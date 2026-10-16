// Layer 1's n-gram embedding block (llama.cpp's build_ple). Per token, the host hashes the bigram and trigram that
// end at it into 16 rows of the 320M-row table (memory-mapped from disk, Q8_0 rows of 160) and gathers them; here:
//   emb   = the 16 rows, dequantized and concatenated              [T][D]
//   key   = W_key emb [T][HC][D]   value = W_value emb [T][D]      (projections run as ordinary GEMVs / GEMMs)
//   g_s   = sigmoid(signed sqrt(rmsnorm(key_s) . rmsnorm(res_s) / sqrt(D)))      per stream s
//   gated = value * g_s,  y = silu(depthwise causal conv over rmsnorm(gated), kernel 4, dilation 3)
//   res  += gated + y
#include "kernels.h"

constexpr int Q8_ROW = PLE_DIM / 32 * 34;     // 170 bytes per table row

// rows [T][PLE_HEADS][Q8_ROW] (host-mapped or device) -> emb fp32 [T][D] and int8 blocks / fp16
__global__ __launch_bounds__(160) void k_ple_embed(const uint8_t *rows, float *of, half *oh, QAct oq) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;      // 160 threads: one value of one head's row
    const uint8_t *r = rows + ((size_t)t * PLE_HEADS + h) * Q8_ROW + (i / 32) * 34;
    const float d = __half2float(__ushort_as_half((uint16_t)(r[0] | (r[1] << 8))));
    const float v = d * (int8_t)r[2 + i % 32];
    const size_t o = (size_t)t * D + h * PLE_DIM + i;
    if (of) of[o] = v;
    if (oh) oh[o] = __float2half(v);
    if (oq.q) {
        const float dd = warp_max(fabsf(v)) / 127.0f;
        oq.q[o] = (int8_t)__float2int_rn(dd > 0 ? v / dd : 0.0f);
        if ((i & 31) == 0) oq.d[o / 32] = dd;
    }
}
void ple_embed(const uint8_t *rows, int T, float *of, half *oh, QAct oq, hipStream_t st) {
    k_ple_embed<<<dim3(PLE_HEADS, T), 160, 0, st>>>(rows, of, oh, oq);
}

__device__ __forceinline__ float bsum256(float v, float *red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
    for (int i = 0; i < 8; i++) s += red[i];
    __syncthreads();
    return s;
}

// one block per (stream, token): the gate, res += gated, and the conv input xconv = rmsnorm(gated) * gamma_conv
__global__ __launch_bounds__(256) void k_ple_gate(const float *key, const float *value, float *res, const float *nk, const float *nq,
                                                  const float *nc, float *xconv) {
    __shared__ float red[8];
    const int s = blockIdx.x, t = blockIdx.y;
    const size_t b = ((size_t)t * HC + s) * D;
    float kk = 0, rr = 0;
    for (int i = threadIdx.x; i < D; i += 256) { kk += key[b + i] * key[b + i]; rr += res[b + i] * res[b + i]; }
    const float ik = rsqrtf(bsum256(kk, red) / D + EPS), ir = rsqrtf(bsum256(rr, red) / D + EPS);
    float dot = 0, vv = 0;
    for (int i = threadIdx.x; i < D; i += 256) {
        dot += key[b + i] * ik * nk[s * D + i] * res[b + i] * ir * nq[s * D + i];
        const float v = value[(size_t)t * D + i];
        vv += v * v;
    }
    const float sc = bsum256(dot, red) * rsqrtf((float)D);
    const float g = sigmoid(copysignf(sqrtf(fmaxf(fabsf(sc), 1e-6f)), sc));
    const float ig = rsqrtf(g * g * bsum256(vv, red) / D + EPS);       // rms of gated = |g| * rms(value)
    for (int i = threadIdx.x; i < D; i += 256) {
        const float gv = value[(size_t)t * D + i] * g;
        xconv[b + i] = gv * ig * nc[s * D + i];
        res[b + i] += gv;
    }
}

// res += silu(conv(xconv)): taps at t-9, t-6, t-3, t; state [PLE_HIST][HCD] holds positions pos0-9 .. pos0-1.
__global__ __launch_bounds__(256) void k_ple_conv(const float *xconv, const float *state, const float *w, float *res) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    float acc = 0;
#pragma unroll
    for (int k = 0; k < PLE_K; k++) {
        const int src = t - (PLE_K - 1 - k) * PLE_DIL;
        const float x = src >= 0 ? xconv[(size_t)src * HCD + c] : state[(size_t)(PLE_HIST + src) * HCD + c];
        acc += w[c * PLE_K + k] * x;
    }
    res[(size_t)t * HCD + c] += silu(acc);
}

// new state = the last PLE_HIST positions of [old state, xconv[0..n)] (n accepted tokens of this pass)
__global__ void k_ple_commit(float *state, const float *xconv, int n) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    float h[PLE_HIST];
#pragma unroll
    for (int j = 0; j < PLE_HIST; j++) {
        const int src = n - PLE_HIST + j;                  // position relative to the pass start
        h[j] = src >= 0 ? xconv[(size_t)src * HCD + c] : state[(size_t)(PLE_HIST + src) * HCD + c];
    }
#pragma unroll
    for (int j = 0; j < PLE_HIST; j++) state[(size_t)j * HCD + c] = h[j];
}

void ple_block(const float *key, const float *value, float *res, const float *nk, const float *nq, const float *nc,
               const float *conv_w, const float *state, float *xconv, int T, hipStream_t st) {
    k_ple_gate<<<dim3(HC, T), 256, 0, st>>>(key, value, res, nk, nq, nc, xconv);
    k_ple_conv<<<dim3(HCD / 256, T), 256, 0, st>>>(xconv, state, conv_w, res);
}
void ple_commit(float *state, const float *xconv, int n, hipStream_t st) {
    k_ple_commit<<<HCD / 256, 256, 0, st>>>(state, xconv, n);
}
