// Small fused ops: RMSNorm (with fp16 / int8 / fp32 outputs), SwiGLU, top-k for sampling.
#include "kernels.h"

__device__ __forceinline__ float block_sum(float v, float *red) {   // blockDim.x == 256
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
    for (int i = 0; i < 8; i++) s += red[i];
    __syncthreads();
    return s;
}

// One block per row: out = x * rsqrt(mean(x^2) + eps) * w, written as any of fp32 / fp16 / int8 blocks.
__global__ __launch_bounds__(256) void k_rmsnorm(const float *x, const float *w, int n, int ldx, int ldo,
                                                  float *of, half *oh, QAct oq) {
    __shared__ float red[8];
    const float *xr = x + (size_t)blockIdx.x * ldx;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + EPS);
    const size_t o = (size_t)blockIdx.x * ldo;
    for (int i = threadIdx.x; i < n; i += 256) {      // 256 threads = 8 waves -> each wave covers whole 32-blocks
        const float v = xr[i] * inv * w[i];
        if (of) of[o + i] = v;
        if (oh) oh[o + i] = __float2half(v);
        if (oq.q) {
            const float d = warp_max(fabsf(v)) / 127.0f;
            oq.q[o + i] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
            if ((threadIdx.x & 31) == 0) oq.d[(o + i) / 32] = d;
        }
    }
}

void rmsnorm(const float *x, const float *w, int T, int n, int ldx, int ldo, float *of, half *oh, QAct oq, hipStream_t st) {
    k_rmsnorm<<<T, 256, 0, st>>>(x, w, n, ldx, ldo, of, oh, oq);
}

// Decode-path RMSNorm (row length D, T <= 8 rows): D/256 blocks per row each reduce the whole row
// (cheap, L2-resident) and normalize their own 256 elements, so the norm is one short, wide launch.
__global__ __launch_bounds__(256) void k_norm_rows(const float *x, const float *w, float *of, half *oh, QAct oq) {
    __shared__ float red[8];
    const float *xr = x + (size_t)blockIdx.y * D;
    float ss = 0;
    for (int i = threadIdx.x; i < D; i += 256) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss, red) / D + EPS);
    const int i = blockIdx.x * 256 + threadIdx.x;
    const size_t o = (size_t)blockIdx.y * D + i;
    const float v = xr[i] * inv * w[i];
    if (of) of[o] = v;
    if (oh) oh[o] = __float2half(v);
    if (oq.q) {
        const float d = warp_max(fabsf(v)) / 127.0f;
        oq.q[o] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
        if ((threadIdx.x & 31) == 0) oq.d[o / 32] = d;
    }
}
void norm_rows(const float *x, const float *w, int T, float *of, half *oh, QAct oq, hipStream_t st) {
    k_norm_rows<<<dim3(D / 256, T), 256, 0, st>>>(x, w, of, oh, oq);
}

// g = silu(g) * u in place, for the prefill FFN (gate/up come from separate fp16 GEMMs).
__global__ void k_swiglu_h(half *g, const half *u, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) g[i] = __float2half(silu(__half2float(g[i])) * __half2float(u[i]));
}
void swiglu_h(half *g, const half *u, int64_t n, hipStream_t st) {
    k_swiglu_h<<<(n + 255) / 256, 256, 0, st>>>(g, u, n);
}

// y[t][i] (+)= x[t][i] for a [T][n] fp16 matrix into fp32 with row stride ldy.
__global__ void k_store_f16(const half *x, int n, float *y, int ldy, int accumulate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float v = __half2float(x[(size_t)t * n + i]);
    float *o = y + (size_t)t * ldy + i;
    *o = accumulate ? *o + v : v;
}
void store_f16(const half *x, int n, int T, float *y, int ldy, bool accumulate, hipStream_t st) {
    k_store_f16<<<dim3((n + 255) / 256, T), 256, 0, st>>>(x, n, y, ldy, accumulate);
}

__global__ void k_to_half(const float *x, half *y, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
void to_half(const float *x, half *y, int64_t n, hipStream_t st) {
    k_to_half<<<(n + 255) / 256, 256, 0, st>>>(x, y, n);
}

// ---- top-k over vocabulary rows: two passes of bitonic sorts in LDS ---------------------------
// Pass 1: each block sorts a 4096-wide slice of one row and keeps its best TOPK.
// Pass 2: one block per row merges the candidates and keeps the global best TOPK.

template <int N>
__device__ void bitonic_desc(float *v, int *ix) {
    for (int k = 2; k <= N; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int p = i ^ j;
                if (p > i) {
                    const bool desc = (i & k) == 0;
                    if ((v[i] < v[p]) == desc) {
                        float tv = v[i]; v[i] = v[p]; v[p] = tv;
                        int ti = ix[i]; ix[i] = ix[p]; ix[p] = ti;
                    }
                }
            }
            __syncthreads();
        }
}

constexpr int SLICE = 4096;
__global__ __launch_bounds__(512) void k_topk1(const float *logits, int n, int ld, float *cv, int *ci) {
    __shared__ float v[SLICE];
    __shared__ int ix[SLICE];
    const int row = blockIdx.y, base = blockIdx.x * SLICE;
    for (int i = threadIdx.x; i < SLICE; i += blockDim.x) {
        const int j = base + i;
        v[i] = j < n ? logits[(size_t)row * ld + j] : -INFINITY;
        ix[i] = j;
    }
    __syncthreads();
    bitonic_desc<SLICE>(v, ix);
    for (int i = threadIdx.x; i < TOPK; i += blockDim.x) {
        const size_t o = ((size_t)row * gridDim.x + blockIdx.x) * TOPK + i;
        cv[o] = v[i]; ci[o] = ix[i];
    }
}
__global__ __launch_bounds__(512) void k_topk2(const float *cv, const int *ci, int ncand, float *ov, int *oi) {
    __shared__ float v[SLICE];
    __shared__ int ix[SLICE];
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < SLICE; i += blockDim.x) {
        v[i] = i < ncand ? cv[(size_t)row * ncand + i] : -INFINITY;
        ix[i] = i < ncand ? ci[(size_t)row * ncand + i] : 0;
    }
    __syncthreads();
    bitonic_desc<SLICE>(v, ix);
    for (int i = threadIdx.x; i < TOPK; i += blockDim.x) { ov[row * TOPK + i] = v[i]; oi[row * TOPK + i] = ix[i]; }
}

// logits [T][ld] -> out values/ids [T][TOPK]; scratch needs T * ceil(n/SLICE) * TOPK floats + ints.
void topk(const float *logits, int T, int n, int ld, float *scratch_v, int *scratch_i, float *ov, int *oi, hipStream_t st) {
    const int nb = (n + SLICE - 1) / SLICE;
    if (nb * TOPK > SLICE) FAIL("topk: too many candidates");
    k_topk1<<<dim3(nb, T), 512, 0, st>>>(logits, n, ld, scratch_v, scratch_i);
    k_topk2<<<T, 512, 0, st>>>(scratch_v, scratch_i, nb * TOPK, ov, oi);
}

// Greedy pick for drafts: argmax of each row, written to a device token buffer.
__global__ __launch_bounds__(1024) void k_argmax(const float *logits, int n, int ld, int *out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float *r = logits + (size_t)blockIdx.x * ld;
    float best = -INFINITY; int bidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (r[i] > best) { best = r[i]; bidx = i; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(best, o, 32); const int oi = __shfl_xor(bidx, o, 32);
        if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = bidx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; w++) if (bv[w] > best || (bv[w] == best && bi[w] < bidx)) { best = bv[w]; bidx = bi[w]; }
        out[blockIdx.x] = bidx;
    }
}
void argmax(const float *logits, int T, int n, int ld, int *out, hipStream_t st) {
    k_argmax<<<T, 1024, 0, st>>>(logits, n, ld, out);
}
