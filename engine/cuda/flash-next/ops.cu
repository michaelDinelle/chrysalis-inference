#include "kernels.h"

__device__ __forceinline__ float block_sum(float v, float *red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
    for (int i = 0; i < 8; i++) s += red[i];
    __syncthreads();
    return s;
}

__global__ __launch_bounds__(256) void k_rmsnorm(const float *x, const float *w, int n, int ldx, int ldo,
                                                  float *of, half *oh, QAct oq) {
    __shared__ float red[8];
    const float *xr = x + (size_t)blockIdx.x * ldx;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + EPS);
    const size_t o = (size_t)blockIdx.x * ldo;
    for (int i = threadIdx.x; i < n; i += 256) {
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

void rmsnorm(const float *x, const float *w, int T, int n, int ldx, int ldo, float *of, half *oh, QAct oq, cudaStream_t st) {
    k_rmsnorm<<<T, 256, 0, st>>>(x, w, n, ldx, ldo, of, oh, oq);
}

__global__ void k_swiglu_h(half *g, const half *u, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) g[i] = __float2half(silu(__half2float(g[i])) * __half2float(u[i]));
}
void swiglu_h(half *g, const half *u, int64_t n, cudaStream_t st) {
    k_swiglu_h<<<(n + 255) / 256, 256, 0, st>>>(g, u, n);
}

__global__ void k_store_f16(const half *x, int n, float *y, int ldy, int accumulate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float v = __half2float(x[(size_t)t * n + i]);
    float *o = y + (size_t)t * ldy + i;
    *o = accumulate ? *o + v : v;
}
void store_f16(const half *x, int n, int T, float *y, int ldy, bool accumulate, cudaStream_t st) {
    k_store_f16<<<dim3((n + 255) / 256, T), 256, 0, st>>>(x, n, y, ldy, accumulate);
}


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

void topk(const float *logits, int T, int n, int ld, float *scratch_v, int *scratch_i, float *ov, int *oi, cudaStream_t st) {
    const int nb = (n + SLICE - 1) / SLICE;
    if (nb * TOPK > SLICE) FAIL("topk: too many candidates");
    k_topk1<<<dim3(nb, T), 512, 0, st>>>(logits, n, ld, scratch_v, scratch_i);
    k_topk2<<<T, 512, 0, st>>>(scratch_v, scratch_i, nb * TOPK, ov, oi);
}

__global__ __launch_bounds__(1024) void k_logit_stats(const float *logits, int n, int ld, float *out) {
    __shared__ float tops[32];
    __shared__ double red[1024];
    const float *r = logits + (size_t)blockIdx.x * ld;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 1024) m = fmaxf(m, r[i]);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) tops[threadIdx.x >> 5] = m;
    __syncthreads();
    m = tops[0];
    for (int w = 1; w < 32; w++) m = fmaxf(m, tops[w]);
    float acc[5] = {0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = r[i];
        if (!(v > -INFINITY)) continue;
        const float d = v - m, e = expf(d);
        acc[0] += 1; acc[1] += v; acc[2] += v * v; acc[3] += e; acc[4] += d * e;
    }
    float *o = out + (size_t)blockIdx.x * LOGIT_STATS;
    for (int k = 0; k < 5; k++) {
        red[threadIdx.x] = acc[k];
        __syncthreads();
        for (int w = 512; w > 0; w >>= 1) {
            if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
            __syncthreads();
        }
        if (threadIdx.x == 0) o[1 + k] = (float)red[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) o[0] = (float)n;
}
void logit_stats(const float *logits, int T, int n, int ld, float *out, cudaStream_t st) {
    k_logit_stats<<<T, 1024, 0, st>>>(logits, n, ld, out);
}

__global__ void k_mask_logits(float *logits, const uint32_t *mask, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    logits += (size_t)blockIdx.y * n;
    mask += (size_t)blockIdx.y * ((n + 31) / 32);
    if (i < n && !((mask[i >> 5] >> (i & 31)) & 1u)) logits[i] = -INFINITY;
}
void mask_logits(float *logits, const uint32_t *mask, int n, int rows, cudaStream_t st) {
    k_mask_logits<<<dim3((n + 255) / 256, rows), 256, 0, st>>>(logits, mask, n);
}

__global__ __launch_bounds__(1024) void k_argmax(const float *logits, int n, int ld, int *out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float *r = logits + (size_t)blockIdx.x * ld;
    float best = -INFINITY; int bidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (r[i] > best) { best = r[i]; bidx = i; }
    wave_argmax(best, bidx);
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = bidx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; w++) if (bv[w] > best || (bv[w] == best && bi[w] < bidx)) { best = bv[w]; bidx = bi[w]; }
        out[blockIdx.x] = bidx;
    }
}
void argmax(const float *logits, int T, int n, int ld, int *out, cudaStream_t st) {
    k_argmax<<<T, 1024, 0, st>>>(logits, n, ld, out);
}


__global__ void k_hc_init(const float *x, float *res) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= D) return;
    const float v = x[(size_t)t * D + i];
#pragma unroll
    for (int s = 0; s < HC; s++) res[((size_t)t * HC + s) * D + i] = v;
}
void hc_init(const float *x, float *res, int T, cudaStream_t st) {
    k_hc_init<<<dim3(D / 256, T), 256, 0, st>>>(x, res);
}

__global__ __launch_bounds__(256) void k_hc_norm(const float *res, const float *gamma, float *of, half *oh, QAct oq) {
    __shared__ float red[8];
    const int s = blockIdx.x, t = blockIdx.y;
    const size_t base = ((size_t)t * HC + s) * D;
    float ss = 0;
    for (int i = threadIdx.x; i < D; i += 256) ss += res[base + i] * res[base + i];
    const float inv = rsqrtf(block_sum(ss, red) / D + EPS);
    for (int i = threadIdx.x; i < D; i += 256) {
        const float v = res[base + i] * inv * gamma[s * D + i];
        if (of) of[base + i] = v;
        if (oh) oh[base + i] = __float2half(v);
        if (oq.q) {
            const float d = warp_max(fabsf(v)) / 127.0f;
            oq.q[base + i] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
            if ((threadIdx.x & 31) == 0) oq.d[(base + i) / 32] = d;
        }
    }
}
void hc_norm(const float *res, const float *gamma, int T, float *of, half *oh, QAct oq, cudaStream_t st) {
    k_hc_norm<<<dim3(HC, T), 256, 0, st>>>(res, gamma, of, oh, oq);
}

__global__ void k_hc_lo(float *lo, int64_t n, half *oh, QAct oq) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const float v = i < n ? silu(lo[i] * (1.0f / HC)) : 0.0f;
    if (i < n) { lo[i] = v; if (oh) oh[i] = __float2half(v); }
    if (oq.q) {
        const float d = warp_max(fabsf(v)) / 127.0f;
        if (i < n) oq.q[i] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
        if ((threadIdx.x & 31) == 0 && i < n) oq.d[i / 32] = d;
    }
}
void hc_lo(float *lo, int T, half *oh, QAct oq, cudaStream_t st) {
    const int64_t n = (int64_t)T * HCR;
    k_hc_lo<<<(n + 255) / 256, 256, 0, st>>>(lo, n, oh, oq);
}

__global__ __launch_bounds__(256) void k_hc_mix(const float *xn, const float *gate, float *of, half *oh, QAct oq) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    float v = 0;
#pragma unroll
    for (int s = 0; s < HC; s++) {
        const size_t j = ((size_t)t * HC + s) * D + i;
        v += xn[j] * sigmoid(gate[j]);
    }
    v *= 1.0f / HC;
    const size_t o = (size_t)t * D + i;
    if (of) of[o] = v;
    if (oh) oh[o] = __float2half(v);
    if (oq.q) {
        const float d = warp_max(fabsf(v)) / 127.0f;
        oq.q[o] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
        if ((threadIdx.x & 31) == 0) oq.d[o / 32] = d;
    }
}
void hc_mix(const float *xn, const float *gate, int T, float *of, half *oh, QAct oq, cudaStream_t st) {
    k_hc_mix<<<dim3(D / 256, T), 256, 0, st>>>(xn, gate, of, oh, oq);
}

__global__ __launch_bounds__(256) void k_hc_combine(float *res, const float *inject, const float *out, const float *out2, int ld_inject) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    const float o = out[(size_t)t * D + i] + (out2 ? out2[(size_t)t * D + i] : 0.0f);
#pragma unroll
    for (int s = 0; s < HC; s++) {
        const float w = 2.0f * sigmoid(inject[(size_t)t * ld_inject + s] * (1.0f / HC));
        res[((size_t)t * HC + s) * D + i] += w * o;
    }
}
void hc_combine(float *res, const float *inject, int ld_inject, const float *out, const float *out2, int T, cudaStream_t st) {
    k_hc_combine<<<dim3(D / 256, T), 256, 0, st>>>(res, inject, out, out2, ld_inject);
}

__global__ __launch_bounds__(256) void k_mtp_pack(const float *e, const float *xn, float *out, half *out_h) {
    const int i = blockIdx.x * 256 + threadIdx.x, s = blockIdx.y, t = blockIdx.z;
    const float v = i < D ? e[(size_t)t * D + i] : xn[((size_t)t * HC + s) * D + i - D];
    const size_t o = ((size_t)t * HC + s) * 2 * D + i;
    if (out) out[o] = v;
    else out_h[o] = __float2half(v);
}
void mtp_pack(const float *e, const float *xn, int T, float *out, half *out_h, cudaStream_t st) {
    k_mtp_pack<<<dim3(2 * D / 256, HC, T), 256, 0, st>>>(e, xn, out, out_h);
}

__device__ __forceinline__ int dot16i(int4 w, int4 x, int acc) {
    return dot4(w.w, x.w, dot4(w.z, x.z, dot4(w.y, x.y, dot4(w.x, x.x, acc))));
}

__device__ unsigned g_mix_bar[2][2];

__device__ __forceinline__ void mix_barrier(unsigned *bar) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned gen = load_relaxed_gpu(&bar[1]);
        if (atomicAdd(&bar[0], 1u) == gridDim.x - 1) {
            store_relaxed_gpu(&bar[0], 0u);
            store_gpu(&bar[1], gen + 1);
        } else {
            while (load_gpu(&bar[1]) == gen) __nanosleep(64);
        }
    }
    __syncthreads();
    __threadfence();
}

constexpr int MIX_T = 4;
__global__ __launch_bounds__(256) void k_mix(const float *res, const float *gamma, QTensor down, QTensor inj, QTensor up, QAct xq,
                                             float *inv, float *lo, float *inject, int ld_inject, float *mx, half *hh, QAct hq, int T,
                                             int slot, unsigned *flags, unsigned gen, int role) {
    __shared__ float part[32];
    __shared__ float iv;
    __shared__ int4 lq4[MIX_T][HCR / 16];
    __shared__ float ld_[MIX_T][HCR / 32];
    __shared__ float g[MIX_T][HC][64];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    if (role == 1 && blockIdx.x == 0 && tid == 0) store_gpu(&flags[0], gen);
    for (int u = blockIdx.x; u < HC * T; u += gridDim.x) {
        const int s = u % HC, t = u / HC;
        if (tid < 32) part[tid] = 0.0f;
        __syncthreads();
        const float4 *r = (const float4 *)(res + ((size_t)t * HC + s) * D);
        for (int w = wave; w < D / 4 / 32; w += 8) {
            float ss = 0;
            const float4 v = r[w * 32 + lane];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            const float sw = warp_sum(ss);
            if (lane == 0) part[w] = sw;
        }
        __syncthreads();
        if (tid == 0) {
            float v = 0;
            for (int w = 0; w < 32; w++) v += part[w];
            iv = rsqrtf(v / D + EPS);
            inv[t * HC + s] = iv;
        }
        __syncthreads();
        for (int blk = wave; blk < D / 32; blk += 8) {
            const int c = s * D + blk * 32 + lane;
            const float v = res[(size_t)t * HCD + c] * iv * gamma[c];
            const float d = warp_max(fabsf(v)) / 127.0f;
            xq.q[(size_t)t * HCD + c] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
            if (lane == 0) xq.d[((size_t)t * HCD + c) / 32] = d;
        }
        __syncthreads();
    }
    mix_barrier(g_mix_bar[slot]);
    const int rows = HCR + (inject ? HC : 0);
    for (int task = blockIdx.x; task * 8 < rows; task += gridDim.x) {
        const int row = task * 8 + wave;
        if (row >= rows) continue;
        const QTensor *W = row < HCR ? &down : &inj;
        const int r = row < HCR ? row : row - HCR;
        float acc[MIX_T] = {};
#pragma unroll
        for (int k = 0; k < HCD / 32 / 32; k++) {
            const int ch = lane + 32 * k;
            const int8_t *wr = (const int8_t *)W->q + (size_t)r * HCD + ch * 32;
            const int4 w0 = *(const int4 *)wr, w1 = *(const int4 *)(wr + 16);
            const float dw = __half2float(W->d[(size_t)r * (HCD / 32) + ch]);
#pragma unroll
            for (int t = 0; t < MIX_T; t++) {
                if (t >= T) break;
                const int4 *x4 = (const int4 *)(xq.q + (size_t)t * HCD + ch * 32);
                acc[t] += dw * xq.d[(size_t)t * (HCD / 32) + ch] * dot16i(w1, x4[1], dot16i(w0, x4[0], 0));
            }
        }
#pragma unroll
        for (int t = 0; t < MIX_T; t++) {
            if (t >= T) break;
            const float v = warp_sum(acc[t]);
            if (lane == 0) {
                if (row < HCR) lo[(size_t)t * HCR + r] = v;
                else inject[(size_t)t * ld_inject + r] = v;
            }
        }
    }
    mix_barrier(g_mix_bar[slot]);
    int8_t *lq = (int8_t *)lq4;
    for (int u = blockIdx.x; u < D / 64; u += gridDim.x) {
        const int i0 = u * 64;
        for (int k = wave; k < T * HCR / 32; k += 8) {
            const int t = k / (HCR / 32), blk = k % (HCR / 32);
            const float v = silu(lo[(size_t)t * HCR + blk * 32 + lane] * (1.0f / HC));
            const float d = warp_max(fabsf(v)) / 127.0f;
            lq[(size_t)t * HCR + blk * 32 + lane] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
            if (lane == 0) ld_[t][blk] = d;
        }
        __syncthreads();
        const int s = tid / 64, p = tid % 64, row = s * D + i0 + p;
        const int8_t *wr = (const int8_t *)up.q + (size_t)row * HCR;
        float acc[MIX_T] = {};
#pragma unroll
        for (int ch = 0; ch < HCR / 32; ch++) {
            const int4 w0 = *(const int4 *)(wr + ch * 32), w1 = *(const int4 *)(wr + ch * 32 + 16);
            const float dw = __half2float(up.d[(size_t)row * (HCR / 32) + ch]);
#pragma unroll
            for (int t = 0; t < MIX_T; t++) {
                if (t >= T) break;
                acc[t] += dw * ld_[t][ch] * dot16i(w1, lq4[t][ch * 2 + 1], dot16i(w0, lq4[t][ch * 2], 0));
            }
        }
#pragma unroll
        for (int t = 0; t < MIX_T; t++) g[t][s][p] = acc[t];
        __syncthreads();
        if (tid < 64) {
            const int i = i0 + tid;
            for (int t = 0; t < T; t++) {
                float m = 0;
#pragma unroll
                for (int s2 = 0; s2 < HC; s2++) {
                    const int c = s2 * D + i;
                    m += res[(size_t)t * HCD + c] * inv[t * HC + s2] * gamma[c] * sigmoid(g[t][s2][tid]);
                }
                m *= 1.0f / HC;
                const size_t o = (size_t)t * D + i;
                mx[o] = m;
                if (hh) hh[o] = __float2half(m);
                if (hq.q) {
                    const float d = warp_max(fabsf(m)) / 127.0f;
                    hq.q[o] = (int8_t)__float2int_rn(d > 0 ? m / d : 0.0f);
                    if (lane == 0) hq.d[o / 32] = d;
                }
            }
        }
        __syncthreads();
    }
    if (role != 2) return;
    __threadfence();
    __syncthreads();
    if (tid == 0 && atomicAdd(&flags[2], 1u) == gridDim.x - 1) {
        store_relaxed_gpu(&flags[2], 0u);
        store_gpu(&flags[1], gen);
    }
}

void hc_mix_fused(const float *res, const float *gamma, const QTensor &down, const QTensor &up, const QTensor *inj, int T,
                  float *lo, float *inject, int ld_inject, float *inv, float *mx, half *hh, QAct hq, QAct xq, int slot, cudaStream_t st,
                  unsigned *flags, unsigned gen, int role) {
    if (T < 1 || T > MIX_T) FAIL("hc_mix_fused: T=%d", T);
    const QTensor none{};
    const int grid = std::max((HCR + (inj ? HC : 0) + 7) / 8, D / 64);
    k_mix<<<grid, 256, 0, st>>>(res, gamma, down, inj ? *inj : none, up, xq, inv, lo, inj ? inject : nullptr, ld_inject, mx, hh, hq, T, slot,
                                flags, gen, role);
}
