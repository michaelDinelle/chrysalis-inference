// Routing and the parts of the MoE around the expert matrices: router logits, softmax top-10 with renormalized
// weights (llama.cpp's build_moe_ffn: SOFTMAX gating, norm_w), the hand-off of cold picks to the CPU, the wait for
// its answer, and the final sum with the shared expert (scaled by its own sigmoid gate).
#include "kernels.h"
#include "cpu_moe.h"

// logits [T][NE] = W x for T <= 8 fp32 rows; W fp16 [NE][D]. The extra workgroup computes the shared expert's gate
// sg[t] = sigmoid(wsg . x[t]).
__global__ __launch_bounds__(256) void k_router(const half *W, const float *wsg, const float *x, int T, float *logits, float *sg) {
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (blockIdx.x == NE / 8) {                       // shared expert gate, one wave per token
        if (wave >= T) return;
        float s = 0;
        for (int i = lane; i < D; i += 32) s += wsg[i] * x[(size_t)wave * D + i];
        s = warp_sum(s);
        if (lane == 0) sg[wave] = sigmoid(s);
        return;
    }
    const int e = blockIdx.x * 8 + wave;
    const half2 *w = (const half2 *)(W + (size_t)e * D);
    float acc[8] = {};
    for (int i = lane; i < D / 2; i += 32) {
        const float2 f = __half22float2(w[i]);
        for (int t = 0; t < T; t++) { const float2 xv = ((const float2 *)(x + (size_t)t * D))[i]; acc[t] += f.x * xv.x + f.y * xv.y; }
    }
    for (int t = 0; t < T; t++) { const float s = warp_sum(acc[t]); if (lane == 0) logits[(size_t)t * NE + e] = s; }
}
void router_dec(const half *W, const float *wsg, const float *x, int T, float *logits, float *sg, hipStream_t st) {
    k_router<<<NE / 8 + 1, 256, 0, st>>>(W, wsg, x, T, logits, sg);
}

// Softmax over the 512 logits, top 10, weights renormalized to sum 1. One 512-thread workgroup does all T tokens
// (T <= 8 at decode) so it can publish the cold picks to the CPU with one fence: slot_of maps an expert to its
// hot slot (>= 0) or cold slot (-1 - slot); hot_slot gets the hot slot per pick (-1 if cold).
__global__ __launch_bounds__(512) void k_route(const float *logits, int T, const int *slot_of, int *ids, float *wts,
                                               int *hot_slot, MoeShared *sh, const float *x, int layer, uint32_t gen) {
    __shared__ float v[NE], redv[16];
    __shared__ int redi[16], pick[TOPK_E];
    __shared__ float pw[TOPK_E];
    const int e = threadIdx.x, wave = e / 32, lane = e % 32;
    for (int t = 0; t < T; t++) {
        v[e] = logits[(size_t)t * NE + e];
        __syncthreads();
        // top 10 by logit (softmax is monotonic); ties go to the lower expert id
        for (int k = 0; k < TOPK_E; k++) {
            float best = v[e]; int bi = e;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor(best, o, 32); const int oi = __shfl_xor(bi, o, 32);
                if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            if (lane == 0) { redv[wave] = best; redi[wave] = bi; }
            __syncthreads();
            if (e == 0) {
                float b = redv[0]; int i = redi[0];
                for (int w = 1; w < 16; w++) if (redv[w] > b || (redv[w] == b && redi[w] < i)) { b = redv[w]; i = redi[w]; }
                pick[k] = i; pw[k] = b; v[i] = -INFINITY;
            }
            __syncthreads();
        }
        if (e == 0) {                                   // weights: softmax restricted to the picks
            const float m0 = pw[0]; float s = 0;
            for (int k = 0; k < TOPK_E; k++) { pw[k] = __expf(pw[k] - m0); s += pw[k]; }
            for (int k = 0; k < TOPK_E; k++) {
                const int j = t * TOPK_E + k, ex = pick[k], sl = slot_of[ex];
                const float w = pw[k] / s;
                ids[j] = ex; wts[j] = w;
                hot_slot[j] = sl >= 0 ? sl : -1;
                if (sh) { sh->slot[j] = sl < 0 ? -1 - sl : -1; sh->w[j] = w; }
            }
        }
        __syncthreads();
    }
    if (!sh) return;
    for (int i = e; i < T * D; i += 512) sh->x[i] = x[i];
    __threadfence_system();
    __syncthreads();
    if (e == 0) {
        sh->layer = layer; sh->T = T;
        __threadfence_system();
        __hip_atomic_store(&sh->go, gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}
void route_dec(const float *logits, int T, const int *slot_of, int *ids, float *wts, int *hot_slot, MoeShared *sh,
               const float *x, int layer, uint32_t gen, hipStream_t st) {
    k_route<<<1, 512, 0, st>>>(logits, T, slot_of, ids, wts, hot_slot, sh, x, layer, gen);
}

// spin until the CPU has answered generation gen
__global__ void k_moe_wait(const uint32_t *done, uint32_t gen) {
    while (__hip_atomic_load(done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != gen) __builtin_amdgcn_s_sleep(1);
}
void moe_wait(const uint32_t *done, uint32_t gen, hipStream_t st) { k_moe_wait<<<1, 1, 0, st>>>(done, gen); }

// out[t] = sg[t] * shared[t] + sum of the hot picks' y + cold[t] (the CPU's answer, read from host memory)
__global__ __launch_bounds__(256) void k_moe_sum(const float *y, const int *hot_slot, const float *shared, const float *sg,
                                                 const float *cold, float *out) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    float s = sg[t] * shared[(size_t)t * D + i];
    for (int k = 0; k < TOPK_E; k++) {
        const int j = t * TOPK_E + k;
        if (hot_slot[j] >= 0) s += y[(size_t)j * D + i];
    }
    if (cold) s += cold[(size_t)t * D + i];
    out[(size_t)t * D + i] = s;
}
void moe_sum(const float *y, const int *hot_slot, const float *shared, const float *sg, const float *cold, int T, float *out, hipStream_t st) {
    k_moe_sum<<<dim3(D / 256, T), 256, 0, st>>>(y, hot_slot, shared, sg, cold, out);
}

// ---- prefill (T > 32 tokens): routing for many tokens, gather by expert, final sum -------------------------

// one 512-thread workgroup per token: top 10 of the logits, softmax over them
__global__ __launch_bounds__(512) void k_route_many(const float *logits, int *ids, float *wts) {
    __shared__ float v[NE], redv[16];
    __shared__ int redi[16], pick[TOPK_E];
    __shared__ float pw[TOPK_E];
    const int t = blockIdx.x, e = threadIdx.x, wave = e / 32, lane = e % 32;
    v[e] = logits[(size_t)t * NE + e];
    __syncthreads();
    for (int k = 0; k < TOPK_E; k++) {
        float best = v[e]; int bi = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(best, o, 32); const int oi = __shfl_xor(bi, o, 32);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) { redv[wave] = best; redi[wave] = bi; }
        __syncthreads();
        if (e == 0) {
            float b = redv[0]; int i = redi[0];
            for (int w = 1; w < 16; w++) if (redv[w] > b || (redv[w] == b && redi[w] < i)) { b = redv[w]; i = redi[w]; }
            pick[k] = i; pw[k] = b; v[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (e == 0) {
        const float m0 = pw[0]; float s = 0;
        for (int k = 0; k < TOPK_E; k++) { pw[k] = __expf(pw[k] - m0); s += pw[k]; }
        for (int k = 0; k < TOPK_E; k++) { ids[t * TOPK_E + k] = pick[k]; wts[t * TOPK_E + k] = pw[k] / s; }
    }
}
__global__ void k_shared_gate(const float *wsg, const float *x, float *sg) {   // one wave per token
    const int t = blockIdx.x, lane = threadIdx.x;
    float s = 0;
    for (int i = lane; i < D; i += 32) s += wsg[i] * x[(size_t)t * D + i];
    s = warp_sum(s);
    if (lane == 0) sg[t] = sigmoid(s);
}
void route_many(const float *logits, const float *wsg, const float *x, int T, int *ids, float *wts, float *sg, hipStream_t st) {
    k_route_many<<<T, 512, 0, st>>>(logits, ids, wts);
    k_shared_gate<<<T, 32, 0, st>>>(wsg, x, sg);
}

// dst[i] = src[idx[i]] for fp16 rows of D
__global__ void k_gather_rows(const half *src, const int *idx, half *dst) {
    const int i = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    dst[(size_t)i * D + c] = src[(size_t)idx[i] * D + c];
}
void gather_rows(const half *src, const int *idx, int n, half *dst, hipStream_t st) {
    if (n) k_gather_rows<<<dim3(D / 256, n), 256, 0, st>>>(src, idx, dst);
}

// out[t] = sg[t] * shared[t] + sum_k Y[pos[t*10+k]]  (picks in router order: deterministic)
__global__ __launch_bounds__(256) void k_moe_combine(const half *Y, const int *pos, const float *shared, const float *sg, float *out) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    float s = sg[t] * shared[(size_t)t * D + i];
    for (int k = 0; k < TOPK_E; k++) s += __half2float(Y[(size_t)pos[t * TOPK_E + k] * D + i]);
    out[(size_t)t * D + i] = s;
}
void moe_combine(const half *Y, const int *pos, const float *shared, const float *sg, int T, float *out, hipStream_t st) {
    k_moe_combine<<<dim3(D / 256, T), 256, 0, st>>>(Y, pos, shared, sg, out);
}
