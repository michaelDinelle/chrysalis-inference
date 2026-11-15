#include "kernels.h"
#include "cpu_moe.h"

__global__ __launch_bounds__(256) void k_router_route(const half *W, const float *wsg, const float *x, int T, float *logits,
        float *sg, const int *slot_of, int n_ram, int *ids, float *wts, int *hot_slot, MoeShared *sh, float gfrac, int layer,
        uint32_t gen, unsigned *counter, unsigned *counts, unsigned starts, float xfrac, bool hand_x) {
    __shared__ bool last;
    __shared__ int ram_n[8], gpu_n[8];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    {
        const int e = blockIdx.x * 8 + wave;
        const int4 *w = (const int4 *)(W + (size_t)e * D);
        constexpr int NI = D / 8 / 32;
        int4 q[NI];
#pragma unroll
        for (int k = 0; k < NI; k++) q[k] = w[lane + 32 * k];
        float acc[8] = {};
#pragma unroll
        for (int k = 0; k < NI; k++) {
            const int i = lane + 32 * k;
            const half2 *h = (const half2 *)&q[k];
            const float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]), f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
            for (int t = 0; t < T; t++) {
                const float4 a = ((const float4 *)(x + (size_t)t * D))[2 * i], b = ((const float4 *)(x + (size_t)t * D))[2 * i + 1];
                acc[t] += f0.x * a.x + f0.y * a.y + f1.x * a.z + f1.y * a.w + f2.x * b.x + f2.y * b.y + f3.x * b.z + f3.y * b.w;
            }
        }
        for (int t = 0; t < T; t++) { const float v = warp_sum(acc[t]); if (lane == 0) logits[(size_t)t * NE + e] = v; }
        if (blockIdx.x == 0 && wave < T) {
            constexpr int NV = D / 4 / 32;
            const float4 *w4 = (const float4 *)wsg, *x4 = (const float4 *)(x + (size_t)wave * D);
            float4 wv[NV], xv[NV];
#pragma unroll
            for (int k = 0; k < NV; k++) { wv[k] = w4[lane + 32 * k]; xv[k] = x4[lane + 32 * k]; }
            float v = 0;
#pragma unroll
            for (int k = 0; k < NV; k++) v += wv[k].x * xv[k].x + wv[k].y * xv[k].y + wv[k].z * xv[k].z + wv[k].w * xv[k].w;
            v = warp_sum(v);
            if (lane == 0) sg[wave] = sigmoid(v);
        }
    }
    if (sh && hand_x) {
        const int n = T * D, per = (n + gridDim.x - 1) / gridDim.x, a = blockIdx.x * per, z = min(n, a + per);
        for (int i = a + tid; i < z; i += 256) sh->x[i] = x[i];
        __threadfence_system();
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    float pv = -INFINITY; int pe = 0;
    if (wave < T) {
        float v[16];
#pragma unroll
        for (int k = 0; k < 16; k++) v[k] = __ldcg(&logits[(size_t)wave * NE + lane + 32 * k]);
        for (int r = 0; r < TOPK_E; r++) {
            float b = v[0]; int bi = lane;
#pragma unroll
            for (int k = 1; k < 16; k++) { const bool tk = v[k] > b; b = tk ? v[k] : b; bi = tk ? lane + 32 * k : bi; }
            wave_argmax(b, bi);
            if (lane == r) { pv = b; pe = bi; }
#pragma unroll
            for (int k = 0; k < 16; k++) v[k] = lane + 32 * k == bi ? -INFINITY : v[k];
        }
    }
    const float top = __shfl_sync(0xffffffffu, pv, 0);
    const float p = lane < TOPK_E ? __expf(pv - top) : 0.0f;
    const float wk = p / warp_sum(p);
    const int sl = wave < T && lane < TOPK_E ? slot_of[pe] : 0;
    const bool cold = wave < T && lane < TOPK_E && sl < 0, ram = cold && -1 - sl < n_ram;
    const uint32_t bal = (uint32_t)__ballot_sync(0xffffffffu, ram);
    if (lane == 0) ram_n[wave] = __popc(bal);
    __syncthreads();
    int nc = 0;
    for (int w = wave; w > 0 && !(starts >> w & 1u); w--) nc += ram_n[w - 1];
    const int before = nc + __popc(bal & ((1u << lane) - 1u));
    const bool gpu = ram && floorf((before + 1) * gfrac) > floorf(before * gfrac);
    const uint32_t gbal = (uint32_t)__ballot_sync(0xffffffffu, gpu);
    if (lane == 0) gpu_n[wave] = __popc(gbal);
    __syncthreads();
    int gc = 0;
    for (int w = wave; w > 0 && !(starts >> w & 1u); w--) gc += gpu_n[w - 1];
    const int gb = gc + __popc(gbal & ((1u << lane) - 1u));
    const bool emu = gpu && floorf((gb + 1) * xfrac) <= floorf(gb * xfrac);
    if (wave < T && lane < TOPK_E) {
        const int j = wave * TOPK_E + lane;
        if (counts && layer < NL) atomicAdd(&counts[layer * NE + pe], 1u);
        ids[j] = pe; wts[j] = wk;
        hot_slot[j] = sl >= 0 ? sl : emu ? EMU_PICK : gpu ? -2 - (-1 - sl) : -1;
        if (sh) { sh->slot[j] = cold && !gpu ? -1 - sl : emu ? -2 - (-1 - sl) : -1; sh->w[j] = wk; }
    }
    if (tid == 0) *counter = 0;
    if (!sh) return;
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        sh->layer = layer; sh->T = T;
        __threadfence_system();
        store_sys(&sh->go, gen);
    }
}
void router_route_dec(const half *W, const float *wsg, const float *x, int T, float *logits, float *sg, const int *slot_of,
                      int n_ram, int *ids, float *wts, int *hot_slot, MoeShared *sh, float gfrac, int layer, uint32_t gen,
                      unsigned *counter, cudaStream_t st, unsigned *counts, unsigned starts, float xfrac, bool hand_x) {
    k_router_route<<<NE / 8, 256, 0, st>>>(W, wsg, x, T, logits, sg, slot_of, n_ram, ids, wts, hot_slot, sh, gfrac, layer, gen,
                                           counter, counts, starts, xfrac, hand_x);
}
constexpr unsigned WAIT_SPINS = 1u << 22;
__global__ void k_wait_flag(const unsigned *flag, unsigned gen, unsigned *lost) {
    for (unsigned i = 0; i < WAIT_SPINS; i++) {
        if ((int)(load_gpu(flag) - gen) >= 0) return;
        __nanosleep(128);
    }
    *lost = 1u;
    __threadfence_system();
}
void wait_flag(const unsigned *flag, unsigned gen, unsigned *lost, cudaStream_t st) { k_wait_flag<<<1, 1, 0, st>>>(flag, gen, lost); }

__global__ __launch_bounds__(256) void k_moe_finish(const uint32_t *done, uint32_t gen, const float *y, const int *hot_slot,
                                                    const float *shared, const float *sg, const float *cold, const float *inject, float *res,
                                                    const float *ye) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (done) {
        if (threadIdx.x == 0)
            while (load_sys(done) != gen) __nanosleep(64);
        __syncthreads();
        __threadfence_system();
    }
    float s = (sg ? sg[t] : 1.0f) * shared[(size_t)t * D + i];
    for (int k = 0; k < TOPK_E; k++) {
        const int j = t * TOPK_E + k;
        const int hs = hot_slot[j];
        if (hs != -1) s += hs == EMU_PICK ? __ldcv(&ye[(size_t)j * D + i]) : y[(size_t)j * D + i];
    }
    if (cold) s += __ldcv(&cold[(size_t)t * D + i]);
#pragma unroll
    for (int st = 0; st < HC; st++) res[((size_t)t * HC + st) * D + i] += 2.0f * sigmoid(inject[t * HC + st] * (1.0f / HC)) * s;
}
void moe_finish(const uint32_t *done, uint32_t gen, const float *y, const int *hot_slot, const float *shared, const float *sg,
                const float *cold, const float *inject, float *res, int T, cudaStream_t st, const float *ye) {
    k_moe_finish<<<dim3(D / 256, T), 256, 0, st>>>(done, gen, y, hot_slot, shared, sg, cold, inject, res, ye);
}

__global__ __launch_bounds__(32) void k_emu_glu(const int *hot_slot, MoeEmu *me, uint32_t gen) {
    const int j = blockIdx.y, lane = threadIdx.x, r = blockIdx.x * 32 + lane;
    if (hot_slot[j] != EMU_PICK) return;
    if (lane == 0)
        while (load_sys(&me->ready) != gen) __nanosleep(64);
    __syncthreads();
    __threadfence_system();
    const float v = silu(__ldcv(&me->gu[j][r])) * __ldcv(&me->gu[j][FE + r]), d = warp_max(fabsf(v)) / 127.0f;
    me->hq[j][r] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
    if (lane == 0) me->hd[j][blockIdx.x] = d;
    __threadfence_system();
    if (lane == 0) store_sys(&me->flag[j][blockIdx.x], gen);
}
void emu_glu(const int *hot_slot, MoeEmu *me, uint32_t gen, int T, cudaStream_t st) {
    k_emu_glu<<<dim3(FE / 32, T * TOPK_E), 32, 0, st>>>(hot_slot, me, gen);
}


__global__ __launch_bounds__(512) void k_route_many(const float *logits, int *ids, float *wts) {
    __shared__ float v[NE], redv[16];
    __shared__ int redi[16], pick[TOPK_E];
    __shared__ float pw[TOPK_E];
    const int t = blockIdx.x, e = threadIdx.x, wave = e / 32, lane = e % 32;
    v[e] = logits[(size_t)t * NE + e];
    __syncthreads();
    for (int k = 0; k < TOPK_E; k++) {
        float best = v[e]; int bi = e;
        wave_argmax(best, bi);
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
__global__ void k_shared_gate(const float *wsg, const float *x, float *sg) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float s = 0;
    for (int i = lane; i < D; i += 32) s += wsg[i] * x[(size_t)t * D + i];
    s = warp_sum(s);
    if (lane == 0) sg[t] = sigmoid(s);
}
void route_many(const float *logits, const float *wsg, const float *x, int T, int *ids, float *wts, float *sg, cudaStream_t st) {
    k_route_many<<<T, 512, 0, st>>>(logits, ids, wts);
    k_shared_gate<<<T, 32, 0, st>>>(wsg, x, sg);
}

__global__ void k_gather_rows(const half *src, const int *idx, half *dst) {
    const int i = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    dst[(size_t)i * D + c] = src[(size_t)idx[i] * D + c];
}
void gather_rows(const half *src, const int *idx, int n, half *dst, cudaStream_t st) {
    if (n) k_gather_rows<<<dim3(D / 256, n), 256, 0, st>>>(src, idx, dst);
}

__global__ __launch_bounds__(256) void k_moe_shared(const float *shared, const float *sg, float *moe) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    moe[(size_t)t * D + i] = sg[t] * shared[(size_t)t * D + i];
}
void moe_shared(const float *shared, const float *sg, int T, float *moe, cudaStream_t st) {
    k_moe_shared<<<dim3(D / 256, T), 256, 0, st>>>(shared, sg, moe);
}

__global__ __launch_bounds__(256) void k_moe_accum(const half *Y, const int *pos, int p0, int p1, const int *toks, float *moe) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = toks[blockIdx.y];
    float s = 0;
#pragma unroll
    for (int k = 0; k < TOPK_E; k++) {
        const int q = pos[t * TOPK_E + k];
        if (q >= p0 && q < p1) s += __half2float(Y[(size_t)(q - p0) * D + i]);
    }
    moe[(size_t)t * D + i] += s;
}
void moe_accum(const half *Y, const int *pos, int p0, int p1, const int *toks, int ntok, float *moe, cudaStream_t st) {
    if (ntok) k_moe_accum<<<dim3(D / 256, ntok), 256, 0, st>>>(Y, pos, p0, p1, toks, moe);
}

