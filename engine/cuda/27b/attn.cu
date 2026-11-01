#include "kernels.h"
#include <algorithm>
#include <initializer_list>

__device__ __forceinline__ const uint8_t *kv_head(const KVRef &kv, int layer, int head, int pos) {
    return kv.pool + (size_t)kv.pages[pos / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + (size_t)head * KV_HEAD;
}

__device__ __forceinline__ float fwht256(float x, float *buf) {
    const int i = threadIdx.x;
    {
        float o;
        o = lane_xor<1>(x); x = (i & 1) ? o - x : x + o;
        o = lane_xor<2>(x); x = (i & 2) ? o - x : x + o;
        o = lane_xor<4>(x); x = (i & 4) ? o - x : x + o;
        o = lane_xor<8>(x); x = (i & 8) ? o - x : x + o;
        o = lane_xor<16>(x); x = (i & 16) ? o - x : x + o;
    }
    for (int h = 32; h < 256; h <<= 1) {
        buf[i] = x;
        __syncthreads();
        const float o = buf[i ^ h];
        __syncthreads();
        x = (i & h) ? o - x : x + o;
    }
    return x * 0.0625f;
}

__device__ __forceinline__ float block_sum256(float v, float *red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
    for (int i = 0; i < 8; i++) s += red[i];
    __syncthreads();
    return s;
}

__global__ __launch_bounds__(256) void k_attn_prep(const float *qg, const float *kin, const float *vin, const float *qnw,
                                                   const float *knw, int pos0, int rope0, const int *rpos, KVRef kv, int layer, half *Q16) {
    __shared__ float red[8], buf[HD];
    const int job = blockIdx.x, t = blockIdx.y, i = threadIdx.x, pos = pos0 + t;
    float x;
    if (job < NH) x = qg[((size_t)t * NH + job) * 2 * HD + i];
    else if (job < NH + NKV) x = kin[((size_t)t * NKV + job - NH) * HD + i];
    else x = vin[((size_t)t * NKV + job - NH - NKV) * HD + i];
    if (job < NH + NKV) {
        x *= rsqrtf(block_sum256(x * x, red) / HD + EPS) * (job < NH ? qnw : knw)[i];
        if (i < ROT) buf[i] = x;
        __syncthreads();
        if (i < ROT) {
            const int f = i % (ROT / 2);
            const int sec = f % 3 == 1 && f < 33 ? 1 : f % 3 == 2 && f < 30 ? 2 : 0;
            const int rp = rpos ? rpos[t * 3 + sec] : rope0 + t;
            float s, c;
            sincosf(rp * powf(ROPE_BASE, -2.0f * f / ROT), &s, &c);
            x = i < ROT / 2 ? x * c - buf[i + ROT / 2] * s : x * c + buf[i - ROT / 2] * s;
        }
    }
    if (job < NH) {
        Q16[((size_t)t * NH + job) * HD + i] = __float2half(x * 0.0625f);
        return;
    }
    const bool isv = job >= NH + NKV;
    const int head = (job - NH) % NKV, slot = pos % PAGE;
    uint8_t *blk = (uint8_t *)kv_head(kv, layer, head, pos);
    const float amax = warp_max(fabsf(x));
    if (!isv) {
        const float d = amax / 127.0f;
        ((int8_t *)blk)[slot * HD + i] = (int8_t)__float2int_rn(d > 0 ? x / d : 0.0f);
        if ((i & 31) == 0) ((half *)(blk + KS_OFF))[slot * (HD / 32) + i / 32] = __float2half(d);
    } else {
        x = fwht256(x, buf);
        const float d = warp_max(fabsf(x)) / 15.0f;
        const uint32_t u = (uint32_t)(__float2int_rn(d > 0 ? x / d : 0.0f) + 16);
        uint32_t *lo = (uint32_t *)(blk + VLO_OFF) + (i * (PAGE / 2) + slot / 2) / 4;
        const int ls = ((slot / 2) % 4) * 8 + (slot % 2) * 4;
        atomicAnd(lo, ~(15u << ls));
        atomicOr(lo, (u & 15) << ls);
        uint32_t *hb = (uint32_t *)(blk + VHI_OFF) + (i * (PAGE / 8) + slot / 8) / 4;
        const int hs = ((slot / 8) % 4) * 8 + slot % 8;
        atomicAnd(hb, ~(1u << hs));
        atomicOr(hb, (u >> 4) << hs);
        if ((i & 31) == 0) ((half *)(blk + VS_OFF))[(i / 32) * PAGE + slot] = __float2half(d);
    }
}

void attn_prep(const float *qg, const float *k, const float *v, const float *qnw, const float *knw, int T, int pos0,
               int rope0, const int *rpos, KVRef kv, int layer, half *Q16, cudaStream_t st) {
    k_attn_prep<<<dim3(NH + 2 * NKV, T), 256, 0, st>>>(qg, k, v, qnw, knw, pos0, rope0, rpos, kv, layer, Q16);
}

constexpr int KST = 32;
constexpr int KLD = HD + 8;
constexpr int VLD = KST + 8;

template <int AW>
struct AttnRaw {
    static constexpr int NK = KST * HD / 8 / (AW * 32), NV = HD * (KST / 8) / (AW * 32);
    uint2 kw[NK]; half ks[NK];
    uint32_t vlo[NV], vhb[NV]; uint4 vsr[NV];
};

template <int AW>
__device__ __forceinline__ void attn_fetch(AttnRaw<AW> &r, const uint8_t *page, int slot0, int nvalid, int tid) {
#pragma unroll
    for (int m = 0; m < AttnRaw<AW>::NK; m++) {
        const int i = tid + m * AW * 32, key = i / (HD / 8), d8 = i % (HD / 8);
        r.kw[m] = *(const uint2 *)(page + (size_t)(slot0 + key) * HD + d8 * 8);
        r.ks[m] = key < nvalid ? ((const half *)(page + KS_OFF))[(slot0 + key) * (HD / 32) + d8 / 4] : __float2half(0.0f);
    }
#pragma unroll
    for (int m = 0; m < AttnRaw<AW>::NV; m++) {
        const int i = tid + m * AW * 32, dim = i / (KST / 8), k8 = i % (KST / 8), s = slot0 + k8 * 8;
        r.vlo[m] = *(const uint32_t *)(page + VLO_OFF + dim * (PAGE / 2) + s / 2);
        r.vhb[m] = page[VHI_OFF + dim * (PAGE / 8) + s / 8];
        r.vsr[m] = *(const uint4 *)(page + VS_OFF + ((size_t)(dim / 32) * PAGE + s) * 2);
    }
}

template <int AW>
__device__ __forceinline__ void attn_store(const AttnRaw<AW> &r, half (*Ks)[KLD], half (*Vt)[VLD], int nvalid, int tid) {
#pragma unroll
    for (int m = 0; m < AttnRaw<AW>::NK; m++) {
        const int i = tid + m * AW * 32, key = i / (HD / 8), d8 = i % (HD / 8);
        const half2 s2 = __halves2half2(r.ks[m], r.ks[m]), base = __float2half2_rn(1152.0f);
        const uint32_t ux = r.kw[m].x ^ 0x80808080u, uy = r.kw[m].y ^ 0x80808080u;
        const uint32_t raw[4] = {__byte_perm(ux, 0x64646464u, 0x5140), __byte_perm(ux, 0x64646464u, 0x7362),
                                 __byte_perm(uy, 0x64646464u, 0x5140), __byte_perm(uy, 0x64646464u, 0x7362)};
        uint32_t o[4];
#pragma unroll
        for (int j = 0; j < 4; j++) { const half2 v = __hmul2(__hsub2(*(const half2 *)&raw[j], base), s2); o[j] = *(const uint32_t *)&v; }
        *(uint4 *)&Ks[key][d8 * 8] = make_uint4(o[0], o[1], o[2], o[3]);
    }
#pragma unroll
    for (int m = 0; m < AttnRaw<AW>::NV; m++) {
        const int i = tid + m * AW * 32, dim = i / (KST / 8), k8 = i % (KST / 8);
        const half *sc = (const half *)&r.vsr[m];
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int u = (int)(((r.vlo[m] >> (4 * j)) & 15) | (((r.vhb[m] >> j) & 1) << 4)) - 16;
            v[j] = k8 * 8 + j < nvalid ? __half2float(sc[j]) * u : 0.0f;
        }
        *(uint4 *)&Vt[dim][k8 * 8] = make_uint4(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]), pack_h2(v[4], v[5]), pack_h2(v[6], v[7]));
    }
}

template <int AW>
__global__ __launch_bounds__(AW * 32) void k_attn(AttnArgs a) {
    constexpr int NB = AW == 8 ? 2 : 1;
    extern __shared__ __align__(16) uint8_t attn_smem[];
    half (*Ks)[KST][KLD] = (half (*)[KST][KLD])attn_smem;
    half (*Vt)[HD][VLD] = (half (*)[HD][VLD])(attn_smem + sizeof(half) * NB * KST * KLD);
    const int g = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32, gq = lane / 4, tig = lane % 4;
    const int nrows = a.T * 6, tile = blockIdx.x * AW + warp;
    const int r0 = tile * 16 + gq, r1 = r0 + 8;
    const bool v0 = r0 < nrows, v1 = r1 < nrows;
    const int qp0 = v0 ? a.pos0 + r0 / 6 : -1, qp1 = v1 ? a.pos0 + r1 / 6 : -1;
    const half *q0 = a.Q16 + ((size_t)(v0 ? r0 / 6 : 0) * NH + g * 6 + (v0 ? r0 % 6 : 0)) * HD;
    const half *q1 = a.Q16 + ((size_t)(v1 ? r1 / 6 : 0) * NH + g * 6 + (v1 ? r1 % 6 : 0)) * HD;
    const int last = min(nrows, (blockIdx.x * AW + AW) * 16) - 1;
    const int kbeg = blockIdx.z * a.split_len, kend = min(a.pos0 + last / 6 + 1, kbeg + a.split_len);

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0, l1 = 0;
    float acc[HD / 8][4];
#pragma unroll
    for (int d = 0; d < HD / 8; d++) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.0f;

    AttnRaw<AW> raw;
    if (kbeg < kend) {
        attn_fetch<AW>(raw, kv_head(a.kv, a.layer, g, kbeg), kbeg % PAGE, kend - kbeg, tid);
        attn_store<AW>(raw, Ks[0], Vt[0], kend - kbeg, tid);
    }
    __syncthreads();
    for (int k0 = kbeg, cur = 0; k0 < kend; k0 += KST, cur ^= NB - 1) {
        const int kn = k0 + KST;
        if (kn < kend) attn_fetch<AW>(raw, kv_head(a.kv, a.layer, g, kn), kn % PAGE, kend - kn, tid);

        float s[KST / 8][4];
#pragma unroll
        for (int n = 0; n < KST / 8; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            const int d = kk * 16 + tig * 2;
            const uint32_t A[4] = {*(const uint32_t *)(q0 + d), *(const uint32_t *)(q1 + d), *(const uint32_t *)(q0 + d + 8),
                                   *(const uint32_t *)(q1 + d + 8)};
#pragma unroll
            for (int n = 0; n < KST / 8; n++) {
                const uint32_t B[2] = {*(const uint32_t *)&Ks[cur][n * 8 + gq][d], *(const uint32_t *)&Ks[cur][n * 8 + gq][d + 8]};
                mma_f16(s[n], A, B);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < KST / 8; n++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int key = k0 + n * 8 + tig * 2 + (i & 1);
                if (key >= kend || key > (i < 2 ? qp0 : qp1)) s[n][i] = -INFINITY;
                if (i < 2) mx0 = fmaxf(mx0, s[n][i]); else mx1 = fmaxf(mx1, s[n][i]);
            }
        mx0 = fmaxf(mx0, lane_xor<1>(mx0)); mx0 = fmaxf(mx0, lane_xor<2>(mx0));
        mx1 = fmaxf(mx1, lane_xor<1>(mx1)); mx1 = fmaxf(mx1, lane_xor<2>(mx1));
        float al0 = 1.0f, al1 = 1.0f;
        if (mx0 > m0) { al0 = m0 == -INFINITY ? 0.0f : __expf(m0 - mx0); m0 = mx0; }
        if (mx1 > m1) { al1 = m1 == -INFINITY ? 0.0f : __expf(m1 - mx1); m1 = mx1; }
        float rs0 = 0, rs1 = 0;
#pragma unroll
        for (int n = 0; n < KST / 8; n++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float m = i < 2 ? m0 : m1;
                s[n][i] = m == -INFINITY ? 0.0f : __expf(s[n][i] - m);
                if (i < 2) rs0 += s[n][i]; else rs1 += s[n][i];
            }
        rs0 += lane_xor<1>(rs0); rs0 += lane_xor<2>(rs0);
        rs1 += lane_xor<1>(rs1); rs1 += lane_xor<2>(rs1);
        l0 = l0 * al0 + rs0; l1 = l1 * al1 + rs1;
#pragma unroll
        for (int d = 0; d < HD / 8; d++) { acc[d][0] *= al0; acc[d][1] *= al0; acc[d][2] *= al1; acc[d][3] *= al1; }
        uint32_t P[KST / 16][4];
#pragma unroll
        for (int ks = 0; ks < KST / 16; ks++) {
            P[ks][0] = pack_h2(s[2 * ks][0], s[2 * ks][1]);
            P[ks][1] = pack_h2(s[2 * ks][2], s[2 * ks][3]);
            P[ks][2] = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            P[ks][3] = pack_h2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
        }
#pragma unroll
        for (int dt = 0; dt < HD / 8; dt++)
#pragma unroll
            for (int ks = 0; ks < KST / 16; ks++) {
                const uint32_t B[2] = {*(const uint32_t *)&Vt[cur][dt * 8 + gq][ks * 16 + tig * 2], *(const uint32_t *)&Vt[cur][dt * 8 + gq][ks * 16 + 8 + tig * 2]};
                mma_f16(acc[dt], P[ks], B);
            }
        if constexpr (NB == 1) __syncthreads();
        if (kn < kend) attn_store<AW>(raw, Ks[cur ^ (NB - 1)], Vt[cur ^ (NB - 1)], kend - kn, tid);
        __syncthreads();
    }

#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int r = h ? r1 : r0;
        if (r >= nrows) continue;
        const size_t row = (size_t)(r / 6) * NH + g * 6 + r % 6;
        float *po = a.part_o + ((size_t)blockIdx.z * a.T * NH + row) * HD;
#pragma unroll
        for (int dt = 0; dt < HD / 8; dt++) *(float2 *)&po[dt * 8 + tig * 2] = make_float2(acc[dt][2 * h], acc[dt][2 * h + 1]);
        if (tig == 0) {
            a.part_ml[((size_t)blockIdx.z * a.T * NH + row) * 2] = h ? m1 : m0;
            a.part_ml[((size_t)blockIdx.z * a.T * NH + row) * 2 + 1] = h ? l1 : l0;
        }
    }
}

__global__ __launch_bounds__(256) void k_attn_combine(const float *part_o, const float *part_ml, int nsplit, int rows,
                                                      const float *qg, half *oh, QAct oq) {
    __shared__ float buf[HD], ws[MAX_SPLITS], red[8], part[8][HD];
    const int row = blockIdx.x, d = threadIdx.x;
    const float ms = d < nsplit ? part_ml[((size_t)d * rows + row) * 2] : -INFINITY;
    float M = warp_max(ms);
    if ((d & 31) == 0) red[d >> 5] = M;
    __syncthreads();
    M = red[0];
    for (int i = 1; i < 8; i++) M = fmaxf(M, red[i]);
    __syncthreads();
    const float wt = ms == -INFINITY ? 0.0f : __expf(ms - M);
    if (d < nsplit) ws[d] = wt;
    const float L = block_sum256(d < nsplit ? wt * part_ml[((size_t)d * rows + row) * 2 + 1] : 0.0f, red);
    {
        const int v = d / 32, ln = d % 32;
        float4 a0 = {0, 0, 0, 0}, a1 = {0, 0, 0, 0};
#pragma unroll 4
        for (int s = v; s < nsplit; s += 8) {
            const float w = ws[s];
            const float4 *src = (const float4 *)(part_o + ((size_t)s * rows + row) * HD) + 2 * ln;
            const float4 x0 = src[0], x1 = src[1];
            a0.x += w * x0.x; a0.y += w * x0.y; a0.z += w * x0.z; a0.w += w * x0.w;
            a1.x += w * x1.x; a1.y += w * x1.y; a1.z += w * x1.z; a1.w += w * x1.w;
        }
        *(float4 *)&part[v][8 * ln] = a0;
        *(float4 *)&part[v][8 * ln + 4] = a1;
    }
    __syncthreads();
    float o = 0;
#pragma unroll
    for (int v = 0; v < 8; v++) o += part[v][d];
    const float y = fwht256(L > 0 ? o / L : 0.0f, buf) * sigmoid(qg[(size_t)row * 2 * HD + HD + d]);
    const size_t idx = (size_t)row * HD + d;
    if (oh) oh[idx] = __float2half(y);
    if (oq.q) quant_put(oq, idx, y);
}

static int sm_count() {
    static const int n = [] { int dev = 0, v = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev); return v; }();
    return n;
}

void attention(AttnArgs a, half *oh, QAct oq, cudaStream_t st) {
    const int ntiles = (a.T * 6 + 15) / 16;
    const int nkeys = a.pos0 + a.T;
    a.out_h = oh;
    const int aw = ntiles >= 64 ? 8 : 4;
    const int rowblocks = (ntiles + aw - 1) / aw, wgs = rowblocks * NKV;
    if (a.T <= 8) {
        a.split_len = nkeys <= 65536 ? 256 : nkeys <= 131072 ? 512 : 1024;
    } else {
        const int n = std::max(1, std::min({(4 * sm_count() + wgs - 1) / wgs, (nkeys + KST - 1) / KST, MAX_SPLITS, MAX_SPLITS * 8 / a.T}));
        a.split_len = ((nkeys + n - 1) / n + KST - 1) / KST * KST;
    }
    a.nsplit = (nkeys + a.split_len - 1) / a.split_len;
    if (a.nsplit > MAX_SPLITS) FAIL("attention: %d keys need %d splits", nkeys, a.nsplit);
    a.partial = 1;
    const dim3 grid(rowblocks, NKV, a.nsplit);
    constexpr int SMEM = sizeof(half) * (KST * KLD + HD * VLD);
    static const bool attr = [] { CK(cudaFuncSetAttribute(k_attn<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * SMEM)); return true; }();
    (void)attr;
    if (aw == 8) k_attn<8><<<grid, 256, 2 * SMEM, st>>>(a);
    else k_attn<4><<<grid, 128, SMEM, st>>>(a);
    k_attn_combine<<<a.T * NH, 256, 0, st>>>(a.part_o, a.part_ml, a.nsplit, a.T * NH, a.qg, oh, oq);
}
