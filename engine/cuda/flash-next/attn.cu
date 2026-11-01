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

__device__ __forceinline__ int mrope_section(int f) { return f % 3 == 1 && f < 33 ? 1 : f % 3 == 2 && f < 30 ? 2 : 0; }

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
            const int rp = rpos ? rpos[t * 3 + mrope_section(f)] : rope0 + t;
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
__global__ __launch_bounds__(AW * 32) void k_attn(AttnArgs a) {
    __shared__ __align__(16) half Ks[KST][KLD];
    __shared__ __align__(16) half Vt[HD][VLD];
    const int g = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32, gq = lane / 4, tig = lane % 4;
    const int nrows = a.T * QPK, tile = blockIdx.x * AW + warp;
    const int r0 = tile * 16 + gq, r1 = r0 + 8;
    const bool v0 = r0 < nrows, v1 = r1 < nrows;
    const int qp0 = v0 ? a.pos0 + r0 / QPK : -1, qp1 = v1 ? a.pos0 + r1 / QPK : -1;
    const half *q0 = a.Q16 + ((size_t)(v0 ? r0 / QPK : 0) * NH + g * QPK + (v0 ? r0 % QPK : 0)) * HD;
    const half *q1 = a.Q16 + ((size_t)(v1 ? r1 / QPK : 0) * NH + g * QPK + (v1 ? r1 % QPK : 0)) * HD;
    const int last = min(nrows, (blockIdx.x * AW + AW) * 16) - 1;
    const int kbeg = blockIdx.z * a.split_len, kend = min(a.pos0 + last / QPK + 1, kbeg + a.split_len);

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0, l1 = 0;
    float acc[HD / 8][4];
#pragma unroll
    for (int d = 0; d < HD / 8; d++) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.0f;

    for (int k0 = kbeg; k0 < kend; k0 += KST) {
        __syncthreads();
        const uint8_t *page = kv_head(a.kv, a.layer, g, k0);
        const int slot0 = k0 % PAGE, nvalid = kend - k0;
        for (int i = tid; i < KST * HD / 8; i += AW * 32) {
            const int key = i / (HD / 8), d8 = i % (HD / 8);
            const uint2 w = *(const uint2 *)(page + (size_t)(slot0 + key) * HD + d8 * 8);
            const float s = key < nvalid ? __half2float(((const half *)(page + KS_OFF))[(slot0 + key) * (HD / 32) + d8 / 4]) : 0.0f;
            const int8_t *b = (const int8_t *)&w;
            uint32_t o[4];
#pragma unroll
            for (int j = 0; j < 4; j++) o[j] = pack_h2(s * b[2 * j], s * b[2 * j + 1]);
            *(uint4 *)&Ks[key][d8 * 8] = make_uint4(o[0], o[1], o[2], o[3]);
        }
        for (int i = tid; i < HD * (KST / 8); i += AW * 32) {
            const int dim = i / (KST / 8), k8 = i % (KST / 8), s = slot0 + k8 * 8;
            const uint32_t lo = *(const uint32_t *)(page + VLO_OFF + dim * (PAGE / 2) + s / 2);
            const uint32_t hb = page[VHI_OFF + dim * (PAGE / 8) + s / 8];
            const uint4 sr = *(const uint4 *)(page + VS_OFF + ((size_t)(dim / 32) * PAGE + s) * 2);
            const half *sc = (const half *)&sr;
            float v[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int u = (int)(((lo >> (4 * j)) & 15) | (((hb >> j) & 1) << 4)) - 16;
                v[j] = k8 * 8 + j < nvalid ? __half2float(sc[j]) * u : 0.0f;
            }
            *(uint4 *)&Vt[dim][k8 * 8] = make_uint4(pack_h2(v[0], v[1]), pack_h2(v[2], v[3]), pack_h2(v[4], v[5]), pack_h2(v[6], v[7]));
        }
        __syncthreads();

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
                const uint32_t B[2] = {*(const uint32_t *)&Ks[n * 8 + gq][d], *(const uint32_t *)&Ks[n * 8 + gq][d + 8]};
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
                const uint32_t B[2] = {*(const uint32_t *)&Vt[dt * 8 + gq][ks * 16 + tig * 2], *(const uint32_t *)&Vt[dt * 8 + gq][ks * 16 + 8 + tig * 2]};
                mma_f16(acc[dt], P[ks], B);
            }
    }

#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int r = h ? r1 : r0;
        if (r >= nrows) continue;
        const size_t row = (size_t)(r / QPK) * NH + g * QPK + r % QPK;
        float *po = a.part_o + ((size_t)blockIdx.z * a.T * NH + row) * HD;
#pragma unroll
        for (int dt = 0; dt < HD / 8; dt++) *(float2 *)&po[dt * 8 + tig * 2] = make_float2(acc[dt][2 * h], acc[dt][2 * h + 1]);
        if (tig == 0) {
            a.part_ml[((size_t)blockIdx.z * a.T * NH + row) * 2] = h ? m1 : m0;
            a.part_ml[((size_t)blockIdx.z * a.T * NH + row) * 2 + 1] = h ? l1 : l0;
        }
    }
}

static int sm_count() {
    static const int n = [] { int dev = 0, v = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev); return v; }();
    return n;
}

__global__ __launch_bounds__(256) void k_attn_combine(const float *part_o, const float *part_ml, int nsplit, int rows,
                                                      const float *qg, half *oh, QAct oq) {
    __shared__ float buf[HD];
    const int row = blockIdx.x, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; s++) M = fmaxf(M, part_ml[((size_t)s * rows + row) * 2]);
    float L = 0, o = 0;
    for (int s = 0; s < nsplit; s++) {
        const float ms = part_ml[((size_t)s * rows + row) * 2];
        if (ms == -INFINITY) continue;
        const float w = __expf(ms - M);
        L += w * part_ml[((size_t)s * rows + row) * 2 + 1];
        o += w * part_o[((size_t)s * rows + row) * HD + d];
    }
    const float y = fwht256(L > 0 ? o / L : 0.0f, buf) * sigmoid(qg[(size_t)row * 2 * HD + HD + d]);
    const size_t idx = (size_t)row * HD + d;
    if (oh) oh[idx] = __float2half(y);
    if (oq.q) {
        const float sd = warp_max(fabsf(y)) / 127.0f;
        oq.q[idx] = (int8_t)__float2int_rn(sd > 0 ? y / sd : 0.0f);
        if ((d & 31) == 0) oq.d[idx / 32] = sd;
    }
}

void attention(AttnArgs a, half *oh, QAct oq, cudaStream_t st) {
    const int ntiles = (a.T * QPK + 15) / 16;
    const int nkeys = a.pos0 + a.T;
    a.out_h = oh;
    const int aw = ntiles >= 64 ? 8 : 4;
    const int rowblocks = (ntiles + aw - 1) / aw, wgs = rowblocks * NKV;
    if (a.T <= 8) {
        a.split_len = 256;
    } else {
        const int n = std::max(1, std::min({(4 * sm_count() + wgs - 1) / wgs, (nkeys + KST - 1) / KST, PART_ROWS / a.T}));
        a.split_len = ((nkeys + n - 1) / n + KST - 1) / KST * KST;
    }
    a.nsplit = (nkeys + a.split_len - 1) / a.split_len;
    if (a.nsplit * a.T > PART_ROWS) FAIL("attention: %d tokens x %d splits exceed the partial buffers", a.T, a.nsplit);
    a.partial = 1;
    const dim3 grid(rowblocks, NKV, a.nsplit);
    if (aw == 8) k_attn<8><<<grid, 256, 0, st>>>(a);
    else k_attn<4><<<grid, 128, 0, st>>>(a);
    k_attn_combine<<<a.T * NH, 256, 0, st>>>(a.part_o, a.part_ml, a.nsplit, a.T * NH, a.qg, oh, oq);
}


__device__ __forceinline__ half *idx_pool(const KVRef &kv, int layer, int block) {
    const int cell = block * KPOOL;
    return (half *)(kv.pool + (size_t)kv.pages[cell / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + NKV * KV_HEAD) +
           (size_t)((cell % PAGE) / KPOOL) * IDX_D;
}

__device__ __forceinline__ float norm_rope128(float x, const float *w, Pos3 p, float *red, float *buf) {
    const int i = threadIdx.x;
    float ss = warp_sum(x * x);
    if ((i & 31) == 0) red[i >> 5] = ss;
    __syncthreads();
    ss = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    x *= rsqrtf(ss / IDX_D + EPS) * w[i];
    if (i < ROT) buf[i] = x;
    __syncthreads();
    if (i < ROT) {
        const int f = i % (ROT / 2), sec = mrope_section(f);
        float s, c;
        sincosf((sec == 0 ? p.t : sec == 1 ? p.h : p.w) * powf(ROPE_BASE, -2.0f * f / ROT), &s, &c);
        x = i < ROT / 2 ? x * c - buf[i + ROT / 2] * s : x * c + buf[i - ROT / 2] * s;
    }
    __syncthreads();
    return x;
}
__device__ __forceinline__ Pos3 pos_at(int t, int rope0, const int *rpos) {
    return rpos ? Pos3{rpos[t * 3], rpos[t * 3 + 1], rpos[t * 3 + 2]} : Pos3{rope0 + t, rope0 + t, rope0 + t};
}

__global__ __launch_bounds__(128) void k_idx_prep(const float *kraw, const float *qraw, const float *qnw, int rope0, const int *rpos,
                                                  half *praw, half *q16) {
    __shared__ float red[4], buf[ROT];
    const int job = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    if (job == 0) {
        praw[(size_t)t * IDX_D + i] = __float2half(kraw[(size_t)t * IDX_D + i]);
        return;
    }
    const int h = job - 1;
    const float x = norm_rope128(qraw[((size_t)t * IDX_H + h) * IDX_D + i], qnw, pos_at(t, rope0, rpos), red, buf);
    q16[((size_t)t * IDX_H + h) * IDX_D + i] = __float2half(x);
}

__global__ __launch_bounds__(128) void k_idx_pool(int b0, int pos0, int rope0, const int *rpos, Pos3 tail_pos, const half *tail,
                                                  const half *praw, KVRef kv, int layer, const float *knw) {
    __shared__ float red[4], buf[ROT];
    const int b = b0 + blockIdx.x, i = threadIdx.x;
    float x = 0;
    for (int c = 0; c < KPOOL; c++) {
        const int cell = b * KPOOL + c;
        x += __half2float(cell < pos0 ? tail[(cell % KPOOL) * IDX_D + i] : praw[(size_t)(cell - pos0) * IDX_D + i]);
    }
    const int first = b * KPOOL;
    x = norm_rope128(x * (1.0f / KPOOL), knw, first < pos0 ? tail_pos : pos_at(first - pos0, rope0, rpos), red, buf);
    idx_pool(kv, layer, b)[i] = __float2half(x);
}

__global__ __launch_bounds__(128) void k_idx_tail(half *tail, const half *praw, size_t praw_layer, int pos0, int n_end) {
    half *tl = tail + (size_t)blockIdx.x * IDX_TAIL;
    const half *pr = praw + (size_t)blockIdx.x * praw_layer;
    for (int cell = max(pos0, n_end / KPOOL * KPOOL); cell < n_end; cell++)
        tl[(cell % KPOOL) * IDX_D + threadIdx.x] = pr[(size_t)(cell - pos0) * IDX_D + threadIdx.x];
}

void idx_prep(const float *kraw, const float *qraw, const float *qnw, const float *knw, int T, int pos0, int rope0, const int *rpos,
              Pos3 tail_pos, KVRef kv, int layer, half *tail, half *praw, bool keep_tail, half *q16, cudaStream_t st) {
    k_idx_prep<<<dim3(1 + IDX_H, T), 128, 0, st>>>(kraw, qraw, qnw, rope0, rpos, praw, q16);
    const int first = pos0 / KPOOL, last = (pos0 + T) / KPOOL - 1;
    if (last >= first) k_idx_pool<<<last - first + 1, 128, 0, st>>>(first, pos0, rope0, rpos, tail_pos, tail, praw, kv, layer, knw);
    if (keep_tail) k_idx_tail<<<1, 128, 0, st>>>(tail, praw, 0, pos0, pos0 + T);
}

void idx_tail_commit(half *tails, const half *praw, size_t praw_layer, int layers, int pos0, int n_end, cudaStream_t st) {
    k_idx_tail<<<layers, 128, 0, st>>>(tails, praw, praw_layer, pos0, n_end);
}

__global__ __launch_bounds__(256) void k_idx_score(const half *q16, int pos0, int nblk, KVRef kv, int layer, float *score) {
    __shared__ float q[IDX_H][IDX_D];
    const int t = blockIdx.y, b = blockIdx.x * 256 + threadIdx.x, pos = pos0 + t;
    for (int i = threadIdx.x; i < IDX_H * IDX_D; i += 256) q[i / IDX_D][i % IDX_D] = __half2float(q16[(size_t)t * IDX_H * IDX_D + i]);
    __syncthreads();
    if (b >= nblk) return;
    float s = -INFINITY;
    if ((b + 1) * KPOOL <= pos + 1) {
        const half2 *k = (const half2 *)idx_pool(kv, layer, b);
        float acc[IDX_H] = {};
        for (int i = 0; i < IDX_D / 2; i++) {
            const float2 kv2 = __half22float2(k[i]);
#pragma unroll
            for (int h = 0; h < IDX_H; h++) acc[h] += q[h][2 * i] * kv2.x + q[h][2 * i + 1] * kv2.y;
        }
        s = 0;
#pragma unroll
        for (int h = 0; h < IDX_H; h++) s += fmaxf(acc[h], 0.0f);
        s *= 0.08838834764831845f;
    }
    score[(size_t)t * nblk + b] = s;
}

__global__ __launch_bounds__(1024) void k_idx_select(const float *score, int pos0, int nblk, int *sel) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t prefix, need, eq_taken;
    __shared__ int nout;
    const int t = blockIdx.x, tid = threadIdx.x, pos = pos0 + t;
    const int nvis = min(nblk, (pos + 1) / KPOOL);
    const float *s = score + (size_t)t * nblk;
    int *out = sel + (size_t)t * (SEL_BLOCKS + 1);
    const int want = min(nvis, SEL_BLOCKS);
    if (tid == 0) { prefix = 0; need = want; nout = 0; eq_taken = 0; }
    __syncthreads();
    uint32_t mask = 0;
    for (int shift = 24; shift >= 0 && want < nvis; shift -= 8) {
        for (int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int b = tid; b < nvis; b += 1024) {
            const uint32_t u = __float_as_uint(s[b]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0; int d = 255;
            for (; d > 0; d--) { if (acc + hist[d] >= need) break; acc += hist[d]; }
            need -= acc; prefix |= (uint32_t)d << shift;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const bool all = want >= nvis;
    for (int base = 0; base < nvis; base += 1024) {
        const int b = base + tid;
        bool take = false, eq = false;
        if (b < nvis) {
            const uint32_t u = __float_as_uint(s[b]);
            take = all || u > thr;
            eq = !all && u == thr;
        }
        const unsigned long long bt = (unsigned long long)__ballot_sync(0xffffffffu, take), be = (unsigned long long)__ballot_sync(0xffffffffu, eq);
        __shared__ int wt[32], we[32];
        const int wave = tid / 32, lane = tid % 32;
        const unsigned long long below = (1ull << lane) - 1;
        if (lane == 0) { wt[wave] = __popcll(bt); we[wave] = __popcll(be); }
        __syncthreads();
        int ot = 0, oe = 0;
        for (int w = 0; w < wave; w++) { ot += wt[w]; oe += we[w]; }
        int tot_t = 0, tot_e = 0;
        for (int w = 0; w < 32; w++) { tot_t += wt[w]; tot_e += we[w]; }
        const int my_e = oe + __popcll(be & below);
        const bool take_eq = eq && (int)eq_taken + my_e < (int)need;
        const unsigned long long bte = (unsigned long long)__ballot_sync(0xffffffffu, take || take_eq);
        __shared__ int wte[32];
        if (lane == 0) wte[wave] = __popcll(bte);
        __syncthreads();
        int o = 0;
        for (int w = 0; w < wave; w++) o += wte[w];
        if (take || take_eq) out[nout + o + __popcll(bte & below)] = b;
        __syncthreads();
        if (tid == 0) {
            int add = 0;
            for (int w = 0; w < 32; w++) add += wte[w];
            nout += add;
            eq_taken += min((int)need - (int)eq_taken, tot_e);
            (void)tot_t;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int n = nout;
        if ((pos + 1) % KPOOL != 0) out[n++] = (pos + 1) / KPOOL;
        for (; n < SEL_BLOCKS + 1; n++) out[n] = -1;
    }
}

constexpr int SEL_BINS = 1024;
__global__ __launch_bounds__(256) void k_idx_score2(const half *q16, int pos0, int nblk, KVRef kv, int layer, float *score,
                                                    uint32_t *hist) {
    __shared__ uint32_t lh[SEL_BINS];
    const int t = blockIdx.y, tid = threadIdx.x, g = tid / 16, l = tid % 16, pos = pos0 + t;
    const int nvis = min(nblk, (pos + 1) / KPOOL);
    for (int i = tid; i < SEL_BINS; i += 256) lh[i] = 0;
    float q[IDX_H][8];
#pragma unroll
    for (int h = 0; h < IDX_H; h++) {
        const int4 v = *(const int4 *)(q16 + ((size_t)t * IDX_H + h) * IDX_D + l * 8);
        const half2 *hv = (const half2 *)&v;
#pragma unroll
        for (int k = 0; k < 4; k++) { const float2 f = __half22float2(hv[k]); q[h][2 * k] = f.x; q[h][2 * k + 1] = f.y; }
    }
    __syncthreads();
    const int b0 = blockIdx.x * 256;
#pragma unroll 4
    for (int it = 0; it < 16; it++) {
        const int b = b0 + it * 16 + g;
        if (b0 + it * 16 + (g & ~1) >= nblk) break;
        float acc[IDX_H] = {};
        if (b < nvis) {
            const int4 v = *(const int4 *)(idx_pool(kv, layer, b) + l * 8);
            const half2 *hv = (const half2 *)&v;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float2 f = __half22float2(hv[k]);
#pragma unroll
                for (int h = 0; h < IDX_H; h++) acc[h] += q[h][2 * k] * f.x + q[h][2 * k + 1] * f.y;
            }
        }
#pragma unroll
        for (int h = 0; h < IDX_H; h++) {
            acc[h] += lane_xor<8>(acc[h]); acc[h] += lane_xor<4>(acc[h]); acc[h] += lane_xor<2>(acc[h]); acc[h] += lane_xor<1>(acc[h]);
        }
        if (l == 0) {
            float sc = -INFINITY;
            if (b < nvis) {
                sc = 0;
#pragma unroll
                for (int h = 0; h < IDX_H; h++) sc += fmaxf(acc[h], 0.0f);
                sc *= 0.08838834764831845f;
                atomicAdd(&lh[__float_as_uint(sc) >> 22], 1u);
            }
            if (b < nblk) score[(size_t)t * nblk + b] = sc;
        }
    }
    __syncthreads();
    for (int i = tid; i < SEL_BINS; i += 256) if (lh[i]) atomicAdd(&hist[(size_t)t * SEL_BINS + i], lh[i]);
}

__device__ __forceinline__ int wave_incl(int v, int lane) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int u = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += u; }
    return v;
}
__device__ __forceinline__ int block_excl(int v, int *wsum, int *tot) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int inc = wave_incl(v, lane);
    if (lane == 31) wsum[wave] = inc;
    __syncthreads();
    if (wave == 0) {
        const int w = wave_incl(wsum[lane], lane);
        wsum[lane] = w;
    }
    __syncthreads();
    *tot = wsum[31];
    return inc - v + (wave ? wsum[wave - 1] : 0);
}
__device__ __forceinline__ void digit_of(const uint32_t *h, int n, uint32_t need, int lane, int &d_out, uint32_t &above) {
    const int per = n / 32;
    uint32_t loc = 0;
    for (int k = 0; k < per; k++) loc += h[n - 1 - (lane * per + k)];
    const int inc = wave_incl((int)loc, lane), exc = inc - (int)loc;
    const bool mine = (uint32_t)exc < need && (uint32_t)inc >= need;
    const unsigned long long m = (unsigned long long)__ballot_sync(0xffffffffu, mine);
    const int owner = m ? __ffsll((long long)m) - 1 : 31;
    int d = 0; uint32_t ab = 0;
    if (lane == owner) {
        uint32_t acc = (uint32_t)exc;
        int k = 0;
        for (; k < per - 1; k++) { const uint32_t c = h[n - 1 - (lane * per + k)]; if (acc + c >= need) break; acc += c; }
        d = n - 1 - (lane * per + k); ab = acc;
    }
    d_out = __shfl_sync(0xffffffffu, d, owner); above = (uint32_t)__shfl_sync(0xffffffffu, (int)ab, owner);
}

__global__ __launch_bounds__(1024) void k_idx_select2(const float *score, const uint32_t *hist, int pos0, int nblk, int *sel) {
    __shared__ uint32_t hb[SEL_BINS], h256[256], hw[32][256];
    __shared__ uint32_t s_prefix, s_need, s_mask;
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid % 32, wave = tid / 32, pos = pos0 + t;
    const int nvis = min(nblk, (pos + 1) / KPOOL);
    const float *s = score + (size_t)t * nblk;
    int *out = sel + (size_t)t * (SEL_BLOCKS + 1);
    const int want = min(nvis, SEL_BLOCKS);
    const bool all = want >= nvis;
    if (!all) {
        hb[tid] = hist[(size_t)t * SEL_BINS + tid];
        __syncthreads();
        if (wave == 0) {
            int d; uint32_t above;
            digit_of(hb, SEL_BINS, (uint32_t)want, lane, d, above);
            if (lane == 0) { s_prefix = (uint32_t)d << 22; s_mask = 0xFFC00000u; s_need = want - above; }
        }
        __syncthreads();
        for (int shift = 14; shift >= -6; shift -= 8) {
            const int sh_ = shift < 0 ? 0 : shift, bits = shift < 0 ? 6 : 8;
            for (int i = tid; i < 32 * 256; i += 1024) hw[i / 256][i % 256] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix, mask = s_mask;
            for (int b0 = tid; b0 < nvis; b0 += 8 * 1024) {
                uint32_t u[8];
#pragma unroll
                for (int k = 0; k < 8; k++) { const int b = b0 + k * 1024; u[k] = b < nvis ? __float_as_uint(s[b]) : ~prefix; }
#pragma unroll
                for (int k = 0; k < 8; k++)
                    if ((u[k] & mask) == prefix) atomicAdd(&hw[wave][(u[k] >> sh_) & ((1u << bits) - 1)], 1u);
            }
            __syncthreads();
            if (tid < 256) { uint32_t c = 0; for (int v = 0; v < 32; v++) c += hw[v][tid]; h256[tid] = c; }
            __syncthreads();
            if (wave == 0) {
                int d; uint32_t above;
                digit_of(h256, 1 << bits, s_need, lane, d, above);
                if (lane == 0) { s_need -= above; s_prefix = prefix | ((uint32_t)d << sh_); s_mask = mask | (((1u << bits) - 1) << sh_); }
            }
            __syncthreads();
        }
    }
    const uint32_t thr = all ? 0 : s_prefix, need = all ? 0 : s_need;
    const int per = ((nvis + 31) / 32 + 31) / 32 * 32, lo = min(nvis, wave * per), hi = min(nvis, lo + per);
    int n_take = 0, n_eq = 0;
    for (int b0 = lo; b0 < hi; b0 += 32 * 4) {
        uint32_t u[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const int b = b0 + k * 32 + lane; u[k] = b < hi ? __float_as_uint(s[b]) : 0u; }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int b = b0 + k * 32 + lane;
            n_take += __popcll((unsigned long long)__ballot_sync(0xffffffffu, b < hi && (all || u[k] > thr)));
            n_eq += __popcll((unsigned long long)__ballot_sync(0xffffffffu, b < hi && !all && u[k] == thr));
        }
    }
    __shared__ int w_take[32], w_eq[32];
    if (lane == 0) { w_take[wave] = n_take; w_eq[wave] = n_eq; }
    __syncthreads();
    int eq_before = 0, out_before = 0;
    for (int v = 0; v < wave; v++) eq_before += w_eq[v];
    int tot = 0;
    for (int v = 0; v < 32; v++) {
        const int e = max(0, min(w_eq[v], (int)need - (tot == 0 && v == 0 ? 0 : 0)));
        (void)e;
    }
    {
        int eqb = 0, ob = 0;
        for (int v = 0; v < 32; v++) {
            const int taken_eq = max(0, min(w_eq[v], (int)need - eqb));
            if (v < wave) ob += w_take[v] + taken_eq;
            tot += w_take[v] + taken_eq;
            eqb += w_eq[v];
        }
        out_before = ob;
    }
    int w = out_before, eq_seen = eq_before;
    for (int b0 = lo; b0 < hi; b0 += 32 * 4) {
        uint32_t u[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const int b = b0 + k * 32 + lane; u[k] = b < hi ? __float_as_uint(s[b]) : 0u; }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int b = b0 + k * 32 + lane;
            const bool in = b < hi;
            const bool eq = in && !all && u[k] == thr;
            const unsigned long long me = (unsigned long long)__ballot_sync(0xffffffffu, eq), below = (1ull << lane) - 1;
            const bool take_eq = eq && eq_seen + __popcll(me & below) < (int)need;
            const bool take = in && (all || u[k] > thr || take_eq);
            const unsigned long long mt = (unsigned long long)__ballot_sync(0xffffffffu, take);
            if (take) out[w + __popcll(mt & below)] = b;
            w += __popcll(mt);
            eq_seen += __popcll(me);
        }
    }
    if (tid == 0) {
        int n = tot;
        if ((pos + 1) % KPOOL != 0) out[n++] = (pos + 1) / KPOOL;
        for (; n < SEL_BLOCKS + 1; n++) out[n] = -1;
    }
}

__global__ __launch_bounds__(128) void k_idx_score_mm(const half *q16, int G, int pos0, int nblk, KVRef kv, int layer, float *score) {
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, gq = lane / 4, tig = lane % 4;
    const int t0 = blockIdx.y * 16, bb = (blockIdx.x * 4 + wave) * 16;
    const half *kp[2] = {idx_pool(kv, layer, min(bb + gq, nblk - 1)), idx_pool(kv, layer, min(bb + 8 + gq, nblk - 1))};
    const int ta = min(t0 + gq, G - 1), tb = min(t0 + 8 + gq, G - 1);
    float sum[2][4] = {};
#pragma unroll
    for (int h = 0; h < IDX_H; h++) {
        const half *qa = q16 + ((size_t)ta * IDX_H + h) * IDX_D, *qb = q16 + ((size_t)tb * IDX_H + h) * IDX_D;
        float acc[2][4] = {};
#pragma unroll
        for (int k = 0; k < IDX_D / 16; k++) {
            const int kk = 16 * k + 2 * tig;
            const uint32_t A[4] = {*(const uint32_t *)(qa + kk), *(const uint32_t *)(qb + kk), *(const uint32_t *)(qa + kk + 8),
                                   *(const uint32_t *)(qb + kk + 8)};
#pragma unroll
            for (int n = 0; n < 2; n++) {
                const uint32_t B[2] = {*(const uint32_t *)(kp[n] + kk), *(const uint32_t *)(kp[n] + kk + 8)};
                mma_f16(acc[n], A, B);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
            for (int i = 0; i < 4; i++) sum[n][i] += fmaxf(acc[n][i], 0.0f);
    }
#pragma unroll
    for (int n = 0; n < 2; n++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int t = t0 + gq + 8 * (i >> 1), b = bb + 8 * n + 2 * tig + (i & 1);
            if (t >= G || b >= nblk) continue;
            const bool vis = (b + 1) * KPOOL <= pos0 + t + 1;
            score[(size_t)t * nblk + b] = vis ? sum[n][i] * 0.08838834764831845f : -INFINITY;
        }
}

__global__ __launch_bounds__(256) void k_idx_hist(const float *score, int pos0, int nblk, uint32_t *hist) {
    __shared__ uint32_t lh[SEL_BINS];
    const int t = blockIdx.y, tid = threadIdx.x;
    const int nvis = min(nblk, (pos0 + t + 1) / KPOOL);
    for (int i = tid; i < SEL_BINS; i += 256) lh[i] = 0;
    __syncthreads();
    const int b1 = min(nvis, (int)(blockIdx.x + 1) * 4096);
    for (int b = blockIdx.x * 4096 + tid; b < b1; b += 256) atomicAdd(&lh[__float_as_uint(score[(size_t)t * nblk + b]) >> 22], 1u);
    __syncthreads();
    for (int i = tid; i < SEL_BINS; i += 256) if (lh[i]) atomicAdd(&hist[(size_t)t * SEL_BINS + i], lh[i]);
}

void idx_select(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, uint32_t *hist, int *sel, cudaStream_t st) {
    CK(cudaMemsetAsync(hist, 0, (size_t)G * SEL_BINS * 4, st));
    if (G == 1) k_idx_score2<<<dim3((nblk + 255) / 256, 1), 256, 0, st>>>(q16, pos0, nblk, kv, layer, score, hist);
    else {
        k_idx_score_mm<<<dim3((nblk + 63) / 64, (G + 15) / 16), 128, 0, st>>>(q16, G, pos0, nblk, kv, layer, score);
        k_idx_hist<<<dim3((nblk + 4095) / 4096, G), 256, 0, st>>>(score, pos0, nblk, hist);
    }
    k_idx_select2<<<G, 1024, 0, st>>>(score, hist, pos0, nblk, sel);
}
void idx_select_ref(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, int *sel, cudaStream_t st) {
    k_idx_score<<<dim3((nblk + 255) / 256, G), 256, 0, st>>>(q16, pos0, nblk, kv, layer, score);
    k_idx_select<<<G, 1024, 0, st>>>(score, pos0, nblk, sel);
}

constexpr int SST = 32;

__global__ __launch_bounds__(256) void k_attn_sel(AttnArgs a, const int *sel, int stages_per_split) {
    __shared__ __align__(16) half Qs[16][HD + 8];
    __shared__ float rmx[8][16];
    __shared__ __align__(16) half Ps[8][16][16 + 8];
    __shared__ int bid[SEL_BLOCKS + 1], bpg[SEL_BLOCKS + 1];
    const int t = blockIdx.x, g = blockIdx.y, split = blockIdx.z, tid = threadIdx.x;
    const int w = tid / 32, lane = tid % 32, gq = lane / 4, tig = lane % 4;
    const int qpos = a.pos0 + t;
    const int *list = sel + (size_t)t * (SEL_BLOCKS + 1);
    const int bbeg = split * stages_per_split * SST, nb = min(SEL_BLOCKS + 1, bbeg + stages_per_split * SST) - bbeg;
    for (int i = tid; i < 16 * (HD / 8); i += 256) {
        const int r = i / (HD / 8), c = i % (HD / 8);
        int4 v = {0, 0, 0, 0};
        if (r < QPK) v = *(const int4 *)(a.Q16 + ((size_t)t * NH + g * QPK + r) * HD + c * 8);
        *(int4 *)&Qs[r][c * 8] = v;
    }
    for (int i = tid; i < nb; i += 256) {
        const int id = list[bbeg + i];
        bid[i] = id;
        bpg[i] = id >= 0 ? a.kv.pages[id * KPOOL / PAGE] : a.kv.pages[0];
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0, l1 = 0;
    float acc[4][4] = {};
    __syncthreads();
    const size_t head_off = (size_t)a.layer * KV_LAYER + (size_t)g * KV_HEAD;
    auto block_base = [&](int i) { return a.kv.pool + (size_t)bpg[i] * KV_PAGE + head_off; };

    for (int s0 = 0; s0 < nb; s0 += SST) {
        float sc[2][4] = {};
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const int key = 8 * n + gq, bi = s0 + 4 * w + key / 4;
            const int id = bi < nb ? bid[bi] : -1;
            const uint8_t *page = block_base(bi < nb ? bi : 0);
            const int slot = (id >= 0 ? id * KPOOL % PAGE : 0) + key % 4;
            const int4 *kq = (const int4 *)((const int8_t *)page + slot * HD + tig * 64);
            const half *ks = (const half *)(page + KS_OFF) + slot * (HD / 32) + tig * 2;
            int4 kr[4];
#pragma unroll
            for (int c = 0; c < 4; c++) kr[c] = kq[c];
            const float ka = __half2float(ks[0]), kb = __half2float(ks[1]);
#pragma unroll
            for (int st4 = 0; st4 < 16; st4++) {
                const int8_t *b = (const int8_t *)&kr[st4 / 4] + (st4 % 4) * 4;
                const float ksc = st4 < 8 ? ka : kb;
                const uint32_t B[2] = {pack_h2(ksc * b[0], ksc * b[1]), pack_h2(ksc * b[2], ksc * b[3])};
                const int d = tig * 64 + 4 * st4;
                const uint32_t A[4] = {*(const uint32_t *)&Qs[gq][d], *(const uint32_t *)&Qs[gq + 8][d], *(const uint32_t *)&Qs[gq][d + 2],
                                       *(const uint32_t *)&Qs[gq + 8][d + 2]};
                mma_f16(sc[n], A, B);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int key = 8 * n + 2 * tig + (i & 1), kb = s0 + 4 * w + key / 4;
                const int kid = kb < nb ? bid[kb] : -1;
                const int row = gq + 8 * (i >> 1);
                if (kid < 0 || kid * KPOOL + key % 4 > qpos || row >= QPK) sc[n][i] = -INFINITY;
                if (i < 2) mx0 = fmaxf(mx0, sc[n][i]); else mx1 = fmaxf(mx1, sc[n][i]);
            }
        mx0 = fmaxf(mx0, lane_xor<1>(mx0)); mx0 = fmaxf(mx0, lane_xor<2>(mx0));
        mx1 = fmaxf(mx1, lane_xor<1>(mx1)); mx1 = fmaxf(mx1, lane_xor<2>(mx1));
        if (tig == 0) { rmx[w][gq] = mx0; rmx[w][gq + 8] = mx1; }
        __syncthreads();
        {
            float M0 = m0, M1 = m1;
#pragma unroll
            for (int ww = 0; ww < 8; ww++) { M0 = fmaxf(M0, rmx[ww][gq]); M1 = fmaxf(M1, rmx[ww][gq + 8]); }
            const float al0 = M0 == -INFINITY ? 1.0f : __expf(m0 - M0), al1 = M1 == -INFINITY ? 1.0f : __expf(m1 - M1);
            m0 = M0; m1 = M1;
            float rs0 = 0, rs1 = 0;
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float M = i < 2 ? M0 : M1;
                    sc[n][i] = M == -INFINITY ? 0.0f : __expf(sc[n][i] - M);
                    if (i < 2) rs0 += sc[n][i]; else rs1 += sc[n][i];
                }
            rs0 += lane_xor<1>(rs0); rs0 += lane_xor<2>(rs0);
            rs1 += lane_xor<1>(rs1); rs1 += lane_xor<2>(rs1);
            l0 = l0 * al0 + rs0; l1 = l1 * al1 + rs1;
#pragma unroll
            for (int d = 0; d < 4; d++) { acc[d][0] *= al0; acc[d][1] *= al0; acc[d][2] *= al1; acc[d][3] *= al1; }
#pragma unroll
            for (int n = 0; n < 2; n++) {
                *(uint32_t *)&Ps[w][gq][8 * n + 2 * tig] = pack_h2(sc[n][0], sc[n][1]);
                *(uint32_t *)&Ps[w][gq + 8][8 * n + 2 * tig] = pack_h2(sc[n][2], sc[n][3]);
            }
        }
        __syncthreads();
#pragma unroll 2
        for (int kt = 0; kt < 8; kt++) {
            if (s0 + 4 * kt >= nb) break;
            const uint32_t A[4] = {*(const uint32_t *)&Ps[kt][gq][2 * tig], *(const uint32_t *)&Ps[kt][gq + 8][2 * tig],
                                   *(const uint32_t *)&Ps[kt][gq][8 + 2 * tig], *(const uint32_t *)&Ps[kt][gq + 8][8 + 2 * tig]};
            const uint8_t *pq[2]; int sl[2]; half2 vsc[2];
#pragma unroll
            for (int hk = 0; hk < 2; hk++) {
                const int b2 = s0 + 4 * kt + 2 * hk + tig / 2, id2 = b2 < nb ? bid[b2] : -1;
                pq[hk] = block_base(b2 < nb ? b2 : 0);
                sl[hk] = (id2 >= 0 ? id2 * KPOOL % PAGE : 0) + (tig % 2) * 2;
                vsc[hk] = *(const half2 *)((const half *)(pq[hk] + VS_OFF) + w * PAGE + sl[hk]);
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int dim = 32 * w + 8 * nt + gq;
                uint32_t B[2];
#pragma unroll
                for (int hk = 0; hk < 2; hk++) {
                    const uint32_t lo = pq[hk][VLO_OFF + dim * (PAGE / 2) + sl[hk] / 2];
                    const uint32_t hb = pq[hk][VHI_OFF + dim * (PAGE / 8) + sl[hk] / 8] >> (sl[hk] % 8);
                    const float u0 = (float)((int)((lo & 15) | ((hb & 1) << 4)) - 16), u1 = (float)((int)((lo >> 4) | (((hb >> 1) & 1) << 4)) - 16);
                    const float2 sc2 = __half22float2(vsc[hk]);
                    B[hk] = pack_h2(u0 * sc2.x, u1 * sc2.y);
                }
                mma_f16(acc[nt], A, B);
            }
        }
    }
    if (tig == 0) { rmx[w][gq] = l0; rmx[w][gq + 8] = l1; }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int r = gq + 8 * h;
        if (r >= QPK) continue;
        const size_t row = (size_t)t * NH + g * QPK + r;
        float *po = a.part_o + ((size_t)split * a.T * NH + row) * HD + 32 * w;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) *(float2 *)&po[8 * nt + 2 * tig] = make_float2(acc[nt][2 * h], acc[nt][2 * h + 1]);
        if (w == 0 && tig == 0) {
            float L = 0;
#pragma unroll
            for (int ww = 0; ww < 8; ww++) L += rmx[ww][r];
            a.part_ml[((size_t)split * a.T * NH + row) * 2] = h ? m1 : m0;
            a.part_ml[((size_t)split * a.T * NH + row) * 2 + 1] = L;
        }
    }
}

void attention_sel(AttnArgs a, const int *sel, half *oh, QAct oq, cudaStream_t st) {
    const int stages = (SEL_BLOCKS + 1 + SST - 1) / SST;
    const int nsplit = std::max(1, std::min({stages, 192 / std::max(1, a.T * NKV), PART_ROWS / a.T}));
    const int sps = (stages + nsplit - 1) / nsplit;
    a.nsplit = (stages + sps - 1) / sps;
    if (a.nsplit * a.T > PART_ROWS) FAIL("attention_sel: %d tokens x %d splits exceed the partial buffers", a.T, a.nsplit);
    k_attn_sel<<<dim3(a.T, NKV, a.nsplit), 256, 0, st>>>(a, sel, sps);
    k_attn_combine<<<a.T * NH, 256, 0, st>>>(a.part_o, a.part_ml, a.nsplit, a.T * NH, a.qg, oh, oq);
}
