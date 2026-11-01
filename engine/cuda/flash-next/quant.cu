#include "kernels.h"
#include <algorithm>
#include <vector>

void qtensor_rows(QTensor &t, const QTensor &W, int r0) {
    const size_t n = (size_t)r0 * W.cols;
    switch (W.type) {
        case F32: t.f = (float *)W.f + n; break;
        case F16: t.f = (half *)W.f + n; break;
        case Q8_0: t.q = W.q + n; t.d = W.d + n / 32; break;
        case Q6_K: t.q = W.q + n / 2; t.qh = W.qh + n / 4; t.sc = W.sc + n / 16; t.d = W.d + n / 256; break;
        case IQ4_XS: t.q = W.q + n / 2; t.sc = W.sc + n / 64; t.sh = W.sh + n / 256; t.d = W.d + n / 256; break;
        case IQ4_NL: t.q = W.q + n / 2; t.d = W.d + n / 32; break;
        case Q3_K: t.q = W.q + n / 4; t.qh = W.qh + n / 8; t.sc = W.sc + n / 16; t.d = W.d + n / 256; break;
        case Q2_K: t.q = W.q + n / 4; t.sc = W.sc + n / 16; t.d = W.d + n / 256; t.dm = W.dm + n / 256; break;
        default: FAIL("qtensor_rows: type %d", W.type);
    }
}

QTensor make_qtensor(Arena &a, int type, int rows, int cols) {
    QTensor t; t.type = type; t.rows = rows; t.cols = cols;
    const size_t n = (size_t)rows * cols;
    switch (type) {
        case F32: t.f = a.get<float>(n); break;
        case F16: t.f = a.get<half>(n); break;
        case Q8_0: t.q = a.get(n); t.d = a.get<half>(n / 32); break;
        case Q6_K: t.q = a.get(n / 2); t.qh = a.get(n / 4); t.sc = a.get(n / 16); t.d = a.get<half>(n / 256); break;
        case IQ4_XS: t.q = a.get(n / 2); t.sc = a.get(n / 64); t.sh = a.get<uint16_t>(n / 256); t.d = a.get<half>(n / 256); break;
        case IQ4_NL: t.q = a.get(n / 2); t.d = a.get<half>(n / 32); break;
        case Q3_K: t.q = a.get(n / 4); t.qh = a.get(n / 8); t.sc = a.get(n / 16); t.d = a.get<half>(n / 256); break;
        case Q2_K: t.q = a.get(n / 4); t.sc = a.get(n / 16); t.d = a.get<half>(n / 256); t.dm = a.get<half>(n / 256); break;
        default: FAIL("unsupported tensor type %d", type);
    }
    return t;
}

__device__ __forceinline__ int2 iq4_lookup(uint32_t q) {
    constexpr uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;
    uint32_t r[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const uint32_t s = q >> (16 * h);
        const uint32_t lo = __byte_perm(T0, T1, s), hi = __byte_perm(T2, T3, s);
        r[h] = __byte_perm(lo, hi, 0x3210u | ((s & 0x8888u) >> 1));
    }
    return make_int2((int)__byte_perm(r[0], r[1], 0x6420u), (int)__byte_perm(r[0], r[1], 0x7531u));
}
__device__ __forceinline__ int iq4_scale(const QTensor &W, size_t sb, int ib) {
    return (((W.sc[sb * 4 + (ib >> 1)] >> 4 * (ib & 1)) & 0xf) | (((W.sh[sb] >> 2 * ib) & 3) << 4)) - 32;
}
__device__ __forceinline__ int q6_unpack(uint32_t lo4, uint32_t hi2) {
    const uint32_t x = (lo4 & 0x0F0F0F0Fu) | ((hi2 & 0x03030303u) << 4);
    return (int)((x & 0x1F1F1F1Fu) | ((~x & 0x20202020u) * 7));
}
__device__ __forceinline__ void iq4_block(int4 w, int4 &lo, int4 &hi) {
    const int2 p0 = iq4_lookup(w.x), p1 = iq4_lookup(w.y), p2 = iq4_lookup(w.z), p3 = iq4_lookup(w.w);
    lo = make_int4(p0.x, p1.x, p2.x, p3.x); hi = make_int4(p0.y, p1.y, p2.y, p3.y);
}
__device__ __forceinline__ int dot16(int4 w, int4 x, int acc) {
    return dot4(w.w, x.w, dot4(w.z, x.z, dot4(w.y, x.y, dot4(w.x, x.x, acc))));
}
__device__ __forceinline__ int sum16(int4 x) { return dot16(make_int4(0x01010101, 0x01010101, 0x01010101, 0x01010101), x, 0); }
__device__ __forceinline__ int q2_vals(uint32_t w, int j) { return (int)((w >> (2 * j)) & 0x03030303u); }
__device__ __forceinline__ int q3_vals(uint32_t w, uint32_t hm, int j, int bit) {
    const uint32_t u = ((w >> (2 * j)) & 0x03030303u) | (((hm >> bit) & 0x01010101u) << 2);
    return (int)(((u | 0x80808080u) - 0x04040404u) ^ 0x80808080u);
}


template <int QT, int T, int R>
__device__ __forceinline__ void rows_dot(const QTensor *const W[R], const int row[R], int lane, QAct x, float acc[R][T],
                                         int part = 0, int nparts = 1) {
    const int K = W[0]->cols;
    const int n32 = K / 32, b32 = part * n32 / nparts, e32 = (part + 1) * n32 / nparts;
    const int n64 = K / 64, b64 = part * n64 / nparts, e64 = (part + 1) * n64 / nparts;
    if constexpr (QT == Q8_0) {
        for (int ch = b32 + lane; ch < e32; ch += 32) {
            int4 w0[R], w1[R]; float dw[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const int8_t *wr = (const int8_t *)W[r]->q + (size_t)row[r] * K + ch * 32;
                w0[r] = *(const int4 *)wr; w1[r] = *(const int4 *)(wr + 16);
                dw[r] = __half2float(W[r]->d[(size_t)row[r] * (K / 32) + ch]);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int8_t *xp = x.q + (size_t)t * K + ch * 32;
                const int4 a = *(const int4 *)xp, b = *(const int4 *)(xp + 16);
                const float xd = x.d[t * (K / 32) + ch];
#pragma unroll
                for (int r = 0; r < R; r++) acc[r][t] = __fmaf_rn(dw[r] * xd, (float)dot16(w1[r], b, dot16(w0[r], a, 0)), acc[r][t]);
            }
        }
    } else if constexpr (QT == IQ4_XS) {
        for (int ch = b32 + lane; ch < e32; ch += 32) {
            const int sb = ch >> 3, ib = ch & 7;
            int4 lo[R], hi[R]; float dl[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const size_t gsb = (size_t)row[r] * (K / 256) + sb;
                const int4 w = *(const int4 *)(W[r]->q + (size_t)row[r] * (K / 2) + sb * 128 + ib * 16);
                const int2 p0 = iq4_lookup(w.x), p1 = iq4_lookup(w.y), p2 = iq4_lookup(w.z), p3 = iq4_lookup(w.w);
                lo[r] = make_int4(p0.x, p1.x, p2.x, p3.x); hi[r] = make_int4(p0.y, p1.y, p2.y, p3.y);
                dl[r] = __half2float(W[r]->d[gsb]) * iq4_scale(*W[r], gsb, ib);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int8_t *xp = x.q + (size_t)t * K + ch * 32;
                const int4 a = *(const int4 *)xp, b = *(const int4 *)(xp + 16);
                const float xd = x.d[t * (K / 32) + ch];
#pragma unroll
                for (int r = 0; r < R; r++) acc[r][t] = __fmaf_rn(dl[r] * xd, (float)dot16(hi[r], b, dot16(lo[r], a, 0)), acc[r][t]);
            }
        }
    } else if constexpr (QT == IQ4_NL) {
        for (int ch = b32 + lane; ch < e32; ch += 32) {
            int4 lo[R], hi[R]; float dw[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const int4 w = *(const int4 *)(W[r]->q + (size_t)row[r] * (K / 2) + ch * 16);
                iq4_block(w, lo[r], hi[r]);
                dw[r] = __half2float(W[r]->d[(size_t)row[r] * (K / 32) + ch]);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int8_t *xp = x.q + (size_t)t * K + ch * 32;
                const int4 a = *(const int4 *)xp, b = *(const int4 *)(xp + 16);
                const float xd = x.d[t * (K / 32) + ch];
#pragma unroll
                for (int r = 0; r < R; r++) acc[r][t] = __fmaf_rn(dw[r] * xd, (float)dot16(hi[r], b, dot16(lo[r], a, 0)), acc[r][t]);
            }
        }
    } else if constexpr (QT == Q3_K || QT == Q2_K) {
        for (int ch = b32 + lane; ch < e32; ch += 32) {
            const int sb = ch >> 3, n = (ch >> 2) & 1, j = ch & 3, g = 8 * n + 2 * j;
            int4 lo[R], hi[R]; float s0[R], s1[R], m0[R], m1[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const size_t gsb = (size_t)row[r] * (K / 256) + sb;
                const uint8_t *qp = W[r]->q + gsb * 64 + 32 * n;
                const int4 A = *(const int4 *)qp, B = *(const int4 *)(qp + 16);
                const float d = __half2float(W[r]->d[gsb]);
                if constexpr (QT == Q3_K) {
                    const int4 HA = *(const int4 *)(W[r]->qh + gsb * 32), HB = *(const int4 *)(W[r]->qh + gsb * 32 + 16);
                    const int bit = 4 * n + j;
                    lo[r] = make_int4(q3_vals(A.x, HA.x, j, bit), q3_vals(A.y, HA.y, j, bit), q3_vals(A.z, HA.z, j, bit), q3_vals(A.w, HA.w, j, bit));
                    hi[r] = make_int4(q3_vals(B.x, HB.x, j, bit), q3_vals(B.y, HB.y, j, bit), q3_vals(B.z, HB.z, j, bit), q3_vals(B.w, HB.w, j, bit));
                    const int8_t *sc = (const int8_t *)W[r]->sc + gsb * 16 + g;
                    s0[r] = d * sc[0]; s1[r] = d * sc[1]; m0[r] = m1[r] = 0;
                } else {
                    lo[r] = make_int4(q2_vals(A.x, j), q2_vals(A.y, j), q2_vals(A.z, j), q2_vals(A.w, j));
                    hi[r] = make_int4(q2_vals(B.x, j), q2_vals(B.y, j), q2_vals(B.z, j), q2_vals(B.w, j));
                    const uint8_t *sc = W[r]->sc + gsb * 16 + g;
                    const float dm = __half2float(W[r]->dm[gsb]);
                    s0[r] = d * (sc[0] & 15); s1[r] = d * (sc[1] & 15); m0[r] = dm * (sc[0] >> 4); m1[r] = dm * (sc[1] >> 4);
                }
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int8_t *xp = x.q + (size_t)t * K + ch * 32;
                const int4 a = *(const int4 *)xp, b = *(const int4 *)(xp + 16);
                const float xd = x.d[t * (K / 32) + ch];
                int sa = 0, sbb = 0;
                if constexpr (QT == Q2_K) { sa = sum16(a); sbb = sum16(b); }
#pragma unroll
                for (int r = 0; r < R; r++) {
                    float v = __fmaf_rn(s0[r], (float)dot16(lo[r], a, 0), s1[r] * (float)dot16(hi[r], b, 0));
                    if constexpr (QT == Q2_K) v = v - __fmaf_rn(m0[r], (float)sa, m1[r] * (float)sbb);
                    acc[r][t] = __fmaf_rn(xd, v, acc[r][t]);
                }
            }
        }
    } else if constexpr (QT == Q6_K) {
        for (int ch = b64 + lane; ch < e64; ch += 32) {
            const int sb = ch >> 2, h = (ch >> 1) & 1, c = ch & 1;
            int4 q[R][4]; float sc[R][4];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const size_t gsb = (size_t)row[r] * (K / 256) + sb;
                const uint8_t *ql = W[r]->q + (size_t)row[r] * (K / 2) + sb * 128 + h * 64 + c * 16;
                const int4 A = *(const int4 *)ql, B = *(const int4 *)(ql + 32);
                const int4 H = *(const int4 *)(W[r]->qh + (size_t)row[r] * (K / 4) + sb * 64 + h * 32 + c * 16);
                const int8_t *s8 = (const int8_t *)W[r]->sc + gsb * 16 + h * 8 + c;
                const float dd = __half2float(W[r]->d[gsb]);
                q[r][0] = make_int4(q6_unpack(A.x, H.x), q6_unpack(A.y, H.y), q6_unpack(A.z, H.z), q6_unpack(A.w, H.w));
                q[r][1] = make_int4(q6_unpack(B.x, H.x >> 2), q6_unpack(B.y, H.y >> 2), q6_unpack(B.z, H.z >> 2), q6_unpack(B.w, H.w >> 2));
                q[r][2] = make_int4(q6_unpack(A.x >> 4, H.x >> 4), q6_unpack(A.y >> 4, H.y >> 4), q6_unpack(A.z >> 4, H.z >> 4), q6_unpack(A.w >> 4, H.w >> 4));
                q[r][3] = make_int4(q6_unpack(B.x >> 4, H.x >> 6), q6_unpack(B.y >> 4, H.y >> 6), q6_unpack(B.z >> 4, H.z >> 6), q6_unpack(B.w >> 4, H.w >> 6));
#pragma unroll
                for (int k = 0; k < 4; k++) sc[r][k] = dd * s8[2 * k];
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int blk = sb * 8 + h * 4 + k;
                    const int4 a = *(const int4 *)(x.q + (size_t)t * K + blk * 32 + c * 16);
                    const float xd = x.d[t * (K / 32) + blk];
#pragma unroll
                    for (int r = 0; r < R; r++) acc[r][t] = __fmaf_rn(sc[r][k] * xd, (float)dot16(q[r][k], a, 0), acc[r][t]);
                }
            }
        }
    }
}

template <int T, int R, int QT, int... More>
__device__ __forceinline__ void rows_dot_any(const QTensor *const W[R], const int row[R], int lane, QAct x, float acc[R][T]) {
    if constexpr (sizeof...(More) == 0) rows_dot<QT, T, R>(W, row, lane, x, acc);
    else if (W[0]->type == QT) rows_dot<QT, T, R>(W, row, lane, x, acc);
    else rows_dot_any<T, R, More...>(W, row, lane, x, acc);
}

template <int T, int R>
__global__ __launch_bounds__(256) void k_gemv(GemvArgs a) {
    int row = (blockIdx.x * 8 + threadIdx.x / 32) * R, s = 0;
    const int lane = threadIdx.x % 32;
    while (s < a.nseg - 1 && row >= a.seg[s].W.rows) row -= a.seg[s++].W.rows;
    if (row >= a.seg[s].W.rows) return;
    const QTensor *W[R]; int rows[R];
    float acc[R][T] = {};
#pragma unroll
    for (int r = 0; r < R; r++) { W[r] = &a.seg[s].W; rows[r] = row + r; }
    rows_dot_any<T, R, Q6_K, Q8_0>(W, rows, lane, a.x, acc);
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) {
                if (a.mode == GEMV_HC) {
#pragma unroll
                    for (int h = 0; h < HC; h++)
                        a.res[((size_t)t * HC + h) * D + row + r] += 2.0f * sigmoid(a.inj[t * HC + h] * (1.0f / HC)) * v;
                } else {
                    float *o = a.seg[s].y + (size_t)t * a.seg[s].ldy + row + r;
                    *o = a.mode == GEMV_ADD ? *o + v : v;
                }
            }
        }
}

constexpr int GEMV_SK = 4;
template <int QT, int T>
__global__ __launch_bounds__(256) void k_gemv_sk(GemvArgs a) {
    __shared__ float part[8][T];
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    int row = blockIdx.x * (8 / GEMV_SK) + wave / GEMV_SK, s = 0;
    while (s < a.nseg - 1 && row >= a.seg[s].W.rows) row -= a.seg[s++].W.rows;
    const bool valid = row < a.seg[s].W.rows;
    float acc[1][T] = {};
    if (valid) {
        const QTensor *W[1] = {&a.seg[s].W};
        const int rows[1] = {row};
        rows_dot<QT, T, 1>(W, rows, lane, a.x, acc, wave % GEMV_SK, GEMV_SK);
    }
#pragma unroll
    for (int t = 0; t < T; t++) { const float v = warp_sum(acc[0][t]); if (lane == 0) part[wave][t] = v; }
    __syncthreads();
    if (!valid || wave % GEMV_SK || lane >= T) return;
    const int t = lane;
    float v = 0;
#pragma unroll
    for (int k = 0; k < GEMV_SK; k++) v += part[wave + k][t];
    if (a.mode == GEMV_HC) {
#pragma unroll
        for (int h = 0; h < HC; h++) a.res[((size_t)t * HC + h) * D + row] += 2.0f * sigmoid(a.inj[t * HC + h] * (1.0f / HC)) * v;
    } else {
        float *o = a.seg[s].y + (size_t)t * a.seg[s].ldy + row;
        *o = a.mode == GEMV_ADD ? *o + v : v;
    }
}

template <int T>
static void gemv_launch(const GemvArgs &a, int rows, cudaStream_t st) {
    if (a.seg[0].W.type == Q8_0 && a.seg[0].W.cols >= 4096 && rows >= 1024 && rows <= 8192 && a.nseg == 1)
        k_gemv_sk<Q8_0, T><<<(rows + 1) / 2, 256, 0, st>>>(a);
    else if (T >= 5) k_gemv<T, 2><<<(rows + 15) / 16, 256, 0, st>>>(a);
    else k_gemv<T, 1><<<(rows + 7) / 8, 256, 0, st>>>(a);
}

void gemv(const GemvArgs &a, cudaStream_t st) {
    int rows = 0;
    for (int i = 0; i < a.nseg; i++) rows += a.seg[i].W.rows;
    switch (a.T) {
        case 1: gemv_launch<1>(a, rows, st); break;
        case 2: gemv_launch<2>(a, rows, st); break;
        case 3: gemv_launch<3>(a, rows, st); break;
        case 4: gemv_launch<4>(a, rows, st); break;
        case 5: gemv_launch<5>(a, rows, st); break;
        case 6: gemv_launch<6>(a, rows, st); break;
        case 7: gemv_launch<7>(a, rows, st); break;
        case 8: gemv_launch<8>(a, rows, st); break;
        default: FAIL("gemv: T=%d unsupported", a.T);
    }
}

void gemv(const QTensor &W, QAct x, int T, float *y, int ldy, int mode, cudaStream_t st) {
    GemvArgs a;
    a.seg[0] = {W, y, ldy}; a.x = x; a.T = T; a.mode = mode;
    gemv(a, st);
}
void gemv_hc(const QTensor &W, QAct x, int T, float *res, const float *inj, cudaStream_t st) {
    GemvArgs a;
    a.seg[0] = {W, nullptr, 0}; a.x = x; a.T = T; a.mode = GEMV_HC; a.inj = inj; a.res = res;
    gemv(a, st);
}


__device__ __forceinline__ uint4 to_h8(int2 q, float d) {
    const half2 dd = __float2half2_rn(d);
    const int8_t *b = (const int8_t *)&q;
    uint4 r;
    uint32_t *o = (uint32_t *)&r;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const half2 v = __hmul2(__floats2half2_rn((float)b[2 * i], (float)b[2 * i + 1]), dd);
        o[i] = *(const uint32_t *)&v;
    }
    return r;
}

__device__ __forceinline__ uint4 to_h8m(int2 q, float d, float m) {
    const half2 dd = __float2half2_rn(d), mm = __float2half2_rn(-m);
    const int8_t *b = (const int8_t *)&q;
    uint4 r;
    uint32_t *o = (uint32_t *)&r;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const half2 v = __hfma2(__floats2half2_rn((float)b[2 * i], (float)b[2 * i + 1]), dd, mm);
        o[i] = *(const uint32_t *)&v;
    }
    return r;
}

struct SbScales { float d, dm; int4 raw; };
template <int QT>
__device__ __forceinline__ SbScales sb_scales(const QTensor &W, int row, int sb) {
    const size_t gsb = (size_t)row * (W.cols / 256) + sb;
    if constexpr (QT == IQ4_XS) return {__half2float(W.d[gsb]), 0.0f, make_int4(((const int *)W.sc)[gsb], W.sh[gsb], 0, 0)};
    else if constexpr (QT == Q6_K || QT == Q3_K) return {__half2float(W.d[gsb]), 0.0f, ((const int4 *)W.sc)[gsb]};
    else if constexpr (QT == Q2_K) return {__half2float(W.d[gsb]), __half2float(W.dm[gsb]), ((const int4 *)W.sc)[gsb]};
    else return {1.0f, 0.0f, ((const int4 *)W.d)[gsb]};
}
template <int QT>
__device__ __forceinline__ float frag_scale(const SbScales &s, int f) {
    const uint32_t w[4] = {(uint32_t)s.raw.x, (uint32_t)s.raw.y, (uint32_t)s.raw.z, (uint32_t)s.raw.w};
    if constexpr (QT == IQ4_XS) return s.d * ((int)(((w[0] >> 4 * (f >> 1)) & 0xf) | (((w[1] >> 2 * (f >> 1)) & 3) << 4)) - 32);
    else if constexpr (QT == Q6_K || QT == Q3_K) return s.d * (int8_t)(w[f >> 2] >> 8 * (f & 3));
    else if constexpr (QT == Q2_K) return s.d * ((w[f >> 2] >> 8 * (f & 3)) & 0xf);
    else return __half2float(__ushort_as_half((uint16_t)(w[f >> 2] >> 16 * ((f >> 1) & 1))));
}
__device__ __forceinline__ float frag_min_q2k(const SbScales &s, int f) {
    const uint32_t w[4] = {(uint32_t)s.raw.x, (uint32_t)s.raw.y, (uint32_t)s.raw.z, (uint32_t)s.raw.w};
    return s.dm * ((w[f >> 2] >> (8 * (f & 3) + 4)) & 0xf);
}
template <int QT>
__device__ __forceinline__ uint4 frag_h8(int2 q, const SbScales &s, int f) {
    if constexpr (QT == Q2_K) return to_h8m(q, frag_scale<QT>(s, f), frag_min_q2k(s, f));
    else return to_h8(q, frag_scale<QT>(s, f));
}

template <int QT>
__device__ __forceinline__ void frags(const QTensor &W, int row, int sb, int c, int hf, int2 q[4], int f[4]) {
    const int K = W.cols;
    if constexpr (QT == IQ4_XS) {
        const uint8_t *p = W.q + (size_t)row * (K / 2) + sb * 128 + c * 32 + 8 * hf;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int2 w = *(const int2 *)(p + 16 * h);
            const int2 a = iq4_lookup(w.x), b = iq4_lookup(w.y);
            q[2 * h] = make_int2(a.x, b.x); q[2 * h + 1] = make_int2(a.y, b.y);
            f[2 * h] = 4 * c + 2 * h; f[2 * h + 1] = 4 * c + 2 * h + 1;
        }
    } else if constexpr (QT == Q6_K) {
        const int h = c >> 1, cc = c & 1;
        const uint8_t *ql = W.q + (size_t)row * (K / 2) + sb * 128 + h * 64 + cc * 16 + 8 * hf;
        const int2 A = *(const int2 *)ql, B = *(const int2 *)(ql + 32);
        const int2 H = *(const int2 *)(W.qh + (size_t)row * (K / 4) + sb * 64 + h * 32 + cc * 16 + 8 * hf);
        q[0] = make_int2(q6_unpack(A.x, H.x), q6_unpack(A.y, H.y));
        q[1] = make_int2(q6_unpack(B.x, H.x >> 2), q6_unpack(B.y, H.y >> 2));
        q[2] = make_int2(q6_unpack(A.x >> 4, H.x >> 4), q6_unpack(A.y >> 4, H.y >> 4));
        q[3] = make_int2(q6_unpack(B.x >> 4, H.x >> 6), q6_unpack(B.y >> 4, H.y >> 6));
#pragma unroll
        for (int g = 0; g < 4; g++) f[g] = 8 * h + cc + 2 * g;
    } else if constexpr (QT == Q3_K || QT == Q2_K) {
        const int n = c >> 1, k = c & 1;
        const size_t gsb = (size_t)row * (K / 256) + sb;
        const int2 w = *(const int2 *)(W.q + gsb * 64 + 32 * n + 16 * k + 8 * hf);
        if constexpr (QT == Q3_K) {
            const int2 hm = *(const int2 *)(W.qh + gsb * 32 + 16 * k + 8 * hf);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                q[j] = make_int2(q3_vals(w.x, hm.x, j, 4 * n + j), q3_vals(w.y, hm.y, j, 4 * n + j));
                f[j] = 8 * n + 2 * j + k;
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) { q[j] = make_int2(q2_vals(w.x, j), q2_vals(w.y, j)); f[j] = 8 * n + 2 * j + k; }
        }
    } else if constexpr (QT == IQ4_NL) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int blk = sb * 8 + 2 * c + h;
            const int2 w = *(const int2 *)(W.q + (size_t)row * (K / 2) + blk * 16 + 8 * hf);
            const int2 a = iq4_lookup(w.x), b = iq4_lookup(w.y);
            q[2 * h] = make_int2(a.x, b.x); q[2 * h + 1] = make_int2(a.y, b.y);
            f[2 * h] = 4 * c + 2 * h; f[2 * h + 1] = 4 * c + 2 * h + 1;
        }
    } else {
        const int8_t *p = (const int8_t *)W.q + (size_t)row * K + sb * 256 + c * 64 + 8 * hf;
#pragma unroll
        for (int g = 0; g < 4; g++) { q[g] = *(const int2 *)(p + 16 * g); f[g] = 4 * c + g; }
    }
}


__device__ __forceinline__ void store16(half *o, int4 q, float d) {
    const int8_t *b = (const int8_t *)&q;
    half2 h[8];
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = __floats2half2_rn(d * b[2 * i], d * b[2 * i + 1]);
    ((int4 *)o)[0] = *(const int4 *)&h[0];
    ((int4 *)o)[1] = *(const int4 *)&h[4];
}

__global__ void k_dequant(QTensor W, half *out, int64_t nchunks) {
    const int64_t ch = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (ch >= nchunks) return;
    const int K = W.cols;
    if (W.type == Q8_0) {
        const int row = ch / (K / 32), c = ch % (K / 32);
        const int8_t *q = (const int8_t *)W.q + (size_t)row * K + c * 32;
        const float dw = __half2float(W.d[ch]);
        half *o = out + (size_t)row * K + c * 32;
        store16(o, *(const int4 *)q, dw);
        store16(o + 16, *(const int4 *)(q + 16), dw);
    } else {
        const int row = ch / (K / 64), c64 = ch % (K / 64), sb = c64 >> 2, h = (c64 >> 1) & 1, c = c64 & 1;
        const size_t gsb = (size_t)row * (K / 256) + sb;
        const uint8_t *ql = W.q + (size_t)row * (K / 2) + sb * 128 + h * 64 + c * 16;
        const int4 A = *(const int4 *)ql, B = *(const int4 *)(ql + 32);
        const int4 H = *(const int4 *)(W.qh + (size_t)row * (K / 4) + sb * 64 + h * 32 + c * 16);
        const int8_t *sc = (const int8_t *)W.sc + gsb * 16 + h * 8 + c;
        const float dd = __half2float(W.d[gsb]);
        half *o = out + (size_t)row * K + sb * 256 + h * 128 + c * 16;
        store16(o, make_int4(q6_unpack(A.x, H.x), q6_unpack(A.y, H.y), q6_unpack(A.z, H.z), q6_unpack(A.w, H.w)), dd * sc[0]);
        store16(o + 32, make_int4(q6_unpack(B.x, H.x >> 2), q6_unpack(B.y, H.y >> 2), q6_unpack(B.z, H.z >> 2), q6_unpack(B.w, H.w >> 2)), dd * sc[2]);
        store16(o + 64, make_int4(q6_unpack(A.x >> 4, H.x >> 4), q6_unpack(A.y >> 4, H.y >> 4), q6_unpack(A.z >> 4, H.z >> 4), q6_unpack(A.w >> 4, H.w >> 4)), dd * sc[4]);
        store16(o + 96, make_int4(q6_unpack(B.x >> 4, H.x >> 6), q6_unpack(B.y >> 4, H.y >> 6), q6_unpack(B.z >> 4, H.z >> 6), q6_unpack(B.w >> 4, H.w >> 6)), dd * sc[6]);
    }
}

void dequant(const QTensor &W, half *out, cudaStream_t st) {
    if (W.type != Q8_0 && W.type != Q6_K) FAIL("dequant: type %d", W.type);
    const int64_t n = (int64_t)W.rows * W.cols;
    const int64_t nchunks = W.type == Q6_K ? n / 64 : n / 32;
    k_dequant<<<(nchunks + 255) / 256, 256, 0, st>>>(W, out, nchunks);
}

__global__ void k_embed(QTensor W, const int *tokens, const half *const *rows, float *out) {
    const int t = blockIdx.y, K = W.cols;
    if (rows && rows[t]) {
        for (int c = threadIdx.x; c < K; c += blockDim.x) out[(size_t)t * K + c] = __half2float(rows[t][c]);
        return;
    }
    const int row = tokens[t];
    for (int c = threadIdx.x; c < K / 32; c += blockDim.x) {
        float *o = out + (size_t)t * K + c * 32;
        if (W.type == Q8_0) {
            const uint8_t *blk = W.q + ((size_t)row * (K / 32) + c) * 34;
            const float d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
            for (int j = 0; j < 32; j++) o[j] = d * (int8_t)blk[2 + j];
            continue;
        }
        const int sb = c >> 3, ib = c & 7;
        const size_t gsb = (size_t)row * (K / 256) + sb;
        const float dl = __half2float(W.d[gsb]) * iq4_scale(W, gsb, ib);
        const uint32_t *w = (const uint32_t *)(W.q + (size_t)row * (K / 2) + sb * 128 + ib * 16);
        for (int k = 0; k < 4; k++) {
            const int2 p = iq4_lookup(w[k]);
            for (int j = 0; j < 4; j++) {
                o[4 * k + j] = dl * (int8_t)(p.x >> 8 * j);
                o[16 + 4 * k + j] = dl * (int8_t)(p.y >> 8 * j);
            }
        }
    }
}

void embed(const QTensor &W, const int *tokens, const half *const *rows, int T, float *out, cudaStream_t st) {
    if (W.type != IQ4_XS && W.type != Q8_0) FAIL("embed: token embeddings must be IQ4_XS or Q8_0 (GGUF blocks)");
    k_embed<<<dim3(1, T), 256, 0, st>>>(W, tokens, rows, out);
}


__global__ void k_quant_act(const float *x, int64_t n, QAct out) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.0f;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    if (i < n) {
        out.q[i] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
        if ((threadIdx.x & 31) == 0) out.d[i / 32] = d;
    }
}

void quant_act(const float *x, int64_t n, QAct out, cudaStream_t st) {
    k_quant_act<<<(n + 255) / 256, 256, 0, st>>>(x, n, out);
}



constexpr int CG_ROWS = 8;
constexpr int CD_ROWS = 32;

template <int QT>
__device__ __forceinline__ float aos_dot(const uint8_t *row, int lane, const int8_t *xq, const float *xd, const int (*xs16)[2]) {
    float acc = 0;
    for (int i = 0, ch = lane; ch < D / 32; ch += 32, i++) {
        const int4 a = *(const int4 *)(xq + ch * 32), b = *(const int4 *)(xq + ch * 32 + 16);
        if constexpr (QT == IQ4_XS) {
            const int sb = ch >> 3, ib = ch & 7;
            const uint8_t *blk = row + sb * 136;
            const float dl = __half2float(*(const __half *)blk) *
                             ((((blk[4 + (ib >> 1)] >> 4 * (ib & 1)) & 0xf) | (((*(const uint16_t *)(blk + 2) >> 2 * ib) & 3) << 4)) - 32);
            const uint2 q0 = *(const uint2 *)(blk + 8 + 16 * ib), q1 = *(const uint2 *)(blk + 16 + 16 * ib);
            int4 lo, hi;
            iq4_block(make_int4(q0.x, q0.y, q1.x, q1.y), lo, hi);
            acc = __fmaf_rn(dl * xd[ch], (float)dot16(hi, b, dot16(lo, a, 0)), acc);
        } else if constexpr (QT == Q3_K) {
            const int sb = ch >> 3, n = (ch >> 2) & 1, j = ch & 3, g = 8 * n + 2 * j, bit = 4 * n + j;
            const uint8_t *blk = row + sb * 110;
            const uint16_t *h16 = (const uint16_t *)blk, *q16 = (const uint16_t *)(blk + 32 + 32 * n), *s16 = (const uint16_t *)(blk + 96);
            uint32_t qw[8], hw[8];
#pragma unroll
            for (int k = 0; k < 8; k++) { qw[k] = q16[2 * k] | (uint32_t)q16[2 * k + 1] << 16; hw[k] = h16[2 * k] | (uint32_t)h16[2 * k + 1] << 16; }
            const int4 lo = make_int4(q3_vals(qw[0], hw[0], j, bit), q3_vals(qw[1], hw[1], j, bit), q3_vals(qw[2], hw[2], j, bit), q3_vals(qw[3], hw[3], j, bit));
            const int4 hi = make_int4(q3_vals(qw[4], hw[4], j, bit), q3_vals(qw[5], hw[5], j, bit), q3_vals(qw[6], hw[6], j, bit), q3_vals(qw[7], hw[7], j, bit));
            const uint32_t l2 = s16[(g & 7) >> 1], h2 = s16[4 + ((g & 3) >> 1)];
            const int sl = 4 * (g >> 3), sh = 2 * (g >> 2);
            const int sc0 = (int)(((l2 >> sl) & 0xf) | (((h2 >> sh) & 3) << 4)) - 32;
            const int sc1 = (int)(((l2 >> (8 + sl)) & 0xf) | (((h2 >> (8 + sh)) & 3) << 4)) - 32;
            const float d = __half2float(__ushort_as_half(*(const uint16_t *)(blk + 108)));
            acc = __fmaf_rn(d * xd[ch], (float)(sc0 * dot16(lo, a, 0) + sc1 * dot16(hi, b, 0)), acc);
        } else {
            const int sb = ch >> 3, n = (ch >> 2) & 1, j = ch & 3, g = 8 * n + 2 * j;
            const uint8_t *blk = row + sb * 84;
            const uint32_t *q32 = (const uint32_t *)(blk + 16 + 32 * n);
            const int4 lo = make_int4(q2_vals(q32[0], j), q2_vals(q32[1], j), q2_vals(q32[2], j), q2_vals(q32[3], j));
            const int4 hi = make_int4(q2_vals(q32[4], j), q2_vals(q32[5], j), q2_vals(q32[6], j), q2_vals(q32[7], j));
            const uint32_t sc = *(const uint16_t *)(blk + g), dd = *(const uint32_t *)(blk + 80);
            const float d = __half2float(__ushort_as_half((uint16_t)dd)), dm = __half2float(__ushort_as_half((uint16_t)(dd >> 16)));
            acc = __fmaf_rn(xd[ch], __fmaf_rn(d, (float)((int)(sc & 0xf) * dot16(lo, a, 0) + (int)((sc >> 8) & 0xf) * dot16(hi, b, 0)),
                                              -(dm * (float)((int)((sc >> 4) & 0xf) * xs16[i][0] + (int)(sc >> 12) * xs16[i][1]))), acc);
        }
    }
    return acc;
}

template <int QT>
__device__ __forceinline__ void cold_glu(const uint8_t *gate, const uint8_t *up, int slot, int j, QAct x, QAct hq, int4 *tile0,
                                         int4 *tile1, float *act) {
    constexpr int RB = D / 256 * qbytes_of(QT);
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, t = j / TOPK_E;
    const int8_t *xq = x.q + (size_t)t * D;
    const float *xd = x.d + (size_t)t * (D / 32);
    int xs16[3][2] = {};
    if constexpr (QT == Q2_K)
        for (int i = 0, ch = lane; ch < D / 32; ch += 32, i++) {
            xs16[i][0] = sum16(*(const int4 *)(xq + ch * 32)); xs16[i][1] = sum16(*(const int4 *)(xq + ch * 32 + 16));
        }
    for (int k = 0; k < 32 / CG_ROWS; k++) {
        const size_t r0 = (size_t)slot * FE + blockIdx.x * 32 + k * CG_ROWS;
        const int4 *g = (const int4 *)(gate + r0 * RB), *u = (const int4 *)(up + r0 * RB);
        for (int i = threadIdx.x; i < CG_ROWS * RB / 16; i += 256) { tile0[i] = g[i]; tile1[i] = u[i]; }
        __syncthreads();
        const float ag = aos_dot<QT>((const uint8_t *)tile0 + wave * RB, lane, xq, xd, xs16);
        const float au = aos_dot<QT>((const uint8_t *)tile1 + wave * RB, lane, xq, xd, xs16);
        const float v = silu(warp_sum(ag)) * warp_sum(au);
        if (lane == 0) act[k * CG_ROWS + wave] = v;
        __syncthreads();
    }
    if (wave == 0) {
        const float a = act[lane], d = warp_max(fabsf(a)) / 127.0f;
        hq.q[(size_t)j * FE + blockIdx.x * 32 + lane] = (int8_t)__float2int_rn(d > 0 ? a / d : 0.0f);
        if (lane == 0) hq.d[((size_t)j * FE + blockIdx.x * 32) / 32] = d;
    }
}

__device__ __forceinline__ void cold_down(const uint8_t *down, int slot, int j, int r0, QAct hq, const float *wts, float *y, int4 *tile) {
    constexpr int RB = FE / 32 * 18;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int4 *src = (const int4 *)(down + ((size_t)slot * D + r0) * RB);
    for (int i = threadIdx.x; i < CD_ROWS * RB / 16; i += 256) tile[i] = src[i];
    __syncthreads();
    const int8_t *h = hq.q + (size_t)j * FE;
    const float *hd = hq.d + (size_t)j * (FE / 32);
    for (int rr = 0; rr < CD_ROWS / 8; rr++) {
        const int r = wave * (CD_ROWS / 8) + rr;
        float acc = 0;
        if (lane < FE / 32) {
            const uint8_t *blk = (const uint8_t *)tile + r * RB + lane * 18;
            const uint16_t *q16 = (const uint16_t *)(blk + 2);
            int4 lo, hi;
            iq4_block(make_int4(q16[0] | (uint32_t)q16[1] << 16, q16[2] | (uint32_t)q16[3] << 16,
                                q16[4] | (uint32_t)q16[5] << 16, q16[6] | (uint32_t)q16[7] << 16), lo, hi);
            const int4 a = *(const int4 *)(h + lane * 32), b = *(const int4 *)(h + lane * 32 + 16);
            acc = __half2float(*(const __half *)blk) * hd[lane] * dot16(hi, b, dot16(lo, a, 0));
        }
        const float v = warp_sum(acc);
        if (lane == 0) y[(size_t)j * D + r0 + r] = wts[j] * v;
    }
}

__global__ __launch_bounds__(256) void k_glu_dec(QTensor gate, QTensor up, const uint8_t *fmt, QTensor sgate, QTensor sup, ColdDec c,
                                                 const int *hot_slot, int T, QAct x, QAct hq) {
    __shared__ int4 tile[2][CG_ROWS * (D / 256 * 136) / 16];
    __shared__ float act[32];
    const int j = blockIdx.y;
    const bool shared = j >= T * TOPK_E;
    const int hs = shared ? 0 : hot_slot[j];
    if (hs == -1 || hs == EMU_PICK || (hs >= 0 && !shared && gate.type < 0) || (hs <= -2 && !c.down)) return;
    if (hs <= -2) {
        const int slot = -2 - hs, grp = slot >= c.n0, s = grp ? slot - c.n0 : slot;
        switch (c.type[grp]) {
            case Q3_K: cold_glu<Q3_K>(c.gate[grp], c.up[grp], s, j, x, hq, tile[0], tile[1], act); break;
            case Q2_K: cold_glu<Q2_K>(c.gate[grp], c.up[grp], s, j, x, hq, tile[0], tile[1], act); break;
            default: cold_glu<IQ4_XS>(c.gate[grp], c.up[grp], s, j, x, hq, tile[0], tile[1], act); break;
        }
        return;
    }
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, t = shared ? j - T * TOPK_E : j / TOPK_E;
    const int f = shared || !fmt ? -1 : fmt[hs];
    const QTensor G = shared ? sgate : f >= 0 ? slot_view(gate, hs, f) : gate, U = shared ? sup : f >= 0 ? slot_view(up, hs, f) : up;
    const int r0 = f >= 0 ? 0 : hs * FE;
    const QTensor *W[8];
    int rows[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        W[i] = &G; W[4 + i] = &U;
        rows[i] = rows[4 + i] = r0 + blockIdx.x * 32 + wave * 4 + i;
    }
    const QAct xt{x.q + (size_t)t * D, x.d + (size_t)t * (D / 32)};
    float acc[8][1] = {};
    rows_dot_any<1, 8, IQ4_XS, Q6_K, Q3_K, Q2_K>(W, rows, lane, xt, acc);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float v = silu(warp_sum(acc[i][0])) * warp_sum(acc[4 + i][0]);
        if (lane == 0) act[wave * 4 + i] = v;
    }
    __syncthreads();
    if (wave == 0) {
        const float a = act[lane], d = warp_max(fabsf(a)) / 127.0f;
        hq.q[(size_t)j * FE + blockIdx.x * 32 + lane] = (int8_t)__float2int_rn(d > 0 ? a / d : 0.0f);
        if (lane == 0) hq.d[((size_t)j * FE + blockIdx.x * 32) / 32] = d;
    }
}

__global__ __launch_bounds__(256) void k_down_dec(QTensor down, QTensor sdown, const uint8_t *cdown, const int *hot_slot, int T,
                                                  QAct hq, const float *wts, const float *sg, float *y) {
    __shared__ int4 tile[CD_ROWS * (FE / 32 * 18) / 16];
    const int j = blockIdx.y, r0 = blockIdx.x * CD_ROWS;
    const bool shared = j >= T * TOPK_E;
    const int hs = shared ? 0 : hot_slot[j];
    if (hs == -1 || hs == EMU_PICK || (hs >= 0 && !shared && down.type < 0) || (hs <= -2 && !cdown)) return;
    if (hs <= -2) { cold_down(cdown, -2 - hs, j, r0, hq, wts, y, tile); return; }
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    int d0 = 0;
    const QTensor dv = shared ? sdown : down_view(down, hs, d0);
    const QTensor *W[4];
    int rows[4];
#pragma unroll
    for (int i = 0; i < 4; i++) { W[i] = &dv; rows[i] = d0 + r0 + wave * 4 + i; }
    const QAct h{hq.q + (size_t)j * FE, hq.d + (size_t)j * (FE / 32)};
    float acc[4][1] = {};
    rows_dot_any<1, 4, IQ4_NL, Q8_0>(W, rows, lane, h, acc);
    const float wj = shared ? sg[j - T * TOPK_E] : wts[j];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float v = warp_sum(acc[i][0]);
        if (lane == 0) y[(size_t)j * D + r0 + wave * 4 + i] = wj * v;
    }
}

void moe_dec(const QTensor &gate, const QTensor &up, const QTensor &down, const uint8_t *fmt, const QTensor &sgate, const QTensor &sup,
             const QTensor &sdown, const ColdDec *c, const int *hot_slot, int T, QAct x, QAct hq, const float *wts, const float *sg,
             float *y, cudaStream_t st) {
    if (c)
        for (int g = 0; g < 2; g++)
            if (c->gate[g] && c->type[g] != IQ4_XS && c->type[g] != Q3_K && c->type[g] != Q2_K) FAIL("moe_dec: cold gate/up type %d", c->type[g]);
    if (fmt && gate.type != IQ4_XS) FAIL("moe_dec: an expert pool of type %d", gate.type);
    const ColdDec none{};
    const int picks = T * TOPK_E + (sgate.type >= 0 ? T : 0);
    k_glu_dec<<<dim3(FE / 32, picks), 256, 0, st>>>(gate, up, fmt, sgate, sup, c ? *c : none, hot_slot, T, x, hq);
    k_down_dec<<<dim3(D / CD_ROWS, picks), 256, 0, st>>>(down, sdown, c ? c->down : nullptr, hot_slot, T, hq, wts, sg, y);
}
void moe_experts_dec(const QTensor &gate, const QTensor &up, const QTensor &down, const uint8_t *fmt, const QTensor &sgate,
                     const QTensor &sup, const QTensor &sdown, const int *hot_slot, int T, QAct x, QAct hq, const float *wts,
                     const float *sg, float *y, cudaStream_t st) {
    moe_dec(gate, up, down, fmt, sgate, sup, sdown, nullptr, hot_slot, T, x, hq, wts, sg, y, st);
}
void moe_cold_dec(const ColdDec &c, const int *hot_slot, int T, QAct x, QAct hq, const float *wts, float *y, cudaStream_t st) {
    const QTensor none{};
    moe_dec(none, none, none, nullptr, none, none, none, &c, hot_slot, T, x, hq, wts, nullptr, y, st);
}


template <int QT>
__device__ __forceinline__ void frags64(const QTensor &W, int row, int ch, int hf, int2 q[4], float s[4]) {
    const int K = W.cols;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int blk = 2 * ch + h;
        const int2 w = *(const int2 *)(W.q + (size_t)row * (K / 2) + blk * 16 + 8 * hf);
        const int2 a = iq4_lookup(w.x), b = iq4_lookup(w.y);
        const int2 lo = make_int2(a.x, b.x), hi = make_int2(a.y, b.y);
        const float d = __half2float(W.d[(size_t)row * (K / 32) + blk]);
        q[2 * h] = lo; q[2 * h + 1] = hi; s[2 * h] = d; s[2 * h + 1] = d;
    }
}

constexpr int MOE_NW_GU = 8, MOE_NW_DN = 16;
template <int QT, bool SLOT = false>
__global__ __launch_bounds__(32 * MOE_NW_GU) void k_moe_gateup(QTensor g0, QTensor u0, QTensor g1, QTensor u1, const MoeWork *work,
                                                              const half *X, half *H) {
    __shared__ __align__(16) half Ws[2][32][128 + 8];
    const MoeWork wk = work[blockIdx.y];
    QTensor gs, us;
    if constexpr (SLOT) { gs = slot_view(g0, wk.slot, QT); us = slot_view(u0, wk.slot, QT); }
    const QTensor &G = SLOT ? gs : wk.tier ? g1 : g0, &U = SLOT ? us : wk.tier ? u1 : u0;
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, gq = lane / 4, tig = lane % 4;
    const int row0 = (SLOT ? 0 : wk.slot * FE) + blockIdx.x * 32;
    for (int tt = 0; tt < wk.n; tt += MOE_NW_GU * 16) {
        const int nt = min(MOE_NW_GU * 16, wk.n - tt), m0 = wk.m0 + tt;
        const bool live = 16 * wave < nt;
        const half *xr[2] = {X + (size_t)(m0 + min(16 * wave + gq, nt - 1)) * D, X + (size_t)(m0 + min(16 * wave + 8 + gq, nt - 1)) * D};
        float ag[2][2][4] = {}, au[2][2][4] = {};
        for (int hs = 0; hs < 2 * (D / 256); hs++) {
            const int sb = hs >> 1, h2 = hs & 1;
#pragma unroll 1
            for (int u = tid; u < 2 * 32 * 2 * 2; u += 32 * MOE_NW_GU) {
                const int mat = u >> 7, row = (u >> 2) & 31, c = 2 * h2 + ((u >> 1) & 1), h = u & 1;
                const QTensor &W = mat ? U : G;
                const SbScales sc = sb_scales<QT>(W, row0 + row, sb);
                int2 q[4]; int f[4];
                frags<QT>(W, row0 + row, sb, c, h, q, f);
#pragma unroll
                for (int g = 0; g < 4; g++) *(uint4 *)&Ws[mat][row][16 * (f[g] - 8 * h2) + 8 * h] = frag_h8<QT>(q[g], sc, f[g]);
            }
            __syncthreads();
            if (live) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const int kk = 16 * k + 2 * tig, xk = sb * 256 + h2 * 128 + kk;
                    uint32_t B[2][2];
#pragma unroll
                    for (int n = 0; n < 2; n++) { B[n][0] = *(const uint32_t *)(xr[n] + xk); B[n][1] = *(const uint32_t *)(xr[n] + xk + 8); }
#pragma unroll
                    for (int m = 0; m < 2; m++) {
                        const int r = 16 * m + gq;
                        const uint32_t A0[4] = {*(const uint32_t *)&Ws[0][r][kk], *(const uint32_t *)&Ws[0][r + 8][kk],
                                                *(const uint32_t *)&Ws[0][r][kk + 8], *(const uint32_t *)&Ws[0][r + 8][kk + 8]};
                        const uint32_t A1[4] = {*(const uint32_t *)&Ws[1][r][kk], *(const uint32_t *)&Ws[1][r + 8][kk],
                                                *(const uint32_t *)&Ws[1][r][kk + 8], *(const uint32_t *)&Ws[1][r + 8][kk + 8]};
#pragma unroll
                        for (int n = 0; n < 2; n++) { mma_f16(ag[m][n], A0, B[n]); mma_f16(au[m][n], A1, B[n]); }
                    }
                }
            }
            __syncthreads();
        }
        if (!live) continue;
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int t = 16 * wave + 8 * n + 2 * tig + (i & 1), r = 16 * m + gq + 8 * (i >> 1);
                    if (t < nt) H[(size_t)(m0 + t) * FE + blockIdx.x * 32 + r] = __float2half(silu(ag[m][n][i]) * au[m][n][i]);
                }
    }
}

template <int QT>
__global__ __launch_bounds__(32 * MOE_NW_DN) void k_moe_down_g(QTensor d0, QTensor d1, const MoeWork *work, const half *H, const float *wts, half *Y) {
    __shared__ __align__(16) half Ws[32][FE + 8];
    const MoeWork wk = work[blockIdx.y];
    int rs;
    const QTensor W = down_view(wk.tier ? d1 : d0, wk.slot, rs);
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, gq = lane / 4, tig = lane % 4;
    const int row0 = rs + blockIdx.x * 32;
    for (int u = tid; u < 32 * (FE / 64) * 2; u += 32 * MOE_NW_DN) {
        const int row = u / (FE / 64 * 2), ch = (u >> 1) % (FE / 64), h = u & 1;
        int2 q[4]; float sc[4];
        frags64<QT>(W, row0 + row, ch, h, q, sc);
#pragma unroll
        for (int gi = 0; gi < 4; gi++) *(uint4 *)&Ws[row][ch * 64 + (gi >> 1) * 32 + (gi & 1) * 16 + 8 * h] = to_h8(q[gi], sc[gi]);
    }
    __syncthreads();
    for (int tt = 0; tt < wk.n; tt += MOE_NW_DN * 16) {
        const int nt = min(MOE_NW_DN * 16, wk.n - tt), m0 = wk.m0 + tt;
        if (16 * wave >= nt) continue;
        const half *hr[2] = {H + (size_t)(m0 + min(16 * wave + gq, nt - 1)) * FE, H + (size_t)(m0 + min(16 * wave + 8 + gq, nt - 1)) * FE};
        float acc[2][2][4] = {};
#pragma unroll 4
        for (int k = 0; k < FE / 16; k++) {
            const int kk = 16 * k + 2 * tig;
            uint32_t B[2][2];
#pragma unroll
            for (int n = 0; n < 2; n++) { B[n][0] = *(const uint32_t *)(hr[n] + kk); B[n][1] = *(const uint32_t *)(hr[n] + kk + 8); }
#pragma unroll
            for (int m = 0; m < 2; m++) {
                const int r = 16 * m + gq;
                const uint32_t A[4] = {*(const uint32_t *)&Ws[r][kk], *(const uint32_t *)&Ws[r + 8][kk], *(const uint32_t *)&Ws[r][kk + 8],
                                       *(const uint32_t *)&Ws[r + 8][kk + 8]};
#pragma unroll
                for (int n = 0; n < 2; n++) mma_f16(acc[m][n], A, B[n]);
            }
        }
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int t = 16 * wave + 8 * n + 2 * tig + (i & 1), r = 16 * m + gq + 8 * (i >> 1);
                    if (t >= nt) continue;
                    const size_t e = (size_t)(m0 + t);
                    Y[e * D + blockIdx.x * 32 + r] = __float2half(wts[e] * acc[m][n][i]);
                }
    }
}

void moe_gemm_gateup(const QTensor &g0, const QTensor &u0, const QTensor &g1, const QTensor &u1, const MoeWork *work, int nwork,
                     const half *X, half *H, cudaStream_t st, int slot_type) {
    if (!nwork) return;
    const dim3 grid(FE / 32, nwork);
    if (slot_type >= 0 && slot_type != IQ4_XS) {
        if (g0.type != IQ4_XS) FAIL("moe_gemm_gateup: an expert pool of type %d", g0.type);
        if (slot_type == Q3_K) k_moe_gateup<Q3_K, true><<<grid, 32 * MOE_NW_GU, 0, st>>>(g0, u0, g1, u1, work, X, H);
        else if (slot_type == Q2_K) k_moe_gateup<Q2_K, true><<<grid, 32 * MOE_NW_GU, 0, st>>>(g0, u0, g1, u1, work, X, H);
        else FAIL("moe_gemm_gateup: pool slots of type %d", slot_type);
        return;
    }
    const int t = g0.type;
    if (g1.q && g1.type != t) FAIL("moe_gemm_gateup: tiers must share a format per call");
    if (t == IQ4_XS) k_moe_gateup<IQ4_XS><<<grid, 32 * MOE_NW_GU, 0, st>>>(g0, u0, g1, u1, work, X, H);
    else if (t == Q3_K) k_moe_gateup<Q3_K><<<grid, 32 * MOE_NW_GU, 0, st>>>(g0, u0, g1, u1, work, X, H);
    else if (t == Q2_K) k_moe_gateup<Q2_K><<<grid, 32 * MOE_NW_GU, 0, st>>>(g0, u0, g1, u1, work, X, H);
    else FAIL("moe_gemm_gateup: type %d", t);
}
void moe_gemm_down(const QTensor &d0, const QTensor &d1, const MoeWork *work, int nwork, const half *H, const float *wts, half *Y, cudaStream_t st) {
    if (!nwork) return;
    const dim3 grid(D / 32, nwork);
    if (d0.type == IQ4_NL) k_moe_down_g<IQ4_NL><<<grid, 32 * MOE_NW_DN, 0, st>>>(d0, d1, work, H, wts, Y);
    else FAIL("moe_gemm_down: type %d", d0.type);
}
