#include "kernels.h"
#include <algorithm>
#include <vector>

QTensor make_qtensor(Arena &a, int type, int rows, int cols) {
    QTensor t; t.type = type; t.rows = rows; t.cols = cols;
    const size_t n = (size_t)rows * cols;
    switch (type) {
        case F32: t.f = a.get<float>(n); break;
        case F16: t.f = a.get<half>(n); break;
        case Q8_0: t.q = a.get(n); t.d = a.get<half>(n / 32); break;
        case Q6_K: t.q = a.get(n / 2); t.qh = a.get(n / 4); t.sc = a.get(n / 16); t.d = a.get<half>(n / 256); break;
        case IQ4_XS: t.q = a.get(n / 2); t.sc = a.get(n / 64); t.sh = a.get<uint16_t>(n / 256); t.d = a.get<half>(n / 256); break;
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
__device__ __forceinline__ int dot16(int4 w, int4 x, int acc) {
    return dot4(w.w, x.w, dot4(w.z, x.z, dot4(w.y, x.y, dot4(w.x, x.x, acc))));
}


template <int QT, int T, int R>
__device__ __forceinline__ void rows_dot(const QTensor *const W[R], const int row[R], int lane, QAct x, float acc[R][T]) {
    const int K = W[0]->cols;
    if constexpr (QT == Q8_0) {
        for (int ch = lane; ch < K / 32; ch += 32) {
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
        for (int ch = lane; ch < K / 32; ch += 32) {
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
    } else if constexpr (QT == Q6_K) {
        for (int ch = lane; ch < K / 64; ch += 32) {
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

template <int T, int R>
__device__ __forceinline__ void rows_dot_any(const QTensor *const W[R], const int row[R], int lane, QAct x, float acc[R][T]) {
    if (W[0]->type == IQ4_XS) rows_dot<IQ4_XS, T, R>(W, row, lane, x, acc);
    else if (W[0]->type == Q6_K) rows_dot<Q6_K, T, R>(W, row, lane, x, acc);
    else rows_dot<Q8_0, T, R>(W, row, lane, x, acc);
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
    rows_dot_any<T, R>(W, rows, lane, a.x, acc);
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) {
                float *o = a.seg[s].y + (size_t)t * a.seg[s].ldy + row + r;
                *o = a.mode == GEMV_ADD ? *o + v : v;
            }
        }
}

template <int T>
__global__ __launch_bounds__(1024) void k_gemv_glu(GemvArgs a) {
    __shared__ float act[T][32];
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, N = a.seg[0].W.rows;
    const int base = blockIdx.x * 32, row = base + wave;
    const QTensor *W[2] = {&a.seg[0].W, &a.up};
    const int rows[2] = {row, row};
    float acc[2][T] = {};
    rows_dot_any<T, 2>(W, rows, lane, a.x, acc);
#pragma unroll
    for (int t = 0; t < T; t++) {
        const float v = silu(warp_sum(acc[0][t])) * warp_sum(acc[1][t]);
        if (lane == 0) act[t][wave] = v;
    }
    __syncthreads();
    if (wave < T) quant_put(a.gq, (size_t)wave * N + base + lane, act[wave][lane]);
}

template <int T, int CW, int P = 1> struct Slabs {
    alignas(16) int8_t q[2][T][32 * CW * P];
    float d[2][T][CW * P];
};

template <int T, int CW, int P>
__device__ __forceinline__ void stage(Slabs<T, CW, P> &S, int buf, QAct x, int K, int s) {
    constexpr int SLAB = 32 * CW * P;
    const int k0 = s * SLAB, n = min(SLAB, K - k0), nb = n / 32;
    for (int i = threadIdx.x * 16; i < T * n; i += 256 * 16) {
        const int t = i / n, k = i - t * n;
        *(int4 *)&S.q[buf][t][k] = *(const int4 *)(x.q + (size_t)t * K + k0 + k);
    }
    for (int i = threadIdx.x; i < T * nb; i += 256) {
        const int t = i / nb, j = i - t * nb;
        S.d[buf][t][j] = x.d[t * (K / 32) + k0 / 32 + j];
    }
}

template <int QT, int T, int R, int CW, int P>
__device__ __forceinline__ void chunk_dot(const QTensor &W, int row0, int ch, const Slabs<T, CW, P> &S, int buf, int c, float acc[R][T]) {
    const int K = W.cols;
    if constexpr (QT == Q8_0) {
        int4 w0[R], w1[R]; float dw[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            const int8_t *wr = (const int8_t *)W.q + (size_t)row * K + ch * 32;
            w0[r] = *(const int4 *)wr; w1[r] = *(const int4 *)(wr + 16);
            dw[r] = __half2float(W.d[(size_t)row * (K / 32) + ch]);
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
            const int4 a = *(const int4 *)&S.q[buf][t][c * 32], b = *(const int4 *)&S.q[buf][t][c * 32 + 16];
            const float xd = S.d[buf][t][c];
#pragma unroll
            for (int r = 0; r < R; r++) acc[r][t] = __fmaf_rn(dw[r] * xd, (float)dot16(w1[r], b, dot16(w0[r], a, 0)), acc[r][t]);
        }
    } else if constexpr (QT == IQ4_XS) {
        const int sb = ch >> 3, ib = ch & 7;
        int4 lo[R], hi[R]; float dl[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            const size_t gsb = (size_t)row * (K / 256) + sb;
            const int4 w = *(const int4 *)(W.q + (size_t)row * (K / 2) + sb * 128 + ib * 16);
            const int2 p0 = iq4_lookup(w.x), p1 = iq4_lookup(w.y), p2 = iq4_lookup(w.z), p3 = iq4_lookup(w.w);
            lo[r] = make_int4(p0.x, p1.x, p2.x, p3.x); hi[r] = make_int4(p0.y, p1.y, p2.y, p3.y);
            dl[r] = __half2float(W.d[gsb]) * iq4_scale(W, gsb, ib);
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
            const int4 a = *(const int4 *)&S.q[buf][t][c * 32], b = *(const int4 *)&S.q[buf][t][c * 32 + 16];
            const float xd = S.d[buf][t][c];
#pragma unroll
            for (int r = 0; r < R; r++) acc[r][t] = __fmaf_rn(dl[r] * xd, (float)dot16(hi[r], b, dot16(lo[r], a, 0)), acc[r][t]);
        }
    } else if constexpr (QT == Q6_K) {
        const int sb = ch >> 2, h = (ch >> 1) & 1, cc = ch & 1, lsb = c >> 2;
        int4 q[R][4]; float sc[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int row = row0 + r;
            const size_t gsb = (size_t)row * (K / 256) + sb;
            const uint8_t *ql = W.q + (size_t)row * (K / 2) + sb * 128 + h * 64 + cc * 16;
            const int4 qa = *(const int4 *)ql, qb = *(const int4 *)(ql + 32);
            const int4 H = *(const int4 *)(W.qh + (size_t)row * (K / 4) + sb * 64 + h * 32 + cc * 16);
            const int8_t *s8 = (const int8_t *)W.sc + gsb * 16 + h * 8 + cc;
            const float dd = __half2float(W.d[gsb]);
            q[r][0] = make_int4(q6_unpack(qa.x, H.x), q6_unpack(qa.y, H.y), q6_unpack(qa.z, H.z), q6_unpack(qa.w, H.w));
            q[r][1] = make_int4(q6_unpack(qb.x, H.x >> 2), q6_unpack(qb.y, H.y >> 2), q6_unpack(qb.z, H.z >> 2), q6_unpack(qb.w, H.w >> 2));
            q[r][2] = make_int4(q6_unpack(qa.x >> 4, H.x >> 4), q6_unpack(qa.y >> 4, H.y >> 4), q6_unpack(qa.z >> 4, H.z >> 4), q6_unpack(qa.w >> 4, H.w >> 4));
            q[r][3] = make_int4(q6_unpack(qb.x >> 4, H.x >> 6), q6_unpack(qb.y >> 4, H.y >> 6), q6_unpack(qb.z >> 4, H.z >> 6), q6_unpack(qb.w >> 4, H.w >> 6));
#pragma unroll
            for (int k = 0; k < 4; k++) sc[r][k] = dd * s8[2 * k];
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int blk = lsb * 8 + h * 4 + k;
                const int4 a = *(const int4 *)&S.q[buf][t][blk * 32 + cc * 16];
                const float xd = S.d[buf][t][blk];
#pragma unroll
                for (int r = 0; r < R; r++) acc[r][t] = __fmaf_rn(sc[r][k] * xd, (float)dot16(q[r][k], a, 0), acc[r][t]);
            }
        }
    }
}

template <int T, int R, int CW, int P>
__device__ __forceinline__ void rows_staged(const QTensor &W, int row0, bool live, QAct x, Slabs<T, CW, P> &S, float acc[R][T]) {
    const int K = W.cols, lane = threadIdx.x % 32, nslab = (K + 32 * CW * P - 1) / (32 * CW * P);
    stage<T, CW, P>(S, 0, x, K, 0);
    for (int s = 0; s < nslab; s++) {
        const int buf = s & 1, nch = min(32 * P, (K - s * 32 * CW * P) / CW);
        __syncthreads();
        if (s + 1 < nslab) stage<T, CW, P>(S, buf ^ 1, x, K, s + 1);
#pragma unroll
        for (int p = 0; p < P; p++) {
            const int c = p * 32 + lane;
            if (live && c < nch) {
                if constexpr (CW == 64) chunk_dot<Q6_K, T, R, CW, P>(W, row0, s * 32 * P + c, S, buf, c, acc);
                else if (W.type == IQ4_XS) chunk_dot<IQ4_XS, T, R, CW, P>(W, row0, s * 32 * P + c, S, buf, c, acc);
                else chunk_dot<Q8_0, T, R, CW, P>(W, row0, s * 32 * P + c, S, buf, c, acc);
            }
        }
    }
}

template <int T, int R, int CW, int P = 1>
__global__ __launch_bounds__(256) void k_gemv_staged(GemvArgs a) {
    __shared__ Slabs<T, CW, P> S;
    int row = (blockIdx.x * 8 + threadIdx.x / 32) * R, s = 0;
    const int lane = threadIdx.x % 32;
    while (s < a.nseg - 1 && row >= a.seg[s].W.rows) row -= a.seg[s++].W.rows;
    const QTensor &W = a.seg[s].W;
    const bool live = row < W.rows;
    float acc[R][T] = {};
    rows_staged<T, R, CW, P>(W, row, live, a.x, S, acc);
    if (!live) return;
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) {
                float *o = a.seg[s].y + (size_t)t * a.seg[s].ldy + row + r;
                *o = a.mode == GEMV_ADD ? *o + v : v;
            }
        }
}

template <int T>
static void gemv_launch(const GemvArgs &a, int rows, cudaStream_t st) {
    if (a.mode == GEMV_GLU) k_gemv_glu<T><<<rows / 32, 1024, 0, st>>>(a);
    else if constexpr (T <= 4) k_gemv<T, 1><<<(rows + 7) / 8, 256, 0, st>>>(a);
    else if (a.seg[0].W.type == Q6_K) k_gemv_staged<T, 2, 64><<<(rows + 15) / 16, 256, 0, st>>>(a);
    else if constexpr (T < 8) k_gemv_staged<T, 2, 32, 2><<<(rows + 15) / 16, 256, 0, st>>>(a);
    else k_gemv_staged<T, 2, 32><<<(rows + 15) / 16, 256, 0, st>>>(a);
}

void gemv(const GemvArgs &a, cudaStream_t st) {
    int rows = 0;
    for (int i = 0; i < a.nseg; i++) {
        if ((a.seg[i].W.type == Q6_K) != (a.seg[0].W.type == Q6_K)) FAIL("gemv: Q6_K and other types in one launch");
        if (a.T > 4 && a.seg[i].W.rows % 2) FAIL("gemv: %d rows (two per wave)", a.seg[i].W.rows);
        rows += a.seg[i].W.rows;
    }
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


template <int QT>
__device__ __forceinline__ void mmq_a(const QTensor &W, int row, int kb, int tig, int &lo, int &hi, float &slo, float &shi) {
    const int K = W.cols;
    if constexpr (QT == IQ4_XS) {
        const int sb = kb >> 3, ib = kb & 7;
        const size_t gsb = (size_t)row * (K / 256) + sb;
        const int2 v = iq4_lookup(*(const uint32_t *)(W.q + (size_t)row * (K / 2) + sb * 128 + ib * 16 + tig * 4));
        lo = v.x; hi = v.y;
        slo = shi = __half2float(W.d[gsb]) * iq4_scale(W, gsb, ib);
    } else if constexpr (QT == Q8_0) {
        const int8_t *p = (const int8_t *)W.q + (size_t)row * K + kb * 32 + tig * 4;
        lo = *(const int *)p; hi = *(const int *)(p + 16);
        slo = shi = __half2float(W.d[(size_t)row * (K / 32) + kb]);
    } else {
        const int sb = kb >> 3, j = kb & 7, h = j >> 2, grp = j & 3;
        const size_t gsb = (size_t)row * (K / 256) + sb;
        const uint8_t *ql = W.q + (size_t)row * (K / 2) + sb * 128 + h * 64 + (grp & 1) * 32 + tig * 4;
        const uint8_t *qh = W.qh + (size_t)row * (K / 4) + sb * 64 + h * 32 + tig * 4;
        const int ns = (grp >> 1) * 4, hs = grp * 2;
        lo = q6_unpack(*(const uint32_t *)ql >> ns, *(const uint32_t *)qh >> hs);
        hi = q6_unpack(*(const uint32_t *)(ql + 16) >> ns, *(const uint32_t *)(qh + 16) >> hs);
        const int8_t *s8 = (const int8_t *)W.sc + gsb * 16 + h * 8 + grp * 2;
        const float dd = __half2float(W.d[gsb]);
        slo = dd * s8[0]; shi = dd * s8[1];
    }
}

constexpr int MMQ_WARPS = 4, MMQ_NT = 8;

struct MmqX {
    const int8_t *q[MMQ_NT], *r[MMQ_NT];
    const float *d[MMQ_NT][2];
    bool live[MMQ_NT];
    __device__ MmqX(QAct x, int T, int K, int t0, int g, int tig) {
#pragma unroll
        for (int n = 0; n < MMQ_NT; n++) {
            const int tb = min(t0 + n * 8 + g, T - 1), ta = t0 + n * 8 + tig * 2;
            live[n] = t0 + n * 8 < T;
            q[n] = x.q + (size_t)tb * K + tig * 4;
            r[n] = x.r ? x.r + (size_t)tb * K + tig * 4 : q[n];
            d[n][0] = x.d + (size_t)min(ta, T - 1) * (K / 32);
            d[n][1] = x.d + (size_t)min(ta + 1, T - 1) * (K / 32);
        }
    }
};

template <int QT, bool TWO>
__device__ __forceinline__ void mmq_step(const int a0[2], const int a1[2], const float s0[2], const float s1[2], const MmqX &X, int n, int kb,
                                         float acc[4]) {
    const int b0 = __ldg((const int *)(X.q[n] + kb * 32)), b1 = __ldg((const int *)(X.q[n] + kb * 32 + 16));
    constexpr float RS = TWO ? 1.0f / 254.0f : 1.0f;
    const float da = __ldg(X.d[n][0] + kb) * RS, db = __ldg(X.d[n][1] + kb) * RS;
    if constexpr (QT == Q6_K) {
        int clo[4] = {0, 0, 0, 0}, chi[4] = {0, 0, 0, 0};
        const int alo[2] = {a0[0], a1[0]}, ahi[2] = {a0[1], a1[1]};
        mma_s8_k16(clo, alo, b0);
        mma_s8_k16(chi, ahi, b1);
        if constexpr (TWO) {
            const int r0 = __ldg((const int *)(X.r[n] + kb * 32)), r1 = __ldg((const int *)(X.r[n] + kb * 32 + 16));
#pragma unroll
            for (int i = 0; i < 4; i++) { clo[i] *= 254; chi[i] *= 254; }
            mma_s8_k16(clo, alo, r0);
            mma_s8_k16(chi, ahi, r1);
        }
        acc[0] = __fmaf_rn(__fmaf_rn((float)clo[0], s0[0], (float)chi[0] * s0[1]), da, acc[0]);
        acc[1] = __fmaf_rn(__fmaf_rn((float)clo[1], s0[0], (float)chi[1] * s0[1]), db, acc[1]);
        acc[2] = __fmaf_rn(__fmaf_rn((float)clo[2], s1[0], (float)chi[2] * s1[1]), da, acc[2]);
        acc[3] = __fmaf_rn(__fmaf_rn((float)clo[3], s1[0], (float)chi[3] * s1[1]), db, acc[3]);
    } else {
        int c[4] = {0, 0, 0, 0};
        const int a[4] = {a0[0], a1[0], a0[1], a1[1]}, b[2] = {b0, b1};
        mma_s8(c, a, b);
        if constexpr (TWO) {
            const int rb[2] = {__ldg((const int *)(X.r[n] + kb * 32)), __ldg((const int *)(X.r[n] + kb * 32 + 16))};
#pragma unroll
            for (int i = 0; i < 4; i++) c[i] *= 254;
            mma_s8(c, a, rb);
        }
        acc[0] = __fmaf_rn((float)c[0], s0[0] * da, acc[0]);
        acc[1] = __fmaf_rn((float)c[1], s0[0] * db, acc[1]);
        acc[2] = __fmaf_rn((float)c[2], s1[0] * da, acc[2]);
        acc[3] = __fmaf_rn((float)c[3], s1[0] * db, acc[3]);
    }
}

template <int QT, bool TWO>
__device__ __forceinline__ void mmq_rows(const QTensor &W, int row0, const MmqX &X, int g, int tig, float acc[MMQ_NT][4]) {
    const int nkb = W.cols / 32;
    for (int kb = 0; kb < nkb; kb++) {
        int a0[2], a1[2];
        float s0[2], s1[2];
        mmq_a<QT>(W, row0 + g, kb, tig, a0[0], a0[1], s0[0], s0[1]);
        mmq_a<QT>(W, row0 + g + 8, kb, tig, a1[0], a1[1], s1[0], s1[1]);
#pragma unroll
        for (int n = 0; n < MMQ_NT; n++)
            if (X.live[n]) mmq_step<QT, TWO>(a0, a1, s0, s1, X, n, kb, acc[n]);
    }
}

template <int QT, bool TWO>
__global__ __launch_bounds__(MMQ_WARPS * 32) void k_mmq(QTensor W, QAct x, int T, void *y, int ldy, int mode, int y_f16) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, tig = lane % 4;
    const int row0 = (blockIdx.y * MMQ_WARPS + warp) * 16, t0 = blockIdx.x * 8 * MMQ_NT;
    if (row0 >= W.rows) return;
    const MmqX X(x, T, W.cols, t0, g, tig);
    float acc[MMQ_NT][4] = {};
    mmq_rows<QT, TWO>(W, row0, X, g, tig, acc);
#pragma unroll
    for (int n = 0; n < MMQ_NT; n++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int t = t0 + n * 8 + tig * 2 + (i & 1), row = row0 + g + (i >> 1) * 8;
            if (t >= T) continue;
            const size_t o = (size_t)t * ldy + row;
            if (y_f16) ((half *)y)[o] = __float2half(acc[n][i]);
            else ((float *)y)[o] = mode == GEMV_ADD ? ((float *)y)[o] + acc[n][i] : acc[n][i];
        }
}

template <int QT, bool TWO>
__global__ __launch_bounds__(MMQ_WARPS * 32) void k_mmq_glu(QTensor Wg, QTensor Wu, QAct x, int T, QAct out) {
    __shared__ float amax[MMQ_WARPS][8 * MMQ_NT];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, tig = lane % 4;
    const int row0 = (blockIdx.y * MMQ_WARPS + warp) * 16, t0 = blockIdx.x * 8 * MMQ_NT, N = Wg.rows;
    const MmqX X(x, T, Wg.cols, t0, g, tig);
    float ag[MMQ_NT][4] = {}, au[MMQ_NT][4] = {};
    mmq_rows<QT, TWO>(Wg, row0, X, g, tig, ag);
    mmq_rows<QT, TWO>(Wu, row0, X, g, tig, au);
#pragma unroll
    for (int n = 0; n < MMQ_NT; n++) {
        float m[2];
#pragma unroll
        for (int i = 0; i < 4; i++) ag[n][i] = silu(ag[n][i]) * au[n][i];
#pragma unroll
        for (int c = 0; c < 2; c++) {
            m[c] = fmaxf(fabsf(ag[n][c]), fabsf(ag[n][c + 2]));
            m[c] = fmaxf(m[c], lane_xor<4>(m[c])); m[c] = fmaxf(m[c], lane_xor<8>(m[c])); m[c] = fmaxf(m[c], lane_xor<16>(m[c]));
        }
        if (g == 0) { amax[warp][n * 8 + tig * 2] = m[0]; amax[warp][n * 8 + tig * 2 + 1] = m[1]; }
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < MMQ_NT; n++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int tc = n * 8 + tig * 2 + (i & 1), t = t0 + tc, row = row0 + g + (i >> 1) * 8;
            if (t >= T) continue;
            const float d = fmaxf(amax[warp][tc], amax[warp ^ 1][tc]) / 127.0f, v = ag[n][i];
            const float q = d > 0 ? rintf(v / d) : 0.0f;
            const size_t o = (size_t)t * N + row;
            out.q[o] = (int8_t)q;
            if (out.r) out.r[o] = (int8_t)(d > 0 ? fminf(127.0f, fmaxf(-127.0f, rintf((v - q * d) * (254.0f / d)))) : 0.0f);
            if (g == 0 && (i >> 1) == 0 && (warp & 1) == 0) out.d[o / 32] = d;
        }
}

template <bool TWO>
static void mmq_launch(const QTensor &W, QAct x, int T, void *y, int ldy, int mode, bool y_f16, cudaStream_t st) {
    const dim3 grid((T + 8 * MMQ_NT - 1) / (8 * MMQ_NT), (W.rows + 16 * MMQ_WARPS - 1) / (16 * MMQ_WARPS));
    if (W.type == IQ4_XS) k_mmq<IQ4_XS, TWO><<<grid, MMQ_WARPS * 32, 0, st>>>(W, x, T, y, ldy, mode, y_f16);
    else if (W.type == Q6_K) k_mmq<Q6_K, TWO><<<grid, MMQ_WARPS * 32, 0, st>>>(W, x, T, y, ldy, mode, y_f16);
    else if (W.type == Q8_0) k_mmq<Q8_0, TWO><<<grid, MMQ_WARPS * 32, 0, st>>>(W, x, T, y, ldy, mode, y_f16);
    else FAIL("gemm_q: type %d", W.type);
}

void gemm_q(const QTensor &W, QAct x, int T, void *y, int ldy, int mode, bool y_f16, cudaStream_t st) {
    if (W.rows % 16 || W.cols % 256) FAIL("gemm_q: %dx%d not supported", W.rows, W.cols);
    if (x.r) mmq_launch<true>(W, x, T, y, ldy, mode, y_f16, st);
    else mmq_launch<false>(W, x, T, y, ldy, mode, y_f16, st);
}

template <bool TWO>
static void mmq_glu_launch(const QTensor &Wg, const QTensor &Wu, QAct x, int T, QAct out, cudaStream_t st) {
    const dim3 grid((T + 8 * MMQ_NT - 1) / (8 * MMQ_NT), Wg.rows / (16 * MMQ_WARPS));
    if (Wg.type == IQ4_XS) k_mmq_glu<IQ4_XS, TWO><<<grid, MMQ_WARPS * 32, 0, st>>>(Wg, Wu, x, T, out);
    else if (Wg.type == Q6_K) k_mmq_glu<Q6_K, TWO><<<grid, MMQ_WARPS * 32, 0, st>>>(Wg, Wu, x, T, out);
    else if (Wg.type == Q8_0) k_mmq_glu<Q8_0, TWO><<<grid, MMQ_WARPS * 32, 0, st>>>(Wg, Wu, x, T, out);
    else FAIL("gemm_q_glu: type %d", Wg.type);
}

void gemm_q_glu(const QTensor &Wg, const QTensor &Wu, QAct x, int T, QAct out, cudaStream_t st) {
    if (Wg.rows % (16 * MMQ_WARPS) || Wg.cols % 256 || Wg.type != Wu.type || Wg.rows != Wu.rows || Wg.cols != Wu.cols)
        FAIL("gemm_q_glu: %dx%d types %d and %d not supported", Wg.rows, Wg.cols, Wg.type, Wu.type);
    if (x.r) mmq_glu_launch<true>(Wg, Wu, x, T, out, st);
    else mmq_glu_launch<false>(Wg, Wu, x, T, out, st);
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
    } else if (W.type == IQ4_XS) {
        const int row = ch / (K / 32), c = ch % (K / 32), sb = c >> 3, ib = c & 7;
        const size_t gsb = (size_t)row * (K / 256) + sb;
        const float dl = __half2float(W.d[gsb]) * iq4_scale(W, gsb, ib);
        const int4 w = *(const int4 *)(W.q + (size_t)row * (K / 2) + sb * 128 + ib * 16);
        const int2 p0 = iq4_lookup(w.x), p1 = iq4_lookup(w.y), p2 = iq4_lookup(w.z), p3 = iq4_lookup(w.w);
        half *o = out + (size_t)row * K + c * 32;
        store16(o, make_int4(p0.x, p1.x, p2.x, p3.x), dl);
        store16(o + 16, make_int4(p0.y, p1.y, p2.y, p3.y), dl);
    } else if (W.type == Q6_K) {
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
    } else if (W.type == F16) {
        const int4 *f = (const int4 *)((const half *)W.f + ch * 32);
        int4 *o = (int4 *)(out + ch * 32);
        o[0] = f[0]; o[1] = f[1]; o[2] = f[2]; o[3] = f[3];
    }
}

void dequant(const QTensor &W, half *out, cudaStream_t st) {
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
        const int sb = c >> 3, ib = c & 7;
        const size_t gsb = (size_t)row * (K / 256) + sb;
        const float dl = __half2float(W.d[gsb]) * iq4_scale(W, gsb, ib);
        const uint32_t *w = (const uint32_t *)(W.q + (size_t)row * (K / 2) + sb * 128 + ib * 16);
        float *o = out + (size_t)t * K + c * 32;
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
    if (W.type != IQ4_XS) FAIL("embed: expected IQ4_XS token embeddings");
    k_embed<<<dim3(1, T), 256, 0, st>>>(W, tokens, rows, out);
}


__global__ void k_quant_act(const float *x, int64_t n, QAct out) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) quant_put(out, i, x[i]);
}

void quant_act(const float *x, int64_t n, QAct out, cudaStream_t st) {
    k_quant_act<<<(n + 255) / 256, 256, 0, st>>>(x, n, out);
}
