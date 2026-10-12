// Quantized weights: their structure-of-arrays layout (filled by relayout.hip), decode GEMVs built on int8
// dot products, dequantization to fp16 for the prefill GEMMs, and activation quantization.
#include "kernels.h"
#include <algorithm>
#include <vector>

// ---- weight arrays (relayout from GGUF blocks: relayout.hip) -------------------------------------

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

// ---- unpacking helpers --------------------------------------------------------------------

// IQ4_XS non-linear codebook, as four little-endian words for v_perm_b32 lookups.
__device__ __forceinline__ int2 iq4_lookup(uint32_t q) {
    constexpr uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;  // -127 ... 113
    const uint32_t lo = q & 0x07070707u, hi = (q >> 4) & 0x07070707u;
    const uint32_t a0 = __builtin_amdgcn_perm(T1, T0, lo), a1 = __builtin_amdgcn_perm(T3, T2, lo);
    const uint32_t b0 = __builtin_amdgcn_perm(T1, T0, hi), b1 = __builtin_amdgcn_perm(T3, T2, hi);
    const uint32_t ma = 0x03020100u | ((q & 0x08080808u) >> 1), mb = 0x03020100u | (((q >> 4) & 0x08080808u) >> 1);
    return make_int2(__builtin_amdgcn_perm(a1, a0, ma), __builtin_amdgcn_perm(b1, b0, mb));
}
__device__ __forceinline__ int iq4_scale(const QTensor &W, size_t sb, int ib) {   // sb = super-block index
    return (((W.sc[sb * 4 + (ib >> 1)] >> 4 * (ib & 1)) & 0xf) | (((W.sh[sb] >> 2 * ib) & 3) << 4)) - 32;
}
// Q6_K: 4 low bits + 2 high bits, minus 32, for 4 packed bytes at once.
__device__ __forceinline__ int q6_unpack(uint32_t lo4, uint32_t hi2) {
    const uint32_t x = (lo4 & 0x0F0F0F0Fu) | ((hi2 & 0x03030303u) << 4);
    return (int)((x & 0x1F1F1F1Fu) | ((~x & 0x20202020u) * 7));
}
__device__ __forceinline__ int dot16(int4 w, int4 x, int acc) {
    return dot4(w.w, x.w, dot4(w.z, x.z, dot4(w.y, x.y, dot4(w.x, x.x, acc))));
}

// ---- decode GEMV: y[t][row] (+)= W[row] . x[t] for T <= 8 tokens ----------------------------------
// A wave computes R rows at once (R consecutive rows of one matrix, or the gate and up rows of the
// FFN), so every activation load is reused R times; lanes stride over 32- or 64-weight chunks.
// One launch covers up to 3 matrices stacked by rows (e.g. q|k|v) with their own types and outputs.
// Modes: STORE, ADD (residual stream), GLU: silu(gate) * up quantized straight into the int8 input of
// the down projection (1024-thread workgroups = 32 rows = one int8 block per token).

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
                for (int r = 0; r < R; r++) acc[r][t] += dw[r] * xd * dot16(w1[r], b, dot16(w0[r], a, 0));
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
                for (int r = 0; r < R; r++) acc[r][t] += dl[r] * xd * dot16(hi[r], b, dot16(lo[r], a, 0));
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
                    for (int r = 0; r < R; r++) acc[r][t] += sc[r][k] * xd * dot16(q[r][k], a, 0);
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
    if (wave < T) {
        const float v = act[wave][lane], d = warp_max(fabsf(v)) / 127.0f;
        a.gq.q[(size_t)wave * N + base + lane] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
        if (lane == 0) a.gq.d[((size_t)wave * N + base) / 32] = d;
    }
}

template <int T>
static void gemv_launch(const GemvArgs &a, int rows, hipStream_t st) {
    if (a.mode == GEMV_GLU) k_gemv_glu<T><<<rows / 32, 1024, 0, st>>>(a);
    else if (T >= 5) k_gemv<T, 2><<<(rows + 15) / 16, 256, 0, st>>>(a);   // amortize activation loads
    else k_gemv<T, 1><<<(rows + 7) / 8, 256, 0, st>>>(a);
}

void gemv(const GemvArgs &a, hipStream_t st) {
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

void gemv(const QTensor &W, QAct x, int T, float *y, int ldy, int mode, hipStream_t st) {
    GemvArgs a;
    a.seg[0] = {W, y, ldy}; a.x = x; a.T = T; a.mode = mode;
    gemv(a, st);
}

// ---- small-batch GEMM: y[t][row] (+)= W[row] . x[t] for prefill chunks of 9..384 tokens --------------
// Weights are dequantized in registers straight into WMMA fragments (fp16 x fp16 -> fp32), so a pass
// reads the quantized weights once instead of writing and re-reading an fp16 copy of every matrix (that
// costs ~140 ms per forward and dominated short prefills). A workgroup is 4 waves on the same 32 rows and
// 16*NT tokens, each wave taking a quarter of K; the partial sums are added through LDS.
//
// WMMA wants every 16-wide K fragment in both half-waves. Here the lower half produces K 0..7 and the
// upper half K 8..15 (dequantizing or loading only 8 values per lane), then the halves swap with
// v_permlanex16. The upper half ends up holding (8..15, 0..7): each half-wave's outputs are computed from
// its own operands, so a K order that A and B share is exact (checked in tools/test_quant.hip).

typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));

__device__ __forceinline__ h16 join_halves(u4v own) {
    u8v r;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = own[i];
        r[4 + i] = __builtin_amdgcn_permlanex16(own[i], own[i], 0x76543210u, 0xfedcba98u, false, false);
    }
    return __builtin_bit_cast(h16, r);
}

// 8 int8 values times d -> 8 halves. int8 -> fp16 is exact via the bits of 1024 + (q + 128).
__device__ __forceinline__ u4v to_h8(int2 q, float d) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const h2 dd = {(_Float16)d, (_Float16)d}, off = {(_Float16)-1152.0f, (_Float16)-1152.0f};
    u4v r;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const uint32_t u = (i ? q.y : q.x) ^ 0x80808080u;
        r[2 * i] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)) + off) * dd);
        r[2 * i + 1] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)) + off) * dd);
    }
    return r;
}

// Scale data of one 256-weight super-block of a row, and the scale of its fragment f (weights 16f..16f+15).
struct SbScales { float d; int4 raw; };
template <int QT>
__device__ __forceinline__ SbScales sb_scales(const QTensor &W, int row, int sb) {
    const size_t gsb = (size_t)row * (W.cols / 256) + sb;
    if constexpr (QT == IQ4_XS) return {__half2float(W.d[gsb]), make_int4(((const int *)W.sc)[gsb], W.sh[gsb], 0, 0)};
    else if constexpr (QT == Q6_K) return {__half2float(W.d[gsb]), ((const int4 *)W.sc)[gsb]};
    else return {1.0f, ((const int4 *)W.d)[gsb]};                  // Q8_0: 8 fp16 block scales
}
template <int QT>
__device__ __forceinline__ float frag_scale(const SbScales &s, int f) {
    const uint32_t w[4] = {(uint32_t)s.raw.x, (uint32_t)s.raw.y, (uint32_t)s.raw.z, (uint32_t)s.raw.w};
    if constexpr (QT == IQ4_XS) return s.d * ((int)(((w[0] >> 4 * (f >> 1)) & 0xf) | (((w[1] >> 2 * (f >> 1)) & 3) << 4)) - 32);
    else if constexpr (QT == Q6_K) return s.d * (int8_t)(w[f >> 2] >> 8 * (f & 3));
    else return __half2float(__ushort_as_half((uint16_t)(w[f >> 2] >> 16 * ((f >> 1) & 1))));
}

// This lane half's 8 int8 values of the 4 fragments in step c (weights 64c..64c+63) of super-block sb, and
// the fragments' indices (Q6_K interleaves its 16-weight groups, so they are not consecutive).
template <int QT>
__device__ __forceinline__ void frags(const QTensor &W, int row, int sb, int c, int hf, int2 q[4], int f[4]) {
    const int K = W.cols;
    if constexpr (QT == IQ4_XS) {
        const uint8_t *p = W.q + (size_t)row * (K / 2) + sb * 128 + c * 32 + 8 * hf;     // sub-blocks 2c, 2c+1
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int2 w = *(const int2 *)(p + 16 * h);
            const int2 a = iq4_lookup(w.x), b = iq4_lookup(w.y);                     // low / high nibbles
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
    } else {
        const int8_t *p = (const int8_t *)W.q + (size_t)row * K + sb * 256 + c * 64 + 8 * hf;
#pragma unroll
        for (int g = 0; g < 4; g++) { q[g] = *(const int2 *)(p + 16 * g); f[g] = 4 * c + g; }
    }
}

template <int QT, int NT>
__global__ __launch_bounds__(128) void k_gemm_q(QTensor W, const half *x, int ldx, int T, void *y, int ldy, int mode, int y_f16) {
    __shared__ float red[4][16 * NT][33];
    const int wave = threadIdx.x / 32, l = threadIdx.x % 32, hf = l >> 4, r = l & 15;
    const int row0 = blockIdx.x * 32, t0 = blockIdx.y * 16 * NT, nsb = W.cols / 256, sb1 = (wave + 1) * nsb / 4;
    const _Float16 *xl = (const _Float16 *)x + (size_t)(t0 + r) * ldx + 8 * hf;   // B fragments: lane = token
    f8 acc[2][NT] = {};
    for (int sb = wave * nsb / 4; sb < sb1; sb++) {
        const SbScales s0 = sb_scales<QT>(W, row0 + r, sb), s1 = sb_scales<QT>(W, row0 + 16 + r, sb);
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int2 q0[4], q1[4]; int f[4];
            frags<QT>(W, row0 + r, sb, c, hf, q0, f);                                 // A fragments: lane = row
            frags<QT>(W, row0 + 16 + r, sb, c, hf, q1, f);
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const h16 a0 = join_halves(to_h8(q0[g], frag_scale<QT>(s0, f[g])));
                const h16 a1 = join_halves(to_h8(q1[g], frag_scale<QT>(s1, f[g])));
#pragma unroll
                for (int j = 0; j < NT; j++) {
                    const h16 b = join_halves(*(const u4v *)(xl + (size_t)16 * j * ldx + sb * 256 + 16 * f[g]));
                    acc[0][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc[0][j]);
                    acc[1][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc[1][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) red[wave][16 * j + r][16 * i + 2 * e + hf] = acc[i][j][e];
    __syncthreads();
    for (int o = threadIdx.x; o < 32 * 16 * NT; o += 128) {
        const int m = o & 31, t = o >> 5;
        if (t0 + t >= T) break;
        const float v = red[0][t][m] + red[1][t][m] + red[2][t][m] + red[3][t][m];
        const size_t i = (size_t)(t0 + t) * ldy + row0 + m;
        if (y_f16) ((half *)y)[i] = __float2half(v);
        else ((float *)y)[i] = mode == GEMV_ADD ? ((float *)y)[i] + v : v;
    }
}

template <int QT>
static void gemm_q_launch(const QTensor &W, const half *x, int ldx, int T, void *y, int ldy, int mode, bool y_f16, hipStream_t st) {
    const int nt = std::min(4, (T + 15) / 16);
    const dim3 grid(W.rows / 32, (T + 16 * nt - 1) / (16 * nt));
    switch (nt) {
        case 1: k_gemm_q<QT, 1><<<grid, 128, 0, st>>>(W, x, ldx, T, y, ldy, mode, y_f16); break;
        case 2: k_gemm_q<QT, 2><<<grid, 128, 0, st>>>(W, x, ldx, T, y, ldy, mode, y_f16); break;
        case 3: k_gemm_q<QT, 3><<<grid, 128, 0, st>>>(W, x, ldx, T, y, ldy, mode, y_f16); break;
        default: k_gemm_q<QT, 4><<<grid, 128, 0, st>>>(W, x, ldx, T, y, ldy, mode, y_f16); break;
    }
}

void gemm_q(const QTensor &W, const half *x, int ldx, int T, void *y, int ldy, int mode, bool y_f16, hipStream_t st) {
    if (W.rows % 32 || W.cols % 256) FAIL("gemm_q: %dx%d not supported", W.rows, W.cols);
    if (W.type == IQ4_XS) gemm_q_launch<IQ4_XS>(W, x, ldx, T, y, ldy, mode, y_f16, st);
    else if (W.type == Q6_K) gemm_q_launch<Q6_K>(W, x, ldx, T, y, ldy, mode, y_f16, st);
    else if (W.type == Q8_0) gemm_q_launch<Q8_0>(W, x, ldx, T, y, ldy, mode, y_f16, st);
    else FAIL("gemm_q: type %d", W.type);
}

// ---- dequantization to fp16 (long prefill chunks feed hipBLASLt) ------------------------------

// 16 int8 values times a scale -> 16 halves, written as two 16-byte stores.
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

void dequant(const QTensor &W, half *out, hipStream_t st) {
    const int64_t n = (int64_t)W.rows * W.cols;
    const int64_t nchunks = W.type == Q6_K ? n / 64 : n / 32;
    k_dequant<<<(nchunks + 255) / 256, 256, 0, st>>>(W, out, nchunks);
}

// Dequantize selected rows (token embeddings) to fp32.
__global__ void k_embed(QTensor W, const int *tokens, float *out) {
    const int t = blockIdx.y, K = W.cols, row = tokens[t];
    for (int c = threadIdx.x; c < K / 32; c += blockDim.x) {          // IQ4_XS rows
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

void embed(const QTensor &W, const int *tokens, int T, float *out, hipStream_t st) {
    if (W.type != IQ4_XS) FAIL("embed: expected IQ4_XS token embeddings");
    k_embed<<<dim3(1, T), 256, 0, st>>>(W, tokens, out);
}

// ---- activation quantization: fp32 -> int8 with one scale per 32 values ------------------------

__global__ void k_quant_act(const float *x, int64_t n, QAct out) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;   // one lane per value, one wave per block
    const float v = i < n ? x[i] : 0.0f;
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    if (i < n) {
        out.q[i] = (int8_t)__float2int_rn(d > 0 ? v / d : 0.0f);
        if ((threadIdx.x & 31) == 0) out.d[i / 32] = d;
    }
}

void quant_act(const float *x, int64_t n, QAct out, hipStream_t st) {
    k_quant_act<<<(n + 255) / 256, 256, 0, st>>>(x, n, out);
}
