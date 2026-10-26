#include "kernels/quant.h"
#include <metal_simdgroup_matrix>

constant constexpr int BK = 32;
constant constexpr uint MD = 2560, MF = 640, TOPK = 10, NEXP = 512;
constant constexpr uint XDN = 2560 * 640 / 32 * 18, RB_DN = 640 / 32 * 18;

struct PfMmArgs {
    uint rows, cols, rb;
    uint T, ldx, ldy;
    uint add;
    float scale;
};
struct PfSortArgs { uint P, bt; };
struct PfMoeArgs { uint layer, ldx, P, pad; };
struct XPtr { device const uchar *p; };

typedef simdgroup_float8x8 acc_t;
template <int F> constexpr float wpre() { return F == T_BF16 || F == T_F32 ? 2048.f : 1.f; }

template <int F, int BM, int BT, int SGM>
inline void mm_loop(device const uchar *wr, device const float *xr, uint units, threadgroup half *Ws, threadgroup half *Xs,
                    ushort tid, ushort sg, thread acc_t (&acc)[BT * SGM / 32][BM / SGM / 8]) {
    constexpr int SGT = 4 / SGM, TT = BT / SGT, TM = BM / SGM, LDW = BM + 8, LDX = BK + 8;
    constexpr int XPT = BT * BK / 128, XTPR = BK / XPT;
    const int sm = (sg % SGM) * TM, st = (sg / SGM) * TT;
    threadgroup half4 *xs = (threadgroup half4 *)(Xs + (tid / XTPR) * LDX + (tid % XTPR) * XPT);
    for (uint u = 0; u < units; u++) {
        if (BM >= 128 || tid < BM) {
            float w[32];
            deq32<F>(wr, u, w);
            for (int i = 0; i < 32; i++) Ws[i * LDW + tid] = half(w[i] * wpre<F>());
        }
        device const float4 *xp = (device const float4 *)(xr + u * BK);
        for (int i = 0; i < XPT / 4; i++) xs[i] = half4(xp[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 A[TT / 8], B[TM / 8];
            for (int i = 0; i < TT / 8; i++) simdgroup_load(A[i], Xs + (st + 8 * i) * LDX + kk, LDX);
            for (int j = 0; j < TM / 8; j++) simdgroup_load(B[j], Ws + kk * LDW + sm + 8 * j, LDW);
            for (int i = 0; i < TT / 8; i++)
                for (int j = 0; j < TM / 8; j++) simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <int F, int BM, int BT, int SGM>
kernel void pf_mm(constant PfMmArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]], device const float *X [[buffer(2)]],
                  device float *Y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int SGT = 4 / SGM, TT = BT / SGT, TM = BM / SGM, XPT = BT * BK / 128, XTPR = BK / XPT;
    threadgroup half Ws[BK * (BM + 8)], Xs[BT * (BK + 8)];
    const uint n0 = tg.x * BM, t0 = tg.y * BT;
    device const uchar *wr = W + (size_t)min(n0 + tid, a.rows - 1) * a.rb;
    device const float *xr = X + (size_t)min(t0 + tid / XTPR, a.T - 1) * a.ldx + (tid % XTPR) * XPT;
    acc_t acc[TT / 8][TM / 8];
    for (int i = 0; i < TT / 8; i++) for (int j = 0; j < TM / 8; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    mm_loop<F, BM, BT, SGM>(wr, xr, a.cols / 32, Ws, Xs, tid, sg, acc);
    threadgroup float *scr = (threadgroup float *)Ws + sg * 64;
    const uint sn = n0 + (sg % SGM) * TM, st = t0 + (sg / SGM) * TT;
    const float sc = a.scale / wpre<F>();
    for (int i = 0; i < TT / 8; i++)
        for (int j = 0; j < TM / 8; j++) {
            const uint t = st + 8 * i, n = sn + 8 * j;
            if (t >= a.T || n >= a.rows) continue;
            if (!a.add && t + 8 <= a.T && n + 8 <= a.rows) {
                if (sc != 1.f) simdgroup_multiply(acc[i][j], acc[i][j], acc_t(sc));
                simdgroup_store(acc[i][j], Y + (size_t)t * a.ldy + n, a.ldy);
            } else {
                simdgroup_store(acc[i][j], scr, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int q = 0; q < 2; q++) {
                    const uint e = lane * 2 + q, r = e / 8, c = e % 8;
                    if (t + r < a.T && n + c < a.rows) {
                        device float *d = Y + (size_t)(t + r) * a.ldy + n + c;
                        *d = (a.add ? *d : 0.f) + scr[e] * sc;
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
}

kernel void pf_moe_sort(constant PfSortArgs &a [[buffer(0)]], device const int *ids [[buffer(1)]], device uint *offs [[buffer(2)]],
                        device int *rows [[buffer(3)]], device int2 *tiles [[buffer(4)]], device uint *ntiles [[buffer(5)]],
                        ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint cnt[NEXP];
    threadgroup uint s_r[16], s_t[16];
    if (tid < NEXP) atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < a.P; p += 1024) atomic_fetch_add_explicit(&cnt[ids[p]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c = 0, nt = 0, pr = 0, pt = 0;
    if (tid < NEXP) {
        c = atomic_load_explicit(&cnt[tid], memory_order_relaxed);
        nt = (c + a.bt - 1) / a.bt;
        pr = simd_prefix_exclusive_sum(c);
        pt = simd_prefix_exclusive_sum(nt);
        if (lane == 31) { s_r[sg] = pr + c; s_t[sg] = pt + nt; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NEXP) {
        for (ushort i = 0; i < sg; i++) { pr += s_r[i]; pt += s_t[i]; }
        offs[tid] = pr;
        for (uint k = 0; k < nt; k++) tiles[pt + k] = int2(tid, pr + k * a.bt);
        if (tid == NEXP - 1) { offs[NEXP] = pr + c; *ntiles = pt + nt; }
        atomic_store_explicit(&cnt[tid], pr, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < a.P; p += 1024) rows[atomic_fetch_add_explicit(&cnt[ids[p]], 1u, memory_order_relaxed)] = p;
}

template <int F, int BT>
inline void moe_gu_loop(device const uchar *E, uint f0, ushort tid, device const float *xr, threadgroup half *Ws, threadgroup half *Xs,
                        ushort sg, thread acc_t (&acc)[BT / 8][4]) {
    const uint rb = row_bytes(F, MD), w = tid % 32;
    device const uchar *wr = E + XDN + (w < 16 ? 0 : MF * rb) + (f0 + 16 * (tid / 32) + w % 16) * rb;
    mm_loop<F, 128, BT, 4>(wr, xr, MD / 32, Ws, Xs, tid, sg, acc);
}

template <int BT>
kernel void pf_moe_gu(constant PfMoeArgs &a [[buffer(0)]], device const XPtr *table [[buffer(1)]], device const uint *offs [[buffer(2)]],
                      device const int *rows [[buffer(3)]], device const int2 *tiles [[buffer(4)]], device const uint *ntiles [[buffer(5)]],
                      device const float *X [[buffer(6)]], device float *H [[buffer(7)]], device const uchar *fmt [[buffer(8)]],
                      uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int XPT = BT * BK / 128, XTPR = BK / XPT;
    threadgroup half Ws[BK * (128 + 8)];
    threadgroup half Xs[BT * (BK + 8)];
    if (tg.y >= *ntiles) return;
    const int2 tl = tiles[tg.y];
    const uint e = tl.x, r0 = tl.y, nr = min((uint)BT, offs[e + 1] - r0), f0 = tg.x * 64;
    device const uchar *E = table[a.layer * NEXP + e].p;
    const uint lt = min((uint)(tid / XTPR), nr - 1);
    device const float *xr = X + (size_t)(rows[r0 + lt] / TOPK) * a.ldx + (tid % XTPR) * XPT;
    acc_t acc[BT / 8][4];
    for (int i = 0; i < BT / 8; i++) for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    const uint f = fmt[a.layer * NEXP + e];
    if (f == T_IQ4_XS) moe_gu_loop<T_IQ4_XS, BT>(E, f0, tid, xr, Ws, Xs, sg, acc);
    else if (f == T_Q3_K) moe_gu_loop<T_Q3_K, BT>(E, f0, tid, xr, Ws, Xs, sg, acc);
    else moe_gu_loop<T_Q2_K, BT>(E, f0, tid, xr, Ws, Xs, sg, acc);
    threadgroup float *scr = (threadgroup float *)Ws + sg * 128;
    const uint r = lane / 4, c = (lane % 4) * 2;
    for (int i = 0; i < BT / 8; i++) {
        const uint t = 8 * i + r;
        device float *dst = H + (size_t)(r0 + min(t, nr - 1)) * MF + f0 + 16 * sg + c;
        for (int j = 0; j < 2; j++) {
            simdgroup_store(acc[i][j], scr, 8);
            simdgroup_store(acc[i][j + 2], scr + 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const float2 g = *(threadgroup float2 *)(scr + lane * 2), u = *(threadgroup float2 *)(scr + 64 + lane * 2);
            if (t < nr) *(device float2 *)(dst + 8 * j) = g / (1.f + exp(-g)) * u;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template <int BT>
kernel void pf_moe_down(constant PfMoeArgs &a [[buffer(0)]], device const XPtr *table [[buffer(1)]], device const uint *offs [[buffer(2)]],
                        device const int *rows [[buffer(3)]], device const int2 *tiles [[buffer(4)]], device const uint *ntiles [[buffer(5)]],
                        device const float *H [[buffer(6)]], device float *Yp [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int XPT = BT * BK / 128, XTPR = BK / XPT;
    threadgroup half Ws[BK * (128 + 8)];
    threadgroup half Xs[BT * (BK + 8)];
    if (tg.y >= *ntiles) return;
    const int2 tl = tiles[tg.y];
    const uint e = tl.x, r0 = tl.y, nr = min((uint)BT, offs[e + 1] - r0), d0 = tg.x * 128;
    device const uchar *wr = table[a.layer * NEXP + e].p + (d0 + tid) * RB_DN;
    const uint lt = min((uint)(tid / XTPR), nr - 1);
    device const float *xr = H + (size_t)(r0 + lt) * MF + (tid % XTPR) * XPT;
    acc_t acc[BT / 8][4];
    for (int i = 0; i < BT / 8; i++) for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    mm_loop<T_IQ4_NL, 128, BT, 4>(wr, xr, MF / 32, Ws, Xs, tid, sg, acc);
    threadgroup float *scr = (threadgroup float *)Ws + sg * 64;
    const uint r = lane / 4, c = (lane % 4) * 2;
    for (int i = 0; i < BT / 8; i++) {
        const uint t = 8 * i + r;
        device float *dst = Yp + (size_t)rows[r0 + min(t, nr - 1)] * MD + d0 + 32 * sg + c;
        for (int j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], scr, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (t < nr) *(device float2 *)(dst + 8 * j) = *(threadgroup float2 *)(scr + lane * 2);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void pf_moe_wsum(constant uint &T [[buffer(0)]], device const float4 *Yp [[buffer(1)]], device const float *wts [[buffer(2)]],
                        device float4 *Y [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= T * (MD / 4)) return;
    const uint t = i / (MD / 4), d = i % (MD / 4);
    float4 s = 0;
    for (uint k = 0; k < TOPK; k++) s += wts[t * TOPK + k] * Yp[(t * TOPK + k) * (MD / 4) + d];
    Y[i] = s;
}

#define PF_TYPES(X) X(T_F32, f32) X(T_F16, f16) X(T_BF16, bf16) X(T_Q2_0, q2_0) X(T_Q4_0, q4_0) X(T_Q5_0, q5_0) X(T_Q8_0, q8_0) \
    X(T_IQ4_NL, iq4_nl) X(T_IQ4_XS, iq4_xs) X(T_Q3_K, q3_k) X(T_Q4_K, q4_k) X(T_Q5_K, q5_k) X(T_Q6_K, q6_k)
#define PF_MM(F, name) \
    template [[host_name("pf_mm_" #name)]] kernel void pf_mm<F, 64, 64, 2>(constant PfMmArgs &, device const uchar *, \
        device const float *, device float *, uint2, ushort, ushort, ushort);
PF_TYPES(PF_MM)

#define PF_MOE(BT) \
    template [[host_name("pf_moe_gu" #BT)]] kernel void pf_moe_gu<BT>(constant PfMoeArgs &, device const XPtr *, device const uint *, \
        device const int *, device const int2 *, device const uint *, device const float *, device float *, device const uchar *, uint2, ushort, ushort, ushort); \
    template [[host_name("pf_moe_down" #BT)]] kernel void pf_moe_down<BT>(constant PfMoeArgs &, device const XPtr *, device const uint *, \
        device const int *, device const int2 *, device const uint *, device const float *, device float *, uint2, ushort, ushort, ushort);
PF_MOE(16) PF_MOE(32)
