#include "kcommon.h"

constant int GQT [[function_constant(0)]];
constant constexpr int BN = 64, BK = 64, LDS = BK + 8;

struct RawU { uint w[8]; float s0, s1; };
template <int QT> inline RawU raw_load(const WArgs w, ulong u) {
    RawU r;
    const int ui = u % 4;
    if (QT == T_IQ4_XS || QT == T_Q6_K) {
        device const uchar *grp = w.q + (u / 4) * 64 + ui * 4;
        for (int g = 0; g < 4; g++) r.w[g] = *(device const uint *)(grp + g * 16);
        if (QT == T_IQ4_XS) {
            r.s0 = r.s1 = float(w.d[u / 8]) * float(iq4_ls(w, u));
        } else {
            device const uchar *hg = w.qh + (u / 4) * 32 + ui * 2;
            for (int g = 0; g < 4; g++) r.w[4 + g] = *(device const ushort *)(hg + g * 8);
            const float d = float(w.d[u / 8]);
            r.s0 = d * float(w.sc[u * 2]);
            r.s1 = d * float(w.sc[u * 2 + 1]);
        }
    } else {
        device const uchar *grp = w.q + (u / 4) * 128 + ui * 8;
        for (int g = 0; g < 4; g++) { const uint2 q = *(device const uint2 *)(grp + g * 32); r.w[2 * g] = q.x; r.w[2 * g + 1] = q.y; }
        r.s0 = r.s1 = float(w.d[u]);
    }
    return r;
}
template <int QT> inline void raw_store(thread const RawU &r, threadgroup const half2 *tab2, threadgroup half *dst) {
    for (int g = 0; g < 4; g++) {
        half2 l0, l1, h0, h1;
        if (QT == T_Q8_0) {
            const float4 x = float4(as_type<char4>(r.w[2 * g])) * r.s0, y = float4(as_type<char4>(r.w[2 * g + 1])) * r.s0;
            l0 = half2(x.xy); l1 = half2(x.zw); h0 = half2(y.xy); h1 = half2(y.zw);
        } else {
            const uint q = r.w[g];
            float2 a0, a1, b0, b1;
            if (QT == T_IQ4_XS) {
                const half2 c0 = tab2[q & 255], c1 = tab2[(q >> 8) & 255], c2 = tab2[(q >> 16) & 255], c3 = tab2[q >> 24];
                a0 = float2(c0.x, c1.x); a1 = float2(c2.x, c3.x); b0 = float2(c0.y, c1.y); b1 = float2(c2.y, c3.y);
            } else {
                const uint h = r.w[4 + g];
                const uint hl = (h & 0x0303u) | ((h & 0x0c0cu) << 14), hh = ((h >> 4) & 0x0303u) | (((h >> 4) & 0x0c0cu) << 14);
                const float4 lo = float4(as_type<uchar4>((q & 0x0f0f0f0fu) | (hl << 4))) - 32.0f;
                const float4 hi = float4(as_type<uchar4>(((q >> 4) & 0x0f0f0f0fu) | (hh << 4))) - 32.0f;
                a0 = lo.xy; a1 = lo.zw; b0 = hi.xy; b1 = hi.zw;
            }
            l0 = half2(a0 * r.s0); l1 = half2(a1 * r.s0); h0 = half2(b0 * r.s1); h1 = half2(b1 * r.s1);
        }
        *(threadgroup half2 *)(dst + 2 * g) = l0;
        *(threadgroup half2 *)(dst + 8 + 2 * g) = l1;
        *(threadgroup half2 *)(dst + 16 + 2 * g) = h0;
        *(threadgroup half2 *)(dst + 24 + 2 * g) = h1;
    }
}

kernel void gemm_q(constant GemmA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int BT = 64;
    threadgroup half buf[(BN + BT) * LDS];
    threadgroup half2 tab2[256];
    threadgroup half *Ws = buf, *Xs = buf + BN * LDS;
    if (GQT == T_IQ4_XS)
        for (int i = tid; i < 256; i += 128) tab2[i] = half2(IQ4_TAB[i & 15], IQ4_TAB[i >> 4]);
    const int n0 = tg.x * BN, t0 = tg.y * BT;
    constexpr int TR = 4, NR = 4;
    const int st = (sg / 2) * 32, sn = (sg % 2) * 32;
    simdgroup_float8x8 acc[TR][NR];
    for (int i = 0; i < TR; i++)
        for (int j = 0; j < NR; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const int nu = a.cols / 32;
    const int wr = tid / 2, wu = tid % 2;
    const ulong ub = (ulong)(n0 + wr) * nu + wu;
    RawU rw = GQT == T_IQ4_XS ? raw_load<T_IQ4_XS>(a.w, ub) : GQT == T_Q6_K ? raw_load<T_Q6_K>(a.w, ub) : raw_load<T_Q8_0>(a.w, ub);
    uint4 xr[BT * BK / 8 / 128];
    for (int j = 0; j < BT * BK / 8 / 128; j++) {
        const int i = tid + j * 128, t = i / (BK / 8), c = (i % (BK / 8)) * 8;
        xr[j] = *(device const uint4 *)(a.x + (ulong)(t0 + t) * a.ldx + c);
    }
    for (int k0 = 0; k0 < a.cols; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup half *d = Ws + wr * LDS + wu * 32;
            if (GQT == T_IQ4_XS) raw_store<T_IQ4_XS>(rw, tab2, d);
            else if (GQT == T_Q6_K) raw_store<T_Q6_K>(rw, tab2, d);
            else raw_store<T_Q8_0>(rw, tab2, d);
        }
        for (int j = 0; j < BT * BK / 8 / 128; j++) {
            const int i = tid + j * 128, t = i / (BK / 8), c = (i % (BK / 8)) * 8;
            *(threadgroup uint4 *)(Xs + t * LDS + c) = xr[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < a.cols) {
            const ulong un = ub + (k0 + BK) / 32;
            rw = GQT == T_IQ4_XS ? raw_load<T_IQ4_XS>(a.w, un) : GQT == T_Q6_K ? raw_load<T_Q6_K>(a.w, un) : raw_load<T_Q8_0>(a.w, un);
            for (int j = 0; j < BT * BK / 8 / 128; j++) {
                const int i = tid + j * 128, t = i / (BK / 8), c = (i % (BK / 8)) * 8;
                xr[j] = *(device const uint4 *)(a.x + (ulong)(t0 + t) * a.ldx + k0 + BK + c);
            }
        }
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 A[TR], B[NR];
            for (int i = 0; i < TR; i++) simdgroup_load(A[i], Xs + (st + 8 * i) * LDS + kk, LDS);
            for (int j = 0; j < NR; j++) simdgroup_load(B[j], Ws + (sn + 8 * j) * LDS + kk, LDS, ulong2(0, 0), true);
            for (int i = 0; i < TR; i++)
                for (int j = 0; j < NR; j++) simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *Cs = (threadgroup float *)buf;
    for (int i = 0; i < TR; i++)
        for (int j = 0; j < NR; j++) simdgroup_store(acc[i][j], Cs + (st + 8 * i) * BN + sn + 8 * j, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < BT * BN; i += 128) {
        const int t = i / BN, n = i % BN;
        if (t0 + t >= a.T) continue;
        const ulong o = (ulong)(t0 + t) * a.ldy + n0 + n;
        if (a.y_f16) ((device half *)a.y)[o] = half(Cs[i]);
        else if (a.add) ((device float *)a.y)[o] += Cs[i];
        else ((device float *)a.y)[o] = Cs[i];
    }
}
