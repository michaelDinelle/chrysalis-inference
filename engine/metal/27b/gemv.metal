#include "kcommon.h"

constant int QT [[function_constant(0)]];

kernel void dequant(constant DeqA &a [[buffer(0)]], uint gid [[thread_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float tab[16];
    if (tid < 16) tab[tid] = IQ4_TAB[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong nu = (ulong)a.rows * a.cols / 32;
    if (gid >= nu) return;
    float v[32], s0, s1;
    if (QT == T_IQ4_XS) unit_load<T_IQ4_XS>(a.w, gid, tab, v, s0, s1);
    else if (QT == T_Q6_K) unit_load<T_Q6_K>(a.w, gid, tab, v, s0, s1);
    else unit_load<T_Q8_0>(a.w, gid, tab, v, s0, s1);
    device half4 *o = (device half4 *)(a.out + gid * 32);
    for (int i = 0; i < 8; i++) {
        const float s = i < 4 ? s0 : s1;
        o[i] = half4(float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]) * s);
    }
}

kernel void embed(constant EmbedA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float tab[16];
    if (tid < 16) tab[tid] = IQ4_TAB[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int t = tg.y, u = tg.x * 32 + tid;
    if (u >= D / 32) return;
    float v[32], s0, s1;
    unit_load<T_IQ4_XS>(a.w, (ulong)a.tokens[t] * (D / 32) + u, tab, v, s0, s1);
    device float4 *o = (device float4 *)(a.out + (ulong)t * D + u * 32);
    for (int i = 0; i < 8; i++) o[i] = float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]) * s0;
}

template <int Q> inline void wgroup(const WArgs w, ulong row, int nu, int G, int g, threadgroup const half2 *tab2,
                                    thread float4 *e, thread float4 *f) {
    const ulong u0 = row * nu + G * 4;
    if (Q == T_IQ4_XS || Q == T_Q6_K) {
        const uint4 qv = *(device const uint4 *)(w.q + (u0 / 4) * 64 + g * 16);
        const uint q[4] = {qv.x, qv.y, qv.z, qv.w};
        if (Q == T_IQ4_XS) {
            const float d = float(w.d[u0 / 8]);
            const char4 sc = iq4_ls4(w, u0);
            for (int ui = 0; ui < 4; ui++) {
                const float2 a0 = float2(tab2[q[ui] & 255]), a1 = float2(tab2[(q[ui] >> 8) & 255]);
                const float2 a2 = float2(tab2[(q[ui] >> 16) & 255]), a3 = float2(tab2[q[ui] >> 24]);
                const float s = d * float(sc[ui]);
                e[ui] = float4(a0.x, a1.x, a2.x, a3.x) * s;
                f[ui] = float4(a0.y, a1.y, a2.y, a3.y) * s;
            }
        } else {
            const uint2 hv = *(device const uint2 *)(w.qh + (u0 / 4) * 32 + g * 8);
            const float d = float(w.d[u0 / 8]);
            const char4 sa = *(device const char4 *)(w.sc + u0 * 2), sb = *(device const char4 *)(w.sc + u0 * 2 + 4);
            for (int ui = 0; ui < 4; ui++) {
                const uint h = ((ui < 2 ? hv.x : hv.y) >> (16 * (ui & 1))) & 0xffffu;
                const uint lo = q[ui] & 0x0f0f0f0fu, hi = (q[ui] >> 4) & 0x0f0f0f0fu;
                const uint hl = (h & 0x0303u) | ((h & 0x0c0cu) << 14), hh = ((h >> 4) & 0x0303u) | (((h >> 4) & 0x0c0cu) << 14);
                const float4 v0 = float4(as_type<uchar4>(lo | (hl << 4))) - 32.0f, v1 = float4(as_type<uchar4>(hi | (hh << 4))) - 32.0f;
                const char s0 = ui < 2 ? sa[2 * ui] : sb[2 * ui - 4], s1 = ui < 2 ? sa[2 * ui + 1] : sb[2 * ui - 3];
                e[ui] = v0 * (d * float(s0));
                f[ui] = v1 * (d * float(s1));
            }
        }
    } else {
        device const uint4 *p = (device const uint4 *)(w.q + (u0 / 4) * 128 + g * 32);
        const uint4 qa = p[0], qb = p[1];
        const uint q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        const uint2 dv = *(device const uint2 *)(w.d + u0);
        const half dd[4] = {as_type<half2>(dv.x)[0], as_type<half2>(dv.x)[1], as_type<half2>(dv.y)[0], as_type<half2>(dv.y)[1]};
        for (int ui = 0; ui < 4; ui++) {
            const float s = float(dd[ui]);
            e[ui] = float4(as_type<char4>(q[2 * ui])) * s;
            f[ui] = float4(as_type<char4>(q[2 * ui + 1])) * s;
        }
    }
}

template <int Q> inline void wgroup_h(const WArgs w, ulong row, int nu, int G, int g, threadgroup const half2 *tab2,
                                      thread half4 *e, thread half4 *f) {
    const ulong u0 = row * nu + G * 4;
    if (Q == T_IQ4_XS || Q == T_Q6_K) {
        const uint4 qv = *(device const uint4 *)(w.q + (u0 / 4) * 64 + g * 16);
        const uint q[4] = {qv.x, qv.y, qv.z, qv.w};
        if (Q == T_IQ4_XS) {
            const float d = float(w.d[u0 / 8]);
            const char4 sc = iq4_ls4(w, u0);
            for (int ui = 0; ui < 4; ui++) {
                const half2 a0 = tab2[q[ui] & 255], a1 = tab2[(q[ui] >> 8) & 255], a2 = tab2[(q[ui] >> 16) & 255], a3 = tab2[q[ui] >> 24];
                const float s = d * float(sc[ui]);
                e[ui] = half4(float4(half4(a0.x, a1.x, a2.x, a3.x)) * s);
                f[ui] = half4(float4(half4(a0.y, a1.y, a2.y, a3.y)) * s);
            }
        } else {
            const uint2 hv = *(device const uint2 *)(w.qh + (u0 / 4) * 32 + g * 8);
            const float d = float(w.d[u0 / 8]);
            const char4 sa = *(device const char4 *)(w.sc + u0 * 2), sb = *(device const char4 *)(w.sc + u0 * 2 + 4);
            for (int ui = 0; ui < 4; ui++) {
                const uint h = ((ui < 2 ? hv.x : hv.y) >> (16 * (ui & 1))) & 0xffffu;
                const uint lo = q[ui] & 0x0f0f0f0fu, hi = (q[ui] >> 4) & 0x0f0f0f0fu;
                const uint hl = (h & 0x0303u) | ((h & 0x0c0cu) << 14), hh = ((h >> 4) & 0x0303u) | (((h >> 4) & 0x0c0cu) << 14);
                const half4 v0 = half4(as_type<uchar4>(lo | (hl << 4))) - 32.0h, v1 = half4(as_type<uchar4>(hi | (hh << 4))) - 32.0h;
                const char s0 = ui < 2 ? sa[2 * ui] : sb[2 * ui - 4], s1 = ui < 2 ? sa[2 * ui + 1] : sb[2 * ui - 3];
                e[ui] = half4(float4(v0) * (d * float(s0)));
                f[ui] = half4(float4(v1) * (d * float(s1)));
            }
        }
    } else {
        device const uint4 *p = (device const uint4 *)(w.q + (u0 / 4) * 128 + g * 32);
        const uint4 qa = p[0], qb = p[1];
        const uint q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        const uint2 dv = *(device const uint2 *)(w.d + u0);
        const half dd[4] = {as_type<half2>(dv.x)[0], as_type<half2>(dv.x)[1], as_type<half2>(dv.y)[0], as_type<half2>(dv.y)[1]};
        for (int ui = 0; ui < 4; ui++) {
            e[ui] = half4(as_type<char4>(q[2 * ui])) * dd[ui];
            f[ui] = half4(as_type<char4>(q[2 * ui + 1])) * dd[ui];
        }
    }
}

inline void load_tab2(threadgroup half2 *tab2, ushort tid) {
    for (int i = tid; i < 256; i += 128) tab2[i] = half2(IQ4_TAB[i & 15], IQ4_TAB[i >> 4]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <int Q, int RT, int MODE>
kernel void gemv1(constant GemvA &a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NM = MODE == 2 ? 2 : 1;
    threadgroup half2 tab2[256];
    threadgroup float red[4][NM][RT * 8];
    threadgroup float rs[4];
    if (Q == T_IQ4_XS) load_tab2(tab2, tid);
    const int fm = (lane & 16) / 4 + (lane & 6) / 2, g = ((lane >> 3) & 1) * 2 + (lane & 1);
    const int row0 = tg * RT * 8, nu = a.cols / 32, ng = nu / 4;
    float inv = 1.0f;
    const bool norm = a.nw;
    if (norm) {
        float ss = 0;
        for (int i = tid; i < a.cols / 4; i += 128) { const float4 v = ((device const float4 *)a.x)[i]; ss += dot(v, v); }
        ss = simd_sum(ss);
        if (lane == 0) rs[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv = rsqrt((rs[0] + rs[1] + rs[2] + rs[3]) / a.cols + EPS);
    }
    const int G0 = sg * ng / 4, G1 = (sg + 1) * ng / 4;
    float acc[NM][RT];
    for (int k = 0; k < NM; k++)
        for (int r = 0; r < RT; r++) acc[k][r] = 0.0f;
    for (int G = G0; G < G1; G++) {
        float2 x[4][4];
        device const float *xp = a.x + G * 128 + 2 * g;
        for (int ui = 0; ui < 4; ui++)
            for (int kt = 0; kt < 4; kt++) x[ui][kt] = *(device const float2 *)(xp + ui * 32 + kt * 8);
        if (norm) {
            device const float *wp = a.nw + G * 128 + 2 * g;
            for (int ui = 0; ui < 4; ui++)
                for (int kt = 0; kt < 4; kt++) x[ui][kt] *= *(device const float2 *)(wp + ui * 32 + kt * 8) * inv;
        }
        for (int k = 0; k < NM; k++)
            for (int r = 0; r < RT; r++) {
                float4 e[4], f[4];
                wgroup<Q>(k ? a.w2 : a.w, row0 + r * 8 + fm, nu, G, g, tab2, e, f);
                float s = 0;
                for (int ui = 0; ui < 4; ui++)
                    s += dot(e[ui], float4(x[ui][0], x[ui][1])) + dot(f[ui], float4(x[ui][2], x[ui][3]));
                acc[k][r] += s;
            }
    }
    for (int k = 0; k < NM; k++)
        for (int r = 0; r < RT; r++) {
            float v = acc[k][r];
            v += simd_shuffle_xor(v, 1);
            v += simd_shuffle_xor(v, 8);
            if (g == 0) red[sg][k][r * 8 + fm] = v;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < RT * 8) {
        float s = red[0][0][tid] + red[1][0][tid] + red[2][0][tid] + red[3][0][tid];
        device float *y = a.y + row0 + tid;
        if (MODE == GEMV_ADD) *y += s;
        else if (MODE == 2) *y = silu(s) * (red[0][NM - 1][tid] + red[1][NM - 1][tid] + red[2][NM - 1][tid] + red[3][NM - 1][tid]);
        else *y = s;
    }
}
#define G1(Q, RT, M) template [[host_name("gemv1_" #Q "_" #RT "_" #M)]] kernel decltype(gemv1<Q, RT, M>) gemv1<Q, RT, M>;
#define G1M(Q, RT) G1(Q, RT, 0) G1(Q, RT, 1) G1(Q, RT, 2)
G1M(23, 1) G1M(23, 4) G1M(23, 8) G1M(14, 2) G1M(8, 2)

template <int Q, int RT, int GLU, int NTT>
kernel void gemv_mma(constant GemvA &a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half2 tab2[256];
    threadgroup float red[4][RT * (GLU + 1)][64 * NTT];
    if (Q == T_IQ4_XS) {
        for (int i = tid; i < 256; i += 128) tab2[i] = half2(IQ4_TAB[i & 15], IQ4_TAB[i >> 4]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int fm = (lane & 16) / 4 + (lane & 6) / 2, g = ((lane >> 3) & 1) * 2 + (lane & 1);
    const int row0 = tg * RT * 8, nu = a.cols / 32, ng = nu / 4;
    const int G0 = sg * ng / 4, G1 = (sg + 1) * ng / 4;
    simdgroup_float8x8 C[RT][GLU + 1][NTT];
    for (int r = 0; r < RT; r++)
        for (int k = 0; k <= GLU; k++)
            for (int n = 0; n < NTT; n++) C[r][k][n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (NTT == 1) {
        for (int G = G0; G < G1; G++) {
            simdgroup_half8x8 B[16];
            for (int i = 0; i < 16; i++) simdgroup_load(B[i], a.x16 + G * 128 + i * 8, a.ldx, ulong2(0, 0), true);
            for (int r = 0; r < RT; r++)
                for (int k = 0; k <= GLU; k++) {
                    half4 e[4], f[4];
                    wgroup_h<Q>(k ? a.w2 : a.w, row0 + r * 8 + fm, nu, G, g, tab2, e, f);
                    for (int ui = 0; ui < 4; ui++) {
                        simdgroup_half8x8 A;
                        A.thread_elements()[0] = e[ui].x; A.thread_elements()[1] = e[ui].y;
                        simdgroup_multiply_accumulate(C[r][k][0], A, B[ui * 4], C[r][k][0]);
                        A.thread_elements()[0] = e[ui].z; A.thread_elements()[1] = e[ui].w;
                        simdgroup_multiply_accumulate(C[r][k][0], A, B[ui * 4 + 1], C[r][k][0]);
                        A.thread_elements()[0] = f[ui].x; A.thread_elements()[1] = f[ui].y;
                        simdgroup_multiply_accumulate(C[r][k][0], A, B[ui * 4 + 2], C[r][k][0]);
                        A.thread_elements()[0] = f[ui].z; A.thread_elements()[1] = f[ui].w;
                        simdgroup_multiply_accumulate(C[r][k][0], A, B[ui * 4 + 3], C[r][k][0]);
                    }
                }
        }
    } else
    for (int G = G0; G < G1; G++)
        for (int k = 0; k <= GLU; k++) {
            half4 e[RT][4], f[RT][4];
            for (int r = 0; r < RT; r++) wgroup_h<Q>(k ? a.w2 : a.w, row0 + r * 8 + fm, nu, G, g, tab2, e[r], f[r]);
            for (int ui = 0; ui < 4; ui++) {
                simdgroup_half8x8 B[NTT][4];
                for (int n = 0; n < NTT; n++)
                    for (int kt = 0; kt < 4; kt++)
                        simdgroup_load(B[n][kt], a.x16 + (ulong)n * 8 * a.ldx + G * 128 + ui * 32 + kt * 8, a.ldx, ulong2(0, 0), true);
                for (int r = 0; r < RT; r++) {
                    simdgroup_half8x8 A[4];
                    A[0].thread_elements()[0] = e[r][ui].x; A[0].thread_elements()[1] = e[r][ui].y;
                    A[1].thread_elements()[0] = e[r][ui].z; A[1].thread_elements()[1] = e[r][ui].w;
                    A[2].thread_elements()[0] = f[r][ui].x; A[2].thread_elements()[1] = f[r][ui].y;
                    A[3].thread_elements()[0] = f[r][ui].z; A[3].thread_elements()[1] = f[r][ui].w;
                    for (int n = 0; n < NTT; n++)
                        for (int kt = 0; kt < 4; kt++) simdgroup_multiply_accumulate(C[r][k][n], A[kt], B[n][kt], C[r][k][n]);
                }
            }
        }
    for (int r = 0; r < RT; r++)
        for (int k = 0; k <= GLU; k++)
            for (int n = 0; n < NTT; n++) simdgroup_store(C[r][k][n], &red[sg][r * (GLU + 1) + k][n * 64], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < RT * 64 * NTT; i += 128) {
        const int r = i / (64 * NTT), e = i % (64 * NTT), row = (e % 64) / 8, t = (e / 64) * 8 + e % 8;
        if (t >= a.T) continue;
        float s = 0, s2 = 0;
        for (int q = 0; q < 4; q++) {
            s += red[q][r * (GLU + 1)][e];
            if (GLU) s2 += red[q][r * (GLU + 1) + 1][e];
        }
        const ulong o = (ulong)t * a.ldy + row0 + r * 8 + row;
        if (GLU) a.y16[o] = half(silu(s) * s2);
        else if (a.y16) a.y16[o] = half(s);
        else if (a.add) a.y[o] += s;
        else a.y[o] = s;
    }
}
#define GM(Q, RT, GLU, N) template [[host_name("gemv_mma_" #Q "_" #RT "_" #GLU "_" #N)]] kernel decltype(gemv_mma<Q, RT, GLU, N>) gemv_mma<Q, RT, GLU, N>;
#define GMN(N) GM(23, 4, 0, N) GM(23, 4, 1, N) GM(14, 4, 0, N) GM(8, 4, 0, N) GM(8, 4, 1, N)
GMN(1) GMN(2)

template <int Q, int NR, int KS, int P, int MODE>
kernel void gemv1k(constant GemvA &a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NM = MODE == 2 ? 2 : 1;
    threadgroup half2 tab2[256];
    threadgroup float red[KS][NM][P * 2 * NR];
    threadgroup float rs[KS * P];
    const int ks = sg % KS, pr = sg / KS;
    if (Q == T_IQ4_XS)
        for (int i = tid; i < 256; i += 32 * KS * P) tab2[i] = half2(IQ4_TAB[i & 15], IQ4_TAB[i >> 4]);
    const int h = lane >> 4, c = lane & 15;
    const int nu = a.cols / 32, steps = a.cols / 512;
    const int rowb = (tg * P + pr) * 2 * NR + h * NR;
    float inv = 1.0f;
    const bool norm = a.nw;
    if (norm) {
        float ss = 0;
        for (int i = tid; i < a.cols / 4; i += 32 * KS * P) { const float4 v = ((device const float4 *)a.x)[i]; ss += dot(v, v); }
        ss = simd_sum(ss);
        if (lane == 0) rs[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (norm) {
        float ss = 0;
        for (int k = 0; k < KS * P; k++) ss += rs[k];
        inv = rsqrt(ss / a.cols + EPS);
    }
    float acc[NM][NR];
    for (int k = 0; k < NM; k++)
        for (int r = 0; r < NR; r++) acc[k][r] = 0.0f;
    for (int st = ks; st < steps; st += KS) {
        const int ch = st * 16 + c, G = ch >> 2, g = ch & 3;
        uint4 qv[NM][NR];
        uint2 hv[NM][NR];
        char4 sc[NM][NR], sb[NM][NR];
        float d[NM][NR];
        for (int k = 0; k < NM; k++)
            for (int r = 0; r < NR; r++) {
                const WArgs w = k ? a.w2 : a.w;
                const ulong u0 = (ulong)(rowb + r) * nu + G * 4;
                qv[k][r] = *(device const uint4 *)(w.q + u0 * 16 + g * 16);
                if (Q == T_Q6_K) {
                    hv[k][r] = *(device const uint2 *)(w.qh + u0 * 8 + g * 8);
                    sc[k][r] = *(device const char4 *)(w.sc + u0 * 2);
                    sb[k][r] = *(device const char4 *)(w.sc + u0 * 2 + 4);
                } else {
                    sc[k][r] = iq4_ls4(w, u0);
                }
                d[k][r] = float(w.d[u0 / 8]);
            }
        float4 x[4][2];
        device const float *xp = a.x + G * 128 + 2 * g;
        for (int ui = 0; ui < 4; ui++) {
            x[ui][0] = float4(*(device const float2 *)(xp + ui * 32), *(device const float2 *)(xp + ui * 32 + 8));
            x[ui][1] = float4(*(device const float2 *)(xp + ui * 32 + 16), *(device const float2 *)(xp + ui * 32 + 24));
        }
        if (norm) {
            device const float *wp = a.nw + G * 128 + 2 * g;
            for (int ui = 0; ui < 4; ui++) {
                x[ui][0] *= float4(*(device const float2 *)(wp + ui * 32), *(device const float2 *)(wp + ui * 32 + 8)) * inv;
                x[ui][1] *= float4(*(device const float2 *)(wp + ui * 32 + 16), *(device const float2 *)(wp + ui * 32 + 24)) * inv;
            }
        }
        for (int k = 0; k < NM; k++)
            for (int r = 0; r < NR; r++) {
                const uint q[4] = {qv[k][r].x, qv[k][r].y, qv[k][r].z, qv[k][r].w};
                float s = 0;
                for (int ui = 0; ui < 4; ui++) {
                    if (Q == T_Q6_K) {
                        const uint hh = ((ui < 2 ? hv[k][r].x : hv[k][r].y) >> (16 * (ui & 1))) & 0xffffu;
                        const uint lo = q[ui] & 0x0f0f0f0fu, hi = (q[ui] >> 4) & 0x0f0f0f0fu;
                        const uint hl = (hh & 0x0303u) | ((hh & 0x0c0cu) << 14), hu = ((hh >> 4) & 0x0303u) | (((hh >> 4) & 0x0c0cu) << 14);
                        const float4 v0 = float4(as_type<uchar4>(lo | (hl << 4))) - 32.0f, v1 = float4(as_type<uchar4>(hi | (hu << 4))) - 32.0f;
                        const char s0 = ui < 2 ? sc[k][r][2 * ui] : sb[k][r][2 * ui - 4], s1 = ui < 2 ? sc[k][r][2 * ui + 1] : sb[k][r][2 * ui - 3];
                        s += dot(v0, x[ui][0]) * float(s0) + dot(v1, x[ui][1]) * float(s1);
                    } else {
                        const float2 a0 = float2(tab2[q[ui] & 255]), a1 = float2(tab2[(q[ui] >> 8) & 255]);
                        const float2 a2 = float2(tab2[(q[ui] >> 16) & 255]), a3 = float2(tab2[q[ui] >> 24]);
                        const float t = dot(float4(a0.x, a1.x, a2.x, a3.x), x[ui][0]) + dot(float4(a0.y, a1.y, a2.y, a3.y), x[ui][1]);
                        s += t * float(sc[k][r][ui]);
                    }
                }
                acc[k][r] += s * d[k][r];
            }
    }
    for (int k = 0; k < NM; k++)
        for (int r = 0; r < NR; r++) {
            float v = acc[k][r];
            v += simd_shuffle_xor(v, 1);
            v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 4);
            v += simd_shuffle_xor(v, 8);
            if (c == 0) red[ks][k][pr * 2 * NR + h * NR + r] = v;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < P * 2 * NR) {
        float s = 0, s2 = 0;
        for (int q = 0; q < KS; q++) { s += red[q][0][tid]; s2 += red[q][NM - 1][tid]; }
        device float *y = a.y + tg * P * 2 * NR + tid;
        if (MODE == GEMV_ADD) *y += s;
        else if (MODE == 2) *y = silu(s) * s2;
        else *y = s;
    }
}
#define GK1(Q, P, KS, M) template [[host_name("gemv1k_" #Q "_" #P "_" #KS "_" #M)]] kernel decltype(gemv1k<Q, 2, KS, P, M>) gemv1k<Q, 2, KS, P, M>;
#define GK1M(Q, P, KS) GK1(Q, P, KS, 0) GK1(Q, P, KS, 1) GK1(Q, P, KS, 2)
GK1M(23, 4, 2) GK1M(23, 1, 8) GK1M(14, 4, 2)
