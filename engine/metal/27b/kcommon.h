#pragma once
#include <metal_stdlib>
using namespace metal;
#include "args.h"

constant constexpr int D = 5120, FF = 17408, NH = 24, NKV = 4, HD = 256, ROT = 64;
constant constexpr int GK = 16, GV = 48, GD = 128, QKV = 2 * GK * GD + GV * GD;
constant constexpr float EPS = 1e-6f, ROPE_BASE = 1e7f;
constant constexpr int PAGE = 64, IMG_BASE = 1 << 30;
constant constexpr ulong KV_V = PAGE * HD, KV_KS = 2 * KV_V, KV_VS = KV_KS + PAGE * (HD / 32) * 2;
constant constexpr ulong KV_HEAD = KV_VS + PAGE * (HD / 32) * 2, KV_LAYER = KV_HEAD * NKV, KV_PAGE = KV_LAYER * 17;
inline uint k_off(int slot, int dim) { return slot * HD + ((dim >> 1) & 3) * 64 + (dim >> 3) * 2 + (dim & 1); }
inline uint v_off(int slot, int dim) { return (((slot >> 3) * 8 + (dim & 7)) * 4 + ((slot >> 1) & 3)) * 64 + (dim >> 3) * 2 + (slot & 1); }
enum { T_F32 = 0, T_Q8_0 = 8, T_Q6_K = 14, T_IQ4_XS = 23 };

constant float IQ4_TAB[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float silu(float x) { return x / (1.0f + exp(-x)); }

inline int iq4_ls(const WArgs w, ulong u) {
    const uint lo = ((device const uchar *)w.sc)[u >> 1], hi = w.qh[u >> 2];
    return int(((lo >> (4 * (u & 1))) & 15) | (((hi >> (2 * (u & 3))) & 3) << 4)) - 32;
}
inline char4 iq4_ls4(const WArgs w, ulong u0) {
    const uint lo = *(device const ushort *)(w.sc + u0 / 2), hi = w.qh[u0 / 4];
    return char4(((int4(lo, lo >> 4, lo >> 8, lo >> 12) & 15) | ((int4(hi, hi >> 2, hi >> 4, hi >> 6) & 3) << 4)) - 32);
}
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

template <int QT> inline void unit_load(const WArgs w, ulong u, threadgroup const float *tab, thread float *v,
                                        thread float &s0, thread float &s1) {
    const int ui = u % 4;
    if (QT == T_IQ4_XS || QT == T_Q6_K) {
        device const uchar *grp = w.q + (u / 4) * 64 + ui * 4;
        uint hw[4] = {0, 0, 0, 0};
        if (QT == T_Q6_K) {
            device const uchar *hg = w.qh + (u / 4) * 32 + ui * 2;
            for (int g = 0; g < 4; g++) hw[g] = *(device const ushort *)(hg + g * 8);
        }
        for (int g = 0; g < 4; g++) {
            const uint q = *(device const uint *)(grp + g * 16);
            for (int b = 0; b < 4; b++) {
                const int j = (b >> 1) * 8 + 2 * g + (b & 1);
                const uint byte = (q >> (8 * b)) & 0xffu;
                if (QT == T_IQ4_XS) {
                    v[j] = tab[byte & 15];
                    v[j + 16] = tab[byte >> 4];
                } else {
                    const uint hb = (hw[g] >> (8 * (b & 1))) & 0xffu, sh = 2 * (j / 8);
                    v[j] = float(int((byte & 15) | (((hb >> sh) & 3) << 4)) - 32);
                    v[j + 16] = float(int((byte >> 4) | (((hb >> (sh + 4)) & 3) << 4)) - 32);
                }
            }
        }
        if (QT == T_IQ4_XS) {
            s0 = s1 = float(w.d[u / 8]) * float(iq4_ls(w, u));
        } else {
            const float d = float(w.d[u / 8]);
            s0 = d * float(w.sc[u * 2]);
            s1 = d * float(w.sc[u * 2 + 1]);
        }
    } else {
        device const uchar *grp = w.q + (u / 4) * 128 + ui * 8;
        for (int g = 0; g < 4; g++) {
            const uint2 q = *(device const uint2 *)(grp + g * 32);
            for (int b = 0; b < 8; b++) {
                const int j = (b >> 1) * 8 + 2 * g + (b & 1);
                v[j] = float(char(((b < 4 ? q.x : q.y) >> (8 * (b & 3))) & 0xffu));
            }
        }
        s0 = s1 = float(w.d[u]);
    }
}

template <int QT> inline void unit_store_h(const WArgs w, ulong u, threadgroup const half2 *tab2, threadgroup half *dst) {
    const int ui = u % 4;
    if (QT == T_IQ4_XS || QT == T_Q6_K) {
        device const uchar *grp = w.q + (u / 4) * 64 + ui * 4;
        float s0, s1;
        if (QT == T_IQ4_XS) {
            s0 = s1 = float(w.d[u / 8]) * float(iq4_ls(w, u));
        } else {
            const float d = float(w.d[u / 8]);
            s0 = d * float(w.sc[u * 2]);
            s1 = d * float(w.sc[u * 2 + 1]);
        }
        device const uchar *hg = w.qh + (u / 4) * 32 + ui * 2;
        for (int g = 0; g < 4; g++) {
            const uint q = *(device const uint *)(grp + g * 16);
            half2 l0, l1, h0, h1;
            if (QT == T_IQ4_XS) {
                const half2 a0 = tab2[q & 255], a1 = tab2[(q >> 8) & 255], a2 = tab2[(q >> 16) & 255], a3 = tab2[q >> 24];
                l0 = half2(a0.x, a1.x); l1 = half2(a2.x, a3.x); h0 = half2(a0.y, a1.y); h1 = half2(a2.y, a3.y);
            } else {
                const uint h = *(device const ushort *)(hg + g * 8);
                const uint hl = (h & 0x0303u) | ((h & 0x0c0cu) << 14), hh = ((h >> 4) & 0x0303u) | (((h >> 4) & 0x0c0cu) << 14);
                const half4 lo = half4(as_type<uchar4>((q & 0x0f0f0f0fu) | (hl << 4))) - 32.0h;
                const half4 hi = half4(as_type<uchar4>(((q >> 4) & 0x0f0f0f0fu) | (hh << 4))) - 32.0h;
                l0 = lo.xy; l1 = lo.zw; h0 = hi.xy; h1 = hi.zw;
            }
            *(threadgroup half2 *)(dst + 2 * g) = half2(float2(l0) * s0);
            *(threadgroup half2 *)(dst + 8 + 2 * g) = half2(float2(l1) * s0);
            *(threadgroup half2 *)(dst + 16 + 2 * g) = half2(float2(h0) * s1);
            *(threadgroup half2 *)(dst + 24 + 2 * g) = half2(float2(h1) * s1);
        }
    } else {
        device const uchar *grp = w.q + (u / 4) * 128 + ui * 8;
        const half s = w.d[u];
        for (int g = 0; g < 4; g++) {
            const uint2 q = *(device const uint2 *)(grp + g * 32);
            const half4 a = half4(as_type<char4>(q.x)) * s, b = half4(as_type<char4>(q.y)) * s;
            *(threadgroup half2 *)(dst + 2 * g) = a.xy;
            *(threadgroup half2 *)(dst + 8 + 2 * g) = a.zw;
            *(threadgroup half2 *)(dst + 16 + 2 * g) = b.xy;
            *(threadgroup half2 *)(dst + 24 + 2 * g) = b.zw;
        }
    }
}
