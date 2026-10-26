#pragma once
#include <metal_stdlib>
using namespace metal;

#define T_F32 0
#define T_F16 1
#define T_Q4_0 2
#define T_Q5_0 6
#define T_Q8_0 8
#define T_Q2_K 10
#define T_Q3_K 11
#define T_Q4_K 12
#define T_Q5_K 13
#define T_Q6_K 14
#define T_IQ4_NL 20
#define T_IQ4_XS 23
#define T_BF16 30
#define T_Q2_0 42

constant float kv_iq4nl[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};

inline float rdh(device const uchar *p) { return float(as_type<half>(*(device const ushort *)p)); }
inline uint rd16(device const uchar *p) { return *(device const ushort *)p; }
inline uint rd32(device const uchar *p) { return rd16(p) | (rd16(p + 2) << 16); }

inline uint row_bytes(int F, uint cols) {
    switch (F) {
        case T_F32: return cols * 4;
        case T_F16: case T_BF16: return cols * 2;
        case T_Q4_0: case T_IQ4_NL: return cols / 32 * 18;
        case T_Q5_0: return cols / 32 * 22;
        case T_Q8_0: return cols / 32 * 34;
        case T_Q2_0: return cols / 64 * 18;
        case T_Q2_K: return cols / 256 * 84;
        case T_Q3_K: return cols / 256 * 110;
        case T_Q4_K: return cols / 256 * 144;
        case T_Q5_K: return cols / 256 * 176;
        case T_Q6_K: return cols / 256 * 210;
        case T_IQ4_XS: return cols / 256 * 136;
    }
    return 0;
}

inline void scale_min_k4(int j, device const uchar *q, thread float &sc, thread float &m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

template <int F> inline void deq32(device const uchar *row, uint u, thread float *w);

template <> inline void deq32<T_F32>(device const uchar *row, uint u, thread float *w) {
    device const float4 *p = (device const float4 *)(row + u * 128);
    for (int i = 0; i < 8; i++) { float4 v = p[i]; w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w; }
}
template <> inline void deq32<T_F16>(device const uchar *row, uint u, thread float *w) {
    device const half4 *p = (device const half4 *)(row + u * 64);
    for (int i = 0; i < 8; i++) { float4 v = float4(p[i]); w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w; }
}
template <> inline void deq32<T_BF16>(device const uchar *row, uint u, thread float *w) {
    device const uint4 *p = (device const uint4 *)(row + u * 64);
    for (int i = 0; i < 4; i++) {
        uint4 v = p[i];
        for (int k = 0; k < 4; k++) { w[8 * i + 2 * k] = as_type<float>(v[k] << 16); w[8 * i + 2 * k + 1] = as_type<float>(v[k] & 0xFFFF0000u); }
    }
}
template <> inline void deq32<T_Q2_0>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + (u / 2) * 18;
    const float d = rdh(b);
    device const uchar *q = b + 2 + (u % 2) * 8;
    for (int i = 0; i < 8; i++) {
        const uint x = q[i];
        w[4 * i] = (float(x & 3) - 1.f) * d; w[4 * i + 1] = (float((x >> 2) & 3) - 1.f) * d;
        w[4 * i + 2] = (float((x >> 4) & 3) - 1.f) * d; w[4 * i + 3] = (float(x >> 6) - 1.f) * d;
    }
}
template <> inline void deq32<T_Q4_0>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + u * 18;
    const float d = rdh(b);
    for (int i = 0; i < 16; i++) { const uint x = b[2 + i]; w[i] = (float(x & 15) - 8.f) * d; w[i + 16] = (float(x >> 4) - 8.f) * d; }
}
template <> inline void deq32<T_Q5_0>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + u * 22;
    const float d = rdh(b);
    const uint qh = rd32(b + 2);
    for (int i = 0; i < 16; i++) {
        const uint x = b[6 + i];
        w[i] = (float((x & 15) | (((qh >> i) & 1) << 4)) - 16.f) * d;
        w[i + 16] = (float((x >> 4) | (((qh >> (i + 16)) & 1) << 4)) - 16.f) * d;
    }
}
template <> inline void deq32<T_Q8_0>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + u * 34;
    const float d = rdh(b);
    device const char *q = (device const char *)(b + 2);
    for (int i = 0; i < 32; i++) w[i] = float(q[i]) * d;
}
template <> inline void deq32<T_IQ4_NL>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + u * 18;
    const float d = rdh(b);
    for (int i = 0; i < 16; i++) { const uint x = b[2 + i]; w[i] = kv_iq4nl[x & 15] * d; w[i + 16] = kv_iq4nl[x >> 4] * d; }
}
template <> inline void deq32<T_IQ4_XS>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + (u / 8) * 136;
    const uint ib = u % 8;
    const uint sh = rd16(b + 2);
    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = rdh(b) * float(ls - 32);
    device const uchar *q = b + 8 + 16 * ib;
    for (int i = 0; i < 16; i++) { const uint x = q[i]; w[i] = kv_iq4nl[x & 15] * dl; w[i + 16] = kv_iq4nl[x >> 4] * dl; }
}
template <> inline void deq32<T_Q4_K>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + (u / 8) * 144;
    const uint j = u % 8;
    float sc, m;
    scale_min_k4(j, b + 4, sc, m);
    const float d = rdh(b) * sc, mn = rdh(b + 2) * m;
    device const uchar *q = b + 16 + 32 * (j / 2);
    const uint sft = (j % 2) * 4;
    for (int i = 0; i < 32; i++) w[i] = d * float((q[i] >> sft) & 15) - mn;
}
template <> inline void deq32<T_Q5_K>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + (u / 8) * 176;
    const uint j = u % 8;
    float sc, m;
    scale_min_k4(j, b + 4, sc, m);
    const float d = rdh(b) * sc, mn = rdh(b + 2) * m;
    device const uchar *qh = b + 16, *q = b + 48 + 32 * (j / 2);
    const uint sft = (j % 2) * 4;
    for (int i = 0; i < 32; i++) w[i] = d * float(((q[i] >> sft) & 15) | (((qh[i] >> j) & 1) << 4)) - mn;
}
template <> inline void deq32<T_Q6_K>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + (u / 8) * 210;
    const uint j = u % 8, n = j / 4, qq = j % 4;
    device const uchar *ql = b + 64 * n + (qq % 2) * 32, *qh = b + 128 + 32 * n;
    device const char *sc = (device const char *)(b + 192 + 8 * n + 2 * qq);
    const float d = rdh(b + 208);
    const uint ls = (qq / 2) * 4, hs = qq * 2;
    const float d0 = d * float(sc[0]), d1 = d * float(sc[1]);
    for (int i = 0; i < 32; i++) {
        const int q = int(((ql[i] >> ls) & 15) | (((qh[i] >> hs) & 3) << 4)) - 32;
        w[i] = (i < 16 ? d0 : d1) * float(q);
    }
}
template <> inline void deq32<T_Q3_K>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + (u / 8) * 110;
    const uint j = u % 8, n = j / 4, jj = j % 4;
    device const uchar *hm = b, *q = b + 32 + 32 * n, *s = b + 96;
    const float d = rdh(b + 108);
    float dl[2];
    for (int k = 0; k < 2; k++) {
        const uint is = 8 * n + 2 * jj + k;
        const uint lo = is < 8 ? (s[is] & 15) : (s[is - 8] >> 4);
        const uint hi = (s[8 + is % 4] >> (2 * (is / 4))) & 3;
        dl[k] = d * (float(lo | (hi << 4)) - 32.f);
    }
    const uint sft = 2 * jj, hb = 4 * n + jj;
    for (int i = 0; i < 32; i++) {
        const int v = int((q[i] >> sft) & 3) - (((hm[i] >> hb) & 1) ? 0 : 4);
        w[i] = dl[i / 16] * float(v);
    }
}

template <> inline void deq32<T_Q2_K>(device const uchar *row, uint u, thread float *w) {
    device const uchar *b = row + (u / 8) * 84;
    const uint j = u % 8, sft = 2 * (j % 4);
    device const uchar *q = b + 16 + 32 * (j / 4);
    const float d = rdh(b + 80), dm = rdh(b + 82);
    for (int k = 0; k < 2; k++) {
        const uint sc = b[2 * j + k];
        const float dl = d * float(sc & 15), ml = dm * float(sc >> 4);
        for (int i = 0; i < 16; i++) w[16 * k + i] = dl * float((q[16 * k + i] >> sft) & 3) - ml;
    }
}

template <int F> inline float dot32(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *kvt) {
    float w[32];
    deq32<F>(row, u, w);
    float s = 0;
    for (int i = 0; i < 32; i++) s += w[i] * x[i];
    return s;
}
template <> inline float dot32<T_Q2_0>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *) {
    device const uchar *b = row + (u / 2) * 18;
    device const ushort *q = (device const ushort *)(b + 2 + (u % 2) * 8);
    float s = 0;
    for (int k = 0; k < 4; k++) {
        const uint w = q[k];
        thread const float *xk = x + 8 * k;
        s += float(w & 3) * xk[0] + float((w >> 2) & 3) * xk[1] + float((w >> 4) & 3) * xk[2] + float((w >> 6) & 3) * xk[3]
           + float((w >> 8) & 3) * xk[4] + float((w >> 10) & 3) * xk[5] + float((w >> 12) & 3) * xk[6] + float(w >> 14) * xk[7];
    }
    return rdh(b) * (s - sx.x - sx.y);
}
template <> inline float dot32<T_Q4_K>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *) {
    device const uchar *b = row + (u / 8) * 144;
    const uint j = u % 8, sft = (j % 2) * 4;
    float sc, m;
    scale_min_k4(j, b + 4, sc, m);
    device const uint *q = (device const uint *)(b + 16 + 32 * (j / 2));
    float s = 0;
    for (int k = 0; k < 8; k++) {
        const uint w = q[k] >> sft;
        s += float(w & 15) * x[4 * k] + float((w >> 8) & 15) * x[4 * k + 1] + float((w >> 16) & 15) * x[4 * k + 2] + float((w >> 24) & 15) * x[4 * k + 3];
    }
    return rdh(b) * sc * s - rdh(b + 2) * m * (sx.x + sx.y);
}
template <> inline float dot32<T_Q5_K>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *) {
    device const uchar *b = row + (u / 8) * 176;
    const uint j = u % 8, sft = (j % 2) * 4;
    float sc, m;
    scale_min_k4(j, b + 4, sc, m);
    device const uint *qh = (device const uint *)(b + 16), *q = (device const uint *)(b + 48 + 32 * (j / 2));
    float s = 0;
    for (int k = 0; k < 8; k++) {
        const uint w = q[k] >> sft, h = (qh[k] >> j) << 4;
        s += float((w & 15) | (h & 16)) * x[4 * k] + float(((w >> 8) & 15) | ((h >> 8) & 16)) * x[4 * k + 1]
           + float(((w >> 16) & 15) | ((h >> 16) & 16)) * x[4 * k + 2] + float(((w >> 24) & 15) | ((h >> 24) & 16)) * x[4 * k + 3];
    }
    return rdh(b) * sc * s - rdh(b + 2) * m * (sx.x + sx.y);
}
template <> inline float dot32<T_Q3_K>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *) {
    device const uchar *b = row + (u / 8) * 110;
    const uint j = u % 8, n = j / 4, jj = j % 4;
    device const ushort *hm = (device const ushort *)b, *q = (device const ushort *)(b + 32 + 32 * n);
    device const uchar *sb = b + 96;
    const float d = rdh(b + 108);
    float dl[2];
    for (int k = 0; k < 2; k++) {
        const uint is = 8 * n + 2 * jj + k;
        const uint lo = is < 8 ? (sb[is] & 15) : (sb[is - 8] >> 4);
        const uint hi = (sb[8 + is % 4] >> (2 * (is / 4))) & 3;
        dl[k] = d * (float(lo | (hi << 4)) - 32.f);
    }
    const uint sft = 2 * jj, hb = 4 * n + jj;
    float s0 = 0, s1 = 0;
    for (int k = 0; k < 8; k++) {
        const uint wq = q[k], wh = hm[k], wq1 = q[8 + k], wh1 = hm[8 + k];
        s0 += float(((wq >> sft) & 3) | (((wh >> hb) & 1) << 2)) * x[2 * k] + float(((wq >> (8 + sft)) & 3) | (((wh >> (8 + hb)) & 1) << 2)) * x[2 * k + 1];
        s1 += float(((wq1 >> sft) & 3) | (((wh1 >> hb) & 1) << 2)) * x[16 + 2 * k] + float(((wq1 >> (8 + sft)) & 3) | (((wh1 >> (8 + hb)) & 1) << 2)) * x[17 + 2 * k];
    }
    return dl[0] * (s0 - 4.f * sx.x) + dl[1] * (s1 - 4.f * sx.y);
}
template <> inline float dot32<T_Q6_K>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *) {
    device const uchar *b = row + (u / 8) * 210;
    const uint j = u % 8, n = j / 4, qq = j % 4;
    device const ushort *ql = (device const ushort *)(b + 64 * n + (qq % 2) * 32), *qh = (device const ushort *)(b + 128 + 32 * n);
    device const char *sc = (device const char *)(b + 192 + 8 * n + 2 * qq);
    const float d = rdh(b + 208);
    const uint ls = (qq / 2) * 4, hs = qq * 2;
    float s0 = 0, s1 = 0;
    for (int k = 0; k < 8; k++) {
        const uint wl = ql[k], wh = qh[k], wl1 = ql[8 + k], wh1 = qh[8 + k];
        s0 += float(((wl >> ls) & 15) | (((wh >> hs) & 3) << 4)) * x[2 * k] + float(((wl >> (8 + ls)) & 15) | (((wh >> (8 + hs)) & 3) << 4)) * x[2 * k + 1];
        s1 += float(((wl1 >> ls) & 15) | (((wh1 >> hs) & 3) << 4)) * x[16 + 2 * k] + float(((wl1 >> (8 + ls)) & 15) | (((wh1 >> (8 + hs)) & 3) << 4)) * x[17 + 2 * k];
    }
    return d * float(sc[0]) * (s0 - 32.f * sx.x) + d * float(sc[1]) * (s1 - 32.f * sx.y);
}
template <> inline float dot32<T_IQ4_XS>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *kvt) {
    device const uchar *b = row + (u / 8) * 136;
    const uint ib = u % 8, sh = rd16(b + 2);
    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    device const uint *q = (device const uint *)(b + 8 + 16 * ib);
    float s = 0;
    for (int k = 0; k < 4; k++) {
        const uint w = q[k];
        for (int c = 0; c < 4; c++) s += kvt[(w >> (8 * c)) & 15] * x[4 * k + c] + kvt[(w >> (8 * c + 4)) & 15] * x[16 + 4 * k + c];
    }
    return rdh(b) * float(ls - 32) * s;
}
template <> inline float dot32<T_IQ4_NL>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *kvt) {
    device const uchar *b = row + u * 18;
    device const ushort *q = (device const ushort *)(b + 2);
    float s = 0;
    for (int k = 0; k < 8; k++) {
        const uint w = q[k];
        s += kvt[w & 15] * x[2 * k] + kvt[(w >> 8) & 15] * x[2 * k + 1] + kvt[(w >> 4) & 15] * x[16 + 2 * k] + kvt[(w >> 12) & 15] * x[17 + 2 * k];
    }
    return rdh(b) * s;
}

template <> inline float dot32<T_Q2_K>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *) {
    device const uchar *b = row + (u / 8) * 84;
    const uint j = u % 8, sft = 2 * (j % 4);
    device const uint *q = (device const uint *)(b + 16 + 32 * (j / 4));
    float s0 = 0, s1 = 0;
    for (int k = 0; k < 4; k++) {
        const uint w0 = q[k] >> sft, w1 = q[4 + k] >> sft;
        s0 += float(w0 & 3) * x[4 * k] + float((w0 >> 8) & 3) * x[4 * k + 1] + float((w0 >> 16) & 3) * x[4 * k + 2] + float((w0 >> 24) & 3) * x[4 * k + 3];
        s1 += float(w1 & 3) * x[16 + 4 * k] + float((w1 >> 8) & 3) * x[17 + 4 * k] + float((w1 >> 16) & 3) * x[18 + 4 * k] + float((w1 >> 24) & 3) * x[19 + 4 * k];
    }
    const uint sc0 = b[2 * j], sc1 = b[2 * j + 1];
    const float d = rdh(b + 80), dm = rdh(b + 82);
    return d * (float(sc0 & 15) * s0 + float(sc1 & 15) * s1) - dm * (float(sc0 >> 4) * sx.x + float(sc1 >> 4) * sx.y);
}
template <> inline float dot32<T_Q8_0>(device const uchar *row, uint u, thread const float *x, float2 sx, threadgroup const float *) {
    device const uchar *b = row + u * 34;
    device const char2 *q = (device const char2 *)(b + 2);
    float s = 0;
    for (int k = 0; k < 16; k++) { const float2 v = float2(q[k]); s += v.x * x[2 * k] + v.y * x[2 * k + 1]; }
    return rdh(b) * s;
}

inline float bf16_round(float x) { uint u = as_type<uint>(x); u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u; return as_type<float>(u); }
inline void q8_0_round(thread float *x) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = max(amax, fabs(x[i]));
    const float d = amax / 127.f, id = d != 0 ? 1.f / d : 0.f, dh = float(half(d));
    for (int i = 0; i < 32; i++) x[i] = rint(x[i] * id) * dh;
}
inline void q8_K_round(thread float *x) {
    float amax = 0, mx = 0;
    for (int i = 0; i < 32; i++) { const float a = fabs(x[i]); if (a > amax) { amax = a; mx = x[i]; } }
    for (int o = 1; o < 8; o *= 2) {
        const float oa = simd_shuffle_xor(amax, o), om = simd_shuffle_xor(mx, o);
        if (oa > amax) { amax = oa; mx = om; }
    }
    if (amax == 0) { for (int i = 0; i < 32; i++) x[i] = 0; return; }
    const float iscale = -127.f / mx, d = 1.f / iscale;
    for (int i = 0; i < 32; i++) x[i] = min(127.f, rint(iscale * x[i])) * d;
}
template <int F> inline void ggml_round(thread float *x) {
    if (F == T_BF16) { for (int i = 0; i < 32; i++) x[i] = bf16_round(x[i]); }
    else if (F == T_Q2_K || F == T_Q3_K || F == T_Q4_K || F == T_Q5_K || F == T_Q6_K || F == T_IQ4_XS) q8_K_round(x);
    else if (F != T_F32 && F != T_F16) q8_0_round(x);
}
