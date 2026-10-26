#include "quant.h"

constant uint D = 2560, FE = 640, XDN = 2560 * 640 / 32 * 18, RB_DN = 640 / 32 * 18;
constant uint HC = 4, HCD = 10240;
inline float sigm(float x) { return 1.f / (1.f + exp(-x)); }
inline float silu(float x) { return x / (1.f + exp(-x)); }

struct XPtr { device const uchar *p; };

struct MoeArgs {
    uint T;
    uint layer;
    uint phase;
    uint ggml;
};

template <int F>
inline void gu_rows(device const uchar *E, uint j0, device const float *xt, uint ggml, threadgroup const float *kvt, uint lane,
                    thread float *g, thread float *v) {
    const uint rb = row_bytes(F, D);
    device const uchar *wg = E + XDN + j0 * rb, *wu = E + XDN + FE * rb + j0 * rb;
    for (int k = 0; k < 4; k++) g[k] = v[k] = 0;
    for (uint u = lane; u < D / 32; u += 32) {
        float xs[32];
        float2 sx = 0;
        device const float4 *xp = (device const float4 *)(xt + u * 32);
        for (int i = 0; i < 8; i++) {
            const float4 q = xp[i];
            xs[4 * i] = q.x; xs[4 * i + 1] = q.y; xs[4 * i + 2] = q.z; xs[4 * i + 3] = q.w;
        }
        if (ggml) ggml_round<F>(xs);
        for (int i = 0; i < 32; i++) sx[i / 16] += xs[i];
        for (int k = 0; k < 4; k++) {
            g[k] += dot32<F>(wg + k * rb, u, xs, sx, kvt);
            v[k] += dot32<F>(wu + k * rb, u, xs, sx, kvt);
        }
    }
}

kernel void moe_gu(constant MoeArgs &a [[buffer(0)]], device const XPtr *table [[buffer(1)]], device const int *ids [[buffer(2)]],
                   device const float *x [[buffer(3)]], device float *h [[buffer(4)]], device uint *done [[buffer(5)]],
                   device const uchar *fmt [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float kvt[16];
    if (sg == 0 && lane < 16) kvt[lane] = kv_iq4nl[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = (tg * 4 + sg) * 4, p = r / FE, j0 = r % FE;
    if (p >= a.T * 10) return;
    const uint key = a.layer * 512 + ids[p];
    device const uchar *E = table[key].p;
    if (a.phase == 0) { if (j0 == 0 && lane == 0) done[p] = E != nullptr; if (!E) return; }
    else if (done[p]) return;
    device const float *xt = x + (p / 10) * D;
    float g[4], v[4];
    const uint f = fmt[key];
    if (f == T_IQ4_XS) gu_rows<T_IQ4_XS>(E, j0, xt, a.ggml, kvt, lane, g, v);
    else if (f == T_Q3_K) gu_rows<T_Q3_K>(E, j0, xt, a.ggml, kvt, lane, g, v);
    else gu_rows<T_Q2_K>(E, j0, xt, a.ggml, kvt, lane, g, v);
    for (int k = 0; k < 4; k++) {
        const float gs = simd_sum(g[k]), vs = simd_sum(v[k]);
        if (lane == 0) h[p * FE + j0 + k] = silu(gs) * vs;
    }
}

kernel void moe_down(constant MoeArgs &a [[buffer(0)]], device const XPtr *table [[buffer(1)]], device const int *ids [[buffer(2)]],
                     device const float *h [[buffer(3)]], device float *y [[buffer(4)]], device const uint *done [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float kvt[16];
    if (sg == 0 && lane < 16) kvt[lane] = kv_iq4nl[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = (tg * 4 + sg) * 4 + lane / 8, p = r / D, row = r % D, sl = lane % 8;
    float s = 0;
    const bool mine = p < a.T * 10 && (a.phase == 0 ? done[p] != 0 : done[p] == 0);
    if (mine) {
        device const uchar *wd = table[a.layer * 512 + ids[p]].p + row * RB_DN;
        device const float *hp = h + p * FE;
        for (uint u = sl; u < FE / 32; u += 8) {
            float xs[32];
            for (int i = 0; i < 32; i++) xs[i] = hp[u * 32 + i];
            if (a.ggml) q8_0_round(xs);
            s += dot32<T_IQ4_NL>(wd, u, xs, float2(0), kvt);
        }
    }
    s += simd_shuffle_xor(s, 4); s += simd_shuffle_xor(s, 2); s += simd_shuffle_xor(s, 1);
    if (sl == 0 && mine) y[p * D + row] = s;
}

kernel void moe_combine(constant MoeArgs &a [[buffer(0)]], device const float *y [[buffer(1)]], device const float *wts [[buffer(2)]],
                        device const float *sgate [[buffer(3)]], device const float *shared [[buffer(4)]], device const float *inj [[buffer(5)]],
                        device float *res [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.T * D) return;
    const uint t = i / D, d = i % D;
    float o = sigm(sgate[t]) * shared[i];
    for (int k = 0; k < 10; k++) o += wts[t * 10 + k] * y[(t * 10 + k) * D + d];
    for (uint c = 0; c < HC; c++) res[t * HCD + c * D + d] += o * 2.f * sigm(inj[t * HC + c] * 0.25f);
}

kernel void moe_combine_sum(constant uint &T [[buffer(0)]], device const float *Y [[buffer(1)]], device const float *sgate [[buffer(2)]],
                            device const float *shared [[buffer(3)]], device const float *inj [[buffer(4)]], device float *res [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= T * D) return;
    const uint t = i / D, d = i % D;
    const float o = Y[i] + sigm(sgate[t]) * shared[i];
    for (uint c = 0; c < HC; c++) res[t * HCD + c * D + d] += o * 2.f * sigm(inj[t * HC + c] * 0.25f);
}
