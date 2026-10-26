#include "quant.h"

constant float EPS = 1e-6f;
constant int D = 2560, HC = 4, HCD = 10240, HCR = 320;

inline float sigm(float x) { return 1.f / (1.f + exp(-x)); }
inline float silu(float x) { return x / (1.f + exp(-x)); }

inline float tg_sum256(float v, threadgroup float *red, uint tid) {
    v = simd_sum(v);
    if (tid % 32 == 0) red[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (int i = 0; i < 8; i++) s += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

template <int F>
kernel void embed(device const int *tok [[buffer(0)]], device const uchar *W [[buffer(1)]], constant uint &rb [[buffer(2)]],
                  device float *res [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint u [[thread_index_in_threadgroup]]) {
    float w[32];
    deq32<F>(W + (size_t)tok[t] * rb, u, w);
    for (int c = 0; c < HC; c++)
        for (int i = 0; i < 32; i++) res[(size_t)t * HCD + c * D + u * 32 + i] = w[i];
}
template [[host_name("embed_q8_0")]] kernel void embed<T_Q8_0>(device const int *, device const uchar *, constant uint &, device float *, uint, uint);

kernel void hc_norm(device const float *res [[buffer(0)]], device const float *gamma [[buffer(1)]], device float *xn [[buffer(2)]],
                    device half *xh [[buffer(3)]], constant uint &mode [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];
    const size_t b = (size_t)g.y * HCD + g.x * D;
    float ss = 0;
    for (int i = tid; i < D; i += 256) ss += res[b + i] * res[b + i];
    const float inv = rsqrt(tg_sum256(ss, red, tid) / D + EPS);
    for (int i = tid; i < D; i += 256) {
        const float v = res[b + i] * inv * gamma[g.x * D + i];
        if (mode & 1) xn[b + i] = v;
        if (mode & 2) xh[b + i] = half(v);
    }
}

struct HcArgs { uint T, inject, ggml; };
template <int F, int R>
kernel void hc_pre(constant HcArgs &a [[buffer(0)]], device const float *res [[buffer(1)]], device const float *gamma [[buffer(2)]],
                   device const uchar *Wd [[buffer(3)]], device const uchar *Wi [[buffer(4)]], device float *lo [[buffer(5)]],
                   device float *inj [[buffer(6)]], device float *inv_out [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float inv[8][HC];
    threadgroup float part[4][R * 8];
    const uint nrows = HCR + (a.inject ? HC : 0), r0 = tg * R;
    for (uint t = 0; t < a.T; t++) {
        device const float4 *p = (device const float4 *)(res + (size_t)t * HCD + sg * D);
        float ss = 0;
        for (uint i = lane; i < D / 4; i += 32) { const float4 v = p[i]; ss += dot(v, v); }
        ss = simd_sum(ss);
        if (lane == 0) inv[t][sg] = rsqrt(ss / D + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tg == 0 && tid < a.T * HC) inv_out[tid] = inv[tid / HC][tid % HC];
    device const uchar *w[R];
    for (int k = 0; k < R; k++) {
        const uint r = min(r0 + k, nrows - 1);
        w[k] = (r < (uint)HCR ? Wd + (size_t)r * row_bytes(F, HCD) : Wi + (size_t)(r - HCR) * row_bytes(F, HCD));
    }
    for (uint t = 0; t < a.T; t++) {
        float s[R];
        for (int k = 0; k < R; k++) s[k] = 0;
        for (uint u = tid; u < HCD / 32; u += 128) {
            const float iv = inv[t][u / 80];
            float x[32];
            device const float4 *xp = (device const float4 *)(res + (size_t)t * HCD + u * 32);
            device const float4 *gp = (device const float4 *)(gamma + u * 32);
            for (int i = 0; i < 8; i++) { const float4 v = xp[i] * gp[i] * iv; x[4 * i] = v.x; x[4 * i + 1] = v.y; x[4 * i + 2] = v.z; x[4 * i + 3] = v.w; }
            if (a.ggml) ggml_round<F>(x);
            float2 sx = 0;
            for (int i = 0; i < 32; i++) sx[i / 16] += x[i];
            for (int k = 0; k < R; k++) s[k] += dot32<F>(w[k], u, x, sx, nullptr);
        }
        for (int k = 0; k < R; k++) {
            const float v = simd_sum(s[k]);
            if (lane == 0) part[sg][k * 8 + t] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R * a.T) {
        const uint k = tid / a.T, t = tid % a.T, r = r0 + k;
        const float v = part[0][k * 8 + t] + part[1][k * 8 + t] + part[2][k * 8 + t] + part[3][k * 8 + t];
        if (r < (uint)HCR) lo[t * HCR + r] = silu(v * 0.25f);
        else if (r < nrows) inj[t * HC + r - HCR] = v;
    }
}
template [[host_name("hc_pre")]] kernel void hc_pre<T_Q8_0, 2>(constant HcArgs &, device const float *, device const float *, device const uchar *,
    device const uchar *, device float *, device float *, device float *, uint, uint, uint, uint);
template [[host_name("hc_pre1")]] kernel void hc_pre<T_Q8_0, 1>(constant HcArgs &, device const float *, device const float *, device const uchar *,
    device const uchar *, device float *, device float *, device float *, uint, uint, uint, uint);
template [[host_name("hc_pre4")]] kernel void hc_pre<T_Q8_0, 4>(constant HcArgs &, device const float *, device const float *, device const uchar *,
    device const uchar *, device float *, device float *, device float *, uint, uint, uint, uint);

template <int F>
kernel void hc_post(constant HcArgs &a [[buffer(0)]], device const float *res [[buffer(1)]], device const float *gamma [[buffer(2)]],
                    device const float *inv [[buffer(3)]], device const float *lo [[buffer(4)]], device const uchar *Wu [[buffer(5)]],
                    device float *mx [[buffer(6)]], device half *mh [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint d = tg * 4 + sg, c = lane / 8, sl = lane % 8;
    device const uchar *w = Wu + (size_t)(c * D + d) * row_bytes(F, HCR);
    for (uint t = 0; t < a.T; t++) {
        float s = 0;
        for (uint u = sl; u < HCR / 32; u += 8) {
            float x[32];
            device const float *lp = lo + t * HCR + u * 32;
            for (int i = 0; i < 32; i++) x[i] = lp[i];
            if (a.ggml) ggml_round<F>(x);
            float2 sx = 0;
            for (int i = 0; i < 32; i++) sx[i / 16] += x[i];
            s += dot32<F>(w, u, x, sx, nullptr);
        }
        s += simd_shuffle_xor(s, 4); s += simd_shuffle_xor(s, 2); s += simd_shuffle_xor(s, 1);
        const size_t k = (size_t)t * HCD + c * D + d;
        float v = res[k] * inv[t * HC + c] * gamma[c * D + d] * sigm(s);
        v += simd_shuffle_xor(v, 8); v += simd_shuffle_xor(v, 16);
        if (lane == 0) { mx[t * D + d] = v * 0.25f; mh[t * D + d] = half(v * 0.25f); }
    }
}
template [[host_name("hc_post")]] kernel void hc_post<T_Q8_0>(constant HcArgs &, device const float *, device const float *, device const float *,
    device const float *, device const uchar *, device float *, device half *, uint, uint, uint);

kernel void hc_lo(device float *lo [[buffer(0)]], device half *loh [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float v = silu(lo[i] * 0.25f);
    lo[i] = v;
    loh[i] = half(v);
}

kernel void hc_mix(device const float *xn [[buffer(0)]], device const float *gate [[buffer(1)]], device float *mx [[buffer(2)]],
                   device half *mh [[buffer(3)]], constant uint &T [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= T * D) return;
    const uint t = i / D, d = i % D;
    float s = 0;
    for (int c = 0; c < HC; c++) { const size_t k = (size_t)t * HCD + c * D + d; s += xn[k] * sigm(gate[k]); }
    mx[i] = s * 0.25f;
    mh[i] = half(s * 0.25f);
}

kernel void hc_combine(device float *res [[buffer(0)]], device const float *inj [[buffer(1)]], device const float *out [[buffer(2)]],
                       constant uint &T [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= T * HCD) return;
    const uint t = i / HCD, c = (i / D) % HC, d = i % D;
    res[i] += out[t * D + d] * 2.f * sigm(inj[t * HC + c] * 0.25f);
}

kernel void ple_embed(device const uchar *rows [[buffer(0)]], device float *emb [[buffer(1)]], device half *eh [[buffer(2)]],
                      constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float w[32];
    deq32<T_Q8_0>(rows + (size_t)(i / 5) * row_bytes(T_Q8_0, 160), i % 5, w);
    const size_t o = (size_t)(i / 80) * D + (i % 80) * 32;
    for (int k = 0; k < 32; k++) { emb[o + k] = w[k]; eh[o + k] = half(w[k]); }
}

kernel void ple_gate(device const float *key [[buffer(0)]], device const float *value [[buffer(1)]], device float *res [[buffer(2)]],
                     device const float *nk [[buffer(3)]], device const float *nq [[buffer(4)]], device const float *nc [[buffer(5)]],
                     device float *xconv [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];
    const uint s = g.x, t = g.y;
    const size_t b = (size_t)t * HCD + s * D;
    float kk = 0, rr = 0;
    for (int i = tid; i < D; i += 256) { kk += key[b + i] * key[b + i]; rr += res[b + i] * res[b + i]; }
    const float ik = rsqrt(tg_sum256(kk, red, tid) / D + EPS), ir = rsqrt(tg_sum256(rr, red, tid) / D + EPS);
    float dot = 0, vv = 0;
    for (int i = tid; i < D; i += 256) {
        dot += key[b + i] * ik * nk[s * D + i] * res[b + i] * ir * nq[s * D + i];
        const float v = value[(size_t)t * D + i];
        vv += v * v;
    }
    const float sc = tg_sum256(dot, red, tid) * rsqrt((float)D);
    const float gg = sigm(copysign(sqrt(max(fabs(sc), 1e-6f)), sc));
    const float ig = rsqrt(gg * gg * tg_sum256(vv, red, tid) / D + EPS);
    for (int i = tid; i < D; i += 256) {
        const float gv = value[(size_t)t * D + i] * gg;
        xconv[b + i] = gv * ig * nc[s * D + i];
        res[b + i] += gv;
    }
}

kernel void ple_conv(device const float *xconv [[buffer(0)]], device const float *state [[buffer(1)]], device const float *w [[buffer(2)]],
                     device float *res [[buffer(3)]], constant uint &T [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= T * HCD) return;
    const int t = i / HCD, c = i % HCD;
    float acc = 0;
    for (int k = 0; k < 4; k++) {
        const int src = t - (3 - k) * 3;
        const float x = src >= 0 ? xconv[(size_t)src * HCD + c] : state[(size_t)(9 + src) * HCD + c];
        acc += w[c * 4 + k] * x;
    }
    res[i] += silu(acc);
}

kernel void ple_commit(device float *state [[buffer(0)]], device const float *xconv [[buffer(1)]], constant uint &n [[buffer(2)]],
                       uint c [[thread_position_in_grid]]) {
    if (c >= HCD) return;
    float h[9];
    for (int j = 0; j < 9; j++) {
        const int src = (int)n - 9 + j;
        h[j] = src >= 0 ? xconv[(size_t)src * HCD + c] : state[(size_t)(9 + src) * HCD + c];
    }
    for (int j = 0; j < 9; j++) state[(size_t)j * HCD + c] = h[j];
}

kernel void route_topk(device const float *logits [[buffer(0)]], device int *ids [[buffer(1)]], device float *wts [[buffer(2)]],
                       uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float v[16];
    for (int i = 0; i < 16; i++) v[i] = logits[t * 512 + i * 32 + lane];
    float topv[10];
    int topi[10];
    for (int k = 0; k < 10; k++) {
        float m = v[0]; int mi = lane;
        for (int i = 1; i < 16; i++) if (v[i] > m) { m = v[i]; mi = i * 32 + lane; }
        for (int o = 16; o > 0; o /= 2) {
            const float om = simd_shuffle_xor(m, o); const int oi = simd_shuffle_xor(mi, o);
            if (om > m || (om == m && oi < mi)) { m = om; mi = oi; }
        }
        topv[k] = m; topi[k] = mi;
        if ((uint)(mi % 32) == lane) v[mi / 32] = -INFINITY;
    }
    if (lane == 0) {
        float s = 0;
        for (int k = 0; k < 10; k++) s += exp(topv[k] - topv[0]);
        for (int k = 0; k < 10; k++) { ids[t * 10 + k] = topi[k]; wts[t * 10 + k] = exp(topv[k] - topv[0]) / s; }
    }
}

kernel void swiglu(device float *a [[buffer(0)]], device const float *b [[buffer(1)]], device half *ah [[buffer(2)]],
                   constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float v = silu(a[i]) * b[i];
    a[i] = v;
    ah[i] = half(v);
}

kernel void topk64(device const float *v [[buffer(0)]], device const int *vi [[buffer(1)]], constant uint2 &a [[buffer(2)]],
                   device float *ov [[buffer(3)]], device int *oi [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                   uint2 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int K = 64, CH = 2048;
    threadgroup float sv[CH];
    threadgroup int si[CH];
    threadgroup float bv[8];
    threadgroup int bi[8], bp[8];
    const uint n = a.x, use_ids = a.y, row = g.y, c0 = g.x * CH;
    for (uint i = tid; i < CH; i += 256) {
        const uint k = c0 + i;
        sv[i] = k < n ? v[(size_t)row * n + k] : -INFINITY;
        si[i] = k < n ? (use_ids ? vi[(size_t)row * n + k] : (int)k) : 0x7fffffff;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < K; r++) {
        float m = -INFINITY; int mi = 0x7fffffff, mp = 0;
        for (uint i = tid; i < CH; i += 256) if (sv[i] > m || (sv[i] == m && si[i] < mi)) { m = sv[i]; mi = si[i]; mp = i; }
        for (int o = 16; o > 0; o /= 2) {
            const float om = simd_shuffle_xor(m, o); const int oi2 = simd_shuffle_xor(mi, o), op = simd_shuffle_xor(mp, o);
            if (om > m || (om == m && oi2 < mi)) { m = om; mi = oi2; mp = op; }
        }
        if (tid % 32 == 0) { bv[tid / 32] = m; bi[tid / 32] = mi; bp[tid / 32] = mp; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float bm = bv[0]; int bmi = bi[0], bmp = bp[0];
            for (int i = 1; i < 8; i++) if (bv[i] > bm || (bv[i] == bm && bi[i] < bmi)) { bm = bv[i]; bmi = bi[i]; bmp = bp[i]; }
            const size_t o = ((size_t)row * ng.x + g.x) * K + r;
            ov[o] = bm; oi[o] = bmi;
            sv[bmp] = -INFINITY; si[bmp] = 0x7fffffff;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void logit_stats(device const float *logits [[buffer(0)]], device float *out [[buffer(1)]], constant uint &n [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    device const float *r = logits + (size_t)row * n;
    float m = -INFINITY;
    for (uint i = tid; i < n; i += 1024) {
        const float v = r[i];
        if ((as_type<uint>(v) & 0x7f800000u) != 0x7f800000u) m = max(m, v);
    }
    m = simd_max(m);
    if ((tid & 31) == 0) red[tid >> 5] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (int w = 1; w < 32; w++) m = max(m, red[w]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[5] = {0, 0, 0, 0, 0};
    for (uint i = tid; i < n; i += 1024) {
        const float v = r[i];
        if ((as_type<uint>(v) & 0x7f800000u) == 0x7f800000u) continue;
        const float d = v - m, e = exp(d);
        acc[0] += 1; acc[1] += v; acc[2] += v * v; acc[3] += e; acc[4] += d * e;
    }
    device float *o = out + (size_t)row * 6;
    for (int k = 0; k < 5; k++) {
        const float s = simd_sum(acc[k]);
        if ((tid & 31) == 0) red[tid >> 5] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float t = 0;
            for (int w = 0; w < 32; w++) t += red[w];
            o[1 + k] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) o[0] = n;
}

kernel void mask_logits(device float *logits [[buffer(0)]], device const uint *mask [[buffer(1)]], constant uint &n [[buffer(2)]],
                        uint2 g [[thread_position_in_grid]]) {
    if (g.x >= n) return;
    if (!((mask[g.y * (n / 32) + g.x / 32] >> (g.x % 32)) & 1)) logits[(size_t)g.y * n + g.x] = -INFINITY;
}

kernel void copy_f32(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}
kernel void zero_u32(device uint *dst [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = 0;
}

kernel void spec_check(device const uint *done [[buffer(0)]], constant uint &P [[buffer(1)]], device const uint4 *desc [[buffer(2)]],
                       constant uint &layer [[buffer(3)]], device const uint *grids [[buffer(4)]], device uint *args [[buffer(5)]],
                       uint tid [[thread_index_in_threadgroup]]) {
    bool all = true;
    for (uint p = 0; p < P; p++) all = all && done[p] != 0;
    const uint4 d = desc[layer];
    for (uint i = d.x + tid; i < d.y; i += 256) for (int k = 0; k < 3; k++) args[3 * i + k] = all ? grids[3 * i + k] : 0;
    for (uint i = d.z + tid; i < d.w; i += 256) for (int k = 0; k < 3; k++) args[3 * i + k] = all ? 0 : grids[3 * i + k];
}

kernel void keep_warm(device float *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float x = tid;
    for (int i = 0; i < 2000; i++) x = x * 1.0001f + 0.5f;
    if (x == 12345.f) out[0] = x;
}
