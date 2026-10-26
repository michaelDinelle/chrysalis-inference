#include "kcommon.h"

inline float tg_sum(float v, threadgroup float *red, ushort tid) {
    v = simd_sum(v);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (int i = 0; i < 8; i++) s += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

kernel void rmsnorm(constant NormA &a [[buffer(0)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];
    device const float *x = a.x + (ulong)row * a.ldx;
    float ss = 0;
    for (int i = tid; i < a.n; i += 256) ss += x[i] * x[i];
    const float inv = rsqrt(tg_sum(ss, red, tid) / a.n + EPS);
    const ulong o = (ulong)row * a.ldo;
    for (int i = tid; i < a.n; i += 256) {
        const float v = x[i] * inv * a.w[i];
        if (a.of) a.of[o + i] = v;
        if (a.oh) a.oh[o + i] = half(v);
    }
}

kernel void swiglu_h(constant SwigluA &a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if ((int)i < a.n) a.g[i] = half(silu(float(a.g[i])) * float(a.u[i]));
}

kernel void copy16(constant CopyA &a [[buffer(0)]], uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    for (ulong j = i; j < a.n / 16; j += n) ((device uint4 *)a.dst)[j] = ((device const uint4 *)a.src)[j];
}
kernel void fill16(constant FillA &a [[buffer(0)]], uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    for (ulong j = i; j < a.n / 16; j += n) ((device uint4 *)a.dst)[j] = uint4(0);
}

constant constexpr int SLICE = 4096, TOPK = 64;
inline void bitonic_desc(threadgroup float *v, threadgroup int *ix, ushort tid) {
    for (int k = 2; k <= SLICE; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < SLICE; i += 512) {
                const int p = i ^ j;
                if (p > i) {
                    const bool desc = (i & k) == 0;
                    const bool sw = desc ? (v[i] < v[p] || (v[i] == v[p] && ix[i] > ix[p])) : (v[i] > v[p] || (v[i] == v[p] && ix[i] < ix[p]));
                    if (sw) { const float tv = v[i]; v[i] = v[p]; v[p] = tv; const int ti = ix[i]; ix[i] = ix[p]; ix[p] = ti; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
}
kernel void topk1(constant TopkA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]],
                  ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[SLICE];
    threadgroup int ix[SLICE];
    const int row = tg.y, base = tg.x * SLICE;
    for (int i = tid; i < SLICE; i += 512) {
        const int j = base + i;
        v[i] = j < a.n ? a.logits[(ulong)row * a.ld + j] : -INFINITY;
        ix[i] = j;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bitonic_desc(v, ix, tid);
    if (tid < TOPK) {
        const ulong o = ((ulong)row * ng.x + tg.x) * TOPK + tid;
        a.cv[o] = v[tid]; a.ci[o] = ix[tid];
    }
}
kernel void topk2(constant TopkA &a [[buffer(0)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[SLICE];
    threadgroup int ix[SLICE];
    for (int i = tid; i < SLICE; i += 512) {
        v[i] = i < a.ncand ? a.cv[(ulong)row * a.ncand + i] : -INFINITY;
        ix[i] = i < a.ncand ? a.ci[(ulong)row * a.ncand + i] : 0x7fffffff;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bitonic_desc(v, ix, tid);
    if (tid < TOPK) { a.ov[row * TOPK + tid] = v[tid]; a.oi[row * TOPK + tid] = ix[tid]; }
}

kernel void logit_stats(constant StatsA &a [[buffer(0)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    device const float *r = a.logits + (ulong)row * a.ld;
    float m = -INFINITY;
    for (int i = tid; i < a.n; i += 1024) m = max(m, r[i]);
    m = simd_max(m);
    if ((tid & 31) == 0) red[tid >> 5] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (int w = 1; w < 32; w++) m = max(m, red[w]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[5] = {0, 0, 0, 0, 0};
    for (int i = tid; i < a.n; i += 1024) {
        const float v = r[i];
        if (!(v > -INFINITY)) continue;
        const float d = v - m, e = exp(d);
        acc[0] += 1; acc[1] += v; acc[2] += v * v; acc[3] += e; acc[4] += d * e;
    }
    device float *o = a.out + (ulong)row * 6;
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
    if (tid == 0) o[0] = a.n;
}

kernel void mask_logits(constant MaskA &a [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    const int i = g.x;
    device float *l = a.logits + (ulong)g.y * a.n;
    device const uint *m = a.mask + (ulong)g.y * ((a.n + 31) / 32);
    if (i < a.n && !((m[i >> 5] >> (i & 31)) & 1u)) l[i] = -INFINITY;
}

kernel void argmax(constant ArgmaxA &a [[buffer(0)]], uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    device const float *r = a.logits + (ulong)row * a.ld;
    float best = -INFINITY; int idx = 0x7fffffff;
    for (int i = tid; i < a.n; i += 1024) if (r[i] > best) { best = r[i]; idx = i; }
    for (int m = 16; m > 0; m >>= 1) {
        const float ov = simd_shuffle_xor(best, m); const int oi = simd_shuffle_xor(idx, m);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if ((tid & 31) == 0) { bv[tid >> 5] = best; bi[tid >> 5] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (int w = 1; w < 32; w++) if (bv[w] > best || (bv[w] == best && bi[w] < idx)) { best = bv[w]; idx = bi[w]; }
        a.out[row] = idx;
    }
}

kernel void rmsnorm1(constant NormA &a [[buffer(0)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    device const float4 *x = (device const float4 *)a.x, *w = (device const float4 *)a.w;
    const int n4 = a.n / 4;
    float4 v0 = tid < n4 ? x[tid] : 0.0f, v1 = tid + 1024 < n4 ? x[tid + 1024] : 0.0f;
    const float s = simd_sum(dot(v0, v0) + dot(v1, v1));
    if (tid % 32 == 0) red[tid / 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0;
    for (int i = 0; i < 32; i++) ss += red[i];
    const float inv = rsqrt(ss / a.n + EPS);
    device float4 *o = (device float4 *)a.of;
    if (tid < n4) o[tid] = v0 * inv * w[tid];
    if (tid + 1024 < n4) o[tid + 1024] = v1 * inv * w[tid + 1024];
}
