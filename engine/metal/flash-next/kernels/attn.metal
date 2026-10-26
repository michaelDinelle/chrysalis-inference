#include <metal_stdlib>
using namespace metal;

constant int NH = 24, NKV = 2, HD = 256, QPK = 12, IDX_H = 4, IDX_D = 128, PAGE = 64, KPOOL = 4, SEL = 512;
constant size_t KV_K = (size_t)NKV * PAGE * HD * 2, KV_LAYER = 2 * KV_K + (PAGE / KPOOL) * IDX_D * 2, KV_PAGE = KV_LAYER * 12;
constant float EPS = 1e-6f, ROPE_BASE = 1e7f;
inline float sigm(float x) { return 1.f / (1.f + exp(-x)); }

struct KVArgs {
    uint T;
    uint pos0;
    uint layer;
    uint nsplit;
};

inline device half *kv_k(device uchar *kv, device const int *pages, uint layer, uint head, uint cell) {
    return (device half *)(kv + (size_t)pages[cell / PAGE] * KV_PAGE + layer * KV_LAYER + ((size_t)head * PAGE + cell % PAGE) * HD * 2);
}
inline device half *kv_pool(device uchar *kv, device const int *pages, uint layer, uint b) {
    return (device half *)(kv + (size_t)pages[b * KPOOL / PAGE] * KV_PAGE + layer * KV_LAYER + 2 * KV_K + (b % (PAGE / KPOOL)) * IDX_D * 2);
}

inline float tg_sum(float v, threadgroup float *red, uint tid, uint nsg) {
    v = simd_sum(v);
    if (tid % 32 == 0) red[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint i = 0; i < nsg; i++) s += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

inline float norm_rope(float x, device const float *w, uint pos, uint n, threadgroup float *red, threadgroup float *buf, uint i) {
    x *= rsqrt(tg_sum(x * x, red, i, n / 32) / n + EPS) * w[i];
    if (i < 64) buf[i] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < 64) {
        const uint f = i % 32;
        const float th = float(pos) * pow(ROPE_BASE, -float(f) / 32.f);
        const float s = precise::sin(th), c = precise::cos(th);
        x = i < 32 ? x * c - buf[i + 32] * s : x * c + buf[i - 32] * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return x;
}

kernel void attn_prep(constant KVArgs &a [[buffer(0)]], device const float *qg [[buffer(1)]], device const float *kin [[buffer(2)]],
                      device const float *vin [[buffer(3)]], device const float *qnw [[buffer(4)]], device const float *knw [[buffer(5)]],
                      device uchar *kv [[buffer(6)]], device const int *pages [[buffer(7)]], device float *Q [[buffer(8)]],
                      uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    threadgroup float red[8], buf[64];
    const uint job = g.x, t = g.y, cell = a.pos0 + t;
    if (job < (uint)NH) {
        const float x = norm_rope(qg[((size_t)t * NH + job) * 2 * HD + i], qnw, cell, HD, red, buf, i);
        Q[((size_t)t * NH + job) * HD + i] = x * 0.0625f;
    } else if (job < (uint)(NH + NKV)) {
        const uint head = job - NH;
        const float x = norm_rope(kin[((size_t)t * NKV + head) * HD + i], knw, cell, HD, red, buf, i);
        kv_k(kv, pages, a.layer, head, cell)[i] = half(x);
    } else {
        const uint head = job - NH - NKV;
        kv_k(kv, pages, a.layer, head, cell)[KV_K / 2 + i] = half(vin[((size_t)t * NKV + head) * HD + i]);
    }
}

kernel void idx_qprep(constant KVArgs &a [[buffer(0)]], device const float *qraw [[buffer(1)]], device const float *qnw [[buffer(2)]],
                      device float *Qi [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    threadgroup float red[4], buf[64];
    const size_t o = ((size_t)g.y * IDX_H + g.x) * IDX_D + i;
    Qi[o] = norm_rope(qraw[o], qnw, a.pos0 + g.y, IDX_D, red, buf, i);
}

kernel void idx_pool(constant KVArgs &a [[buffer(0)]], device const float *kraw [[buffer(1)]], device const half *tail [[buffer(2)]],
                     device const float *knw [[buffer(3)]], device uchar *kv [[buffer(4)]], device const int *pages [[buffer(5)]],
                     uint gb [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    threadgroup float red[4], buf[64];
    const uint b = a.pos0 / KPOOL + gb;
    float x = 0;
    for (uint c = 0; c < (uint)KPOOL; c++) {
        const uint cell = b * KPOOL + c;
        x += cell < a.pos0 ? float(tail[(cell % KPOOL) * IDX_D + i]) : kraw[(size_t)(cell - a.pos0) * IDX_D + i];
    }
    x = norm_rope(x * 0.25f, knw, b * KPOOL, IDX_D, red, buf, i);
    kv_pool(kv, pages, a.layer, b)[i] = half(x);
}

kernel void idx_tail(constant KVArgs &a [[buffer(0)]], device const float *kraw [[buffer(1)]], device half *tail [[buffer(2)]],
                     uint i [[thread_index_in_threadgroup]]) {
    const uint n_end = a.pos0 + a.T;
    for (uint cell = max(a.pos0, n_end / KPOOL * KPOOL); cell < n_end; cell++)
        tail[(cell % KPOOL) * IDX_D + i] = half(kraw[(size_t)(cell - a.pos0) * IDX_D + i]);
}

kernel void idx_score(constant KVArgs &a [[buffer(0)]], device const float *Qi [[buffer(1)]], device uchar *kv [[buffer(2)]],
                      device const int *pages [[buffer(3)]], device float *score [[buffer(4)]], constant uint &ld [[buffer(5)]],
                      uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float q[IDX_H][IDX_D];
    const uint t = g.y, b = g.x * 256 + tid, pos = a.pos0 + t;
    for (uint k = tid; k < (uint)(IDX_H * IDX_D); k += 256) q[k / IDX_D][k % IDX_D] = Qi[(size_t)t * IDX_H * IDX_D + k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b >= ld || (b + 1) * KPOOL > pos + 1) return;
    device const half4 *k = (device const half4 *)kv_pool(kv, pages, a.layer, b);
    float acc[IDX_H] = {0, 0, 0, 0};
    for (int c = 0; c < IDX_D / 4; c++) {
        const float4 kk = float4(k[c]);
        for (int h = 0; h < IDX_H; h++) acc[h] += q[h][4 * c] * kk.x + q[h][4 * c + 1] * kk.y + q[h][4 * c + 2] * kk.z + q[h][4 * c + 3] * kk.w;
    }
    score[(size_t)t * ld + b] = max(acc[0], 0.f) + max(acc[1], 0.f) + max(acc[2], 0.f) + max(acc[3], 0.f);
}

kernel void idx_select(constant KVArgs &a [[buffer(0)]], device const float *score [[buffer(1)]], constant uint &ld [[buffer(2)]],
                       device int *sel [[buffer(3)]], device int *nsel [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint prefix, need, sgt[32], seq[32];
    const uint pos = a.pos0 + t, nvis = (pos + 1) / KPOOL;
    device const float *s = score + (size_t)t * ld;
    device int *out = sel + (size_t)t * (SEL + 1);
    uint n = 0;
    if (nvis <= (uint)SEL) {
        for (uint b = tid; b < nvis; b += 1024) out[b] = b;
        n = nvis;
    } else {
        if (tid == 0) { prefix = 0; need = SEL; }
        uint mask = 0;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (uint k = tid; k < 256; k += 1024) atomic_store_explicit(&hist[k], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint pf = prefix;
            for (uint b = tid; b < nvis; b += 1024) {
                const uint u = as_type<uint>(s[b]);
                if ((u & mask) == pf) atomic_fetch_add_explicit(&hist[(u >> shift) & 255], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                uint acc = 0; int d = 255;
                for (; d > 0; d--) { const uint h = atomic_load_explicit(&hist[d], memory_order_relaxed); if (acc + h >= need) break; acc += h; }
                need -= acc; prefix |= (uint)d << shift;
            }
            mask |= 255u << shift;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint thr = prefix, take_eq = need;
        const uint chunk = (nvis + 1023) / 1024, b0 = min(nvis, tid * chunk), b1 = min(nvis, b0 + chunk);
        uint gt = 0, eq = 0;
        for (uint b = b0; b < b1; b++) { const uint u = as_type<uint>(s[b]); gt += u > thr; eq += u == thr; }
        const uint pgt = simd_prefix_exclusive_sum(gt), peq = simd_prefix_exclusive_sum(eq);
        if (tid % 32 == 31) { sgt[tid / 32] = pgt + gt; seq[tid / 32] = peq + eq; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint bgt = 0, beq = 0;
        for (uint w = 0; w < tid / 32; w++) { bgt += sgt[w]; beq += seq[w]; }
        const uint eq_before = beq + peq, eq_mine = eq_before >= take_eq ? 0 : min(eq, take_eq - eq_before);
        uint o = bgt + pgt + min(eq_before, take_eq), used = 0;
        for (uint b = b0; b < b1; b++) {
            const uint u = as_type<uint>(s[b]);
            if (u > thr) out[o++] = b;
            else if (u == thr && used < eq_mine) { out[o++] = b; used++; }
        }
        n = SEL;
    }
    if (tid == 0) {
        if ((pos + 1) % KPOOL) out[n++] = (pos + 1) / KPOOL;
        nsel[t] = n;
    }
}

kernel void attn_dec(constant KVArgs &a [[buffer(0)]], device const float *Q [[buffer(1)]], device uchar *kv [[buffer(2)]],
                     device const int *pages [[buffer(3)]], device const int *sel [[buffer(4)]], device const int *nsel [[buffer(5)]],
                     device float *part_o [[buffer(6)]], device float *part_ml [[buffer(7)]], uint3 g [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint split = g.x, t = g.y, kh = g.z, head = kh * QPK + sg, pos = a.pos0 + t;
    const int n = nsel[t];
    const int per = (n + a.nsplit - 1) / a.nsplit, s0 = split * per, s1 = min(n, s0 + per);
    float q[8];
    device const float *qp = Q + ((size_t)t * NH + head) * HD + lane * 8;
    for (int k = 0; k < 8; k++) q[k] = qp[k];
    float m = -INFINITY, l = 0, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    device const int *sl = sel + (size_t)t * (SEL + 1);
    for (int si = s0; si < s1; si++) {
        const uint b = sl[si];
        for (uint c = 0; c < (uint)KPOOL; c++) {
            const uint cell = b * KPOOL + c;
            if (cell > pos) break;
            device const half4 *kp = (device const half4 *)(kv_k(kv, pages, a.layer, kh, cell) + lane * 8);
            const float4 k0 = float4(kp[0]), k1 = float4(kp[1]);
            float sc = q[0] * k0.x + q[1] * k0.y + q[2] * k0.z + q[3] * k0.w + q[4] * k1.x + q[5] * k1.y + q[6] * k1.z + q[7] * k1.w;
            sc = simd_sum(sc);
            const float mn = max(m, sc), al = exp(m - mn), p = exp(sc - mn);
            device const half4 *vp = (device const half4 *)(kv_k(kv, pages, a.layer, kh, cell) + KV_K / 2 + lane * 8);
            const float4 v0 = float4(vp[0]), v1 = float4(vp[1]);
            acc[0] = acc[0] * al + p * v0.x; acc[1] = acc[1] * al + p * v0.y; acc[2] = acc[2] * al + p * v0.z; acc[3] = acc[3] * al + p * v0.w;
            acc[4] = acc[4] * al + p * v1.x; acc[5] = acc[5] * al + p * v1.y; acc[6] = acc[6] * al + p * v1.z; acc[7] = acc[7] * al + p * v1.w;
            l = l * al + p;
            m = mn;
        }
    }
    const size_t pi = ((size_t)t * NH + head) * a.nsplit + split;
    for (int k = 0; k < 8; k++) part_o[pi * HD + lane * 8 + k] = acc[k];
    if (lane == 0) { part_ml[pi * 2] = m; part_ml[pi * 2 + 1] = l; }
}

kernel void attn_combine(constant KVArgs &a [[buffer(0)]], device const float *part_o [[buffer(1)]], device const float *part_ml [[buffer(2)]],
                         device const float *qg [[buffer(3)]], device float *att [[buffer(4)]], device half *atth [[buffer(5)]],
                         uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    const uint head = g.x, t = g.y;
    const size_t pi = ((size_t)t * NH + head) * a.nsplit;
    float M = -INFINITY;
    for (uint s = 0; s < a.nsplit; s++) M = max(M, part_ml[(pi + s) * 2]);
    float L = 0, o = 0;
    for (uint s = 0; s < a.nsplit; s++) {
        const float ms = part_ml[(pi + s) * 2];
        if (ms == -INFINITY) continue;
        const float w = exp(ms - M);
        L += w * part_ml[(pi + s) * 2 + 1];
        o += w * part_o[(pi + s) * HD + i];
    }
    const float y = o / L * sigm(qg[((size_t)t * NH + head) * 2 * HD + HD + i]);
    att[((size_t)t * NH + head) * HD + i] = y;
    atth[((size_t)t * NH + head) * HD + i] = half(y);
}
