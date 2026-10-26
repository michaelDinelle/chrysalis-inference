#include "kcommon.h"

inline float sum128(float v, threadgroup float *red, ushort i) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((i & 31) == 0) red[i >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0] + red[1] + red[2] + red[3];
}

kernel void gdn_conv(constant ConvA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort i [[thread_index_in_threadgroup]]) {
    threadgroup float red[4];
    const int head = tg.x, t0 = tg.y * 16, c = head * GD + i;
    const float w0 = a.w[c * 4], w1 = a.w[c * 4 + 1], w2 = a.w[c * 4 + 2], w3 = a.w[c * 4 + 3];
    auto in = [&](int t) { return t >= 0 ? a.raw[(ulong)t * QKV + c] : a.state[c * 3 + 3 + t]; };
    float s0 = in(t0 - 3), s1 = in(t0 - 2), s2 = in(t0 - 1);
    const int t1 = min(a.T, t0 + 16);
    for (int t = t0; t < t1; t++) {
        const float x = a.raw[(ulong)t * QKV + c];
        float y = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
        s0 = s1; s1 = s2; s2 = x;
        if (head < 2 * GK) {
            y *= rsqrt(sum128(y * y, red, i) + EPS);
            (head < GK ? a.qn : a.kn)[((ulong)t * GK + (head % GK)) * GD + i] = y;
        } else {
            a.vc[((ulong)t * GV + head - 2 * GK) * GD + i] = y;
        }
    }
}
kernel void gdn_conv_state(constant ConvStateA &a [[buffer(0)]], uint c [[thread_position_in_grid]]) {
    if ((int)c < QKV) for (int k = 0; k < 3; k++) a.state[c * 3 + k] = a.raw[(ulong)(a.T - 3 + k) * QKV + c];
}

kernel void gdn_conv_commit(constant ConvCommitA &a [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    const int c = g.x, layer = g.y;
    if (c >= QKV) return;
    device float *s = a.state + ((ulong)layer * QKV + c) * 3;
    device const float *r = a.raw + (ulong)layer * a.raw_layer + c;
    float w[3] = {s[0], s[1], s[2]};
    for (int t = 0; t < a.n; t++) { w[0] = w[1]; w[1] = w[2]; w[2] = r[(ulong)t * QKV]; }
    s[0] = w[0]; s[1] = w[1]; s[2] = w[2];
}

constant constexpr int CT = 16;
kernel void gdn_scan(constant ScanA &a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float ks[CT][GD], qs[CT][GD], vs[CT][32], dec[CT], bet[CT];
    const int h = tg.x, kh = h % GK, layer = tg.z;
    const int j = tg.y * 32 + tid / 4, part = tid % 4;
    device float *Sj = a.S + (ulong)layer * a.S_layer + ((ulong)h * GD + j) * GD + part * 32;
    device const float *qn = a.qn + (ulong)layer * a.in_layer, *kn = a.kn + (ulong)layer * a.in_layer;
    device const float *vc = a.vc + (ulong)layer * a.v_layer, *ab = a.ab + (ulong)layer * a.ab_layer;
    float s[32];
    for (int i = 0; i < 32; i += 4) { const float4 v = *(device const float4 *)&Sj[i]; s[i] = v.x; s[i + 1] = v.y; s[i + 2] = v.z; s[i + 3] = v.w; }
    const float An = a.A[layer * GV + h], db = a.dtb[layer * GV + h];
    const bool out = a.out;
    for (int t0 = 0; t0 < a.T; t0 += CT) {
        const int n = min(CT, a.T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < n * GD; idx += 128) {
            const int t = idx / GD, c = idx % GD;
            ks[t][c] = kn[((ulong)(t0 + t) * GK + kh) * GD + c];
            if (out) qs[t][c] = qn[((ulong)(t0 + t) * GK + kh) * GD + c];
        }
        for (int idx = tid; idx < n * 32; idx += 128)
            vs[idx / 32][idx % 32] = vc[((ulong)(t0 + idx / 32) * GV + h) * GD + tg.y * 32 + idx % 32];
        if (tid < n) {
            dec[tid] = exp(An * softplus(ab[(t0 + tid) * 2 * GV + h] + db));
            bet[tid] = sigmoid(ab[(t0 + tid) * 2 * GV + GV + h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < n; t++) {
            const float decay = dec[t], beta = bet[t];
            threadgroup const float *kp = &ks[t][part * 32];
            float k4[4] = {0, 0, 0, 0};
            for (int i = 0; i < 32; i++) k4[i % 4] += s[i] * kp[i];
            float kv = (k4[0] + k4[1]) + (k4[2] + k4[3]);
            kv += simd_shuffle_xor(kv, 1); kv += simd_shuffle_xor(kv, 2);
            const float delta = beta * (vs[t][tid / 4] - decay * kv);
            for (int i = 0; i < 32; i++) s[i] = decay * s[i] + kp[i] * delta;
            if (out) {
                threadgroup const float *qp = &qs[t][part * 32];
                float o4[4] = {0, 0, 0, 0};
                for (int i = 0; i < 32; i++) o4[i % 4] += s[i] * qp[i];
                float o = (o4[0] + o4[1]) + (o4[2] + o4[3]);
                o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2);
                if (part == 0) a.out[((ulong)(t0 + t) * GV + h) * GD + j] = o * 0.08838834764831845f;
            }
        }
    }
    if (a.write_state)
        for (int i = 0; i < 32; i += 4) *(device float4 *)&Sj[i] = float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
}

kernel void gdn_out(constant GdnOutA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort i [[thread_index_in_threadgroup]]) {
    threadgroup float red[4];
    const ulong idx = ((ulong)tg.y * GV + tg.x) * GD + i;
    const float v = a.o[idx];
    const float y = v * rsqrt(sum128(v * v, red, i) / GD + EPS) * a.w[i] * silu(a.z[idx]);
    if (a.of) a.of[idx] = y;
    if (a.oh) a.oh[idx] = half(y);
}

kernel void gdn_ab(constant AbA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort i [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];
    device const bfloat *w = a.W + (ulong)tg.x * D;
    device const half *x = a.x + (ulong)tg.y * D;
    float s = 0;
    for (int k = i; k < D; k += 256) s += float(w[k]) * float(x[k]);
    s = simd_sum(s);
    if ((i & 31) == 0) red[i >> 5] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i == 0) {
        float t = 0;
        for (int k = 0; k < 8; k++) t += red[k];
        a.out[tg.y * 2 * GV + tg.x] = t;
    }
}

template <int TT>
kernel void gdn_step(constant StepA &a [[buffer(0)]], uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[8][GD], ks[8][GD], vs[8][GD], os[8][GD];
    threadgroup float red[16][16], gates[8][2], rms[8];
    const int kh = h % GK, warp = tid / 32, lane = tid % 32, T = TT ? TT : a.T;
    const int j = tid / 4, part = tid % 4;
    device float *Sj = a.S + ((ulong)h * GD + j) * GD + part * 32;
    float st[32];
    for (int i = 0; i < 32; i += 4) { const float4 v = *(device const float4 *)&Sj[i]; st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w; }

    float pa[8] = {0, 0, 0, 0, 0, 0, 0, 0}, pb[8] = {0, 0, 0, 0, 0, 0, 0, 0}, ps[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    threadgroup float red2[16][8];
    device const bfloat *wa = a.Wab + (ulong)h * D, *wb = a.Wab + (ulong)(GV + h) * D;
    const bool norm = a.nw;
    for (int i = tid; i < D; i += 512) {
        const float nwi = norm ? a.nw[i] : 1.0f, fa = float(wa[i]) * nwi, fb = float(wb[i]) * nwi;
        for (int t = 0; t < T; t++) {
            const float x = a.x[(ulong)t * D + i];
            pa[t] += fa * x;
            pb[t] += fb * x;
            ps[t] += x * x;
        }
    }
    for (int t = 0; t < T; t++) {
        const float sa = simd_sum(pa[t]), sb = simd_sum(pb[t]), sq = simd_sum(ps[t]);
        if (lane == 0) { red[warp][2 * t] = sa; red[warp][2 * t + 1] = sb; red2[warp][t] = sq; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < T) {
        float al = 0, be = 0, sq = 0;
        for (int w = 0; w < 16; w++) { al += red[w][2 * tid]; be += red[w][2 * tid + 1]; sq += red2[w][tid]; }
        if (norm) { const float inv = rsqrt(sq / D + EPS); al *= inv; be *= inv; }
        gates[tid][0] = exp(a.A[h] * softplus(al + a.dt[h]));
        gates[tid][1] = sigmoid(be);
        if (a.verify) { a.ab_out[tid * 2 * GV + h] = al; a.ab_out[tid * 2 * GV + GV + h] = be; }
    }
    if (tid < 3 * GD) {
        const int which = tid / GD, i = tid % GD;
        const int c = which == 0 ? kh * GD + i : which == 1 ? GK * GD + kh * GD + i : 2 * GK * GD + h * GD + i;
        const float w0 = a.conv_w[c * 4], w1 = a.conv_w[c * 4 + 1], w2 = a.conv_w[c * 4 + 2], w3 = a.conv_w[c * 4 + 3];
        device const float *cs = a.conv_state + c * 3;
        float s0 = cs[0], s1 = cs[1], s2 = cs[2];
        for (int t = 0; t < T; t++) {
            const float x = a.raw[(ulong)t * QKV + c];
            const float y = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
            if (which == 0) qs[t][i] = y; else if (which == 1) ks[t][i] = y; else vs[t][i] = y;
            s0 = s1; s1 = s2; s2 = x;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = warp; r < 2 * T; r += 16) {
        threadgroup float *rw = (r & 1) ? ks[r / 2] : qs[r / 2];
        float4 v = *(threadgroup float4 *)&rw[lane * 4];
        const float inv = rsqrt(simd_sum(dot(v, v)) + EPS);
        *(threadgroup float4 *)&rw[lane * 4] = v * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a.verify) {
        for (int idx = tid; idx < T * GD; idx += 512) {
            const int t = idx / GD, i = idx % GD;
            if ((int)h < GK) a.kn_out[((ulong)t * GK + kh) * GD + i] = ks[t][i];
            a.vc_out[((ulong)t * GV + h) * GD + i] = vs[t][i];
        }
    }
    for (int t = 0; t < T; t++) {
        const float decay = gates[t][0], beta = gates[t][1];
        threadgroup const float *kp = &ks[t][part * 32], *qp = &qs[t][part * 32];
        float kv = 0;
        for (int i = 0; i < 32; i++) kv += st[i] * kp[i];
        kv += simd_shuffle_xor(kv, 1); kv += simd_shuffle_xor(kv, 2);
        const float delta = beta * (vs[t][j] - decay * kv);
        float o = 0;
        for (int i = 0; i < 32; i++) { st[i] = decay * st[i] + kp[i] * delta; o += st[i] * qp[i]; }
        o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2);
        if (part == 0) os[t][j] = o * 0.08838834764831845f;
    }
    if (!a.verify)
        for (int i = 0; i < 32; i += 4) *(device float4 *)&Sj[i] = float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp < T) {
        const float4 v = *(threadgroup float4 *)&os[warp][lane * 4];
        const float ss = simd_sum(dot(v, v));
        if (lane == 0) rms[warp] = rsqrt(ss / GD + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < T * GD; idx += 512) {
        const int t = idx / GD, col = idx % GD;
        const ulong o = (ulong)t * GV * GD + h * GD + col;
        const float y = os[t][col] * rms[t] * a.norm_w[col] * silu(a.z[o]);
        a.out[o] = y;
        a.out16[o] = half(y);
    }
}
template [[host_name("gdn_step")]] kernel decltype(gdn_step<0>) gdn_step<0>;
template [[host_name("gdn_step1")]] kernel decltype(gdn_step<1>) gdn_step<1>;

kernel void gdn_step1s(constant StepA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[GD], ks[GD], vs[32], red[4][2], gates[2];
    const int h = tg.x, jb = tg.y, kh = h % GK, warp = tid / 32, lane = tid % 32;
    const int jl = tid / 4, part = tid % 4, j = jb * 32 + jl;
    device float *Sj = a.S + ((ulong)h * GD + j) * GD + part * 32;
    float st[32];
    for (int i = 0; i < 32; i += 4) { const float4 v = *(device const float4 *)&Sj[i]; st[i] = v.x; st[i + 1] = v.y; st[i + 2] = v.z; st[i + 3] = v.w; }

    device const float4 *cw = (device const float4 *)a.conv_w;
    auto conv = [&](int c) {
        const float4 w = cw[c];
        device const float *cs = a.conv_state + c * 3;
        return silu(w.x * cs[0] + w.y * cs[1] + w.z * cs[2] + w.w * a.raw[c]);
    };
    const float qv = conv(kh * GD + tid), kv0 = conv(GK * GD + kh * GD + tid);
    if (tid < 32) vs[tid] = conv(2 * GK * GD + h * GD + jb * 32 + tid);
    const float sq = simd_sum(qv * qv), sk = simd_sum(kv0 * kv0);
    if (lane == 0) { red[warp][0] = sq; red[warp][1] = sk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qq = 0, kk = 0;
    for (int w = 0; w < 4; w++) { qq += red[w][0]; kk += red[w][1]; }
    const float al = a.ab_in[h], be = a.ab_in[GV + h];
    qs[tid] = qv * rsqrt(qq + EPS);
    ks[tid] = kv0 * rsqrt(kk + EPS);
    if (tid == 0) { gates[0] = exp(a.A[h] * softplus(al + a.dt[h])); gates[1] = sigmoid(be); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float decay = gates[0], beta = gates[1];
    threadgroup const float *kp = &ks[part * 32], *qp = &qs[part * 32];
    float kv = 0;
    for (int i = 0; i < 32; i++) kv += st[i] * kp[i];
    kv += simd_shuffle_xor(kv, 1); kv += simd_shuffle_xor(kv, 2);
    const float delta = beta * (vs[jl] - decay * kv);
    float o = 0;
    for (int i = 0; i < 32; i++) { st[i] = decay * st[i] + kp[i] * delta; o += st[i] * qp[i]; }
    o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2);
    if (part == 0) a.out[h * GD + j] = o * 0.08838834764831845f;
    for (int i = 0; i < 32; i += 4) *(device float4 *)&Sj[i] = float4(st[i], st[i + 1], st[i + 2], st[i + 3]);
}

kernel void gdn_ab1(constant StepA &a [[buffer(0)]], uint r [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[4][2];
    device const bfloat4 *w = (device const bfloat4 *)(a.Wab + (ulong)r * D);
    device const float4 *x4 = (device const float4 *)a.x, *n4 = (device const float4 *)a.nw;
    float p = 0, ps = 0;
    for (int i = tid; i < D / 4; i += 128) {
        const float4 x = x4[i];
        p += dot(float4(w[i]) * (a.nw ? n4[i] : 1.0f), x);
        ps += dot(x, x);
    }
    p = simd_sum(p); ps = simd_sum(ps);
    if (tid % 32 == 0) { red[tid / 32][0] = p; red[tid / 32][1] = ps; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0, ss = 0;
        for (int k = 0; k < 4; k++) { s += red[k][0]; ss += red[k][1]; }
        a.ab_out[r] = a.nw ? s * rsqrt(ss / D + EPS) : s;
    }
}
