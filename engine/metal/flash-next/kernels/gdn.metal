#include <metal_stdlib>
using namespace metal;

constant int GK = 16, GV = 48, GD = 128, QKV = 10240;
constant float EPS = 1e-6f;
inline float sigm(float x) { return 1.f / (1.f + exp(-x)); }
inline float silu(float x) { return x / (1.f + exp(-x)); }
inline float softplus(float x) { return x > 20.f ? x : log(1.f + exp(x)); }

struct GdnStepArgs {
    uint T;
    uint write_state;
};

kernel void gdn_step(constant GdnStepArgs &a [[buffer(0)]], device const float *raw [[buffer(1)]], device const float *ab [[buffer(2)]],
                     device const float *z [[buffer(3)]], device const float *conv_w [[buffer(4)]], device const float *conv_state [[buffer(5)]],
                     device const float *A [[buffer(6)]], device const float *dtb [[buffer(7)]], device float *S [[buffer(8)]],
                     device const float *norm_w [[buffer(9)]], device float *out [[buffer(10)]],
                     uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qs[8][GD], ks[8][GD], vs[8][GD], os[8][GD];
    threadgroup float gates[8][2], rms[8];
    const uint kh = h % GK, T = a.T, sg = tid / 32, lane = tid % 32;
    if (tid < T) {
        gates[tid][0] = exp(A[h] * softplus(ab[tid * 96 + h] + dtb[h]));
        gates[tid][1] = sigm(ab[tid * 96 + 48 + h]);
    }
    if (tid < 3 * GD) {
        const uint which = tid / GD, i = tid % GD;
        const uint c = which == 0 ? kh * GD + i : which == 1 ? GK * GD + kh * GD + i : 2 * GK * GD + h * GD + i;
        const float w0 = conv_w[c * 4], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], w3 = conv_w[c * 4 + 3];
        float s0 = conv_state[c * 3], s1 = conv_state[c * 3 + 1], s2 = conv_state[c * 3 + 2];
        threadgroup float (*dst)[GD] = which == 0 ? qs : which == 1 ? ks : vs;
        for (uint t = 0; t < T; t++) {
            const float x = raw[t * QKV + c];
            dst[t][i] = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
            s0 = s1; s1 = s2; s2 = x;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = sg; r < 2 * T; r += 16) {
        threadgroup float *row = (r & 1) ? ks[r / 2] : qs[r / 2];
        float4 v = float4(row[lane * 4], row[lane * 4 + 1], row[lane * 4 + 2], row[lane * 4 + 3]);
        const float inv = rsqrt(simd_sum(dot(v, v)) + EPS);
        for (int k = 0; k < 4; k++) row[lane * 4 + k] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint j = tid / 4, part = tid % 4;
    device float *Sj = S + ((size_t)h * GD + j) * GD + part * 32;
    float st[32];
    for (int i = 0; i < 32; i++) st[i] = Sj[i];
    for (uint t = 0; t < T; t++) {
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
    if (a.write_state) for (int i = 0; i < 32; i++) Sj[i] = st[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < T) {
        float s = 0;
        for (int k = 0; k < 4; k++) s += os[sg][lane * 4 + k] * os[sg][lane * 4 + k];
        s = simd_sum(s);
        if (lane == 0) rms[sg] = rsqrt(s / GD + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < T * GD; idx += 512) {
        const uint t = idx / GD, col = idx % GD;
        const size_t o = (size_t)t * GV * GD + h * GD + col;
        out[o] = os[t][col] * rms[t] * norm_w[col] * sigm(z[o]);
    }
}

kernel void gdn_conv_commit(device float *state [[buffer(0)]], device const float *raw [[buffer(1)]], constant uint &n [[buffer(2)]],
                            uint c [[thread_position_in_grid]]) {
    if (c >= (uint)QKV) return;
    float w[3] = {state[c * 3], state[c * 3 + 1], state[c * 3 + 2]};
    for (uint t = 0; t < n; t++) { w[0] = w[1]; w[1] = w[2]; w[2] = raw[t * QKV + c]; }
    state[c * 3] = w[0]; state[c * 3 + 1] = w[1]; state[c * 3 + 2] = w[2];
}

kernel void gdn_conv_pre(device const float *raw [[buffer(0)]], constant uint &T [[buffer(1)]], device const float *w [[buffer(2)]],
                         device const float *state [[buffer(3)]], device float *qn [[buffer(4)]], device float *kn [[buffer(5)]],
                         device float *vc [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    threadgroup float red[4];
    const uint head = g.x, t0 = g.y * 16, c = head * GD + i;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    auto in = [&](int t) { return t >= 0 ? raw[(size_t)t * QKV + c] : state[c * 3 + 3 + t]; };
    float s0 = in((int)t0 - 3), s1 = in((int)t0 - 2), s2 = in((int)t0 - 1);
    for (uint t = t0; t < min(T, t0 + 16); t++) {
        const float x = raw[(size_t)t * QKV + c];
        float y = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
        s0 = s1; s1 = s2; s2 = x;
        if (head < 2 * GK) {
            const float ss = simd_sum(y * y);
            if (i % 32 == 0) red[i / 32] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            y *= rsqrt(red[0] + red[1] + red[2] + red[3] + EPS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (head < (uint)GK ? qn : kn)[((size_t)t * GK + head % GK) * GD + i] = y;
        } else {
            vc[((size_t)t * GV + head - 2 * GK) * GD + i] = y;
        }
    }
}

constant int CT = 16;
kernel void gdn_scan(device const float *qn [[buffer(0)]], device const float *kn [[buffer(1)]], device float *vc [[buffer(2)]],
                     device const float *ab [[buffer(3)]], device const float *A [[buffer(4)]], device const float *dtb [[buffer(5)]],
                     device float *S [[buffer(6)]], constant uint &T [[buffer(7)]], uint2 g [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float ks[CT][GD], qs[CT][GD], vs[CT][32], dec[CT], bet[CT];
    const uint h = g.x, kh = h % GK, j = g.y * 32 + tid / 4, part = tid % 4;
    device float *Sj = S + ((size_t)h * GD + j) * GD + part * 32;
    float s[32];
    for (int i = 0; i < 32; i++) s[i] = Sj[i];
    const float An = A[h], db = dtb[h];
    for (uint t0 = 0; t0 < T; t0 += CT) {
        const uint n = min((uint)CT, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < n * GD; idx += 128) {
            const uint t = idx / GD, c = idx % GD;
            ks[t][c] = kn[((size_t)(t0 + t) * GK + kh) * GD + c];
            qs[t][c] = qn[((size_t)(t0 + t) * GK + kh) * GD + c];
        }
        for (uint idx = tid; idx < n * 32; idx += 128)
            vs[idx / 32][idx % 32] = vc[((size_t)(t0 + idx / 32) * GV + h) * GD + g.y * 32 + idx % 32];
        if (tid < n) {
            dec[tid] = exp(An * softplus(ab[(t0 + tid) * 96 + h] + db));
            bet[tid] = sigm(ab[(t0 + tid) * 96 + 48 + h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < n; t++) {
            const float decay = dec[t], beta = bet[t];
            threadgroup const float *kp = &ks[t][part * 32], *qp = &qs[t][part * 32];
            float kv = 0;
            for (int i = 0; i < 32; i++) kv += s[i] * kp[i];
            kv += simd_shuffle_xor(kv, 1); kv += simd_shuffle_xor(kv, 2);
            const float delta = beta * (vs[t][tid / 4] - decay * kv);
            float o = 0;
            for (int i = 0; i < 32; i++) { s[i] = decay * s[i] + kp[i] * delta; o += s[i] * qp[i]; }
            o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2);
            if (part == 0) vc[((size_t)(t0 + t) * GV + h) * GD + j] = o * 0.08838834764831845f;
        }
    }
    for (int i = 0; i < 32; i++) Sj[i] = s[i];
}

kernel void gdn_out(device float *o [[buffer(0)]], device const float *z [[buffer(1)]], device const float *w [[buffer(2)]],
                    uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
    threadgroup float red[4];
    const size_t idx = ((size_t)g.y * GV + g.x) * GD + i;
    const float v = o[idx];
    const float ss = simd_sum(v * v);
    if (i % 32 == 0) red[i / 32] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[idx] = v * rsqrt((red[0] + red[1] + red[2] + red[3]) / GD + EPS) * w[i] * sigm(z[idx]);
}
