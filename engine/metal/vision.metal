#include <metal_stdlib>
using namespace metal;

constant int C = 1152, HEADS = 16, HDIM = 72, GRID = 48;

struct VisRows { int r0, n, gh, gw; };

inline float erf_(float x) {
    const float z = fabs(x), t = 1.0f / (1.0f + 0.3275911f * z);
    const float y = 1.0f - t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f)))) * exp(-z * z);
    return copysign(y, x);
}

inline int2 patch_rc(int p, int gw) {
    const int blk = p >> 2, in = p & 3, bw = gw >> 1;
    return int2((blk / bw) * 2 + (in >> 1), (blk % bw) * 2 + (in & 1));
}

kernel void vis_embed(device const float *y [[buffer(0)]], device const float *bias [[buffer(1)]], device const float *pos [[buffer(2)]],
                      device float *x [[buffer(3)]], constant VisRows &a [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    const int p = a.r0 + (int)t;
    const int2 rc = patch_rc(p, a.gw);
    const float sh = float(rc.x * (GRID - 1)) / float(max(a.gh - 1, 1)), sw = float(rc.y * (GRID - 1)) / float(max(a.gw - 1, 1));
    const int h0 = int(floor(sh)), w0 = int(floor(sw)), h1 = min(h0 + 1, GRID - 1), w1 = min(w0 + 1, GRID - 1);
    const float fh = sh - h0, fw = sw - w0;
    const float k00 = (1 - fh) * (1 - fw), k01 = (1 - fh) * fw, k10 = fh * (1 - fw), k11 = fh * fw;
    device const float *p00 = pos + (size_t)(h0 * GRID + w0) * C, *p01 = pos + (size_t)(h0 * GRID + w1) * C;
    device const float *p10 = pos + (size_t)(h1 * GRID + w0) * C, *p11 = pos + (size_t)(h1 * GRID + w1) * C;
    for (int c = tid; c < C; c += 256)
        x[(size_t)p * C + c] = y[(size_t)t * C + c] + bias[c] + (k00 * p00[c] + k01 * p01[c] + k10 * p10[c] + k11 * p11[c]);
}

kernel void vis_ln(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device const float *b [[buffer(2)]],
                   device float *out [[buffer(3)]], constant int &rows [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int row = tg * 8 + sg;
    if (row >= rows) return;
    device const float *xr = x + (size_t)row * C;
    float v[C / 32], s = 0;
    for (int i = 0; i < C / 32; i++) { v[i] = xr[lane + 32 * i]; s += v[i]; }
    const float mean = simd_sum(s) / C;
    float q = 0;
    for (int i = 0; i < C / 32; i++) { v[i] -= mean; q += v[i] * v[i]; }
    const float inv = 1.0f / sqrt(simd_sum(q) / C + 1e-6f);
    for (int i = 0; i < C / 32; i++) {
        const int c = lane + 32 * i;
        out[(size_t)row * C + c] = v[i] * inv * w[c] + b[c];
    }
}

kernel void vis_qkv(device const float *y [[buffer(0)]], device const float *bias [[buffer(1)]], device float *qkv [[buffer(2)]],
                    constant VisRows &a [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int p = a.r0 + (int)t;
    const int2 rc = patch_rc(p, a.gw);
    device const float *yr = y + (size_t)t * 3 * C;
    device float *row = qkv + (size_t)p * 3 * C;
    for (int i = tid; i < 2 * HEADS * 36; i += 256) {
        const int part = i / (HEADS * 36), hj = i % (HEADS * 36), h = hj / 36, j = hj % 36;
        const int d0 = part * C + h * HDIM + j, d1 = d0 + 36;
        const int f = j % 18;
        const float inv = 1.0f / pow(10000.0f, float(2 * f) / 36.0f);
        const float ang = float(j < 18 ? rc.x : rc.y) * inv;
        const float s = sin(ang), c = cos(ang);
        const float u = yr[d0] + bias[d0], v = yr[d1] + bias[d1];
        const float scale = part == 0 ? 0.11785113019775792f : 1.0f;
        row[d0] = (u * c - v * s) * scale;
        row[d1] = (v * c + u * s) * scale;
    }
    for (int i = tid; i < C; i += 256) row[2 * C + i] = yr[2 * C + i] + bias[2 * C + i];
}

kernel void vis_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], device const float *bias [[buffer(2)]],
                    constant int &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if ((int)i < n) x[i] += y[i] + bias[i % C];
}

struct VisGelu { int n, cols, exact; };

kernel void vis_gelu(device float *y [[buffer(0)]], device const float *bias [[buffer(1)]], constant VisGelu &a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if ((int)i >= a.n) return;
    const float v = y[i] + bias[i % a.cols];
    const float g = a.exact ? 0.5f * v * (1.0f + erf_(v * 0.70710678118654752f)) : 0.5f * v * (1.0f + tanh(0.79788456080286536f * (v + 0.044715f * v * v * v)));
    y[i] = g;
}

kernel void vis_out(device const float *y [[buffer(0)]], device const float *bias [[buffer(1)]], device half *out [[buffer(2)]],
                    constant int2 &a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if ((int)i < a.x) out[i] = half(y[i] + bias[i % a.y]);
}

constant constexpr int FB = 32, FLD = HDIM + 4, FSG = 8;

kernel void vis_attn(device float *qkv [[buffer(0)]], constant int &n [[buffer(1)]], uint2 tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                     ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float Ks[FB * FLD], Vs[FB * FLD];
    threadgroup float Ss[FSG][8 * FB];
    threadgroup float Al[FSG][8];
    const int h = tg.y, r0 = (tg.x * FSG + sg) * 8;
    const ulong ld = 3 * C;
    device float *qp = qkv + (ulong)r0 * ld + h * HDIM;
    simdgroup_float8x8 Q[9], O[9];
    for (int d = 0; d < 9; d++) {
        simdgroup_load(Q[d], qp + d * 8, ld);
        O[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const int row = lane / 4, part = lane % 4, fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    float m = -INFINITY, l = 0;
    for (int k0 = 0; k0 < n; k0 += FB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < FB * HDIM / 4; i += FSG * 32) {
            const int kk = i / (HDIM / 4), piece = i % (HDIM / 4), key = k0 + kk;
            float4 kv = 0, vv = 0;
            if (key < n) {
                device const float *src = qkv + (ulong)key * ld + h * HDIM + piece * 4;
                kv = *(device const float4 *)(src + C);
                vv = *(device const float4 *)(src + 2 * C);
            }
            *(threadgroup float4 *)&Ks[kk * FLD + piece * 4] = kv;
            *(threadgroup float4 *)&Vs[kk * FLD + piece * 4] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < FB / 8; j++) {
            simdgroup_float8x8 Sj = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int d = 0; d < 9; d++) {
                simdgroup_float8x8 Kt;
                simdgroup_load(Kt, &Ks[j * 8 * FLD + d * 8], FLD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(Sj, Q[d], Kt, Sj);
            }
            simdgroup_store(Sj, &Ss[sg][j * 8], FB);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float s[8], mx = -INFINITY;
        for (int j = 0; j < 8; j++) {
            const int c = part * 8 + j;
            s[j] = k0 + c < n ? Ss[sg][row * FB + c] : -INFINITY;
            mx = max(mx, s[j]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx);
        const float alpha = m == -INFINITY ? 0.0f : exp(m - mn);
        float ps = 0;
        for (int j = 0; j < 8; j++) {
            const float p = s[j] == -INFINITY ? 0.0f : exp(s[j] - mn);
            Ss[sg][row * FB + part * 8 + j] = p;
            ps += p;
        }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 2);
        l = l * alpha + ps;
        m = mn;
        if (part == 0) Al[sg][row] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float al = Al[sg][fm];
        for (int d = 0; d < 9; d++) {
            O[d].thread_elements()[0] *= al;
            O[d].thread_elements()[1] *= al;
        }
        for (int j = 0; j < FB / 8; j++) {
            simdgroup_float8x8 P;
            simdgroup_load(P, &Ss[sg][j * 8], FB);
            for (int d = 0; d < 9; d++) {
                simdgroup_float8x8 V;
                simdgroup_load(V, &Vs[j * 8 * FLD + d * 8], FLD);
                simdgroup_multiply_accumulate(O[d], P, V, O[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (part == 0) Al[sg][row] = 1.0f / l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = Al[sg][fm];
    for (int d = 0; d < 9; d++) {
        O[d].thread_elements()[0] *= inv;
        O[d].thread_elements()[1] *= inv;
        simdgroup_store(O[d], qp + d * 8, ld);
    }
}
