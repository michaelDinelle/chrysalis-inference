#include "quant.h"

struct GemvArgs {
    uint rows, cols, rb;
    uint T, ldx, ldy;
    uint add;
    float scale;
    uint ggml;
};

template <int F, int LPR, int R, int MAXT>
kernel void gemv(constant GemvArgs &a [[buffer(0)]], device const uchar *W [[buffer(1)]], device const float *x [[buffer(2)]],
                 device float *y [[buffer(3)]], device const float *inj [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int RPS = 32 / LPR;
    const uint row0 = ((tg * 4 + sg) * RPS + lane / LPR) * R, sl = lane % LPR;
    const uint units = a.cols / 32;
    float acc[R][MAXT];
    for (int r = 0; r < R; r++) for (int t = 0; t < MAXT; t++) acc[r][t] = 0;
    const int nr = row0 >= a.rows ? 0 : min((int)R, (int)(a.rows - row0));
    device const uchar *wr = W + (size_t)row0 * a.rb;
    threadgroup float kvt[16];
    if (sg == 0 && lane < 16) kvt[lane] = kv_iq4nl[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < a.T; t++) {
        for (uint u = sl; u < units; u += LPR) {
            float xs[32];
            device const float4 *xp = (device const float4 *)(x + (size_t)t * a.ldx + u * 32);
            float2 sx = 0;
            for (int i = 0; i < 8; i++) {
                const float4 v = xp[i];
                xs[4 * i] = v.x; xs[4 * i + 1] = v.y; xs[4 * i + 2] = v.z; xs[4 * i + 3] = v.w;
            }
            if (a.ggml) ggml_round<F>(xs);
            for (int i = 0; i < 32; i++) sx[i / 16] += xs[i];
            for (int r = 0; r < R; r++) {
                if (r >= nr) break;
                acc[r][t] += dot32<F>(wr + r * a.rb, u, xs, sx, kvt);
            }
        }
    }
    for (int r = 0; r < R; r++) {
        for (uint t = 0; t < a.T; t++) {
            float s = acc[r][t];
            for (int o = LPR / 2; o > 0; o /= 2) s += simd_shuffle_xor(s, o);
            if (sl == 0 && r < nr) {
                if (a.add == 2) {
                    for (int c = 0; c < 4; c++) y[(size_t)t * 10240 + c * 2560 + row0 + r] += s * a.scale * 2.f / (1.f + exp(-inj[t * 4 + c] * 0.25f));
                } else {
                    device float *dst = y + (size_t)t * a.ldy + row0 + r;
                    *dst = (a.add ? *dst : 0.f) + s * a.scale;
                }
            }
        }
    }
}

template <int F>
kernel void deq_rows(constant uint4 &a [[buffer(0)]], device const uchar *W [[buffer(1)]], device float *out [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint u = gid.x, r = gid.y;
    if (u >= a.x / 32 || r >= a.z) return;
    float w[32];
    deq32<F>(W + (size_t)r * a.y, u, w);
    for (int i = 0; i < 32; i++) out[(size_t)r * a.x + u * 32 + i] = w[i];
}

#define GEMV_TYPES(X) X(T_F32, f32) X(T_F16, f16) X(T_BF16, bf16) X(T_Q2_0, q2_0) X(T_Q4_0, q4_0) X(T_Q5_0, q5_0) X(T_Q8_0, q8_0) \
    X(T_IQ4_NL, iq4_nl) X(T_IQ4_XS, iq4_xs) X(T_Q2_K, q2_k) X(T_Q3_K, q3_k) X(T_Q4_K, q4_k) X(T_Q5_K, q5_k) X(T_Q6_K, q6_k)

#define INST_GEMV(F, name) \
    template [[host_name("gemv32_" #name)]] kernel void gemv<F, 32, 4, 1>(constant GemvArgs &, device const uchar *, device const float *, device float *, device const float *, uint, uint, uint); \
    template [[host_name("gemv8_" #name)]] kernel void gemv<F, 8, 2, 1>(constant GemvArgs &, device const uchar *, device const float *, device float *, device const float *, uint, uint, uint); \
    template [[host_name("gemv32t_" #name)]] kernel void gemv<F, 32, 4, 8>(constant GemvArgs &, device const uchar *, device const float *, device float *, device const float *, uint, uint, uint); \
    template [[host_name("gemv8t_" #name)]] kernel void gemv<F, 8, 2, 8>(constant GemvArgs &, device const uchar *, device const float *, device float *, device const float *, uint, uint, uint); \
    template [[host_name("gemvr1_" #name)]] kernel void gemv<F, 32, 1, 1>(constant GemvArgs &, device const uchar *, device const float *, device float *, device const float *, uint, uint, uint); \
    template [[host_name("gemvr2_" #name)]] kernel void gemv<F, 32, 2, 1>(constant GemvArgs &, device const uchar *, device const float *, device float *, device const float *, uint, uint, uint); \
    template [[host_name("gemvr8_" #name)]] kernel void gemv<F, 32, 8, 1>(constant GemvArgs &, device const uchar *, device const float *, device float *, device const float *, uint, uint, uint); \
    template [[host_name("deq_rows_" #name)]] kernel void deq_rows<F>(constant uint4 &, device const uchar *, device float *, uint2);
GEMV_TYPES(INST_GEMV)
