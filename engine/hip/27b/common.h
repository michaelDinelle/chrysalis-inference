#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

[[noreturn, gnu::cold, gnu::noinline, gnu::format(printf, 1, 2)]] inline void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("chrysalis: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) fail("HIP error '%s' at %s:%d", hipGetErrorString(e_), __FILE__, __LINE__); } while (0)
#define FAIL(...) fail(__VA_ARGS__)

constexpr int D = 5120;
constexpr int FF = 17408;
constexpr int VOCAB = 248320;
constexpr int NL = 64;
constexpr int NH = 24, NKV = 4, HD = 256, ROT = 64;
constexpr int GK = 16, GV = 48, GD = 128;
constexpr bool SIGMOID_GATE = false;
constexpr int QKV = 2 * GK * GD + GV * GD;
constexpr int NATT = NL / 4;
constexpr int NKVL = NATT + 1;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 1e7f;
inline bool is_attn_layer(int l) { return l % 4 == 3; }

constexpr int PAGE = 64;
constexpr size_t KS_OFF = (size_t)PAGE * HD;
constexpr size_t VLO_OFF = KS_OFF + (size_t)PAGE * (HD / 32) * 2;
constexpr size_t VHI_OFF = VLO_OFF + (size_t)HD * PAGE / 2;
constexpr size_t VS_OFF = VHI_OFF + (size_t)HD * PAGE / 8;
constexpr size_t KV_HEAD = VS_OFF + (size_t)(HD / 32) * PAGE * 2;
constexpr size_t KV_LAYER = KV_HEAD * NKV;
constexpr size_t KV_PAGE = KV_LAYER * NKVL;
struct KVRef { uint8_t *pool; const int *pages; };

enum QType : int { F32 = 0, F16 = 1, Q8_0 = 8, Q6_K = 14, IQ4_XS = 23 };

struct QTensor {
    int type = -1;
    int rows = 0, cols = 0;
    uint8_t *q = nullptr;
    uint8_t *qh = nullptr;
    uint8_t *sc = nullptr;
    uint16_t *sh = nullptr;
    half *d = nullptr;
    void *f = nullptr;
};

struct QAct { int8_t *q; float *d; };

struct Arena {
    uint8_t *base = nullptr;
    size_t used = 0, cap = 0;
    void init(size_t bytes) { CK(hipMalloc(&base, bytes)); cap = bytes; used = 0; }
    template <typename T = uint8_t> T *get(size_t n) {
        size_t bytes = (n * sizeof(T) + 255) & ~size_t(255);
        if (used + bytes > cap) FAIL("arena out of memory (%zu + %zu > %zu)", used, bytes, cap);
        T *p = (T *)(base + used); used += bytes; return p;
    }
};

template <int M> __device__ __forceinline__ int lane_xor(int v) {
    static_assert(M == 1 || M == 2 || M == 4 || M == 8 || M == 16, "lane_xor: M");
    if constexpr (M == 1) return __builtin_amdgcn_update_dpp(0, v, 0xb1, 0xf, 0xf, true);
    else if constexpr (M == 2) return __builtin_amdgcn_update_dpp(0, v, 0x4e, 0xf, 0xf, true);
    else if constexpr (M == 4) return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xf, 0xf, true);
    else if constexpr (M == 8) return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xf, 0xf, true);
    else return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, true, false);
}
template <int M> __device__ __forceinline__ float lane_xor(float v) { return __int_as_float(lane_xor<M>(__float_as_int(v))); }

__device__ __forceinline__ float warp_sum(float v) {
    v += lane_xor<16>(v); v += lane_xor<8>(v); v += lane_xor<4>(v); v += lane_xor<2>(v); v += lane_xor<1>(v);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    v = fmaxf(v, lane_xor<16>(v)); v = fmaxf(v, lane_xor<8>(v)); v = fmaxf(v, lane_xor<4>(v));
    v = fmaxf(v, lane_xor<2>(v)); v = fmaxf(v, lane_xor<1>(v));
    return v;
}
__device__ __forceinline__ void wave_argmax(float &b, int &bi) {
#define CHRYSALIS_ARGMAX_STEP(M) { const float ov = lane_xor<M>(b); const int oi = lane_xor<M>(bi); \
                              const bool tk = (ov > b) | ((ov == b) & (oi < bi)); b = tk ? ov : b; bi = tk ? oi : bi; }
    CHRYSALIS_ARGMAX_STEP(16) CHRYSALIS_ARGMAX_STEP(8) CHRYSALIS_ARGMAX_STEP(4) CHRYSALIS_ARGMAX_STEP(2) CHRYSALIS_ARGMAX_STEP(1)
#undef CHRYSALIS_ARGMAX_STEP
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ int dot4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
