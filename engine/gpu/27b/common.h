#pragma once
#include "gpu_api.h"
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "../../os.h"

[[noreturn, gnu::cold, gnu::noinline, gnu::format(printf, 1, 2)]] inline void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("chrysalis: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}
#define CK(x) do { gpuError_t e_ = (x); if (e_ != gpuSuccess) fail(GPU_CK_FMT, gpuGetErrorString(e_), __FILE__, __LINE__); } while (0)
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

#if defined(__HIP_PLATFORM_AMD__)
struct QAct { int8_t *q; float *d; };
#else
struct QAct { int8_t *q; float *d; int8_t *r = nullptr; };
#endif

struct Arena {
    uint8_t *base = nullptr;
    size_t used = 0, cap = 0;
    void init(size_t bytes) { CK(gpuMalloc(&base, bytes)); cap = bytes; used = 0; }
    template <typename T = uint8_t> T *get(size_t n) {
        size_t bytes = (n * sizeof(T) + 255) & ~size_t(255);
        if (used + bytes > cap) FAIL("arena out of memory (%zu + %zu > %zu)", used, bytes, cap);
        T *p = (T *)(base + used); used += bytes; return p;
    }
};

#if defined(__HIP_PLATFORM_AMD__)
template <int M> __device__ __forceinline__ int lane_xor(int v) {
    static_assert(M == 1 || M == 2 || M == 4 || M == 8 || M == 16, "lane_xor: M");
    if constexpr (M == 1) return __builtin_amdgcn_update_dpp(0, v, 0xb1, 0xf, 0xf, true);
    else if constexpr (M == 2) return __builtin_amdgcn_update_dpp(0, v, 0x4e, 0xf, 0xf, true);
    else if constexpr (M == 4) return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xf, 0xf, true);
    else if constexpr (M == 8) return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xf, 0xf, true);
    else return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, true, false);
}
template <int M> __device__ __forceinline__ float lane_xor(float v) { return __int_as_float(lane_xor<M>(__float_as_int(v))); }
#else
template <int M> __device__ __forceinline__ int lane_xor(int v) { return __shfl_xor_sync(0xffffffffu, v, M); }
template <int M> __device__ __forceinline__ float lane_xor(float v) { return __shfl_xor_sync(0xffffffffu, v, M); }
#endif

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
#if !defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ void quant_put(QAct o, size_t i, float v) {
    const float d = warp_max(fabsf(v)) / 127.0f;
    const float q = d > 0 ? rintf(v / d) : 0.0f;
    o.q[i] = (int8_t)q;
    if (o.r) o.r[i] = (int8_t)(d > 0 ? fminf(127.0f, fmaxf(-127.0f, rintf((v - q * d) * (254.0f / d)))) : 0.0f);
    if ((threadIdx.x & 31) == 0) o.d[i / 32] = d;
}
#endif
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ int dot4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
#else
__device__ __forceinline__ int dot4(int a, int b, int c) { return __dp4a(a, b, c); }

__device__ __forceinline__ void mma_s8(int c[4], const int a[4], const int b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_s8_k16(int c[4], const int a[2], int b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b));
}
__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *)&h;
}
#endif
