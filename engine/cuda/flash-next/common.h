#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) fail("CUDA error '%s' at %s:%d", cudaGetErrorString(e_), __FILE__, __LINE__); } while (0)
#define FAIL(...) fail(__VA_ARGS__)

constexpr int D = 2560;
constexpr int HC = 4;
constexpr int HCD = HC * D;
constexpr int HCR = 320;
constexpr int VOCAB = 248320;
constexpr int NL = 48;
constexpr int NH = 24, NKV = 2, HD = 256, ROT = 64;
constexpr int QPK = NH / NKV;
constexpr int GK = 16, GV = 48, GD = 128;
constexpr bool SIGMOID_GATE = true;
constexpr int QKV = 2 * GK * GD + GV * GD;
constexpr int NGDN = 36;
constexpr int NATT = NL / 4;
constexpr int NKVL = NATT + 1;
constexpr int NE = 512, TOPK_E = 10, FE = 640;
constexpr int FS = 640;
constexpr int IDX_H = 4, IDX_D = 128;
constexpr int KPOOL = 4;
constexpr int SEL_BLOCKS = 512;
constexpr int PLE_LAYER = 1;
constexpr int PLE_HEADS = 16, PLE_DIM = 160;
constexpr int PLE_K = 4, PLE_DIL = 3;
constexpr int PLE_HIST = (PLE_K - 1) * PLE_DIL;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 1e7f;
constexpr int EOS_IM_END = 248046, EOS_TEXT = 248044;
constexpr int IMG_BASE = 1 << 30, IMG_ROWS = 1 << 14, IMAGE_PAD = 248056;
struct Pos3 { int t, h, w; };
inline bool is_attn_layer(int l) { return l % 4 == 3; }

constexpr int PAGE = 64;
constexpr size_t KS_OFF = (size_t)PAGE * HD;
constexpr size_t VLO_OFF = KS_OFF + (size_t)PAGE * (HD / 32) * 2;
constexpr size_t VHI_OFF = VLO_OFF + (size_t)HD * PAGE / 2;
constexpr size_t VS_OFF = VHI_OFF + (size_t)HD * PAGE / 8;
constexpr size_t KV_HEAD = VS_OFF + (size_t)(HD / 32) * PAGE * 2;
constexpr size_t IDX_POOL = (size_t)(PAGE / KPOOL) * IDX_D * 2;
constexpr size_t KV_LAYER = KV_HEAD * NKV + IDX_POOL;
constexpr size_t IDX_TAIL = (size_t)KPOOL * IDX_D;
constexpr size_t KV_PAGE = KV_LAYER * NKVL;
struct KVRef { uint8_t *pool; const int *pages; };

enum QType : int { F32 = 0, F16 = 1, Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q6_K = 14, IQ4_NL = 20, IQ4_XS = 23 };

struct QTensor {
    int type = -1;
    int rows = 0, cols = 0;
    uint8_t *q = nullptr;
    uint8_t *qh = nullptr;
    uint8_t *sc = nullptr;
    uint16_t *sh = nullptr;
    half *d = nullptr;
    half *dm = nullptr;
    void *f = nullptr;
};
__host__ __device__ constexpr int qk_of(int type) { return (type == Q8_0 || type == IQ4_NL) ? 32 : type == F32 || type == F16 ? 1 : 256; }
__host__ __device__ constexpr int qbytes_of(int type) {
    return type == Q8_0 ? 34 : type == IQ4_NL ? 18 : type == Q2_K ? 84 : type == Q3_K ? 110 : type == Q6_K ? 210
         : type == IQ4_XS ? 136 : type == F32 ? 4 : type == F16 ? 2 : 0;
}
inline size_t qrow_bytes(int type, int cols) { return (size_t)cols / qk_of(type) * qbytes_of(type); }

constexpr size_t XGU = (size_t)FE * D / 4 + (size_t)FE * D / 8 + (size_t)FE * D / 16 + (size_t)FE * D / 128;
constexpr size_t XDN = (size_t)FE * D / 2 + (size_t)FE * D / 16;
constexpr size_t XSB = 2 * XGU + XDN;
constexpr int MAX_EXTRA = 8192;

__host__ __device__ inline QTensor slot_view(const QTensor &pool, int s, int type) {
    constexpr size_t n = (size_t)FE * D;
    QTensor t;
    t.type = type; t.rows = FE; t.cols = D;
    const int nb = pool.rows / FE;
    uint8_t *base = s < nb ? pool.q + (size_t)s * (n / 2) : (uint8_t *)pool.f + (size_t)(s - nb) * XSB;
    if (type == Q3_K) {
        t.q = base; t.qh = base + n / 4; t.sc = t.qh + n / 8; t.d = (half *)(t.sc + n / 16);
    } else if (type == Q2_K) {
        t.q = base; t.sc = base + n / 4; t.d = (half *)(t.sc + n / 16); t.dm = t.d + n / 256;
    } else {
        t.q = base; t.sc = pool.sc + (size_t)s * (n / 64); t.sh = pool.sh + (size_t)s * (n / 256); t.d = pool.d + (size_t)s * (n / 256);
    }
    return t;
}

__host__ __device__ inline QTensor down_view(const QTensor &pool, int s, int &row0) {
    const int nb = pool.rows / D;
    if (s < nb || !pool.f) { row0 = s * D; return pool; }
    QTensor t = pool;
    uint8_t *base = (uint8_t *)pool.f + (size_t)(s - nb) * XSB;
    t.q = base; t.d = (half *)(base + (size_t)FE * D / 2); t.rows = D;
    row0 = 0;
    return t;
}

struct QAct { int8_t *q; float *d; int8_t *r = nullptr; };

template <typename T> void zalloc(T **p, size_t bytes) {
    CK(cudaMalloc((void **)p, bytes));
    CK(cudaMemset(*p, 0, bytes));
}

struct Arena {
    uint8_t *base = nullptr;
    size_t used = 0, cap = 0;
    void init(size_t bytes) { zalloc(&base, bytes); cap = bytes; used = 0; }
    template <typename T = uint8_t> T *get(size_t n) {
        size_t bytes = (n * sizeof(T) + 255) & ~size_t(255);
        if (used + bytes > cap) FAIL("arena out of memory (%zu + %zu > %zu)", used, bytes, cap);
        T *p = (T *)(base + used); used += bytes; return p;
    }
};

template <int M> __device__ __forceinline__ int lane_xor(int v) { return __shfl_xor_sync(0xffffffffu, v, M); }
template <int M> __device__ __forceinline__ float lane_xor(float v) { return __shfl_xor_sync(0xffffffffu, v, M); }

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
__device__ __forceinline__ int dot4(int a, int b, int c) { return __dp4a(a, b, c); }
__device__ __forceinline__ void quant_put(QAct o, size_t i, float v) {
    const float d = warp_max(fabsf(v)) / 127.0f;
    const float q = d > 0 ? rintf(v / d) : 0.0f;
    o.q[i] = (int8_t)q;
    if (o.r) o.r[i] = (int8_t)(d > 0 ? fminf(127.0f, fmaxf(-127.0f, rintf((v - q * d) * (254.0f / d)))) : 0.0f);
    if ((threadIdx.x & 31) == 0) o.d[i / 32] = d;
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
__device__ __forceinline__ unsigned load_sys(const unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned load_gpu(const unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned load_relaxed_gpu(const unsigned *p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void store_sys(unsigned *p, unsigned v) { asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void store_gpu(unsigned *p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void store_relaxed_gpu(unsigned *p, unsigned v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
