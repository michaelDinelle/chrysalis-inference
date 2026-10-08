// Shared definitions for the kiln engine: model constants, quant formats, small device helpers.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "kiln: HIP error '%s' at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)
#define FAIL(...) do { fprintf(stderr, "kiln: " __VA_ARGS__); fprintf(stderr, "\n"); abort(); } while (0)

// ---- Qwen3.8-27B ----------------------------------------------------------------------------
constexpr int D = 5120;            // hidden size
constexpr int FF = 17408;          // FFN intermediate
constexpr int VOCAB = 248320;
constexpr int NL = 64;             // decoder layers (every 4th is full attention)
constexpr int NH = 24, NKV = 4, HD = 256, ROT = 64;  // full attention: q heads, kv heads, head dim, rotary dims
constexpr int GK = 16, GV = 48, GD = 128;            // gated delta net: q/k heads, v heads, head dim
constexpr int QKV = 2 * GK * GD + GV * GD;           // 10240 conv channels (q | k | v)
constexpr int NATT = NL / 4;                         // 16 full-attention layers
constexpr int NKVL = NATT + 1;                       // KV-cache layers: 16 + the MTP layer
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 1e7f;
inline bool is_attn_layer(int l) { return l % 4 == 3; }

// ---- paged KV cache -------------------------------------------------------------------------------
// page -> layer (16 attention layers + MTP) -> kv head -> block of PAGE tokens:
//   K int8 [PAGE][HD] | K scales fp16 [PAGE][HD/32] |
//   V 5-bit, transposed: low nibbles [HD][PAGE/2] | high bits [HD][PAGE/8] | V scales fp16 [HD/32][PAGE]
// One scale per token per 32 dims for both. V uses 5 bits (symmetric, stored +16) and is transposed so a
// lane reads 16 consecutive keys of one dimension, the operand layout of the P.V matrix multiply.
constexpr int PAGE = 64;                                   // tokens per page
constexpr size_t KS_OFF = (size_t)PAGE * HD;
constexpr size_t VLO_OFF = KS_OFF + (size_t)PAGE * (HD / 32) * 2;
constexpr size_t VHI_OFF = VLO_OFF + (size_t)HD * PAGE / 2;
constexpr size_t VS_OFF = VHI_OFF + (size_t)HD * PAGE / 8;
constexpr size_t KV_HEAD = VS_OFF + (size_t)(HD / 32) * PAGE * 2;
constexpr size_t KV_LAYER = KV_HEAD * NKV;
constexpr size_t KV_PAGE = KV_LAYER * NKVL;
struct KVRef { uint8_t *pool; const int *pages; };

// ---- quantized weight formats --------------------------------------------------------------
// GGUF stores blocks as packed structs; at load time we split each tensor into separate arrays
// (same bits, "structure of arrays") so every array is aligned and can be read with wide loads.
enum QType : int { F32 = 0, F16 = 1, Q8_0 = 8, Q6_K = 14, IQ4_XS = 23 };

struct QTensor {
    int type = -1;
    int rows = 0, cols = 0;        // rows = output features, cols = input features (K)
    uint8_t *q = nullptr;          // packed quants   Q8_0: int8[rows*cols]   Q6_K: ql[rows*cols/2]   IQ4_XS: nibbles[rows*cols/2]
    uint8_t *qh = nullptr;         // Q6_K high bits [rows*cols/4]
    uint8_t *sc = nullptr;         // Q6_K int8 scales [rows*cols/16]   IQ4_XS scales_l [rows*cols/64]
    uint16_t *sh = nullptr;        // IQ4_XS scales_h [rows*cols/256]
    half *d = nullptr;             // Q8_0: [rows*cols/32]   Q6_K, IQ4_XS: [rows*cols/256]
    void *f = nullptr;             // F32/F16 data
};

// Activations for the int8 dot-product GEMVs: per 32 values one fp32 scale.
struct QAct { int8_t *q; float *d; };

// Bump allocator over one device allocation.
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

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
// RDNA3 v_dot4_i32_iu8 with both operands signed
__device__ __forceinline__ int dot4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
// WMMA operands (wave32, 16x16x16): A/B fragments hold 16 halves per lane, the accumulator 8 floats.
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
