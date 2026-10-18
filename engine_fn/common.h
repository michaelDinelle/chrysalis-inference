#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "kiln: HIP error '%s' at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)
#define FAIL(...) do { fprintf(stderr, "kiln: " __VA_ARGS__); fprintf(stderr, "\n"); abort(); } while (0)

// Qwen3.8-Flash-Next (GGUF architecture qwen4_exp)
constexpr int D = 2560;            // hidden size (one hyper-connection stream)
constexpr int HC = 4;              // hyper-connection streams: the residual is HC x D wide
constexpr int HCD = HC * D;
constexpr int HCR = 320;           // hyper-connection bottleneck rank
constexpr int VOCAB = 248320;
constexpr int NL = 48;             // decoder layers; every 4th (l % 4 == 3) is QSA attention, the rest Gated DeltaNet
constexpr int NH = 24, NKV = 2, HD = 256, ROT = 64;  // attention: q heads, kv heads, head dim, rotary dims
constexpr int QPK = NH / NKV;                        // query heads per kv head (12)
constexpr int GK = 16, GV = 48, GD = 128;            // gated delta net: q/k heads, v heads, head dim
constexpr int QKV = 2 * GK * GD + GV * GD;           // 10240 conv channels (q | k | v)
constexpr int NGDN = 36;                             // Gated DeltaNet layers
constexpr int NATT = NL / 4;                         // 12 QSA layers
constexpr int NKVL = NATT + 1;                       // KV-cache layers: 12 + the MTP layer
constexpr int NE = 512, TOPK_E = 10, FE = 640;       // routed experts, experts per token, expert FFN width
constexpr int FS = 640;                              // shared expert FFN width
constexpr int IDX_H = 4, IDX_D = 128;                // QSA indexer: query heads, head dim (one shared key head)
constexpr int KPOOL = 4;                             // QSA pools keys over blocks of 4 cells
constexpr int SEL_BLOCKS = 512;                      // QSA attends the top 512 blocks (2048 cells) plus the tail
constexpr int PLE_LAYER = 1;                         // n-gram embedding layer
constexpr int PLE_HEADS = 16, PLE_DIM = 160;         // 8 bigram + 8 trigram rows of 160 per token -> 2560
constexpr int PLE_K = 4, PLE_DIL = 3;                // its depthwise conv: kernel 4, dilation 3 (taps t-9, t-6, t-3, t)
constexpr int PLE_HIST = (PLE_K - 1) * PLE_DIL;      // 9 positions of conv history
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 1e7f;
constexpr int EOS_IM_END = 248046, EOS_TEXT = 248044;   // <|im_end|>, <|endoftext|>
// Image tokens: IMG_BASE + picture * IMG_ROWS + row names row `row` of a picture's embeddings (as in engine/model.h); the
// n-gram block hashes them as the <|image_pad|> they stand for, as the original's ple_input_ids hold it.
constexpr int IMG_BASE = 1 << 30, IMG_ROWS = 1 << 14, IMAGE_PAD = 248056;
struct Pos3 { int t, h, w; };      // an M-RoPE position (text: all three equal)
inline bool is_attn_layer(int l) { return l % 4 == 3; }

// KV cache: page -> layer (12 QSA layers + MTP) -> kv head -> block of PAGE tokens, kiln's format:
//   K int8 [PAGE][HD] | K scales fp16 [PAGE][HD/32] |
//   V 5-bit, transposed: low nibbles [HD][PAGE/2] | high bits [HD][PAGE/8] | V scales fp16 [HD/32][PAGE]
// then, per layer, the QSA indexer cache: pooled keys fp16 [PAGE/KPOOL][IDX_D] (normed and rotated to the block's
// first cell). Raw indexer keys are only needed until their block of KPOOL cells is complete: they live in the
// sequence (idx_tail, the incomplete block) and in a per-pass buffer, not in the pages (22% less KV per token).
constexpr int PAGE = 64;
constexpr size_t KS_OFF = (size_t)PAGE * HD;
constexpr size_t VLO_OFF = KS_OFF + (size_t)PAGE * (HD / 32) * 2;
constexpr size_t VHI_OFF = VLO_OFF + (size_t)HD * PAGE / 2;
constexpr size_t VS_OFF = VHI_OFF + (size_t)HD * PAGE / 8;
constexpr size_t KV_HEAD = VS_OFF + (size_t)(HD / 32) * PAGE * 2;
constexpr size_t IDX_POOL = (size_t)(PAGE / KPOOL) * IDX_D * 2;
constexpr size_t KV_LAYER = KV_HEAD * NKV + IDX_POOL;
constexpr size_t IDX_TAIL = (size_t)KPOOL * IDX_D;     // halfs per layer: raw keys of a sequence's incomplete block
constexpr size_t KV_PAGE = KV_LAYER * NKVL;
struct KVRef { uint8_t *pool; const int *pages; };

// GGUF stores blocks as packed structs; at load time tensors in VRAM are split into separate arrays (same
// bits, "structure of arrays") so every array is aligned for wide loads. Cold experts stay in GGUF block
// order in host memory (the CPU's ggml kernels and the GPU's zero-copy prefill read them as they are).
enum QType : int { F32 = 0, F16 = 1, Q5_0 = 6, Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q6_K = 14, IQ4_NL = 20, IQ4_XS = 23 };

// Q2_K / Q3_K super-blocks (256 weights) in 16 groups of 16; group g = 8n + 2j + k holds weights 128n + 32j + 16k + l
// (l < 16), whose 2 low bits are bits 2j..2j+1 of byte 32n + 16k + l of the 64 quant bytes (and, Q3_K, its third bit is
// bit 4n + j of byte 16k + l of the 32 hmask bytes).
struct QTensor {
    int type = -1;
    int rows = 0, cols = 0;        // rows = output features, cols = input features (K)
    uint8_t *q = nullptr;          // packed quants   Q8_0: int8[rows*cols]  Q6_K, IQ4_XS, IQ4_NL, Q5_0: nibbles[rows*cols/2]
                                   //                 Q2_K, Q3_K: 2-bit [rows*cols/4]
    uint8_t *qh = nullptr;         // Q6_K high bits [rows*cols/4]   Q5_0 fifth bits [rows*cols/8] (one u32 per block)
                                   // Q3_K hmask [rows*cols/8]
    uint8_t *sc = nullptr;         // Q6_K int8 scales [rows*cols/16]   IQ4_XS scales_l [rows*cols/64]
                                   // Q3_K int8 scales (unpacked, -32 applied) [rows*cols/16]   Q2_K scale | min << 4 [rows*cols/16]
    uint16_t *sh = nullptr;        // IQ4_XS scales_h [rows*cols/256]
    half *d = nullptr;             // Q8_0, Q5_0, IQ4_NL: [rows*cols/32]   Q6_K, IQ4_XS, Q2_K, Q3_K: [rows*cols/256]
    half *dm = nullptr;            // Q2_K min scales [rows*cols/256]
    void *f = nullptr;             // F32/F16 data
};
// GGUF block size in weights and bytes
__host__ __device__ constexpr int qk_of(int type) { return (type == Q8_0 || type == Q5_0 || type == IQ4_NL) ? 32 : type == F32 || type == F16 ? 1 : 256; }
__host__ __device__ constexpr int qbytes_of(int type) {
    return type == Q8_0 ? 34 : type == Q5_0 ? 22 : type == IQ4_NL ? 18 : type == Q2_K ? 84 : type == Q3_K ? 110 : type == Q6_K ? 210
         : type == IQ4_XS ? 136 : type == F32 ? 4 : type == F16 ? 2 : 0;
}
inline size_t qrow_bytes(int type, int cols) { return (size_t)cols / qk_of(type) * qbytes_of(type); }

// The expert pool in VRAM: one IQ4_XS gate, one IQ4_XS up and one IQ4_NL down tensor for the 48 layers' VRAM experts, slot
// s owning rows s * FE.. of gate and up and s * D.. of down. A slot may hold a RAM-tier expert in its own format instead:
// its gate (or up) is then that format's arrays for FE rows (as make_qtensor carves them) inside the slot's share of the
// IQ4_XS nibble array, FE * D / 2 bytes, which Q3_K (729,600 bytes) and Q2_K (537,600) fit; its down is IQ4_NL like any
// other. slot_view: slot s's gate or up (pool = the pool's gate or up tensor) as a tensor of FE rows of `type`.
__host__ __device__ inline QTensor slot_view(const QTensor &pool, int s, int type) {
    constexpr size_t n = (size_t)FE * D;
    QTensor t;
    t.type = type; t.rows = FE; t.cols = D;
    uint8_t *base = pool.q + (size_t)s * (n / 2);
    if (type == Q3_K) {
        t.q = base; t.qh = base + n / 4; t.sc = t.qh + n / 8; t.d = (half *)(t.sc + n / 16);
    } else if (type == Q2_K) {
        t.q = base; t.sc = base + n / 4; t.d = (half *)(t.sc + n / 16); t.dm = t.d + n / 256;
    } else {
        t.q = base; t.sc = pool.sc + (size_t)s * (n / 64); t.sh = pool.sh + (size_t)s * (n / 256); t.d = pool.d + (size_t)s * (n / 256);
    }
    return t;
}

// Activations for the int8 dot-product GEMVs: per 32 values one fp32 scale.
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

// The value of lane (lane ^ M), M = 1, 2, 4, 8 or 16, as one VALU instruction (DPP quad_perm / row_xmask, or
// v_permlanex16 across the wave's halves) instead of __shfl_xor's ds_bpermute, an LDS round trip: reductions and top-k
// selections are chains of these. All lanes of the wave must be active.
template <int M> __device__ __forceinline__ int lane_xor(int v) {
    static_assert(M == 1 || M == 2 || M == 4 || M == 8 || M == 16, "lane_xor: M");
    if constexpr (M == 1) return __builtin_amdgcn_update_dpp(0, v, 0xb1, 0xf, 0xf, true);     // quad_perm [1, 0, 3, 2]
    else if constexpr (M == 2) return __builtin_amdgcn_update_dpp(0, v, 0x4e, 0xf, 0xf, true);   // quad_perm [2, 3, 0, 1]
    else if constexpr (M == 4) return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xf, 0xf, true);  // row_xmask:4
    else if constexpr (M == 8) return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xf, 0xf, true);  // row_xmask:8
    else return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, true, false);      // lane i of the other half
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
// Wave-wide (max value, lowest index among equals): every lane ends with the winner. Branch-free (a branchy compare
// makes each step a cascade of exec-mask updates).
__device__ __forceinline__ void wave_argmax(float &b, int &bi) {
#define KILN_ARGMAX_STEP(M) { const float ov = lane_xor<M>(b); const int oi = lane_xor<M>(bi); \
                              const bool tk = (ov > b) | ((ov == b) & (oi < bi)); b = tk ? ov : b; bi = tk ? oi : bi; }
    KILN_ARGMAX_STEP(16) KILN_ARGMAX_STEP(8) KILN_ARGMAX_STEP(4) KILN_ARGMAX_STEP(2) KILN_ARGMAX_STEP(1)
#undef KILN_ARGMAX_STEP
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
// RDNA3 v_dot4_i32_iu8 with both operands signed
__device__ __forceinline__ int dot4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }
// WMMA operands (wave32, 16x16x16): A/B fragments hold 16 halves per lane, the accumulator 8 floats.
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
