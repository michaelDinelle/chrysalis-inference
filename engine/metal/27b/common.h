#pragma once
#include "mtl.h"
#include <cstdint>

using half_t = __fp16;

constexpr int D = 5120;
constexpr int FF = 17408;
constexpr int VOCAB = 248320;
constexpr int NL = 64;
constexpr int NH = 24, NKV = 4, HD = 256, ROT = 64;
constexpr int GK = 16, GV = 48, GD = 128;
constexpr int QKV = 2 * GK * GD + GV * GD;
constexpr int NATT = NL / 4;
constexpr int NKVL = NATT + 1;
inline bool is_attn_layer(int l) { return l % 4 == 3; }

constexpr int PAGE = 64;
constexpr size_t KV_V = (size_t)PAGE * HD;
constexpr size_t KV_KS = 2 * KV_V;
constexpr size_t KV_VS = KV_KS + (size_t)PAGE * (HD / 32) * 2;
constexpr size_t KV_HEAD = KV_VS + (size_t)PAGE * (HD / 32) * 2;
constexpr size_t KV_LAYER = KV_HEAD * NKV;
constexpr size_t KV_PAGE = KV_LAYER * NKVL;

enum QType : int { F32 = 0, F16 = 1, Q8_0 = 8, Q6_K = 14, IQ4_XS = 23 };

struct QTensor {
    int type = -1;
    int rows = 0, cols = 0;
    DP<uint8_t> q, qh, sc, d, f;
};

struct Arena {
    DP<uint8_t> base;
    size_t used = 0, cap = 0;
    template <typename T = uint8_t> DP<T> get(size_t n) {
        const size_t bytes = (n * sizeof(T) + 255) & ~size_t(255);
        if (used + bytes > cap) FAIL("arena out of memory (%zu + %zu > %zu)", used, bytes, cap);
        DP<T> p = (base + used).template as<T>();
        used += bytes;
        return p;
    }
};

inline QTensor make_qtensor(Arena &a, int type, int rows, int cols) {
    QTensor t; t.type = type; t.rows = rows; t.cols = cols;
    const size_t n = (size_t)rows * cols;
    switch (type) {
        case F32: t.f = a.get<float>(n).as<uint8_t>(); break;
        case Q8_0: t.q = a.get(n); t.d = a.get<uint16_t>(n / 32).as<uint8_t>(); break;
        case Q6_K: t.q = a.get(n / 2); t.qh = a.get(n / 4); t.sc = a.get(n / 16); t.d = a.get<uint16_t>(n / 256).as<uint8_t>(); break;
        case IQ4_XS: t.q = a.get(n / 2); t.sc = a.get(n / 64); t.qh = a.get(n / 128); t.d = a.get<uint16_t>(n / 256).as<uint8_t>(); break;
        default: FAIL("unsupported tensor type %d", type);
    }
    return t;
}
