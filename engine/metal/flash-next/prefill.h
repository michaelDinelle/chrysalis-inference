#pragma once
#include "mtl.h"
#include <string>

struct PfMmArgs { uint32_t rows, cols, rb, T, ldx, ldy, add; float scale; };
struct PfSortArgs { uint32_t P, bt; };
struct PfMoeArgs { uint32_t layer, ldx, P, pad; };

inline const char *pf_tname(int t) {
    switch (t) {
        case 0: return "f32"; case 1: return "f16"; case 30: return "bf16"; case 42: return "q2_0"; case 2: return "q4_0";
        case 6: return "q5_0"; case 8: return "q8_0"; case 20: return "iq4_nl"; case 23: return "iq4_xs"; case 11: return "q3_k";
        case 12: return "q4_k"; case 13: return "q5_k"; case 14: return "q6_k";
    }
    FAIL("prefill: no kernels for ggml type %d", t);
}

constexpr int PF_BM = 64, PF_BT = 64;
inline void pf_mm(Enc &e, int type, View W, uint32_t rows, uint32_t cols, size_t rb, View X, uint32_t ldx, View Y, uint32_t ldy,
                  uint32_t T, bool add = false, float scale = 1.f) {
    if (cols % 32) FAIL("pf_mm: %u columns", cols);
    PfMmArgs a{rows, cols, (uint32_t)rb, T, ldx, ldy, add ? 1u : 0u, scale};
    e.k(std::string("pf_mm_") + pf_tname(type)).c(0, a).b(1, W).b(2, X).b(3, Y).grid((rows + PF_BM - 1) / PF_BM, (T + PF_BT - 1) / PF_BT, 1, 128);
}

struct PfMoeBufs { View offs, ntiles, rows, tiles, H, Yp; };
inline uint32_t pf_max_tiles(uint32_t P, int bt) { return P / bt + 513; }
inline int pf_moe_bt(uint32_t T) { return T * 10 / 512 >= 24 ? 32 : 16; }
inline size_t pf_moe_scratch(uint32_t P, int bt) {
    auto al = [](size_t n) { return (n + 255) & ~size_t(255); };
    return al(513 * 4) + al(4) + al((size_t)P * 4) + al((size_t)pf_max_tiles(P, bt) * 8) + al((size_t)P * 640 * 4) + al((size_t)P * 2560 * 4);
}
inline PfMoeBufs pf_moe_bufs(View base, uint32_t P, int bt) {
    auto al = [](size_t n) { return (n + 255) & ~size_t(255); };
    PfMoeBufs s;
    size_t o = 0;
    for (auto [v, n] : {std::pair<View *, size_t>{&s.offs, 513 * 4}, {&s.ntiles, 4}, {&s.rows, (size_t)P * 4},
                        {&s.tiles, (size_t)pf_max_tiles(P, bt) * 8}, {&s.H, (size_t)P * 640 * 4}, {&s.Yp, (size_t)P * 2560 * 4}}) {
        *v = base.at(o);
        o += al(n);
    }
    return s;
}
inline void pf_moe(Enc &e, View table, View fmt, int layer, View ids, View wts, View X, uint32_t ldx, uint32_t T, const PfMoeBufs &s, View Y,
                   int bt = 0, bool wsum = true) {
    if (!bt) bt = pf_moe_bt(T);
    const uint32_t P = T * 10, mt = pf_max_tiles(P, bt);
    const PfSortArgs sa{P, (uint32_t)bt};
    e.k("pf_moe_sort").c(0, sa).b(1, ids).b(2, s.offs).b(3, s.rows).b(4, s.tiles).b(5, s.ntiles).grid(1, 1024);
    e.barrier();
    const PfMoeArgs ma{(uint32_t)layer, ldx, P, 0};
    const std::string b = std::to_string(bt);
    e.k("pf_moe_gu" + b).c(0, ma).b(1, table).b(2, s.offs).b(3, s.rows).b(4, s.tiles).b(5, s.ntiles).b(6, X).b(7, s.H).b(8, fmt).grid(10, mt, 1, 128);
    e.barrier();
    e.k("pf_moe_down" + b).c(0, ma).b(1, table).b(2, s.offs).b(3, s.rows).b(4, s.tiles).b(5, s.ntiles).b(6, s.H).b(7, s.Yp).grid(20, mt, 1, 128);
    if (!wsum) return;
    e.barrier();
    e.k("pf_moe_wsum").c(0, T).b(1, s.Yp).b(2, wts).b(3, Y).grid((T * 640 + 255) / 256, 256);
}
