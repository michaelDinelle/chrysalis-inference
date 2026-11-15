#pragma once
#include <cstddef>
#include <cstdint>

struct ColdLayer {
    const uint8_t *gate[3] = {}, *up[3] = {}, *down = nullptr, *down2 = nullptr;
    const uint8_t *file_gate = nullptr, *file_up = nullptr, *file_down2 = nullptr;
    int type_gu[3] = {-1, -1, -1}, type_dn = -1;
    size_t row_gu[3] = {}, row_dn = 0;
    int n0 = 0, n = 0, n2 = 0;
    int grp(int s) const { return s < n0 ? 0 : s < n ? 1 : 2; }
    int idx(int s) const { return s < n0 ? s : s < n ? s - n0 : s - n; }
    const uint8_t *gu_row(bool is_up, int s, int r) const {
        const int g = grp(s);
        return (is_up ? up[g] : gate[g]) + ((size_t)idx(s) * 640 + r) * row_gu[g];
    }
    const uint8_t *dn_row(int s, int r) const {
        return s < n ? down + ((size_t)s * 2560 + r) * row_dn : down2 + ((size_t)(s - n) * 2560 + r) * row_dn;
    }
    size_t bytes(int s) const { return 2 * 640 * row_gu[grp(s)] + 2560 * row_dn; }
};

struct EmuAct {
    alignas(64) int8_t q[2560];
    alignas(64) float d[80];
    alignas(64) int32_t s16[160];
    alignas(64) int32_t c32[80];
};

struct ExactScratch {
    uint8_t *w = nullptr; float *ws = nullptr, *wa = nullptr, *wb = nullptr;
    uint32_t *xi = nullptr; float *xs = nullptr; int32_t *xc = nullptr;
    float *xf = nullptr;
    float *h = nullptr; uint32_t *hi = nullptr; float *hs = nullptr; int32_t *hc = nullptr;
    float *y = nullptr;
    uint32_t *xa = nullptr, *ha = nullptr; int32_t *xb = nullptr, *hb = nullptr;
    void reserve();
    ~ExactScratch();
};

enum ExactPath { EXACT_AVX512VNNI, EXACT_AVXVNNI, EXACT_AVX512, EXACT_AVX2, EXACT_PATHS };

struct ExactKernels {
    void (*quant_x)(const float *x, int8_t *q, float *s, int32_t *sum);
    void (*expert)(const ColdLayer &c, int slot, int n, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
                   const float *w, uint16_t *const *out, ExactScratch &S);
    float (*dot_gu1)(const uint8_t *row, int type, const int8_t *xq, const float *xs, const int32_t *xsum, const float *s16,
                     const float *f16, ExactScratch &S);
    void (*dot_gu2)(const uint8_t *r0, const uint8_t *r1, const int8_t *xq, const float *xs, const float *xf, float *out);
    void (*quant_h)(const float *h, int8_t *q, float *s, float *c, int32_t *b4);
    float (*dot_dn1)(const uint8_t *row, const int8_t *hq, const float *hs, const float *hc, const int32_t *hb);
    void (*emu_quant_x)(const float *x, EmuAct &a);
    float (*emu_gu_row)(const uint8_t *row, int type, const EmuAct &a, ExactScratch &S);
    float (*emu_dn_row)(const uint8_t *row, const int8_t *hq, const float *hd, const int32_t *hc, ExactScratch &S);
};
extern const ExactKernels *exact_kernels;

const char *exact_path_name(int path);
bool exact_path_supported(int path);
int exact_path();
bool exact_use_path(int path);
inline bool exact_supported() { return exact_path() >= 0; }
bool exact_formats(const ColdLayer &c);

void exact_act1(const float *xs, const int32_t *xsum, float *s16, float *f16);
void exact_act_x(const int32_t *xsum, float *xf);
inline void exact_quant_x(const float *x, int8_t *q, float *s, int32_t *sum) { exact_kernels->quant_x(x, q, s, sum); }
inline void expert_exact(const ColdLayer &c, int slot, int n, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
                         const float *w, uint16_t *const *out, ExactScratch &S) {
    exact_kernels->expert(c, slot, n, xq, xs, xsum, w, out, S);
}
inline float exact_dot_gu1(const uint8_t *row, int type, const int8_t *xq, const float *xs, const int32_t *xsum, const float *s16,
                           const float *f16, ExactScratch &S) {
    return exact_kernels->dot_gu1(row, type, xq, xs, xsum, s16, f16, S);
}
inline void exact_dot_gu2(const uint8_t *r0, const uint8_t *r1, const int8_t *xq, const float *xs, const float *xf, float *out) {
    exact_kernels->dot_gu2(r0, r1, xq, xs, xf, out);
}
inline void exact_quant_h(const float *h, int8_t *q, float *s, float *c, int32_t *b4) { exact_kernels->quant_h(h, q, s, c, b4); }
inline float exact_dot_dn1(const uint8_t *row, const int8_t *hq, const float *hs, const float *hc, const int32_t *hb) {
    return exact_kernels->dot_dn1(row, hq, hs, hc, hb);
}
inline void emu_quant_x(const float *x, EmuAct &a) { exact_kernels->emu_quant_x(x, a); }
inline float emu_gu_row(const uint8_t *row, int type, const EmuAct &a, ExactScratch &S) { return exact_kernels->emu_gu_row(row, type, a, S); }
inline float emu_dn_row(const uint8_t *row, const int8_t *hq, const float *hd, const int32_t *hc, ExactScratch &S) {
    return exact_kernels->emu_dn_row(row, hq, hd, hc, S);
}
