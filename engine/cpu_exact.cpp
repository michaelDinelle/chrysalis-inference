#include "cpu_exact.h"
#include "os.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int D_ = 2560, FE_ = 640;
constexpr int NB_GU = D_ / 32, NB_DN = FE_ / 32;
alignas(16) const uint8_t KV128[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};

namespace avx512vnni {
#define TGT __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c,avx2,fma,bmi2")))
#define EXACT_WIDE 1
#define EXACT_DOT 1
#include "cpu_exact_isa.h"
#undef EXACT_DOT
#undef EXACT_WIDE
#undef TGT
}

namespace avxvnni {
#define TGT __attribute__((target("avxvnni,f16c,avx2,fma,bmi2")))
#define EXACT_WIDE 0
#define EXACT_DOT 2
#include "cpu_exact_isa.h"
#undef EXACT_DOT
#undef EXACT_WIDE
#undef TGT
}

namespace vnni256 {
#define TGT __attribute__((target("avx512f,avx512vl,avx512vnni,f16c,avx2,fma,bmi2")))
#define EXACT_WIDE 0
#define EXACT_DOT 1
#include "cpu_exact_isa.h"
#undef EXACT_DOT
#undef EXACT_WIDE
#undef TGT
}

namespace avx512 {
#define TGT __attribute__((target("avx512f,avx512bw,avx512vl,f16c,avx2,fma,bmi2")))
#define EXACT_WIDE 1
#define EXACT_DOT 0
#include "cpu_exact_isa.h"
#undef EXACT_DOT
#undef EXACT_WIDE
#undef TGT
}

namespace avx2 {
#define TGT __attribute__((target("f16c,avx2,fma")))
#define EXACT_WIDE 0
#define EXACT_DOT 0
#include "cpu_exact_isa.h"
#undef EXACT_DOT
#undef EXACT_WIDE
#undef TGT
}

bool cpu_has(int path) {
#if __has_builtin(__builtin_cpu_init)
    __builtin_cpu_init();
#endif
    const bool base = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    const bool bmi2 = __builtin_cpu_supports("bmi2");
    const bool z = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl");
    const bool zv = z && __builtin_cpu_supports("avx512vnni");
    switch (path) {
    case EXACT_AVX512VNNI: return base && bmi2 && zv;
    case EXACT_AVXVNNI: return base && bmi2 && (__builtin_cpu_supports("avxvnni") || zv);
    case EXACT_AVX512: return base && bmi2 && z;
    case EXACT_AVX2: return base;
    }
    return false;
}

const ExactKernels *table(int path) {
    switch (path) {
    case EXACT_AVX512VNNI: return &avx512vnni::kernels;
    case EXACT_AVXVNNI: return __builtin_cpu_supports("avxvnni") ? &avxvnni::kernels : &vnni256::kernels;
    case EXACT_AVX512: return &avx512::kernels;
    case EXACT_AVX2: return &avx2::kernels;
    }
    return nullptr;
}

int best_path() {
    for (int p = 0; p < EXACT_PATHS; p++)
        if (cpu_has(p)) return p;
    return -1;
}

int active = best_path();
}

const ExactKernels *exact_kernels = active < 0 ? nullptr : table(active);

const char *exact_path_name(int path) {
    static const char *const names[EXACT_PATHS] = {"AVX-512 VNNI", "AVX-VNNI", "AVX-512", "AVX2"};
    if (path == EXACT_AVXVNNI && !__builtin_cpu_supports("avxvnni")) return "AVX-VNNI EVEX";
    return path >= 0 && path < EXACT_PATHS ? names[path] : "none";
}
bool exact_path_supported(int path) { return path >= 0 && path < EXACT_PATHS && cpu_has(path); }
int exact_path() { return active; }
bool exact_use_path(int path) {
    if (!exact_path_supported(path)) return false;
    exact_kernels = table(path);
    active = path;
    return true;
}

bool exact_formats(const ColdLayer &c) {
    for (int g = 0; g < 3; g++)
        if (c.gate[g] && c.type_gu[g] != 23 && c.type_gu[g] != 11 && c.type_gu[g] != 10) return false;
    return c.type_dn == 20;
}

void exact_act1(const float *xs, const int32_t *xsum, float *s16, float *f16) {
    for (int g = 0; g < D_ / 16; g++) { s16[g] = xs[g / 2]; f16[g] = xs[g / 2] * (float)xsum[g]; }
}
void exact_act_x(const int32_t *xsum, float *xf) {
    for (int b = 0; b < NB_GU; b++) xf[b] = (float)(xsum[2 * b] + xsum[2 * b + 1]);
}

void ExactScratch::reserve() {
    if (w) return;
    auto get = [](size_t bytes) { return os::aligned(64, (bytes + 63) / 64 * 64); };
    w = (uint8_t *)get(D_); ws = (float *)get(NB_GU * 4); wa = (float *)get(NB_GU * 2 * 4); wb = (float *)get(NB_GU * 2 * 4);
    xi = (uint32_t *)get(NB_GU * 8 * 16 * 4); xs = (float *)get(NB_GU * 16 * 4); xc = (int32_t *)get(NB_GU * 16 * 4);
    xf = (float *)get(NB_GU * 2 * 16 * 4);
    h = (float *)get(FE_ * 16 * 4); hi = (uint32_t *)get(NB_DN * 8 * 16 * 4); hs = (float *)get(NB_DN * 16 * 4); hc = (int32_t *)get(NB_DN * 16 * 4);
    y = (float *)get((size_t)D_ * 16 * 4);
    xa = (uint32_t *)get(NB_GU * 8 * 16 * 4); xb = (int32_t *)get(NB_GU * 16 * 4);
    ha = (uint32_t *)get(NB_DN * 8 * 16 * 4); hb = (int32_t *)get(NB_DN * 16 * 4);
}
ExactScratch::~ExactScratch() {
    for (void *p : {(void *)w, (void *)ws, (void *)wa, (void *)wb, (void *)xi, (void *)xs, (void *)xc, (void *)xf, (void *)h, (void *)hi,
                    (void *)hs, (void *)hc, (void *)y, (void *)xa, (void *)xb, (void *)ha, (void *)hb}) os::aligned_free(p);
}
