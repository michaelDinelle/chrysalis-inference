// The CPU's share of a decode step in isolation: one layer handoff (T tokens, P cold picks) through CpuMoe's decode path,
// on a synthetic cold tier larger than the L3 (Q3_K gate/up, IQ4_NL down), with the picks' weights in DRAM or already
// fetched into the caches by the workers' prefetch (what a lookahead could do while the GPU runs the token mixer).
//   build_fn/bench_cpudec_fn [picks] [tokens]
#include "../engine_fn/cpu_moe.h"
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/mman.h>
#include <thread>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 9, T = argc > 2 ? atoi(argv[2]) : 3, NEXP = 900, W = 8;
    const size_t row_gu = 1100, row_dn = 360;                    // Q3_K row of 2560, IQ4_NL row of 640
    const size_t gu_bytes = (size_t)NEXP * 640 * row_gu, dn_bytes = (size_t)NEXP * 2560 * row_dn;
    auto alloc = [](size_t n) {
        auto *p = (uint8_t *)mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        std::mt19937 rng(7);
        for (size_t i = 0; i < n; i += 4) { const uint32_t r = rng() & 0x3b3b3b3b; memcpy(p + i, &r, std::min<size_t>(4, n - i)); }
        return p;
    };
    uint8_t *gate = alloc(gu_bytes), *up = alloc(gu_bytes), *down = alloc(dn_bytes);
    ColdLayer c;
    c.gate[0] = gate; c.up[0] = up; c.down = down; c.type_gu[0] = 11; c.type_dn = 20;
    c.row_gu[0] = row_gu; c.row_dn = row_dn; c.n0 = NEXP; c.n = NEXP;
    std::vector<ColdLayer> L(1, c);
    auto *sh = (MoeShared *)aligned_alloc(64, sizeof(MoeShared));
    memset(sh, 0, sizeof(MoeShared));
    CpuMoe cpu;
    cpu.start(sh, nullptr, L, W);
    cpu.arm();
    std::mt19937 rng(3);
    for (int i = 0; i < T * 2560; i++) sh->x[i] = std::uniform_real_distribution<float>(-1, 1)(rng);
    const double exp_mb = (2.0 * 640 * row_gu + 2560.0 * row_dn) / 1e6;
    printf("%d picks over %d tokens, %.2f MB per expert, %d workers\n", P, T, exp_mb, W);
    uint32_t gen = 0;
    for (int mode = 0; mode < 3; mode++) {
        std::vector<double> ts;
        for (int trial = 0; trial < 60; trial++) {
            std::vector<int> ex;
            while ((int)ex.size() < P) { const int e = rng() % NEXP; if (std::find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e); }
            for (int j = 0; j < T * MOE_MAX_E; j++) { sh->slot[j] = -1; sh->w[j] = 0.1f; }
            for (int p = 0; p < P; p++) sh->slot[(p % T) * MOE_MAX_E + p / T] = ex[p];
            sh->layer = 0; sh->T = T;
            if (mode > 0) {   // fetch the picks' rows into the caches first, as W workers would: prefetch (1) or loads (2)
                std::vector<std::thread> th;
                for (int w = 0; w < W; w++) th.emplace_back([&, w] {
                    volatile uint64_t sink = 0;
                    for (int p = 0; p < P; p++) {
                        const uint8_t *seg[3] = {c.gu_row(false, ex[p], 0), c.gu_row(true, ex[p], 0), c.dn_row(ex[p], 0)};
                        const size_t len[3] = {640 * row_gu, 640 * row_gu, 2560 * row_dn};
                        for (int k = 0; k < 3; k++)
                            for (size_t o = len[k] * w / W / 64 * 64; o < len[k] * (w + 1) / W; o += 64) {
                                if (mode == 1) _mm_prefetch((const char *)seg[k] + o, _MM_HINT_T1);
                                else sink += *(const volatile uint64_t *)(seg[k] + o);
                            }
                    }
                });
                for (auto &t : th) t.join();
            }
            const double t0 = now();
            __atomic_store_n(&sh->go, ++gen, __ATOMIC_RELEASE);
            while (__atomic_load_n(&sh->done, __ATOMIC_ACQUIRE) != gen) _mm_pause();
            ts.push_back(now() - t0);
        }
        std::sort(ts.begin(), ts.end());
        printf("%-22s median %6.1f us  p10 %6.1f  p90 %6.1f\n", mode == 0 ? "weights in DRAM" : mode == 1 ? "prefetched (T1)" : "loaded into caches",
               ts[ts.size() / 2] * 1e6, ts[ts.size() / 10] * 1e6, ts[ts.size() * 9 / 10] * 1e6);
    }
    cpu.stop();
}
