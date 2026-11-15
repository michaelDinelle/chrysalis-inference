#pragma once
#include "../../cpu_exact.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

constexpr int MOE_MAX_T = 8;
constexpr int MOE_MAX_E = 10;
constexpr int EXACT_MIN_TOKENS = 3;

struct MoeShared {
    alignas(64) uint32_t go;
    alignas(64) uint32_t done;
    alignas(64) int layer, T;
    int slot[MOE_MAX_T * MOE_MAX_E];
    float w[MOE_MAX_T * MOE_MAX_E];
    alignas(64) float x[MOE_MAX_T * 2560];
    alignas(64) float y[MOE_MAX_T * 2560];
};

struct MoeEmu {
    alignas(64) uint32_t ready;
    alignas(64) uint32_t flag[MOE_MAX_T * MOE_MAX_E][20];
    alignas(64) float gu[MOE_MAX_T * MOE_MAX_E][2 * 640];
    alignas(64) float y[MOE_MAX_T * MOE_MAX_E][2560];
    alignas(64) int8_t hq[MOE_MAX_T * MOE_MAX_E][640];
    float hd[MOE_MAX_T * MOE_MAX_E][20];
};

struct MoeJob {
    int layer = 0, T = 0, n_exp = 0;
    const uint16_t *x = nullptr;
    const int *slot = nullptr;
    const int *first = nullptr;
    const int *tok = nullptr;
    const float *w = nullptr;
    uint16_t *out = nullptr;
};

class CpuMoe {
public:
    void start(MoeShared *shared, MoeShared *pred, const std::vector<ColdLayer> &layers, int threads, MoeEmu *emu = nullptr);
    bool can_emulate() const { return me && exact_ok; }
    void stop();
    void arm();
    void disarm();
    void submit(const MoeJob &job);
    void wait();
    bool use_exact(int on) { const bool was = exact; exact = on && exact_ok; return was; }
    ~CpuMoe() { stop(); }

private:
    void worker(int id);
    void prefill_work(int id);
    void preload(int id, uint32_t real, uint32_t job);
    MoeShared *sh = nullptr, *pr = nullptr;
    MoeEmu *me = nullptr;
    EmuAct ea[MOE_MAX_T];
    std::vector<ColdLayer> L;
    std::vector<std::thread> th;
    int n = 0;
    std::atomic<int> armed{0};
    std::atomic<uint32_t> wake{0};
    std::atomic<bool> quit{false};
    std::atomic<uint32_t> seen{0};
    std::atomic<int> bar_count{0};
    std::atomic<uint32_t> bar_gen{0};
    void barrier();
    std::vector<float> gu;
    std::vector<uint8_t> xq[3];
    std::vector<uint8_t> hq;
    std::vector<int8_t> dxq, dhq; std::vector<float> dxs, ds16, df16, dxf, dhs, dhc; std::vector<int32_t> dxsum;
    size_t xq_row[3] = {}, hq_row = 0;
    int dot_gu[3] = {-1, -1, -1};
    bool dot_same = true;
    MoeJob job;
    std::atomic<uint32_t> jgen{0};
    uint32_t jstart = 0;
    std::atomic<uint64_t> sink_{0};

    std::atomic<int> jtok{0}, jexp{0}, jfin{0};
    std::vector<uint8_t> pxq[3];
    std::vector<int8_t> exq; std::vector<float> exs; std::vector<int32_t> exsum;
    bool exact = false, exact_ok = false;
    struct Scratch { std::vector<float> gu, y; std::vector<uint8_t> hq; ExactScratch v; };
    std::vector<Scratch> scr;
};
