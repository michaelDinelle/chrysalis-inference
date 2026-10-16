// CPU side of the routed experts: the cold experts (GGUF blocks in system RAM) are computed here with ggml's AVX-512
// kernels while the GPU runs the shared expert and the hot experts.
// Decode: per layer the GPU writes its tokens' MoE inputs and their cold expert picks into pinned host memory and
// bumps `go`; the workers answer with the weighted sum of those experts and bump `done`, which a GPU kernel spins on.
// Prefill: the host hands over a MoeJob (the chunk's cold experts that the CPU takes, with their tokens) and the
// workers write each pick's weighted expert output, in place of the GPU computing it from a copy in VRAM.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

constexpr int MOE_MAX_T = 8;           // tokens per handoff (decode and MTP verify passes)
constexpr int MOE_MAX_E = 10;          // routed experts per token

struct MoeShared {                     // pinned, coherent host memory shared with the GPU
    alignas(64) uint32_t go;           // generation written by the GPU when the inputs below are ready
    alignas(64) uint32_t done;         // generation written by the CPU when y is ready
    alignas(64) int layer, T;
    int slot[MOE_MAX_T * MOE_MAX_E];   // cold-tier slot of each pick, -1 for a hot pick (the GPU does those)
    float w[MOE_MAX_T * MOE_MAX_E];    // router weight of each pick
    alignas(64) float x[MOE_MAX_T * 2560];
    alignas(64) float y[MOE_MAX_T * 2560];
};

struct ColdLayer {                     // one layer's cold experts in system RAM, GGUF block order
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    int type_gu = -1, type_dn = -1;    // ggml types
    size_t row_gu = 0, row_dn = 0;     // bytes per row
};

struct MoeJob {                        // a prefill chunk's cold experts that the CPU computes (all host memory)
    int layer = 0, T = 0, n_exp = 0;
    const float *x = nullptr;          // [T][2560] MoE inputs of the chunk
    const int *slot = nullptr;         // [n_exp] cold slot of each expert, in the order the workers take them
    const int *first = nullptr;        // [n_exp + 1] each expert's range of picks in tok / w / out
    const int *tok = nullptr;          // [picks] token of each pick
    const float *w = nullptr;          // [picks] router weight of each pick
    uint16_t *out = nullptr;           // [picks][2560] fp16: w * expert(x[tok])
};

class CpuMoe {
public:
    // shared: device-visible host memory (hipHostMalloc coherent); threads: worker count (not pinned: the scheduler
    // moves a worker off a core that another process is using)
    void start(MoeShared *shared, const std::vector<ColdLayer> &layers, int threads);
    void stop();
    // workers spin on `go` only between arm() and disarm(), and sleep otherwise
    void arm();
    void disarm();
    // prefill: start a job (workers must be armed) and wait for it; the caller enqueues GPU work in between
    void submit(const MoeJob &job);
    void wait();
    ~CpuMoe() { stop(); }

private:
    void worker(int id);
    void prefill_work(int id);
    MoeShared *sh = nullptr;
    std::vector<ColdLayer> L;
    std::vector<std::thread> th;
    int n = 0;
    std::atomic<int> armed{0};
    std::atomic<bool> quit{false};
    std::atomic<uint32_t> seen{0};               // last generation the workers finished
    std::atomic<int> bar_count{0};
    std::atomic<uint32_t> bar_gen{0};
    void barrier();
    // scratch shared by the workers within one handoff
    std::vector<float> gu;                       // [picks][2 * 640] gate | up results
    std::vector<uint8_t> xq;                     // [T] inputs quantized to the gate/up dot type
    std::vector<uint8_t> hq;                     // [picks] silu(g) * u quantized to the down dot type
    size_t xq_row = 0, hq_row = 0;
    // prefill jobs
    MoeJob job;
    std::atomic<uint32_t> jgen{0};               // bumped by submit()
    std::atomic<int> jtok{0}, jexp{0}, jfin{0};  // next token to quantize, next expert, workers finished
    std::vector<uint8_t> pxq;                    // [T] the job's inputs quantized to the gate/up dot type
    struct Scratch { std::vector<float> gu, y; std::vector<uint8_t> hq; };
    std::vector<Scratch> scr;                    // per worker
};
