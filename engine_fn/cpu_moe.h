// CPU side of the routed experts at decode: the cold experts (GGUF blocks in system RAM) of every layer are
// computed here with ggml's AVX-512 kernels while the GPU runs the shared expert and the hot experts.
// Per layer the GPU writes its tokens' MoE inputs and their cold expert picks into pinned host memory and bumps
// `go`; the workers answer with the weighted sum of those experts and bump `done`, which a GPU kernel spins on.
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

class CpuMoe {
public:
    // shared: device-visible host memory (hipHostMalloc coherent); threads: worker count (pinned to cores 0..n-1)
    void start(MoeShared *shared, const std::vector<ColdLayer> &layers, int threads);
    void stop();
    // workers spin on `go` only between arm() and disarm(), and sleep otherwise
    void arm();
    void disarm();
    ~CpuMoe() { stop(); }

private:
    void worker(int id);
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
};
