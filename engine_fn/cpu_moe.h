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
    // gate / up in up to two format groups (slots below n0, then the rest), down one tensor for all slots
    const uint8_t *gate[2] = {}, *up[2] = {}, *down = nullptr;
    int type_gu[2] = {-1, -1}, type_dn = -1;   // ggml types
    size_t row_gu[2] = {}, row_dn = 0;         // bytes per row
    int n0 = 0, n = 0;                 // slots in group 0, in all
    int grp(int s) const { return s >= n0; }
    const uint8_t *gu_row(bool is_up, int s, int r) const {   // row r (< 640) of expert s's gate or up matrix
        const int g = s >= n0;
        return (is_up ? up[g] : gate[g]) + ((size_t)(g ? s - n0 : s) * 640 + r) * row_gu[g];
    }
    const uint8_t *dn_row(int s, int r) const { return down + ((size_t)s * 2560 + r) * row_dn; }
};

struct MoeJob {                        // a prefill chunk's cold experts that the CPU computes (all host memory)
    int layer = 0, T = 0, n_exp = 0;
    const uint16_t *x = nullptr;       // [T][2560] MoE inputs of the pass (fp16)
    const int *slot = nullptr;         // [n_exp] cold slot of each expert, in the order the workers take them
    const int *first = nullptr;        // [n_exp + 1] each expert's range of picks in tok / w / out
    const int *tok = nullptr;          // [picks] token of each pick
    const float *w = nullptr;          // [picks] router weight of each pick
    uint16_t *out = nullptr;           // [picks][2560] fp16: w * expert(x[tok])
};

// Per-thread buffers of the VNNI expert kernel (cpu_vnni.cpp), 64-byte aligned, allocated on first use.
struct VnniScratch {
    uint8_t *w = nullptr; float *ws = nullptr, *wa = nullptr, *wb = nullptr;   // one decoded weight row: unsigned values,
                                                                       // per-32 scales (IQ4), per-16 scales and offsets (K-quants)
    uint32_t *xi = nullptr; float *xs = nullptr; int32_t *xc = nullptr;   // a group's gate/up activations, interleaved
    float *xf = nullptr;                                               // their per-16 sums times the block scale
    float *h = nullptr; uint32_t *hi = nullptr; float *hs = nullptr; int32_t *hc = nullptr;   // silu(g) * u, then int8
    float *y = nullptr;                                                // down rows per lane
    void reserve();
    ~VnniScratch();
};
bool vnni_supported();
bool vnni_formats(const ColdLayer &c);                                  // the kernel handles this layer's formats
void vnni_quant_x(const float *x, int8_t *q, float *s, int32_t *sum);   // one token: int8 per 32, scales, sums per 16
void expert_vnni(const ColdLayer &c, int slot, int n, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
                 const float *w, uint16_t *const *out, VnniScratch &S);

class CpuMoe {
public:
    // shared: device-visible host memory (hipHostMalloc coherent); threads: worker count (not pinned: the scheduler
    // moves a worker off a core that another process is using)
    void start(MoeShared *shared, const std::vector<ColdLayer> &layers, int threads);
    void stop();
    // workers spin on `go` only between arm() and disarm() (decode: the GPU hands over a layer's cold picks every few
    // hundred microseconds), and sleep otherwise until a prefill job arrives
    void arm();
    void disarm();
    // prefill: start a job (workers must be armed) and wait for it; the caller enqueues GPU work in between
    void submit(const MoeJob &job);
    void wait();
    bool use_vnni(int on) { const bool was = vnni; vnni = on && vnni_ok; return was; }   // experiments
    ~CpuMoe() { stop(); }

private:
    void worker(int id);
    void prefill_work(int id);
    MoeShared *sh = nullptr;
    std::vector<ColdLayer> L;
    std::vector<std::thread> th;
    int n = 0;
    std::atomic<int> armed{0};
    std::atomic<uint32_t> wake{0};               // bumped by arm(), submit() and stop(): sleeping workers wait on it
    std::atomic<bool> quit{false};
    std::atomic<uint32_t> seen{0};               // last generation the workers finished
    std::atomic<int> bar_count{0};
    std::atomic<uint32_t> bar_gen{0};
    void barrier();
    // scratch shared by the workers within one handoff
    std::vector<float> gu;                       // [picks][2 * 640] gate | up results
    // inputs quantized to each gate/up group's dot type (group 1 shares group 0's when the types agree)
    std::vector<uint8_t> xq[2];                  // [T]
    std::vector<uint8_t> hq;                     // [picks] silu(g) * u quantized to the down dot type
    size_t xq_row[2] = {}, hq_row = 0;
    int dot_gu[2] = {-1, -1};                    // vec_dot types of the gate/up groups
    bool dot_same = true;
    // prefill jobs
    MoeJob job;
    std::atomic<uint32_t> jgen{0};               // bumped by submit()
    std::atomic<int> jtok{0}, jexp{0}, jfin{0};  // next token to quantize, next expert, workers finished
    std::vector<uint8_t> pxq[2];                 // [T] the job's inputs quantized to each group's gate/up dot type
    std::vector<int8_t> vxq; std::vector<float> vxs; std::vector<int32_t> vxsum;   // [T] int8 per 32 + scales + sums per 16 (VNNI)
    bool vnni = false, vnni_ok = false;
    struct Scratch { std::vector<float> gu, y; std::vector<uint8_t> hq; VnniScratch v; };
    std::vector<Scratch> scr;                    // per worker
};
