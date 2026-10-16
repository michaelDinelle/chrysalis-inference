// CPU cold-expert workers: see cpu_moe.h. Built with -march=native; the dot products are ggml's own kernels.
#include "cpu_moe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <immintrin.h>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
constexpr int D_ = 2560, FE_ = 640;
inline uint32_t load_acq(const uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void store_rel(uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
}

void CpuMoe::start(MoeShared *shared, const std::vector<ColdLayer> &layers, int threads) {
    ggml_cpu_init();
    sh = shared;
    L = layers;
    n = threads;
    int tgu = -1, tdn = -1;
    for (const ColdLayer &c : L) if (c.gate) { tgu = c.type_gu; tdn = c.type_dn; break; }
    if (tgu < 0) return;                                        // no cold experts at all
    const ggml_type dt_gu = ggml_get_type_traits_cpu((ggml_type)tgu)->vec_dot_type;
    const ggml_type dt_dn = ggml_get_type_traits_cpu((ggml_type)tdn)->vec_dot_type;
    xq_row = ggml_row_size(dt_gu, D_);
    hq_row = ggml_row_size(dt_dn, FE_);
    gu.resize((size_t)MOE_MAX_T * MOE_MAX_E * 2 * FE_);
    xq.resize(xq_row * MOE_MAX_T);
    hq.resize(hq_row * MOE_MAX_T * MOE_MAX_E);
    seen = load_acq(&sh->go);
    scr.assign(n, Scratch{});
    for (int i = 0; i < n; i++) th.emplace_back(&CpuMoe::worker, this, i);
}

void CpuMoe::stop() {
    if (th.empty()) return;
    quit = true;
    armed = 1;
    armed.notify_all();
    for (auto &t : th) t.join();
    th.clear();
    quit = false;
}

void CpuMoe::arm() { armed.store(1, std::memory_order_release); armed.notify_all(); }
void CpuMoe::disarm() { armed.store(0, std::memory_order_release); }

void CpuMoe::barrier() {                           // spin barrier for the n workers
    const uint32_t g = bar_gen.load(std::memory_order_acquire);
    if (bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
        bar_count.store(0, std::memory_order_relaxed);
        bar_gen.fetch_add(1, std::memory_order_release);
    } else {
        while (bar_gen.load(std::memory_order_acquire) == g) _mm_pause();
    }
}

void CpuMoe::submit(const MoeJob &j) {
    job = j;
    if (pxq.size() < xq_row * j.T) pxq.resize(xq_row * j.T);
    jtok.store(0, std::memory_order_relaxed); jexp.store(0, std::memory_order_relaxed); jfin.store(0, std::memory_order_relaxed);
    jgen.fetch_add(1, std::memory_order_release);
}

void CpuMoe::wait() {
    while (jfin.load(std::memory_order_acquire) < n) _mm_pause();
}

// One prefill job: quantize the chunk's inputs, then take experts one at a time in the host's order (largest first).
// An expert is computed whole by one worker (rows outer, its tokens inner, so each weight row is read once while it
// sits in L1), which keeps the results independent of how the experts land on the workers.
void CpuMoe::prefill_work(int id) {
    const MoeJob &J = job;
    const ColdLayer &c = L[J.layer];
    const auto *tr_gu = ggml_get_type_traits_cpu((ggml_type)c.type_gu);
    const auto *tr_dn = ggml_get_type_traits_cpu((ggml_type)c.type_dn);
    const auto from_gu = ggml_get_type_traits_cpu(tr_gu->vec_dot_type)->from_float;
    const auto from_dn = ggml_get_type_traits_cpu(tr_dn->vec_dot_type)->from_float;
    for (int t; (t = jtok.fetch_add(1, std::memory_order_relaxed)) < J.T;) from_gu(J.x + (size_t)t * D_, pxq.data() + xq_row * t, D_);
    barrier();
    Scratch &s = scr[id];
    for (int e; (e = jexp.fetch_add(1, std::memory_order_relaxed)) < J.n_exp;) {
        const int p0 = J.first[e], np = J.first[e + 1] - p0, slot = J.slot[e];
        if (s.gu.size() < (size_t)np * 2 * FE_) { s.gu.resize((size_t)np * 2 * FE_); s.y.resize((size_t)np * D_); s.hq.resize(hq_row * np); }
        for (int r = 0; r < 2 * FE_; r++) {
            const uint8_t *row = (r < FE_ ? c.gate : c.up) + ((size_t)slot * FE_ + r % FE_) * c.row_gu;
            for (int i = 0; i < np; i++)
                tr_gu->vec_dot(D_, &s.gu[(size_t)i * 2 * FE_ + r], 0, row, 0, pxq.data() + xq_row * J.tok[p0 + i], 0, 1);
        }
        for (int i = 0; i < np; i++) {
            float h[FE_];
            const float *g2 = &s.gu[(size_t)i * 2 * FE_];
            for (int k = 0; k < FE_; k++) { const float a = g2[k]; h[k] = a / (1.0f + expf(-a)) * g2[FE_ + k]; }
            from_dn(h, s.hq.data() + hq_row * i, FE_);
        }
        for (int r = 0; r < D_; r++) {
            const uint8_t *row = c.down + ((size_t)slot * D_ + r) * c.row_dn;
            for (int i = 0; i < np; i++) tr_dn->vec_dot(FE_, &s.y[(size_t)i * D_ + r], 0, row, 0, s.hq.data() + hq_row * i, 0, 1);
        }
        for (int i = 0; i < np; i++) {
            float *y = &s.y[(size_t)i * D_];
            const float w = J.w[p0 + i];
            for (int k = 0; k < D_; k++) y[k] *= w;
            ggml_cpu_fp32_to_fp16(y, (ggml_fp16_t *)J.out + (size_t)(p0 + i) * D_, D_);
        }
    }
    jfin.fetch_add(1, std::memory_order_release);
}

void CpuMoe::worker(int id) {
    uint32_t last = seen.load(), jlast = jgen.load();
    for (;;) {
        // wait for the next generation (GPU decode handoff) or host job: spin while armed, sleep otherwise
        uint32_t g;
        bool host_job = false;
        for (;;) {
            if (quit.load(std::memory_order_acquire)) return;
            const uint32_t j = jgen.load(std::memory_order_acquire);
            if (j != jlast) { jlast = j; host_job = true; break; }
            g = load_acq(&sh->go);
            if (g != last) break;
            if (!armed.load(std::memory_order_acquire)) { armed.wait(0, std::memory_order_acquire); continue; }
            _mm_pause();
        }
        if (host_job) { prefill_work(id); continue; }
        last = g;
        const int layer = sh->layer, T = sh->T;
        const ColdLayer &c = L[layer];
        // picks handled here
        int pick[MOE_MAX_T * MOE_MAX_E], np = 0;
        for (int j = 0; j < T * MOE_MAX_E; j++) if (sh->slot[j] >= 0) pick[np++] = j;
        const auto *tr_gu = ggml_get_type_traits_cpu((ggml_type)c.type_gu);
        const auto *tr_dn = ggml_get_type_traits_cpu((ggml_type)c.type_dn);
        if (np > 0) {
            // inputs to the gate/up dot type (one token per worker)
            if (id < T) ggml_get_type_traits_cpu(tr_gu->vec_dot_type)->from_float(sh->x + (size_t)id * D_, xq.data() + xq_row * id, D_);
            barrier();
            // gate | up rows of all picks, split evenly
            const int R1 = np * 2 * FE_;
            for (int r = id * R1 / n; r < (id + 1) * R1 / n; r++) {
                const int p = r / (2 * FE_), k = r % (2 * FE_), j = pick[p], slot = sh->slot[j];
                const uint8_t *row = (k < FE_ ? c.gate : c.up) + ((size_t)slot * FE_ + (k % FE_)) * c.row_gu;
                tr_gu->vec_dot(D_, &gu[(size_t)p * 2 * FE_ + k], 0, row, 0, xq.data() + xq_row * (j / MOE_MAX_E), 0, 1);
            }
            barrier();
            // silu(gate) * up, quantized for the down rows
            for (int p = id; p < np; p += n) {
                float h[FE_];
                const float *g2 = &gu[(size_t)p * 2 * FE_];
                for (int i = 0; i < FE_; i++) { const float a = g2[i]; h[i] = a / (1.0f + expf(-a)) * g2[FE_ + i]; }
                ggml_get_type_traits_cpu(tr_dn->vec_dot_type)->from_float(h, hq.data() + hq_row * p, FE_);
            }
            barrier();
        }
        // down rows: each worker owns a slice of the output rows for every token (no races, fixed order)
        for (int r = id * D_ / n; r < (id + 1) * D_ / n; r++) {
            float acc[MOE_MAX_T] = {};
            for (int p = 0; p < np; p++) {
                const int j = pick[p], slot = sh->slot[j];
                float v;
                tr_dn->vec_dot(FE_, &v, 0, c.down + ((size_t)slot * D_ + r) * c.row_dn, 0, hq.data() + hq_row * p, 0, 1);
                acc[j / MOE_MAX_E] += sh->w[j] * v;
            }
            for (int t = 0; t < T; t++) sh->y[(size_t)t * D_ + r] = acc[t];
        }
        barrier();
        if (id == 0) { std::atomic_thread_fence(std::memory_order_release); store_rel(&sh->done, g); }
    }
}
