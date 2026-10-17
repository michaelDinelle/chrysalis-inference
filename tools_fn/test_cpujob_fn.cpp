// The CPU prefill job (CpuMoe::submit/wait): (1) its per-pick outputs against a float reference built from ggml's
// dequantization, on 8 random cold experts in two gate/up format groups (Q3_K + Q2_K as in kiln-fn.gguf, and IQ4_XS),
// per path (VNNI kernel, ggml's dot products); (2) its speed on layer-sized work per format: 392 cold experts of random
// bytes (~0.9 GB, far past the L3), picks for T tokens, the time per layer and the rates it implies; decode handoffs.
// Build: make build_fn/test_cpujob_fn. Usage: test_cpujob_fn [threads]
#include "../engine_fn/cpu_moe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

constexpr int D = 2560, FE = 640;

struct Job {                           // a MoeJob and the arrays it points into
    std::vector<int> slot, first, tok; std::vector<float> w; std::vector<uint16_t> out, x16; MoeJob j;
};

// picks: T tokens x k distinct experts among n (uniform), grouped by expert as the engine does
static Job make_job(int T, int k, int n, std::mt19937 &rng, std::vector<float> &x) {
    std::vector<std::vector<int>> by(n);
    std::vector<int> e(n);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) e[i] = i;
        std::shuffle(e.begin(), e.end(), rng);
        for (int i = 0; i < k; i++) by[e[i]].push_back(t);
    }
    Job J;
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return by[a].size() > by[b].size(); });   // largest first
    J.first.push_back(0);
    std::uniform_real_distribution<float> u(0.05f, 0.3f);
    for (int s : order) {
        if (by[s].empty()) continue;
        J.slot.push_back(s);
        for (int t : by[s]) { J.tok.push_back(t); J.w.push_back(u(rng)); }
        J.first.push_back((int)J.tok.size());
    }
    J.out.resize(J.tok.size() * (size_t)D);
    J.x16.resize(x.size());                    // the job takes fp16 inputs: round x to them, for the reference too
    for (size_t i = 0; i < x.size(); i++) { J.x16[i] = ggml_fp32_to_fp16(x[i]); x[i] = ggml_fp16_to_fp32(J.x16[i]); }
    J.j.layer = 0; J.j.T = T; J.j.n_exp = (int)J.slot.size(); J.j.x = J.x16.data();
    J.j.slot = J.slot.data(); J.j.first = J.first.data(); J.j.tok = J.tok.data(); J.j.w = J.w.data(); J.j.out = J.out.data();
    return J;
}

int main(int argc, char **argv) {
    ggml_cpu_init();
    const int threads = argc > 1 ? atoi(argv[1]) : 8;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    MoeShared *sh = new MoeShared{};
    // ---- (1) correctness: 8 quantized random experts in two gate/up format groups (5 + 3), T = 24, 4 picks per token
    //      (experts with >= 3 tokens take the VNNI kernel, the others ggml's dot products) ----
    struct Fmt { ggml_type g0, g1; const char *name; };
    for (const Fmt f : {Fmt{GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, "Q3_K + Q2_K"}, Fmt{GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, "IQ4_XS"}}) {
        const int NC = 8, NC0 = 5, T = 24;
        std::normal_distribution<float> wd(0.f, 0.03f);
        auto quant = [&](ggml_type gt, int rows, int cols, std::vector<uint8_t> &raw, std::vector<float> &deq) {
            std::vector<float> w((size_t)rows * cols), im(cols, 1.0f);
            for (auto &v : w) v = wd(rng);
            const size_t rs = ggml_row_size(gt, cols);
            raw.resize(rs * rows);
            ggml_quantize_chunk(gt, w.data(), raw.data(), 0, rows, cols, im.data());
            deq.resize(w.size());
            for (int r = 0; r < rows; r++) ggml_get_type_traits(gt)->to_float(raw.data() + rs * r, deq.data() + (size_t)r * cols, cols);
        };
        std::vector<uint8_t> rg[2], ru[2], rd; std::vector<float> dg[2], du[2], ddn, fg, fu;
        for (int g = 0; g < 2; g++) {
            const ggml_type gt = g ? f.g1 : f.g0;
            const int ng = g ? NC - NC0 : NC0;
            quant(gt, ng * FE, D, rg[g], dg[g]); quant(gt, ng * FE, D, ru[g], du[g]);
            fg.insert(fg.end(), dg[g].begin(), dg[g].end()); fu.insert(fu.end(), du[g].begin(), du[g].end());
        }
        quant(GGML_TYPE_IQ4_NL, NC * D, FE, rd, ddn);
        ColdLayer cl;
        for (int g = 0; g < 2; g++) {
            cl.gate[g] = rg[g].data(); cl.up[g] = ru[g].data();
            cl.type_gu[g] = g ? f.g1 : f.g0; cl.row_gu[g] = ggml_row_size(g ? f.g1 : f.g0, D);
        }
        cl.down = rd.data(); cl.type_dn = GGML_TYPE_IQ4_NL; cl.row_dn = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
        cl.n0 = NC0; cl.n = NC;
        CpuMoe cpu; cpu.start(sh, {cl}, 8); cpu.arm();
        std::vector<float> x((size_t)T * D);
        for (auto &v : x) v = nd(rng);
        Job J = make_job(T, 4, NC, rng, x);
        cpu.submit(J.j); cpu.wait();
        double num[2] = {}, den[2] = {};                     // [ggml path, VNNI path]
        for (int e = 0; e < J.j.n_exp; e++) {
            const int path = J.first[e + 1] - J.first[e] >= 3;
            for (int p = J.first[e]; p < J.first[e + 1]; p++) {
                const int s = J.slot[e]; const float *xt = &x[(size_t)J.tok[p] * D];
                std::vector<double> h(FE);
                for (int r = 0; r < FE; r++) {
                    double g = 0, u = 0;
                    for (int k = 0; k < D; k++) { g += (double)fg[((size_t)s * FE + r) * D + k] * xt[k]; u += (double)fu[((size_t)s * FE + r) * D + k] * xt[k]; }
                    h[r] = g / (1 + exp(-g)) * u;
                }
                for (int r = 0; r < D; r++) {
                    double y = 0;
                    for (int k = 0; k < FE; k++) y += (double)ddn[((size_t)s * D + r) * FE + k] * h[k];
                    y *= J.w[p];
                    const double got = ggml_fp16_to_fp32(J.out[(size_t)p * D + r]);
                    num[path] += (got - y) * (got - y); den[path] += y * y;
                }
            }
        }
        for (int path = 0; path < 2; path++) {
            if (den[path] == 0) continue;
            const double r = sqrt(num[path] / den[path]);
            printf("CPU prefill job, %s (8 experts, T=24), %s path: rel diff %.2e %s\n", f.name, path ? "VNNI" : "ggml", r, r < 2e-2 ? "ok" : "FAIL");
            fails += r >= 2e-2;
        }
        cpu.disarm(); cpu.stop();
    }
    // ---- (2) speed on a layer's worth of cold experts (random bytes: the timing does not depend on the values), per
    //      gate/up format, down IQ4_NL ----
    for (const auto [gt, name, bb, doff] : {std::tuple{GGML_TYPE_Q3_K, "Q3_K", 110, 108}, std::tuple{GGML_TYPE_Q2_K, "Q2_K", 84, 80},
                                            std::tuple{GGML_TYPE_IQ4_XS, "IQ4_XS", 136, 0}}) {
        const int NC = 392;
        const size_t rgu = ggml_row_size(gt, D), rdn = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
        std::vector<uint8_t> g((size_t)NC * FE * rgu), u((size_t)NC * FE * rgu), dn((size_t)NC * D * rdn);
        std::uniform_int_distribution<int> byte(0, 255);
        for (auto *v : {&g, &u, &dn}) for (size_t i = 0; i < v->size(); i += 64) (*v)[i] = (uint8_t)byte(rng);
        // block scales: small finite fp16 values so no lane sees NaN / inf (Q2_K: d and dmin)
        auto fix = [](std::vector<uint8_t> &v, size_t bb, size_t off, int n) {
            for (size_t i = 0; i + bb <= v.size(); i += bb) for (int k = 0; k < n; k++) { v[i + off + 2 * k] = 0x00; v[i + off + 2 * k + 1] = 0x20; }
        };
        fix(g, bb, doff, gt == GGML_TYPE_Q2_K ? 2 : 1); fix(u, bb, doff, gt == GGML_TYPE_Q2_K ? 2 : 1); fix(dn, 18, 0, 1);
        ColdLayer cl; cl.gate[0] = g.data(); cl.up[0] = u.data(); cl.down = dn.data();
        cl.type_gu[0] = gt; cl.type_dn = GGML_TYPE_IQ4_NL; cl.row_gu[0] = rgu; cl.row_dn = rdn; cl.n0 = cl.n = NC;
        CpuMoe cpu; cpu.start(sh, {cl}, threads); cpu.arm();
        const double expert_bytes = 2.0 * FE * rgu + (double)D * rdn, macs = 3.0 * FE * D;
        printf("-- %s gate/up + IQ4_NL down, %d experts (%.2f GB)\n", name, NC, NC * expert_bytes / 1e9);
        for (int T : {8, 16, 32, 64, 128, 256, 512}) {
            std::vector<float> x((size_t)T * D);
            for (auto &v : x) v = nd(rng);
            Job J = make_job(T, 5, NC, rng, x);                  // ~5 cold picks per token
            cpu.submit(J.j); cpu.wait();                           // warm-up (page faults, scratch growth)
            double best = 1e9;
            for (int rep = 0; rep < 3; rep++) {
                const auto t0 = std::chrono::steady_clock::now();
                cpu.submit(J.j); cpu.wait();
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            const double bytes = J.j.n_exp * expert_bytes, mac = J.tok.size() * macs;
            printf("T=%4d: %3d experts, %5zu picks: %6.2f ms per layer  (%5.1f GB/s of weights, %5.0f GMAC/s)\n",
                   T, J.j.n_exp, J.tok.size(), best * 1e3, bytes / best / 1e9, mac / best / 1e9);
        }
        // decode handoffs (what the GPU triggers per layer): T tokens, k cold picks each, distinct experts
        for (int T : {1, 2}) for (int k : {5, 10}) {
            std::vector<int> e(NC);
            for (int i = 0; i < NC; i++) e[i] = i;
            double best = 1e9, tot = 0; int reps = 0;
            for (int rep = 0; rep < 40; rep++) {
                std::shuffle(e.begin(), e.end(), rng);
                sh->layer = 0; sh->T = T;
                for (int j = 0; j < T * MOE_MAX_E; j++) { const int kk = j % MOE_MAX_E; sh->slot[j] = kk < k ? e[j] : -1; sh->w[j] = 0.1f; }
                for (int i = 0; i < T * D; i++) sh->x[i] = nd(rng);
                const uint32_t gn = sh->go + 1;
                const auto t0 = std::chrono::steady_clock::now();
                __atomic_store_n(&sh->go, gn, __ATOMIC_RELEASE);
                while (__atomic_load_n(&sh->done, __ATOMIC_ACQUIRE) != gn) {}
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (rep >= 4) { best = std::min(best, dt); tot += dt; reps++; }
            }
            const double bytes = (double)T * k * expert_bytes;
            printf("decode handoff T=%d, %2d cold picks/token: best %5.0f us, mean %5.0f us  (%4.1f GB/s at best, %4.1f mean)\n",
                   T, k, best * 1e6, tot / reps * 1e6, bytes / best / 1e9, bytes / (tot / reps) / 1e9);
        }
        cpu.disarm(); cpu.stop();
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
