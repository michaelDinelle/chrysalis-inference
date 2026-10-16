// The CPU prefill job (CpuMoe::submit/wait): (1) its per-pick outputs against a float reference built from ggml's
// dequantization, on 8 random cold experts; (2) its speed on layer-sized work: 423 cold experts of random IQ4_XS /
// IQ4_NL bytes (1.1 GB, far past the L3), picks for T tokens, the time per layer and the rates it implies.
// Build: g++ -O3 -march=native -std=c++20 (see the command in the Makefile's test_fn target). Usage: test_cpujob_fn [threads]
#include "../engine_fn/cpu_moe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr int D = 2560, FE = 640;

struct Job {                           // a MoeJob and the arrays it points into
    std::vector<int> slot, first, tok; std::vector<float> w; std::vector<uint16_t> out; MoeJob j;
};

// picks: T tokens x k distinct experts among n (uniform), grouped by expert as the engine does
static Job make_job(int T, int k, int n, std::mt19937 &rng, const std::vector<float> &x) {
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
    J.j.layer = 0; J.j.T = T; J.j.n_exp = (int)J.slot.size(); J.j.x = x.data();
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
    // ---- (1) correctness: 8 quantized random experts, T = 24, 4 picks per token ----
    {
        const int NC = 8, T = 24;
        std::normal_distribution<float> wd(0.f, 0.03f);
        std::vector<uint8_t> raw[3]; std::vector<float> deq[3];
        const int rows[3] = {NC * FE, NC * FE, NC * D}, cols[3] = {D, D, FE};
        const ggml_type gt[3] = {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
        for (int m = 0; m < 3; m++) {
            std::vector<float> w((size_t)rows[m] * cols[m]), im(cols[m], 1.0f);
            for (auto &v : w) v = wd(rng);
            const size_t rs = ggml_row_size(gt[m], cols[m]);
            raw[m].resize(rs * rows[m]);
            ggml_quantize_chunk(gt[m], w.data(), raw[m].data(), 0, rows[m], cols[m], im.data());
            deq[m].resize(w.size());
            for (int r = 0; r < rows[m]; r++) ggml_get_type_traits(gt[m])->to_float(raw[m].data() + rs * r, deq[m].data() + (size_t)r * cols[m], cols[m]);
        }
        ColdLayer cl; cl.gate = raw[0].data(); cl.up = raw[1].data(); cl.down = raw[2].data();
        cl.type_gu = GGML_TYPE_IQ4_XS; cl.type_dn = GGML_TYPE_IQ4_NL;
        cl.row_gu = ggml_row_size(GGML_TYPE_IQ4_XS, D); cl.row_dn = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
        CpuMoe cpu; cpu.start(sh, {cl}, 8); cpu.arm();
        std::vector<float> x((size_t)T * D);
        for (auto &v : x) v = nd(rng);
        Job J = make_job(T, 4, NC, rng, x);
        cpu.submit(J.j); cpu.wait();
        double num = 0, den = 0;
        for (int e = 0; e < J.j.n_exp; e++)
            for (int p = J.first[e]; p < J.first[e + 1]; p++) {
                const int s = J.slot[e]; const float *xt = &x[(size_t)J.tok[p] * D];
                std::vector<double> h(FE);
                for (int r = 0; r < FE; r++) {
                    double g = 0, u = 0;
                    for (int k = 0; k < D; k++) { g += (double)deq[0][((size_t)s * FE + r) * D + k] * xt[k]; u += (double)deq[1][((size_t)s * FE + r) * D + k] * xt[k]; }
                    h[r] = g / (1 + exp(-g)) * u;
                }
                for (int r = 0; r < D; r++) {
                    double y = 0;
                    for (int k = 0; k < FE; k++) y += (double)deq[2][((size_t)s * D + r) * FE + k] * h[k];
                    y *= J.w[p];
                    const double got = ggml_fp16_to_fp32(J.out[(size_t)p * D + r]);
                    num += (got - y) * (got - y); den += y * y;
                }
            }
        const double r = sqrt(num / den);
        printf("CPU prefill job (8 experts, T=24): rel diff %.2e %s\n", r, r < 2e-2 ? "ok" : "FAIL");
        fails += r >= 2e-2;
        cpu.disarm(); cpu.stop();
    }
    // ---- (2) speed on a layer's worth of cold experts (random bytes: the timing does not depend on the values) ----
    {
        const int NC = 423;
        const size_t rgu = ggml_row_size(GGML_TYPE_IQ4_XS, D), rdn = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
        std::vector<uint8_t> g((size_t)NC * FE * rgu), u((size_t)NC * FE * rgu), dn((size_t)NC * D * rdn);
        std::uniform_int_distribution<int> byte(0, 255);
        for (auto *v : {&g, &u, &dn}) for (size_t i = 0; i < v->size(); i += 64) (*v)[i] = (uint8_t)byte(rng);
        // block scales: small finite fp16 values so no lane sees NaN / inf
        auto fix = [](std::vector<uint8_t> &v, size_t bb) { for (size_t i = 0; i + bb <= v.size(); i += bb) { v[i] = 0x00; v[i + 1] = 0x20; } };
        fix(g, 136); fix(u, 136); fix(dn, 18);
        ColdLayer cl; cl.gate = g.data(); cl.up = u.data(); cl.down = dn.data();
        cl.type_gu = GGML_TYPE_IQ4_XS; cl.type_dn = GGML_TYPE_IQ4_NL; cl.row_gu = rgu; cl.row_dn = rdn;
        CpuMoe cpu; cpu.start(sh, {cl}, threads); cpu.arm();
        const double expert_bytes = 2.0 * FE * rgu + (double)D * rdn, macs = 3.0 * FE * D;
        for (int T : {8, 16, 32, 64, 128, 256, 512}) {
            std::vector<float> x((size_t)T * D);
            for (auto &v : x) v = nd(rng);
            Job J = make_job(T, 6, NC, rng, x);                  // ~6 cold picks per token
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
        cpu.disarm(); cpu.stop();
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
