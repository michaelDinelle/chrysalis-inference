#include "../engine_fn/cpu_moe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

constexpr int D = 2560, FE = 640;

struct Job {
    std::vector<int> slot, first, tok; std::vector<float> w; std::vector<uint16_t> out, x16; MoeJob j;
};

static Job make_job(const std::vector<std::vector<int>> &picks, int n, std::mt19937 &rng, std::vector<float> &x) {
    const int T = (int)picks.size();
    std::vector<std::vector<int>> by(n);
    for (int t = 0; t < T; t++)
        for (int e : picks[t]) by[e].push_back(t);
    Job J;
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return by[a].size() > by[b].size(); });
    J.first.push_back(0);
    std::uniform_real_distribution<float> u(0.05f, 0.3f);
    for (int s : order) {
        if (by[s].empty()) continue;
        J.slot.push_back(s);
        for (int t : by[s]) { J.tok.push_back(t); J.w.push_back(u(rng)); }
        J.first.push_back((int)J.tok.size());
    }
    J.out.resize(J.tok.size() * (size_t)D);
    J.x16.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) { J.x16[i] = ggml_fp32_to_fp16(x[i]); x[i] = ggml_fp16_to_fp32(J.x16[i]); }
    J.j.layer = 0; J.j.T = T; J.j.n_exp = (int)J.slot.size(); J.j.x = J.x16.data();
    J.j.slot = J.slot.data(); J.j.first = J.first.data(); J.j.tok = J.tok.data(); J.j.w = J.w.data(); J.j.out = J.out.data();
    return J;
}

int main() {
    ggml_cpu_init();
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    MoeShared *sh = new MoeShared{};
    static_assert(VNNI_MIN_TOKENS >= 3 && VNNI_MIN_TOKENS <= 24, "the picks give both paths experts in both groups");
    const int NC = 8, NC0 = 5, T = 24;
    std::vector<int> third = {0, 1, 1, 2, 2, 2, 5};
    third.insert(third.end(), 7, 3);
    third.insert(third.end(), 10, 7);
    std::shuffle(third.begin(), third.end(), rng);
    std::vector<std::vector<int>> picks(T);
    for (int t = 0; t < T; t++) picks[t] = {4, 6, third[t]};
    struct Fmt { ggml_type g0, g1; const char *name; };
    for (const Fmt f : {Fmt{GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, "Q3_K + Q2_K"}, Fmt{GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, "IQ4_XS"}}) {
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
        CpuMoe cpu; cpu.start(sh, nullptr, {cl}, 8); cpu.arm();
        const bool vnni = cpu.use_vnni(1);
        std::vector<float> x((size_t)T * D);
        for (auto &v : x) v = nd(rng);
        Job J = make_job(picks, NC, rng, x);
        cpu.submit(J.j); cpu.wait();
        double num[2][2] = {}, den[2][2] = {};
        for (int e = 0; e < J.j.n_exp; e++) {
            const int s = J.slot[e], path = vnni && J.first[e + 1] - J.first[e] >= VNNI_MIN_TOKENS, grp = s >= NC0;
            for (int p = J.first[e]; p < J.first[e + 1]; p++) {
                const float *xt = &x[(size_t)J.tok[p] * D];
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
                    num[path][grp] += (got - y) * (got - y); den[path][grp] += y * y;
                }
            }
        }
        for (int path = 0; path < 2; path++)
            for (int grp = 0; grp < 2; grp++) {
                char what[96];
                snprintf(what, sizeof what, "CPU prefill job, %s, %s path, group %d", f.name, path ? "VNNI" : "ggml", grp);
                if (den[path][grp] == 0) {
                    const bool expected = !path || vnni;
                    printf("%-48s %s\n", what, expected ? "no expert took it  FAIL" : "not run: no AVX-512 VNNI on this CPU");
                    fails += expected;
                    continue;
                }
                const double r = sqrt(num[path][grp] / den[path][grp]);
                printf("%-48s rel diff %.2e %s\n", what, r, r < 2e-2 ? "ok" : "FAIL");
                fails += r >= 2e-2;
            }
        cpu.disarm(); cpu.stop();
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
