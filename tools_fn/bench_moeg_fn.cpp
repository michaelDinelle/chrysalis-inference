// Flash-Next prefill MoE GEMMs in isolation: one 8K-token pass's hot-tier work (125 IQ4_XS / IQ4_NL experts, the pass's
// picks spread evenly, ~160 tokens each in work items of at most 64), timed per call and as TFLOPS.
//   build_fn/bench_moeg_fn [tokens per expert]
#include "../engine_fn/kernels.h"
#include <random>
#include <vector>

int main(int argc, char **argv) {
    const int per = argc > 1 ? atoi(argv[1]) : 160, n_exp = 125;
    hipStream_t st; CK(hipStreamCreate(&st));
    Arena a; a.init((size_t)2 << 30);
    std::mt19937 rng(3);
    auto fill = [&](const QTensor &t) {
        std::vector<std::pair<const void *, size_t>> arrs;
        qtensor_arrays(t, arrs);
        for (auto &p : arrs) { std::vector<uint8_t> h(p.second); for (auto &b : h) b = rng() & 0x3b; CK(hipMemcpy((void *)p.first, h.data(), h.size(), hipMemcpyHostToDevice)); }
    };
    QTensor G = make_qtensor(a, IQ4_XS, n_exp * FE, D), U = make_qtensor(a, IQ4_XS, n_exp * FE, D), Dn = make_qtensor(a, IQ4_NL, n_exp * D, FE);
    fill(G); fill(U); fill(Dn);
    std::vector<MoeWork> w;
    const int rows = n_exp * per;
    for (int e = 0; e < n_exp; e++)
        for (int t0 = 0; t0 < per; t0 += MOE_ITEM) w.push_back({e, e * per + t0, std::min(MOE_ITEM, per - t0), 0});
    MoeWork *dw = a.get<MoeWork>(w.size());
    CK(hipMemcpy(dw, w.data(), w.size() * sizeof(MoeWork), hipMemcpyHostToDevice));
    half *X = a.get<half>((size_t)rows * D), *H = a.get<half>((size_t)rows * FE), *Y = a.get<half>((size_t)rows * D);
    float *wts = a.get<float>(rows);
    CK(hipMemset(X, 0x11, (size_t)rows * D * 2)); CK(hipMemset(wts, 0, rows * 4));
    const QTensor none{};
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            if (k == 0) moe_gemm_gateup(G, U, none, none, dw, (int)w.size(), X, H, st);
            else moe_gemm_down(Dn, none, dw, (int)w.size(), H, wts, Y, st);
        }
        CK(hipEventRecord(e0, st));
        for (int i = 0; i < 10; i++) {
            if (k == 0) moe_gemm_gateup(G, U, none, none, dw, (int)w.size(), X, H, st);
            else moe_gemm_down(Dn, none, dw, (int)w.size(), H, wts, Y, st);
        }
        CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= 10;
        const double flops = 2.0 * rows * (double)D * FE * (k == 0 ? 2 : 1);
        printf("%s: %d experts x %d tokens (%zu work items): %.3f ms  %.1f TFLOPS\n", k == 0 ? "gate/up" : "down   ", n_exp, per, w.size(), ms, flops / (ms * 1e9));
    }
}
