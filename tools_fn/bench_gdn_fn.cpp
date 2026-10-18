// Flash-Next Gated DeltaNet prefill scan alone: one layer, T tokens, random normalized q/k, time per call. Its
// correctness is tools_fn/test_gdn_fn.cpp.
//   build_fn/bench_gdn_fn [T]
#include "../engine_fn/kernels.h"
#include <cmath>
#include <random>
#include <vector>

int main(int argc, char **argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 1024;
    hipStream_t st; CK(hipStreamCreate(&st));
    std::mt19937 rng(5); std::normal_distribution<float> nd;
    std::vector<float> q((size_t)T * GK * GD), k(q.size()), v((size_t)T * GV * GD), ab((size_t)T * 2 * GV), A(GV), dt(GV), S0((size_t)GV * GD * GD);
    auto norm = [&](std::vector<float> &x) {
        for (size_t r = 0; r < x.size() / GD; r++) {
            double ss = 0; for (int i = 0; i < GD; i++) ss += x[r * GD + i] * x[r * GD + i];
            for (int i = 0; i < GD; i++) x[r * GD + i] /= sqrt(ss);
        }
    };
    for (auto &x : q) x = nd(rng); for (auto &x : k) x = nd(rng); norm(q); norm(k);
    for (auto &x : v) x = nd(rng); for (auto &x : ab) x = nd(rng);
    for (auto &x : A) x = -expf(nd(rng) * 0.5f); for (auto &x : dt) x = nd(rng) * 0.5f;
    for (auto &x : S0) x = nd(rng) * 0.1f;
    float *dq, *dk, *dv, *dab, *dA, *ddt, *dS, *dout;
    CK(hipMalloc(&dq, q.size() * 4)); CK(hipMalloc(&dk, k.size() * 4)); CK(hipMalloc(&dv, v.size() * 4)); CK(hipMalloc(&dab, ab.size() * 4));
    CK(hipMalloc(&dA, GV * 4)); CK(hipMalloc(&ddt, GV * 4)); CK(hipMalloc(&dS, S0.size() * 4)); CK(hipMalloc(&dout, v.size() * 4));
    CK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dab, ab.data(), ab.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dA, A.data(), GV * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(ddt, dt.data(), GV * 4, hipMemcpyHostToDevice));
    GdnScanArgs a{};
    a.S = dS; a.qn = dq; a.kn = dk; a.vc = dv; a.ab = dab; a.A = dA; a.dtb = ddt; a.T = T; a.write_state = 1; a.out = dout;
    CK(hipMemcpy(dS, S0.data(), S0.size() * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int i = 0; i < 3; i++) gdn_scan(a, 1, st);
    CK(hipEventRecord(e0, st));
    for (int i = 0; i < 20; i++) gdn_scan(a, 1, st);
    CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
    printf("gdn_scan T=%d, one layer: %.3f ms per call\n", T, ms / 20);
}
