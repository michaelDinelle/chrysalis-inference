// Flash-Next Gated DeltaNet prefill scan alone: one layer, T tokens, random normalized q/k, time per call; and a check
// against a CPU recurrent reference (state and outputs).
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
    // check: one call from S0 against the CPU recurrence
    CK(hipMemcpy(dS, S0.data(), S0.size() * 4, hipMemcpyHostToDevice));
    gdn_scan(a, 1, st);
    std::vector<float> gS(S0.size()), gO(v.size());
    CK(hipMemcpy(gS.data(), dS, gS.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(gO.data(), dout, gO.size() * 4, hipMemcpyDeviceToHost));
    double eo = 0, ro = 0, es = 0, rs = 0;
    for (int h = 0; h < GV; h++) {
        std::vector<double> S(S0.begin() + (size_t)h * GD * GD, S0.begin() + (size_t)(h + 1) * GD * GD);   // S[j][i]
        const int kh = h % GK;
        for (int t = 0; t < T; t++) {
            const double x = ab[(size_t)t * 2 * GV + h] + dt[h];
            const double decay = exp(A[h] * (x > 20 ? x : log1p(exp(x)))), beta = 1 / (1 + exp(-ab[(size_t)t * 2 * GV + GV + h]));
            const float *kk = &k[((size_t)t * GK + kh) * GD], *qq = &q[((size_t)t * GK + kh) * GD], *vv = &v[((size_t)t * GV + h) * GD];
            for (int j = 0; j < GD; j++) {
                double kv = 0;
                for (int i = 0; i < GD; i++) { S[j * GD + i] *= decay; kv += S[j * GD + i] * kk[i]; }
                const double delta = beta * (vv[j] - kv);
                double o = 0;
                for (int i = 0; i < GD; i++) { S[j * GD + i] += kk[i] * delta; o += S[j * GD + i] * qq[i]; }
                const double g = gO[((size_t)t * GV + h) * GD + j], w = o / sqrt(128.0);
                eo += (g - w) * (g - w); ro += w * w;
            }
        }
        for (size_t i = 0; i < S.size(); i++) { const double g = gS[(size_t)h * GD * GD + i]; es += (g - S[i]) * (g - S[i]); rs += S[i] * S[i]; }
    }
    printf("T=%d vs the CPU recurrence: outputs rel %.2e, state rel %.2e %s\n", T, sqrt(eo / ro), sqrt(es / rs),
           sqrt(eo / ro) < 1e-4 && sqrt(es / rs) < 1e-4 ? "ok" : "FAIL");
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int i = 0; i < 3; i++) gdn_scan(a, 1, st);
    CK(hipEventRecord(e0, st));
    for (int i = 0; i < 20; i++) gdn_scan(a, 1, st);
    CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
    printf("gdn_scan T=%d, one layer: %.3f ms per call\n", T, ms / 20);
}
