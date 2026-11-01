#ifndef CHRYSALIS_GDN_KERNELS
#include "../../engine/cuda/27b/kernels.h"
#endif
#include <cmath>
#include <random>
#include <vector>

static void scan_ref(std::vector<double> &S, const float *q, const float *k, const float *v, const float *ab, const float *A,
                     const float *dt, int T, int qk_stride, int v_stride, int ab_stride, std::vector<double> *out) {
    for (int h = 0; h < GV; h++) {
        double *Sh = &S[(size_t)h * GD * GD];
        for (int t = 0; t < T; t++) {
            const double x = ab[(size_t)t * ab_stride + h] + dt[h];
            const double decay = exp(A[h] * (x > 20 ? x : log1p(exp(x)))), beta = 1 / (1 + exp(-ab[(size_t)t * ab_stride + GV + h]));
            const float *kk = k + (size_t)t * qk_stride + (h % GK) * GD, *vv = v + (size_t)t * v_stride + h * GD;
            for (int j = 0; j < GD; j++) {
                double kv = 0;
                for (int i = 0; i < GD; i++) { Sh[j * GD + i] *= decay; kv += Sh[j * GD + i] * kk[i]; }
                const double delta = beta * (vv[j] - kv);
                double o = 0;
                for (int i = 0; i < GD; i++) {
                    Sh[j * GD + i] += kk[i] * delta;
                    if (out) o += Sh[j * GD + i] * q[(size_t)t * qk_stride + (h % GK) * GD + i];
                }
                if (out) (*out)[((size_t)t * GV + h) * GD + j] = o / sqrt(128.0);
            }
        }
    }
}

static double rel(const std::vector<float> &got, const std::vector<double> &want) {
    double e = 0, r = 0;
    for (size_t i = 0; i < want.size(); i++) { e += (got[i] - want[i]) * (got[i] - want[i]); r += want[i] * want[i]; }
    return sqrt(e / r);
}

int main() {
    cudaStream_t st; CK(cudaStreamCreate(&st));
    std::mt19937 rng(5); std::normal_distribution<float> nd;
    auto rows_l2 = [](std::vector<float> &x) {
        for (size_t r = 0; r < x.size() / GD; r++) {
            double ss = 0; for (int i = 0; i < GD; i++) ss += x[r * GD + i] * x[r * GD + i];
            for (int i = 0; i < GD; i++) x[r * GD + i] /= sqrt(ss);
        }
    };
    auto dev = [&](const std::vector<float> &h) {
        float *d; CK(cudaMalloc(&d, h.size() * 4)); CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); return d;
    };
    auto host = [&](const float *d, size_t n) {
        std::vector<float> h(n); CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h;
    };
    int fails = 0;
    {
        const int T = 100;
        std::vector<float> q((size_t)T * GK * GD), k(q.size()), v((size_t)T * GV * GD), ab((size_t)T * 2 * GV), A(GV), dt(GV),
            S0((size_t)GV * GD * GD);
        for (auto &x : q) x = nd(rng); for (auto &x : k) x = nd(rng); rows_l2(q); rows_l2(k);
        for (auto &x : v) x = nd(rng); for (auto &x : ab) x = nd(rng);
        for (auto &x : A) x = -expf(nd(rng) * 0.5f); for (auto &x : dt) x = nd(rng) * 0.5f;
        for (auto &x : S0) x = nd(rng) * 0.1f;
        float *dq = dev(q), *dk = dev(k), *dv = dev(v), *dab = dev(ab), *dA = dev(A), *ddt = dev(dt), *dS = dev(S0);
        GdnScanArgs a{};
        a.S = dS; a.qn = dq; a.kn = dk; a.vc = dv; a.ab = dab; a.A = dA; a.dtb = ddt; a.T = T; a.write_state = 1; a.out = dv;
        gdn_scan(a, 1, st);
        CK(cudaStreamSynchronize(st));
        std::vector<double> S(S0.begin(), S0.end()), out(v.size());
        scan_ref(S, q.data(), k.data(), v.data(), ab.data(), A.data(), dt.data(), T, GK * GD, GV * GD, 2 * GV, &out);
        const double ro = rel(host(dv, v.size()), out), rs = rel(host(dS, S0.size()), S);
        const bool ok = ro < 1e-4 && rs < 1e-4;
        printf("prefill, T=%d, outputs over v: outputs rel %.2e, state rel %.2e %s\n", T, ro, rs, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    {
        const int NLC = 3, T = 5, CAP = 8;
        std::vector<float> k((size_t)NLC * CAP * GK * GD), v((size_t)NLC * CAP * GV * GD), ab((size_t)NLC * CAP * 2 * GV),
            A((size_t)NLC * GV), dt(A.size()), S0((size_t)NLC * GV * GD * GD);
        for (auto &x : k) x = nd(rng); rows_l2(k);
        for (auto &x : v) x = nd(rng); for (auto &x : ab) x = nd(rng);
        for (auto &x : A) x = -expf(nd(rng) * 0.5f); for (auto &x : dt) x = nd(rng) * 0.5f;
        for (auto &x : S0) x = nd(rng) * 0.1f;
        float *dk = dev(k), *dv = dev(v), *dab = dev(ab), *dA = dev(A), *ddt = dev(dt), *dS = dev(S0);
        GdnScanArgs a{};
        a.S = dS; a.S_layer = (size_t)GV * GD * GD;
        a.kn = dk; a.in_layer = (size_t)CAP * GK * GD;
        a.vc = dv; a.v_layer = (size_t)CAP * GV * GD;
        a.ab = dab; a.ab_layer = (size_t)CAP * 2 * GV;
        a.A = dA; a.dtb = ddt; a.T = T; a.write_state = 1; a.out = nullptr;
        gdn_scan(a, NLC, st);
        CK(cudaStreamSynchronize(st));
        const std::vector<float> gS = host(dS, S0.size());
        double worst = 0;
        for (int l = 0; l < NLC; l++) {
            std::vector<double> S(S0.begin() + (size_t)l * GV * GD * GD, S0.begin() + (size_t)(l + 1) * GV * GD * GD);
            scan_ref(S, nullptr, &k[(size_t)l * CAP * GK * GD], &v[(size_t)l * CAP * GV * GD], &ab[(size_t)l * CAP * 2 * GV],
                     &A[(size_t)l * GV], &dt[(size_t)l * GV], T, GK * GD, GV * GD, 2 * GV, nullptr);
            const std::vector<float> got(gS.begin() + (size_t)l * GV * GD * GD, gS.begin() + (size_t)(l + 1) * GV * GD * GD);
            worst = std::max(worst, rel(got, S));
        }
        printf("commit, %d layers, T=%d of %d: worst layer state rel %.2e %s\n", NLC, T, CAP, worst, worst < 1e-4 ? "ok" : "FAIL");
        fails += worst >= 1e-4;
    }
    printf(fails ? "%d FAILURES\n" : "all ok\n", fails);
    return fails != 0;
}
