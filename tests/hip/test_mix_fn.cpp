#include "../../engine/hip/flash-next/kernels.h"
#include <cmath>
#include <random>
#include <vector>

static QTensor rand_q8(Arena &a, int rows, int cols, std::mt19937 &rng) {
    QTensor t = make_qtensor(a, Q8_0, rows, cols);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> d(0.0005f, 0.003f);
    std::vector<int8_t> hq((size_t)rows * cols);
    std::vector<half> hd((size_t)rows * cols / 32);
    for (auto &v : hq) v = (int8_t)q(rng);
    for (auto &v : hd) v = __float2half(d(rng));
    CK(hipMemcpy(t.q, hq.data(), hq.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(t.d, hd.data(), hd.size() * 2, hipMemcpyHostToDevice));
    return t;
}

static double rel(const std::vector<float> &a, const std::vector<float> &b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); i++) { n += (a[i] - b[i]) * (a[i] - b[i]); d += (double)b[i] * b[i]; }
    return sqrt(n / d);
}

int main() {
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    hipStream_t st; CK(hipStreamCreate(&st));
    Arena a; a.init((size_t)256 << 20);
    QTensor down = rand_q8(a, HCR, HCD, rng), up = rand_q8(a, HCD, HCR, rng), inj = rand_q8(a, HC, HCD, rng);
    const int TM = 4;
    std::vector<float> res((size_t)TM * HCD), gamma(HCD);
    for (size_t i = 0; i < res.size(); i++) res[i] = nd(rng) * (1.0f + (i / D) % HC);
    for (auto &v : gamma) v = 1.0f + 0.2f * nd(rng);
    float *dres = a.get<float>(res.size()), *dgam = a.get<float>(HCD);
    CK(hipMemcpy(dres, res.data(), res.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dgam, gamma.data(), HCD * 4, hipMemcpyHostToDevice));
    float *xn = a.get<float>((size_t)TM * HCD), *lo = a.get<float>((size_t)TM * HCR), *gate = a.get<float>((size_t)TM * HCD);
    float *mx1 = a.get<float>((size_t)TM * D), *mx2 = a.get<float>((size_t)TM * D), *in1 = a.get<float>(TM * HC), *in2 = a.get<float>(TM * HC);
    float *lo2 = a.get<float>((size_t)TM * HCR), *inv = a.get<float>(TM * HC);
    half *hh1 = a.get<half>((size_t)TM * D), *hh2 = a.get<half>((size_t)TM * D);
    auto qact = [&](size_t n) { QAct q; q.q = a.get<int8_t>(n); q.d = a.get<float>(n / 32); return q; };
    QAct xq = qact((size_t)TM * HCD), lq = qact((size_t)TM * HCR), hq1 = qact((size_t)TM * D), hq2 = qact((size_t)TM * D);
    auto old_path = [&](int T) {
        hc_norm(dres, dgam, T, xn, nullptr, xq, st);
        gemv(down, xq, T, lo, HCR, GEMV_STORE, st);
        hc_lo(lo, T, nullptr, lq, st);
        gemv(up, lq, T, gate, HCD, GEMV_STORE, st);
        gemv(inj, xq, T, in1, HC, GEMV_STORE, st);
        hc_mix(xn, gate, T, mx1, hh1, hq1, st);
    };
    auto new_path = [&](int T) { hc_mix_fused(dres, dgam, down, up, &inj, T, lo2, in2, HC, inv, mx2, hh2, hq2, st); };
    int fails = 0;
    for (int T = 1; T <= TM; T++) {
        old_path(T); new_path(T);
        CK(hipStreamSynchronize(st));
        std::vector<float> a1((size_t)T * D), a2((size_t)T * D), i1(T * HC), i2(T * HC);
        CK(hipMemcpy(a1.data(), mx1, a1.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(a2.data(), mx2, a2.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(i1.data(), in1, i1.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(i2.data(), in2, i2.size() * 4, hipMemcpyDeviceToHost));
        std::vector<int8_t> q1((size_t)T * D), q2((size_t)T * D);
        CK(hipMemcpy(q1.data(), hq1.q, q1.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(q2.data(), hq2.q, q2.size(), hipMemcpyDeviceToHost));
        int qdiff = 0;
        for (size_t i = 0; i < q1.size(); i++) qdiff += abs(q1[i] - q2[i]) > 1;
        const double rm = rel(a2, a1), ri = rel(i2, i1);
        const bool ok = rm < 1e-5 && ri < 1e-5 && qdiff == 0;
        printf("T=%d: mixed rel diff %.2e, inject rel diff %.2e, int8 values off by >1: %d %s\n", T, rm, ri, qdiff, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    {
        QTensor W = rand_q8(a, D, 6144, rng);
        QAct xq6 = qact((size_t)TM * 6144);
        std::vector<float> x6((size_t)TM * 6144), inj(TM * HC);
        for (auto &v : x6) v = nd(rng);
        for (auto &v : inj) v = nd(rng);
        float *dx6 = a.get<float>(x6.size()), *dinj = a.get<float>(inj.size()), *y = a.get<float>((size_t)TM * D);
        float *r1 = a.get<float>((size_t)TM * HCD), *r2 = a.get<float>((size_t)TM * HCD);
        CK(hipMemcpy(dx6, x6.data(), x6.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dinj, inj.data(), inj.size() * 4, hipMemcpyHostToDevice));
        quant_act(dx6, (int64_t)x6.size(), xq6, st);
        {
            std::vector<int8_t> wq((size_t)D * 6144), xq8((size_t)TM * 6144); std::vector<half> wd((size_t)D * 6144 / 32);
            std::vector<float> xdh((size_t)TM * 6144 / 32), yg((size_t)TM * D);
            CK(hipMemcpy(wq.data(), W.q, wq.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(wd.data(), W.d, wd.size() * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(xq8.data(), xq6.q, xq8.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(xdh.data(), xq6.d, xdh.size() * 4, hipMemcpyDeviceToHost));
            gemv(W, xq6, TM, y, D, GEMV_STORE, st);
            CK(hipMemcpy(yg.data(), y, yg.size() * 4, hipMemcpyDeviceToHost));
            std::vector<float> yr((size_t)TM * D);
            for (int t = 0; t < TM; t++)
                for (int r = 0; r < D; r++) {
                    double acc = 0;
                    for (int ch = 0; ch < 6144 / 32; ch++) {
                        int dot = 0;
                        for (int i = 0; i < 32; i++) dot += wq[(size_t)r * 6144 + ch * 32 + i] * xq8[(size_t)t * 6144 + ch * 32 + i];
                        acc += (double)__half2float(wd[(size_t)r * 192 + ch]) * xdh[(size_t)t * 192 + ch] * dot;
                    }
                    yr[(size_t)t * D + r] = (float)acc;
                }
            const double r = rel(yg, yr);
            printf("split-K GEMV 2560x6144, T=1..%d, against the host: rel diff %.2e %s\n", TM, r, r < 1e-5 ? "ok" : "FAIL");
            fails += r >= 1e-5;
        }
        for (int T = 1; T <= TM; T++) {
            CK(hipMemcpy(r1, res.data(), (size_t)T * HCD * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(r2, res.data(), (size_t)T * HCD * 4, hipMemcpyHostToDevice));
            gemv(W, xq6, T, y, D, GEMV_STORE, st);
            hc_combine(r1, dinj, HC, y, nullptr, T, st);
            gemv_hc(W, xq6, T, r2, dinj, st);
            std::vector<float> h1((size_t)T * HCD), h2((size_t)T * HCD);
            CK(hipMemcpy(h1.data(), r1, h1.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(h2.data(), r2, h2.size() * 4, hipMemcpyDeviceToHost));
            const double r = rel(h2, h1);
            printf("T=%d: output projection + HC update fused vs separate: rel diff %.2e %s\n", T, r, r < 1e-6 ? "ok" : "FAIL");
            fails += r >= 1e-6;
        }
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
