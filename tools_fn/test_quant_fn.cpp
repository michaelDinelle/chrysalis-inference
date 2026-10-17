// Checks the Flash-Next engine's GEMV, small-batch GEMM and dequantization for every weight format it uses
// against ggml's own quantizer and dequantizer (so the block encodings are exactly llama.cpp's).
#include "../engine_fn/kernels.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

static int fails = 0;
static void check(const char *what, const float *got, const double *ref, size_t n, double tol) {
    double num = 0, den = 0, worst = 0;
    for (size_t i = 0; i < n; i++) { const double d = got[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; worst = std::max(worst, fabs(d)); }
    const double rel = sqrt(num / std::max(den, 1e-30));
    const bool ok = rel < tol;
    if (!ok) fails++;
    printf("  %-34s rel %.2e  max abs %.2e  %s\n", what, rel, worst, ok ? "ok" : "FAIL");
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    hipStream_t st; CK(hipStreamCreate(&st));
    Arena a; a.init((size_t)1 << 30);
    struct Case { ggml_type gt; int type; int rows, cols; };
    const Case cases[] = {
        {GGML_TYPE_Q5_0, Q5_0, 256, 2560}, {GGML_TYPE_Q5_0, Q5_0, 256, 640}, {GGML_TYPE_IQ4_NL, IQ4_NL, 256, 2560},
        {GGML_TYPE_IQ4_NL, IQ4_NL, 256, 640}, {GGML_TYPE_Q8_0, Q8_0, 256, 2560}, {GGML_TYPE_Q8_0, Q8_0, 512, 320},
        {GGML_TYPE_Q6_K, Q6_K, 256, 2560}, {GGML_TYPE_IQ4_XS, IQ4_XS, 256, 2560},
        {GGML_TYPE_Q3_K, Q3_K, 256, 2560}, {GGML_TYPE_Q2_K, Q2_K, 256, 2560},
        {GGML_TYPE_Q3_K, Q3_K, 1024, 4096}, {GGML_TYPE_Q2_K, Q2_K, 1024, 4096},   // split-K GEMV
    };
    for (const Case &c : cases) {
        printf("%s %dx%d\n", ggml_type_name(c.gt), c.rows, c.cols);
        const size_t n = (size_t)c.rows * c.cols;
        std::vector<float> w(n), imat(c.cols);
        for (auto &v : w) v = 0.02f * nd(rng);
        for (auto &v : imat) v = 0.5f + std::abs(nd(rng));
        const size_t rs = ggml_row_size(c.gt, c.cols);
        std::vector<uint8_t> raw(rs * c.rows);
        ggml_quantize_chunk(c.gt, w.data(), raw.data(), 0, c.rows, c.cols, imat.data());
        std::vector<float> wd(n);
        for (int r = 0; r < c.rows; r++) ggml_get_type_traits(c.gt)->to_float(raw.data() + r * rs, wd.data() + (size_t)r * c.cols, c.cols);
        // device tensor via the relayout the loader uses
        const size_t mark = a.used;
        QTensor W = make_qtensor(a, c.type, c.rows, c.cols);
        uint8_t *draw = a.get(raw.size());
        CK(hipMemcpy(draw, raw.data(), raw.size(), hipMemcpyHostToDevice));
        relayout(W, draw, st);
        // dequant
        half *dq = a.get<half>(n);
        dequant(W, dq, st);
        std::vector<half> dqh(n); CK(hipMemcpyAsync(dqh.data(), dq, n * 2, hipMemcpyDeviceToHost, st)); CK(hipStreamSynchronize(st));
        std::vector<float> dqf(n); std::vector<double> wdd(n);
        for (size_t i = 0; i < n; i++) { dqf[i] = __half2float(dqh[i]); wdd[i] = wd[i]; }
        check("dequant vs ggml to_float", dqf.data(), wdd.data(), n, 2e-3);
        // GEMV, T = 1..8 (int8 activations, so the tolerance covers their rounding)
        for (int T : {1, 3, 8}) {
            std::vector<float> x((size_t)T * c.cols);
            for (auto &v : x) v = nd(rng);
            float *dx = a.get<float>(x.size()), *dy = a.get<float>((size_t)T * c.rows);
            QAct qa{a.get<int8_t>(x.size()), a.get<float>(x.size() / 32)};
            CK(hipMemcpyAsync(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice, st));
            quant_act(dx, (int64_t)x.size(), qa, st);
            gemv(W, qa, T, dy, c.rows, GEMV_STORE, st);
            std::vector<float> y((size_t)T * c.rows);
            CK(hipMemcpyAsync(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost, st)); CK(hipStreamSynchronize(st));
            std::vector<double> ref(y.size());
            for (int t = 0; t < T; t++)
                for (int r = 0; r < c.rows; r++) {
                    double s = 0;
                    for (int k = 0; k < c.cols; k++) s += (double)wd[(size_t)r * c.cols + k] * x[(size_t)t * c.cols + k];
                    ref[(size_t)t * c.rows + r] = s;
                }
            char what[64]; snprintf(what, sizeof what, "gemv T=%d", T);
            check(what, y.data(), ref.data(), y.size(), 1.5e-2);
        }
        // small-batch GEMM with fp16 inputs (K a multiple of 256 only)
        if (c.cols % 256 == 0) for (int T : {16, 40, 64}) {
            std::vector<float> x((size_t)T * c.cols);
            for (auto &v : x) v = nd(rng);
            std::vector<half> xh(x.size());
            for (size_t i = 0; i < x.size(); i++) { xh[i] = __float2half(x[i]); x[i] = __half2float(xh[i]); }
            half *dx = a.get<half>(x.size()); float *dy = a.get<float>((size_t)T * c.rows);
            CK(hipMemcpyAsync(dx, xh.data(), xh.size() * 2, hipMemcpyHostToDevice, st));
            gemm_q(W, dx, c.cols, T, dy, c.rows, GEMV_STORE, false, st);
            std::vector<float> y((size_t)T * c.rows);
            CK(hipMemcpyAsync(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost, st)); CK(hipStreamSynchronize(st));
            std::vector<double> ref(y.size());
            for (int t = 0; t < T; t++)
                for (int r = 0; r < c.rows; r++) {
                    double s = 0;
                    for (int k = 0; k < c.cols; k++) s += (double)wd[(size_t)r * c.cols + k] * x[(size_t)t * c.cols + k];
                    ref[(size_t)t * c.rows + r] = s;
                }
            char what[64]; snprintf(what, sizeof what, "gemm_q T=%d", T);
            check(what, y.data(), ref.data(), y.size(), 2e-3);
        }
        a.used = mark;
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
