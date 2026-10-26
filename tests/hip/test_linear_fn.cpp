#include "../../engine/hip/flash-next/model.h"
#include "ggml.h"
#include <cmath>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    Engine e;
    CK(hipStreamCreate(&e.st));
    hipblasLtCreate(&e.lt);
    CK(hipMalloc(&e.lt_ws, e.lt_ws_size));
    e.umax = 1024;
    Arena a; a.init((size_t)1 << 30);
    e.b.deq = a.get<half>(Engine::DEQ_ELEMS);
    e.b.gemm16 = a.get<half>((size_t)e.umax * 12288);
    int fails = 0;
    struct Case { ggml_type gt; int type, rows, cols; };
    for (const Case c : {Case{GGML_TYPE_Q6_K, Q6_K, 12288, 2560}, Case{GGML_TYPE_Q6_K, Q6_K, 10240, 2560}, Case{GGML_TYPE_Q8_0, Q8_0, 2560, 6144}}) {
        const size_t n = (size_t)c.rows * c.cols, rs = ggml_row_size(c.gt, c.cols);
        std::vector<float> w(n), im(c.cols, 1.0f);
        for (auto &v : w) v = 0.02f * nd(rng);
        std::vector<uint8_t> raw(rs * c.rows);
        ggml_quantize_chunk(c.gt, w.data(), raw.data(), 0, c.rows, c.cols, im.data());
        const size_t mark = a.used;
        QTensor W = make_qtensor(a, c.type, c.rows, c.cols);
        uint8_t *draw = a.get(raw.size());
        CK(hipMemcpy(draw, raw.data(), raw.size(), hipMemcpyHostToDevice));
        relayout(W, draw, e.st);
        for (int T : {400, 826}) {
            std::vector<half> xh((size_t)T * c.cols);
            for (auto &v : xh) v = __float2half(nd(rng));
            half *x = a.get<half>(xh.size());
            CK(hipMemcpy(x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
            float *y1 = a.get<float>((size_t)T * c.rows), *y2 = a.get<float>((size_t)T * c.rows);
            half *h1 = a.get<half>((size_t)T * c.rows), *h2 = a.get<half>((size_t)T * c.rows);
            for (int acc = 0; acc < 2; acc++) {
                CK(hipMemset(y1, 0, (size_t)T * c.rows * 4)); CK(hipMemset(y2, 0, (size_t)T * c.rows * 4));
                e.linear(W, T, QAct{}, x, c.cols, y1, c.rows, acc ? GEMV_ADD : GEMV_STORE);
                gemm_q(W, x, c.cols, T, y2, c.rows, acc ? GEMV_ADD : GEMV_STORE, false, e.st);
                e.linear(W, T, QAct{}, x, c.cols, h1, c.rows, GEMV_STORE, true);
                gemm_q(W, x, c.cols, T, h2, c.rows, GEMV_STORE, true, e.st);
                std::vector<float> a1((size_t)T * c.rows), a2(a1.size()); std::vector<half> b1(a1.size()), b2(a1.size());
                CK(hipMemcpy(a1.data(), y1, a1.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(a2.data(), y2, a2.size() * 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(b1.data(), h1, b1.size() * 2, hipMemcpyDeviceToHost)); CK(hipMemcpy(b2.data(), h2, b2.size() * 2, hipMemcpyDeviceToHost));
                double nf = 0, df = 0, nh = 0, dh = 0;
                for (size_t i = 0; i < a1.size(); i++) {
                    nf += (a1[i] - a2[i]) * (a1[i] - a2[i]); df += (double)a2[i] * a2[i];
                    const float p = __half2float(b1[i]), q = __half2float(b2[i]); nh += (p - q) * (p - q); dh += (double)q * q;
                }
                const double rf = sqrt(nf / df), rh = sqrt(nh / dh);
                const bool ok = rf < 3e-3 && rh < 3e-3;
                printf("%s %dx%d T=%d %s: fp32 rel diff %.2e, fp16 rel diff %.2e %s\n", ggml_type_name(c.gt), c.rows, c.cols, T,
                       acc ? "accumulate" : "store", rf, rh, ok ? "ok" : "FAIL");
                fails += !ok;
            }
        }
        a.used = mark;
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
