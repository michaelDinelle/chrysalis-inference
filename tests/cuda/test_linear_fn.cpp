#include "../../engine/cuda/flash-next/model.h"
#include "ggml.h"
#include <cmath>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    Engine e;
    CK(cudaStreamCreate(&e.st));
    cublasLtCreate(&e.lt);
    CK(cudaMalloc(&e.lt_ws, e.lt_ws_size));
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
        CK(cudaMemcpy(draw, raw.data(), raw.size(), cudaMemcpyHostToDevice));
        relayout(W, draw, e.st);
        std::vector<float> wd(n);
        for (int r = 0; r < c.rows; r++) ggml_get_type_traits(c.gt)->to_float(raw.data() + r * rs, wd.data() + (size_t)r * c.cols, c.cols);
        for (int T : {16, 400, 826}) {
            std::vector<half> xh((size_t)T * c.cols);
            for (auto &v : xh) v = __float2half(nd(rng));
            half *x = a.get<half>(xh.size());
            CK(cudaMemcpy(x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
            std::vector<float> y0((size_t)T * c.rows);
            for (auto &v : y0) v = nd(rng);
            float *y1 = a.get<float>((size_t)T * c.rows);
            half *h1 = a.get<half>((size_t)T * c.rows);
            for (int acc = 0; acc < 3; acc++) {
                CK(cudaMemcpy(y1, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice));
                if (acc == 2) e.linear(W, T, QAct{}, x, c.cols, h1, c.rows, GEMV_STORE, true);
                else e.linear(W, T, QAct{}, x, c.cols, y1, c.rows, acc ? GEMV_ADD : GEMV_STORE);
                CK(cudaStreamSynchronize(e.st));
                std::vector<float> got((size_t)T * c.rows);
                if (acc == 2) {
                    std::vector<half> gh(got.size());
                    CK(cudaMemcpy(gh.data(), h1, gh.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < got.size(); i++) got[i] = __half2float(gh[i]);
                } else CK(cudaMemcpy(got.data(), y1, got.size() * 4, cudaMemcpyDeviceToHost));
                double num = 0, den = 0;
                for (int t = 0; t < T; t++)
                    for (int r = t % 13; r < c.rows; r += 13) {
                        double s = acc == 1 ? y0[(size_t)t * c.rows + r] : 0;
                        for (int k = 0; k < c.cols; k++) s += (double)wd[(size_t)r * c.cols + k] * __half2float(xh[(size_t)t * c.cols + k]);
                        const double d = got[(size_t)t * c.rows + r] - s;
                        num += d * d; den += s * s;
                    }
                const double rel = sqrt(num / den);
                const bool ok = rel < (acc == 2 ? 2e-3 : 1e-3);
                printf("%s %dx%d T=%d %s: rel err %.2e against ggml's dequantized weights %s\n", ggml_type_name(c.gt), c.rows, c.cols, T,
                       acc == 0 ? "store" : acc == 1 ? "accumulate" : "fp16", rel, ok ? "ok" : "FAIL");
                fails += !ok;
            }
        }
        a.used = mark;
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
