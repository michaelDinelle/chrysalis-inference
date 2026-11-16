#include "../../engine/gpu/27b/kernels.h"
#include "../../engine/gguf.h"
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

static float h2f(const uint8_t *p) { __half h; memcpy(&h, p, 2); return __half2float(h); }
static const int8_t kIQ4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static void cpu_dequant(int type, const uint8_t *raw, int64_t n, float *y) {
    if (type == Q8_0) {
        for (int64_t b = 0; b < n / 32; b++) {
            const uint8_t *s = raw + b * 34; float d = h2f(s);
            for (int i = 0; i < 32; i++) y[b * 32 + i] = d * (int8_t)s[2 + i];
        }
    } else if (type == IQ4_XS) {
        for (int64_t b = 0; b < n / 256; b++) {
            const uint8_t *s = raw + b * 136; float d = h2f(s); uint16_t sh; memcpy(&sh, s + 2, 2);
            const uint8_t *sl = s + 4, *qs = s + 8; float *o = y + b * 256;
            for (int ib = 0; ib < 8; ib++) {
                int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++) { o[ib * 32 + j] = dl * kIQ4[qs[ib * 16 + j] & 15]; o[ib * 32 + j + 16] = dl * kIQ4[qs[ib * 16 + j] >> 4]; }
            }
        }
    } else if (type == Q6_K) {
        for (int64_t b = 0; b < n / 256; b++) {
            const uint8_t *s = raw + b * 210, *ql = s, *qh = s + 128; const int8_t *sc = (const int8_t *)(s + 192);
            float d = h2f(s + 208), *o = y + b * 256;
            for (int h = 0; h < 2; h++, ql += 64, qh += 32, sc += 8, o += 128)
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    o[l] = d * sc[is] * ((int)((ql[l] & 15) | ((qh[l] & 3) << 4)) - 32);
                    o[l + 32] = d * sc[is + 2] * ((int)((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    o[l + 64] = d * sc[is + 4] * ((int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    o[l + 96] = d * sc[is + 6] * ((int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                }
        }
    }
}

int main(int argc, char **argv) {
    GGUF g; if (!g.open(argc > 1 ? argv[1] : "data/models/qwen3.8-27b/qwen3.8-27b.gguf")) FAIL("open gguf");
    cudaStream_t st; CK(cudaStreamCreate(&st));
    Arena arena; arena.init(2ull << 30);
    const char *names[] = {"blk.0.ffn_up.weight", "blk.0.ssm_out.weight", "blk.3.attn_k.weight", "blk.0.ffn_gate.weight"};
    std::vector<QTensor> W; std::vector<std::vector<float>> ref;
    int fails = 0;
    for (const char *name : names) {
        const GGUFTensor *gt = g.get(name);
        QTensor t = make_qtensor(arena, gt->type, gt->ne[1], gt->ne[0]);
        uint8_t *raw; CK(cudaMalloc(&raw, gt->nbytes)); CK(cudaMemcpy(raw, gt->data, gt->nbytes, cudaMemcpyHostToDevice));
        relayout(t, raw, st);
        const int64_t n = (int64_t)t.rows * t.cols;
        std::vector<float> r(n); cpu_dequant(t.type, gt->data, n, r.data());
        half *dq; CK(cudaMalloc(&dq, n * 2)); dequant(t, dq, st);
        std::vector<__half> hq(n); CK(cudaMemcpy(hq.data(), dq, n * 2, cudaMemcpyDeviceToHost));
        double err = 0, mag = 0;
        for (int64_t i = 0; i < n; i++) { err = fmax(err, fabs((double)__half2float(hq[i]) - r[i])); mag = fmax(mag, fabs(r[i])); }
        bool ok = err <= mag * 1e-3;
        printf("dequant %-24s type %2d %6dx%-6d max|err| %.3g (max|w| %.3g) %s\n", name, t.type, t.rows, t.cols, err, mag, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(cudaFree(raw)); CK(cudaFree(dq));
        W.push_back(t); ref.push_back(std::move(r));
    }
    std::mt19937 rng(1); std::normal_distribution<float> nd;
    for (int T : {1, 3, 8}) for (int wi = 0; wi < 3; wi++) {
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        std::vector<float> x((size_t)T * K); for (auto &v : x) v = nd(rng);
        float *dx, *dy; QAct qa;
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy, (size_t)T * N * 4));
        CK(cudaMalloc(&qa.q, x.size())); CK(cudaMalloc(&qa.d, x.size() / 32 * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        quant_act(dx, x.size(), qa, st);
        gemv(t, qa, T, dy, N, GEMV_STORE, st);
        std::vector<float> y((size_t)T * N); CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        double e2 = 0, r2 = 0;
        for (int tk = 0; tk < T; tk++) for (int n = 0; n < N; n++) {
            double s = 0; for (int k = 0; k < K; k++) s += (double)ref[wi][(size_t)n * K + k] * x[(size_t)tk * K + k];
            e2 += (y[(size_t)tk * N + n] - s) * (y[(size_t)tk * N + n] - s); r2 += s * s;
        }
        double rel = sqrt(e2 / r2); bool ok = rel < 0.02;
        printf("gemv    %-24s T=%d rel err %.4f %s\n", names[wi], T, rel, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(qa.q)); CK(cudaFree(qa.d));
    }
    for (int wi = 0; wi < 3; wi++) {
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        std::vector<float> x((size_t)8 * K); for (auto &v : x) v = nd(rng);
        float *dx, *y8, *y6, *y4, *y1; QAct qa;
        CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&qa.q, x.size())); CK(cudaMalloc(&qa.d, x.size() / 32 * 4));
        CK(cudaMalloc(&y8, (size_t)8 * N * 4)); CK(cudaMalloc(&y6, (size_t)6 * N * 4)); CK(cudaMalloc(&y4, (size_t)4 * N * 4));
        CK(cudaMalloc(&y1, (size_t)N * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        quant_act(dx, x.size(), qa, st);
        gemv(t, qa, 8, y8, N, GEMV_STORE, st); gemv(t, qa, 6, y6, N, GEMV_STORE, st); gemv(t, qa, 4, y4, N, GEMV_STORE, st);
        gemv(t, qa, 1, y1, N, GEMV_STORE, st);
        std::vector<float> a((size_t)8 * N), e((size_t)6 * N), b((size_t)4 * N), c(N);
        CK(cudaMemcpy(a.data(), y8, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), y4, b.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c.data(), y1, c.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(e.data(), y6, e.size() * 4, cudaMemcpyDeviceToHost));
        const bool ok = !memcmp(a.data(), e.data(), e.size() * 4) && !memcmp(a.data(), b.data(), b.size() * 4) && !memcmp(a.data(), c.data(), c.size() * 4);
        printf("gemv    %-24s T=8 = T=6 = T=4 = T=1 on their rows: %s\n", names[wi], ok ? "bit-identical ok" : "FAIL");
        fails += !ok;
        CK(cudaFree(dx)); CK(cudaFree(qa.q)); CK(cudaFree(qa.d)); CK(cudaFree(y8)); CK(cudaFree(y6)); CK(cudaFree(y4)); CK(cudaFree(y1));
    }
    auto quantized = [&](const std::vector<float> &x, bool two, QAct &qa, std::vector<float> &xq) {
        float *dx; CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&qa.q, x.size())); CK(cudaMalloc(&qa.d, x.size() / 32 * 4));
        qa.r = nullptr;
        if (two) CK(cudaMalloc(&qa.r, x.size()));
        quant_act(dx, x.size(), qa, st);
        std::vector<int8_t> q(x.size()), r(x.size(), 0); std::vector<float> d(x.size() / 32);
        CK(cudaMemcpy(q.data(), qa.q, q.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(d.data(), qa.d, d.size() * 4, cudaMemcpyDeviceToHost));
        if (two) CK(cudaMemcpy(r.data(), qa.r, r.size(), cudaMemcpyDeviceToHost));
        xq.resize(x.size());
        double e2 = 0, r2 = 0;
        for (size_t i = 0; i < x.size(); i++) {
            xq[i] = d[i / 32] * (q[i] + r[i] / 254.0f);
            e2 += (xq[i] - x[i]) * (xq[i] - x[i]); r2 += x[i] * x[i];
        }
        CK(cudaFree(dx));
        return sqrt(e2 / r2);
    };
    for (bool two : {false, true}) {
        std::vector<float> x((size_t)64 * 5120); for (auto &v : x) v = nd(rng);
        QAct qa; std::vector<float> xq;
        const double rel = quantized(x, two, qa, xq);
        const bool ok = rel < (two ? 2e-4 : 1e-2);
        printf("quant   %s activations: rel err %.2e %s\n", two ? "two-term" : "one-term", rel, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(cudaFree(qa.q)); CK(cudaFree(qa.d)); if (qa.r) CK(cudaFree(qa.r));
    }
    const bool two = true;
    for (int T : {9, 40, 100, 300}) for (int wi = 0; wi < 3; wi++) for (int mode : {0, 1, 2}) {
        if (mode && T != 40) continue;
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        std::vector<float> x((size_t)T * K); for (auto &v : x) v = nd(rng);
        QAct qa; std::vector<float> xq;
        quantized(x, two, qa, xq);
        std::vector<float> y0((size_t)T * N); for (auto &v : y0) v = nd(rng);
        float *dy; CK(cudaMalloc(&dy, y0.size() * 4));
        CK(cudaMemcpy(dy, y0.data(), y0.size() * 4, cudaMemcpyHostToDevice));
        gemm_q(t, qa, T, dy, N, mode == 1 ? GEMV_ADD : GEMV_STORE, mode == 2, st);
        std::vector<float> y((size_t)T * N);
        if (mode == 2) {
            std::vector<__half> yh(y.size()); CK(cudaMemcpy(yh.data(), dy, yh.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < y.size(); i++) y[i] = __half2float(yh[i]);
        } else CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        double e2 = 0, r2 = 0;
        for (int tk = 0; tk < T; tk++) for (int n = tk % 7; n < N; n += 7) {
            double s = mode == 1 ? y0[(size_t)tk * N + n] : 0;
            for (int k = 0; k < K; k++) s += (double)ref[wi][(size_t)n * K + k] * xq[(size_t)tk * K + k];
            e2 += (y[(size_t)tk * N + n] - s) * (y[(size_t)tk * N + n] - s); r2 += s * s;
        }
        double rel = sqrt(e2 / r2); bool ok = rel < (mode == 2 ? 1e-3 : 2e-5);
        printf("gemm_q  %-24s %s T=%-3d %s rel err %.2e against its activations %s\n", names[wi], two ? "two-term" : "one-term", T,
               mode == 0 ? "store" : mode == 1 ? "add  " : "f16  ", rel, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(cudaFree(dy)); CK(cudaFree(qa.q)); CK(cudaFree(qa.d)); if (qa.r) CK(cudaFree(qa.r));
    }
    for (int T : {9, 100}) {
        const QTensor &tg = W[3], &tu = W[0]; const int K = tg.cols, N = tg.rows;
        std::vector<float> x((size_t)T * K); for (auto &v : x) v = nd(rng);
        QAct qa, out; std::vector<float> xq;
        quantized(x, true, qa, xq);
        CK(cudaMalloc(&out.q, (size_t)T * N)); CK(cudaMalloc(&out.d, (size_t)T * N / 32 * 4));
        out.r = nullptr;
        if (two) CK(cudaMalloc(&out.r, (size_t)T * N));
        gemm_q_glu(tg, tu, qa, T, out, st);
        std::vector<int8_t> q((size_t)T * N), r((size_t)T * N, 0); std::vector<float> d((size_t)T * N / 32);
        CK(cudaMemcpy(q.data(), out.q, q.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(d.data(), out.d, d.size() * 4, cudaMemcpyDeviceToHost));
        if (two) CK(cudaMemcpy(r.data(), out.r, r.size(), cudaMemcpyDeviceToHost));
        double e2 = 0, r2 = 0;
        for (int tk = 0; tk < T; tk++) for (int n = tk % 5; n < N; n += 5) {
            double gs = 0, us = 0;
            for (int k = 0; k < K; k++) {
                gs += (double)ref[3][(size_t)n * K + k] * xq[(size_t)tk * K + k];
                us += (double)ref[0][(size_t)n * K + k] * xq[(size_t)tk * K + k];
            }
            const double want = gs / (1 + exp(-gs)) * us;
            const size_t i = (size_t)tk * N + n;
            const double got = d[i / 32] * (q[i] + r[i] / 254.0);
            e2 += (got - want) * (got - want); r2 += want * want;
        }
        const double rel = sqrt(e2 / r2); const bool ok = rel < (two ? 1e-3 : 2e-2);
        printf("gemm_q_glu gate*up %s output T=%-3d rel err %.2e %s\n", two ? "two-term" : "one-term", T, rel, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(cudaFree(qa.q)); CK(cudaFree(qa.d)); CK(cudaFree(qa.r)); CK(cudaFree(out.q)); CK(cudaFree(out.d)); if (out.r) CK(cudaFree(out.r));
    }
    printf(fails ? "%d FAILURES\n" : "all ok\n", fails);
    return fails != 0;
}
