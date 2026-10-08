// Checks the weight relayout, fp16 dequantization, int8 GEMVs and the small-batch GEMM against a plain
// CPU reference decoded straight from the GGUF bytes. With a second argument it also benchmarks them:
// ./build/test_quant data/models/kiln-mix.gguf bench
#include "../engine/kernels.h"
#include "../engine/gguf.h"
#include <cmath>
#include <random>
#include <vector>

static float h2f(const uint8_t *p) { _Float16 h; memcpy(&h, p, 2); return (float)h; }
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
    GGUF g; if (!g.open(argc > 1 ? argv[1] : "data/models/kiln-mix.gguf")) FAIL("open gguf");
    hipStream_t st; CK(hipStreamCreate(&st));
    Arena arena; arena.init(2ull << 30);
    const char *names[] = {"blk.0.ffn_up.weight", "blk.0.ssm_out.weight", "blk.3.attn_k.weight", "blk.0.ffn_gate.weight"};
    std::vector<QTensor> W; std::vector<std::vector<float>> ref;
    int fails = 0;
    for (const char *name : names) {
        const GGUFTensor *gt = g.get(name);
        QTensor t = make_qtensor(arena, gt->type, gt->ne[1], gt->ne[0]);
        uint8_t *raw; CK(hipMalloc(&raw, gt->nbytes)); CK(hipMemcpy(raw, gt->data, gt->nbytes, hipMemcpyHostToDevice));
        relayout(t, raw, st);
        const int64_t n = (int64_t)t.rows * t.cols;
        std::vector<float> r(n); cpu_dequant(t.type, gt->data, n, r.data());
        half *dq; CK(hipMalloc(&dq, n * 2)); dequant(t, dq, st);
        std::vector<_Float16> hq(n); CK(hipMemcpy(hq.data(), dq, n * 2, hipMemcpyDeviceToHost));
        double err = 0, mag = 0;
        for (int64_t i = 0; i < n; i++) { err = fmax(err, fabs((double)hq[i] - r[i])); mag = fmax(mag, fabs(r[i])); }
        bool ok = err <= mag * 1e-3;
        printf("dequant %-24s type %2d %6dx%-6d max|err| %.3g (max|w| %.3g) %s\n", name, t.type, t.rows, t.cols, err, mag, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(hipFree(raw)); CK(hipFree(dq));
        W.push_back(t); ref.push_back(std::move(r));
    }
    // GEMV against CPU float matmul (int8 activations cost ~0.5% relative error)
    std::mt19937 rng(1); std::normal_distribution<float> nd;
    for (int T : {1, 3, 8}) for (int wi = 0; wi < 3; wi++) {
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        std::vector<float> x((size_t)T * K); for (auto &v : x) v = nd(rng);
        float *dx, *dy; QAct qa;
        CK(hipMalloc(&dx, x.size() * 4)); CK(hipMalloc(&dy, (size_t)T * N * 4));
        CK(hipMalloc(&qa.q, x.size())); CK(hipMalloc(&qa.d, x.size() / 32 * 4));
        CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
        quant_act(dx, x.size(), qa, st);
        gemv(t, qa, T, dy, N, GEMV_STORE, st);
        std::vector<float> y((size_t)T * N); CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
        double e2 = 0, r2 = 0;
        for (int tk = 0; tk < T; tk++) for (int n = 0; n < N; n++) {
            double s = 0; for (int k = 0; k < K; k++) s += (double)ref[wi][(size_t)n * K + k] * x[(size_t)tk * K + k];
            e2 += (y[(size_t)tk * N + n] - s) * (y[(size_t)tk * N + n] - s); r2 += s * s;
        }
        double rel = sqrt(e2 / r2); bool ok = rel < 0.02;
        printf("gemv    %-24s T=%d rel err %.4f %s\n", names[wi], T, rel, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(hipFree(dx)); CK(hipFree(dy)); CK(hipFree(qa.q)); CK(hipFree(qa.d));
    }
    // small-batch GEMM (fp16 activations, weights dequantized in registers) against the CPU, on a sample of rows
    for (int T : {9, 40, 100}) for (int wi = 0; wi < 3; wi++) for (int mode : {0, 1, 2}) {   // fp32 store, fp32 add, fp16 store
        if (mode && T != 40) continue;
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        std::vector<_Float16> x((size_t)T * K); for (auto &v : x) v = (_Float16)nd(rng);
        std::vector<float> y0((size_t)T * N); for (auto &v : y0) v = nd(rng);
        half *dx; float *dy;
        CK(hipMalloc(&dx, x.size() * 2)); CK(hipMalloc(&dy, y0.size() * 4));
        CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dy, y0.data(), y0.size() * 4, hipMemcpyHostToDevice));
        gemm_q(t, dx, K, T, dy, N, mode == 1 ? GEMV_ADD : GEMV_STORE, mode == 2, st);
        std::vector<float> y((size_t)T * N);
        if (mode == 2) {
            std::vector<_Float16> yh(y.size()); CK(hipMemcpy(yh.data(), dy, yh.size() * 2, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < y.size(); i++) y[i] = yh[i];
        } else CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
        double e2 = 0, r2 = 0;
        for (int tk = 0; tk < T; tk++) for (int n = tk % 7; n < N; n += 7) {
            double s = mode == 1 ? y0[(size_t)tk * N + n] : 0;
            for (int k = 0; k < K; k++) s += (double)ref[wi][(size_t)n * K + k] * (float)x[(size_t)tk * K + k];
            e2 += (y[(size_t)tk * N + n] - s) * (y[(size_t)tk * N + n] - s); r2 += s * s;
        }
        double rel = sqrt(e2 / r2); bool ok = rel < 2e-3;
        printf("gemm_q  %-24s T=%-3d %s rel err %.5f %s\n", names[wi], T, mode == 0 ? "store" : mode == 1 ? "add  " : "f16  ", rel, ok ? "ok" : "FAIL");
        fails += !ok;
        CK(hipFree(dx)); CK(hipFree(dy));
    }
    // speed of the small-batch GEMM (weight bytes are what a forward pass has to stream)
    if (argc > 2) for (int wi = 0; wi < 3; wi++) for (int T : {16, 32, 64, 128, 256}) {
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        half *dx; float *dy; CK(hipMalloc(&dx, (size_t)T * K * 2)); CK(hipMalloc(&dy, (size_t)T * N * 4));
        CK(hipMemset(dx, 0, (size_t)T * K * 2));
        hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        for (int r = 0; r < 3; r++) gemm_q(t, dx, K, T, dy, N, GEMV_STORE, false, st);
        CK(hipEventRecord(a, st)); for (int r = 0; r < 20; r++) gemm_q(t, dx, K, T, dy, N, GEMV_STORE, false, st); CK(hipEventRecord(b, st));
        CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b)); ms /= 20;
        printf("bench   gemm_q %-17s T=%-3d %7.1f us  %6.1f GB/s  %5.1f TFLOPS\n", names[wi], T, ms * 1000,
               g.get(names[wi])->nbytes / (ms * 1e6), 2.0 * N * K * T / (ms * 1e9));
        CK(hipFree(dx)); CK(hipFree(dy));
    }
    // bandwidth: effective GB/s of weight bytes streamed per GEMV
    if (argc > 2) for (int wi = 0; wi < 3; wi++) for (int T : {1, 2, 3, 4, 5, 6, 7, 8}) {
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        const double bytes = g.get(names[wi])->nbytes;
        float *dy; QAct qa; CK(hipMalloc(&dy, (size_t)8 * N * 4)); CK(hipMalloc(&qa.q, (size_t)8 * K)); CK(hipMalloc(&qa.d, (size_t)8 * K / 8));
        CK(hipMemset(qa.q, 1, (size_t)8 * K)); CK(hipMemset(qa.d, 0, (size_t)8 * K / 8));
        hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        for (int r = 0; r < 3; r++) gemv(t, qa, T, dy, N, GEMV_STORE, st);
        CK(hipEventRecord(a, st)); for (int r = 0; r < 50; r++) gemv(t, qa, T, dy, N, GEMV_STORE, st); CK(hipEventRecord(b, st));
        CK(hipEventSynchronize(b)); float ms; CK(hipEventElapsedTime(&ms, a, b));
        printf("bench   %-24s T=%d %7.1f us  %6.1f GB/s\n", names[wi], T, ms * 1000 / 50, bytes * 50 / (ms * 1e6));
    }
    printf(fails ? "%d FAILURES\n" : "all ok\n", fails);
    return fails != 0;
}
