// Checks the weight relayout, fp16 dequantization, int8 GEMVs and the small-batch GEMM against a plain
// CPU reference decoded straight from the GGUF bytes, and the GEMVs of different pass sizes against each other. Their speed is tools/bench_quant.hip.
//   build/test_quant data/models/kiln-mix.gguf
#include "../engine/kernels.h"
#include "../engine/gguf.h"
#include <cmath>
#include <cstring>
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
    // A row comes out the same whatever the pass's size (up to 4 tokens and from 5 are different kernels): what a pass of
    // 8 tokens gives its first 4 and its first, bit for bit, against passes of 4 and of 1 (parallel replies and prompt
    // lookup draw what they would alone only so)
    for (int wi = 0; wi < 3; wi++) {
        const QTensor &t = W[wi]; const int K = t.cols, N = t.rows;
        std::vector<float> x((size_t)8 * K); for (auto &v : x) v = nd(rng);
        float *dx, *y8, *y4, *y1; QAct qa;
        CK(hipMalloc(&dx, x.size() * 4)); CK(hipMalloc(&qa.q, x.size())); CK(hipMalloc(&qa.d, x.size() / 32 * 4));
        CK(hipMalloc(&y8, (size_t)8 * N * 4)); CK(hipMalloc(&y4, (size_t)4 * N * 4)); CK(hipMalloc(&y1, (size_t)N * 4));
        CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
        quant_act(dx, x.size(), qa, st);
        gemv(t, qa, 8, y8, N, GEMV_STORE, st); gemv(t, qa, 4, y4, N, GEMV_STORE, st); gemv(t, qa, 1, y1, N, GEMV_STORE, st);
        std::vector<float> a((size_t)8 * N), b((size_t)4 * N), c(N);
        CK(hipMemcpy(a.data(), y8, a.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), y4, b.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(c.data(), y1, c.size() * 4, hipMemcpyDeviceToHost));
        const bool ok = !memcmp(a.data(), b.data(), b.size() * 4) && !memcmp(a.data(), c.data(), c.size() * 4);
        printf("gemv    %-24s T=8 = T=4 = T=1 on their rows: %s\n", names[wi], ok ? "bit-identical ok" : "FAIL");
        fails += !ok;
        CK(hipFree(dx)); CK(hipFree(qa.q)); CK(hipFree(qa.d)); CK(hipFree(y8)); CK(hipFree(y4)); CK(hipFree(y1));
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
    printf(fails ? "%d FAILURES\n" : "all ok\n", fails);
    return fails != 0;
}
