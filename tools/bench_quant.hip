// Speed of the 27B engine's quantized matmuls on real weights from the model file: the small-batch GEMM (gemm_q,
// T = 16..256: time, GB/s of weight bytes, TFLOPS) and the decode GEMV (T = 1..8: GB/s of weight bytes streamed).
// Their correctness is tools/test_quant.hip.
//   build/bench_quant [data/models/kiln-mix.gguf]
#include "../engine/kernels.h"
#include "../engine/gguf.h"
#include <vector>

int main(int argc, char **argv) {
    GGUF g; if (!g.open(argc > 1 ? argv[1] : "data/models/kiln-mix.gguf")) FAIL("open gguf");
    hipStream_t st; CK(hipStreamCreate(&st));
    Arena arena; arena.init(2ull << 30);
    const char *names[] = {"blk.0.ffn_up.weight", "blk.0.ssm_out.weight", "blk.3.attn_k.weight"};
    std::vector<QTensor> W;
    for (const char *name : names) {
        const GGUFTensor *gt = g.get(name);
        QTensor t = make_qtensor(arena, gt->type, gt->ne[1], gt->ne[0]);
        uint8_t *raw; CK(hipMalloc(&raw, gt->nbytes)); CK(hipMemcpy(raw, gt->data, gt->nbytes, hipMemcpyHostToDevice));
        relayout(t, raw, st);
        CK(hipStreamSynchronize(st)); CK(hipFree(raw));
        W.push_back(t);
    }
    // speed of the small-batch GEMM (weight bytes are what a forward pass has to stream)
    for (int wi = 0; wi < 3; wi++) for (int T : {16, 32, 64, 128, 256}) {
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
    for (int wi = 0; wi < 3; wi++) for (int T : {1, 2, 3, 4, 5, 6, 7, 8}) {
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
    return 0;
}
