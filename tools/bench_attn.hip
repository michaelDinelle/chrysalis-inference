// Prefill-attention throughput: a T-token chunk attending to `depth` cached keys (synthetic data).
#include "../engine/kernels.h"
#include <cstdlib>
#include <vector>
int main(int argc, char **argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2048, depth = argc > 2 ? atoi(argv[2]) : 32768;
    const int n = depth + T, npages = (n + PAGE - 1) / PAGE;
    hipStream_t st; CK(hipStreamCreate(&st));
    uint8_t *pool; CK(hipMalloc(&pool, (size_t)npages * KV_PAGE)); CK(hipMemset(pool, 1, (size_t)npages * KV_PAGE));
    std::vector<int> pages(npages); for (int i = 0; i < npages; i++) pages[i] = i;
    int *dp; CK(hipMalloc(&dp, npages * 4)); CK(hipMemcpy(dp, pages.data(), npages * 4, hipMemcpyHostToDevice));
    half *Q16, *out; float *qg;
    CK(hipMalloc(&Q16, (size_t)T * NH * HD * 2)); CK(hipMemset(Q16, 0, (size_t)T * NH * HD * 2));
    CK(hipMalloc(&out, (size_t)T * NH * HD * 2)); CK(hipMalloc(&qg, (size_t)T * NH * 2 * HD * 4)); CK(hipMemset(qg, 0, (size_t)T * NH * 2 * HD * 4));
    float *po, *pml;
    CK(hipMalloc(&po, (size_t)MAX_SPLITS * 8 * NH * HD * 4)); CK(hipMalloc(&pml, (size_t)MAX_SPLITS * 8 * NH * 2 * 4));
    AttnArgs a; a.Q16 = Q16; a.kv = {pool, dp}; a.layer = 0; a.T = T; a.pos0 = depth; a.qg = qg; a.part_o = po; a.part_ml = pml;
    QAct none{nullptr, nullptr};
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    attention(a, out, none, st);
    CK(hipEventRecord(e0, st)); for (int i = 0; i < 5; i++) attention(a, out, none, st); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= 5;
    const double flops = 4.0 * HD * NH * (double)T * (depth + T / 2.0);
    printf("T=%d depth=%d: %.2f ms  %.1f TFLOPS\n", T, depth, ms, flops / (ms * 1e9));
}
