// Microbenchmarks that ground the engine's design: memory bandwidth, launch overhead, FP16 GEMM throughput.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void read_sum(const uint4 *p, size_t n, unsigned *out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *out = acc;  // keep the loads alive
}
__global__ void empty_kernel() {}

static float time_ms(hipEvent_t a, hipEvent_t b) { float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms; }

int main() {
    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    // 1. read bandwidth over a 4 GiB buffer
    size_t bytes = 4ull << 30; uint4 *buf; unsigned *out;
    CK(hipMalloc(&buf, bytes)); CK(hipMalloc(&out, 4)); CK(hipMemset(buf, 1, bytes));
    for (int blocks : {192, 384, 768, 1536, 3072}) {
        for (int threads : {256, 512}) {
            read_sum<<<blocks, threads>>>(buf, bytes / 16, out);
            CK(hipEventRecord(t0));
            for (int r = 0; r < 5; r++) read_sum<<<blocks, threads>>>(buf, bytes / 16, out);
            CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
            printf("read bw blocks=%4d threads=%d: %6.1f GB/s\n", blocks, threads, 5 * bytes / (time_ms(t0, t1) * 1e6));
        }
    }
    // 2. launch overhead: back-to-back empty kernels, plain and via a captured graph
    hipStream_t s; CK(hipStreamCreate(&s));
    for (int i = 0; i < 100; i++) empty_kernel<<<1, 64, 0, s>>>();
    CK(hipEventRecord(t0, s));
    for (int i = 0; i < 1000; i++) empty_kernel<<<1, 64, 0, s>>>();
    CK(hipEventRecord(t1, s)); CK(hipEventSynchronize(t1));
    printf("stream launch: %.2f us/kernel\n", time_ms(t0, t1));
    hipGraph_t g; hipGraphExec_t ge;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < 1000; i++) empty_kernel<<<1, 64, 0, s>>>();
    CK(hipStreamEndCapture(s, &g)); CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(hipGraphLaunch(ge, s)); CK(hipStreamSynchronize(s));
    CK(hipEventRecord(t0, s)); CK(hipGraphLaunch(ge, s)); CK(hipEventRecord(t1, s)); CK(hipEventSynchronize(t1));
    printf("graph launch:  %.2f us/kernel\n", time_ms(t0, t1));
    // 3. rocBLAS FP16 GEMM (fp32 accumulate) at prefill-like shapes
    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_stream(h, s);
    struct { int m, n, k; } shapes[] = {{2048, 17408, 5120}, {2048, 5120, 17408}, {4096, 17408, 5120}, {512, 17408, 5120}, {2048, 10240, 5120}};
    void *A, *B, *C; CK(hipMalloc(&A, 4096ull * 17408 * 2)); CK(hipMalloc(&B, 17408ull * 5120 * 2)); CK(hipMalloc(&C, 4096ull * 17408 * 2));
    CK(hipMemset(A, 0, 4096ull * 17408 * 2)); CK(hipMemset(B, 0, 17408ull * 5120 * 2));
    float alpha = 1, beta = 0;
    for (auto sh : shapes) {
        // C[n x m] = W[n x k] * X[k x m]  (column-major: weights as rows of out features)
        auto run = [&]() {
            rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, sh.n, sh.m, sh.k, &alpha,
                            B, rocblas_datatype_f16_r, sh.k, A, rocblas_datatype_f16_r, sh.k, &beta,
                            C, rocblas_datatype_f16_r, sh.n, C, rocblas_datatype_f16_r, sh.n,
                            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        };
        run(); run(); CK(hipStreamSynchronize(s));
        CK(hipEventRecord(t0, s)); for (int r = 0; r < 10; r++) run(); CK(hipEventRecord(t1, s)); CK(hipEventSynchronize(t1));
        double tf = 2.0 * sh.m * sh.n * sh.k * 10 / (time_ms(t0, t1) * 1e9);
        printf("rocblas hgemm m=%d n=%d k=%d: %.1f TFLOPS\n", sh.m, sh.n, sh.k, tf);
    }
    return 0;
}
