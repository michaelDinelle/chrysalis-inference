#include <hip/hip_runtime.h>
#include "gpu.h"

constexpr unsigned PROBE = 64, STEP = 40503u;

__global__ void k_gpu_runs(unsigned *out) { out[threadIdx.x] = STEP * (threadIdx.x + 1); }

bool gpu_runs() {
    unsigned *d = nullptr, h[PROBE] = {};
    if (hipMalloc(&d, sizeof h) != hipSuccess) return false;
    bool ok = hipMemset(d, 0, sizeof h) == hipSuccess;
    hipLaunchKernelGGL(k_gpu_runs, dim3(1), dim3(PROBE), 0, 0, d);
    ok = ok && hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost) == hipSuccess;
    (void)hipFree(d);
    for (unsigned i = 0; ok && i < PROBE; i++) ok = h[i] == STEP * (i + 1);
    return ok;
}
