#include <hip/hip_runtime.h>
#include "gpu.h"
#include "../os.h"

constexpr unsigned PROBE = 64, STEP = 40503u;

__global__ void k_gpu_runs(unsigned *out) { out[threadIdx.x] = STEP * (threadIdx.x + 1); }

static uint64_t others = 0;

bool gpu_runs() {
    unsigned *d = nullptr, h[PROBE] = {};
    int dev = 0;
    hipDeviceProp_t p{};
    uint64_t total = 0, used = 0;
    if (hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&p, dev) == hipSuccess && os::vram_now(p.pciDomainID, p.pciBusID, p.pciDeviceID, &total, &used)) others = used;
    if (hipMalloc(&d, sizeof h) != hipSuccess) return false;
    bool ok = hipMemset(d, 0, sizeof h) == hipSuccess;
    hipLaunchKernelGGL(k_gpu_runs, dim3(1), dim3(PROBE), 0, 0, d);
    ok = ok && hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost) == hipSuccess;
    (void)hipFree(d);
    for (unsigned i = 0; ok && i < PROBE; i++) ok = h[i] == STEP * (i + 1);
    return ok;
}

__global__ void k_zero(uint32_t *p, size_t words, size_t bytes) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) p[i] = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) for (size_t i = words * 4; i < bytes; i++) ((uint8_t *)p)[i] = 0;
}

void gpu_zero(void *p, size_t bytes, void *stream) {
    const size_t words = bytes / 4, blocks = (words + 255) / 256;
    k_zero<<<(unsigned)(blocks < 1 ? 1 : blocks > 2048 ? 2048 : blocks), 256, 0, (hipStream_t)stream>>>((uint32_t *)p, words, bytes);
}

size_t gpu_free() {
    size_t free_b = 0, total_b = 0;
    int dev = 0;
    hipDeviceProp_t p{};
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess || hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return 0;
    return (size_t)os::vram_grant(0x1002, p.name, p.pciDomainID, p.pciBusID, p.pciDeviceID, free_b, others);
}

#ifdef _WIN32
extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD why, LPVOID ending) {
    if (why == DLL_PROCESS_DETACH && ending) {
        DWORD code = 0;
        if (!GetExitCodeProcess(GetCurrentProcess(), &code) || code == STILL_ACTIVE) code = 0;
        TerminateProcess(GetCurrentProcess(), code);
    }
    return TRUE;
}
#endif
