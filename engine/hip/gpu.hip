#include <hip/hip_runtime.h>
#include "gpu.h"
#include <cstring>
#include "../os.h"

constexpr unsigned PROBE = 64, STEP = 40503u;

__global__ void k_gpu_runs(unsigned *out) { out[threadIdx.x] = STEP * (threadIdx.x + 1); }

bool gpu_runs(void *stream) {
    unsigned *h = nullptr, *d = nullptr;
    if (hipHostMalloc((void **)&h, PROBE * sizeof(unsigned), hipHostMallocMapped) != hipSuccess) return false;
    memset(h, 0, PROBE * sizeof(unsigned));
    bool ok = hipHostGetDevicePointer((void **)&d, h, 0) == hipSuccess;
    if (ok) hipLaunchKernelGGL(k_gpu_runs, dim3(1), dim3(PROBE), 0, (hipStream_t)stream, d);
    ok = ok && hipStreamSynchronize((hipStream_t)stream) == hipSuccess;
    for (unsigned i = 0; ok && i < PROBE; i++) ok = h[i] == STEP * (i + 1);
    (void)hipHostFree(h);
    return ok;
}

size_t gpu_free() {
    size_t free_b = 0, total_b = 0;
    int dev = 0;
    hipDeviceProp_t p{};
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess || hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return 0;
    return (size_t)os::vram_grant(0x1002, p.name, p.pciDomainID, p.pciBusID, p.pciDeviceID, free_b, os::vram_others(p.pciDomainID, p.pciBusID, p.pciDeviceID));
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
