#pragma once
#include <cuda_runtime.h>
#include "../os.h"

inline size_t gpu_free() {
    size_t free_b = 0, total_b = 0;
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
    return (size_t)os::vram_grant(0x10DE, p.name, free_b);
}
