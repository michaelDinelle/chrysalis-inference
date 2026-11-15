#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "gpu.h"

int main() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) {
        fputs("chrysalis: this computer has no GPU that AMD's ROCm can use\n", stderr);
        return 1;
    }
    struct Gpu { size_t bytes; int id; std::string name; };
    std::vector<Gpu> gpus;
    for (int i = 0; i < n; i++) {
        size_t bytes = 0;
        hipDeviceProp_t p{};
        (void)hipDeviceTotalMem(&bytes, i);
        (void)hipGetDeviceProperties(&p, i);
        gpus.push_back({bytes, i, p.name});
    }
    std::stable_sort(gpus.begin(), gpus.end(), [](const Gpu &a, const Gpu &b) { return a.bytes > b.bytes; });
    for (const Gpu &g : gpus) {
        hipStream_t probe = nullptr;
        if (hipSetDevice(g.id) == hipSuccess && hipStreamCreate(&probe) == hipSuccess && gpu_runs(probe)) {
            printf("HIP_VISIBLE_DEVICES=%d\n", g.id);
            return 0;
        }
    }
    fputs("chrysalis: none of this computer's GPUs can run it:", stderr);
    for (const Gpu &g : gpus) fprintf(stderr, " %s (%.0f GB)", g.name.c_str(), g.bytes / 1e9);
    fputc('\n', stderr);
    return 1;
}
