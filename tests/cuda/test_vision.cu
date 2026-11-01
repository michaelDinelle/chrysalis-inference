#include "../../engine/cuda/vision.h"
#include <chrono>
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s mmproj.gguf out_dim gh gw patches.f32 out.f32\n", argv[0]); return 1; }
    const int out_dim = atoi(argv[2]), gh = atoi(argv[3]), gw = atoi(argv[4]), n = gh * gw;
    std::vector<float> patches((size_t)n * Vision::PATCH_IN);
    FILE *f = fopen(argv[5], "rb");
    if (!f || fread(patches.data(), 4, patches.size(), f) != patches.size()) { fprintf(stderr, "cannot read %s\n", argv[5]); return 1; }
    fclose(f);
    Vision v;
    v.open(argv[1], out_dim);
    const size_t need = v.scratch_need(n);
    uint8_t *scratch;
    half *out;
    if (cudaMalloc(&scratch, need) != cudaSuccess || cudaMallocHost(&out, (size_t)n / 4 * out_dim * 2) != cudaSuccess) return 1;
    cudaStream_t st;
    if (cudaStreamCreate(&st) != cudaSuccess) return 1;
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!v.encode(patches.data(), gh, gw, out, scratch, need, st)) return 1;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (rep) best = std::min(best, s);
        else fprintf(stderr, "first picture (weights loading) %.3f s\n", s);
    }
    fprintf(stderr, "%d patches -> %d tokens in %.1f ms (scratch %.0f MB)\n", n, n / 4, best * 1e3, need / 1e6);
    std::vector<float> o((size_t)n / 4 * out_dim);
    for (size_t i = 0; i < o.size(); i++) o[i] = __half2float(out[i]);
    FILE *g = fopen(argv[6], "wb");
    fwrite(o.data(), 4, o.size(), g);
    fclose(g);
    return 0;
}
