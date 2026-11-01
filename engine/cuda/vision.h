#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include "lt_tune.h"
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

struct Vision {
    static constexpr int C = 1152, HEADS = 16, HDIM = 72, FFN = 4304, LAYERS = 27, PATCH_IN = 768, GRID = 48, MERGED = 4 * C;

    void open(const std::string &path, int out_dim);
    size_t scratch_need(int n) const;
    bool encode(const float *patches, int gh, int gw, half *out, uint8_t *scratch, size_t scratch_bytes, cudaStream_t st,
                int frames = 1);
    void release();

  private:
    struct Unit { size_t off = 0, bytes = 0; };
    std::string path;
    int out_dim = 0;
    uint8_t *host = nullptr;
    std::vector<Unit> units;
    size_t slot_bytes = 0;
    cudaStream_t cst = nullptr;
    cudaEvent_t copied[2], used[2];
    cublasLtHandle_t lt = nullptr;
    std::map<std::tuple<int, int, int, int, int>, lt_tune::Plan> plans;
    void load();
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, half *y, int ldy, float alpha, void *ws, cudaStream_t st);
};
