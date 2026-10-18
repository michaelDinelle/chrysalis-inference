// Qwen3.8's vision tower, shared by both engines (they differ only in the width of its output): a 27-block ViT that turns
// one picture's 16 x 16 patches into one embedding per 2 x 2 block of them, the image tokens of the language model.
//
// It takes no VRAM of its own. Its weights (0.92 GB fp16, converted from the mmproj GGUF's bf16 at the first picture: no
// weight changes by more than fp32 rounding, measured) stay in pinned host memory, and each block's are copied into
// VRAM on a second stream while the block before runs. Everything else lives in scratch memory the engine lends it for
// the call: its activation buffers, idle between engine calls (0.96 GB for the 27B holds the largest picture the
// preprocessing allows, 65,536 patches).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include "lt_tune.h"
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

struct Vision {
    static constexpr int C = 1152, HEADS = 16, HDIM = 72, FFN = 4304, LAYERS = 27, PATCH_IN = 768, GRID = 48, MERGED = 4 * C;

    // Reads the GGUF's layout (the weights load at the first encode). out_dim: the language model's hidden size.
    void open(const std::string &path, int out_dim);
    // Scratch bytes encode needs for a picture of n patches.
    size_t scratch_need(int n) const;
    // One picture: patches float32 [gh * gw][768] (host memory, kiln/vision.py's layout), into out: fp16 [gh * gw / 4][out_dim]
    // (host memory the GPU can write, i.e. pinned). frames 2: a video's frame pair, patches [gh * gw][1536] (each patch's two
    // frames, colour by colour, as transformers lays them out), whose tower is a picture's but for the patch embedding (the
    // tower attends within a frame pair). Runs on st; returns false if scratch is too small.
    bool encode(const float *patches, int gh, int gw, half *out, uint8_t *scratch, size_t scratch_bytes, hipStream_t st,
                int frames = 1);
    // Gives the pinned weights back (the next encode reads them again: ~0.35 s from the page cache).
    void release();

  private:
    struct Unit { size_t off = 0, bytes = 0; };   // a piece of the weights copied in at once (offsets in the host copy)
    std::string path;
    int out_dim = 0;
    uint8_t *host = nullptr;                      // pinned weights, unit after unit
    std::vector<Unit> units;                      // the patch embedding, the 27 blocks, the merger's two halves
    size_t slot_bytes = 0;                        // VRAM for one unit (two alternate)
    hipStream_t cst = nullptr;
    hipEvent_t copied[2], used[2];
    hipblasLtHandle_t lt = nullptr;
    std::map<std::tuple<int, int, int, int, int>, lt_tune::Plan> plans;
    void load();
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, half *y, int ldy, float alpha, void *ws, hipStream_t st);
};
