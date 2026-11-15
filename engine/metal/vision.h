#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

struct Vision {
    static constexpr int C = 1152, HEADS = 16, HDIM = 72, FFN = 4304, LAYERS = 27, PATCH_IN = 768, GRID = 48, MERGED = 4 * C;

    void open(const std::string &path, int out_dim, void *device, void *library);
    bool encode(const float *patches, int gh, int gw, int frames, uint16_t *out);
    void release();
    ~Vision();

  private:
    struct Gpu;
    Gpu *gpu = nullptr;
    std::string path;
    int out_dim = 0;
    void load();
};
