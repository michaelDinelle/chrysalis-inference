// Relayout at load time: raw GGUF blocks (in the staging ring) -> the aligned arrays of make_qtensor. A source file
// of its own: the first launch from a file loads that file's whole code object (~40 ms for quant.hip's GEMVs), and the
// loader wants this one to be small.
#include "kernels.h"
#include <algorithm>
#include <vector>

// Each SoA array is one fixed byte range of every GGUF block; copy a range with one thread per 4 bytes.
__global__ void k_copy_range(const uint8_t *raw, int block_bytes, int off, int len, uint8_t *out, int64_t nblocks) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x, words = len / 4;
    if (len == 2) {
        if (i < nblocks) ((uint16_t *)out)[i] = *(const uint16_t *)(raw + i * block_bytes + off);
        return;
    }
    if (i >= nblocks * words) return;
    const uint8_t *src = raw + (i / words) * block_bytes + off + (i % words) * 4;   // at least 2-byte aligned
    const uint16_t *s16 = (const uint16_t *)src;
    ((uint32_t *)out)[i] = s16[0] | ((uint32_t)s16[1] << 16);
}

// Q3_K's 12 bytes of packed 6-bit scales (at byte 96 of a block) -> 16 int8 scales minus 32, one thread per block
__global__ void k_q3k_scales(const uint8_t *raw, int8_t *out, int64_t nblocks) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint16_t *s16 = (const uint16_t *)(raw + i * 110 + 96);              // blocks are 2-byte aligned
    uint8_t sc[12];
    for (int k = 0; k < 6; k++) { sc[2 * k] = (uint8_t)s16[k]; sc[2 * k + 1] = (uint8_t)(s16[k] >> 8); }
    for (int g = 0; g < 16; g++)
        out[i * 16 + g] = (int8_t)((((sc[g & 7] >> (4 * (g >> 3))) & 0xf) | (((sc[8 + (g & 3)] >> (2 * (g >> 2))) & 3) << 4)) - 32);
}

// Blocks b0 .. b0 + nb of the tensor, read from raw (which starts at block b0); nb < 0: the whole tensor.
void relayout(const QTensor &t, const uint8_t *raw_dev, hipStream_t st, int64_t b0, int64_t nb) {
    const int64_t n = (int64_t)t.rows * t.cols;
    struct Range { int off, len; uint8_t *out; };
    std::vector<Range> ranges;
    if (t.type == Q8_0) ranges = {{0, 2, (uint8_t *)t.d}, {2, 32, t.q}};
    else if (t.type == Q6_K) ranges = {{0, 128, t.q}, {128, 64, t.qh}, {192, 16, t.sc}, {208, 2, (uint8_t *)t.d}};
    else if (t.type == IQ4_XS) ranges = {{0, 2, (uint8_t *)t.d}, {2, 2, (uint8_t *)t.sh}, {4, 4, t.sc}, {8, 128, t.q}};
    else if (t.type == Q5_0) ranges = {{0, 2, (uint8_t *)t.d}, {2, 4, t.qh}, {6, 16, t.q}};
    else if (t.type == IQ4_NL) ranges = {{0, 2, (uint8_t *)t.d}, {2, 16, t.q}};
    else if (t.type == Q3_K) ranges = {{0, 32, t.qh}, {32, 64, t.q}, {108, 2, (uint8_t *)t.d}};   // + the scales below
    else if (t.type == Q2_K) ranges = {{0, 16, t.sc}, {16, 64, t.q}, {80, 2, (uint8_t *)t.d}, {82, 2, (uint8_t *)t.dm}};
    else FAIL("relayout: type %d", t.type);
    const int block_bytes = qbytes_of(t.type);
    const int64_t nblocks = nb >= 0 ? nb : n / qk_of(t.type);
    for (const Range &r : ranges) {
        const int64_t threads = r.len == 2 ? nblocks : nblocks * (r.len / 4);
        k_copy_range<<<(threads + 255) / 256, 256, 0, st>>>(raw_dev, block_bytes, r.off, r.len, r.out + b0 * r.len, nblocks);
    }
    if (t.type == Q3_K) k_q3k_scales<<<(nblocks + 255) / 256, 256, 0, st>>>(raw_dev, (int8_t *)t.sc + b0 * 16, nblocks);
}

