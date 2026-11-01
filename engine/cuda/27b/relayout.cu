#include "kernels.h"
#include <algorithm>
#include <vector>

__global__ void k_copy_range(const uint8_t *raw, int block_bytes, int off, int len, uint8_t *out, int64_t nblocks) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x, words = len / 4;
    if (len == 2) {
        if (i < nblocks) ((uint16_t *)out)[i] = *(const uint16_t *)(raw + i * block_bytes + off);
        return;
    }
    if (i >= nblocks * words) return;
    const uint8_t *src = raw + (i / words) * block_bytes + off + (i % words) * 4;
    const uint16_t *s16 = (const uint16_t *)src;
    ((uint32_t *)out)[i] = s16[0] | ((uint32_t)s16[1] << 16);
}

void relayout(const QTensor &t, const uint8_t *raw_dev, cudaStream_t st, int64_t b0, int64_t nb) {
    const int64_t n = (int64_t)t.rows * t.cols;
    struct Range { int off, len; uint8_t *out; };
    int block_bytes;
    std::vector<Range> ranges;
    if (t.type == Q8_0) { block_bytes = 34; ranges = {{0, 2, (uint8_t *)t.d}, {2, 32, t.q}}; }
    else if (t.type == Q6_K) { block_bytes = 210; ranges = {{0, 128, t.q}, {128, 64, t.qh}, {192, 16, t.sc}, {208, 2, (uint8_t *)t.d}}; }
    else if (t.type == IQ4_XS) { block_bytes = 136; ranges = {{0, 2, (uint8_t *)t.d}, {2, 2, (uint8_t *)t.sh}, {4, 4, t.sc}, {8, 128, t.q}}; }
    else FAIL("relayout: type %d", t.type);
    const int64_t nblocks = nb >= 0 ? nb : n / (t.type == Q8_0 ? 32 : 256);
    for (const Range &r : ranges) {
        const int64_t threads = r.len == 2 ? nblocks : nblocks * (r.len / 4);
        k_copy_range<<<(threads + 255) / 256, 256, 0, st>>>(raw_dev, block_bytes, r.off, r.len, r.out + b0 * r.len, nblocks);
    }
}

__global__ void k_f32_to_f16(const float *in, half *out, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}
void f32_to_f16(const float *in, half *out, int64_t n, cudaStream_t st) {
    k_f32_to_f16<<<std::max<int64_t>(1, (n + 255) / 256), 256, 0, st>>>(in, out, n);
}
