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

__global__ void k_q3k_scales(const uint8_t *raw, int8_t *out, int64_t nblocks) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint16_t *s16 = (const uint16_t *)(raw + i * 110 + 96);
    uint8_t sc[12];
    for (int k = 0; k < 6; k++) { sc[2 * k] = (uint8_t)s16[k]; sc[2 * k + 1] = (uint8_t)(s16[k] >> 8); }
    for (int g = 0; g < 16; g++)
        out[i * 16 + g] = (int8_t)((((sc[g & 7] >> (4 * (g >> 3))) & 0xf) | (((sc[8 + (g & 3)] >> (2 * (g >> 2))) & 3) << 4)) - 32);
}

__device__ __forceinline__ void copy_words(const uint8_t *src, uint8_t *dst, int len) {
    const uint16_t *s16 = (const uint16_t *)src;
    for (int w = 0; w < len / 4; w++) ((uint32_t *)dst)[w] = s16[2 * w] | ((uint32_t)s16[2 * w + 1] << 16);
}

__global__ void k_place(PlaceArgs a) {
    constexpr int64_t NGU = (int64_t)FE * D / 256, NDN = (int64_t)D * FE / 32;
    const Place &p = a.p[blockIdx.y];
    const int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 && blockIdx.y == 0)
        for (int i = threadIdx.x; i < a.npatch; i += blockDim.x) {
            if (a.patch[i].slot_of >= 0) a.slot_of[a.patch[i].slot_of] = a.patch[i].value;
            if (a.patch[i].fmt >= 0) a.fmt[a.patch[i].fmt] = (uint8_t)a.patch[i].value;
        }
    if ((int)blockIdx.y >= a.n) return;
    const int bb = qbytes_of(p.type);
    if (t < 2 * NGU) {
        const int m = t >= NGU;
        const int64_t b = t - m * NGU;
        const QTensor v = slot_view(m ? a.up : a.gate, p.slot, p.type);
        const uint8_t *src = p.src + (m * NGU + b) * bb;
        if (p.type == IQ4_XS) {
            v.d[b] = *(const half *)src;
            v.sh[b] = *(const uint16_t *)(src + 2);
            copy_words(src + 4, v.sc + b * 4, 4);
            copy_words(src + 8, v.q + b * 128, 128);
        } else if (p.type == Q3_K) {
            copy_words(src, v.qh + b * 32, 32);
            copy_words(src + 32, v.q + b * 64, 64);
            v.d[b] = *(const half *)(src + 108);
            const uint16_t *s16 = (const uint16_t *)(src + 96);
            uint8_t sc[12];
            for (int k = 0; k < 6; k++) { sc[2 * k] = (uint8_t)s16[k]; sc[2 * k + 1] = (uint8_t)(s16[k] >> 8); }
            for (int g = 0; g < 16; g++)
                ((int8_t *)v.sc)[b * 16 + g] = (int8_t)((((sc[g & 7] >> (4 * (g >> 3))) & 0xf) | (((sc[8 + (g & 3)] >> (2 * (g >> 2))) & 3) << 4)) - 32);
        } else {
            copy_words(src, v.sc + b * 16, 16);
            copy_words(src + 16, v.q + b * 64, 64);
            v.d[b] = *(const half *)(src + 80);
            v.dm[b] = *(const half *)(src + 82);
        }
    } else if (t < 2 * NGU + NDN) {
        int r0;
        const QTensor dv = down_view(a.down, p.slot, r0);
        const int64_t b = t - 2 * NGU, o = (int64_t)r0 * (FE / 32) + b;
        const uint8_t *src = p.src + 2 * NGU * bb + b * 18;
        dv.d[o] = *(const half *)src;
        copy_words(src + 2, dv.q + o * 16, 16);
    }
}

void place_experts(const PlaceArgs &a, hipStream_t st) {
    constexpr int64_t blocks = (2 * ((int64_t)FE * D / 256) + (int64_t)D * FE / 32 + 255) / 256;
    if (a.n) k_place<<<dim3(blocks, a.n), 256, 0, st>>>(a);
    else if (a.npatch) k_place<<<dim3(1, 1), 256, 0, st>>>(a);
}

void relayout(const QTensor &t, const uint8_t *raw_dev, hipStream_t st, int64_t b0, int64_t nb) {
    const int64_t n = (int64_t)t.rows * t.cols;
    struct Range { int off, len; uint8_t *out; };
    std::vector<Range> ranges;
    if (t.type == Q8_0) ranges = {{0, 2, (uint8_t *)t.d}, {2, 32, t.q}};
    else if (t.type == Q6_K) ranges = {{0, 128, t.q}, {128, 64, t.qh}, {192, 16, t.sc}, {208, 2, (uint8_t *)t.d}};
    else if (t.type == IQ4_XS) ranges = {{0, 2, (uint8_t *)t.d}, {2, 2, (uint8_t *)t.sh}, {4, 4, t.sc}, {8, 128, t.q}};
    else if (t.type == Q5_0) ranges = {{0, 2, (uint8_t *)t.d}, {2, 4, t.qh}, {6, 16, t.q}};
    else if (t.type == IQ4_NL) ranges = {{0, 2, (uint8_t *)t.d}, {2, 16, t.q}};
    else if (t.type == Q3_K) ranges = {{0, 32, t.qh}, {32, 64, t.q}, {108, 2, (uint8_t *)t.d}};
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

