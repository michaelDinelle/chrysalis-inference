// QSA attention layers (12 of 48, plus the MTP layer): q/k RMSNorm + partial RoPE + paged int8 KV-cache writes,
// WMMA flash attention (dense: every cell up to the query), the QSA indexer (pooled block keys, block scores,
// top-512 selection) and sparse attention over the selected blocks plus the tail, and the combine step that merges
// key splits and applies the sigmoid output gate.
//
// Query rows of a KV group are numbered r = t * 12 + j (token t, q head 12g + j), so a 16-row WMMA tile
// mixes heads and tokens freely; prefill and decode differ only in T and the number of key splits.
// The kernel computes S^T = K Q^T and O^T = V^T P^T, which keeps each lane on a single query row.
#include "kernels.h"
#include <algorithm>
#include <initializer_list>

__device__ __forceinline__ const uint8_t *kv_head(const KVRef &kv, int layer, int head, int pos) {
    return kv.pool + (size_t)kv.pages[pos / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + (size_t)head * KV_HEAD;
}

// 16 consecutive keys (from slot0, a multiple of 16) of one V dimension, as signed 5-bit values.
__device__ __forceinline__ void v5_load16(const uint8_t *blk, int dim, int slot0, int8_t out[16]) {
    const uint2 lo = *(const uint2 *)(blk + VLO_OFF + dim * (PAGE / 2) + slot0 / 2);
    const uint32_t hi = *(const uint16_t *)(blk + VHI_OFF + dim * (PAGE / 8) + slot0 / 8);
#pragma unroll
    for (int i = 0; i < 16; i++)
        out[i] = (int8_t)((((i < 8 ? lo.x : lo.y) >> (4 * (i % 8))) & 15) | (((hi >> i) & 1) << 4)) - 16;
}

// Normalized 256-point Walsh-Hadamard transform across a block of 256 threads (one value per thread).
// It is its own inverse; V is rotated before 5-bit quantization (spreading outlier channels evenly) and
// the attention output is rotated back.
__device__ __forceinline__ float fwht256(float x, float *buf) {
    const int i = threadIdx.x;
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {                 // within a wave
        const float o = __shfl_xor(x, h, 32);
        x = (i & h) ? o - x : x + o;
    }
    for (int h = 32; h < 256; h <<= 1) {               // across waves
        buf[i] = x;
        __syncthreads();
        const float o = buf[i ^ h];
        __syncthreads();
        x = (i & h) ? o - x : x + o;
    }
    return x * 0.0625f;
}

__device__ __forceinline__ float block_sum256(float v, float *red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0;
    for (int i = 0; i < 8; i++) s += red[i];
    __syncthreads();
    return s;
}

// grid (NH + 2*NKV jobs, T), 256 threads = one head dim each.
__global__ __launch_bounds__(256) void k_attn_prep(const float *qg, const float *kin, const float *vin, const float *qnw,
                                                   const float *knw, int pos0, KVRef kv, int layer, half *Q16) {
    __shared__ float red[8], buf[HD];
    const int job = blockIdx.x, t = blockIdx.y, i = threadIdx.x, pos = pos0 + t;
    float x;
    if (job < NH) x = qg[((size_t)t * NH + job) * 2 * HD + i];
    else if (job < NH + NKV) x = kin[((size_t)t * NKV + job - NH) * HD + i];
    else x = vin[((size_t)t * NKV + job - NH - NKV) * HD + i];
    if (job < NH + NKV) {
        x *= rsqrtf(block_sum256(x * x, red) / HD + EPS) * (job < NH ? qnw : knw)[i];
        if (i < ROT) buf[i] = x;
        __syncthreads();
        if (i < ROT) {               // NeoX-style pairs (i, i + 32) over the first 64 dims
            const int f = i % (ROT / 2);
            float s, c;
            sincosf(pos * powf(ROPE_BASE, -2.0f * f / ROT), &s, &c);
            x = i < ROT / 2 ? x * c - buf[i + ROT / 2] * s : x * c + buf[i - ROT / 2] * s;
        }
    }
    if (job < NH) {
        Q16[((size_t)t * NH + job) * HD + i] = __float2half(x * 0.0625f);   // folds 1/sqrt(256)
        return;
    }
    const bool isv = job >= NH + NKV;
    const int head = (job - NH) % NKV, slot = pos % PAGE;
    uint8_t *blk = (uint8_t *)kv_head(kv, layer, head, pos);
    const float amax = warp_max(fabsf(x));
    if (!isv) {
        const float d = amax / 127.0f;
        ((int8_t *)blk)[slot * HD + i] = (int8_t)__float2int_rn(d > 0 ? x / d : 0.0f);
        if ((i & 31) == 0) ((half *)(blk + KS_OFF))[slot * (HD / 32) + i / 32] = __float2half(d);
    } else {                                 // 5-bit: nibble and high bit go to bit planes shared with other tokens
        x = fwht256(x, buf);
        const float d = warp_max(fabsf(x)) / 15.0f;
        const uint32_t u = (uint32_t)(__float2int_rn(d > 0 ? x / d : 0.0f) + 16);
        uint32_t *lo = (uint32_t *)(blk + VLO_OFF) + (i * (PAGE / 2) + slot / 2) / 4;
        const int ls = ((slot / 2) % 4) * 8 + (slot % 2) * 4;
        atomicAnd(lo, ~(15u << ls));
        atomicOr(lo, (u & 15) << ls);
        uint32_t *hb = (uint32_t *)(blk + VHI_OFF) + (i * (PAGE / 8) + slot / 8) / 4;
        const int hs = ((slot / 8) % 4) * 8 + slot % 8;
        atomicAnd(hb, ~(1u << hs));
        atomicOr(hb, (u >> 4) << hs);
        if ((i & 31) == 0) ((half *)(blk + VS_OFF))[(i / 32) * PAGE + slot] = __float2half(d);
    }
}

void attn_prep(const float *qg, const float *k, const float *v, const float *qnw, const float *knw, int T, int pos0,
               KVRef kv, int layer, half *Q16, hipStream_t st) {
    k_attn_prep<<<dim3(NH + 2 * NKV, T), 256, 0, st>>>(qg, k, v, qnw, knw, pos0, kv, layer, Q16);
}

constexpr int QLD = HD + 8;   // padded LDS row (halves) for K tiles
constexpr int VLD = 16 + 8;   // padded LDS row for the transposed V tile

// Prefill attention: AW waves (16-row query tiles each) share 32-key stages of K and V^T dequantized
// into LDS; each Q fragment (read from L1) feeds two WMMAs, and the accumulators are rescaled only when
// a row's running max actually grows.
constexpr int KST = 32;                       // keys per LDS stage
constexpr int VLD32 = KST + 8;                // padded LDS row for the transposed V stage

template <int AW>
__global__ __launch_bounds__(AW * 32) void k_attn(AttnArgs a) {
    __shared__ _Float16 Ks[KST][QLD];
    __shared__ _Float16 Vt[HD][VLD32];
    const int g = blockIdx.y, tid = threadIdx.x;
    const int wave = tid / 32, lane = tid % 32, r16 = lane % 16, hi = lane / 16;
    const int nrows = a.T * QPK, tile = blockIdx.x * AW + wave;
    const int myrow = tile * 16 + r16;
    const bool valid = myrow < nrows;
    const int qpos = valid ? a.pos0 + myrow / QPK : -1;
    const half *qrow = a.Q16 + ((size_t)(valid ? myrow / QPK : 0) * NH + g * QPK + (valid ? myrow % QPK : 0)) * HD;
    const int last = min(nrows, (blockIdx.x * AW + AW) * 16) - 1;
    const int kbeg = blockIdx.z * a.split_len, kend = min(a.pos0 + last / QPK + 1, kbeg + a.split_len);

    float m = -INFINITY, l = 0;
    f8 acc[16];
#pragma unroll
    for (int d = 0; d < 16; d++) acc[d] = f8{};

    for (int k0 = kbeg; k0 < kend; k0 += KST) {
        __syncthreads();
        const uint8_t *page = kv_head(a.kv, a.layer, g, k0);       // 32 keys never straddle a 64-key page
        const int slot0 = k0 % PAGE, nvalid = kend - k0;
        for (int job = tid; job < KST * 8; job += AW * 32) {      // K: one 32-dim block of one key per job
            const int kk = job >> 3, blk = job & 7;
            const int8_t *q = (const int8_t *)page + (slot0 + kk) * HD + blk * 32;
            const float d = kk < nvalid ? __half2float(((const half *)(page + KS_OFF))[(slot0 + kk) * (HD / 32) + blk]) : 0.0f;
            const int4 w0 = *(const int4 *)q, w1 = *(const int4 *)(q + 16);
            const int8_t *b0 = (const int8_t *)&w0, *b1 = (const int8_t *)&w1;
            h16 x0, x1;
#pragma unroll
            for (int i = 0; i < 16; i++) { x0[i] = (_Float16)(d * b0[i]); x1[i] = (_Float16)(d * b1[i]); }
            *(h16 *)&Ks[kk][blk * 32] = x0;
            *(h16 *)&Ks[kk][blk * 32 + 16] = x1;
        }
        for (int job = tid; job < HD * 2; job += AW * 32) {      // V^T: 16 keys of one dim per job
            const int dim = job >> 1, half16 = job & 1;
            int8_t b[16];
            v5_load16(page, dim, slot0 + half16 * 16, b);
            const half *vs = (const half *)(page + VS_OFF) + (dim / 32) * PAGE + slot0 + half16 * 16;
            h16 v;
#pragma unroll
            for (int i = 0; i < 16; i++) v[i] = (_Float16)(half16 * 16 + i < nvalid ? __half2float(vs[i]) * b[i] : 0.0f);
            *(h16 *)&Vt[dim][half16 * 16] = v;
        }
        __syncthreads();

        f8 s0 = {}, s1 = {};                                      // S^T[key 2i+hi (+16)][row r16]
#pragma unroll
        for (int d = 0; d < 16; d++) {
            const h16 q = *(const h16 *)(qrow + d * 16);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const h16 *)&Ks[r16][d * 16], q, s0);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const h16 *)&Ks[16 + r16][d * 16], q, s1);
        }
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int key = k0 + 2 * i + hi;
            if (key >= kend || key > qpos) s0[i] = -INFINITY;
            if (key + 16 >= kend || key + 16 > qpos) s1[i] = -INFINITY;
            mx = fmaxf(mx, fmaxf(s0[i], s1[i]));
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        if (mx > m) {                                             // rescale only when the max grows
            const float alpha = m == -INFINITY ? 0.0f : __expf(m - mx);
            l *= alpha;
#pragma unroll
            for (int dt = 0; dt < 16; dt++) acc[dt] *= alpha;
            m = mx;
        }
        float p0[8], p1[8], rs = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            p0[i] = m == -INFINITY ? 0.0f : __expf(s0[i] - m);
            p1[i] = m == -INFINITY ? 0.0f : __expf(s1[i] - m);
            rs += p0[i] + p1[i];
        }
        l += rs + __shfl_xor(rs, 16, 32);
        h16 P0, P1;                                               // P^T[key][row r16] for keys 0-15 and 16-31
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o0 = __shfl_xor(p0[i], 16, 32), o1 = __shfl_xor(p1[i], 16, 32);
            P0[2 * i] = (_Float16)(hi ? o0 : p0[i]); P0[2 * i + 1] = (_Float16)(hi ? p0[i] : o0);
            P1[2 * i] = (_Float16)(hi ? o1 : p1[i]); P1[2 * i + 1] = (_Float16)(hi ? p1[i] : o1);
        }
#pragma unroll
        for (int dt = 0; dt < 16; dt++) {
            acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const h16 *)&Vt[dt * 16 + r16][0], P0, acc[dt]);
            acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const h16 *)&Vt[dt * 16 + r16][16], P1, acc[dt]);
        }
    }

    if (!valid) return;                                           // acc[dt][i] = O^T[dim 16dt+2i+hi][row r16]
    const size_t row = (size_t)(myrow / QPK) * NH + g * QPK + myrow % QPK;
    if (a.partial) {
        float *po = a.part_o + ((size_t)blockIdx.z * a.T * NH + row) * HD;
#pragma unroll
        for (int dt = 0; dt < 16; dt++)
#pragma unroll
            for (int i = 0; i < 8; i++) po[dt * 16 + 2 * i + hi] = acc[dt][i];
        if (hi == 0) {
            a.part_ml[((size_t)blockIdx.z * a.T * NH + row) * 2] = m;
            a.part_ml[((size_t)blockIdx.z * a.T * NH + row) * 2 + 1] = l;
        }
        return;
    }
    const float inv = l > 0 ? 1.0f / l : 0.0f;
#pragma unroll
    for (int dt = 0; dt < 16; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int dim = dt * 16 + 2 * i + hi;
            a.out_h[row * HD + dim] = __float2half(acc[dt][i] * inv * sigmoid(a.qg[row * 2 * HD + HD + dim]));
        }
}

// Decode / verify attention (T <= 8): one wave per (16-row query tile, kv head, key split), no LDS.
// Every WMMA operand is read straight from memory: K rows (int8 -> fp16, per-block scale), Q rows
// (fp16) and V^T rows (5-bit, 16 consecutive keys of one dim), with the per-token V scales folded into
// the probabilities. Writes unnormalized partials that k_attn_combine merges.
__global__ __launch_bounds__(32) void k_attn_dec(AttnArgs a) {
    const int tile = blockIdx.x, g = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x, r16 = lane % 16, hi = lane / 16;
    const int nrows = a.T * QPK, myrow = tile * 16 + r16;
    const bool valid = myrow < nrows;
    const int qpos = valid ? a.pos0 + myrow / QPK : -1;
    const half *qrow = a.Q16 + ((size_t)(valid ? myrow / QPK : 0) * NH + g * QPK + (valid ? myrow % QPK : 0)) * HD;
    const int kbeg = split * a.split_len, kend = min(a.pos0 + a.T, kbeg + a.split_len);

    float m = -INFINITY, l = 0;
    f8 acc[16];
#pragma unroll
    for (int d = 0; d < 16; d++) acc[d] = f8{};
    for (int k0 = kbeg; k0 < kend; k0 += 16) {
        const uint8_t *page = kv_head(a.kv, a.layer, g, k0);
        const int slot0 = k0 % PAGE;
        const int8_t *kq = (const int8_t *)page + (slot0 + r16) * HD;
        const half *ks = (const half *)(page + KS_OFF) + (slot0 + r16) * (HD / 32);
        f8 sc = {};                                               // S^T[key 2i+hi][row r16]
#pragma unroll
        for (int d = 0; d < 16; d++) {
            const int4 kr = *(const int4 *)(kq + d * 16);
            const _Float16 s = (_Float16)__half2float(ks[d / 2]);
            const int8_t *kb = (const int8_t *)&kr;
            h16 A;
#pragma unroll
            for (int i = 0; i < 16; i++) A[i] = (_Float16)kb[i] * s;
            sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, *(const h16 *)(qrow + d * 16), sc);
        }
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int key = k0 + 2 * i + hi;
            if (key >= kend || key > qpos) sc[i] = -INFINITY;
            mx = fmaxf(mx, sc[i]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx);
        const float alpha = mn == -INFINITY ? 1.0f : __expf(m - mn);
        float p[8], rs = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) { p[i] = mn == -INFINITY ? 0.0f : __expf(sc[i] - mn); rs += p[i]; }
        rs += __shfl_xor(rs, 16, 32);
        l = l * alpha + rs;
        m = mn;
        float P[16];                                              // this row's probabilities for keys 0..15
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o = __shfl_xor(p[i], 16, 32);
            P[2 * i] = hi ? o : p[i];
            P[2 * i + 1] = hi ? p[i] : o;
        }
#pragma unroll
        for (int dt = 0; dt < 16; dt++) acc[dt] *= alpha;
#pragma unroll
        for (int blk = 0; blk < 8; blk++) {                       // fold the V scales of this 32-dim block into P
            const half *vs = (const half *)(page + VS_OFF) + blk * PAGE + slot0;
            h16 B;
#pragma unroll
            for (int i = 0; i < 16; i++) B[i] = (_Float16)(P[i] * __half2float(vs[i]));
#pragma unroll
            for (int dt = 2 * blk; dt < 2 * blk + 2; dt++) {
                int8_t v8[16];
                v5_load16(page, dt * 16 + r16, slot0, v8);
                h16 A;
#pragma unroll
                for (int i = 0; i < 16; i++) A[i] = (_Float16)v8[i];
                acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, acc[dt]);
            }
        }
    }
    if (!valid) return;
    const size_t row = (size_t)(myrow / QPK) * NH + g * QPK + myrow % QPK;
    float *po = a.part_o + ((size_t)split * a.T * NH + row) * HD;
#pragma unroll
    for (int dt = 0; dt < 16; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) po[dt * 16 + 2 * i + hi] = acc[dt][i];
    if (hi == 0) {
        a.part_ml[((size_t)split * a.T * NH + row) * 2] = m;
        a.part_ml[((size_t)split * a.T * NH + row) * 2 + 1] = l;
    }
}

// Merge key splits, apply the output gate, write fp16 (prefill) and/or int8 blocks (decode).
__global__ __launch_bounds__(256) void k_attn_combine(const float *part_o, const float *part_ml, int nsplit, int rows,
                                                      const float *qg, half *oh, QAct oq) {
    __shared__ float buf[HD];
    const int row = blockIdx.x, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; s++) M = fmaxf(M, part_ml[((size_t)s * rows + row) * 2]);
    float L = 0, o = 0;
    for (int s = 0; s < nsplit; s++) {
        const float ms = part_ml[((size_t)s * rows + row) * 2];
        if (ms == -INFINITY) continue;
        const float w = __expf(ms - M);
        L += w * part_ml[((size_t)s * rows + row) * 2 + 1];
        o += w * part_o[((size_t)s * rows + row) * HD + d];
    }
    const float y = fwht256(L > 0 ? o / L : 0.0f, buf) * sigmoid(qg[(size_t)row * 2 * HD + HD + d]);
    const size_t idx = (size_t)row * HD + d;
    if (oh) oh[idx] = __float2half(y);
    if (oq.q) {
        const float sd = warp_max(fabsf(y)) / 127.0f;
        oq.q[idx] = (int8_t)__float2int_rn(sd > 0 ? y / sd : 0.0f);
        if ((d & 31) == 0) oq.d[idx / 32] = sd;
    }
}

// Full attention for T query tokens at positions pos0.. (keys 0 .. pos0+T-1 must already be cached).
// Decode passes split_len > 0 and gets fp16/int8 output via the combine kernel; prefill uses one split.
void attention(AttnArgs a, half *oh, QAct oq, hipStream_t st) {
    const int ntiles = (a.T * QPK + 15) / 16;
    const int nkeys = a.pos0 + a.T;
    a.out_h = oh;
    if (a.T <= 8) {                       // decode / verify: enough splits to fill the GPU, at most MAX_SPLITS
        const int target = std::min(MAX_SPLITS, std::max(1, 1536 / (ntiles * NKV)));
        a.split_len = std::max(64, ((nkeys + target - 1) / target + 15) / 16 * 16);
        a.nsplit = (nkeys + a.split_len - 1) / a.split_len;
        a.partial = 1;
        k_attn_dec<<<dim3(ntiles, NKV, a.nsplit), 32, 0, st>>>(a);
    } else {                              // prefill: big workgroups share K/V stages; split keys if too few
        if (oq.q) FAIL("attention: prefill writes fp16 output");
        const int aw = ntiles >= 384 ? 16 : ntiles >= 96 ? 8 : 4;
        const int wgs = (ntiles + aw - 1) / aw * NKV;
        a.nsplit = std::max(1, std::min({(192 + wgs - 1) / wgs, nkeys / 1024, MAX_SPLITS * 8 / a.T}));
        if (a.nsplit * a.T > MAX_SPLITS * 8) FAIL("attention: chunk of %d tokens exceeds the partial buffers", a.T);
        a.split_len = ((nkeys + a.nsplit - 1) / a.nsplit + KST - 1) / KST * KST;
        a.nsplit = (nkeys + a.split_len - 1) / a.split_len;
        a.partial = 1;
        const dim3 grid((ntiles + aw - 1) / aw, NKV, a.nsplit);
        if (aw == 16) k_attn<16><<<grid, 512, 0, st>>>(a);
        else if (aw == 8) k_attn<8><<<grid, 256, 0, st>>>(a);
        else k_attn<4><<<grid, 128, 0, st>>>(a);
    }
    if (a.partial) k_attn_combine<<<a.T * NH, 256, 0, st>>>(a.part_o, a.part_ml, a.nsplit, a.T * NH, a.qg, oh, oq);
}

// ---- QSA indexer ------------------------------------------------------------------------------------
// Each cell keeps its raw indexer key (fp16, unnormed, unrotated); every complete block of KPOOL cells keeps one
// pooled key: the mean of its raw keys, RMS-normed and rotated to the position of its first cell. A query's
// indexer heads (normed, rotated) score block b as sum_h relu(q_h . k_b) / sqrt(IDX_D); the query attends the
// cells of its top SEL_BLOCKS complete visible blocks plus the incomplete tail block it sits in.

__device__ __forceinline__ uint8_t *idx_raw(const KVRef &kv, int layer, int cell) {
    return kv.pool + (size_t)kv.pages[cell / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + NKV * KV_HEAD + (size_t)(cell % PAGE) * IDX_D * 2;
}
__device__ __forceinline__ half *idx_pool(const KVRef &kv, int layer, int block) {
    const int cell = block * KPOOL;
    return (half *)(kv.pool + (size_t)kv.pages[cell / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + NKV * KV_HEAD + IDX_RAW) +
           (size_t)((cell % PAGE) / KPOOL) * IDX_D;
}

// rmsnorm over 128 values (one per thread of a 128-thread block) times w, then NeoX rotation of the first ROT dims
__device__ __forceinline__ float norm_rope128(float x, const float *w, int pos, float *red, float *buf) {
    const int i = threadIdx.x;
    float ss = warp_sum(x * x);
    if ((i & 31) == 0) red[i >> 5] = ss;
    __syncthreads();
    ss = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    x *= rsqrtf(ss / IDX_D + EPS) * w[i];
    if (i < ROT) buf[i] = x;
    __syncthreads();
    if (i < ROT) {
        const int f = i % (ROT / 2);
        float s, c;
        sincosf(pos * powf(ROPE_BASE, -2.0f * f / ROT), &s, &c);
        x = i < ROT / 2 ? x * c - buf[i + ROT / 2] * s : x * c + buf[i - ROT / 2] * s;
    }
    __syncthreads();
    return x;
}

// grid (1 + IDX_H, T), 128 threads: job 0 stores the raw key of token t, jobs 1.. norm + rotate its query heads
__global__ __launch_bounds__(128) void k_idx_prep(const float *kraw, const float *qraw, const float *qnw, int pos0, KVRef kv,
                                                  int layer, half *q16) {
    __shared__ float red[4], buf[ROT];
    const int job = blockIdx.x, t = blockIdx.y, i = threadIdx.x, pos = pos0 + t;
    if (job == 0) {
        ((half *)idx_raw(kv, layer, pos))[i] = __float2half(kraw[(size_t)t * IDX_D + i]);
        return;
    }
    const int h = job - 1;
    const float x = norm_rope128(qraw[((size_t)t * IDX_H + h) * IDX_D + i], qnw, pos, red, buf);
    q16[((size_t)t * IDX_H + h) * IDX_D + i] = __float2half(x);
}

// one block per pooled key completed by tokens pos0 .. pos0+T-1 (blocks b0 ..), 128 threads
__global__ __launch_bounds__(128) void k_idx_pool(int b0, KVRef kv, int layer, const float *knw) {
    __shared__ float red[4], buf[ROT];
    const int b = b0 + blockIdx.x, i = threadIdx.x;
    float x = 0;
    for (int c = 0; c < KPOOL; c++) x += __half2float(((const half *)idx_raw(kv, layer, b * KPOOL + c))[i]);
    x = norm_rope128(x * (1.0f / KPOOL), knw, b * KPOOL, red, buf);
    idx_pool(kv, layer, b)[i] = __float2half(x);
}

void idx_prep(const float *kraw, const float *qraw, const float *qnw, const float *knw, int T, int pos0, KVRef kv, int layer,
              half *q16, hipStream_t st) {
    k_idx_prep<<<dim3(1 + IDX_H, T), 128, 0, st>>>(kraw, qraw, qnw, pos0, kv, layer, q16);
    // blocks completed in this pass: b with b*KPOOL + KPOOL-1 in [pos0, pos0 + T - 1]
    const int first = pos0 / KPOOL, last = (pos0 + T) / KPOOL - 1;
    if (last >= first) k_idx_pool<<<last - first + 1, 128, 0, st>>>(first, kv, layer, knw);
}

// scores [G][nblk] for G query tokens at positions pos0 + t (block b visible to t when (b+1)*KPOOL <= pos+1)
__global__ __launch_bounds__(256) void k_idx_score(const half *q16, int pos0, int nblk, KVRef kv, int layer, float *score) {
    __shared__ float q[IDX_H][IDX_D];
    const int t = blockIdx.y, b = blockIdx.x * 256 + threadIdx.x, pos = pos0 + t;
    for (int i = threadIdx.x; i < IDX_H * IDX_D; i += 256) q[i / IDX_D][i % IDX_D] = __half2float(q16[(size_t)t * IDX_H * IDX_D + i]);
    __syncthreads();
    if (b >= nblk) return;
    float s = -INFINITY;
    if ((b + 1) * KPOOL <= pos + 1) {
        const half2 *k = (const half2 *)idx_pool(kv, layer, b);
        float acc[IDX_H] = {};
        for (int i = 0; i < IDX_D / 2; i++) {
            const float2 kv2 = __half22float2(k[i]);
#pragma unroll
            for (int h = 0; h < IDX_H; h++) acc[h] += q[h][2 * i] * kv2.x + q[h][2 * i + 1] * kv2.y;
        }
        s = 0;
#pragma unroll
        for (int h = 0; h < IDX_H; h++) s += fmaxf(acc[h], 0.0f);
        s *= 0.08838834764831845f;                       // 1/sqrt(128)
    }
    score[(size_t)t * nblk + b] = s;
}

// Top SEL_BLOCKS of each token's visible blocks by radix select on the score bits (scores >= 0, so their float bits
// order like unsigned ints); ties at the threshold go to the lower block ids. Writes the chosen blocks in ascending
// order, then the tail block (if the token's own block is incomplete), padding with -1: sel [G][SEL_BLOCKS + 1].
__global__ __launch_bounds__(1024) void k_idx_select(const float *score, int pos0, int nblk, int *sel) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t prefix, need, eq_taken;
    __shared__ int nout;
    const int t = blockIdx.x, tid = threadIdx.x, pos = pos0 + t;
    const int nvis = min(nblk, (pos + 1) / KPOOL);
    const float *s = score + (size_t)t * nblk;
    int *out = sel + (size_t)t * (SEL_BLOCKS + 1);
    const int want = min(nvis, SEL_BLOCKS);
    if (tid == 0) { prefix = 0; need = want; nout = 0; eq_taken = 0; }
    __syncthreads();
    // find the bits of the want-th largest score, 8 bits at a time from the top
    uint32_t mask = 0;
    for (int shift = 24; shift >= 0 && want < nvis; shift -= 8) {
        for (int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int b = tid; b < nvis; b += 1024) {
            const uint32_t u = __float_as_uint(s[b]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0; int d = 255;
            for (; d > 0; d--) { if (acc + hist[d] >= need) break; acc += hist[d]; }
            need -= acc; prefix |= (uint32_t)d << shift;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    // collect in ascending block order: everything above the threshold, then the first `need` equal to it
    const uint32_t thr = prefix;
    const bool all = want >= nvis;
    for (int base = 0; base < nvis; base += 1024) {
        const int b = base + tid;
        bool take = false, eq = false;
        if (b < nvis) {
            const uint32_t u = __float_as_uint(s[b]);
            take = all || u > thr;
            eq = !all && u == thr;
        }
        // stable compaction within this 1024-wide window: count takes before me
        const unsigned long long bt = __ballot(take), be = __ballot(eq);
        __shared__ int wt[32], we[32];
        const int wave = tid / 32, lane = tid % 32;
        const unsigned long long below = (1ull << lane) - 1;
        if (lane == 0) { wt[wave] = __popcll(bt); we[wave] = __popcll(be); }
        __syncthreads();
        int ot = 0, oe = 0;
        for (int w = 0; w < wave; w++) { ot += wt[w]; oe += we[w]; }
        int tot_t = 0, tot_e = 0;
        for (int w = 0; w < 32; w++) { tot_t += wt[w]; tot_e += we[w]; }
        // equals are taken in block order while some of `need` remain
        const int my_e = oe + __popcll(be & below);
        const bool take_eq = eq && (int)eq_taken + my_e < (int)need;
        // write order: within the window, keep ascending block order across takes and taken equals
        const unsigned long long bte = __ballot(take || take_eq);
        __shared__ int wte[32];
        if (lane == 0) wte[wave] = __popcll(bte);
        __syncthreads();
        int o = 0;
        for (int w = 0; w < wave; w++) o += wte[w];
        if (take || take_eq) out[nout + o + __popcll(bte & below)] = b;
        __syncthreads();
        if (tid == 0) {
            int add = 0;
            for (int w = 0; w < 32; w++) add += wte[w];
            nout += add;
            eq_taken += min((int)need - (int)eq_taken, tot_e);
            (void)tot_t;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int n = nout;
        if ((pos + 1) % KPOOL != 0) out[n++] = (pos + 1) / KPOOL;   // the tail block (masked past pos)
        for (; n < SEL_BLOCKS + 1; n++) out[n] = -1;
    }
}

void idx_select(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, int *sel, hipStream_t st) {
    k_idx_score<<<dim3((nblk + 255) / 256, G), 256, 0, st>>>(q16, pos0, nblk, kv, layer, score);
    k_idx_select<<<G, 1024, 0, st>>>(score, pos0, nblk, sel);
}

// ---- sparse attention over selected blocks ------------------------------------------------------------
// One wave per (token, kv head, split): rows are the token's 12 query heads (a 16-row tile, 4 rows unused), keys are
// 16 at a time = 4 selected blocks of 4 cells. Same math and partial outputs as k_attn_dec.
__device__ __forceinline__ void v5_load4(const uint8_t *blk, int dim, int slot0, int8_t out[4]) {   // slot0 % 4 == 0
    const uint32_t lo = *(const uint16_t *)(blk + VLO_OFF + dim * (PAGE / 2) + slot0 / 2);
    const uint32_t hi = (*(const uint8_t *)(blk + VHI_OFF + dim * (PAGE / 8) + slot0 / 8)) >> (slot0 % 8);
#pragma unroll
    for (int i = 0; i < 4; i++) out[i] = (int8_t)((((lo >> (4 * i)) & 15) | (((hi >> i) & 1) << 4)) - 16);
}

__global__ __launch_bounds__(32) void k_attn_sel(AttnArgs a, const int *sel, int groups_per_split) {
    const int t = blockIdx.x, g = blockIdx.y, split = blockIdx.z;
    const int lane = threadIdx.x, r16 = lane % 16, hi = lane / 16;
    const bool valid = r16 < QPK;
    const int qpos = a.pos0 + t;
    const half *qrow = a.Q16 + ((size_t)t * NH + g * QPK + (valid ? r16 : 0)) * HD;
    const int *list = sel + (size_t)t * (SEL_BLOCKS + 1);
    const int gbeg = split * groups_per_split, gend = min((SEL_BLOCKS + 1 + 3) / 4, gbeg + groups_per_split);
    float m = -INFINITY, l = 0;
    f8 acc[16];
#pragma unroll
    for (int d = 0; d < 16; d++) acc[d] = f8{};
    for (int grp = gbeg; grp < gend; grp++) {
        // this lane's key (row r16 of the S^T tile): block list[4*grp + r16/4], cell r16 % 4
        const int bi = 4 * grp + r16 / 4;
        const int blk_id = bi < SEL_BLOCKS + 1 ? list[bi] : -1;
        const int cell = blk_id >= 0 ? blk_id * KPOOL + r16 % 4 : 0;
        const uint8_t *page = kv_head(a.kv, a.layer, g, cell);
        const int slot = cell % PAGE;
        const int8_t *kq = (const int8_t *)page + slot * HD;
        const half *ks = (const half *)(page + KS_OFF) + slot * (HD / 32);
        f8 sc = {};
#pragma unroll
        for (int d = 0; d < 16; d++) {
            const int4 kr = *(const int4 *)(kq + d * 16);
            const _Float16 s = (_Float16)__half2float(ks[d / 2]);
            const int8_t *kb = (const int8_t *)&kr;
            h16 A;
#pragma unroll
            for (int i = 0; i < 16; i++) A[i] = (_Float16)kb[i] * s;
            sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, *(const h16 *)(qrow + d * 16), sc);
        }
        // sc[i] = S^T[key 2i+hi][row r16]: key k's block and cell
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int key = 2 * i + hi;
            const int kb_i = 4 * grp + key / 4;
            const int kb_id = kb_i < SEL_BLOCKS + 1 ? list[kb_i] : -1;
            const int kcell = kb_id * KPOOL + key % 4;
            if (kb_id < 0 || kcell > qpos || !valid) sc[i] = -INFINITY;
            mx = fmaxf(mx, sc[i]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m, mx);
        const float alpha = mn == -INFINITY ? 1.0f : __expf(m - mn);
        float p[8], rs = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) { p[i] = mn == -INFINITY ? 0.0f : __expf(sc[i] - mn); rs += p[i]; }
        rs += __shfl_xor(rs, 16, 32);
        l = l * alpha + rs;
        m = mn;
        float P[16];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o = __shfl_xor(p[i], 16, 32);
            P[2 * i] = hi ? o : p[i];
            P[2 * i + 1] = hi ? p[i] : o;
        }
#pragma unroll
        for (int dt = 0; dt < 16; dt++) acc[dt] *= alpha;
        // V: the 16 keys are 4 runs of 4 consecutive cells (one per block)
        const uint8_t *vpage[4]; int vslot[4];
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) {
            const int b2 = 4 * grp + q4;
            const int id2 = b2 < SEL_BLOCKS + 1 ? list[b2] : -1;
            const int c2 = id2 >= 0 ? id2 * KPOOL : 0;
            vpage[q4] = kv_head(a.kv, a.layer, g, c2); vslot[q4] = c2 % PAGE;
        }
#pragma unroll
        for (int blk = 0; blk < 8; blk++) {
            h16 B;
#pragma unroll
            for (int q4 = 0; q4 < 4; q4++) {
                const half *vs = (const half *)(vpage[q4] + VS_OFF) + blk * PAGE + vslot[q4];
#pragma unroll
                for (int i = 0; i < 4; i++) B[4 * q4 + i] = (_Float16)(P[4 * q4 + i] * __half2float(vs[i]));
            }
#pragma unroll
            for (int dt = 2 * blk; dt < 2 * blk + 2; dt++) {
                h16 A;
#pragma unroll
                for (int q4 = 0; q4 < 4; q4++) {
                    int8_t v4[4];
                    v5_load4(vpage[q4], dt * 16 + r16, vslot[q4], v4);
#pragma unroll
                    for (int i = 0; i < 4; i++) A[4 * q4 + i] = (_Float16)v4[i];
                }
                acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, acc[dt]);
            }
        }
    }
    if (!valid) return;
    const size_t row = (size_t)t * NH + g * QPK + r16;
    float *po = a.part_o + ((size_t)split * a.T * NH + row) * HD;
#pragma unroll
    for (int dt = 0; dt < 16; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) po[dt * 16 + 2 * i + hi] = acc[dt][i];
    if (hi == 0) {
        a.part_ml[((size_t)split * a.T * NH + row) * 2] = m;
        a.part_ml[((size_t)split * a.T * NH + row) * 2 + 1] = l;
    }
}

// sparse attention for T tokens at pos0.. with their block lists `sel`; output like attention()
void attention_sel(AttnArgs a, const int *sel, half *oh, QAct oq, hipStream_t st) {
    const int groups = (SEL_BLOCKS + 1 + 3) / 4;
    const int nsplit = std::max(1, std::min(16, 1536 / std::max(1, a.T * NKV)));
    const int gps = (groups + nsplit - 1) / nsplit;
    a.nsplit = (groups + gps - 1) / gps;
    if (a.nsplit * a.T > MAX_SPLITS * 8) FAIL("attention_sel: %d tokens x %d splits exceed the partial buffers", a.T, a.nsplit);
    k_attn_sel<<<dim3(a.T, NKV, a.nsplit), 32, 0, st>>>(a, sel, gps);
    k_attn_combine<<<a.T * NH, 256, 0, st>>>(a.part_o, a.part_ml, a.nsplit, a.T * NH, a.qg, oh, oq);
}
