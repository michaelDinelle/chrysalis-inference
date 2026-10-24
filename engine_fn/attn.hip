// QSA attention layers (12 of 48, plus the MTP layer): q/k RMSNorm + partial RoPE + paged int8 KV-cache writes,
// WMMA flash attention (dense: every cell up to the query), the QSA indexer (pooled block keys, block scores,
// top-512 selection) and sparse attention over the selected blocks plus the tail, and the combine step that merges
// key splits and applies the sigmoid output gate.
//
// Query rows of a KV group are numbered r = t * 12 + j (token t, q head 12g + j), so a 16-row WMMA tile
// mixes heads and tokens freely.
// The kernels compute S^T = K Q^T and O^T = V^T P^T, which keeps each lane on a single query row.
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
    {                                                  // within a wave
        float o;
        o = lane_xor<1>(x); x = (i & 1) ? o - x : x + o;
        o = lane_xor<2>(x); x = (i & 2) ? o - x : x + o;
        o = lane_xor<4>(x); x = (i & 4) ? o - x : x + o;
        o = lane_xor<8>(x); x = (i & 8) ? o - x : x + o;
        o = lane_xor<16>(x); x = (i & 16) ? o - x : x + o;
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

// M-RoPE (Qwen3.8's interleaved sections [11, 11, 10]): of the 32 frequency pairs, f % 3 == 1 (f < 33) turn with the
// row position, f % 3 == 2 (f < 30) with the column, the rest with the sequence position; text has all three equal.
__device__ __forceinline__ int mrope_section(int f) { return f % 3 == 1 && f < 33 ? 1 : f % 3 == 2 && f < 30 ? 2 : 0; }

// grid (NH + 2*NKV jobs, T), 256 threads = one head dim each.
__global__ __launch_bounds__(256) void k_attn_prep(const float *qg, const float *kin, const float *vin, const float *qnw,
                                                   const float *knw, int pos0, int rope0, const int *rpos, KVRef kv, int layer, half *Q16) {
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
            const int rp = rpos ? rpos[t * 3 + mrope_section(f)] : rope0 + t;
            float s, c;
            sincosf(rp * powf(ROPE_BASE, -2.0f * f / ROT), &s, &c);
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
               int rope0, const int *rpos, KVRef kv, int layer, half *Q16, hipStream_t st) {
    k_attn_prep<<<dim3(NH + 2 * NKV, T), 256, 0, st>>>(qg, k, v, qnw, knw, pos0, rope0, rpos, kv, layer, Q16);
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
        mx = fmaxf(mx, lane_xor<16>(mx));
        if (mx > m) {
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
        l += rs + lane_xor<16>(rs);
        h16 P0, P1;                                               // P^T[key][row r16] for keys 0-15 and 16-31
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o0 = lane_xor<16>(p0[i]), o1 = lane_xor<16>(p1[i]);
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
        mx = fmaxf(mx, lane_xor<16>(mx));
        const float mn = fmaxf(m, mx);
        const float alpha = mn == -INFINITY ? 1.0f : __expf(m - mn);
        float p[8], rs = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) { p[i] = mn == -INFINITY ? 0.0f : __expf(sc[i] - mn); rs += p[i]; }
        rs += lane_xor<16>(rs);
        l = l * alpha + rs;
        m = mn;
        float P[16];                                              // this row's probabilities for keys 0..15
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o = lane_xor<16>(p[i]);
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
// Sets split_len itself (whatever the caller passed) and always merges the splits in k_attn_combine: fp16 and/or
// int8 output for decode / verify, fp16 only for prefill.
void attention(AttnArgs a, half *oh, QAct oq, hipStream_t st) {
    const int ntiles = (a.T * QPK + 15) / 16;
    const int nkeys = a.pos0 + a.T;
    a.out_h = oh;
    if (a.T <= 8) {                       // decode / verify: enough splits to fill the GPU, at most MAX_SPLITS
        const int target = std::min({MAX_SPLITS, PART_ROWS / a.T, std::max(1, 1536 / (ntiles * NKV))});
        a.split_len = std::max(16, ((nkeys + target - 1) / target + 15) / 16 * 16);   // short contexts: many short splits
        a.nsplit = (nkeys + a.split_len - 1) / a.split_len;
        a.partial = 1;
        k_attn_dec<<<dim3(ntiles, NKV, a.nsplit), 32, 0, st>>>(a);
    } else {                              // prefill: big workgroups share K/V stages; split keys if too few
        if (oq.q) FAIL("attention: prefill writes fp16 output");
        const int aw = ntiles >= 384 ? 16 : ntiles >= 96 ? 8 : 4;
        const int wgs = (ntiles + aw - 1) / aw * NKV;
        a.nsplit = std::max(1, std::min({(192 + wgs - 1) / wgs, nkeys / 1024, PART_ROWS / a.T}));
        if (a.nsplit * a.T > PART_ROWS) FAIL("attention: chunk of %d tokens exceeds the partial buffers", a.T);
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

// QSA indexer.
// Each cell keeps its raw indexer key (fp16, unnormed, unrotated); every complete block of KPOOL cells keeps one
// pooled key: the mean of its raw keys, RMS-normed and rotated to the position of its first cell. A query's
// indexer heads (normed, rotated) score block b as sum_h relu(q_h . k_b) / sqrt(IDX_D); the query attends the
// cells of its top SEL_BLOCKS complete visible blocks plus the incomplete tail block it sits in.

__device__ __forceinline__ half *idx_pool(const KVRef &kv, int layer, int block) {
    const int cell = block * KPOOL;
    return (half *)(kv.pool + (size_t)kv.pages[cell / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + NKV * KV_HEAD) +
           (size_t)((cell % PAGE) / KPOOL) * IDX_D;
}

// rmsnorm over 128 values (one per thread of a 128-thread block) times w, then NeoX rotation of the first ROT dims at the
// M-RoPE position p
__device__ __forceinline__ float norm_rope128(float x, const float *w, Pos3 p, float *red, float *buf) {
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
        const int f = i % (ROT / 2), sec = mrope_section(f);
        float s, c;
        sincosf((sec == 0 ? p.t : sec == 1 ? p.h : p.w) * powf(ROPE_BASE, -2.0f * f / ROT), &s, &c);
        x = i < ROT / 2 ? x * c - buf[i + ROT / 2] * s : x * c + buf[i - ROT / 2] * s;
    }
    __syncthreads();
    return x;
}
__device__ __forceinline__ Pos3 pos_at(int t, int rope0, const int *rpos) {
    return rpos ? Pos3{rpos[t * 3], rpos[t * 3 + 1], rpos[t * 3 + 2]} : Pos3{rope0 + t, rope0 + t, rope0 + t};
}

// grid (1 + IDX_H, T), 128 threads: job 0 stores the raw key of token t (fp16, in the pass buffer), jobs 1.. norm +
// rotate its query heads
__global__ __launch_bounds__(128) void k_idx_prep(const float *kraw, const float *qraw, const float *qnw, int rope0, const int *rpos,
                                                  half *praw, half *q16) {
    __shared__ float red[4], buf[ROT];
    const int job = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    if (job == 0) {
        praw[(size_t)t * IDX_D + i] = __float2half(kraw[(size_t)t * IDX_D + i]);
        return;
    }
    const int h = job - 1;
    const float x = norm_rope128(qraw[((size_t)t * IDX_H + h) * IDX_D + i], qnw, pos_at(t, rope0, rpos), red, buf);
    q16[((size_t)t * IDX_H + h) * IDX_D + i] = __float2half(x);
}

// one block per pooled key completed by tokens pos0 .. pos0+T-1 (blocks b0 ..), 128 threads; raw keys of cells before
// pos0 come from the sequence's tail (slot = cell % KPOOL), the rest from this pass's buffer; the key turns with its first
// cell's position (tail_pos when that cell came before the pass)
__global__ __launch_bounds__(128) void k_idx_pool(int b0, int pos0, int rope0, const int *rpos, Pos3 tail_pos, const half *tail,
                                                  const half *praw, KVRef kv, int layer, const float *knw) {
    __shared__ float red[4], buf[ROT];
    const int b = b0 + blockIdx.x, i = threadIdx.x;
    float x = 0;
    for (int c = 0; c < KPOOL; c++) {
        const int cell = b * KPOOL + c;
        x += __half2float(cell < pos0 ? tail[(cell % KPOOL) * IDX_D + i] : praw[(size_t)(cell - pos0) * IDX_D + i]);
    }
    const int first = b * KPOOL;
    x = norm_rope128(x * (1.0f / KPOOL), knw, first < pos0 ? tail_pos : pos_at(first - pos0, rope0, rpos), red, buf);
    idx_pool(kv, layer, b)[i] = __float2half(x);
}

// the tail after the cells before n_end are final: raw keys of n_end's incomplete block that this pass wrote. Grid:
// layers (tail / praw strides per layer), 128 threads.
__global__ __launch_bounds__(128) void k_idx_tail(half *tail, const half *praw, size_t praw_layer, int pos0, int n_end) {
    half *tl = tail + (size_t)blockIdx.x * IDX_TAIL;
    const half *pr = praw + (size_t)blockIdx.x * praw_layer;
    for (int cell = max(pos0, n_end / KPOOL * KPOOL); cell < n_end; cell++)
        tl[(cell % KPOOL) * IDX_D + threadIdx.x] = pr[(size_t)(cell - pos0) * IDX_D + threadIdx.x];
}

void idx_prep(const float *kraw, const float *qraw, const float *qnw, const float *knw, int T, int pos0, int rope0, const int *rpos,
              Pos3 tail_pos, KVRef kv, int layer, half *tail, half *praw, bool keep_tail, half *q16, hipStream_t st) {
    k_idx_prep<<<dim3(1 + IDX_H, T), 128, 0, st>>>(kraw, qraw, qnw, rope0, rpos, praw, q16);
    // blocks completed in this pass: b with b*KPOOL + KPOOL-1 in [pos0, pos0 + T - 1]
    const int first = pos0 / KPOOL, last = (pos0 + T) / KPOOL - 1;
    if (last >= first) k_idx_pool<<<last - first + 1, 128, 0, st>>>(first, pos0, rope0, rpos, tail_pos, tail, praw, kv, layer, knw);
    if (keep_tail) k_idx_tail<<<1, 128, 0, st>>>(tail, praw, 0, pos0, pos0 + T);
}

void idx_tail_commit(half *tails, const half *praw, size_t praw_layer, int layers, int pos0, int n_end, hipStream_t st) {
    k_idx_tail<<<layers, 128, 0, st>>>(tails, praw, praw_layer, pos0, n_end);
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

// Selection at long context (decode: 65K blocks per QSA layer at 262K).
// Scores: 16 lanes per block (16-byte loads, coalesced), 16 blocks per workgroup step; each workgroup also counts the
// top 10 bits of its scores (non-negative floats order like their bits) and adds its non-empty bins to the token's
// histogram, so the selection starts from a 1024-bin histogram instead of 4 radix passes over every score.
constexpr int SEL_BINS = 1024;
__global__ __launch_bounds__(256) void k_idx_score2(const half *q16, int pos0, int nblk, KVRef kv, int layer, float *score,
                                                    uint32_t *hist) {
    __shared__ uint32_t lh[SEL_BINS];
    const int t = blockIdx.y, tid = threadIdx.x, g = tid / 16, l = tid % 16, pos = pos0 + t;
    const int nvis = min(nblk, (pos + 1) / KPOOL);
    for (int i = tid; i < SEL_BINS; i += 256) lh[i] = 0;
    float q[IDX_H][8];
#pragma unroll
    for (int h = 0; h < IDX_H; h++) {
        const int4 v = *(const int4 *)(q16 + ((size_t)t * IDX_H + h) * IDX_D + l * 8);
        const half2 *hv = (const half2 *)&v;
#pragma unroll
        for (int k = 0; k < 4; k++) { const float2 f = __half22float2(hv[k]); q[h][2 * k] = f.x; q[h][2 * k + 1] = f.y; }
    }
    __syncthreads();
    const int b0 = blockIdx.x * 256;
#pragma unroll 4
    for (int it = 0; it < 16; it++) {
        const int b = b0 + it * 16 + g;
        if (b >= nblk) break;
        float acc[IDX_H] = {};
        if (b < nvis) {
            const int4 v = *(const int4 *)(idx_pool(kv, layer, b) + l * 8);
            const half2 *hv = (const half2 *)&v;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float2 f = __half22float2(hv[k]);
#pragma unroll
                for (int h = 0; h < IDX_H; h++) acc[h] += q[h][2 * k] * f.x + q[h][2 * k + 1] * f.y;
            }
        }
#pragma unroll
        for (int h = 0; h < IDX_H; h++) {             // sums over 16 lanes
            acc[h] += lane_xor<8>(acc[h]); acc[h] += lane_xor<4>(acc[h]); acc[h] += lane_xor<2>(acc[h]); acc[h] += lane_xor<1>(acc[h]);
        }
        if (l == 0) {
            float sc = -INFINITY;
            if (b < nvis) {
                sc = 0;
#pragma unroll
                for (int h = 0; h < IDX_H; h++) sc += fmaxf(acc[h], 0.0f);
                sc *= 0.08838834764831845f;           // 1/sqrt(128)
                atomicAdd(&lh[__float_as_uint(sc) >> 22], 1u);
            }
            score[(size_t)t * nblk + b] = sc;
        }
    }
    __syncthreads();
    for (int i = tid; i < SEL_BINS; i += 256) if (lh[i]) atomicAdd(&hist[(size_t)t * SEL_BINS + i], lh[i]);
}

// Top SEL_BLOCKS from the histogram: the threshold bin from suffix sums, its exact threshold value by radix passes over
// the scores in that bin only (bits 21..0), then one compaction pass in block order (each thread a contiguous range,
// one scan): blocks above the threshold, plus the first `need` equal to it (ties to the lower block ids). Same output
// as k_idx_select.
// inclusive prefix sum over a wave (32 lanes)
__device__ __forceinline__ int wave_incl(int v, int lane) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int u = __shfl_up(v, o, 32); if (lane >= o) v += u; }
    return v;
}
// exclusive prefix sum over the 1024 threads of a workgroup (2 barriers); total comes back in *tot
__device__ __forceinline__ int block_excl(int v, int *wsum, int *tot) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int inc = wave_incl(v, lane);
    if (lane == 31) wsum[wave] = inc;
    __syncthreads();
    if (wave == 0) {
        const int w = wave_incl(wsum[lane], lane);
        wsum[lane] = w;                               // inclusive per wave
    }
    __syncthreads();
    *tot = wsum[31];
    return inc - v + (wave ? wsum[wave - 1] : 0);
}
// Wave 0 finds, in a histogram of n bins (counts per digit), the digit d holding the need-th largest element counting
// from the top digit down: returns d and the count of elements above it (all lanes of wave 0 get the result).
__device__ __forceinline__ void digit_of(const uint32_t *h, int n, uint32_t need, int lane, int &d_out, uint32_t &above) {
    const int per = n / 32;                           // bins per lane, lane 0 owns the top ones
    uint32_t loc = 0;
    for (int k = 0; k < per; k++) loc += h[n - 1 - (lane * per + k)];
    const int inc = wave_incl((int)loc, lane), exc = inc - (int)loc;
    const bool mine = (uint32_t)exc < need && (uint32_t)inc >= need;
    const unsigned long long m = __ballot(mine);
    const int owner = m ? __ffsll((long long)m) - 1 : 31;
    int d = 0; uint32_t ab = 0;
    if (lane == owner) {
        uint32_t acc = (uint32_t)exc;
        int k = 0;
        for (; k < per - 1; k++) { const uint32_t c = h[n - 1 - (lane * per + k)]; if (acc + c >= need) break; acc += c; }
        d = n - 1 - (lane * per + k); ab = acc;
    }
    d_out = __shfl(d, owner, 32); above = (uint32_t)__shfl((int)ab, owner, 32);
}

__global__ __launch_bounds__(1024) void k_idx_select2(const float *score, const uint32_t *hist, int pos0, int nblk, int *sel) {
    __shared__ uint32_t hb[SEL_BINS], h256[256], hw[32][256];   // per-wave sub-histograms: ties do not serialize
    __shared__ uint32_t s_prefix, s_need, s_mask;
    __shared__ int wsum[32];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid % 32, wave = tid / 32, pos = pos0 + t;
    const int nvis = min(nblk, (pos + 1) / KPOOL);
    const float *s = score + (size_t)t * nblk;
    int *out = sel + (size_t)t * (SEL_BLOCKS + 1);
    const int want = min(nvis, SEL_BLOCKS);
    const bool all = want >= nvis;
    if (!all) {
        hb[tid] = hist[(size_t)t * SEL_BINS + tid];
        __syncthreads();
        if (wave == 0) {                              // the threshold bin from the top 10 bits
            int d; uint32_t above;
            digit_of(hb, SEL_BINS, (uint32_t)want, lane, d, above);
            if (lane == 0) { s_prefix = (uint32_t)d << 22; s_mask = 0xFFC00000u; s_need = want - above; }
        }
        __syncthreads();
        // the remaining 22 bits, 8 + 8 + 6, over the scores matching the prefix so far
        for (int shift = 14; shift >= -6; shift -= 8) {
            const int sh_ = shift < 0 ? 0 : shift, bits = shift < 0 ? 6 : 8;
            for (int i = tid; i < 32 * 256; i += 1024) hw[i / 256][i % 256] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix, mask = s_mask;
            for (int b0 = tid; b0 < nvis; b0 += 8 * 1024) {     // 8 loads in flight per thread
                uint32_t u[8];
#pragma unroll
                for (int k = 0; k < 8; k++) { const int b = b0 + k * 1024; u[k] = b < nvis ? __float_as_uint(s[b]) : ~prefix; }
#pragma unroll
                for (int k = 0; k < 8; k++)
                    if ((u[k] & mask) == prefix) atomicAdd(&hw[wave][(u[k] >> sh_) & ((1u << bits) - 1)], 1u);
            }
            __syncthreads();
            if (tid < 256) { uint32_t c = 0; for (int v = 0; v < 32; v++) c += hw[v][tid]; h256[tid] = c; }
            __syncthreads();
            if (wave == 0) {
                int d; uint32_t above;
                digit_of(h256, 1 << bits, s_need, lane, d, above);
                if (lane == 0) { s_need -= above; s_prefix = prefix | ((uint32_t)d << sh_); s_mask = mask | (((1u << bits) - 1) << sh_); }
            }
            __syncthreads();
        }
    }
    const uint32_t thr = all ? 0 : s_prefix, need = all ? 0 : s_need;
    // compaction in block order: wave w owns blocks [lo, hi), its lanes read them together (coalesced); order inside a
    // wave from ballots, across waves from one scan
    const int per = ((nvis + 31) / 32 + 31) / 32 * 32, lo = min(nvis, wave * per), hi = min(nvis, lo + per);
    int n_take = 0, n_eq = 0;                         // per wave (lane 0 holds the sums)
    for (int b0 = lo; b0 < hi; b0 += 32 * 4) {
        uint32_t u[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const int b = b0 + k * 32 + lane; u[k] = b < hi ? __float_as_uint(s[b]) : 0u; }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int b = b0 + k * 32 + lane;
            n_take += __popcll(__ballot(b < hi && (all || u[k] > thr)));
            n_eq += __popcll(__ballot(b < hi && !all && u[k] == thr));
        }
    }
    __shared__ int w_take[32], w_eq[32];
    if (lane == 0) { w_take[wave] = n_take; w_eq[wave] = n_eq; }
    __syncthreads();
    int eq_before = 0, out_before = 0;
    for (int v = 0; v < wave; v++) eq_before += w_eq[v];
    int tot = 0;
    for (int v = 0; v < 32; v++) {
        const int e = max(0, min(w_eq[v], (int)need - (tot == 0 && v == 0 ? 0 : 0)));
        (void)e;
    }
    // equals taken by the waves before mine, then my own output offset
    {
        int eqb = 0, ob = 0;
        for (int v = 0; v < 32; v++) {
            const int taken_eq = max(0, min(w_eq[v], (int)need - eqb));
            if (v < wave) ob += w_take[v] + taken_eq;
            tot += w_take[v] + taken_eq;
            eqb += w_eq[v];
        }
        out_before = ob;
    }
    int w = out_before, eq_seen = eq_before;          // running output position and equal rank (block order)
    for (int b0 = lo; b0 < hi; b0 += 32 * 4) {
        uint32_t u[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const int b = b0 + k * 32 + lane; u[k] = b < hi ? __float_as_uint(s[b]) : 0u; }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int b = b0 + k * 32 + lane;
            const bool in = b < hi;
            const bool eq = in && !all && u[k] == thr;
            const unsigned long long me = __ballot(eq), below = (1ull << lane) - 1;
            const bool take_eq = eq && eq_seen + __popcll(me & below) < (int)need;
            const bool take = in && (all || u[k] > thr || take_eq);
            const unsigned long long mt = __ballot(take);
            if (take) out[w + __popcll(mt & below)] = b;
            w += __popcll(mt);
            eq_seen += __popcll(me);
        }
    }
    if (tid == 0) {
        int n = tot;
        if ((pos + 1) % KPOOL != 0) out[n++] = (pos + 1) / KPOOL;   // the tail block (masked past pos)
        for (; n < SEL_BLOCKS + 1; n++) out[n] = -1;
    }
}

// Scores for G > 1 query tokens (prefill) as WMMA tiles: a wave computes 16 tokens x 16 blocks, the 16 pooled keys
// loaded once as B fragments and used for all 4 indexer heads (A = the tokens' query heads), so the keys are read once
// per 16 tokens instead of once per token. Workgroup: 4 waves = 64 blocks.
__global__ __launch_bounds__(128) void k_idx_score_mm(const half *q16, int G, int pos0, int nblk, KVRef kv, int layer, float *score) {
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32, r16 = lane % 16, hi = lane / 16;
    const int t0 = blockIdx.y * 16, b = (blockIdx.x * 4 + wave) * 16 + r16;
    // B fragments: block b's key, 8 slices of 16 dims (blocks past nblk read block 0, masked below)
    const half *kp = idx_pool(kv, layer, b < nblk ? b : 0);
    h16 kb[8];
#pragma unroll
    for (int k = 0; k < 8; k++) kb[k] = *(const h16 *)(kp + k * 16);
    const int ta = min(t0 + r16, G - 1);               // A row: token t0 + r16 (rows past G repeat the last token)
    f8 sum = {};
#pragma unroll
    for (int h = 0; h < IDX_H; h++) {
        f8 acc = {};
        const half *qp = q16 + ((size_t)ta * IDX_H + h) * IDX_D;
#pragma unroll
        for (int k = 0; k < 8; k++) acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const h16 *)(qp + k * 16), kb[k], acc);
#pragma unroll
        for (int i = 0; i < 8; i++) sum[i] += fmaxf(acc[i], 0.0f);
    }
    // lane holds tokens t0 + 2i + hi, block b
    if (b >= nblk) return;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int t = t0 + 2 * i + hi;
        if (t >= G) continue;
        const bool vis = (b + 1) * KPOOL <= pos0 + t + 1;
        score[(size_t)t * nblk + b] = vis ? sum[i] * 0.08838834764831845f : -INFINITY;
    }
}

// the per-token histograms of the top 10 score bits, from a score matrix (one workgroup per token and 4096 blocks)
__global__ __launch_bounds__(256) void k_idx_hist(const float *score, int pos0, int nblk, uint32_t *hist) {
    __shared__ uint32_t lh[SEL_BINS];
    const int t = blockIdx.y, tid = threadIdx.x;
    const int nvis = min(nblk, (pos0 + t + 1) / KPOOL);
    for (int i = tid; i < SEL_BINS; i += 256) lh[i] = 0;
    __syncthreads();
    const int b1 = min(nvis, (int)(blockIdx.x + 1) * 4096);
    for (int b = blockIdx.x * 4096 + tid; b < b1; b += 256) atomicAdd(&lh[__float_as_uint(score[(size_t)t * nblk + b]) >> 22], 1u);
    __syncthreads();
    for (int i = tid; i < SEL_BINS; i += 256) if (lh[i]) atomicAdd(&hist[(size_t)t * SEL_BINS + i], lh[i]);
}

void idx_select(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, uint32_t *hist, int *sel, hipStream_t st) {
    CK(hipMemsetAsync(hist, 0, (size_t)G * SEL_BINS * 4, st));
    if (G == 1) k_idx_score2<<<dim3((nblk + 255) / 256, 1), 256, 0, st>>>(q16, pos0, nblk, kv, layer, score, hist);
    else {                                            // several tokens: WMMA tiles, then their histograms
        k_idx_score_mm<<<dim3((nblk + 63) / 64, (G + 15) / 16), 128, 0, st>>>(q16, G, pos0, nblk, kv, layer, score);
        k_idx_hist<<<dim3((nblk + 4095) / 4096, G), 256, 0, st>>>(score, pos0, nblk, hist);
    }
    k_idx_select2<<<G, 1024, 0, st>>>(score, hist, pos0, nblk, sel);
}
// The two-kernel selection (k_idx_score, k_idx_select): the reference tests/test_attn_fn.cpp compares against.
void idx_select_ref(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, int *sel, hipStream_t st) {
    k_idx_score<<<dim3((nblk + 255) / 256, G), 256, 0, st>>>(q16, pos0, nblk, kv, layer, score);
    k_idx_select<<<G, 1024, 0, st>>>(score, pos0, nblk, sel);
}

// Sparse attention over the selected blocks.
// One 256-thread workgroup (8 waves) per (token, kv head, split of the token's block list); the selected blocks
// (SEL_BLOCKS of KPOOL = 4 cells, then the tail block) in stages of 32 blocks = 128 keys. As the 27B engine's decode
// kernel: wave w scores keys 16w..16w+15 of the stage (4 blocks) for the token's 12 query heads (one 16-row tile), the
// waves agree on each row's running max through LDS and publish P^T (fp16), and wave w accumulates O^T for its own 32
// dims (one V scale block) over the stage's keys, so a wave keeps 16 registers of output instead of 128. Keys come in
// runs of 4 cells: V is 2 bytes of nibbles per run and dim, decoded two cells per instruction (fragment order VPERM4).
constexpr int SST = 32;                       // selected blocks per stage (128 keys)
__device__ constexpr int VPERM4[16] = {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15};
union Frag { h16 h; uint32_t u[8]; };
__device__ __forceinline__ h16 k8_frag(int4 kr, _Float16 s) {   // 16 int8 K values times their scale, as (_Float16)k * s
    const uint32_t w[4] = {(uint32_t)kr.x ^ 0x80808080u, (uint32_t)kr.y ^ 0x80808080u, (uint32_t)kr.z ^ 0x80808080u,
                           (uint32_t)kr.w ^ 0x80808080u};
    Frag f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        f.u[2 * q] = __builtin_amdgcn_perm(0x64646464u, w[q], 0x04010400u);
        f.u[2 * q + 1] = __builtin_amdgcn_perm(0x64646464u, w[q], 0x04030402u);
    }
    return (f.h - (_Float16)1152) * s;
}

__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(10))) void k_attn_sel(AttnArgs a, const int *sel, int stages_per_split) {
    __shared__ _Float16 Qs[16][HD + 8];
    __shared__ float rmx[8][16];
    __shared__ _Float16 Ps[8][16][16 + 8];    // P^T per key tile: [tile][row][16 fragment slots, keys VPERM4]
    __shared__ int bid[SEL_BLOCKS + 1], bpg[SEL_BLOCKS + 1];   // this split's blocks: id (-1: none), page
    const int t = blockIdx.x, g = blockIdx.y, split = blockIdx.z, tid = threadIdx.x;
    const int w = tid / 32, lane = tid % 32, r16 = lane % 16, hi = lane / 16;
    const int qpos = a.pos0 + t, rowvalid = r16 < QPK;
    const int *list = sel + (size_t)t * (SEL_BLOCKS + 1);
    const int bbeg = split * stages_per_split * SST, nb = min(SEL_BLOCKS + 1, bbeg + stages_per_split * SST) - bbeg;
    for (int i = tid; i < 16 * (HD / 8); i += 256) {
        const int r = i / (HD / 8), c = i % (HD / 8);
        int4 v = {0, 0, 0, 0};
        if (r < QPK) v = *(const int4 *)(a.Q16 + ((size_t)t * NH + g * QPK + r) * HD + c * 8);
        *(int4 *)&Qs[r][c * 8] = v;
    }
    for (int i = tid; i < nb; i += 256) {
        const int id = list[bbeg + i];
        bid[i] = id;
        bpg[i] = id >= 0 ? a.kv.pages[id * KPOOL / PAGE] : a.kv.pages[0];
    }
    float m = -INFINITY, l = 0;
    f8 acc[2] = {f8{}, f8{}};
    __syncthreads();
    const size_t head_off = (size_t)a.layer * KV_LAYER + (size_t)g * KV_HEAD;
    auto block_base = [&](int i) { return a.kv.pool + (size_t)bpg[i] * KV_PAGE + head_off; };

    for (int s0 = 0; s0 < nb; s0 += SST) {
        // S^T for this wave's keys: blocks s0 + 4w .. s0 + 4w + 3, cell r16 % 4 of block r16 / 4
        const int bi = s0 + 4 * w + r16 / 4;
        const int id = bi < nb ? bid[bi] : -1;
        f8 sc = {};
        {
            const uint8_t *page = block_base(bi < nb ? bi : 0);
            const int slot = (id >= 0 ? id * KPOOL % PAGE : 0) + r16 % 4;
            const int8_t *kq = (const int8_t *)page + slot * HD;
            const half *ks = (const half *)(page + KS_OFF) + slot * (HD / 32);
#pragma unroll
            for (int d = 0; d < 16; d++) {
                const h16 A = k8_frag(*(const int4 *)(kq + d * 16), (_Float16)__half2float(ks[d / 2]));
                sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, *(const h16 *)&Qs[r16][d * 16], sc);
            }
        }
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) {                // sc[i] = S^T[key 2i+hi][row r16]
            const int key = 2 * i + hi, kb = s0 + 4 * w + key / 4;
            const int kid = kb < nb ? bid[kb] : -1;
            if (kid < 0 || kid * KPOOL + key % 4 > qpos || !rowvalid) sc[i] = -INFINITY;
            mx = fmaxf(mx, sc[i]);
        }
        mx = fmaxf(mx, lane_xor<16>(mx));
        if (hi == 0) rmx[w][r16] = mx;
        __syncthreads();
        {
            float M = m;
#pragma unroll
            for (int ww = 0; ww < 8; ww++) M = fmaxf(M, rmx[ww][r16]);
            const float alpha = M == -INFINITY ? 1.0f : __expf(m - M);
            m = M;
            float p[8], rs = 0;
#pragma unroll
            for (int i = 0; i < 8; i++) { p[i] = M == -INFINITY ? 0.0f : __expf(sc[i] - M); rs += p[i]; }
            l = l * alpha + rs + lane_xor<16>(rs);
            acc[0] *= alpha; acc[1] *= alpha;
            float P16[16];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float o = lane_xor<16>(p[i]);
                P16[2 * i] = hi ? o : p[i];
                P16[2 * i + 1] = hi ? p[i] : o;
            }
            h16 Ph;                                  // lane hi writes fragment slots 8hi .. 8hi+7
#pragma unroll
            for (int j = 0; j < 8; j++) Ph[j] = (_Float16)(hi ? P16[VPERM4[8 + j]] : P16[VPERM4[j]]);
            *(uint4 *)&Ps[w][r16][8 * hi] = *(const uint4 *)&Ph;
        }
        __syncthreads();
#pragma unroll 2
        for (int kt = 0; kt < 8; kt++) {             // O^T for dims 32w + 16dt + r16 over the stage's key tiles
            if (s0 + 4 * kt >= nb) break;
            const uint8_t *pq[4]; int sl[4];
            h16 vsp;                                 // the keys' V scales (block w) in fragment order
#pragma unroll
            for (int q4 = 0; q4 < 4; q4++) {
                const int b2 = s0 + 4 * kt + q4, id2 = b2 < nb ? bid[b2] : -1;
                pq[q4] = block_base(b2 < nb ? b2 : 0); sl[q4] = id2 >= 0 ? id2 * KPOOL % PAGE : 0;
                const half *vs = (const half *)(pq[q4] + VS_OFF) + w * PAGE + sl[q4];
                vsp[4 * q4] = (_Float16)__half2float(vs[0]); vsp[4 * q4 + 1] = (_Float16)__half2float(vs[2]);
                vsp[4 * q4 + 2] = (_Float16)__half2float(vs[1]); vsp[4 * q4 + 3] = (_Float16)__half2float(vs[3]);
            }
            const h16 B = *(const h16 *)&Ps[kt][r16][0] * vsp;
#pragma unroll
            for (int dt = 0; dt < 2; dt++) {
                const int dim = 32 * w + 16 * dt + r16;
                Frag f;
#pragma unroll
                for (int q4 = 0; q4 < 4; q4++) {
                    const uint32_t lo16 = *(const uint16_t *)(pq[q4] + VLO_OFF + dim * (PAGE / 2) + sl[q4] / 2);
                    const uint32_t hb4 = (*(const uint8_t *)(pq[q4] + VHI_OFF + dim * (PAGE / 8) + sl[q4] / 8) >> (sl[q4] % 8)) & 15u;
                    const uint32_t y = (lo16 & 0xffu) | ((lo16 & 0xff00u) << 8);       // nibbles at bits 0, 4, 16, 20
                    f.u[2 * q4] = (y & 0x000f000fu) | ((hb4 & 1u) << 4) | ((hb4 & 4u) << 18) | 0x64006400u;          // cells 0, 2
                    f.u[2 * q4 + 1] = ((y >> 4) & 0x000f000fu) | ((hb4 & 2u) << 3) | ((hb4 & 8u) << 17) | 0x64006400u;  // cells 1, 3
                }
                acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(f.h - (_Float16)1040, B, acc[dt]);
            }
        }
    }
    // partials: O^T rows of this wave's dims; the row sums add up over the waves
    if (hi == 0) rmx[w][r16] = l;
    __syncthreads();
    if (!rowvalid) return;
    const size_t row = (size_t)t * NH + g * QPK + r16;
    float *po = a.part_o + ((size_t)split * a.T * NH + row) * HD + 32 * w;
#pragma unroll
    for (int dt = 0; dt < 2; dt++)
#pragma unroll
        for (int i = 0; i < 8; i++) po[dt * 16 + 2 * i + hi] = acc[dt][i];
    if (w == 0 && hi == 0) {
        float L = 0;
#pragma unroll
        for (int ww = 0; ww < 8; ww++) L += rmx[ww][r16];
        a.part_ml[((size_t)split * a.T * NH + row) * 2] = m;
        a.part_ml[((size_t)split * a.T * NH + row) * 2 + 1] = L;
    }
}

// sparse attention for T tokens at pos0.. with their block lists `sel`; output like attention()
void attention_sel(AttnArgs a, const int *sel, half *oh, QAct oq, hipStream_t st) {
    const int stages = (SEL_BLOCKS + 1 + SST - 1) / SST;
    const int nsplit = std::max(1, std::min({stages, 192 / std::max(1, a.T * NKV), PART_ROWS / a.T}));   // decode: a stage per workgroup
    const int sps = (stages + nsplit - 1) / nsplit;
    a.nsplit = (stages + sps - 1) / sps;
    if (a.nsplit * a.T > PART_ROWS) FAIL("attention_sel: %d tokens x %d splits exceed the partial buffers", a.T, a.nsplit);
    k_attn_sel<<<dim3(a.T, NKV, a.nsplit), 256, 0, st>>>(a, sel, sps);
    k_attn_combine<<<a.T * NH, 256, 0, st>>>(a.part_o, a.part_ml, a.nsplit, a.T * NH, a.qg, oh, oq);
}
