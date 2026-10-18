// Full-attention layers (16 of 64, plus the MTP layer): q/k RMSNorm + partial RoPE + paged int8
// KV-cache writes, WMMA flash-attention kernels for prefill and for decode / verify, and the combine
// step that merges key splits and applies the sigmoid output gate.
//
// Query rows of a KV group are numbered r = t * 6 + j (token t, q head 6g + j), so a 16-row WMMA tile
// mixes heads and tokens freely. Both kernels compute S^T = K Q^T and O^T = V^T P^T, which keeps each
// lane on a single query row.
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

// Decode A fragments with packed fp16 arithmetic (the decode kernel is bound by these instructions, not by WMMA).
// V: 16 consecutive keys of one dimension, in key order VPERM (nibble pairs k, k+4 share a 32-bit word): each pair is
// one and-or into fp16 1024 + q (q = nibble + 16 * high bit), then one packed subtract of 1040.
// K: 16 int8 values times their block scale: bytes biased to unsigned, spread into fp16 lanes 1024 + u by v_perm, then
// an exact packed subtract (1152) and one rounding multiply, the same values as (_Float16)k * s.
__device__ constexpr int VPERM[16] = {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15};
union Frag { h16 h; uint32_t u[8]; };
__device__ __forceinline__ h16 v5_frag_raw(uint2 lo, uint32_t hb) {
    Frag f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const uint32_t x = h ? lo.y : lo.x, b = (hb >> (8 * h)) & 0xffu, hs = (b << 4) | (b << 16);
#pragma unroll
        for (int j = 0; j < 4; j++) f.u[4 * h + j] = ((x >> (4 * j)) & 0x000f000fu) | ((hs >> j) & 0x00100010u) | 0x64006400u;
    }
    return f.h - (_Float16)1040;
}
__device__ __forceinline__ h16 k8_frag(int4 kr, _Float16 s) {
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

// grid (NH + 2*NKV jobs, T), 256 threads = one head dim each.
// M-RoPE (Qwen3.8's interleaved sections [11, 11, 10]): of the 32 frequency pairs, f % 3 == 1 (f < 33) turn with the
// row position, f % 3 == 2 (f < 30) with the column, the rest with the sequence position; text has all three equal.
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
            const int sec = f % 3 == 1 && f < 33 ? 1 : f % 3 == 2 && f < 30 ? 2 : 0;
            const int rp = rpos ? rpos[t * 3 + sec] : rope0 + t;
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
// a row's running max actually grows. A stage's raw K/V bytes are loaded into registers during the previous
// stage's softmax and P.V (after its Q loads: vector loads complete in order), so the DRAM latency hides
// behind WMMA work instead of stalling every stage.
constexpr int KST = 32;                       // keys per LDS stage
constexpr int VLD32 = KST + 8;                // padded LDS row for the transposed V stage

template <int AW>
struct AttnRaw {                              // one thread's share of a stage's raw K/V bytes
    static constexpr int NT = AW * 32, KJ = (KST * 8 + NT - 1) / NT, VJ = (HD * 2 + NT - 1) / NT;
    int4 k[KJ][2]; float kd[KJ];              // K: one 32-dim block of one key per job (int8), its scale
    uint2 vlo[VJ]; uint32_t vhi[VJ]; int4 vs[VJ][2];   // V^T: 16 keys of one dim per job (5-bit), their scales
    __device__ void load(const uint8_t *page, int slot0, int nvalid, int tid) {
#pragma unroll
        for (int j = 0; j < KJ; j++) {
            const int job = tid + j * NT, kk = job >> 3, blk = job & 7;
            if (job < KST * 8) {
                const int8_t *q = (const int8_t *)page + (slot0 + kk) * HD + blk * 32;
                k[j][0] = *(const int4 *)q; k[j][1] = *(const int4 *)(q + 16);
                kd[j] = kk < nvalid ? __half2float(((const half *)(page + KS_OFF))[(slot0 + kk) * (HD / 32) + blk]) : 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < VJ; j++) {
            const int job = tid + j * NT, dim = job >> 1, s16 = slot0 + (job & 1) * 16;
            if (job < HD * 2) {
                vlo[j] = *(const uint2 *)(page + VLO_OFF + dim * (PAGE / 2) + s16 / 2);
                vhi[j] = *(const uint16_t *)(page + VHI_OFF + dim * (PAGE / 8) + s16 / 8);
                const int4 *sp = (const int4 *)((const half *)(page + VS_OFF) + (dim / 32) * PAGE + s16);
                vs[j][0] = sp[0]; vs[j][1] = sp[1];
            }
        }
    }
    __device__ void store(_Float16 (*Ks)[QLD], _Float16 (*Vt)[VLD32], int nvalid, int tid) const {
#pragma unroll
        for (int j = 0; j < KJ; j++) {
            const int job = tid + j * NT, kk = job >> 3, blk = job & 7;
            if (job < KST * 8) {
                const _Float16 d = (_Float16)kd[j];  // exact (an fp16 scale): k8_frag gives (_Float16)k * d, rounded once
                *(h16 *)&Ks[kk][blk * 32] = k8_frag(k[j][0], d);
                *(h16 *)&Ks[kk][blk * 32 + 16] = k8_frag(k[j][1], d);
            }
        }
#pragma unroll
        for (int j = 0; j < VJ; j++) {
            const int job = tid + j * NT, dim = job >> 1, half16 = job & 1;
            if (job < HD * 2) {
                const uint2 lo = vlo[j];
                const uint32_t hb = vhi[j];
                const half *vsc = (const half *)vs[j];
                h16 v;
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const int q = (int)((((i < 8 ? lo.x : lo.y) >> (4 * (i % 8))) & 15) | (((hb >> i) & 1) << 4)) - 16;
                    v[i] = (_Float16)(half16 * 16 + i < nvalid ? __half2float(vsc[i]) * q : 0.0f);
                }
                *(h16 *)&Vt[dim][half16 * 16] = v;
            }
        }
    }
};

template <int AW>
__global__ __launch_bounds__(AW * 32) void k_attn(AttnArgs a) {
    __shared__ _Float16 Ks[KST][QLD];
    __shared__ _Float16 Vt[HD][VLD32];
    const int g = blockIdx.y, tid = threadIdx.x;
    const int wave = tid / 32, lane = tid % 32, r16 = lane % 16, hi = lane / 16;
    const int nrows = a.T * 6, tile = blockIdx.x * AW + wave;
    const int myrow = tile * 16 + r16;
    const bool valid = myrow < nrows;
    const int qpos = valid ? a.pos0 + myrow / 6 : -1;
    const half *qrow = a.Q16 + ((size_t)(valid ? myrow / 6 : 0) * NH + g * 6 + (valid ? myrow % 6 : 0)) * HD;
    const int last = min(nrows, (blockIdx.x * AW + AW) * 16) - 1;
    const int kbeg = blockIdx.z * a.split_len, kend = min(a.pos0 + last / 6 + 1, kbeg + a.split_len);

    float m = -INFINITY, l = 0;
    f8 acc[16];
#pragma unroll
    for (int d = 0; d < 16; d++) acc[d] = f8{};

    AttnRaw<AW> raw;
    if (kbeg < kend) raw.load(kv_head(a.kv, a.layer, g, kbeg), kbeg % PAGE, kend - kbeg, tid);   // 32 keys never straddle a page
    for (int k0 = kbeg; k0 < kend; k0 += KST) {
        __syncthreads();
        raw.store(Ks, Vt, kend - k0, tid);
        __syncthreads();

        f8 s0 = {}, s1 = {};                                      // S^T[key 2i+hi (+16)][row r16]
#pragma unroll
        for (int d = 0; d < 16; d++) {
            const h16 q = *(const h16 *)(qrow + d * 16);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const h16 *)&Ks[r16][d * 16], q, s0);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const h16 *)&Ks[16 + r16][d * 16], q, s1);
        }
        if (k0 + KST < kend) raw.load(kv_head(a.kv, a.layer, g, k0 + KST), (k0 + KST) % PAGE, kend - k0 - KST, tid);
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
    const size_t row = (size_t)(myrow / 6) * NH + g * 6 + myrow % 6;
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

// Decode / verify attention (T <= 8, 6T query rows in NT tiles of 16): one 256-thread workgroup (8 waves) per (key split,
// kv head), keys in stages of 128 (two pages). Per stage, wave w computes S^T = K Q^T for keys 16w..16w+15 of the stage
// (its K rows read straight from memory, int8 -> fp16 with the block scale) for all query rows; the waves agree on each
// row's running max through LDS and publish P^T (fp16); then wave w accumulates O^T = V^T P^T for its own 32 dims (one V
// scale block, folded into P) over all 128 keys. Every K/V byte is read once for all query rows, and a wave keeps 32
// dims of output per tile (16 registers) instead of 256. Writes unnormalized partials that k_attn_combine merges.
constexpr int DST = 128;                      // keys per decode stage
constexpr int DEC_PAGES = 64;                 // page table entries a decode workgroup caches (split_len <= 4096 keys)
constexpr int DEC_SPLITS = 128;               // key splits per kv head at long contexts (fewer: longer workgroups, fewer partials)
template <int NT>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(NT == 3 ? 4 : 8))) void k_attn_dec(AttnArgs a) {
    constexpr int R = 16 * NT;
    __shared__ _Float16 Qs[R][HD + 8];
    __shared__ float rmx[8][R];
    __shared__ _Float16 Ps[8][R][16 + 8];     // P^T per key tile: [tile][row][16 fragment slots, keys VPERM]
    __shared__ int pg[DEC_PAGES];             // the split's pages
    const int split = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int w = tid / 32, lane = tid % 32, r16 = lane % 16, hi = lane / 16;
    const int nrows = a.T * 6;
    const int kbeg = split * a.split_len, kend = min(a.pos0 + a.T, kbeg + a.split_len);
    for (int i = tid; i < R * (HD / 8); i += 256) {      // the group's query rows (row t * 6 + j: token t, head 6g + j)
        const int r = i / (HD / 8), c = i % (HD / 8);
        int4 v = {0, 0, 0, 0};
        if (r < nrows) v = *(const int4 *)(a.Q16 + ((size_t)(r / 6) * NH + g * 6 + r % 6) * HD + c * 8);
        *(int4 *)&Qs[r][c * 8] = v;
    }
    if (tid < (kend - kbeg + PAGE - 1) / PAGE) pg[tid] = a.kv.pages[kbeg / PAGE + tid];
    int qpos[NT];
    float m[NT], l[NT];
    f8 acc[NT][2];
#pragma unroll
    for (int rt = 0; rt < NT; rt++) {
        const int row = rt * 16 + r16;
        qpos[rt] = row < nrows ? a.pos0 + row / 6 : -1;
        m[rt] = -INFINITY; l[rt] = 0;
        acc[rt][0] = f8{}; acc[rt][1] = f8{};
    }
    __syncthreads();
    const size_t head_off = (size_t)a.layer * KV_LAYER + (size_t)g * KV_HEAD;

    for (int k0 = kbeg; k0 < kend; k0 += DST) {
        const int pi = (k0 - kbeg) / PAGE;
        const uint8_t *pp[2];                                // the stage's two pages (wave-uniform)
        pp[0] = a.kv.pool + (size_t)__builtin_amdgcn_readfirstlane(pg[pi]) * KV_PAGE + head_off;
        pp[1] = k0 + PAGE < kend ? a.kv.pool + (size_t)__builtin_amdgcn_readfirstlane(pg[pi + 1]) * KV_PAGE + head_off : pp[0];
        // all of the stage's loads first: this wave's V dims over the 8 key tiles, then its K tile
        uint2 vlo[8][2];
        uint32_t vhi[8][2];
#pragma unroll
        for (int kt = 0; kt < 8; kt++) {
            const uint8_t *page = pp[kt / 4];
            const int slot0 = (kt % 4) * 16;
#pragma unroll
            for (int dt = 0; dt < 2; dt++) {
                const int dim = 32 * w + 16 * dt + r16;
                vlo[kt][dt] = k0 + 16 * kt < kend ? *(const uint2 *)(page + VLO_OFF + dim * (PAGE / 2) + slot0 / 2) : uint2{0, 0};
                vhi[kt][dt] = k0 + 16 * kt < kend ? *(const uint16_t *)(page + VHI_OFF + dim * (PAGE / 8) + slot0 / 8) : 0u;
            }
        }
        const int kt0 = k0 + 16 * w;                         // this wave's key tile
        f8 sc[NT];
#pragma unroll
        for (int rt = 0; rt < NT; rt++) sc[rt] = f8{};
        if (kt0 < kend) {
            const uint8_t *page = pp[w / 4];
            const int slot = (w % 4) * 16 + r16;
            const int8_t *kq = (const int8_t *)page + slot * HD;
            const half *ks = (const half *)(page + KS_OFF) + slot * (HD / 32);
            int4 kr[16];
#pragma unroll
            for (int d = 0; d < 16; d++) kr[d] = *(const int4 *)(kq + d * 16);
#pragma unroll
            for (int d = 0; d < 16; d++) {
                const h16 A = k8_frag(kr[d], (_Float16)__half2float(ks[d / 2]));
#pragma unroll
                for (int rt = 0; rt < NT; rt++)
                    sc[rt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, *(const h16 *)&Qs[rt * 16 + r16][d * 16], sc[rt]);
            }
        }
#pragma unroll
        for (int rt = 0; rt < NT; rt++) {                    // mask, this wave's row maxima
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int key = kt0 + 2 * i + hi;
                if (key >= kend || key > qpos[rt]) sc[rt][i] = -INFINITY;
                mx = fmaxf(mx, sc[rt][i]);
            }
            mx = fmaxf(mx, lane_xor<16>(mx));
            if (hi == 0) rmx[w][rt * 16 + r16] = mx;
        }
        __syncthreads();
#pragma unroll
        for (int rt = 0; rt < NT; rt++) {                    // the stage's row max, P^T of this wave's keys to LDS
            float M = m[rt];
#pragma unroll
            for (int ww = 0; ww < 8; ww++) M = fmaxf(M, rmx[ww][rt * 16 + r16]);
            const float alpha = M == -INFINITY ? 1.0f : __expf(m[rt] - M);
            m[rt] = M;
            float p[8], rs = 0;
#pragma unroll
            for (int i = 0; i < 8; i++) { p[i] = M == -INFINITY ? 0.0f : __expf(sc[rt][i] - M); rs += p[i]; }
            l[rt] = l[rt] * alpha + rs + lane_xor<16>(rs);
            acc[rt][0] *= alpha; acc[rt][1] *= alpha;
            float P16[16];                                   // row r16's keys 0..15 (the partner lane has the other parity)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float o = lane_xor<16>(p[i]);
                P16[2 * i] = hi ? o : p[i];
                P16[2 * i + 1] = hi ? p[i] : o;
            }
            h16 Ph;                                          // lane hi writes fragment slots 8hi .. 8hi+7 (keys VPERM)
#pragma unroll
            for (int j = 0; j < 8; j++) Ph[j] = (_Float16)(hi ? P16[VPERM[8 + j]] : P16[VPERM[j]]);
            *(uint4 *)&Ps[w][rt * 16 + r16][8 * hi] = *(const uint4 *)&Ph;
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 8; kt++) {                     // O^T for dims 32w + 16dt + r16 over the stage's keys
            if (k0 + 16 * kt >= kend) break;
            const half *vs = (const half *)(pp[kt / 4] + VS_OFF) + w * PAGE + (kt % 4) * 16;
            h16 vsp;                                         // the keys' V scales in fragment order
#pragma unroll
            for (int i = 0; i < 16; i++) vsp[i] = (_Float16)__half2float(vs[VPERM[i]]);
            h16 B[NT];
#pragma unroll
            for (int rt = 0; rt < NT; rt++) B[rt] = *(const h16 *)&Ps[kt][rt * 16 + r16][0] * vsp;
#pragma unroll
            for (int dt = 0; dt < 2; dt++) {
                const h16 A = v5_frag_raw(vlo[kt][dt], vhi[kt][dt]);
#pragma unroll
                for (int rt = 0; rt < NT; rt++) acc[rt][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B[rt], acc[rt][dt]);
            }
        }
    }
    // partials: O^T rows of this wave's dims; the row sums add up over the waves
#pragma unroll
    for (int rt = 0; rt < NT; rt++) if (hi == 0) rmx[w][rt * 16 + r16] = l[rt];
    __syncthreads();
#pragma unroll
    for (int rt = 0; rt < NT; rt++) {
        const int myrow = rt * 16 + r16;
        if (myrow >= nrows) continue;
        const size_t row = (size_t)(myrow / 6) * NH + g * 6 + myrow % 6;
        float *po = a.part_o + ((size_t)split * a.T * NH + row) * HD + 32 * w;
#pragma unroll
        for (int dt = 0; dt < 2; dt++)
#pragma unroll
            for (int i = 0; i < 8; i++) po[dt * 16 + 2 * i + hi] = acc[rt][dt][i];
        if (w == 0 && hi == 0) {
            float L = 0;
#pragma unroll
            for (int ww = 0; ww < 8; ww++) L += rmx[ww][myrow];
            a.part_ml[((size_t)split * a.T * NH + row) * 2] = m[rt];
            a.part_ml[((size_t)split * a.T * NH + row) * 2 + 1] = L;
        }
    }
}

// Merge key splits, apply the output gate, write fp16 (prefill) and/or int8 blocks (decode). One workgroup per row:
// thread s weighs split s (running max and sum across the splits), then thread d sums dimension d over the splits.
__global__ __launch_bounds__(256) void k_attn_combine(const float *part_o, const float *part_ml, int nsplit, int rows,
                                                      const float *qg, half *oh, QAct oq) {
    __shared__ float buf[HD], ws[MAX_SPLITS], red[8], part[8][HD];
    const int row = blockIdx.x, d = threadIdx.x;
    const float ms = d < nsplit ? part_ml[((size_t)d * rows + row) * 2] : -INFINITY;
    float M = warp_max(ms);
    if ((d & 31) == 0) red[d >> 5] = M;
    __syncthreads();
    M = red[0];
    for (int i = 1; i < 8; i++) M = fmaxf(M, red[i]);
    __syncthreads();
    const float wt = ms == -INFINITY ? 0.0f : __expf(ms - M);
    if (d < nsplit) ws[d] = wt;
    const float L = block_sum256(d < nsplit ? wt * part_ml[((size_t)d * rows + row) * 2 + 1] : 0.0f, red);
    {                                                   // wave v sums splits v, v + 8, ... for all dims (8 per lane)
        const int v = d / 32, ln = d % 32;
        float4 a0 = {0, 0, 0, 0}, a1 = {0, 0, 0, 0};
#pragma unroll 4
        for (int s = v; s < nsplit; s += 8) {
            const float w = ws[s];
            const float4 *src = (const float4 *)(part_o + ((size_t)s * rows + row) * HD) + 2 * ln;
            const float4 x0 = src[0], x1 = src[1];
            a0.x += w * x0.x; a0.y += w * x0.y; a0.z += w * x0.z; a0.w += w * x0.w;
            a1.x += w * x1.x; a1.y += w * x1.y; a1.z += w * x1.z; a1.w += w * x1.w;
        }
        *(float4 *)&part[v][8 * ln] = a0;
        *(float4 *)&part[v][8 * ln + 4] = a1;
    }
    __syncthreads();
    float o = 0;
#pragma unroll
    for (int v = 0; v < 8; v++) o += part[v][d];
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
    const int ntiles = (a.T * 6 + 15) / 16;
    const int nkeys = a.pos0 + a.T;
    a.out_h = oh;
    if (a.T <= 8) {                       // decode / verify: up to DEC_SPLITS workgroups per kv head, whole stages each
        const int nst = (nkeys + DST - 1) / DST;
        const int target = std::min(DEC_SPLITS, nst);
        if ((nst + target - 1) / target * DST > DEC_PAGES * PAGE) FAIL("attention: %d keys exceed the decode page cache", nkeys);
        a.split_len = (nst + target - 1) / target * DST;
        a.nsplit = (nkeys + a.split_len - 1) / a.split_len;
        a.partial = 1;
        const dim3 grid(a.nsplit, NKV);
        if (ntiles == 1) k_attn_dec<1><<<grid, 256, 0, st>>>(a);
        else if (ntiles == 2) k_attn_dec<2><<<grid, 256, 0, st>>>(a);
        else k_attn_dec<3><<<grid, 256, 0, st>>>(a);
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
