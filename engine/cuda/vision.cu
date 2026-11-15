#include "vision.h"
#include "../os.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define VCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "chrysalis vision: CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)
#define VFAIL(...) do { fprintf(stderr, "chrysalis vision: " __VA_ARGS__); fprintf(stderr, "\n"); abort(); } while (0)

namespace {

constexpr int C = Vision::C, HEADS = Vision::HEADS, HDIM = Vision::HDIM, FFN = Vision::FFN, PIN = Vision::PATCH_IN;
constexpr int GRID = Vision::GRID, MERGED = Vision::MERGED;
constexpr int ROWS = 2048;
constexpr size_t WS = 32 << 20;
constexpr float DOWN = 1.0f / 16, UP = 16;
constexpr size_t al(size_t n) { return (n + 255) & ~size_t(255); }
constexpr int pad64(int n) { return (n + 63) / 64 * 64; }

struct File {
    struct T { int type = -1; int64_t ne[4] = {1, 1, 1, 1}; const uint8_t *data = nullptr; };
    std::map<std::string, T> t;
    std::map<std::string, int64_t> ints;
    uint8_t *base = nullptr;
    size_t size = 0;
    bool open(const char *path) {
        const os::File fd = os::open_read(path);
        if (fd == os::NO_FILE) return false;
        size = os::size(fd);
        base = (uint8_t *)os::map(fd, size);
        os::close(fd);
        if (!base || memcmp(base, "GGUF", 4) != 0) return false;
        const uint8_t *p = base + 4;
        auto rd = [&](auto &v) { memcpy(&v, p, sizeof(v)); p += sizeof(v); };
        auto rs = [&]() { uint64_t n; rd(n); std::string s((const char *)p, n); p += n; return s; };
        uint32_t version; uint64_t nt, nkv;
        rd(version); rd(nt); rd(nkv);
        static const int scalar[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < nkv; i++) {
            const std::string key = rs();
            uint32_t type; rd(type);
            if (type == 8) { rs(); continue; }
            if (type == 9) {
                uint32_t at; uint64_t n; rd(at); rd(n);
                if (at == 8) for (uint64_t j = 0; j < n; j++) rs();
                else p += n * scalar[at];
                continue;
            }
            int64_t v = 0;
            if (type == 4 || type == 5) { int32_t x; rd(x); v = x; }
            else if (type == 10 || type == 11) rd(v);
            else p += scalar[type];
            ints[key] = v;
            if (key == "general.alignment") alignment = v;
        }
        std::map<std::string, uint64_t> off;
        for (uint64_t i = 0; i < nt; i++) {
            const std::string name = rs();
            uint32_t nd; rd(nd);
            T x;
            for (uint32_t d = 0; d < nd; d++) rd(x.ne[d]);
            uint32_t type; rd(type); x.type = type;
            uint64_t o; rd(o); off[name] = o;
            t[name] = x;
        }
        const size_t start = ((p - base) + alignment - 1) / alignment * alignment;
        for (auto &[name, x] : t) x.data = base + start + off[name];
        return true;
    }
    const T &get(const std::string &name, int type, std::initializer_list<int64_t> ne) const {
        auto it = t.find(name);
        if (it == t.end()) VFAIL("the vision GGUF has no %s", name.c_str());
        int k = 0;
        for (int64_t n : ne) if (it->second.ne[k++] != n) VFAIL("%s has an unexpected shape", name.c_str());
        if (it->second.type != type) VFAIL("%s has type %d, expected %d", name.c_str(), it->second.type, type);
        return it->second;
    }
    ~File() { os::unmap(base, size); }
};
constexpr int GF32 = 0, GBF16 = 30;

struct EmbedOff { size_t w, w2, b, pos, end; };
struct BlockOff { size_t qkv_w, proj_w, fc1_w, fc2_w, qkv_b, proj_b, fc1_b, fc2_b, ln1_w, ln1_b, ln2_w, ln2_b, end; };
struct MergeOff { size_t norm_w, norm_b, fc1_w, fc1_b, end; };
struct OutOff { size_t fc2_w, fc2_b, end; };
constexpr EmbedOff embed_off() {
    EmbedOff o{};
    o.w = 0; o.w2 = al(o.w + (size_t)C * PIN * 2); o.b = al(o.w2 + (size_t)C * 2 * PIN * 2); o.pos = al(o.b + C * 4);
    o.end = al(o.pos + (size_t)GRID * GRID * C * 4);
    return o;
}
constexpr BlockOff block_off() {
    BlockOff o{};
    o.qkv_w = 0; o.proj_w = al(o.qkv_w + (size_t)3 * C * C * 2); o.fc1_w = al(o.proj_w + (size_t)C * C * 2);
    o.fc2_w = al(o.fc1_w + (size_t)FFN * C * 2); o.qkv_b = al(o.fc2_w + (size_t)C * FFN * 2);
    o.proj_b = al(o.qkv_b + 3 * C * 4); o.fc1_b = al(o.proj_b + C * 4); o.fc2_b = al(o.fc1_b + FFN * 4);
    o.ln1_w = al(o.fc2_b + C * 4); o.ln1_b = al(o.ln1_w + C * 4); o.ln2_w = al(o.ln1_b + C * 4); o.ln2_b = al(o.ln2_w + C * 4);
    o.end = al(o.ln2_b + C * 4);
    return o;
}
constexpr MergeOff merge_off() {
    MergeOff o{};
    o.norm_w = 0; o.norm_b = al(C * 4); o.fc1_w = al(o.norm_b + C * 4); o.fc1_b = al(o.fc1_w + (size_t)MERGED * MERGED * 2);
    o.end = al(o.fc1_b + MERGED * 4);
    return o;
}
OutOff out_off(int out_dim) {
    OutOff o{};
    o.fc2_w = 0; o.fc2_b = al((size_t)out_dim * MERGED * 2); o.end = al(o.fc2_b + (size_t)out_dim * 4);
    return o;
}

__device__ __forceinline__ float wsum(float v) {
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ int2 patch_rc(int p, int gw) {
    const int blk = p >> 2, in = p & 3, bw = gw >> 1;
    return {(blk / bw) * 2 + (in >> 1), (blk % bw) * 2 + (in & 1)};
}

__device__ __forceinline__ void vmma(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t vpack(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *)&h;
}

__global__ void k_to_f16(const float *x, half *y, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ __launch_bounds__(256) void k_embed(const half *y, const float *bias, const float *pos, float *x, int r0, int gh, int gw) {
    const int t = blockIdx.x, p = r0 + t;
    const int2 rc = patch_rc(p, gw);
    const float sh = (float)(rc.x * (GRID - 1)) / (float)max(gh - 1, 1), sw = (float)(rc.y * (GRID - 1)) / (float)max(gw - 1, 1);
    const int h0 = (int)floorf(sh), w0 = (int)floorf(sw), h1 = min(h0 + 1, GRID - 1), w1 = min(w0 + 1, GRID - 1);
    const float fh = sh - h0, fw = sw - w0;
    const float k00 = (1 - fh) * (1 - fw), k01 = (1 - fh) * fw, k10 = fh * (1 - fw), k11 = fh * fw;
    const float *p00 = pos + (size_t)(h0 * GRID + w0) * C, *p01 = pos + (size_t)(h0 * GRID + w1) * C;
    const float *p10 = pos + (size_t)(h1 * GRID + w0) * C, *p11 = pos + (size_t)(h1 * GRID + w1) * C;
    for (int c = threadIdx.x; c < C; c += 256)
        x[(size_t)p * C + c] = __half2float(y[(size_t)t * C + c]) * UP + bias[c] + (k00 * p00[c] + k01 * p01[c] + k10 * p10[c] + k11 * p11[c]);
}

__global__ __launch_bounds__(256) void k_ln(const float *x, const float *w, const float *b, half *out, int rows, float eps) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float *xr = x + (size_t)row * C;
    float v[C / 32], s = 0;
#pragma unroll
    for (int i = 0; i < C / 32; i++) { v[i] = xr[lane + 32 * i]; s += v[i]; }
    const float mean = wsum(s) / C;
    float q = 0;
#pragma unroll
    for (int i = 0; i < C / 32; i++) { v[i] -= mean; q += v[i] * v[i]; }
    const float inv = rsqrtf(wsum(q) / C + eps);
#pragma unroll
    for (int i = 0; i < C / 32; i++) {
        const int c = lane + 32 * i;
        out[(size_t)row * C + c] = __float2half(v[i] * inv * w[c] + b[c]);
    }
}

__global__ __launch_bounds__(256) void k_qkv(half *qkv, const float *bias, int r0, int gw) {
    const int p = r0 + blockIdx.x;
    const int2 rc = patch_rc(p, gw);
    half *row = qkv + (size_t)p * 3 * C;
    for (int i = threadIdx.x; i < 2 * HEADS * 36; i += 256) {
        const int part = i / (HEADS * 36), hj = i % (HEADS * 36), h = hj / 36, j = hj % 36;
        const int d0 = part * C + h * HDIM + j, d1 = d0 + 36;
        const int f = j % 18;
        const float inv = 1.0f / powf(10000.0f, (float)(2 * f) / 36.0f);
        float s, c;
        sincosf((float)(j < 18 ? rc.x : rc.y) * inv, &s, &c);
        const float a = __half2float(row[d0]) + bias[d0], b = __half2float(row[d1]) + bias[d1];
        const float scale = part == 0 ? 0.11785113019775792f : 1.0f;
        row[d0] = __float2half((a * c - b * s) * scale);
        row[d1] = __float2half((b * c + a * s) * scale);
    }
    for (int i = threadIdx.x; i < C; i += 256) row[2 * C + i] = __float2half(__half2float(row[2 * C + i]) + bias[2 * C + i]);
}

__global__ void k_add(float *x, const half *y, const float *bias, int r0, int T) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= (int64_t)T * C) return;
    x[(size_t)r0 * C + i] += __half2float(y[i]) * UP + bias[i % C];
}

__global__ void k_gelu(half *t, const float *bias, int64_t n, int cols, int erf_) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(t[i]) + bias[i % cols];
    const float g = erf_ ? 0.5f * v * (1.0f + erff(v * 0.70710678118654752f))
                         : 0.5f * v * (1.0f + tanhf(0.79788456080286536f * (v + 0.044715f * v * v * v)));
    t[i] = __float2half(g);
}

__global__ void k_out(const half *y, const float *bias, half *out, int64_t n, int cols, int64_t base) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[base + i] = __float2half(__half2float(y[i]) + bias[i % cols]);
}

constexpr int VD = 80, KST = 32, KLD = VD + 8, VLD = KST + 8;
template <int AW>
__global__ __launch_bounds__(AW * 32) void k_attn(half *qkv, int n) {
    __shared__ __align__(16) half Ks[KST][KLD];
    __shared__ __align__(16) half Vt[VD][VLD];
    const int head = blockIdx.y, tid = threadIdx.x, warp = tid / 32, lane = tid % 32, gq = lane / 4, tig = lane % 4;
    const int r0 = (blockIdx.x * AW + warp) * 16 + gq, r1 = r0 + 8;
    const size_t ld = 3 * C;
    for (int i = tid; i < KST * (VD - HDIM); i += AW * 32) Ks[i / (VD - HDIM)][HDIM + i % (VD - HDIM)] = __float2half(0.0f);
    for (int i = tid; i < (VD - HDIM) * KST; i += AW * 32) Vt[HDIM + i / KST][i % KST] = __float2half(0.0f);
    const half *q0 = qkv + (size_t)min(r0, n - 1) * ld + head * HDIM, *q1 = qkv + (size_t)min(r1, n - 1) * ld + head * HDIM;
    uint32_t Q[VD / 16][4];
#pragma unroll
    for (int kk = 0; kk < VD / 16; kk++) {
        const int d = kk * 16 + tig * 2;
        Q[kk][0] = *(const uint32_t *)(q0 + d);
        Q[kk][1] = *(const uint32_t *)(q1 + d);
        Q[kk][2] = d + 8 < HDIM ? *(const uint32_t *)(q0 + d + 8) : 0u;
        Q[kk][3] = d + 8 < HDIM ? *(const uint32_t *)(q1 + d + 8) : 0u;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0, l1 = 0;
    float acc[VD / 8][4];
#pragma unroll
    for (int d = 0; d < VD / 8; d++) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.0f;
    for (int k0 = 0; k0 < n; k0 += KST) {
        __syncthreads();
        for (int j = tid; j < KST * 9; j += AW * 32) {
            const int key = j / 9, piece = j % 9, gk = k0 + key;
            uint4 kv = {0, 0, 0, 0}, vv = {0, 0, 0, 0};
            if (gk < n) {
                const half *src = qkv + (size_t)gk * ld + head * HDIM + piece * 8;
                kv = *(const uint4 *)(src + C);
                vv = *(const uint4 *)(src + 2 * C);
            }
            *(uint4 *)&Ks[key][piece * 8] = kv;
            const half *ve = (const half *)&vv;
#pragma unroll
            for (int e = 0; e < 8; e++) Vt[piece * 8 + e][key] = ve[e];
        }
        __syncthreads();
        float s[KST / 8][4];
#pragma unroll
        for (int t = 0; t < KST / 8; t++) s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < VD / 16; kk++) {
            const int d = kk * 16 + tig * 2;
#pragma unroll
            for (int t = 0; t < KST / 8; t++) {
                const uint32_t B[2] = {*(const uint32_t *)&Ks[t * 8 + gq][d], *(const uint32_t *)&Ks[t * 8 + gq][d + 8]};
                vmma(s[t], Q[kk], B);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int t = 0; t < KST / 8; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (k0 + t * 8 + tig * 2 + (i & 1) >= n) s[t][i] = -INFINITY;
                if (i < 2) mx0 = fmaxf(mx0, s[t][i]); else mx1 = fmaxf(mx1, s[t][i]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        float al0 = 1.0f, al1 = 1.0f;
        if (mx0 > m0) { al0 = m0 == -INFINITY ? 0.0f : __expf(m0 - mx0); m0 = mx0; }
        if (mx1 > m1) { al1 = m1 == -INFINITY ? 0.0f : __expf(m1 - mx1); m1 = mx1; }
        float rs0 = 0, rs1 = 0;
#pragma unroll
        for (int t = 0; t < KST / 8; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s[t][i] = __expf(s[t][i] - (i < 2 ? m0 : m1));
                if (i < 2) rs0 += s[t][i]; else rs1 += s[t][i];
            }
        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1); rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1); rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
        l0 = l0 * al0 + rs0; l1 = l1 * al1 + rs1;
#pragma unroll
        for (int d = 0; d < VD / 8; d++) { acc[d][0] *= al0; acc[d][1] *= al0; acc[d][2] *= al1; acc[d][3] *= al1; }
        uint32_t P[KST / 16][4];
#pragma unroll
        for (int ks = 0; ks < KST / 16; ks++) {
            P[ks][0] = vpack(s[2 * ks][0], s[2 * ks][1]);
            P[ks][1] = vpack(s[2 * ks][2], s[2 * ks][3]);
            P[ks][2] = vpack(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            P[ks][3] = vpack(s[2 * ks + 1][2], s[2 * ks + 1][3]);
        }
#pragma unroll
        for (int dt = 0; dt < VD / 8; dt++)
#pragma unroll
            for (int ks = 0; ks < KST / 16; ks++) {
                const uint32_t B[2] = {*(const uint32_t *)&Vt[dt * 8 + gq][ks * 16 + tig * 2], *(const uint32_t *)&Vt[dt * 8 + gq][ks * 16 + 8 + tig * 2]};
                vmma(acc[dt], P[ks], B);
            }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int r = h ? r1 : r0;
        if (r >= n) continue;
        const float inv = 1.0f / (h ? l1 : l0);
        half *op = qkv + (size_t)r * ld + head * HDIM;
#pragma unroll
        for (int dt = 0; dt < VD / 8; dt++) {
            const int dim = dt * 8 + tig * 2;
            if (dim < HDIM) *(uint32_t *)(op + dim) = vpack(acc[dt][2 * h] * inv, acc[dt][2 * h + 1] * inv);
        }
    }
}

uint16_t f16_bits(float f) { const half h = __float2half(f); uint16_t u; memcpy(&u, &h, 2); return u; }
float bf16_value(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

}

void Vision::open(const std::string &p, int out) {
    path = p;
    out_dim = out;
    File f;
    if (!f.open(path.c_str())) VFAIL("cannot open %s", path.c_str());
    if (f.ints["clip.vision.block_count"] != LAYERS || f.ints["clip.vision.embedding_length"] != C ||
        f.ints["clip.vision.projection_dim"] != out_dim || f.ints["clip.vision.patch_size"] != 16)
        VFAIL("%s is not Qwen3.8's vision tower for a %d-wide model", path.c_str(), out_dim);
    const size_t eb = embed_off().end, bb = block_off().end, mb = merge_off().end, ob = out_off(out_dim).end;
    units.clear();
    size_t off = 0;
    units.push_back({off, eb}); off += eb;
    for (int l = 0; l < LAYERS; l++) { units.push_back({off, bb}); off += bb; }
    units.push_back({off, mb}); off += mb;
    units.push_back({off, ob}); off += ob;
    slot_bytes = std::max({eb, bb, mb, ob});
}

void Vision::load() {
    File f;
    if (!f.open(path.c_str())) VFAIL("cannot open %s", path.c_str());
    const size_t total = units.back().off + units.back().bytes;
    VCK(cudaMallocHost(&host, total));
    struct Job { const File::T *t; size_t dst; bool to_f16; int64_t n; const File::T *add = nullptr; bool pair = false; };
    std::vector<Job> jobs;
    auto put = [&](const char *name, int type, std::initializer_list<int64_t> ne, size_t dst) {
        const File::T &t = f.get(name, type, ne);
        int64_t n = 1;
        for (int64_t x : ne) n *= x;
        jobs.push_back({&t, dst, type == GBF16, n});
    };
    {
        const EmbedOff o = embed_off();
        const size_t u = units[0].off;
        const File::T &w0 = f.get("v.patch_embd.weight", GF32, {16, 16, 3, C}), &w1 = f.get("v.patch_embd.weight.1", GF32, {16, 16, 3, C});
        jobs.push_back({&w0, u + o.w, true, (int64_t)C * PIN, &w1});
        jobs.push_back({&w0, u + o.w2, true, (int64_t)C * PIN, &w1, true});
        put("v.patch_embd.bias", GF32, {C}, u + o.b);
        put("v.position_embd.weight", GF32, {C, GRID * GRID}, u + o.pos);
    }
    for (int l = 0; l < LAYERS; l++) {
        const BlockOff o = block_off();
        const size_t u = units[1 + l].off;
        const std::string p = "v.blk." + std::to_string(l) + ".";
        put((p + "attn_qkv.weight").c_str(), GBF16, {C, 3 * C}, u + o.qkv_w);
        put((p + "attn_out.weight").c_str(), GBF16, {C, C}, u + o.proj_w);
        put((p + "ffn_up.weight").c_str(), GBF16, {C, FFN}, u + o.fc1_w);
        put((p + "ffn_down.weight").c_str(), GBF16, {FFN, C}, u + o.fc2_w);
        put((p + "attn_qkv.bias").c_str(), GF32, {3 * C}, u + o.qkv_b);
        put((p + "attn_out.bias").c_str(), GF32, {C}, u + o.proj_b);
        put((p + "ffn_up.bias").c_str(), GF32, {FFN}, u + o.fc1_b);
        put((p + "ffn_down.bias").c_str(), GF32, {C}, u + o.fc2_b);
        put((p + "ln1.weight").c_str(), GF32, {C}, u + o.ln1_w);
        put((p + "ln1.bias").c_str(), GF32, {C}, u + o.ln1_b);
        put((p + "ln2.weight").c_str(), GF32, {C}, u + o.ln2_w);
        put((p + "ln2.bias").c_str(), GF32, {C}, u + o.ln2_b);
    }
    {
        const MergeOff o = merge_off();
        const size_t u = units[1 + LAYERS].off;
        put("v.post_ln.weight", GF32, {C}, u + o.norm_w);
        put("v.post_ln.bias", GF32, {C}, u + o.norm_b);
        put("mm.0.weight", GBF16, {MERGED, MERGED}, u + o.fc1_w);
        put("mm.0.bias", GF32, {MERGED}, u + o.fc1_b);
        const OutOff q = out_off(out_dim);
        const size_t v = units[2 + LAYERS].off;
        put("mm.2.weight", GBF16, {MERGED, out_dim}, v + q.fc2_w);
        put("mm.2.bias", GF32, {out_dim}, v + q.fc2_b);
    }
    std::vector<std::thread> th;
    const int nt = 8;
    for (int w = 0; w < nt; w++) th.emplace_back([&, w] {
        for (size_t j = w; j < jobs.size(); j += nt) {
            const Job &jb = jobs[j];
            uint8_t *dst = host + jb.dst;
            if (!jb.to_f16) { memcpy(dst, jb.t->data, jb.n * 4); continue; }
            uint16_t *o = (uint16_t *)dst;
            if (jb.t->type == GBF16) {
                const uint16_t *s = (const uint16_t *)jb.t->data;
                for (int64_t i = 0; i < jb.n; i++) o[i] = f16_bits(bf16_value(s[i]));
            } else if (jb.pair) {
                const float *a = (const float *)jb.t->data, *b = (const float *)jb.add->data;
                for (int64_t i = 0; i < jb.n; i++) {
                    const int64_t c3 = i / 256, px = i % 256;
                    o[(c3 * 2) * 256 + px] = f16_bits(a[i]);
                    o[(c3 * 2 + 1) * 256 + px] = f16_bits(b[i]);
                }
            } else {
                const float *a = (const float *)jb.t->data, *b = (const float *)jb.add->data;
                for (int64_t i = 0; i < jb.n; i++) o[i] = f16_bits(a[i] + b[i]);
            }
        }
    });
    for (std::thread &t : th) t.join();
    VCK(cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking));
    for (int s = 0; s < 2; s++) {
        VCK(cudaEventCreateWithFlags(&copied[s], cudaEventDisableTiming));
        VCK(cudaEventCreateWithFlags(&used[s], cudaEventDisableTiming));
    }
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) VFAIL("cublasLtCreate failed");
}

void Vision::release() {
    if (!host) return;
    VCK(cudaFreeHost(host));
    host = nullptr;
}

size_t Vision::scratch_need(int n) const {
    const size_t np = pad64(n);
    return WS + 2 * al(slot_bytes) + al(np * C * 4) + al(np * 3 * C * 2) + al((size_t)ROWS * 2 * PIN * 2) + al((size_t)ROWS * FFN * 2)
         + al((size_t)std::max(ROWS * C, ROWS / 4 * out_dim) * 2) + al((size_t)ROWS * 2 * PIN * 4);
}

void Vision::gemm(const half *W, int N, int K, const half *x, int ldx, int T, half *y, int ldy, float alpha, void *ws, cudaStream_t st) {
    const auto key = std::make_tuple(N, K, T, ldx, ldy);
    auto it = plans.find(key);
    if (it == plans.end()) it = plans.emplace(key, lt_tune::plan(lt, N, K, T, ldx, ldy, false, W, x, y, ws, WS, st, false)).first;
    const lt_tune::Plan &p = it->second;
    const float beta = 0.0f;
    if (cublasLtMatmul(lt, p.desc, &alpha, W, p.a, x, p.b, &beta, y, p.c, y, p.c, &p.algo, ws, WS, st) != CUBLAS_STATUS_SUCCESS)
        VFAIL("cublasLtMatmul failed (%dx%dx%d)", N, K, T);
}

bool Vision::encode(const float *patches, int gh, int gw, half *out, uint8_t *scratch, size_t scratch_bytes, cudaStream_t st,
                    int frames) {
    const int n = gh * gw, merged = n / 4;
    if (gh % 2 || gw % 2 || n <= 0) VFAIL("a picture's grid must be whole 2 x 2 blocks (got %d x %d)", gh, gw);
    if (scratch_need(n) > scratch_bytes) return false;
    if (!host) load();
    const int np = pad64(n);
    uint8_t *s = scratch;
    auto take = [&](size_t bytes) { uint8_t *p = s; s += al(bytes); return p; };
    void *ws = take(WS);
    uint8_t *slot[2] = {take(slot_bytes), take(slot_bytes)};
    float *x = (float *)take((size_t)np * C * 4);
    half *qkv = (half *)take((size_t)np * 3 * C * 2);
    half *h16 = (half *)take((size_t)ROWS * 2 * PIN * 2);
    half *t16 = (half *)take((size_t)ROWS * FFN * 2);
    half *y16 = (half *)take((size_t)std::max(ROWS * C, ROWS / 4 * out_dim) * 2);
    float *p32 = (float *)take((size_t)ROWS * 2 * PIN * 4);

    const int nunits = (int)units.size();
    auto fetch = [&](int u) {
        if (u >= nunits) return;
        VCK(cudaStreamWaitEvent(cst, used[u % 2], 0));
        VCK(cudaMemcpyAsync(slot[u % 2], host + units[u].off, units[u].bytes, cudaMemcpyHostToDevice, cst));
        VCK(cudaEventRecord(copied[u % 2], cst));
    };
    VCK(cudaEventRecord(used[0], st));
    VCK(cudaEventRecord(used[1], st));
    fetch(0);
    fetch(1);
    auto chunks = [&](auto fn) { for (int r0 = 0; r0 < n; r0 += ROWS) fn(r0, std::min(ROWS, n - r0)); };
    auto blocks = [](int64_t total) { return (unsigned)((total + 255) / 256); };

    {
        VCK(cudaStreamWaitEvent(st, copied[0], 0));
        const EmbedOff o = embed_off();
        const int pin = PIN * frames;
        chunks([&](int r0, int T) {
            VCK(cudaMemcpyAsync(p32, patches + (size_t)r0 * pin, (size_t)T * pin * 4, cudaMemcpyHostToDevice, st));
            k_to_f16<<<blocks((int64_t)pad64(T) * pin), 256, 0, st>>>(p32, h16, (int64_t)T * pin);
            gemm((const half *)(slot[0] + (frames == 2 ? o.w2 : o.w)), C, pin, h16, pin, pad64(T), y16, C, DOWN, ws, st);
            k_embed<<<T, 256, 0, st>>>(y16, (const float *)(slot[0] + o.b), (const float *)(slot[0] + o.pos), x, r0, gh, gw);
        });
        VCK(cudaEventRecord(used[0], st));
        fetch(2);
    }
    const BlockOff o = block_off();
    for (int l = 0; l < LAYERS; l++) {
        const int u = 1 + l;
        uint8_t *w = slot[u % 2];
        auto F = [&](size_t off) { return (const float *)(w + off); };
        auto H = [&](size_t off) { return (const half *)(w + off); };
        VCK(cudaStreamWaitEvent(st, copied[u % 2], 0));
        chunks([&](int r0, int T) {
            k_ln<<<(T + 7) / 8, 256, 0, st>>>(x + (size_t)r0 * C, F(o.ln1_w), F(o.ln1_b), h16, T, 1e-6f);
            gemm(H(o.qkv_w), 3 * C, C, h16, C, pad64(T), qkv + (size_t)r0 * 3 * C, 3 * C, 1.0f, ws, st);
            k_qkv<<<T, 256, 0, st>>>(qkv, F(o.qkv_b), r0, gw);
        });
        k_attn<8><<<dim3((n + 127) / 128, HEADS), 256, 0, st>>>(qkv, n);
        chunks([&](int r0, int T) {
            gemm(H(o.proj_w), C, C, qkv + (size_t)r0 * 3 * C, 3 * C, pad64(T), y16, C, DOWN, ws, st);
            k_add<<<blocks((int64_t)T * C), 256, 0, st>>>(x, y16, F(o.proj_b), r0, T);
            k_ln<<<(T + 7) / 8, 256, 0, st>>>(x + (size_t)r0 * C, F(o.ln2_w), F(o.ln2_b), h16, T, 1e-6f);
            gemm(H(o.fc1_w), FFN, C, h16, C, pad64(T), t16, FFN, 1.0f, ws, st);
            k_gelu<<<blocks((int64_t)pad64(T) * FFN), 256, 0, st>>>(t16, F(o.fc1_b), (int64_t)pad64(T) * FFN, FFN, 0);
            gemm(H(o.fc2_w), C, FFN, t16, FFN, pad64(T), y16, C, DOWN, ws, st);
            k_add<<<blocks((int64_t)T * C), 256, 0, st>>>(x, y16, F(o.fc2_b), r0, T);
        });
        VCK(cudaEventRecord(used[u % 2], st));
        fetch(u + 2);
    }
    {
        const int u1 = 1 + LAYERS, u2 = 2 + LAYERS;
        const MergeOff m = merge_off();
        const OutOff q = out_off(out_dim);
        uint8_t *w1 = slot[u1 % 2], *w2 = slot[u2 % 2];
        VCK(cudaStreamWaitEvent(st, copied[u1 % 2], 0));
        VCK(cudaStreamWaitEvent(st, copied[u2 % 2], 0));
        for (int m0 = 0; m0 < merged; m0 += ROWS / 4) {
            const int T = std::min(ROWS / 4, merged - m0), Tp = pad64(T);
            k_ln<<<(4 * T + 7) / 8, 256, 0, st>>>(x + (size_t)4 * m0 * C, (const float *)(w1 + m.norm_w), (const float *)(w1 + m.norm_b), h16, 4 * T, 1e-6f);
            gemm((const half *)(w1 + m.fc1_w), MERGED, MERGED, h16, MERGED, Tp, t16, MERGED, 1.0f, ws, st);
            k_gelu<<<blocks((int64_t)Tp * MERGED), 256, 0, st>>>(t16, (const float *)(w1 + m.fc1_b), (int64_t)Tp * MERGED, MERGED, 1);
            gemm((const half *)(w2 + q.fc2_w), out_dim, MERGED, t16, MERGED, Tp, y16, out_dim, 1.0f, ws, st);
            k_out<<<blocks((int64_t)T * out_dim), 256, 0, st>>>(y16, (const float *)(w2 + q.fc2_b), out, (int64_t)T * out_dim, out_dim, (int64_t)m0 * out_dim);
        }
        VCK(cudaEventRecord(used[0], st));
        VCK(cudaEventRecord(used[1], st));
    }
    VCK(cudaStreamSynchronize(st));
    return true;
}
