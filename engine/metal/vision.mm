#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "vision.h"
#include "../os.h"
#include <dispatch/dispatch.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#define VFAIL(...) do { fprintf(stderr, "chrysalis vision: " __VA_ARGS__); fprintf(stderr, "\n"); abort(); } while (0)

namespace {

constexpr int C = Vision::C, FFN = Vision::FFN, PIN = Vision::PATCH_IN;
constexpr int GRID = Vision::GRID, MERGED = Vision::MERGED, LAYERS = Vision::LAYERS;
constexpr int ROWS = 2048;
constexpr size_t al(size_t n) { return (n + 255) & ~size_t(255); }
constexpr int GF32 = 0, GBF16 = 30;

struct EmbedOff { size_t w, w2, b, pos, end; };
struct BlockOff { size_t qkv_w, proj_w, fc1_w, fc2_w, qkv_b, proj_b, fc1_b, fc2_b, ln1_w, ln1_b, ln2_w, ln2_b, end; };
struct MergeOff { size_t norm_w, norm_b, fc1_w, fc1_b, end; };
struct OutOff { size_t fc2_w, fc2_b, end; };
constexpr EmbedOff embed_off() {
    EmbedOff o{};
    o.w = 0; o.w2 = al(o.w + (size_t)C * PIN * 4); o.b = al(o.w2 + (size_t)C * 2 * PIN * 4); o.pos = al(o.b + C * 4);
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
size_t unit_off(int u, int out_dim) {
    const size_t eb = embed_off().end, bb = block_off().end, mb = merge_off().end;
    if (u == 0) return 0;
    if (u <= LAYERS) return eb + (size_t)(u - 1) * bb;
    return eb + LAYERS * bb + (u >= LAYERS + 2 ? mb : 0) + (u >= LAYERS + 3 ? out_off(out_dim).end : 0);
}

uint16_t f16_bits(float f) { const __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
float bf16_value(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

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
        if (!base || size < 24 || memcmp(base, "GGUF", 4) != 0) return false;
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
        int64_t n = type == GBF16 ? 2 : 4;
        for (int64_t e : ne) { if (it->second.ne[k++] != e) VFAIL("%s has an unexpected shape", name.c_str()); n *= e; }
        if (it->second.type != type) VFAIL("%s has type %d, expected %d", name.c_str(), it->second.type, type);
        if (it->second.data + n > base + size) VFAIL("%s is cut short", name.c_str());
        return it->second;
    }
    ~File() { os::unmap(base, size); }
};

struct Pass {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> ce = nil;
    id<MTLDevice> dev;
    std::map<std::string, id<MTLComputePipelineState>> &ps;
    id<MTLLibrary> lib;
    Pass &k(const char *name) {
        if (!ce) ce = [cb computeCommandEncoder];
        auto it = ps.find(name);
        if (it == ps.end()) {
            id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!f) VFAIL("no kernel %s in the metallib", name);
            NSError *e = nil;
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
            if (!p) VFAIL("pipeline %s: %s", name, e.localizedDescription.UTF8String);
            it = ps.emplace(name, p).first;
        }
        [ce setComputePipelineState:it->second];
        return *this;
    }
    Pass &b(int i, id<MTLBuffer> buf, size_t off) { [ce setBuffer:buf offset:off atIndex:i]; return *this; }
    template <typename T> Pass &c(int i, const T &v) { [ce setBytes:&v length:sizeof v atIndex:i]; return *this; }
    void go(int gx, int threads) { [ce dispatchThreadgroups:MTLSizeMake(gx, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }
    void end() { if (ce) { [ce endEncoding]; ce = nil; } }
    void mm(id<MTLBuffer> A, size_t ao, int lda, id<MTLBuffer> B, size_t bo, int ldb, id<MTLBuffer> Cb, size_t co, int ldc, int M, int N, int K,
            bool b32 = false) {
        end();
        MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:A offset:ao descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                                     rowBytes:(size_t)lda * 4 dataType:MPSDataTypeFloat32]];
        MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:B offset:bo descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K
                                                                                     rowBytes:(size_t)ldb * (b32 ? 4 : 2)
                                                                                     dataType:b32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16]];
        MPSMatrix *mc = [[MPSMatrix alloc] initWithBuffer:Cb offset:co descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                                                                     rowBytes:(size_t)ldc * 4 dataType:MPSDataTypeFloat32]];
        MPSMatrixMultiplication *g = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:M
                                                                         resultColumns:N interiorColumns:K alpha:1 beta:0];
        [g encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
    }
};

}

struct Vision::Gpu {
    id<MTLDevice> dev = nil;
    id<MTLLibrary> lib = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLBuffer> w = nil;
    std::map<std::string, id<MTLComputePipelineState>> ps;
};

Vision::~Vision() { delete gpu; }

void Vision::open(const std::string &p, int out, void *device, void *library) {
    path = p;
    out_dim = out;
    File f;
    if (!f.open(path.c_str())) VFAIL("cannot open %s", path.c_str());
    if (f.ints["clip.vision.block_count"] != LAYERS || f.ints["clip.vision.embedding_length"] != C ||
        f.ints["clip.vision.projection_dim"] != out_dim || f.ints["clip.vision.patch_size"] != 16)
        VFAIL("%s is not Qwen3.8's vision tower for a %d-wide model", path.c_str(), out_dim);
    if (!gpu) gpu = new Gpu;
    gpu->dev = (__bridge id<MTLDevice>)device;
    gpu->lib = (__bridge id<MTLLibrary>)library;
    if (!gpu->q) gpu->q = [gpu->dev newCommandQueue];
}

void Vision::load() {
    File f;
    if (!f.open(path.c_str())) VFAIL("cannot open %s", path.c_str());
    const size_t total = unit_off(LAYERS + 3, out_dim);
    id<MTLBuffer> buf = [gpu->dev newBufferWithLength:total options:MTLResourceStorageModeShared];
    if (!buf) VFAIL("cannot allocate %.2f GB for the vision tower's weights", total / 1e9);
    uint8_t *host = (uint8_t *)buf.contents;
    enum { COPY, HALF, SUM, PAIR };
    struct Job { const File::T *t; size_t dst; int kind; int64_t n; const File::T *add = nullptr; };
    std::vector<Job> jobs;
    auto put = [&](const std::string &name, int type, std::initializer_list<int64_t> ne, size_t dst) {
        const File::T &t = f.get(name, type, ne);
        int64_t n = 1;
        for (int64_t x : ne) n *= x;
        jobs.push_back({&t, dst, type == GBF16 ? HALF : COPY, n});
    };
    {
        const EmbedOff o = embed_off();
        const File::T &w0 = f.get("v.patch_embd.weight", GF32, {16, 16, 3, C}), &w1 = f.get("v.patch_embd.weight.1", GF32, {16, 16, 3, C});
        jobs.push_back({&w0, o.w, SUM, (int64_t)C * PIN, &w1});
        jobs.push_back({&w0, o.w2, PAIR, (int64_t)C * PIN, &w1});
        put("v.patch_embd.bias", GF32, {C}, o.b);
        put("v.position_embd.weight", GF32, {C, GRID * GRID}, o.pos);
    }
    for (int l = 0; l < LAYERS; l++) {
        const BlockOff o = block_off();
        const size_t u = unit_off(1 + l, out_dim);
        const std::string p = "v.blk." + std::to_string(l) + ".";
        put(p + "attn_qkv.weight", GBF16, {C, 3 * C}, u + o.qkv_w);
        put(p + "attn_out.weight", GBF16, {C, C}, u + o.proj_w);
        put(p + "ffn_up.weight", GBF16, {C, FFN}, u + o.fc1_w);
        put(p + "ffn_down.weight", GBF16, {FFN, C}, u + o.fc2_w);
        put(p + "attn_qkv.bias", GF32, {3 * C}, u + o.qkv_b);
        put(p + "attn_out.bias", GF32, {C}, u + o.proj_b);
        put(p + "ffn_up.bias", GF32, {FFN}, u + o.fc1_b);
        put(p + "ffn_down.bias", GF32, {C}, u + o.fc2_b);
        put(p + "ln1.weight", GF32, {C}, u + o.ln1_w);
        put(p + "ln1.bias", GF32, {C}, u + o.ln1_b);
        put(p + "ln2.weight", GF32, {C}, u + o.ln2_w);
        put(p + "ln2.bias", GF32, {C}, u + o.ln2_b);
    }
    {
        const MergeOff o = merge_off();
        const size_t u = unit_off(1 + LAYERS, out_dim);
        put("v.post_ln.weight", GF32, {C}, u + o.norm_w);
        put("v.post_ln.bias", GF32, {C}, u + o.norm_b);
        put("mm.0.weight", GBF16, {MERGED, MERGED}, u + o.fc1_w);
        put("mm.0.bias", GF32, {MERGED}, u + o.fc1_b);
        const OutOff q = out_off(out_dim);
        const size_t v = unit_off(2 + LAYERS, out_dim);
        put("mm.2.weight", GBF16, {MERGED, out_dim}, v + q.fc2_w);
        put("mm.2.bias", GF32, {out_dim}, v + q.fc2_b);
    }
    constexpr int64_t PIECE = 1 << 20;
    struct Piece { const Job *j; int64_t i0, i1; };
    std::vector<Piece> pieces;
    for (const Job &j : jobs)
        for (int64_t i = 0; i < j.n; i += PIECE) pieces.push_back({&j, i, std::min(j.n, i + PIECE)});
    const Piece *pc = pieces.data();
    dispatch_apply(pieces.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
        const Job &jb = *pc[k].j;
        const int64_t i0 = pc[k].i0, i1 = pc[k].i1;
        uint8_t *dst = host + jb.dst;
        const float *a = (const float *)jb.t->data, *b = jb.add ? (const float *)jb.add->data : nullptr;
        float *o = (float *)dst;
        if (jb.kind == COPY) {
            memcpy(dst + i0 * 4, jb.t->data + i0 * 4, (i1 - i0) * 4);
        } else if (jb.kind == HALF) {
            const uint16_t *s = (const uint16_t *)jb.t->data;
            for (int64_t i = i0; i < i1; i++) ((uint16_t *)dst)[i] = f16_bits(bf16_value(s[i]));
        } else if (jb.kind == PAIR) {
            for (int64_t i = i0; i < i1; i++) {
                const int64_t c3 = i / 256, px = i % 256;
                o[(c3 * 2) * 256 + px] = a[i];
                o[(c3 * 2 + 1) * 256 + px] = b[i];
            }
        } else {
            for (int64_t i = i0; i < i1; i++) o[i] = a[i] + b[i];
        }
    });
    gpu->w = buf;
}

void Vision::release() {
    if (gpu) gpu->w = nil;
}

bool Vision::encode(const float *patches, int gh, int gw, int frames, uint16_t *out) {
    @autoreleasepool {
        const int n = gh * gw, merged = n / 4;
        if (gh % 2 || gw % 2 || n <= 0) VFAIL("a picture's grid must be whole 2 x 2 blocks (got %d x %d)", gh, gw);
        if (!gpu) VFAIL("encode before the vision tower was opened");
        if (!gpu->w) load();
        id<MTLDevice> dev = gpu->dev;
        id<MTLBuffer> W = gpu->w;
        const int pin = PIN * frames;
        const int np = (n + 63) / 64 * 64;
        size_t at = 0;
        auto take = [&](size_t bytes) { const size_t o = at; at += al(bytes); return o; };
        const size_t o_pin = take((size_t)n * pin * 4), o_x = take((size_t)n * C * 4), o_qkv = take((size_t)np * 3 * C * 4);
        const size_t o_h = take((size_t)ROWS * C * 4), o_y = take((size_t)ROWS * 3 * C * 4), o_t = take((size_t)ROWS * FFN * 4);
        const size_t o_out = take((size_t)merged * out_dim * 2);
        id<MTLBuffer> S = [dev newBufferWithLength:at options:MTLResourceStorageModeShared];
        if (!S) return false;
        memcpy((uint8_t *)S.contents + o_pin, patches, (size_t)n * pin * 4);
        struct VisRows { int r0, n, gh, gw; };
        struct VisGelu { int n, cols, exact; };
        std::vector<id<MTLCommandBuffer>> cbs;
        auto pass = [&]() { id<MTLCommandBuffer> cb = [gpu->q commandBuffer]; cbs.push_back(cb); return Pass{cb, nil, dev, gpu->ps, gpu->lib}; };
        auto chunks = [&](auto fn) { for (int r0 = 0; r0 < n; r0 += ROWS) fn(r0, std::min(ROWS, n - r0)); };
        auto groups = [](int64_t total) { return (int)((total + 255) / 256); };
        {
            Pass ps = pass();
            const EmbedOff o = embed_off();
            chunks([&](int r0, int T) {
                ps.mm(S, o_pin + (size_t)r0 * pin * 4, pin, W, frames == 2 ? o.w2 : o.w, pin, S, o_y, C, T, C, pin, true);
                const VisRows a{r0, n, gh, gw};
                ps.k("vis_embed").b(0, S, o_y).b(1, W, o.b).b(2, W, o.pos).b(3, S, o_x).c(4, a).go(T, 256);
            });
            ps.end();
            [ps.cb commit];
        }
        const BlockOff o = block_off();
        for (int l = 0; l < LAYERS; l++) {
            const size_t u = unit_off(1 + l, out_dim);
            Pass ps = pass();
            chunks([&](int r0, int T) {
                ps.k("vis_ln").b(0, S, o_x + (size_t)r0 * C * 4).b(1, W, u + o.ln1_w).b(2, W, u + o.ln1_b).b(3, S, o_h).c(4, T).go((T + 7) / 8, 256);
                ps.mm(S, o_h, C, W, u + o.qkv_w, C, S, o_y, 3 * C, T, 3 * C, C);
                const VisRows a{r0, n, gh, gw};
                ps.k("vis_qkv").b(0, S, o_y).b(1, W, u + o.qkv_b).b(2, S, o_qkv).c(3, a).go(T, 256);
            });
            ps.k("vis_attn").b(0, S, o_qkv).c(1, n);
            [ps.ce dispatchThreadgroups:MTLSizeMake(np / 64, Vision::HEADS, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            chunks([&](int r0, int T) {
                const int nc = T * C;
                ps.mm(S, o_qkv + (size_t)r0 * 3 * C * 4, 3 * C, W, u + o.proj_w, C, S, o_y, C, T, C, C);
                ps.k("vis_add").b(0, S, o_x + (size_t)r0 * C * 4).b(1, S, o_y).b(2, W, u + o.proj_b).c(3, nc).go(groups(nc), 256);
                ps.k("vis_ln").b(0, S, o_x + (size_t)r0 * C * 4).b(1, W, u + o.ln2_w).b(2, W, u + o.ln2_b).b(3, S, o_h).c(4, T).go((T + 7) / 8, 256);
                ps.mm(S, o_h, C, W, u + o.fc1_w, C, S, o_t, FFN, T, FFN, C);
                const VisGelu ga{T * FFN, FFN, 0};
                ps.k("vis_gelu").b(0, S, o_t).b(1, W, u + o.fc1_b).c(2, ga).go(groups((int64_t)T * FFN), 256);
                ps.mm(S, o_t, FFN, W, u + o.fc2_w, FFN, S, o_y, C, T, C, FFN);
                ps.k("vis_add").b(0, S, o_x + (size_t)r0 * C * 4).b(1, S, o_y).b(2, W, u + o.fc2_b).c(3, nc).go(groups(nc), 256);
            });
            ps.end();
            [ps.cb commit];
        }
        {
            Pass ps = pass();
            const size_t u1 = unit_off(1 + LAYERS, out_dim), u2 = unit_off(2 + LAYERS, out_dim);
            const MergeOff m = merge_off();
            const OutOff q = out_off(out_dim);
            for (int m0 = 0; m0 < merged; m0 += ROWS / 4) {
                const int T = std::min(ROWS / 4, merged - m0), rows = 4 * T;
                ps.k("vis_ln").b(0, S, o_x + (size_t)4 * m0 * C * 4).b(1, W, u1 + m.norm_w).b(2, W, u1 + m.norm_b).b(3, S, o_h).c(4, rows).go((rows + 7) / 8, 256);
                ps.mm(S, o_h, MERGED, W, u1 + m.fc1_w, MERGED, S, o_t, MERGED, T, MERGED, MERGED);
                const VisGelu ga{T * MERGED, MERGED, 1};
                ps.k("vis_gelu").b(0, S, o_t).b(1, W, u1 + m.fc1_b).c(2, ga).go(groups((int64_t)T * MERGED), 256);
                ps.mm(S, o_t, MERGED, W, u2 + q.fc2_w, MERGED, S, o_y, out_dim, T, out_dim, MERGED);
                const int oa[2] = {T * out_dim, out_dim};
                ps.k("vis_out").b(0, S, o_y).b(1, W, u2 + q.fc2_b).b(2, S, o_out + (size_t)m0 * out_dim * 2).c(3, oa).go(groups((int64_t)T * out_dim), 256);
            }
            ps.end();
            [ps.cb commit];
        }
        [cbs.back() waitUntilCompleted];
        for (id<MTLCommandBuffer> cb : cbs)
            if (cb.status == MTLCommandBufferStatusError) VFAIL("the GPU failed: %s", cb.error.localizedDescription.UTF8String);
        memcpy(out, (uint8_t *)S.contents + o_out, (size_t)merged * out_dim * 2);
        return true;
    }
}
