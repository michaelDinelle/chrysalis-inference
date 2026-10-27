#pragma once
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <string>
#include <unordered_map>
#include "fn.h"

struct View {
    __unsafe_unretained id<MTLBuffer> buf = nil;
    size_t off = 0;
    template <typename T = void> T *host() const { return (T *)((uint8_t *)buf.contents + off); }
    View at(size_t bytes) const { return {buf, off + bytes}; }
    uint64_t gpu() const { return buf.gpuAddress + off; }
};

struct Mtl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLCommandQueue> q2 = nil;
    id<MTLLibrary> lib = nil;
    id<MTLResidencySet> rs = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> ps;
    std::vector<id<MTLBuffer>> owned;
    size_t allocated = 0;
    id<MTLFence> fence = nil;

    void init() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) FAIL("no Metal device");
        q = [dev newCommandQueueWithMaxCommandBufferCount:64];
        Dl_info info;
        dladdr((const void *)&Mtl::anchor, &info);
        std::string dir = info.dli_fname;
        dir = dir.substr(0, dir.find_last_of('/') + 1);
        NSError *e = nil;
        NSString *p = [NSString stringWithUTF8String:(dir + "chrysalis_fn.metallib").c_str()];
        NSData *bytes = [NSData dataWithContentsOfFile:p options:0 error:&e];
        dispatch_data_t dd = bytes ? dispatch_data_create(bytes.bytes, bytes.length, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT) : nullptr;
        lib = dd ? [dev newLibraryWithData:dd error:&e] : nil;
        if (!lib) FAIL("cannot load %s: %s", p.UTF8String, e.localizedDescription.UTF8String);
        MTLResidencySetDescriptor *d = [MTLResidencySetDescriptor new];
        d.initialCapacity = 256;
        rs = [dev newResidencySetWithDescriptor:d error:&e];
        if (!rs) FAIL("cannot make a residency set: %s", e.localizedDescription.UTF8String);
        [q addResidencySet:rs];
        q2 = [dev newCommandQueue];
        [q2 addResidencySet:rs];
        fence = [dev newFence];
    }
    static void anchor() {}
    id<MTLComputePipelineState> pipe(const std::string &name) {
        auto it = ps.find(name);
        if (it != ps.end()) return it->second;
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (!f) FAIL("no kernel %s in the metallib", name.c_str());
        NSError *e = nil;
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&e];
        if (!p) FAIL("pipeline %s: %s", name.c_str(), e.localizedDescription.UTF8String);
        ps[name] = p;
        return p;
    }
    View alloc(size_t bytes) {
        id<MTLBuffer> b = [dev newBufferWithLength:(bytes + 16383) & ~size_t(16383)
                                           options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (!b) FAIL("cannot allocate a %.2f GB Metal buffer (%.2f GB allocated)", bytes / 1e9, allocated / 1e9);
        owned.push_back(b);
        allocated += b.length;
        [rs addAllocation:b];
        [rs commit];
        return {b, 0};
    }
    void release_all() {
        [rs removeAllAllocations];
        [rs commit];
        for (id<MTLBuffer> b : owned) munlock(b.contents, b.length);
        owned.clear();
        allocated = 0;
    }
};

struct Arena {
    View base; size_t cap = 0, used = 0;
    void init(Mtl &m, size_t bytes) { base = m.alloc(bytes); cap = bytes; used = 0; }
    View get(size_t bytes) {
        bytes = (bytes + 255) & ~size_t(255);
        if (used + bytes > cap) FAIL("arena out of memory (%zu + %zu > %zu)", used, bytes, cap);
        View v = base.at(used);
        used += bytes;
        return v;
    }
};

struct Enc {
    id<MTLComputeCommandEncoder> e = nil;
    Mtl *m = nullptr;
    id<MTLCounterSampleBuffer> ts = nil;
    uint32_t ts_n = 0;
    void begin(Mtl &mt, id<MTLCommandBuffer> cb) {
        m = &mt;
        start(cb);
        [e waitForFence:m->fence];
    }
    void start(id<MTLCommandBuffer> cb) {
        if (!ts) { e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; return; }
        MTLComputePassDescriptor *d = [MTLComputePassDescriptor computePassDescriptor];
        d.dispatchType = MTLDispatchTypeConcurrent;
        d.sampleBufferAttachments[0].sampleBuffer = ts;
        d.sampleBufferAttachments[0].startOfEncoderSampleIndex = ts_n++;
        d.sampleBufferAttachments[0].endOfEncoderSampleIndex = ts_n++;
        e = [cb computeCommandEncoderWithDescriptor:d];
    }
    void end() {
        if (!e) return;
        [e updateFence:m->fence];
        [e endEncoding];
        e = nil;
    }
    Enc &k(const char *name) { [e setComputePipelineState:m->pipe(name)]; return *this; }
    Enc &k(const std::string &name) { return k(name.c_str()); }
    Enc &b(int i, const View &v) { [e setBuffer:v.buf offset:v.off atIndex:i]; return *this; }
    template <typename T> Enc &c(int i, const T &v) { [e setBytes:&v length:sizeof(T) atIndex:i]; return *this; }
    Enc &tgmem(int i, size_t n) { [e setThreadgroupMemoryLength:(n + 15) & ~size_t(15) atIndex:i]; return *this; }
    View args, grids;
    uint32_t nind = 0;
    bool ind = false;
    void grid(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t threads) {
        if (!ind) { [e dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; return; }
        uint32_t *g = grids.host<uint32_t>() + 3 * nind;
        g[0] = gx; g[1] = gy; g[2] = gz;
        [e dispatchThreadgroupsWithIndirectBuffer:args.buf indirectBufferOffset:args.off + 12 * nind threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        nind++;
    }
    void grid(uint32_t gx, uint32_t threads) { grid(gx, 1, 1, threads); }
    void barrier() { [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
};
