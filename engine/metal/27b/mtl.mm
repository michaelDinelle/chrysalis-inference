#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>
#include "mtl.h"
#include <chrono>
#include <unistd.h>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <atomic>
#include <condition_variable>
#include <thread>

#define OBJ(T, p) ((__bridge T)(p))

struct Mtl::Warm { std::atomic<double> idle{0}; std::mutex mu; std::condition_variable cv; };

void Mtl::init(const std::string &metallib) {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) FAIL("no Metal device");
    dev = (void *)CFBridgingRetain(d);
    max_ws = [d recommendedMaxWorkingSetSize];
    NSError *e = nil;
    id<MTLLibrary> l = [d newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&e];
    if (!l) FAIL("cannot load %s: %s", metallib.c_str(), e ? [[e description] UTF8String] : "?");
    lib = (void *)CFBridgingRetain(l);
    archive_path = metallib + ".archive";
    MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
    if (access(archive_path.c_str(), R_OK) == 0) ad.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:archive_path.c_str()]];
    id<MTLBinaryArchive> ar = [d newBinaryArchiveWithDescriptor:ad error:&e];
    if (!ar) { ad.url = nil; ar = [d newBinaryArchiveWithDescriptor:ad error:&e]; }
    archive = (void *)CFBridgingRetain(ar);
    id<MTLCommandQueue> q = [d newCommandQueueWithMaxCommandBufferCount:64];
    queue = (void *)CFBridgingRetain(q);
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.initialCapacity = 256;
    id<MTLResidencySet> rs = [d newResidencySetWithDescriptor:rd error:&e];
    if (!rs) FAIL("residency set: %s", [[e description] UTF8String]);
    rset = (void *)CFBridgingRetain(rs);
    [q addResidencySet:rs];
}

DP<uint8_t> Mtl::alloc(size_t bytes, const char *what) {
    const DP<uint8_t> p = try_alloc(bytes);
    if (!p) FAIL("cannot allocate %.2f GB for %s (%.2f GB allocated)", bytes / 1e9, what, allocated / 1e9);
    return p;
}

DP<uint8_t> Mtl::try_alloc(size_t bytes) {
    bytes = (bytes + 16383) & ~size_t(16383);
    id<MTLBuffer> b = [OBJ(id<MTLDevice>, dev) newBufferWithLength:bytes options:MTLResourceStorageModeShared |
                                                                     MTLResourceHazardTrackingModeTracked];
    if (!b) return {};
    id<MTLResidencySet> rs = OBJ(id<MTLResidencySet>, rset);
    [rs addAllocation:b];
    [rs commit];
    if (resident) [rs requestResidency];
    bufs.push_back((void *)CFBridgingRetain(b));
    ranges.push_back({[b gpuAddress], bytes});
    allocated += bytes;
    return {[b gpuAddress], (uint8_t *)[b contents]};
}

void Mtl::make_resident() { resident = true; [OBJ(id<MTLResidencySet>, rset) requestResidency]; }

void Mtl::free_all() {
    sync();
    id<MTLResidencySet> rs = OBJ(id<MTLResidencySet>, rset);
    [rs removeAllAllocations];
    [rs commit];
    for (void *b : bufs) CFBridgingRelease(b);
    bufs.clear();
    ranges.clear();
    allocated = 0;
}

void *Mtl::pipeline(const char *name, const int *consts, int nconsts) {
    static std::map<std::string, void *> cache;
    static std::mutex mu;
    static const std::vector<std::string> skips = [] {
        std::vector<std::string> v;
        if (const char *e = getenv("CHRYSALIS_SKIP"))
            for (std::string s = e; !s.empty();) { const size_t c = s.find(','); v.push_back(s.substr(0, c)); s = c == std::string::npos ? "" : s.substr(c + 1); }
        return v;
    }();
    for (const std::string &k : skips) if (!strncmp(name, k.c_str(), k.size())) return (void *)1;
    std::string key = name;
    for (int i = 0; i < nconsts; i++) key += "," + std::to_string(consts[i]);
    {
        std::lock_guard<std::mutex> g(mu);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }
    NSError *e = nil;
    id<MTLFunction> f;
    id<MTLLibrary> l = OBJ(id<MTLLibrary>, lib);
    NSString *n = [NSString stringWithUTF8String:name];
    if (nconsts) {
        MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
        for (int i = 0; i < nconsts; i++) [cv setConstantValue:&consts[i] type:MTLDataTypeInt atIndex:i];
        f = [l newFunctionWithName:n constantValues:cv error:&e];
    } else {
        f = [l newFunctionWithName:n];
    }
    if (!f) FAIL("no kernel %s: %s", key.c_str(), e ? [[e description] UTF8String] : "");
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = f;
    id<MTLBinaryArchive> ar = OBJ(id<MTLBinaryArchive>, archive);
    pd.binaryArchives = @[ar];
    id<MTLDevice> d = OBJ(id<MTLDevice>, dev);
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                  reflection:nil error:&e];
    if (!p) {
        p = [d newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&e];
        std::lock_guard<std::mutex> g(mu);
        if (p && [ar addComputePipelineFunctionsWithDescriptor:pd error:nil]) archive_dirty = true;
    }
    if (!p) FAIL("pipeline %s: %s", key.c_str(), [[e description] UTF8String]);
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    void *r = (void *)CFBridgingRetain(p);
    cache[key] = r;
    return r;
}

void Mtl::save_archive() {
    if (!archive_dirty) return;
    NSError *e = nil;
    if (![OBJ(id<MTLBinaryArchive>, archive) serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:archive_path.c_str()]] error:&e])
        fprintf(stderr, "chrysalis: cannot write %s: %s\n", archive_path.c_str(), [[e description] UTF8String]);
    archive_dirty = false;
}

void Mtl::begin() {
    id<MTLCommandBuffer> c = [OBJ(id<MTLCommandQueue>, queue) commandBufferWithUnretainedReferences];
    cb = (void *)CFBridgingRetain(c);
}

void *Mtl::enc_begin() {
    if (!cb) begin();
    if (!enc) {
        id<MTLComputeCommandEncoder> en = [OBJ(id<MTLCommandBuffer>, cb) computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        enc = (void *)CFBridgingRetain(en);
        skip_barrier = true;
        for (void *b : bufs) [en useResource:OBJ(id<MTLBuffer>, b) usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    return enc;
}

void Mtl::gemm(uint64_t A, int lda, uint64_t B, uint64_t C, int ldc, bool c_f32, int M, int N, int K, float beta, int ldb) {
    auto find = [&](uint64_t g, size_t &off) -> id<MTLBuffer> {
        for (size_t i = 0; i < ranges.size(); i++)
            if (g >= ranges[i].first && g < ranges[i].first + ranges[i].second) { off = g - ranges[i].first; return OBJ(id<MTLBuffer>, bufs[i]); }
        FAIL("gemm: address %llx is in no buffer", (unsigned long long)g);
    };
    size_t oa, ob, oc;
    id<MTLBuffer> ba = find(A, oa), bb = find(B, ob), bc = find(C, oc);
    if (!cb) begin();
    if (enc) { [OBJ(id<MTLComputeCommandEncoder>, enc) endEncoding]; CFBridgingRelease(enc); enc = nullptr; }
    MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:ba offset:oa descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:(size_t)lda * 2 dataType:MPSDataTypeFloat16]];
    MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:bb offset:ob descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:(size_t)(ldb ? ldb : K) * 2 dataType:MPSDataTypeFloat16]];
    MPSMatrix *mc = [[MPSMatrix alloc] initWithBuffer:bc offset:oc descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
        rowBytes:(size_t)ldc * (c_f32 ? 4 : 2) dataType:c_f32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16]];
    static std::map<std::tuple<int, int, int, float>, MPSMatrixMultiplication *> plans;
    MPSMatrixMultiplication *mm = plans[{M, N, K, beta}];
    if (!mm) {
        mm = [[MPSMatrixMultiplication alloc] initWithDevice:OBJ(id<MTLDevice>, dev) transposeLeft:NO transposeRight:YES
                  resultRows:M resultColumns:N interiorColumns:K alpha:1 beta:beta];
        plans[{M, N, K, beta}] = mm;
    }
    [mm encodeToCommandBuffer:OBJ(id<MTLCommandBuffer>, cb) leftMatrix:ma rightMatrix:mb resultMatrix:mc];
}

void Mtl::run(void *pso, const void *args, size_t nargs, int gx, int gy, int gz, int tx, int ty, int tz) {
    if (gx <= 0 || gy <= 0 || gz <= 0 || pso == (void *)1) return;
    id<MTLComputeCommandEncoder> en = OBJ(id<MTLComputeCommandEncoder>, enc_begin());
    static const bool nobar = getenv("CHRYSALIS_NOBAR");
    if (!skip_barrier && !nobar) [en memoryBarrierWithScope:MTLBarrierScopeBuffers];
    skip_barrier = false;
    [en setComputePipelineState:OBJ(id<MTLComputePipelineState>, pso)];
    [en setBytes:args length:nargs atIndex:0];
    [en dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
    dispatches++;
}

void Mtl::commit() {
    if (!cb) return;
    if (enc) { [OBJ(id<MTLComputeCommandEncoder>, enc) endEncoding]; CFBridgingRelease(enc); }
    id<MTLCommandBuffer> c = (__bridge_transfer id<MTLCommandBuffer>)cb;
    commit_time = CACurrentMediaTime();
    [c commit];
    if (last_cb) CFBridgingRelease(last_cb);
    last_cb = (void *)CFBridgingRetain(c);
    cb = enc = nullptr;
    if (warm) warm->idle = 0;
}

void Mtl::warm_start() {
    warm = new Warm;
    Warm *w = warm;
    id<MTLCommandQueue> q2 = [OBJ(id<MTLDevice>, dev) newCommandQueue];
    id<MTLComputePipelineState> pso = OBJ(id<MTLComputePipelineState>, pipeline("fill16"));
    std::thread([w, q2, pso] {
        const uint64_t none[2] = {0, 0};
        std::unique_lock<std::mutex> lk(w->mu);
        for (double done = 0;;) {
            w->cv.wait(lk, [&] { const double t = w->idle.load(); return t != 0 && t != done; });
            const double t0 = w->idle.load();
            lk.unlock();
            while (w->idle.load() == t0 && CACurrentMediaTime() - t0 < 0.005) {
                @autoreleasepool {
                    id<MTLCommandBuffer> c = [q2 commandBufferWithUnretainedReferences];
                    id<MTLComputeCommandEncoder> en = [c computeCommandEncoder];
                    [en setComputePipelineState:pso];
                    [en setBytes:none length:sizeof none atIndex:0];
                    [en dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [en endEncoding];
                    [c commit];
                }
                usleep(50);
            }
            done = t0;
            lk.lock();
        }
    }).detach();
}

void Mtl::sync() {
    commit();
    if (!last_cb) return;
    id<MTLCommandBuffer> c = OBJ(id<MTLCommandBuffer>, last_cb);
    [c waitUntilCompleted];
    if (c.status == MTLCommandBufferStatusError) FAIL("command buffer failed: %s", [[c.error description] UTF8String]);
    if (!warm) warm_start();
    {
        std::lock_guard<std::mutex> g(warm->mu);
        warm->idle = CACurrentMediaTime();
    }
    warm->cv.notify_one();
    last_gpu_ms = (c.GPUEndTime - c.GPUStartTime) * 1e3;
    static const bool trace = getenv("CHRYSALIS_TRACE");
    if (trace) {
        const double t = CACurrentMediaTime();
        last_start_ms = (c.GPUStartTime - commit_time) * 1e3;
        last_wake_ms = (t - c.GPUEndTime) * 1e3;
    }
    CFBridgingRelease(last_cb);
    last_cb = nullptr;
}
