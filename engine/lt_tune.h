// hipBLASLt's first suggestion for a shape is often not its fastest on gfx1100, so a shape's first 64 suggestions are timed
// once on the real buffers (plan) and the fastest kept, and the file KILN_LT_CACHE names (kiln/engine.py:
// data/cache/hipblaslt.txt) remembers the choice by device, library version and shape: a machine times each shape once
// (0.3-0.6 s a shape). Measured against the first suggestions, per pass: the 27B's 2048-token prefill chunks 1337 ms where
// 1388, 1000 tokens 710 where 818; Flash-Next's 1000 tokens 620 where 714 (longer prompts wait on its CPU experts). Asking
// for more than 64 never found one more than 1.5% faster. Every suggestion measured gave the same bits run after run (none
// splits K with atomics), so the choice changes speed, not repeatability; suggestions differ from each other by a rounding
// here and there. The vision tower keeps the first suggestions (tune = false): tuned it was 1.5% faster (1080p: 245 ms
// where 249, its attention taking 146), and those roundings carried through its 27 blocks moved its most sensitive tokens
// further from transformers' on every test picture (the photo's relative error 1.14% where 0.80%). KILN_LT_TUNE=0 takes
// the first suggestions everywhere, to measure.
#pragma once
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace lt_tune {

constexpr int CANDIDATES = 64, FINALISTS = 6;
constexpr float LEAD_MS = 10, MEASURE_MS = 20;

struct Plan { hipblasLtMatmulDesc_t desc; hipblasLtMatrixLayout_t a, b, c; hipblasLtMatmulAlgo_t algo; };

struct Known {
    std::mutex mu;
    std::map<std::string, int> pick;                 // key -> place in the heuristic's list, which is the same each time
    Known() {
        if (const char *path = getenv("KILN_LT_CACHE"))
            if (FILE *f = fopen(path, "r")) {
                char line[512];
                while (fgets(line, sizeof line, f)) {
                    const std::string s(line);
                    const size_t sp = s.rfind(' ');
                    if (sp != std::string::npos && sp > 0) pick[s.substr(0, sp)] = atoi(s.c_str() + sp + 1);
                }
                fclose(f);
            }
    }
};

inline Known &known() { static Known k; return k; }

// The device and its compute units (a 7900 XT is a gfx1100 too) and hipBLASLt's version.
inline const std::string &device(hipblasLtHandle_t lt) {
    static const std::string d = [lt] {
        int dev = 0, ver = 0;
        (void)hipGetDevice(&dev);
        hipDeviceProp_t p;
        (void)hipGetDeviceProperties(&p, dev);
        hipblasLtGetVersion(lt, &ver);
        const std::string arch = p.gcnArchName;
        return arch.substr(0, arch.find(':')) + " " + std::to_string(p.multiProcessorCount) + " " + std::to_string(ver);
    }();
    return d;
}

// D[T][N] = B[T][K] A[N][K]^T (column-major, as hipBLASLt sees it: D = A^T B): A and B fp16, B's rows ldb apart, D fp16 or
// fp32 (d32) with rows ldd apart, fp32 accumulation. The suggestions are timed on the caller's buffers, D included, at
// beta 0: the caller's own run overwrites what they leave.
inline Plan plan(hipblasLtHandle_t lt, int N, int K, int T, int ldb, int ldd, bool d32, const void *A, const void *B,
                 void *D, void *ws, size_t ws_bytes, hipStream_t st, bool tune = true) {
    Plan p;
    hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16F, K, N, K);
    hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16F, K, T, ldb);
    hipblasLtMatrixLayoutCreate(&p.c, d32 ? HIP_R_32F : HIP_R_16F, N, T, ldd);
    std::vector<hipblasLtMatmulHeuristicResult_t> res(CANDIDATES);
    int n = 0;
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
    hipblasLtMatmulAlgoGetHeuristic(lt, p.desc, p.a, p.b, p.c, p.c, pref, CANDIDATES, res.data(), &n);
    hipblasLtMatmulPreferenceDestroy(pref);
    char shape[128];
    snprintf(shape, sizeof shape, "%zu %d %d %d %d %d %d", ws_bytes, N, K, T, ldb, ldd, (int)d32);
    if (n < 1) {
        fprintf(stderr, "kiln: hipBLASLt has no algorithm for %s\n", shape);
        abort();
    }
    p.algo = res[0].algo;
    const char *env = getenv("KILN_LT_TUNE");
    if (!tune || n == 1 || (env && !strcmp(env, "0"))) return p;
    const std::string key = device(lt) + " " + shape;
    Known &kn = known();
    std::lock_guard<std::mutex> lock(kn.mu);
    const auto it = kn.pick.find(key);
    if (it != kn.pick.end() && it->second >= 0 && it->second < n) {
        p.algo = res[it->second].algo;
        return p;
    }

    // A burst of one run each (after the run that loads the kernel) picks finalists, then each finalist runs back to back
    // long enough for the card's power limit to act: a long prefill holds this card at its 301 W cap, where a kernel
    // faster in a burst may draw more and run slower (measured: 2170 MHz where the first suggestion's ran at 2420, the
    // pass 2% slower; the ranking in back-to-back runs held from 30 ms to 1.6 s). hipBLASLt's first is always a finalist.
    const float one = 1.0f, zero = 0.0f;
    auto run = [&](int i) {
        return hipblasLtMatmul(lt, p.desc, &one, A, p.a, B, p.b, &zero, D, p.c, D, p.c, &res[i].algo, ws, ws_bytes, st)
            == HIPBLAS_STATUS_SUCCESS;
    };
    if (!run(0)) return p;                           // the caller's own run reports it
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
    auto timed = [&](int i, int reps) {
        (void)hipEventRecord(e0, st);
        for (int r = 0; r < reps; r++) run(i);
        (void)hipEventRecord(e1, st);
        (void)hipEventSynchronize(e1);
        float t = 0;
        (void)hipEventElapsedTime(&t, e0, e1);
        return std::max(t, 1e-3f) / reps;
    };
    for (int r = std::max(2, (int)(LEAD_MS / timed(0, 1))); r > 0; r--) run(0);   // clocks up before the burst
    std::vector<float> burst(n, 1e30f);
    for (int i = 0; i < n; i++)
        if (run(i)) burst[i] = timed(i, 1);
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int x, int y) { return burst[x] < burst[y]; });
    std::vector<int> finalists;
    if (burst[0] < 1e30f) finalists.push_back(0);
    for (int i : order)
        if ((int)finalists.size() < FINALISTS && i != 0 && burst[i] < 1e30f) finalists.push_back(i);
    int pick = 0;
    float fastest = 1e30f;
    for (int i : finalists) {
        for (int r = std::max(2, (int)(LEAD_MS / burst[i])); r > 0; r--) run(i);
        const float t = timed(i, std::max(4, (int)(MEASURE_MS / burst[i])));
        if (t < fastest) fastest = t, pick = i;
    }
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    kn.pick[key] = pick;
    if (const char *path = getenv("KILN_LT_CACHE"))
        if (FILE *f = fopen(path, "a")) {
            fprintf(f, "%s %d\n", key.c_str(), pick);
            fclose(f);
        }
    p.algo = res[pick].algo;
    return p;
}

}  // namespace lt_tune
