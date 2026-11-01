#pragma once
#include <cuda_runtime.h>
#include <cublasLt.h>
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

struct Plan { cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; };

struct Known {
    std::mutex mu;
    std::map<std::string, int> pick;
    Known() {
        if (const char *path = getenv("CHRYSALIS_LT_CACHE"))
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

inline const std::string &device() {
    static const std::string d = [] {
        int dev = 0;
        (void)cudaGetDevice(&dev);
        cudaDeviceProp p;
        (void)cudaGetDeviceProperties(&p, dev);
        std::string name = p.name;
        std::replace(name.begin(), name.end(), ' ', '_');
        return name + " sm" + std::to_string(p.major * 10 + p.minor) + " " + std::to_string(p.multiProcessorCount) + " " +
               std::to_string(cublasLtGetVersion());
    }();
    return d;
}

inline Plan plan(cublasLtHandle_t lt, int N, int K, int T, int ldb, int ldd, bool d32, const void *A, const void *B,
                 void *D, void *ws, size_t ws_bytes, cudaStream_t st, bool tune = true) {
    Plan p;
    cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, K, N, K);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, K, T, ldb);
    cublasLtMatrixLayoutCreate(&p.c, d32 ? CUDA_R_32F : CUDA_R_16F, N, T, ldd);
    std::vector<cublasLtMatmulHeuristicResult_t> res(CANDIDATES);
    int n = 0;
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
    cublasLtMatmulAlgoGetHeuristic(lt, p.desc, p.a, p.b, p.c, p.c, pref, CANDIDATES, res.data(), &n);
    cublasLtMatmulPreferenceDestroy(pref);
    char shape[128];
    snprintf(shape, sizeof shape, "%zu %d %d %d %d %d %d", ws_bytes, N, K, T, ldb, ldd, (int)d32);
    if (n < 1) {
        fprintf(stderr, "chrysalis: cuBLASLt has no algorithm for %s\n", shape);
        abort();
    }
    p.algo = res[0].algo;
    const char *env = getenv("CHRYSALIS_LT_TUNE");
    if (!tune || n == 1 || (env && !strcmp(env, "0"))) return p;
    const std::string key = device() + " " + shape;
    Known &kn = known();
    std::lock_guard<std::mutex> lock(kn.mu);
    const auto it = kn.pick.find(key);
    if (it != kn.pick.end() && it->second >= 0 && it->second < n) {
        p.algo = res[it->second].algo;
        return p;
    }

    const float one = 1.0f, zero = 0.0f;
    auto run = [&](int i) {
        return cublasLtMatmul(lt, p.desc, &one, A, p.a, B, p.b, &zero, D, p.c, D, p.c, &res[i].algo, ws, ws_bytes, st) == CUBLAS_STATUS_SUCCESS;
    };
    if (!run(0)) return p;
    cudaEvent_t e0, e1;
    (void)cudaEventCreate(&e0);
    (void)cudaEventCreate(&e1);
    auto timed = [&](int i, int reps) {
        (void)cudaEventRecord(e0, st);
        for (int r = 0; r < reps; r++) run(i);
        (void)cudaEventRecord(e1, st);
        (void)cudaEventSynchronize(e1);
        float t = 0;
        (void)cudaEventElapsedTime(&t, e0, e1);
        return std::max(t, 1e-3f) / reps;
    };
    for (int r = std::max(2, (int)(LEAD_MS / timed(0, 1))); r > 0; r--) run(0);
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
    (void)cudaEventDestroy(e0);
    (void)cudaEventDestroy(e1);
    kn.pick[key] = pick;
    if (const char *path = getenv("CHRYSALIS_LT_CACHE"))
        if (FILE *f = fopen(path, "a")) {
            fprintf(f, "%s %d\n", key.c_str(), pick);
            fclose(f);
        }
    p.algo = res[pick].algo;
    return p;
}

}
