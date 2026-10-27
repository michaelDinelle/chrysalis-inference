#include "engine/metal/flash-next/prefill.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <ggml.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <random>
#include <thread>

template <class F> static void par(int n, F f) {
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (unsigned k = 0; k < std::thread::hardware_concurrency(); k++)
        th.emplace_back([&] { for (int i; (i = next++) < n;) f(i); });
    for (auto &t : th) t.join();
}

static void deq_rows(int type, const uint8_t *src, size_t rb, int cols, int r0, int n, float *dst) {
    const ggml_type_traits *tr = ggml_get_type_traits((ggml_type)type);
    for (int r = 0; r < n; r++) {
        const uint8_t *p = src + (size_t)(r0 + r) * rb;
        if (type == 0) memcpy(dst + (size_t)r * cols, p, cols * 4);
        else tr->to_float(p, dst + (size_t)r * cols, cols);
    }
}

static View load(Mtl &m, GGUF &g, const GGUFTensor &t) {
    View v = m.alloc(t.nbytes);
    for (size_t o = 0; o < t.nbytes;) {
        const ssize_t r = pread(g.fds[t.file], v.host<uint8_t>() + o, t.nbytes - o, t.off + o);
        if (r <= 0) FAIL("read failed");
        o += r;
    }
    return v;
}

template <class F> static double gpu_time(Mtl &m, int reps, F enc) {
    double best = 1e9;
    for (int k = 0; k < 2; k++) {
        id<MTLCommandBuffer> cb = [m.q commandBuffer];
        Enc e; e.begin(m, cb);
        for (int i = 0; i < reps; i++) { enc(e); e.barrier(); }
        e.end();
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
        best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / reps);
    }
    return best;
}
static void run(Mtl &m, const std::function<void(Enc &)> &enc) {
    id<MTLCommandBuffer> cb = [m.q commandBuffer];
    Enc e; e.begin(m, cb);
    enc(e);
    e.end();
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
}

static std::mt19937 rng(1);
static void fill(View v, size_t n, float lo = -1, float hi = 1) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (size_t i = 0; i < n; i++) v.host<float>()[i] = d(rng);
}

struct Dense { std::string name; int type, rows, cols; size_t rb; View w; };

static Dense dense(Mtl &m, GGUF &g, const std::string &name) {
    const GGUFTensor &t = g.tensors.at(name);
    return {name, t.type, (int)(t.ne[1] * t.ne[2]), (int)t.ne[0], ggml_row_bytes(t.type, t.ne[0]), load(m, g, t)};
}

static std::vector<Dense> pick_dense(Mtl &m, GGUF &g) {
    std::map<int, std::string> pick;
    for (auto &[n, t] : g.tensors) {
        if (n.find("_hot.") != std::string::npos || n.find("_cold") != std::string::npos || n.find("_mtp.") != std::string::npos ||
            n.find("token_embd") != std::string::npos || n == "output.weight") continue;
        if (t.ne[1] < 64 || t.ne[1] > 16384 || t.ne[2] > 1 || t.ne[0] % 32 || t.type == 0 || t.type == 1) continue;
        auto it = pick.find(t.type);
        if (it == pick.end() || t.ne[1] > g.tensors[it->second].ne[1]) pick[t.type] = n;
    }
    std::vector<Dense> v;
    for (auto &[type, n] : pick) v.push_back(dense(m, g, n));
    return v;
}

static bool check_dense(Mtl &m, const Dense &d, int T, bool add, float scale, double tol) {
    View x = m.alloc((size_t)T * d.cols * 4), y = m.alloc((size_t)T * d.rows * 4);
    fill(x, (size_t)T * d.cols);
    fill(y, (size_t)T * d.rows);
    std::vector<float> y0(y.host<float>(), y.host<float>() + (size_t)T * d.rows);
    run(m, [&](Enc &e) { pf_mm(e, d.type, d.w, d.rows, d.cols, d.rb, x, d.cols, y, d.rows, T, add, scale); });
    std::vector<int> rows;
    const int ns = std::min(d.rows, 384);
    for (int i = 0; i < ns; i++) rows.push_back((int)((int64_t)i * d.rows / ns));
    rows.back() = d.rows - 1;
    std::vector<double> worst(rows.size(), 0);
    std::vector<int> wt(rows.size(), 0);
    par((int)rows.size(), [&](int i) {
        std::vector<float> w(d.cols);
        deq_rows(d.type, d.w.host<uint8_t>(), d.rb, d.cols, rows[i], 1, w.data());
        for (int t = 0; t < T; t++) {
            const float *xt = x.host<float>() + (size_t)t * d.cols;
            double s = 0, mag = 0;
            for (int c = 0; c < d.cols; c++) { s += (double)w[c] * xt[c]; mag += fabs((double)w[c] * xt[c]); }
            const size_t o = (size_t)t * d.rows + rows[i];
            const double ref = (add ? y0[o] : 0.0) + scale * s;
            const double err = fabs(y.host<float>()[o] - ref) / (fabs(scale) * mag + 1e-30);
            if (err > worst[i]) { worst[i] = err; wt[i] = t; }
        }
    });
    const size_t wi = std::max_element(worst.begin(), worst.end()) - worst.begin();
    const double wst = worst[wi];
    const bool ok = wst < tol;
    printf("%-4s pf_mm %-6s %-30s %5d x %5d T=%d%s: max |err| / sum|wx| = %.3g (t %d, n %d)\n", ok ? "ok" : "FAIL", pf_tname(d.type), d.name.c_str(),
           d.rows, d.cols, T, add ? " add, scale 0.5" : "", wst, wt[wi], rows[wi]);
    return ok;
}

struct Moe {
    int layer = 3;
    View slots, table, fmt;
    XSrc src[NE];
    Dense router;
    std::vector<int> slot_of;
};

static Moe moe_load(Mtl &m, GGUF &g, int layer) {
    Moe x;
    x.layer = layer;
    const std::string p = "blk." + std::to_string(layer) + ".";
    expert_sources(g, layer, x.src);
    x.slots = m.alloc(XSLOT * NE);
    x.table = m.alloc((size_t)NL * NE * 8);
    x.fmt = m.alloc((size_t)NL * NE);
    memset(x.table.host(), 0, (size_t)NL * NE * 8);
    for (int e = 0; e < NE; e++) x.fmt.host<uint8_t>()[layer * NE + e] = (uint8_t)x.src[e].type;
    x.slot_of.resize(NE);
    for (int e = 0; e < NE; e++) x.slot_of[e] = (e * 7 + 3) % NE;
    par(NE * 3, [&](int i) {
        const int e = i / 3, k = i % 3;
        const size_t gu = xgu_bytes(x.src[e].type), n = k ? gu : XDN, at = k ? XDN + (k - 1) * gu : 0;
        uint8_t *dst = x.slots.host<uint8_t>() + (size_t)x.slot_of[e] * XSLOT + at;
        if (pread(g.fds[0], dst, n, x.src[e].off[k]) != (ssize_t)n) FAIL("read experts");
    });
    for (int e = 0; e < NE; e++) x.table.host<uint64_t>()[layer * NE + e] = x.slots.gpu() + (size_t)x.slot_of[e] * XSLOT;
    x.router = dense(m, g, p + "ffn_router.weight");
    return x;
}

static void route(Mtl &m, const Moe &x, View X, int T, View ids, View wts) {
    View lg = m.alloc((size_t)T * NE * 4);
    run(m, [&](Enc &e) { pf_mm(e, x.router.type, x.router.w, NE, D, x.router.rb, X, D, lg, NE, T); });
    par(T, [&](int t) {
        const float *l = lg.host<float>() + (size_t)t * NE;
        int idx[NE];
        for (int i = 0; i < NE; i++) idx[i] = i;
        std::partial_sort(idx, idx + 10, idx + NE, [&](int a, int b) { return l[a] > l[b]; });
        double s = 0;
        for (int k = 0; k < 10; k++) s += exp(l[idx[k]] - l[idx[0]]);
        for (int k = 0; k < 10; k++) { ids.host<int>()[t * 10 + k] = idx[k]; wts.host<float>()[t * 10 + k] = (float)(exp(l[idx[k]] - l[idx[0]]) / s); }
    });
}

static bool check_moe(Mtl &m, const Moe &x, int T, double tol) {
    const uint32_t P = T * 10;
    View X = m.alloc((size_t)T * D * 4), ids = m.alloc(P * 4), wts = m.alloc(P * 4), Y = m.alloc((size_t)T * D * 4);
    fill(X, (size_t)T * D);
    route(m, x, X, T, ids, wts);
    std::vector<int> toks;
    for (int t = 7; t < T; t += 20) toks.push_back(t);
    std::vector<std::vector<std::pair<int, int>>> by_e(NE);
    for (size_t i = 0; i < toks.size(); i++) for (int k = 0; k < 10; k++) by_e[ids.host<int>()[toks[i] * 10 + k]].push_back({(int)i, k});
    std::vector<double> yk(toks.size() * 10 * D);
    par(NE, [&](int e) {
        if (by_e[e].empty()) return;
        const uint8_t *E = x.slots.host<uint8_t>() + (size_t)x.slot_of[e] * XSLOT;
        std::vector<float> wg((size_t)FE * D), wu((size_t)FE * D), wd((size_t)D * FE);
        const int ty = x.src[e].type;
        const size_t gu = xgu_bytes(ty);
        deq_rows(T_IQ4_NL, E, ggml_row_bytes(T_IQ4_NL, FE), FE, 0, D, wd.data());
        deq_rows(ty, E + XDN, ggml_row_bytes(ty, D), D, 0, FE, wg.data());
        deq_rows(ty, E + XDN + gu, ggml_row_bytes(ty, D), D, 0, FE, wu.data());
        std::vector<double> h(FE);
        for (auto [i, k] : by_e[e]) {
            const float *xt = X.host<float>() + (size_t)toks[i] * D;
            for (int j = 0; j < FE; j++) {
                double g = 0, u = 0;
                for (int c = 0; c < D; c++) { g += (double)wg[(size_t)j * D + c] * xt[c]; u += (double)wu[(size_t)j * D + c] * xt[c]; }
                h[j] = g / (1 + exp(-g)) * u;
            }
            double *o = &yk[((size_t)i * 10 + k) * D];
            for (int d = 0; d < D; d++) { double s = 0; for (int j = 0; j < FE; j++) s += wd[(size_t)d * FE + j] * h[j]; o[d] = s; }
        }
    });
    bool all = true;
    for (int bt : {16, 32}) {
        View scr = m.alloc(pf_moe_scratch(P, bt));
        const PfMoeBufs s = pf_moe_bufs(scr, P, bt);
        memset(Y.host(), 0xff, (size_t)T * D * 4);
        run(m, [&](Enc &e) { pf_moe(e, x.table, x.fmt, x.layer, ids, wts, X, D, T, s, Y, bt); });
        double worst = 0, worst_max = 0;
        for (size_t i = 0; i < toks.size(); i++) {
            double e2 = 0, r2 = 0, emax = 0, rmax = 0;
            for (int d = 0; d < D; d++) {
                double ref = 0;
                for (int k = 0; k < 10; k++) ref += wts.host<float>()[toks[i] * 10 + k] * yk[(i * 10 + k) * D + d];
                const double err = Y.host<float>()[(size_t)toks[i] * D + d] - ref;
                e2 += err * err; r2 += ref * ref; emax = std::max(emax, fabs(err)); rmax = std::max(rmax, fabs(ref));
            }
            const double rel = sqrt(e2 / r2);
            worst = std::max(worst, std::isnan(rel) ? 1e9 : rel);
            worst_max = std::max(worst_max, std::isnan(emax) ? 1e9 : emax / rmax);
        }
        const uint32_t *offs = s.offs.host<uint32_t>();
        std::vector<int> seen(P, 0);
        bool sort_ok = offs[NE] == P;
        for (int e = 0; e < NE && sort_ok; e++)
            for (uint32_t r = offs[e]; r < offs[e + 1]; r++) {
                const int p = s.rows.host<int>()[r];
                if (p < 0 || p >= (int)P || seen[p]++ || ids.host<int>()[p] != e) { sort_ok = false; break; }
            }
        const bool ok = sort_ok && worst < tol;
        all = all && ok;
        printf("%-4s pf_moe layer %d T=%d bt=%d: %zu tokens checked, worst |err|_2 / |ref|_2 = %.3g, max |err| / max |ref| = %.3g%s\n",
               ok ? "ok" : "FAIL", x.layer, T, bt, toks.size(), worst, worst_max, sort_ok ? "" : ", SORT BROKEN");
    }
    return all;
}

static void bench_dense(Mtl &m, const Dense &d, const std::vector<int> &Ts) {
    const int TM = *std::max_element(Ts.begin(), Ts.end());
    View x = m.alloc((size_t)TM * d.cols * 4), y = m.alloc((size_t)TM * d.rows * 4);
    fill(x, (size_t)TM * d.cols);
    for (int T : Ts) {
        const double fl = 2.0 * T * d.rows * d.cols;
        const int reps = std::max(2, std::min(50, (int)(0.15 / (fl / 8e12))));
        const double s = gpu_time(m, reps, [&](Enc &e) { pf_mm(e, d.type, d.w, d.rows, d.cols, d.rb, x, d.cols, y, d.rows, T); });
        printf("pf_mm %-6s %-30s %5d x %5d T=%5d: %8.3f ms %6.2f TFLOPS\n", pf_tname(d.type), d.name.c_str(), d.rows, d.cols, T,
               s * 1e3, fl / s / 1e12);
    }
}

static void bench_mps(Mtl &m, int N, int K, const std::vector<int> &Ts) {
    const int TM = *std::max_element(Ts.begin(), Ts.end());
    View a = m.alloc((size_t)TM * K * 2), b = m.alloc((size_t)N * K * 2), c = m.alloc((size_t)TM * N * 2);
    for (size_t i = 0; i < (size_t)TM * K; i++) a.host<__fp16>()[i] = (__fp16)(0.01f * (i % 97));
    for (size_t i = 0; i < (size_t)N * K; i++) b.host<__fp16>()[i] = (__fp16)(0.01f * (i % 89));
    for (int T : Ts) {
        MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:a.buf offset:a.off descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:T columns:K rowBytes:K * 2 dataType:MPSDataTypeFloat16]];
        MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:b.buf offset:b.off descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * 2 dataType:MPSDataTypeFloat16]];
        MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:c.buf offset:c.off descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:T columns:N rowBytes:N * 2 dataType:MPSDataTypeFloat16]];
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:m.dev transposeLeft:NO transposeRight:YES resultRows:T
                                                                       resultColumns:N interiorColumns:K alpha:1 beta:0];
        const double fl = 2.0 * T * N * K;
        const int reps = std::max(2, std::min(50, (int)(0.15 / (fl / 8e12))));
        double best = 1e9;
        for (int k = 0; k < 2; k++) {
            id<MTLCommandBuffer> cb = [m.q commandBuffer];
            for (int i = 0; i < reps; i++) [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
            [cb commit]; [cb waitUntilCompleted];
            best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / reps);
        }
        printf("MPS fp16 GEMM (fp16 out)              %5d x %5d T=%5d: %8.3f ms %6.2f TFLOPS\n", N, K, T, best * 1e3, fl / best / 1e12);
    }
}

static void bench_moe(Mtl &m, const Moe &x, const std::vector<int> &Ts) {
    for (int T : Ts) {
        const uint32_t P = T * 10;
        View X = m.alloc((size_t)T * D * 4), ids = m.alloc(P * 4), wts = m.alloc(P * 4), Y = m.alloc((size_t)T * D * 4);
        fill(X, (size_t)T * D);
        route(m, x, X, T, ids, wts);
        std::vector<int> cnt(NE, 0);
        for (uint32_t p = 0; p < P; p++) cnt[ids.host<int>()[p]]++;
        const int used = (int)std::count_if(cnt.begin(), cnt.end(), [](int c) { return c > 0; });
        const double fl = 3.0 * 2 * P * FE * D;
        for (int bt : {16, 32}) {
            View scr = m.alloc(pf_moe_scratch(P, bt));
            const PfMoeBufs s = pf_moe_bufs(scr, P, bt);
            const int reps = std::max(2, std::min(20, (int)(0.15 / (fl / 6e12))));
            const double all = gpu_time(m, reps, [&](Enc &e) { pf_moe(e, x.table, x.fmt, x.layer, ids, wts, X, D, T, s, Y, bt); });
            const double srt = gpu_time(m, reps, [&](Enc &e) { const PfSortArgs sa{P, (uint32_t)bt};
                e.k("pf_moe_sort").c(0, sa).b(1, ids).b(2, s.offs).b(3, s.rows).b(4, s.tiles).b(5, s.ntiles).grid(1, 1024); });
            const PfMoeArgs ma{(uint32_t)x.layer, D, P, 0};
            const uint32_t mt = pf_max_tiles(P, bt);
            const double gu = gpu_time(m, reps, [&](Enc &e) {
                e.k("pf_moe_gu" + std::to_string(bt)).c(0, ma).b(1, x.table).b(2, s.offs).b(3, s.rows).b(4, s.tiles).b(5, s.ntiles).b(6, X).b(7, s.H).b(8, x.fmt).grid(10, mt, 1, 128); });
            const double dn = gpu_time(m, reps, [&](Enc &e) {
                e.k("pf_moe_down" + std::to_string(bt)).c(0, ma).b(1, x.table).b(2, s.offs).b(3, s.rows).b(4, s.tiles).b(5, s.ntiles).b(6, s.H).b(7, s.Yp).grid(20, mt, 1, 128); });
            printf("pf_moe layer %d T=%5d bt=%2d (%3d experts used, %u tiles): %8.3f ms = %7.0f tok/s, %5.2f TFLOPS | sort %.3f gu %.3f (%.2f TF) down %.3f (%.2f TF) ms\n",
                   x.layer, T, bt, used, *s.ntiles.host<uint32_t>(), all * 1e3, T / all, fl / all / 1e12, srt * 1e3, gu * 1e3,
                   fl * 2 / 3 / gu / 1e12, dn * 1e3, fl / 3 / dn / 1e12);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <qwen3.8-flash-next.gguf> [check | dense | bench]\n", argv[0]); return 2; }
    const std::string mode = argc > 2 ? argv[2] : "check";
    GGUF g;
    if (!g.open(argv[1])) FAIL("cannot open %s", argv[1]);
    Mtl m;
    m.init();
    std::vector<Dense> ds = pick_dense(m, g);
    if (mode == "bench") {
        for (auto &d : ds) bench_dense(m, d, {512, 2048, 8192});
        bench_dense(m, dense(m, g, "blk.0.hc_attn_down.weight"), {512, 2048, 8192});
        Dense f{"random fp16", 1, 12288, 2560, 2560 * 2, m.alloc((size_t)12288 * 2560 * 2)};
        for (size_t i = 0; i < (size_t)12288 * 2560; i++) f.w.host<__fp16>()[i] = (__fp16)(0.01f * (int)(i % 89) - 0.44f);
        bench_dense(m, f, {512, 2048, 8192});
        bench_mps(m, 12288, 2560, {512, 2048, 8192});
        Dense f2{"random fp16", 1, 10240, 320, 320 * 2, m.alloc((size_t)10240 * 320 * 2)};
        for (size_t i = 0; i < (size_t)10240 * 320; i++) f2.w.host<__fp16>()[i] = (__fp16)(0.01f * (int)(i % 89) - 0.44f);
        bench_dense(m, f2, {2048});
        bench_mps(m, 10240, 320, {2048});
        bench_moe(m, moe_load(m, g, 3), {512, 2048, 8192});
        return 0;
    }
    int failed = 0, checked = 0;
    for (auto &d : ds) { checked++; failed += !check_dense(m, d, 133, false, 1.f, 2e-3); }
    for (const char *n : {"blk.3.ffn_router.weight", "blk.0.ssm_ab.weight", "blk.3.hc_attn_inject.weight", "blk.0.hc_attn_down.weight"}) {
        checked++; failed += !check_dense(m, dense(m, g, n), 133, false, 1.f, 2e-3);
    }
    checked++; failed += !check_dense(m, dense(m, g, "blk.0.ssm_ab.weight"), 133, true, 0.5f, 2e-3);
    checked++; failed += !check_dense(m, ds[0], 70, true, 0.5f, 2e-3);
    std::map<int, bool> types;
    for (auto &d : ds) types[d.type] = true;
    for (int t : {8, 14}) if (!types[t]) { fprintf(stderr, "FAIL: no dense tensor of type %d\n", t); failed++; }
    if (mode != "dense") { checked++; failed += !check_moe(m, moe_load(m, g, 3), 1000, 5e-3); }
    printf("%d of %d checks failed\n", failed, checked);
    return failed ? 1 : 0;
}
