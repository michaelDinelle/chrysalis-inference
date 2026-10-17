// MoE checks on random experts: 16 experts, 8 hot in VRAM and 8 cold in system RAM, routing over them, against a float
// reference built from ggml's dequantization. The cold experts come in two gate / up format groups as in kiln-fn.gguf
// (5 Q3_K, then 3 Q2_K, down IQ4_NL for all); the hot tier runs once per VRAM format (IQ4_XS as in the file, Q3_K as the
// MTP experts, Q2_K and Q5_0 for completeness).
//   decode: route_dec + hot GEMVs + the CPU workers (ggml kernels) + wait + sum, T = 3; then half of the cold picks read
//           by the GPU straight from host memory
//   prefill: grouped WMMA GEMMs on the hot tier and on cold experts staged into VRAM per format group (DMA + relayout)
#include "../engine_fn/kernels.h"
#include "../engine_fn/cpu_moe.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

struct Mat { int gt = -1; std::vector<uint8_t> raw; std::vector<float> deq; };   // a quantized [rows][cols] matrix

static Mat quant_mat(int gt, int rows, int cols, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 0.03f);
    Mat m; m.gt = gt;
    std::vector<float> w((size_t)rows * cols), im(cols, 1.0f);
    for (auto &x : w) x = nd(rng);
    const size_t rs = ggml_row_size((ggml_type)gt, cols);
    m.raw.resize(rs * rows);
    ggml_quantize_chunk((ggml_type)gt, w.data(), m.raw.data(), 0, rows, cols, im.data());
    m.deq.resize(w.size());
    for (int r = 0; r < rows; r++) ggml_get_type_traits((ggml_type)gt)->to_float(m.raw.data() + rs * r, m.deq.data() + (size_t)r * cols, cols);
    return m;
}

// n experts: gate / up in two format groups (slots below n0, the rest), down one matrix; float copies by slot
struct Experts {
    int n = 0, n0 = 0;
    Mat g[2], u[2], d;
    std::vector<float> fg, fu, fd;
};
static Experts make_experts(int n, int n0, int t0, int t1, int tdn, std::mt19937 &rng) {
    Experts x; x.n = n; x.n0 = n0;
    x.g[0] = quant_mat(t0, n0 * FE, D, rng); x.u[0] = quant_mat(t0, n0 * FE, D, rng);
    if (n > n0) { x.g[1] = quant_mat(t1, (n - n0) * FE, D, rng); x.u[1] = quant_mat(t1, (n - n0) * FE, D, rng); }
    x.d = quant_mat(tdn, n * D, FE, rng);
    for (int k = 0; k < 2; k++) {
        x.fg.insert(x.fg.end(), x.g[k].deq.begin(), x.g[k].deq.end());
        x.fu.insert(x.fu.end(), x.u[k].deq.begin(), x.u[k].deq.end());
    }
    x.fd = x.d.deq;
    return x;
}

static void expert_ref(const Experts &t, int slot, const float *x, double *y, double w) {   // y += w * down(silu(g)*u)
    std::vector<double> h(FE);
    for (int r = 0; r < FE; r++) {
        double g = 0, u = 0;
        for (int k = 0; k < D; k++) { g += (double)t.fg[((size_t)slot * FE + r) * D + k] * x[k]; u += (double)t.fu[((size_t)slot * FE + r) * D + k] * x[k]; }
        h[r] = g / (1 + exp(-g)) * u;
    }
    for (int r = 0; r < D; r++) {
        double s = 0;
        for (int k = 0; k < FE; k++) s += (double)t.fd[((size_t)slot * D + r) * FE + k] * h[k];
        y[r] += w * s;
    }
}

static double rel(const std::vector<float> &a, const std::vector<double> &b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); i++) { n += (a[i] - b[i]) * (a[i] - b[i]); d += b[i] * b[i]; }
    return sqrt(n / d);
}

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    hipStream_t st, cst; CK(hipStreamCreate(&st)); CK(hipStreamCreate(&cst));
    const int NH_ = 8, NC_ = 8, NC0 = 5;
    Experts cold = make_experts(NC_, NC0, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_IQ4_NL, rng);
    // slot map: experts 0..15, even hot, odd cold; the rest unused (-inf logits)
    std::vector<int> slot_of(NE, 0);
    for (int e = 0; e < 16; e++) slot_of[e] = e % 2 == 0 ? e / 2 : -1 - e / 2;
    Arena a; a.init((size_t)2 << 30);
    int *dslot = a.get<int>(NE); CK(hipMemcpy(dslot, slot_of.data(), NE * 4, hipMemcpyHostToDevice));
    // cold tier in host memory for the CPU (layer 0) and, registered, for the GPU
    std::vector<uint8_t> *cm[5] = {&cold.g[0].raw, &cold.u[0].raw, &cold.g[1].raw, &cold.u[1].raw, &cold.d.raw};
    uint8_t *dcm[5];
    for (int k = 0; k < 5; k++) {
        CK(hipHostRegister(cm[k]->data(), cm[k]->size(), hipHostRegisterMapped));
        CK(hipHostGetDevicePointer((void **)&dcm[k], cm[k]->data(), 0));
    }
    std::vector<ColdLayer> cl(1);
    ColdDec cd;
    for (int g = 0; g < 2; g++) {
        cl[0].gate[g] = cm[2 * g]->data(); cl[0].up[g] = cm[2 * g + 1]->data();
        cl[0].type_gu[g] = cold.g[g].gt; cl[0].row_gu[g] = ggml_row_size((ggml_type)cold.g[g].gt, D);
        cd.gate[g] = dcm[2 * g]; cd.up[g] = dcm[2 * g + 1]; cd.type[g] = cold.g[g].gt;
    }
    cl[0].down = cold.d.raw.data(); cl[0].type_dn = GGML_TYPE_IQ4_NL; cl[0].row_dn = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
    cl[0].n0 = cd.n0 = NC0; cl[0].n = NC_;
    cd.down = dcm[4];
    MoeShared *sh; CK(hipHostMalloc((void **)&sh, sizeof(MoeShared), hipHostMallocCoherent | hipHostMallocMapped));
    memset(sh, 0, sizeof(MoeShared));
    MoeShared *dsh; CK(hipHostGetDevicePointer((void **)&dsh, sh, 0));
    CpuMoe cpu; cpu.start(sh, nullptr, cl, 8); cpu.arm();
    int fails = 0;
    uint32_t gen = 0;

    // ---- fused router + routing against router_dec + route_dec: random fp16 router, T = 1..4 ----
    {
        std::vector<half> wr((size_t)NE * D); std::vector<float> wsg(D), x((size_t)4 * D);
        std::normal_distribution<float> wd(0.f, 0.05f);
        for (auto &v : wr) v = __float2half(wd(rng));
        for (auto &v : wsg) v = wd(rng);
        for (auto &v : x) v = nd(rng);
        std::vector<int> so(NE);                           // half of the experts hot, the rest cold
        for (int e = 0; e < NE; e++) so[e] = e % 2 ? e / 2 : -1 - e / 2;
        half *dw = a.get<half>(wr.size()); float *dwsg = a.get<float>(D), *dx = a.get<float>(x.size());
        int *dso = a.get<int>(NE); unsigned *cnt = a.get<unsigned>(1);
        CK(hipMemcpy(dw, wr.data(), wr.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dwsg, wsg.data(), D * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dso, so.data(), NE * 4, hipMemcpyHostToDevice));
        CK(hipMemset(cnt, 0, 4));
        float *lg1 = a.get<float>(4 * NE), *lg2 = a.get<float>(4 * NE), *sg1 = a.get<float>(4), *sg2 = a.get<float>(4), *w1 = a.get<float>(40), *w2 = a.get<float>(40);
        int *id1 = a.get<int>(40), *id2 = a.get<int>(40), *h1 = a.get<int>(40), *h2 = a.get<int>(40);
        for (int T = 1; T <= 4; T++) {
            router_dec(dw, dwsg, dx, T, lg1, sg1, st);
            route_dec(lg1, T, dso, id1, w1, h1, nullptr, 0.35f, dx, 0, 0, st);
            router_route_dec(dw, dwsg, dx, T, lg2, sg2, dso, id2, w2, h2, nullptr, 0.35f, 0, 0, cnt, st);
            std::vector<int> a1(T * 10), a2(T * 10), b1(T * 10), b2(T * 10); std::vector<float> c1(T * 10), c2(T * 10), s1(T), s2(T);
            CK(hipMemcpy(a1.data(), id1, T * 40, hipMemcpyDeviceToHost)); CK(hipMemcpy(a2.data(), id2, T * 40, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b1.data(), h1, T * 40, hipMemcpyDeviceToHost)); CK(hipMemcpy(b2.data(), h2, T * 40, hipMemcpyDeviceToHost));
            CK(hipMemcpy(c1.data(), w1, T * 40, hipMemcpyDeviceToHost)); CK(hipMemcpy(c2.data(), w2, T * 40, hipMemcpyDeviceToHost));
            CK(hipMemcpy(s1.data(), sg1, T * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(s2.data(), sg2, T * 4, hipMemcpyDeviceToHost));
            double dwt = 0, dsg = 0; int bad = 0;
            for (int j = 0; j < T * 10; j++) { bad += a1[j] != a2[j] || b1[j] != b2[j]; dwt = std::max(dwt, (double)fabsf(c1[j] - c2[j])); }
            for (int t = 0; t < T; t++) dsg = std::max(dsg, (double)fabsf(s1[t] - s2[t]));
            const bool ok = bad == 0 && dwt < 1e-5 && dsg < 1e-5;
            printf("fused router+route T=%d: %d picks differ, max weight diff %.1e, max shared-gate diff %.1e %s\n", T, bad, dwt, dsg, ok ? "ok" : "FAIL");
            fails += !ok;
        }
    }

    // ---- the zero-copy cold decode kernels (GGUF blocks in host memory) against the VRAM GEMVs on SoA copies of the same
    //      experts, from the same int8 inputs: one token whose 8 picks are the 8 cold experts ----
    {
        const size_t mark = a.used;
        QTensor sg[2], su[2], sdn[2];
        const size_t dn_row = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
        for (int g = 0; g < 2; g++) {
            const int ng = g ? NC_ - NC0 : NC0;
            sg[g] = make_qtensor(a, cold.g[g].gt, ng * FE, D); su[g] = make_qtensor(a, cold.g[g].gt, ng * FE, D);
            sdn[g] = make_qtensor(a, IQ4_NL, ng * D, FE);
            for (auto [t, src, n] : {std::tuple{&sg[g], cold.g[g].raw.data(), cold.g[g].raw.size()}, std::tuple{&su[g], cold.u[g].raw.data(), cold.u[g].raw.size()},
                                     std::tuple{&sdn[g], cold.d.raw.data() + (g ? (size_t)NC0 * D * dn_row : 0), (size_t)ng * D * dn_row}}) {
                uint8_t *d = a.get(n);
                CK(hipMemcpy(d, src, n, hipMemcpyHostToDevice));
                relayout(*t, d, st);
            }
        }
        std::vector<float> x(D), ones(TOPK_E, 1.0f);
        for (auto &v : x) v = nd(rng);
        std::vector<int> hz(TOPK_E, -1), hs0(TOPK_E, -1), hs1(TOPK_E, -1);
        for (int j = 0; j < NC_; j++) { hz[j] = -2 - j; (j < NC0 ? hs0[j] : hs1[j]) = j < NC0 ? j : j - NC0; }
        float *dx = a.get<float>(D), *dw = a.get<float>(TOPK_E), *y1 = a.get<float>((size_t)TOPK_E * D), *y2 = a.get<float>((size_t)TOPK_E * D);
        int *dhz = a.get<int>(TOPK_E), *dh0 = a.get<int>(TOPK_E), *dh1 = a.get<int>(TOPK_E);
        QAct xq{a.get<int8_t>(D), a.get<float>(D / 32)}, hq1{a.get<int8_t>(TOPK_E * FE), a.get<float>(TOPK_E * FE / 32)}, hq2{a.get<int8_t>(TOPK_E * FE), a.get<float>(TOPK_E * FE / 32)};
        CK(hipMemcpy(dx, x.data(), D * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, ones.data(), TOPK_E * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dhz, hz.data(), TOPK_E * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dh0, hs0.data(), TOPK_E * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dh1, hs1.data(), TOPK_E * 4, hipMemcpyHostToDevice));
        quant_act(dx, D, xq, st);
        moe_cold_dec(cd, dhz, 1, xq, hq1, dw, y1, st);
        moe_experts_dec(sg[0], su[0], sdn[0], QTensor{}, QTensor{}, QTensor{}, dh0, 1, xq, hq2, dw, nullptr, y2, st);
        moe_experts_dec(sg[1], su[1], sdn[1], QTensor{}, QTensor{}, QTensor{}, dh1, 1, xq, hq2, dw, nullptr, y2, st);
        std::vector<float> o1((size_t)NC_ * D), o2((size_t)NC_ * D);
        CK(hipMemcpy(o1.data(), y1, o1.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(o2.data(), y2, o2.size() * 4, hipMemcpyDeviceToHost));
        for (int g = 0; g < 2; g++) {
            double n = 0, d = 0;
            for (int j = g ? NC0 : 0; j < (g ? NC_ : NC0); j++)
                for (int i = 0; i < D; i++) { const double e = o1[(size_t)j * D + i] - o2[(size_t)j * D + i]; n += e * e; d += (double)o2[(size_t)j * D + i] * o2[(size_t)j * D + i]; }
            const double r = sqrt(n / d);
            printf("zero-copy cold decode vs VRAM GEMVs, %s experts, same int8 inputs: rel diff %.2e %s\n", g ? "Q2_K" : "Q3_K", r, r < 1e-3 ? "ok" : "FAIL");
            fails += r >= 1e-3;
        }
        a.used = mark;
    }

    struct HotFmt { int gt_gu, gt_dn; const char *name; };
    for (const HotFmt hf : {HotFmt{GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "IQ4_XS"}, HotFmt{GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, "Q3_K"},
                            HotFmt{GGML_TYPE_Q2_K, GGML_TYPE_IQ4_NL, "Q2_K"}, HotFmt{GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, "Q5_0"}}) {
        const size_t mark = a.used;
        Experts hot = make_experts(NH_, NH_, hf.gt_gu, hf.gt_gu, hf.gt_dn, rng);
        QTensor hg = make_qtensor(a, hf.gt_gu, NH_ * FE, D), hu = make_qtensor(a, hf.gt_gu, NH_ * FE, D), hd = make_qtensor(a, hf.gt_dn, NH_ * D, FE);
        auto upload = [&](const QTensor &t, const std::vector<uint8_t> &raw) {
            uint8_t *d = a.get(raw.size());
            CK(hipMemcpy(d, raw.data(), raw.size(), hipMemcpyHostToDevice));
            relayout(t, d, st);
        };
        upload(hg, hot.g[0].raw); upload(hu, hot.u[0].raw); upload(hd, hot.d.raw);

        // ---- decode, T = 3: cold picks all on the CPU, then half of them read by the GPU from host memory ----
        for (int pass = 0; pass < 2; pass++) {
            const int T = 3;
            const float gfrac = pass ? 0.5f : 0.0f;
            std::vector<float> x((size_t)T * D), lg((size_t)T * NE, -INFINITY);
            for (auto &v : x) v = nd(rng);
            for (int t = 0; t < T; t++) for (int e = 0; e < 16; e++) lg[(size_t)t * NE + e] = nd(rng);
            float *dx = a.get<float>(x.size()), *dlg = a.get<float>(lg.size()), *dwts = a.get<float>(80), *dy = a.get<float>((size_t)80 * D);
            float *dsh_out = a.get<float>((size_t)T * D), *dsg = a.get<float>(T);
            int *dids = a.get<int>(80), *dhot = a.get<int>(80);
            QAct xq{a.get<int8_t>(x.size()), a.get<float>(x.size() / 32)}, hq{a.get<int8_t>(80 * FE), a.get<float>(80 * FE / 32)};
            CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dlg, lg.data(), lg.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemset(dsh_out, 0, (size_t)T * D * 4)); CK(hipMemset(dsg, 0, T * 4));
            quant_act(dx, (int64_t)x.size(), xq, st);
            gen++;
            route_dec(dlg, T, dslot, dids, dwts, dhot, dsh, gfrac, dx, 0, gen, st);
            moe_experts_dec(hg, hu, hd, QTensor{}, QTensor{}, QTensor{}, dhot, T, xq, hq, dwts, nullptr, dy, st);
            if (gfrac > 0) moe_cold_dec(cd, dhot, T, xq, hq, dwts, dy, st);
            // moe_finish: wait + sum + the HC update; inject 0 makes every stream's weight 1 and res starts at 0
            float *dres = a.get<float>((size_t)T * HC * D), *dinj = a.get<float>(T * HC);
            CK(hipMemset(dres, 0, (size_t)T * HC * D * 4)); CK(hipMemset(dinj, 0, T * HC * 4));
            moe_finish(&dsh->done, gen, dy, dhot, dsh_out, dsg, dsh->y, dinj, dres, T, st);
            std::vector<float> res4((size_t)T * HC * D), out((size_t)T * D); std::vector<int> ids(T * 10), hs(T * 10); std::vector<float> w(T * 10);
            CK(hipMemcpy(res4.data(), dres, res4.size() * 4, hipMemcpyDeviceToHost));
            for (int t = 0; t < T; t++) for (int i = 0; i < D; i++) out[(size_t)t * D + i] = res4[((size_t)t * HC + 3) * D + i];   // stream 3
            CK(hipMemcpy(ids.data(), dids, ids.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(w.data(), dwts, w.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(hs.data(), dhot, hs.size() * 4, hipMemcpyDeviceToHost));
            int n_gpu = 0, n_cpu = 0, n_g1 = 0;
            for (int v : hs) { n_gpu += v <= -2; n_cpu += v == -1; n_g1 += v <= -2 && -2 - v >= NC0; }
            std::vector<double> ref((size_t)T * D, 0.0);
            for (int t = 0; t < T; t++) {
                // reference routing: top 10 of the 16 logits, softmax over them
                std::vector<std::pair<float, int>> c;
                for (int e = 0; e < 16; e++) c.push_back({lg[(size_t)t * NE + e], e});
                std::sort(c.begin(), c.end(), [](auto &p, auto &q) { return p.first > q.first || (p.first == q.first && p.second < q.second); });
                double s = 0; for (int k = 0; k < 10; k++) s += exp(c[k].first - c[0].first);
                for (int k = 0; k < 10; k++) {
                    const int e = c[k].second; const double wk = exp(c[k].first - c[0].first) / s;
                    if (ids[t * 10 + k] != e) { printf("routing mismatch t%d k%d: %d vs %d\n", t, k, ids[t * 10 + k], e); fails++; }
                    if (slot_of[e] >= 0) expert_ref(hot, slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], wk);
                    else expert_ref(cold, -1 - slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], wk);
                }
            }
            const double r = rel(out, ref);
            printf("decode MoE, hot %-6s (T=3; cold Q3_K/Q2_K: %d on the GPU from host memory (%d Q2_K), %d on the CPU): rel diff %.2e %s\n",
                   hf.name, n_gpu, n_g1, n_cpu, r, r < 2e-2 ? "ok" : "FAIL");
            fails += r >= 2e-2 || (pass && n_gpu == 0);
        }
        // ---- prefill grouped GEMMs, T = 64 ----
        {
            const int T = 64, N = T * 10;
            std::vector<float> x((size_t)T * D);
            for (auto &v : x) v = nd(rng);
            std::vector<half> xh(x.size());
            for (size_t i = 0; i < x.size(); i++) { xh[i] = __float2half(x[i]); x[i] = __half2float(xh[i]); }
            // routing: random picks among the 16 experts (distinct per token), random weights summing to 1
            std::vector<int> ids(N); std::vector<float> w(N);
            for (int t = 0; t < T; t++) {
                std::vector<int> e(16); for (int i = 0; i < 16; i++) e[i] = i;
                std::shuffle(e.begin(), e.end(), rng);
                double s = 0; std::vector<double> r(10);
                for (int k = 0; k < 10; k++) { r[k] = std::abs(nd(rng)) + 0.1; s += r[k]; }
                for (int k = 0; k < 10; k++) { ids[t * 10 + k] = e[k]; w[t * 10 + k] = (float)(r[k] / s); }
            }
            // host-side sort by tier and slot, as Engine::moe_prefill does; cold work items per format group, slots relative
            std::vector<int> key(16), cnt(16, 0), off(16), order(16);
            for (int e = 0; e < 16; e++) { key[e] = slot_of[e] >= 0 ? slot_of[e] : NE + (-1 - slot_of[e]); order[e] = e; }
            std::sort(order.begin(), order.end(), [&](int p, int q) { return key[p] < key[q]; });
            for (int j = 0; j < N; j++) cnt[ids[j]]++;
            for (int i = 0, o = 0; i < 16; i++) { off[order[i]] = o; o += cnt[order[i]]; }
            std::vector<int> cur = off, tok(N), pos(N); std::vector<float> ew(N);
            for (int j = 0; j < N; j++) { const int e = ids[j], p = cur[e]++; tok[p] = j / 10; ew[p] = w[j]; pos[j] = p; }
            std::vector<MoeWork> hw, cw[2];
            for (int i = 0; i < 16; i++) {
                const int e = order[i];
                if (!cnt[e]) continue;
                if (slot_of[e] >= 0) { hw.push_back({slot_of[e], off[e], cnt[e], 0}); continue; }
                const int s = -1 - slot_of[e];
                if (s < NC0) cw[0].push_back({s, off[e], cnt[e], 0}); else cw[1].push_back({s - NC0, off[e], cnt[e], 0});
            }
            half *dX = a.get<half>(x.size()), *dXs = a.get<half>((size_t)N * D), *dH = a.get<half>((size_t)N * FE), *dY = a.get<half>((size_t)N * D);
            int *dtok = a.get<int>(N), *dpos = a.get<int>(N); float *dew = a.get<float>(N), *dshr = a.get<float>((size_t)T * D), *dsg = a.get<float>(T), *dout = a.get<float>((size_t)T * D);
            MoeWork *dhw = a.get<MoeWork>(64), *dcw[2] = {a.get<MoeWork>(64), a.get<MoeWork>(64)};
            CK(hipMemcpy(dX, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(dtok, tok.data(), N * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dpos, pos.data(), N * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(dew, ew.data(), N * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(dhw, hw.data(), hw.size() * sizeof(MoeWork), hipMemcpyHostToDevice));
            for (int g = 0; g < 2; g++) CK(hipMemcpy(dcw[g], cw[g].data(), cw[g].size() * sizeof(MoeWork), hipMemcpyHostToDevice));
            CK(hipMemset(dshr, 0, (size_t)T * D * 4)); CK(hipMemset(dsg, 0, T * 4));
            gather_rows(dX, dtok, N, dXs, st);
            QTensor none{};
            moe_gemm_gateup(hg, hu, none, none, dhw, (int)hw.size(), dXs, dH, st);
            moe_gemm_down(hd, none, dhw, (int)hw.size(), dH, dew, dY, st);
            // cold: DMA the AoS bytes of each format group from registered host memory, relayout to VRAM, same kernels
            uint8_t *rd = a.get(cold.d.raw.size());
            CK(hipMemcpyAsync(rd, cold.d.raw.data(), cold.d.raw.size(), hipMemcpyHostToDevice, cst));
            for (int g = 0; g < 2; g++) {
                const int ng = g ? NC_ - NC0 : NC0;
                uint8_t *rg = a.get(cold.g[g].raw.size()), *ru = a.get(cold.u[g].raw.size());
                CK(hipMemcpyAsync(rg, cold.g[g].raw.data(), cold.g[g].raw.size(), hipMemcpyHostToDevice, cst));
                CK(hipMemcpyAsync(ru, cold.u[g].raw.data(), cold.u[g].raw.size(), hipMemcpyHostToDevice, cst));
                CK(hipStreamSynchronize(cst));
                QTensor cg = make_qtensor(a, cold.g[g].gt, ng * FE, D), cu = make_qtensor(a, cold.g[g].gt, ng * FE, D);
                QTensor cdn = make_qtensor(a, IQ4_NL, ng * D, FE);
                relayout(cg, rg, st); relayout(cu, ru, st);
                relayout(cdn, rd + (g ? (size_t)NC0 * D * ggml_row_size(GGML_TYPE_IQ4_NL, FE) : 0), st);
                moe_gemm_gateup(cg, cu, none, none, dcw[g], (int)cw[g].size(), dXs, dH, st);
                moe_gemm_down(cdn, none, dcw[g], (int)cw[g].size(), dH, dew, dY, st);
            }
            moe_combine(dY, dpos, dshr, dsg, T, dout, st);
            std::vector<float> out((size_t)T * D);
            CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
            std::vector<double> ref((size_t)T * D, 0.0);
            for (int t = 0; t < T; t++)
                for (int k = 0; k < 10; k++) {
                    const int e = ids[t * 10 + k];
                    if (slot_of[e] >= 0) expert_ref(hot, slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], w[t * 10 + k]);
                    else expert_ref(cold, -1 - slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], w[t * 10 + k]);
                }
            const double r = rel(out, ref);
            printf("prefill MoE, hot %-6s (T=64, grouped WMMA, hot + staged cold Q3_K/Q2_K): rel diff %.2e %s\n", hf.name, r, r < 1e-2 ? "ok" : "FAIL");
            fails += r >= 1e-2;
        }
        CK(hipStreamSynchronize(st));
        a.used = mark;
    }
    cpu.disarm(); cpu.stop();
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
