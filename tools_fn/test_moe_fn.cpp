// MoE checks on random experts: 16 experts (8 hot in VRAM as Q5_0, 8 cold in system RAM as IQ4_XS / IQ4_NL), routing
// over them, against a float reference built from ggml's dequantization.
//   decode: route_dec + hot GEMVs + the CPU workers (ggml kernels) + wait + sum, T = 3
//   prefill: grouped WMMA GEMMs on the hot tier and on cold experts staged into VRAM (DMA + relayout), T = 64
#include "../engine_fn/kernels.h"
#include "../engine_fn/cpu_moe.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

struct Tier { int gt_gu, gt_dn, t_gu, t_dn; std::vector<uint8_t> gate, up, down; std::vector<float> fg, fu, fd; };

static void make_tier(Tier &t, int n, std::mt19937 &rng) {
    std::normal_distribution<float> nd(0.f, 0.03f);
    auto q = [&](int gt, int rows, int cols, std::vector<uint8_t> &raw, std::vector<float> &deq) {
        std::vector<float> w((size_t)rows * cols), im(cols, 1.0f);
        for (auto &x : w) x = nd(rng);
        const size_t rs = ggml_row_size((ggml_type)gt, cols);
        raw.resize(rs * rows);
        ggml_quantize_chunk((ggml_type)gt, w.data(), raw.data(), 0, rows, cols, im.data());
        deq.resize(w.size());
        for (int r = 0; r < rows; r++) ggml_get_type_traits((ggml_type)gt)->to_float(raw.data() + rs * r, deq.data() + (size_t)r * cols, cols);
    };
    q(t.gt_gu, n * FE, D, t.gate, t.fg); q(t.gt_gu, n * FE, D, t.up, t.fu); q(t.gt_dn, n * D, FE, t.down, t.fd);
}

static void expert_ref(const Tier &t, int slot, const float *x, double *y, double w) {   // y += w * down(silu(g)*u)
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
    const int NH_ = 8, NC_ = 8;
    Tier hot{GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, Q5_0, Q5_0}, cold{GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, IQ4_XS, IQ4_NL};
    make_tier(hot, NH_, rng); make_tier(cold, NC_, rng);
    // slot map: experts 0..15, even hot, odd cold; the rest unused (-inf logits)
    std::vector<int> slot_of(NE, 0);
    for (int e = 0; e < 16; e++) slot_of[e] = e % 2 == 0 ? e / 2 : -1 - e / 2;
    Arena a; a.init((size_t)1 << 30);
    QTensor hg = make_qtensor(a, Q5_0, NH_ * FE, D), hu = make_qtensor(a, Q5_0, NH_ * FE, D), hd = make_qtensor(a, Q5_0, NH_ * D, FE);
    auto upload = [&](const QTensor &t, const std::vector<uint8_t> &raw) {
        uint8_t *d = a.get(raw.size());
        CK(hipMemcpy(d, raw.data(), raw.size(), hipMemcpyHostToDevice));
        relayout(t, d, st);
    };
    upload(hg, hot.gate); upload(hu, hot.up); upload(hd, hot.down);
    int *dslot = a.get<int>(NE); CK(hipMemcpy(dslot, slot_of.data(), NE * 4, hipMemcpyHostToDevice));
    // cold tier in host memory for the CPU (layer 0)
    MoeShared *sh; CK(hipHostMalloc((void **)&sh, sizeof(MoeShared), hipHostMallocCoherent | hipHostMallocMapped));
    memset(sh, 0, sizeof(MoeShared));
    MoeShared *dsh; CK(hipHostGetDevicePointer((void **)&dsh, sh, 0));
    std::vector<ColdLayer> cl(1);
    cl[0].gate = cold.gate.data(); cl[0].up = cold.up.data(); cl[0].down = cold.down.data();
    cl[0].type_gu = GGML_TYPE_IQ4_XS; cl[0].type_dn = GGML_TYPE_IQ4_NL;
    cl[0].row_gu = ggml_row_size(GGML_TYPE_IQ4_XS, D); cl[0].row_dn = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
    CpuMoe cpu; cpu.start(sh, cl, 8); cpu.arm();
    int fails = 0;

    // ---- decode, T = 3 ----
    {
        const int T = 3;
        std::vector<float> x((size_t)T * D), lg((size_t)T * NE, -INFINITY);
        for (auto &v : x) v = nd(rng);
        for (int t = 0; t < T; t++) for (int e = 0; e < 16; e++) lg[(size_t)t * NE + e] = nd(rng);
        float *dx = a.get<float>(x.size()), *dlg = a.get<float>(lg.size()), *dwts = a.get<float>(80), *dy = a.get<float>((size_t)80 * D);
        float *dsh_out = a.get<float>((size_t)T * D), *dsg = a.get<float>(T), *dout = a.get<float>((size_t)T * D);
        int *dids = a.get<int>(80), *dhot = a.get<int>(80);
        QAct xq{a.get<int8_t>(x.size()), a.get<float>(x.size() / 32)}, hq{a.get<int8_t>(80 * FE), a.get<float>(80 * FE / 32)};
        CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dlg, lg.data(), lg.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemset(dsh_out, 0, (size_t)T * D * 4)); CK(hipMemset(dsg, 0, T * 4));
        quant_act(dx, (int64_t)x.size(), xq, st);
        route_dec(dlg, T, dslot, dids, dwts, dhot, dsh, dx, 0, 1, st);
        moe_experts_dec(hg, hu, hd, dhot, T, xq, hq, dwts, dy, st);
        moe_wait(&dsh->done, 1, st);
        moe_sum(dy, dhot, dsh_out, dsg, dsh->y, T, dout, st);
        std::vector<float> out((size_t)T * D); std::vector<int> ids(T * 10); std::vector<float> w(T * 10);
        CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(ids.data(), dids, ids.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(w.data(), dwts, w.size() * 4, hipMemcpyDeviceToHost));
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
                const Tier &tr = slot_of[e] >= 0 ? hot : cold;
                expert_ref(tr, slot_of[e] >= 0 ? slot_of[e] : -1 - slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], wk);
            }
        }
        // split: the hot part (GPU, per pick) and the cold part (CPU answer in host memory)
        std::vector<float> yh((size_t)80 * D); std::vector<int> hs(80);
        CK(hipMemcpy(yh.data(), dy, yh.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(hs.data(), dhot, 80 * 4, hipMemcpyDeviceToHost));
        std::vector<float> gh((size_t)T * D, 0), gc((size_t)T * D, 0);
        std::vector<double> rh((size_t)T * D, 0), rc((size_t)T * D, 0);
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < 10; k++) {
                const int j = t * 10 + k, e = ids[j];
                if (hs[j] >= 0) for (int i = 0; i < D; i++) gh[(size_t)t * D + i] += yh[(size_t)j * D + i];
                const Tier &tr = slot_of[e] >= 0 ? hot : cold;
                expert_ref(tr, slot_of[e] >= 0 ? slot_of[e] : -1 - slot_of[e], &x[(size_t)t * D], slot_of[e] >= 0 ? &rh[(size_t)t * D] : &rc[(size_t)t * D], w[j]);
            }
            for (int i = 0; i < D; i++) gc[(size_t)t * D + i] = sh->y[(size_t)t * D + i];
        }
        printf("   hot part rel diff %.2e, cold part rel diff %.2e\n", rel(gh, rh), rel(gc, rc));
        const double r = rel(out, ref);
        printf("decode MoE (T=3, hot GPU + cold CPU): rel diff %.2e %s\n", r, r < 2e-2 ? "ok" : "FAIL");
        fails += r >= 2e-2;
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
        // host-side sort by tier and slot, as Engine::moe_prefill does
        std::vector<int> key(16), cnt(16, 0), off(16), order(16);
        for (int e = 0; e < 16; e++) { key[e] = slot_of[e] >= 0 ? slot_of[e] : NE + (-1 - slot_of[e]); order[e] = e; }
        std::sort(order.begin(), order.end(), [&](int p, int q) { return key[p] < key[q]; });
        for (int j = 0; j < N; j++) cnt[ids[j]]++;
        for (int i = 0, o = 0; i < 16; i++) { off[order[i]] = o; o += cnt[order[i]]; }
        std::vector<int> cur = off, tok(N), pos(N); std::vector<float> ew(N);
        for (int j = 0; j < N; j++) { const int e = ids[j], p = cur[e]++; tok[p] = j / 10; ew[p] = w[j]; pos[j] = p; }
        std::vector<MoeWork> hw, cw;
        for (int i = 0; i < 16; i++) {
            const int e = order[i];
            if (!cnt[e]) continue;
            if (slot_of[e] >= 0) hw.push_back({slot_of[e], off[e], cnt[e], 0}); else cw.push_back({-1 - slot_of[e], off[e], cnt[e], 0});
        }
        half *dX = a.get<half>(x.size()), *dXs = a.get<half>((size_t)N * D), *dH = a.get<half>((size_t)N * FE), *dY = a.get<half>((size_t)N * D);
        int *dtok = a.get<int>(N), *dpos = a.get<int>(N); float *dew = a.get<float>(N), *dshr = a.get<float>((size_t)T * D), *dsg = a.get<float>(T), *dout = a.get<float>((size_t)T * D);
        MoeWork *dhw = a.get<MoeWork>(64), *dcw = a.get<MoeWork>(64);
        CK(hipMemcpy(dX, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dtok, tok.data(), N * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dpos, pos.data(), N * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dew, ew.data(), N * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dhw, hw.data(), hw.size() * sizeof(MoeWork), hipMemcpyHostToDevice));
        CK(hipMemcpy(dcw, cw.data(), cw.size() * sizeof(MoeWork), hipMemcpyHostToDevice));
        CK(hipMemset(dshr, 0, (size_t)T * D * 4)); CK(hipMemset(dsg, 0, T * 4));
        gather_rows(dX, dtok, N, dXs, st);
        QTensor none{};
        moe_gemm_gateup(hg, hu, none, none, dhw, (int)hw.size(), dXs, dH, st);
        moe_gemm_down(hd, none, dhw, (int)hw.size(), dH, dew, dY, st);
        // cold: DMA the AoS bytes from registered host memory, relayout to VRAM, same kernels
        for (auto *v : {&cold.gate, &cold.up, &cold.down}) CK(hipHostRegister(v->data(), v->size(), hipHostRegisterDefault));
        uint8_t *rg = a.get(cold.gate.size()), *ru = a.get(cold.up.size()), *rd = a.get(cold.down.size());
        CK(hipMemcpyAsync(rg, cold.gate.data(), cold.gate.size(), hipMemcpyHostToDevice, cst));
        CK(hipMemcpyAsync(ru, cold.up.data(), cold.up.size(), hipMemcpyHostToDevice, cst));
        CK(hipMemcpyAsync(rd, cold.down.data(), cold.down.size(), hipMemcpyHostToDevice, cst));
        CK(hipStreamSynchronize(cst));
        QTensor cg = make_qtensor(a, IQ4_XS, NC_ * FE, D), cu = make_qtensor(a, IQ4_XS, NC_ * FE, D), cd = make_qtensor(a, IQ4_NL, NC_ * D, FE);
        relayout(cg, rg, st); relayout(cu, ru, st); relayout(cd, rd, st);
        moe_gemm_gateup(cg, cu, none, none, dcw, (int)cw.size(), dXs, dH, st);
        moe_gemm_down(cd, none, dcw, (int)cw.size(), dH, dew, dY, st);
        moe_combine(dY, dpos, dshr, dsg, T, dout, st);
        std::vector<float> out((size_t)T * D);
        CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
        std::vector<double> ref((size_t)T * D, 0.0);
        for (int t = 0; t < T; t++)
            for (int k = 0; k < 10; k++) {
                const int e = ids[t * 10 + k];
                const Tier &tr = slot_of[e] >= 0 ? hot : cold;
                expert_ref(tr, slot_of[e] >= 0 ? slot_of[e] : -1 - slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], w[t * 10 + k]);
            }
        const double r = rel(out, ref);
        printf("prefill MoE (T=64, grouped WMMA, hot + staged cold): rel diff %.2e %s\n", r, r < 1e-2 ? "ok" : "FAIL");
        fails += r >= 1e-2;
    }
    cpu.disarm(); cpu.stop();
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
