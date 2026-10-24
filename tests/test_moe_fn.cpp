#include "../engine_fn/kernels.h"
#include "../engine_fn/cpu_moe.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

struct Mat { int gt = -1; std::vector<uint8_t> raw; std::vector<float> deq; };

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

static void expert_ref(const Experts &t, int slot, const float *x, double *y, double w) {
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
    const int NH_ = 8, NC_ = 8, NC0 = 5, NF = 2, NX = 16 + NF;
    Experts cold = make_experts(NC_, NC0, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_IQ4_NL, rng);
    Experts filed = make_experts(NF, NF, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, rng);
    std::vector<int> slot_of(NE, 0);
    for (int e = 0; e < 16; e++) slot_of[e] = e % 2 == 0 ? e / 2 : -1 - e / 2;
    for (int f = 0; f < NF; f++) slot_of[16 + f] = -1 - (NC_ + f);
    auto ref_of = [&](int e, const Experts &hot, const float *x, double *y, double w) {
        const int s = slot_of[e];
        if (s >= 0) expert_ref(hot, s, x, y, w);
        else if (-1 - s < NC_) expert_ref(cold, -1 - s, x, y, w);
        else expert_ref(filed, -1 - s - NC_, x, y, w);
    };
    Arena a; a.init((size_t)2 << 30);
    int *dslot = a.get<int>(NE); CK(hipMemcpy(dslot, slot_of.data(), NE * 4, hipMemcpyHostToDevice));
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
    cl[0].gate[2] = filed.g[0].raw.data(); cl[0].up[2] = filed.u[0].raw.data(); cl[0].down2 = filed.d.raw.data();
    cl[0].type_gu[2] = GGML_TYPE_IQ4_XS; cl[0].row_gu[2] = ggml_row_size(GGML_TYPE_IQ4_XS, D); cl[0].n2 = NF;
    cd.down = dcm[4];
    MoeShared *sh; CK(hipHostMalloc((void **)&sh, sizeof(MoeShared), hipHostMallocCoherent | hipHostMallocMapped));
    memset(sh, 0, sizeof(MoeShared));
    MoeShared *dsh; CK(hipHostGetDevicePointer((void **)&dsh, sh, 0));
    CpuMoe cpu; cpu.start(sh, nullptr, cl, 8); cpu.arm();
    int fails = 0;
    uint32_t gen = 0;
    std::uniform_real_distribution<float> gate(0.2f, 0.8f);
    auto upload = [&](const QTensor &t, const std::vector<uint8_t> &raw) {
        uint8_t *d = a.get(raw.size());
        CK(hipMemcpy(d, raw.data(), raw.size(), hipMemcpyHostToDevice));
        relayout(t, d, st);
    };
    static_assert(FS == FE, "expert_ref computes the shared expert with the routed experts' width");
    Experts shx = make_experts(1, 1, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, rng);
    QTensor sgate = make_qtensor(a, Q6_K, FS, D), sup = make_qtensor(a, Q6_K, FS, D), sdown = make_qtensor(a, Q8_0, D, FS);
    upload(sgate, shx.g[0].raw); upload(sup, shx.u[0].raw); upload(sdown, shx.d.raw);

    {
        std::vector<half> wr((size_t)NE * D); std::vector<float> wsg(D), x((size_t)4 * D);
        std::normal_distribution<float> wd(0.f, 0.05f);
        for (auto &v : wr) v = __float2half(wd(rng));
        for (auto &v : wsg) v = wd(rng);
        for (auto &v : x) v = nd(rng);
        std::vector<int> so(NE);
        for (int e = 0; e < NE; e++) so[e] = e % 2 ? e / 2 : -1 - e / 2;
        half *dw = a.get<half>(wr.size()); float *dwsg = a.get<float>(D), *dx = a.get<float>(x.size());
        int *dso = a.get<int>(NE); unsigned *cnt = a.get<unsigned>(1);
        CK(hipMemcpy(dw, wr.data(), wr.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dwsg, wsg.data(), D * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dso, so.data(), NE * 4, hipMemcpyHostToDevice));
        CK(hipMemset(cnt, 0, 4));
        float *lg1 = a.get<float>(4 * NE), *lg2 = a.get<float>(4 * NE), *sg1 = a.get<float>(4), *sg2 = a.get<float>(4), *w1 = a.get<float>(40), *w2 = a.get<float>(40);
        int *id1 = a.get<int>(40), *id2 = a.get<int>(40), *h1 = a.get<int>(40), *h2 = a.get<int>(40);
        MoeShared *hs1, *hs2, *ds1, *ds2;
        for (MoeShared **h : {&hs1, &hs2}) { CK(hipHostMalloc((void **)h, sizeof(MoeShared), hipHostMallocCoherent | hipHostMallocMapped)); memset(*h, 0, sizeof(MoeShared)); }
        CK(hipHostGetDevicePointer((void **)&ds1, hs1, 0)); CK(hipHostGetDevicePointer((void **)&ds2, hs2, 0));
        for (int T = 1; T <= 4; T++) {
            router_dec(dw, dwsg, dx, T, lg1, sg1, st);
            route_dec(lg1, T, dso, NE / 4, id1, w1, h1, ds1, 0.35f, dx, 7, 100 + T, st);
            router_route_dec(dw, dwsg, dx, T, lg2, sg2, dso, NE / 4, id2, w2, h2, ds2, 0.35f, 7, 100 + T, cnt, st);
            CK(hipStreamSynchronize(st));
            int hand = hs1->go != hs2->go || hs1->go != (uint32_t)(100 + T) || hs1->layer != hs2->layer || hs1->T != hs2->T;
            for (int j = 0; j < T * 10; j++) hand += hs1->slot[j] != hs2->slot[j] || fabsf(hs1->w[j] - hs2->w[j]) > 1e-5f;
            for (int i = 0; i < T * D; i++) hand += hs1->x[i] != hs2->x[i];
            if (hand) { printf("fused router+route T=%d: %d differences in what the CPU is handed FAIL\n", T, hand); fails++; }
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
        moe_experts_dec(sg[0], su[0], sdn[0], nullptr, QTensor{}, QTensor{}, QTensor{}, dh0, 1, xq, hq2, dw, nullptr, y2, st);
        moe_experts_dec(sg[1], su[1], sdn[1], nullptr, QTensor{}, QTensor{}, QTensor{}, dh1, 1, xq, hq2, dw, nullptr, y2, st);
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
        upload(hg, hot.g[0].raw); upload(hu, hot.u[0].raw); upload(hd, hot.d.raw);

        for (int pass = 0; pass < 2; pass++) {
            const int T = 3;
            const float gfrac = pass ? 0.5f : 0.0f;
            std::vector<float> x((size_t)T * D), lg((size_t)T * NE, -INFINITY);
            for (auto &v : x) v = nd(rng);
            for (int t = 0; t < T; t++) for (int e = 0; e < NX; e++) lg[(size_t)t * NE + e] = nd(rng);
            float *dx = a.get<float>(x.size()), *dlg = a.get<float>(lg.size()), *dwts = a.get<float>(80), *dy = a.get<float>((size_t)80 * D);
            std::vector<float> sg(T);
            for (auto &v : sg) v = gate(rng);
            float *dsg = a.get<float>(T);
            int *dids = a.get<int>(80), *dhot = a.get<int>(80);
            QAct xq{a.get<int8_t>(x.size()), a.get<float>(x.size() / 32)}, hq{a.get<int8_t>(80 * FE), a.get<float>(80 * FE / 32)};
            CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dlg, lg.data(), lg.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(dsg, sg.data(), T * 4, hipMemcpyHostToDevice));
            quant_act(dx, (int64_t)x.size(), xq, st);
            gen++;
            route_dec(dlg, T, dslot, NC_, dids, dwts, dhot, dsh, gfrac, dx, 0, gen, st);
            {
                std::vector<int> hs0(T * 10), ids0(T * 10);
                CK(hipMemcpy(hs0.data(), dhot, hs0.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(ids0.data(), dids, ids0.size() * 4, hipMemcpyDeviceToHost));
                int bad = 0;
                for (int j = 0; j < T * 10; j++) bad += hs0[j] <= -2 && -2 - hs0[j] >= NC_;
                if (bad) { printf("decode routing hands %d picks of group 2 to the GPU FAIL\n", bad); fails++; CK(hipDeviceSynchronize()); continue; }
            }
            moe_dec(hg, hu, hd, nullptr, sgate, sup, sdown, gfrac > 0 ? &cd : nullptr, dhot, T, xq, hq, dwts, dsg, dy, st);
            float *dres = a.get<float>((size_t)T * HC * D), *dinj = a.get<float>(T * HC);
            CK(hipMemset(dres, 0, (size_t)T * HC * D * 4)); CK(hipMemset(dinj, 0, T * HC * 4));
            moe_finish(&dsh->done, gen, dy, dhot, dy + (size_t)T * TOPK_E * D, nullptr, dsh->y, dinj, dres, T, st);
            std::vector<float> res4((size_t)T * HC * D), out((size_t)T * D); std::vector<int> ids(T * 10), hs(T * 10); std::vector<float> w(T * 10);
            CK(hipMemcpy(res4.data(), dres, res4.size() * 4, hipMemcpyDeviceToHost));
            for (int t = 0; t < T; t++) for (int i = 0; i < D; i++) out[(size_t)t * D + i] = res4[((size_t)t * HC + 3) * D + i];
            CK(hipMemcpy(ids.data(), dids, ids.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(w.data(), dwts, w.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(hs.data(), dhot, hs.size() * 4, hipMemcpyDeviceToHost));
            int n_gpu = 0, n_cpu = 0, n_g1 = 0, n_g2 = 0, n_g2_gpu = 0;
            for (int j = 0; j < T * 10; j++) {
                const int v = hs[j];
                n_gpu += v <= -2; n_cpu += v == -1; n_g1 += v <= -2 && -2 - v >= NC0;
                n_g2 += ids[j] >= 16; n_g2_gpu += ids[j] >= 16 && v != -1;
            }
            std::vector<double> ref((size_t)T * D, 0.0);
            for (int t = 0; t < T; t++) {
                std::vector<std::pair<float, int>> c;
                for (int e = 0; e < NX; e++) c.push_back({lg[(size_t)t * NE + e], e});
                std::sort(c.begin(), c.end(), [](auto &p, auto &q) { return p.first > q.first || (p.first == q.first && p.second < q.second); });
                double s = 0; for (int k = 0; k < 10; k++) s += exp(c[k].first - c[0].first);
                for (int k = 0; k < 10; k++) {
                    const int e = c[k].second; const double wk = exp(c[k].first - c[0].first) / s;
                    if (ids[t * 10 + k] != e) { printf("routing mismatch t%d k%d: %d vs %d\n", t, k, ids[t * 10 + k], e); fails++; }
                    ref_of(e, hot, &x[(size_t)t * D], &ref[(size_t)t * D], wk);
                }
                expert_ref(shx, 0, &x[(size_t)t * D], &ref[(size_t)t * D], sg[t]);
            }
            const double r = rel(out, ref);
            const bool ok = r < 2e-2 && (!pass || n_gpu > 0) && n_g2 > 0 && n_g2_gpu == 0;
            printf("decode MoE + shared, hot %-6s (T=3; cold Q3_K/Q2_K: %d on the GPU from host memory (%d Q2_K), %d on the CPU, "
                   "%d of them the file's IQ4_XS): rel diff %.2e %s\n", hf.name, n_gpu, n_g1, n_cpu, n_g2, r, ok ? "ok" : "FAIL");
            fails += !ok;
        }
        {
            const int T = 64, N = T * 10;
            std::vector<float> x((size_t)T * D);
            for (auto &v : x) v = nd(rng);
            std::vector<half> xh(x.size());
            for (size_t i = 0; i < x.size(); i++) { xh[i] = __float2half(x[i]); x[i] = __half2float(xh[i]); }
            std::vector<int> ids(N); std::vector<float> w(N);
            for (int t = 0; t < T; t++) {
                std::vector<int> e(16); for (int i = 0; i < 16; i++) e[i] = i;
                std::shuffle(e.begin(), e.end(), rng);
                double s = 0; std::vector<double> r(10);
                for (int k = 0; k < 10; k++) { r[k] = std::abs(nd(rng)) + 0.1; s += r[k]; }
                for (int k = 0; k < 10; k++) { ids[t * 10 + k] = e[k]; w[t * 10 + k] = (float)(r[k] / s); }
            }
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
            std::vector<float> shr((size_t)T * D), sgv(T);
            for (auto &v : shr) v = 0.05f * nd(rng);
            for (auto &v : sgv) v = gate(rng);
            CK(hipMemcpy(dshr, shr.data(), shr.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dsg, sgv.data(), T * 4, hipMemcpyHostToDevice));
            gather_rows(dX, dtok, N, dXs, st);
            QTensor none{};
            moe_gemm_gateup(hg, hu, none, none, dhw, (int)hw.size(), dXs, dH, st);
            moe_gemm_down(hd, none, dhw, (int)hw.size(), dH, dew, dY, st);
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
            moe_shared(dshr, dsg, T, dout, st);
            int *dbtok = a.get<int>(2 * T);
            for (int bt = 0; bt < 2; bt++) {
                const int p0 = bt ? N / 2 : 0, p1 = bt ? N : N / 2;
                std::vector<int> toks, seen(T, 0);
                for (int p = p0; p < p1; p++) if (!seen[tok[p]]) { seen[tok[p]] = 1; toks.push_back(tok[p]); }
                CK(hipMemcpy(dbtok + bt * T, toks.data(), toks.size() * 4, hipMemcpyHostToDevice));
                moe_accum(dY + (size_t)p0 * D, dpos, p0, p1, dbtok + bt * T, (int)toks.size(), dout, st);
            }
            std::vector<float> out((size_t)T * D);
            CK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
            std::vector<double> ref((size_t)T * D, 0.0);
            for (int t = 0; t < T; t++)
                for (int k = 0; k < 10; k++) {
                    const int e = ids[t * 10 + k];
                    if (slot_of[e] >= 0) expert_ref(hot, slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], w[t * 10 + k]);
                    else expert_ref(cold, -1 - slot_of[e], &x[(size_t)t * D], &ref[(size_t)t * D], w[t * 10 + k]);
                }
            for (int t = 0; t < T; t++)
                for (int i = 0; i < D; i++) ref[(size_t)t * D + i] += (double)sgv[t] * shr[(size_t)t * D + i];
            const double r = rel(out, ref);
            printf("prefill MoE, hot %-6s (T=64, grouped WMMA, hot + staged cold Q3_K/Q2_K, 2 batches): rel diff %.2e %s\n", hf.name, r, r < 1e-2 ? "ok" : "FAIL");
            fails += r >= 1e-2;
        }
        CK(hipStreamSynchronize(st));
        a.used = mark;
    }
    {
        const size_t mark = a.used;
        const int NP = 6;
        struct Src { const Experts *x; int g, s, slot; };
        const Src src[NP] = {{&filed, 0, 0, 3}, {&filed, 0, 1, 0}, {&cold, 0, 1, 1}, {&cold, 0, 2, 5}, {&cold, 1, 0, 4}, {&cold, 1, 2, 2}};
        QTensor pg = make_qtensor(a, IQ4_XS, NP * FE, D), pu = make_qtensor(a, IQ4_XS, NP * FE, D), pd = make_qtensor(a, IQ4_NL, NP * D, FE);
        int *dso = a.get<int>(NE); uint8_t *dfmt = a.get<uint8_t>(NP);
        CK(hipMemset(dso, 0xff, NE * 4)); CK(hipMemset(dfmt, 0, NP));
        PlaceArgs pa;
        pa.gate = pg; pa.up = pu; pa.down = pd; pa.slot_of = dso; pa.fmt = dfmt;
        const size_t dn_row = ggml_row_size(GGML_TYPE_IQ4_NL, FE);
        for (int i = 0; i < NP; i++) {
            const Src &q = src[i];
            const int gt = q.x->g[q.g].gt;
            const size_t row = ggml_row_size((ggml_type)gt, D), gsz = (size_t)FE * row, dsz = (size_t)D * dn_row;
            const int ds = q.g ? q.x->n0 + q.s : q.s;
            std::vector<uint8_t> blob(2 * gsz + dsz);
            memcpy(blob.data(), q.x->g[q.g].raw.data() + q.s * gsz, gsz);
            memcpy(blob.data() + gsz, q.x->u[q.g].raw.data() + q.s * gsz, gsz);
            memcpy(blob.data() + 2 * gsz, q.x->d.raw.data() + ds * dsz, dsz);
            uint8_t *d = a.get(blob.size());
            CK(hipMemcpy(d, blob.data(), blob.size(), hipMemcpyHostToDevice));
            pa.p[pa.n++] = {d, q.slot, gt};
            pa.patch[pa.npatch++] = {100 + i, -1, q.slot};
            pa.patch[pa.npatch++] = {-1, q.slot, gt};
        }
        place_experts(pa, st);
        std::vector<int> so(NE); std::vector<uint8_t> fm(NP);
        CK(hipMemcpy(so.data(), dso, NE * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(fm.data(), dfmt, NP, hipMemcpyDeviceToHost));
        bool tables = true;
        for (int i = 0; i < NP; i++) tables &= so[100 + i] == src[i].slot && fm[src[i].slot] == src[i].x->g[src[i].g].gt;
        tables &= so[99] == -1 && so[100 + NP] == -1;
        printf("pool placement: routing table and slot formats patched %s\n", tables ? "ok" : "FAIL");
        fails += !tables;
        auto ref_src = [&](const Src &q, const float *x, double *y, double w) {
            expert_ref(*q.x, q.g ? q.x->n0 + q.s : q.s, x, y, w);
        };
        std::vector<float> x(D), ones(TOPK_E, 1.0f);
        for (auto &v : x) v = nd(rng);
        std::vector<int> hs(TOPK_E, -1);
        for (int i = 0; i < NP; i++) hs[i] = src[i].slot;
        float *dx = a.get<float>(D), *dw = a.get<float>(TOPK_E), *dy = a.get<float>((size_t)TOPK_E * D);
        int *dhs = a.get<int>(TOPK_E);
        QAct xq{a.get<int8_t>(D), a.get<float>(D / 32)}, hq{a.get<int8_t>(TOPK_E * FE), a.get<float>(TOPK_E * FE / 32)};
        CK(hipMemcpy(dx, x.data(), D * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dw, ones.data(), TOPK_E * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dhs, hs.data(), TOPK_E * 4, hipMemcpyHostToDevice));
        quant_act(dx, D, xq, st);
        moe_experts_dec(pg, pu, pd, dfmt, QTensor{}, QTensor{}, QTensor{}, dhs, 1, xq, hq, dw, nullptr, dy, st);
        std::vector<float> y((size_t)TOPK_E * D);
        CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
        double worst = 0;
        for (int i = 0; i < NP; i++) {
            std::vector<double> ref(D, 0.0);
            ref_src(src[i], x.data(), ref.data(), 1.0);
            worst = std::max(worst, rel(std::vector<float>(y.begin() + (size_t)i * D, y.begin() + (size_t)(i + 1) * D), ref));
        }
        printf("pool decode, slots of IQ4_XS, Q3_K and Q2_K experts: worst rel diff %.2e %s\n", worst, worst < 2e-2 ? "ok" : "FAIL");
        fails += worst >= 2e-2;
        {
            const int T = 48, per = 16, N = NP * per;
            std::vector<float> xt((size_t)T * D);
            for (auto &v : xt) v = nd(rng);
            std::vector<half> xh((size_t)N * D);
            std::vector<int> tok(N);
            for (int i = 0; i < NP; i++)
                for (int k = 0; k < per; k++) {
                    const int t = (i * 7 + k * 3) % T;
                    tok[i * per + k] = t;
                    for (int c = 0; c < D; c++) xh[(size_t)(i * per + k) * D + c] = __float2half(xt[(size_t)t * D + c]);
                }
            for (int t = 0; t < T; t++) for (int c = 0; c < D; c++) xt[(size_t)t * D + c] = __half2float(__float2half(xt[(size_t)t * D + c]));
            std::vector<float> ew(N, 1.0f);
            std::vector<MoeWork> items;
            std::vector<int> fmts;
            for (int f : {GGML_TYPE_IQ4_XS, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K})
                for (int i = 0; i < NP; i++)
                    if (src[i].x->g[src[i].g].gt == f) { items.push_back({src[i].slot, i * per, per, 0}); fmts.push_back(f); }
            half *dX = a.get<half>(xh.size()), *dH = a.get<half>((size_t)N * FE), *dY = a.get<half>((size_t)N * D);
            float *dew = a.get<float>(N);
            MoeWork *dwk = a.get<MoeWork>(items.size());
            CK(hipMemcpy(dX, xh.data(), xh.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dew, ew.data(), N * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(dwk, items.data(), items.size() * sizeof(MoeWork), hipMemcpyHostToDevice));
            QTensor none{};
            for (size_t i0 = 0; i0 < items.size();) {
                size_t i1 = i0 + 1;
                while (i1 < items.size() && fmts[i1] == fmts[i0]) i1++;
                moe_gemm_gateup(pg, pu, none, none, dwk + i0, (int)(i1 - i0), dX, dH, st, fmts[i0]);
                i0 = i1;
            }
            moe_gemm_down(pd, none, dwk, (int)items.size(), dH, dew, dY, st);
            std::vector<half> yh((size_t)N * D);
            CK(hipMemcpy(yh.data(), dY, yh.size() * 2, hipMemcpyDeviceToHost));
            double w2 = 0;
            for (int i = 0; i < NP; i++)
                for (int k = 0; k < per; k += 5) {
                    std::vector<double> ref(D, 0.0);
                    ref_src(src[i], &xt[(size_t)tok[i * per + k] * D], ref.data(), 1.0);
                    std::vector<float> got(D);
                    for (int c = 0; c < D; c++) got[c] = __half2float(yh[(size_t)(i * per + k) * D + c]);
                    w2 = std::max(w2, rel(got, ref));
                }
            printf("pool prefill GEMMs, slots of IQ4_XS, Q3_K and Q2_K experts: worst rel diff %.2e %s\n", w2, w2 < 1e-2 ? "ok" : "FAIL");
            fails += w2 >= 1e-2;
        }
        CK(hipStreamSynchronize(st));
        a.used = mark;
    }
    cpu.disarm(); cpu.stop();
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
