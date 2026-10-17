// Flash-Next decode kernels in isolation (T = 1, synthetic weights with the model's shapes and formats): time per launch
// of each piece of a layer, launched back to back (so it includes the gap between launches).
//   build_fn/bench_dec_fn [reps]
#include "../engine_fn/kernels.h"
#include "../engine_fn/cpu_moe.h"
#include <cstring>
#include <functional>
#include <random>
#include <vector>

static float time_us(hipStream_t st, int reps, const std::function<void()> &f) {
    for (int i = 0; i < 10; i++) f();
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0, st));
    for (int i = 0; i < reps; i++) f();
    CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
    float ms; CK(hipEventElapsedTime(&ms, e0, e1));
    return ms * 1000 / reps;
}

static QTensor qt(Arena &a, int type, int rows, int cols, std::mt19937 &rng) {
    QTensor t = make_qtensor(a, type, rows, cols);
    std::vector<std::pair<const void *, size_t>> arrs;
    qtensor_arrays(t, arrs);
    for (auto &p : arrs) {                     // small random bytes: finite scales, realistic memory traffic
        std::vector<uint8_t> h(p.second);
        for (auto &b : h) b = rng() & 0x3b;
        CK(hipMemcpy((void *)p.first, h.data(), h.size(), hipMemcpyHostToDevice));
    }
    return t;
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 300;
    hipStream_t st; CK(hipStreamCreate(&st));
    std::mt19937 rng(1);
    Arena a; a.init((size_t)3 << 30);
    const int n_hot = 125, T = 1;
    // weights of one GDN layer
    QTensor qkv = qt(a, Q6_K, QKV, D, rng), z = qt(a, Q6_K, GV * GD, D, rng), sout = qt(a, Q8_0, D, GV * GD, rng);
    QTensor hdown = qt(a, Q8_0, HCR, HCD, rng), hup = qt(a, Q8_0, HCD, HCR, rng), hinj = qt(a, Q8_0, HC, HCD, rng);
    QTensor hg = qt(a, IQ4_XS, n_hot * FE, D, rng), hu = qt(a, IQ4_XS, n_hot * FE, D, rng), hd = qt(a, IQ4_NL, n_hot * D, FE, rng);
    QTensor sg_ = qt(a, Q6_K, FS, D, rng), su = qt(a, Q6_K, FS, D, rng), sd = qt(a, Q8_0, D, FS, rng);
    half *router = a.get<half>((size_t)NE * D);
    {
        std::vector<half> h((size_t)NE * D); std::normal_distribution<float> nd(0, 0.05f);
        for (auto &v : h) v = __float2half(nd(rng));
        CK(hipMemcpy(router, h.data(), h.size() * 2, hipMemcpyHostToDevice));
    }
    float *wsg = a.get<float>(D), *gamma = a.get<float>(HCD), *res = a.get<float>(HCD * 8), *mx = a.get<float>(D * 8);
    float *lo = a.get<float>(HCR * 8), *inject = a.get<float>(HC * 8), *inv = a.get<float>(64), *logits = a.get<float>(NE * 8);
    float *sgv = a.get<float>(8), *wts = a.get<float>(80), *y = a.get<float>((size_t)88 * D), *proj = a.get<float>(QKV * 8), *zb = a.get<float>(GV * GD * 8);
    half *hh = a.get<half>(D * 8);
    QAct hq{a.get<int8_t>(D * 8), a.get<float>(D * 8 / 32)}, hq6{a.get<int8_t>(GV * GD * 8), a.get<float>(GV * GD * 8 / 32)};
    QAct hqe{a.get<int8_t>((size_t)88 * FE), a.get<float>((size_t)88 * FE / 32)};
    int *slot_of = a.get<int>(NE), *ids = a.get<int>(80), *hot = a.get<int>(80);
    unsigned *counter = a.get<unsigned>(1);
    CK(hipMemset(counter, 0, 4));
    for (float *p : {wsg, gamma, res, mx}) {
        std::vector<float> h(HCD); std::normal_distribution<float> nd(0, 1);
        for (auto &v : h) v = nd(rng);
        CK(hipMemcpy(p, h.data(), (p == mx || p == wsg ? D : HCD) * 4, hipMemcpyHostToDevice));
    }
    std::vector<int> so(NE);
    {
        std::vector<int> perm(NE); for (int i = 0; i < NE; i++) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        int nc = 0;
        for (int i = 0; i < NE; i++) so[perm[i]] = i < n_hot ? i : -1 - nc++;
        CK(hipMemcpy(slot_of, so.data(), NE * 4, hipMemcpyHostToDevice));
    }
    // a decode pick set: 6 hot, 3 for the CPU, 1 for the GPU from RAM, then the shared expert
    std::vector<int> hs = {3, 77, -1, 12, -1, 100, 45, -2 - 5, -1, 9, 0};
    CK(hipMemcpy(hot, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemset(hq.q, 1, D)); CK(hipMemset(hq.d, 0, D / 32 * 4));
    CK(hipMemset(hq6.q, 1, GV * GD)); CK(hipMemset(hq6.d, 0, GV * GD / 32 * 4));
    // cold experts in registered host memory (Q3_K gate / up, IQ4_NL down), 64 slots
    const int n_cold = 64;
    const size_t grow = qrow_bytes(Q3_K, D), drow = qrow_bytes(IQ4_NL, FE);
    const size_t gbytes = (size_t)n_cold * FE * grow, dbytes = (size_t)n_cold * D * drow;
    uint8_t *cold_h; CK(hipHostMalloc(&cold_h, 2 * gbytes + dbytes, hipHostMallocMapped));
    for (size_t i = 0; i < 2 * gbytes + dbytes; i++) cold_h[i] = rng() & 0x3b;
    uint8_t *cold_d; CK(hipHostGetDevicePointer((void **)&cold_d, cold_h, 0));
    ColdDec cd; cd.gate[0] = cold_d; cd.up[0] = cold_d + gbytes; cd.down = cold_d + 2 * gbytes; cd.type[0] = Q3_K; cd.n0 = n_cold;

    struct R { const char *name; float us; };
    std::vector<R> out;
    out.push_back({"hc_mix_fused (2 kernels)", time_us(st, reps, [&] {
        hc_mix_fused(res, gamma, hdown, hup, &hinj, T, lo, inject, HC, inv, mx, hh, hq, st); })});
    out.push_back({"gemv qkv + z (Q6_K, 38 MB)", time_us(st, reps, [&] {
        GemvArgs g; g.seg[0] = {qkv, proj, QKV}; g.seg[1] = {z, zb, GV * GD}; g.nseg = 2; g.x = hq; g.T = T; gemv(g, st); })});
    out.push_back({"gemv_hc ssm_out (Q8_0, 16.7 MB)", time_us(st, reps, [&] { gemv_hc(sout, hq6, T, res, inject, st); })});
    out.push_back({"router_route_dec (no CPU handoff)", time_us(st, reps, [&] {
        router_route_dec(router, wsg, mx, T, logits, sgv, slot_of, ids, wts, hot, nullptr, 0.25f, 0, 0, counter, st); })});
    CK(hipMemcpy(hot, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));   // the router overwrote the pick set
    out.push_back({"moe_experts_dec (6 hot + shared)", time_us(st, reps, [&] {
        moe_experts_dec(hg, hu, hd, sg_, su, sd, hot, T, hq, hqe, wts, sgv, y, st); })});
    out.push_back({"moe_cold_dec (1 pick from RAM)", time_us(st, reps, [&] { moe_cold_dec(cd, hot, T, hq, hqe, wts, y, st); })});
    {   // hot only / cold only / both in the merged launches
        std::vector<int> hot_only = hs, cold_only = hs;
        for (auto &v : hot_only) if (v <= -2) v = -1;
        for (auto &v : cold_only) if (v >= 0) v = -1;
        for (auto [name, set] : {std::pair{"moe_dec hot only (6 + shared)", &hot_only}, std::pair{"moe_dec cold only (1 pick)", &cold_only},
                                 std::pair{"moe_dec both", &hs}}) {
            CK(hipMemcpy(hot, set->data(), set->size() * 4, hipMemcpyHostToDevice));
            out.push_back({name, time_us(st, reps, [&] { moe_dec(hg, hu, hd, sg_, su, sd, &cd, hot, T, hq, hqe, wts, sgv, y, st); })});
        }
        CK(hipMemcpy(hot, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    }
    out.push_back({"moe_finish (no CPU)", time_us(st, reps, [&] {
        moe_finish(nullptr, 0, y, hot, y + (size_t)T * TOPK_E * D, nullptr, nullptr, inject, res, T, st); })});
    float tot = 0;
    for (auto &r : out) { printf("%-36s %7.1f us\n", r.name, r.us); tot += r.us; }
    printf("%-36s %7.1f us\n", "sum", tot);

    // ---- a GDN layer's whole decode sequence, cycling over NLAY copies of the weights (more than the Infinity Cache holds) ----
    constexpr int NLAY = 6;
    struct Lw { QTensor qkv, z, sout, ad, au, ai, fd, fu, fi, hg, hu, hd, sg, su, sd; half *router, *Wab; int *slot_of; };
    std::vector<Lw> lw(NLAY);
    for (auto &l : lw) {
        l.qkv = qt(a, Q6_K, QKV, D, rng); l.z = qt(a, Q6_K, GV * GD, D, rng); l.sout = qt(a, Q8_0, D, GV * GD, rng);
        l.ad = qt(a, Q8_0, HCR, HCD, rng); l.au = qt(a, Q8_0, HCD, HCR, rng); l.ai = qt(a, Q8_0, HC, HCD, rng);
        l.fd = qt(a, Q8_0, HCR, HCD, rng); l.fu = qt(a, Q8_0, HCD, HCR, rng); l.fi = qt(a, Q8_0, HC, HCD, rng);
        l.hg = qt(a, IQ4_XS, n_hot * FE, D, rng); l.hu = qt(a, IQ4_XS, n_hot * FE, D, rng); l.hd = qt(a, IQ4_NL, n_hot * D, FE, rng);
        l.sg = qt(a, Q6_K, FS, D, rng); l.su = qt(a, Q6_K, FS, D, rng); l.sd = qt(a, Q8_0, D, FS, rng);
        l.router = a.get<half>((size_t)NE * D); CK(hipMemcpy(l.router, router, (size_t)NE * D * 2, hipMemcpyDeviceToDevice));
        l.Wab = a.get<half>((size_t)2 * GV * D); CK(hipMemset(l.Wab, 0, (size_t)2 * GV * D * 2));
        l.slot_of = slot_of;
    }
    float *convw = a.get<float>(QKV * 4), *Av = a.get<float>(GV), *dtv = a.get<float>(GV), *nw = a.get<float>(GD);
    float *conv_state = a.get<float>(QKV * 3), *S = a.get<float>((size_t)GV * GD * GD);
    float *vkn = a.get<float>(8 * GK * GD), *vvc = a.get<float>(8 * GV * GD), *vab = a.get<float>(8 * 2 * GV);
    for (float *p : {convw, Av, dtv, nw, conv_state}) CK(hipMemset(p, 0, 4));
    CK(hipMemset(S, 0, (size_t)GV * GD * GD * 4));
    const bool cold_gpu = getenv("BENCH_NO_COLD") == nullptr, pf_on = getenv("BENCH_PREFETCH") != nullptr;
    // the engine's decode prefetch: a layer's dense weights read on a second stream once the previous layer's router ran
    std::vector<PrefetchRange *> pf(NLAY);
    std::vector<int> pfn(NLAY);
    unsigned *sink = a.get<unsigned>(1);
    for (int i = 0; i < NLAY; i++) {
        const Lw &l = lw[i];
        std::vector<std::pair<const void *, size_t>> arrs;
        for (const QTensor *q : {&l.ad, &l.au, &l.ai, &l.qkv, &l.z, &l.sout, &l.fd, &l.fu, &l.fi, &l.sg, &l.su, &l.sd}) qtensor_arrays(*q, arrs);
        arrs.push_back({l.router, (size_t)NE * D * 2}); arrs.push_back({l.Wab, (size_t)2 * GV * D * 2});
        std::vector<PrefetchRange> r;
        for (auto [p, n] : arrs) if (n >= 16) r.push_back({(const int4 *)p, n / 16});
        pf[i] = a.get<PrefetchRange>(r.size()); pfn[i] = (int)r.size();
        CK(hipMemcpy(pf[i], r.data(), r.size() * sizeof(PrefetchRange), hipMemcpyHostToDevice));
    }
    hipStream_t pst; CK(hipStreamCreateWithFlags(&pst, hipStreamNonBlocking));
    hipEvent_t routed; CK(hipEventCreateWithFlags(&routed, hipEventDisableTiming));
    auto layer = [&](int i) {
        const Lw &l = lw[i % NLAY];
        hc_mix_fused(res, gamma, l.ad, l.au, &l.ai, T, lo, inject, HC, inv, mx, hh, hq, st);
        GemvArgs g; g.seg[0] = {l.qkv, proj, QKV}; g.seg[1] = {l.z, zb, GV * GD}; g.nseg = 2; g.x = hq; g.T = T; gemv(g, st);
        GdnStepArgs gs{}; gs.raw = proj; gs.z = zb; gs.hh = hh; gs.Wab = l.Wab; gs.conv_w = convw; gs.A = Av; gs.dt = dtv; gs.norm_w = nw;
        gs.conv_state = conv_state; gs.S = S; gs.T = T; gs.verify = 0; gs.kn_out = vkn; gs.vc_out = vvc; gs.ab_out = vab; gs.out = hq6;
        gdn_step(gs, st);
        gemv_hc(l.sout, hq6, T, res, inject, st);
        hc_mix_fused(res, gamma, l.fd, l.fu, &l.fi, T, lo, inject, HC, inv, mx, hh, hq, st);
        router_route_dec(l.router, wsg, mx, T, logits, sgv, l.slot_of, ids, wts, hot, nullptr, 0.25f, 0, 0, counter, st);
        CK(hipMemcpyAsync(hot, hs.data(), hs.size() * 4, hipMemcpyHostToDevice, st));   // the fixed pick set
        if (pf_on) {
            CK(hipEventRecord(routed, st)); CK(hipStreamWaitEvent(pst, routed, 0));
            prefetch(pf[(i + 1) % NLAY], pfn[(i + 1) % NLAY], sink, pst);
        }
        moe_dec(l.hg, l.hu, l.hd, l.sg, l.su, l.sd, cold_gpu ? &cd : nullptr, hot, T, hq, hqe, wts, sgv, y, st);
        moe_finish(nullptr, 0, y, hot, y + (size_t)T * TOPK_E * D, nullptr, nullptr, inject, res, T, st);
    };
    int li = 0;
    const float us = time_us(st, reps, [&] { layer(li++); });
    printf("GDN layer sequence (%d weight copies, GPU cold pick %s, prefetch %s): %.1f us per layer\n", NLAY, cold_gpu ? "on" : "off",
           pf_on ? "on" : "off", us);
}
