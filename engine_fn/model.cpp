// Engine implementation: sequences, snapshots, forward passes, MTP drafting, speculative steps.
#include "model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>

static const QAct NO_Q{nullptr, nullptr};
static QAct qoff(QAct a, size_t elems) { return {a.q + elems, a.d + elems / 32}; }

// ---- sequences --------------------------------------------------------------------------------

int Engine::seq_new() {
    for (size_t i = 0; i < seqs.size(); i++) {
        if (seqs[i].used) continue;
        seqs[i].used = true;
        seq_reset(seqs[i]);
        return (int)i;
    }
    return -1;
}

void Engine::seq_reset(Seq &s) {
    for (int p : s.pages) page_release(p);
    s.n = 0; s.pending = -1; s.pages.clear(); s.seen.clear(); s.top_v.clear(); s.top_i.clear();
    s.hist[0] = s.hist[1] = -1;
    s.rng.seed(std::random_device{}());
    CK(hipMemsetAsync(s.ssm, 0, ssm_bytes(), st));
    CK(hipMemsetAsync(s.conv, 0, conv_bytes(), st));
    CK(hipMemsetAsync(s.ple, 0, ple_bytes(), st));
    CK(hipMemsetAsync(s.h_last, 0, (size_t)HCD * 4, st));
}

void Engine::seq_free(int id) {
    Seq &s = seqs.at(id);
    for (int p : s.pages) page_release(p);
    s.pages.clear();
    s.used = false;
}

void Engine::seq_truncate(int id, int n) {
    Seq &s = seqs.at(id);
    if (n == s.n) { s.pending = -1; return; }
    if (n != 0) FAIL("truncate to %d: the recurrent state can only be rolled back via snapshots", n);
    seq_reset(s);
}

int Engine::page_alloc() {
    if (free_pages.empty()) throw std::runtime_error("out of KV pages");
    const int p = free_pages.back();
    free_pages.pop_back();
    page_ref[p] = 1;
    return p;
}

void Engine::page_release(int p) {
    if (--page_ref[p] == 0) free_pages.push_back(p);
}

// Make s cover n_tokens, with exclusive ownership of every page written from position `from` on
// (pages shared with snapshots or other sequences are copied first).
void Engine::ensure_pages(Seq &s, int n_tokens, int from) {
    const int need = (n_tokens + PAGE - 1) / PAGE;
    int dirty = (int)s.pages.size();
    int shared = 0;
    for (int i = from / PAGE; i < dirty && i < need; i++) shared += page_ref[s.pages[i]] > 1;
    if ((int)free_pages.size() < need - dirty + shared) throw std::runtime_error("out of KV pages");
    while ((int)s.pages.size() < need) {
        const int p = page_alloc();   // fresh pages start zeroed (WMMA results must not depend on stale masked data)
        CK(hipMemsetAsync(kv_pool + (size_t)p * KV_PAGE, 0, KV_PAGE, st));
        s.pages.push_back(p);
    }
    for (int i = from / PAGE; i < dirty && i < need; i++) {
        if (page_ref[s.pages[i]] == 1) continue;
        const int np = page_alloc();
        CK(hipMemcpyAsync(kv_pool + (size_t)np * KV_PAGE, kv_pool + (size_t)s.pages[i] * KV_PAGE, KV_PAGE, hipMemcpyDeviceToDevice, st));
        page_release(s.pages[i]);
        s.pages[i] = np;
        dirty = std::min(dirty, i);
    }
    if (dirty < need)
        CK(hipMemcpyAsync(s.d_pages + dirty, s.pages.data() + dirty, (need - dirty) * 4, hipMemcpyHostToDevice, st));
}

// ---- snapshots: recurrent state copied to pinned host memory, KV pages shared copy-on-write ----------

int Engine::snapshot(int id) {
    Seq &s = seqs.at(id);
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    if (!sn.host) CK(hipHostMalloc(&sn.host, state_bytes()));
    sn.used = true;
    sn.n = s.n;
    sn.hist[0] = s.hist[0]; sn.hist[1] = s.hist[1];
    sn.pages.assign(s.pages.begin(), s.pages.begin() + (s.n + PAGE - 1) / PAGE);
    for (int p : sn.pages) page_ref[p]++;
    CK(hipStreamWaitEvent(st, copied, 0));
    size_t o = 0;
    CK(hipMemcpyAsync(stage_state + o, s.ssm, ssm_bytes(), hipMemcpyDeviceToDevice, st)); o += ssm_bytes();
    CK(hipMemcpyAsync(stage_state + o, s.conv, conv_bytes(), hipMemcpyDeviceToDevice, st)); o += conv_bytes();
    CK(hipMemcpyAsync(stage_state + o, s.ple, ple_bytes(), hipMemcpyDeviceToDevice, st)); o += ple_bytes();
    CK(hipMemcpyAsync(stage_state + o, s.h_last, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
    CK(hipEventRecord(staged, st));
    CK(hipStreamWaitEvent(cst, staged, 0));
    CK(hipMemcpyAsync(sn.host, stage_state, state_bytes(), hipMemcpyDeviceToHost, cst));
    CK(hipEventRecord(copied, cst));
    return (int)si;
}

void Engine::restore(int id, int snap) {
    Seq &s = seqs.at(id);
    Snap &sn = snaps.at(snap);
    CK(hipStreamWaitEvent(st, copied, 0));
    for (int p : s.pages) page_release(p);
    s.pages = sn.pages;
    for (int p : s.pages) page_ref[p]++;
    if (!s.pages.empty()) CK(hipMemcpyAsync(s.d_pages, s.pages.data(), s.pages.size() * 4, hipMemcpyHostToDevice, st));
    s.n = sn.n;
    s.hist[0] = sn.hist[0]; s.hist[1] = sn.hist[1];
    s.pending = -1;
    s.top_v.clear(); s.top_i.clear(); s.seen.clear();
    const uint8_t *h = (const uint8_t *)sn.host;
    size_t o = 0;
    CK(hipMemcpyAsync(s.ssm, h + o, ssm_bytes(), hipMemcpyHostToDevice, st)); o += ssm_bytes();
    CK(hipMemcpyAsync(s.conv, h + o, conv_bytes(), hipMemcpyHostToDevice, st)); o += conv_bytes();
    CK(hipMemcpyAsync(s.ple, h + o, ple_bytes(), hipMemcpyHostToDevice, st)); o += ple_bytes();
    CK(hipMemcpyAsync(s.h_last, h + o, (size_t)HCD * 4, hipMemcpyHostToDevice, st));
}

void Engine::snap_free(int snap) {
    Snap &sn = snaps.at(snap);
    CK(hipStreamSynchronize(st));
    CK(hipStreamSynchronize(cst));
    for (int p : sn.pages) page_release(p);
    sn.pages.clear();
    sn.used = false;
}

// ---- building blocks --------------------------------------------------------------------------

Engine::LtPlan &Engine::lt_plan(int N, int K, int T, int ldx, int ldd) {
    const auto key = std::make_tuple(N, K, T, ldx, ldd);
    auto it = lt_plans.find(key);
    if (it != lt_plans.end()) return it->second;
    LtPlan p;
    hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16F, K, N, K);
    hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16F, K, T, ldx);
    hipblasLtMatrixLayoutCreate(&p.c, HIP_R_16F, N, T, ldd);
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_size, sizeof(lt_ws_size));
    hipblasLtMatmulHeuristicResult_t res;
    int n = 0;
    hipblasLtMatmulAlgoGetHeuristic(lt, p.desc, p.a, p.b, p.c, p.c, pref, 1, &res, &n);
    hipblasLtMatmulPreferenceDestroy(pref);
    if (n < 1) FAIL("hipBLASLt has no algorithm for %dx%dx%d", N, K, T);
    p.algo = res.algo;
    return lt_plans.emplace(key, p).first->second;
}

// y = x W^T (x fp16 [T][ldx], W fp16 [N][K]); fp32 results go through an fp16 scratch (fast kernels)
void Engine::gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate) {
    const float alpha = 1.0f, beta = 0.0f;
    half *dst = y_f16 && !accumulate ? (half *)y : b.gemm16;
    const int ldd = dst == y ? ldy : N;
    const int Tp = std::min(umax * HC, (T + 63) / 64 * 64);
    LtPlan &p = lt_plan(N, K, Tp, ldx, ldd);
    if (hipblasLtMatmul(lt, p.desc, &alpha, W, p.a, x, p.b, &beta, dst, p.c, dst, p.c, &p.algo, lt_ws, lt_ws_size, st) != HIPBLAS_STATUS_SUCCESS)
        FAIL("hipblasLtMatmul failed (%dx%dx%d)", N, K, T);
    if (dst != y) store_f16(b.gemm16, N, T, (float *)y, ldy, accumulate, st);
}

void Engine::linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16) {
    if (T <= 8) {
        gemv(W, qin, T, (float *)out, ldo, mode, st);
    } else if (T <= GEMM_Q_MAX && W.type != F16 && W.rows % 32 == 0 && W.cols % 256 == 0) {
        gemm_q(W, hin, ldin, T, out, ldo, mode, out_f16, st);
    } else {
        const half *w16 = (const half *)W.f;
        if (W.type != F16) { dequant(W, b.deq, st); w16 = b.deq; }
        gemm(w16, W.rows, W.cols, hin, ldin, T, out, ldo, out_f16, mode == GEMV_ADD);
    }
}

// hyper-connection mixer: residual res [T][HC][D] -> block input b.mx (fp32), b.hq (int8, decode) / b.hh (fp16),
// and the stream weights `inject` [T][HC] when the mixer has them
static void hc_mix_run(Engine &e, const HcMix &m, const float *res, int T, float *inject) {
    auto &b = e.b;
    if (T <= 8) {
        hc_norm(res, m.norm, T, b.xn, nullptr, b.xn_q, e.st);
        gemv(m.down, b.xn_q, T, b.lo, HCR, GEMV_STORE, e.st);
        hc_lo(b.lo, T, nullptr, b.lo_q, e.st);
        gemv(m.up, b.lo_q, T, b.gate, HCD, GEMV_STORE, e.st);
        if (inject) gemv(m.inject, b.xn_q, T, inject, HC, GEMV_STORE, e.st);
        hc_mix(b.xn, b.gate, T, b.mx, b.hh, b.hq, e.st);
    } else {
        hc_norm(res, m.norm, T, b.xn, b.xn_h, NO_Q, e.st);
        e.linear(m.down, T, NO_Q, b.xn_h, HCD, b.lo, HCR, GEMV_STORE);
        hc_lo(b.lo, T, b.lo_h, NO_Q, e.st);
        e.linear(m.up, T, NO_Q, b.lo_h, HCR, b.gate, HCD, GEMV_STORE);
        if (inject) e.linear(m.inject, T, NO_Q, b.xn_h, HCD, inject, HC, GEMV_STORE);
        hc_mix(b.xn, b.gate, T, b.mx, b.hh, NO_Q, e.st);
    }
}

void Engine::attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *out) {
    const bool small = T <= 8;
    if (small) {
        GemvArgs a;
        a.seg[0] = {w.q, b.proj, 2 * NH * HD}; a.seg[1] = {w.k, b.kb, NKV * HD}; a.seg[2] = {w.v, b.vb, NKV * HD};
        a.nseg = 3; a.x = b.hq; a.T = T;
        gemv(a, st);
        GemvArgs ix;
        ix.seg[0] = {w.idx_q, b.qidx, IDX_H * IDX_D}; ix.seg[1] = {w.idx_k, b.kidx, IDX_D}; ix.nseg = 2; ix.x = b.hq; ix.T = T;
        gemv(ix, st);
    } else {
        linear(w.q, T, NO_Q, b.hh, D, b.proj, 2 * NH * HD, GEMV_STORE);
        linear(w.k, T, NO_Q, b.hh, D, b.kb, NKV * HD, GEMV_STORE);
        linear(w.v, T, NO_Q, b.hh, D, b.vb, NKV * HD, GEMV_STORE);
        linear(w.idx_q, T, NO_Q, b.hh, D, b.qidx, IDX_H * IDX_D, GEMV_STORE);
        linear(w.idx_k, T, NO_Q, b.hh, D, b.kidx, IDX_D, GEMV_STORE);
    }
    idx_prep(b.kidx, b.qidx, w.idx_qn, w.idx_kn, T, pos0, kv(s), kvl, b.qidx16, st);
    attn_prep(b.proj, b.kb, b.vb, w.q_norm, w.k_norm, T, pos0, kv(s), kvl, b.Q16, st);
    AttnArgs a;
    a.Q16 = b.Q16; a.kv = kv(s); a.layer = kvl; a.T = T; a.pos0 = pos0; a.qg = b.proj;
    a.part_o = b.part_o; a.part_ml = b.part_ml;
    const int last_pos = pos0 + T - 1;
    if ((last_pos + 1) / KPOOL <= SEL_BLOCKS) {            // every query sees at most SEL_BLOCKS complete blocks: dense
        attention(a, small ? nullptr : b.att_h, small ? b.mq : NO_Q, st);
    } else {
        const int nblk = (last_pos + 1) / KPOOL;
        for (int g0 = 0; g0 < T; g0 += SEL_G) {
            const int G = std::min(SEL_G, T - g0);
            idx_select(b.qidx16 + (size_t)g0 * IDX_H * IDX_D, pos0 + g0, G, nblk, kv(s), kvl, b.score,
                       b.sel + (size_t)g0 * (SEL_BLOCKS + 1), st);
        }
        attention_sel(a, b.sel, small ? nullptr : b.att_h, small ? b.mq : NO_Q, st);
    }
    if (small) gemv(w.o, b.mq, T, out, D, GEMV_STORE, st);
    else linear(w.o, T, NO_Q, b.att_h, NH * HD, out, D, GEMV_STORE);
}

void Engine::gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *out) {
    if (T <= 8) {
        float *raw = b.v_raw + (size_t)gl * 8 * QKV;
        GemvArgs a;
        a.seg[0] = {w.qkv, raw, QKV}; a.seg[1] = {w.z, b.zb, GV * GD}; a.nseg = 2; a.x = b.hq; a.T = T;
        gemv(a, st);
        GdnStepArgs g;
        g.raw = raw; g.z = b.zb; g.hh = b.hh; g.Wab = w.Wab;
        g.conv_w = w.conv; g.A = gdn_A + gl * GV; g.dt = gdn_dt + gl * GV; g.norm_w = w.ssm_norm;
        g.conv_state = s.conv + (size_t)gl * QKV * 3; g.S = s.ssm + (size_t)gl * GV * GD * GD;
        g.T = T; g.verify = verify;
        g.kn_out = b.v_kn + (size_t)gl * 8 * GK * GD; g.vc_out = b.v_vc + (size_t)gl * 8 * GV * GD;
        g.ab_out = b.v_ab + (size_t)gl * 8 * 2 * GV; g.out = b.mq;
        gdn_step(g, st);
        gemv(w.ssm_out, b.mq, T, out, D, GEMV_STORE, st);
        return;
    }
    float *raw = b.proj, *z = b.zb, *ab = b.kb;
    linear(w.qkv, T, NO_Q, b.hh, D, raw, QKV, GEMV_STORE);
    linear(w.z, T, NO_Q, b.hh, D, z, GV * GD, GEMV_STORE);
    gemm(w.Wab, 2 * GV, D, b.hh, D, T, ab, 2 * GV, false, false);
    gdn_conv(raw, T, w.conv, s.conv + (size_t)gl * QKV * 3, true, b.qn, b.kn, b.vc, st);
    GdnScanArgs sa{};
    sa.S = s.ssm + (size_t)gl * GV * GD * GD; sa.qn = b.qn; sa.kn = b.kn; sa.vc = b.vc; sa.ab = ab;
    sa.A = gdn_A + gl * GV; sa.dtb = gdn_dt + gl * GV; sa.T = T; sa.write_state = 1; sa.out = b.vc;
    gdn_scan(sa, 1, st);
    gdn_out(b.vc, z, w.ssm_norm, T, b.att_h, NO_Q, st);
    linear(w.ssm_out, T, NO_Q, b.att_h, GV * GD, out, D, GEMV_STORE);
}

// Routed experts + shared expert for T tokens of b.mx / b.hq. Decode path, also used (8 tokens at a time) by
// prefill until the grouped GEMM path exists: hot picks on the GPU, cold picks on the CPU (cpu_cold).
void Engine::moe_block(const Layer &w, int layer, int T, bool cpu_cold, float *out) {
    if (T > 8) { moe_prefill(w, layer, T, out); return; }
    router_dec(w.router, w.sg_w, b.mx, T, b.logits_e, b.sg, st);
    route_dec(b.logits_e, T, w.slot_of, b.ids, b.wts, b.hot, cpu_cold ? d_sh : nullptr, b.mx, layer, gen + 1, st);
    if (cpu_cold) gen++;
    GemvArgs sg; sg.seg[0] = {w.sh_gate, nullptr, 0}; sg.up = w.sh_up; sg.gq = b.hs_q; sg.x = b.hq; sg.T = T; sg.mode = GEMV_GLU;
    gemv(sg, st);
    gemv(w.sh_down, b.hs_q, T, b.shared, D, GEMV_STORE, st);
    moe_experts_dec(w.hot_gate, w.hot_up, w.hot_down, b.hot, T, b.hq, b.hq_e, b.wts, b.y, st);
    if (cpu_cold) moe_wait(&d_sh->done, gen, st);
    moe_sum(b.y, b.hot, b.shared, b.sg, cpu_cold ? d_sh->y : nullptr, T, out, st);
}

// Cost model of a prefill chunk's cold experts (per layer), measured on this machine: the CPU computes an expert in
// place (bandwidth-bound for experts with few tokens, ~165 GMAC/s of ggml's IQ4 dot products for busy ones), the GPU
// first copies it to VRAM (~26 GB/s over PCIe) and then computes it for nothing. Fixed constants rather than live
// timings, so the split (and with it the output) depends only on the routing.
namespace {
constexpr double CPU_BW = 56e9, CPU_MAC = 165e9, GPU_BW = 26e9, GPU_PER_EXPERT = 4e-6;
constexpr double EXPERT_MAC = 3.0 * FE * D;
double cpu_time(double bytes, double macs) {
    const double a = bytes / CPU_BW, m = macs / CPU_MAC;
    return std::max(a, m) + 0.25 * std::min(a, m);
}
}

// Routed experts for a prefill chunk: picks sorted by expert on the host (one sync per layer). Hot experts run as
// grouped WMMA GEMMs from VRAM. Of the cold experts with picks, the CPU workers take the ones with the fewest tokens
// and compute them where they are, while the GPU copies the rest into a two-slot VRAM staging ring (copy stream),
// relayouts them and runs the same GEMMs; then the per-token sum in router order.
void Engine::moe_prefill(const Layer &w, int layer, int T, float *out) {
    const int N = T * TOPK_E;
    gemm(w.router, NE, D, b.hh, D, T, b.rlog, NE, false, false);
    route_many(b.rlog, w.sg_w, b.mx, T, b.idsT, b.wtsT, b.sgT, st);
    // shared expert as GEMMs: gate | up (fp16) -> silu * up -> down
    half *g = b.att_h, *u = b.att_h + (size_t)T * FS;
    linear(w.sh_gate, T, NO_Q, b.hh, D, g, FS, GEMV_STORE, true);
    linear(w.sh_up, T, NO_Q, b.hh, D, u, FS, GEMV_STORE, true);
    swiglu_h(g, u, (int64_t)T * FS, st);
    linear(w.sh_down, T, NO_Q, g, FS, b.blk, D, GEMV_STORE);
    // picks -> host (and the inputs, for the CPU's share)
    CK(hipMemcpyAsync(b.h_ids, b.idsT, (size_t)N * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(b.h_wts, b.wtsT, (size_t)N * 4, hipMemcpyDeviceToHost, st));
    if (w.n_cold) CK(hipMemcpyAsync(b.h_x, b.mx, (size_t)T * D * 4, hipMemcpyDeviceToHost, st));
    CK(hipStreamSynchronize(st));
    const std::vector<int> &slot_of = w.slot_h;
    std::vector<int> cnt(NE, 0);
    for (int j = 0; j < N; j++) cnt[b.h_ids[j]]++;
    // cold experts with picks, fewest tokens first: the CPU takes a prefix, the GPU the rest
    std::vector<int> ce, hot;
    for (int e = 0; e < NE; e++) if (cnt[e]) (slot_of[e] >= 0 ? hot : ce).push_back(e);
    std::stable_sort(ce.begin(), ce.end(), [&](int a, int c) { return cnt[a] < cnt[c]; });
    const ColdLayer *cl = w.n_cold ? &cold[layer] : nullptr;
    const double eb = cl ? 2.0 * FE * cl->row_gu + (double)D * cl->row_dn : 0;
    int kc = 0;
    {
        double best = 1e30, macs = 0;
        const int K = (int)ce.size();
        for (int k = 0; k <= K; k++) {
            const double t = std::max(cpu_time(k * eb, macs), (K - k) * (eb / GPU_BW + GPU_PER_EXPERT));
            if (t < best) { best = t; kc = k; }
            if (k < K) macs += cnt[ce[k]] * EXPERT_MAC;
        }
    }
    std::vector<int> gpu_cold(ce.begin() + kc, ce.end()), cpu_cold(ce.begin(), ce.begin() + kc);
    auto by_slot = [&](int a, int c) { return slot_of[a] > slot_of[c]; };   // cold slots are -1 - s: descending = ascending s
    std::sort(hot.begin(), hot.end(), [&](int a, int c) { return slot_of[a] < slot_of[c]; });
    std::sort(gpu_cold.begin(), gpu_cold.end(), by_slot);
    std::stable_sort(cpu_cold.begin(), cpu_cold.end(), [&](int a, int c) { return cnt[a] > cnt[c]; });   // largest first
    // pick order: hot experts, the GPU's cold experts, the CPU's (whose outputs then fill one contiguous range of Ys)
    std::vector<int> off(NE, 0), cur(NE);
    int o = 0;
    for (const auto *v : {&hot, &gpu_cold, &cpu_cold}) for (int e : *v) { off[e] = o; o += cnt[e]; }
    int pc0 = 0;
    for (int e : hot) pc0 += cnt[e];
    for (int e : gpu_cold) pc0 += cnt[e];
    cur = off;
    for (int j = 0; j < N; j++) {                 // token order within an expert: stable
        const int e = b.h_ids[j], p = cur[e]++;
        b.h_tok[p] = j / TOPK_E; b.h_entw[p] = b.h_wts[j]; b.h_pos[j] = p;
    }
    if (!cpu_cold.empty()) {                      // the CPU starts right away
        j_slot.clear(); j_first.assign(1, 0);
        j_tok.assign(b.h_tok + pc0, b.h_tok + N); j_w.assign(b.h_entw + pc0, b.h_entw + N);
        for (int e : cpu_cold) { j_slot.push_back(-1 - slot_of[e]); j_first.push_back(off[e] - pc0 + cnt[e]); }
        MoeJob job;
        job.layer = layer; job.T = T; job.n_exp = (int)cpu_cold.size(); job.x = b.h_x;
        job.slot = j_slot.data(); job.first = j_first.data(); job.tok = j_tok.data(); job.w = j_w.data(); job.out = b.h_cout;
        cpu.submit(job);
    }
    // GPU work items: hot experts, then the GPU's cold experts in staging groups (slots relative to the group)
    int nw = 0;
    for (int e : hot) b.h_work[nw++] = {slot_of[e], off[e], cnt[e], 0};
    const int nhot = nw;
    std::vector<std::pair<int, int>> groups;      // (first work item, experts) per group
    for (size_t i = 0; i < gpu_cold.size(); i += STAGE_G) {
        const int n = (int)std::min<size_t>(STAGE_G, gpu_cold.size() - i);
        groups.push_back({nw, n});
        for (int c = 0; c < n; c++) { const int e = gpu_cold[i + c]; b.h_work[nw++] = {c, off[e], cnt[e], 0}; }
    }
    CK(hipMemcpyAsync(b.ent_tok, b.h_tok, (size_t)pc0 * 4, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(b.ent_w, b.h_entw, (size_t)pc0 * 4, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(b.pos, b.h_pos, (size_t)N * 4, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(b.work, b.h_work, (size_t)nw * sizeof(MoeWork), hipMemcpyHostToDevice, st));
    gather_rows(b.hh, b.ent_tok, pc0, b.Xs, st);
    QTensor none{};
    moe_gemm_gateup(w.hot_gate, w.hot_up, none, none, b.work, nhot, b.Xs, b.Hs, st);
    moe_gemm_down(w.hot_down, none, b.work, nhot, b.Hs, b.ent_w, b.Ys, st);
    if (!groups.empty()) {
        const size_t gsz = (size_t)FE * cl->row_gu, dsz = (size_t)D * cl->row_dn;   // bytes per expert: gate (= up), down
        auto dma = [&](int gi) {                  // the group's experts, packed; runs of consecutive slots in one copy
            const int k = gi % 2, i0 = (int)(gi * STAGE_G), n = groups[gi].second;
            CK(hipStreamWaitEvent(cst, raw_free[k], 0));
            uint8_t *r = b.raw[k];
            for (int c = 0; c < n;) {
                const int s0 = -1 - slot_of[gpu_cold[i0 + c]];
                int run = 1;
                while (c + run < n && -1 - slot_of[gpu_cold[i0 + c + run]] == s0 + run) run++;
                CK(hipMemcpyAsync(r + c * gsz, cl->gate + s0 * gsz, run * gsz, hipMemcpyHostToDevice, cst));
                CK(hipMemcpyAsync(r + (STAGE_G + c) * gsz, cl->up + s0 * gsz, run * gsz, hipMemcpyHostToDevice, cst));
                CK(hipMemcpyAsync(r + 2 * STAGE_G * gsz + c * dsz, cl->down + s0 * dsz, run * dsz, hipMemcpyHostToDevice, cst));
                c += run;
            }
            CK(hipEventRecord(raw_ready[k], cst));
        };
        dma(0);
        for (size_t gi = 0; gi < groups.size(); gi++) {
            if (gi + 1 < groups.size()) dma((int)gi + 1);
            const int k = gi % 2, ns = groups[gi].second;
            CK(hipStreamWaitEvent(st, raw_ready[k], 0));
            QTensor G = b.sg_[k], U = b.su_[k], Dn = b.sd_[k];
            G.rows = U.rows = ns * FE; Dn.rows = ns * D;
            relayout(G, b.raw[k], st, 0, (int64_t)ns * FE * D / 256);
            relayout(U, b.raw[k] + STAGE_G * gsz, st, 0, (int64_t)ns * FE * D / 256);
            relayout(Dn, b.raw[k] + 2 * STAGE_G * gsz, st, 0, (int64_t)ns * D * FE / 32);
            CK(hipEventRecord(raw_free[k], st));
            moe_gemm_gateup(G, U, none, none, b.work + groups[gi].first, ns, b.Xs, b.Hs, st);
            moe_gemm_down(Dn, none, b.work + groups[gi].first, ns, b.Hs, b.ent_w, b.Ys, st);
        }
    }
    if (!cpu_cold.empty()) {                      // the CPU's outputs into their rows of Ys
        cpu.wait();
        CK(hipMemcpyAsync(b.Ys + (size_t)pc0 * D, b.h_cout, (size_t)(N - pc0) * D * 2, hipMemcpyHostToDevice, st));
    }
    moe_combine(b.Ys, b.pos, b.blk, b.sgT, T, out, st);
}

// n-gram rows of tokens[0..T) at positions s.n.. : the hash of llama.cpp's llm_graph_input_qwen4exp_ple
// Table rows of the n-gram heads for tokens at the positions after history (h0, h1) (-1 = none): 8 bigram heads, then
// 8 trigram heads; an EOS (or no token) cuts the context at and before it.
void Engine::ple_ids(int h0, int h1, const int *tokens, int T, size_t *rows) const {
    constexpr int64_t EOS = EOS_TEXT;
    for (int t = 0; t < T; t++) {
        const int64_t p1 = t >= 1 ? tokens[t - 1] : h1;
        const int64_t p2 = t >= 2 ? tokens[t - 2] : t == 1 ? h1 : h0;
        bool cut = p1 < 0 || p1 == EOS;
        const int64_t c1 = cut ? EOS : p1;
        cut = cut || p2 < 0 || p2 == EOS;
        const int64_t c2 = cut ? EOS : p2;
        const uint64_t m2 = (uint64_t)tokens[t] * ple_mult[0] ^ (uint64_t)c1 * ple_mult[1];
        const uint64_t m3 = m2 ^ (uint64_t)c2 * ple_mult[2];
        for (int g = 0; g < 8; g++) rows[(size_t)t * PLE_HEADS + g] = m2 % ple_voc[g] + ple_off[g];
        for (int g = 8; g < 16; g++) rows[(size_t)t * PLE_HEADS + g] = m3 % ple_voc[g] + ple_off[g];
    }
}

// Ask the kernel for every page these rows touch at once: cold rows (most of them when RAM is full) are then read from
// the disk in parallel instead of one ~0.1-0.3 ms fault after another.
void Engine::ple_prefetch(const size_t *rows, size_t n) const {
    constexpr size_t RB = PLE_DIM / 32 * 34;
    for (size_t i = 0; i < n; i++) {
        const uintptr_t a = (uintptr_t)(ngram + rows[i] * RB), p0 = a & ~uintptr_t(4095), p1 = (a + RB - 1) & ~uintptr_t(4095);
        madvise((void *)p0, p1 - p0 + 4096, MADV_WILLNEED);
    }
}

void Engine::ple_rows(const Seq &s, const int *tokens, int T) {
    constexpr size_t RB = PLE_DIM / 32 * 34;
    std::vector<size_t> rows((size_t)T * PLE_HEADS);
    ple_ids(s.hist[0], s.hist[1], tokens, T, rows.data());
    ple_prefetch(rows.data(), rows.size());
    for (size_t i = 0; i < rows.size(); i++) memcpy(b.ple_rows + i * RB, ngram + rows[i] * RB, RB);
}

void Engine::ple_block_run(Seq &s, int T, bool verify) {
    const Layer &w = L[PLE_LAYER];
    constexpr size_t RB = PLE_DIM / 32 * 34;
    CK(hipMemcpyAsync(b.d_ple_rows, b.ple_rows, (size_t)T * PLE_HEADS * RB, hipMemcpyHostToDevice, st));
    if (T <= 8) {
        ple_embed(b.d_ple_rows, T, nullptr, nullptr, b.emb_q, st);
        gemv(w.ple_key, b.emb_q, T, b.key, HCD, GEMV_STORE, st);
        gemv(w.ple_value, b.emb_q, T, b.value, D, GEMV_STORE, st);
    } else {
        ple_embed(b.d_ple_rows, T, nullptr, b.emb_h, NO_Q, st);
        linear(w.ple_key, T, NO_Q, b.emb_h, D, b.key, HCD, GEMV_STORE);
        linear(w.ple_value, T, NO_Q, b.emb_h, D, b.value, D, GEMV_STORE);
    }
    ple_block(b.key, b.value, b.res, w.ple_nk, w.ple_nq, w.ple_nc, w.ple_conv, s.ple, b.xconv, T, st);
    if (!verify) ple_commit(s.ple, b.xconv, T, st);
}

// tokens (host) at positions s.n .. s.n+T-1 through the 48 layers; leaves the final residual in b.res
void Engine::forward(Seq &s, const int *tokens, int T, bool verify) {
    ensure_pages(s, s.n + T, s.n);
    ple_rows(s, tokens, T);
    CK(hipMemcpyAsync(b.tokens, tokens, T * 4, hipMemcpyHostToDevice, st));
    embed(embd, b.tokens, T, b.x, st);
    hc_init(b.x, b.res, T, st);
    for (int l = 0, gl = 0; l < NL; l++) {
        if (l == PLE_LAYER) ple_block_run(s, T, verify);
        hc_mix_run(*this, L[l].hca, b.res, T, b.inj_a);
        if (is_attn_layer(l)) attn_block(L[l], l / 4, s, T, s.n, b.blk);
        else gdn_block(L[l], gl++, s, T, verify, b.blk);
        hc_combine(b.res, b.inj_a, HC, b.blk, nullptr, T, st);
        hc_mix_run(*this, L[l].hcf, b.res, T, b.inj_f);
        moe_block(L[l], l, T, true, b.moe);
        hc_combine(b.res, b.inj_f, HC, b.moe, nullptr, T, st);
    }
    if (T <= 8 && !verify) gdn_conv_commit(s.conv, b.v_raw, T, 8 * QKV, NGDN, st);
}

// final mixer + LM head for rows [r0, r0+T) of b.res -> b.logits [T][VOCAB] (rows: vocabulary prefix)
static void head(Engine &e, const HcMix &hm, const float *res, int T, int rows) {
    hc_mix_run(e, hm, res, T, nullptr);
    if (T > 8) quant_act(e.b.mx, (int64_t)T * D, e.b.hq, e.st);
    QTensor w = e.out;
    w.rows = rows;
    for (int t0 = 0; t0 < T; t0 += 8)
        gemv(w, qoff(e.b.hq, (size_t)t0 * D), std::min(8, T - t0), e.b.logits + (size_t)t0 * VOCAB, VOCAB, GEMV_STORE, e.st);
}

void Engine::lm_head(const QAct &in, int T, int rows) {
    QTensor w = out;
    w.rows = rows;
    gemv(w, in, T, b.logits, VOCAB, GEMV_STORE, st);
}

// MTP block over pairs (token at pos0+t, residual of pos0+t-1): h_prev [T][HCD] -> b.mtp_res [T][HCD]; the draft
// logits of the last row come from its head mixer (left in b.mx / b.hq).
void Engine::mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *d_tokens) {
    // eh_proj input per (token, stream): [rmsnorm(e) * enorm ; rmsnorm(h_s) * hnorm_s]
    embed(embd, d_tokens, T, b.mtp_e, st);
    rmsnorm(b.mtp_e, enorm, T, D, D, D, b.mtp_e, nullptr, NO_Q, st);
    hc_norm(h_prev, hnorm, T, b.xn, nullptr, NO_Q, st);
    mtp_pack(b.mtp_e, b.xn, T, b.mtp_in, st);
    const int R = T * HC;
    if (R <= 8) {
        quant_act(b.mtp_in, (int64_t)R * 2 * D, b.mtp_in_q, st);
        gemv(eh_proj, b.mtp_in_q, R, b.mtp_res, D, GEMV_STORE, st);
    } else {
        to_half(b.mtp_in, b.mtp_in_h, (int64_t)R * 2 * D, st);
        linear(eh_proj, R, NO_Q, b.mtp_in_h, 2 * D, b.mtp_res, D, GEMV_STORE);
    }
    const Layer &w = M;
    hc_mix_run(*this, w.hca, b.mtp_res, T, b.inj_a);
    attn_block(w, NATT, s, T, pos0, b.blk);
    hc_combine(b.mtp_res, b.inj_a, HC, b.blk, nullptr, T, st);
    hc_mix_run(*this, w.hcf, b.mtp_res, T, b.inj_f);
    moe_block(w, NL, T, false, b.moe);
    hc_combine(b.mtp_res, b.inj_f, HC, b.moe, nullptr, T, st);
}

// Run the MTP block over the T tokens just processed at pos0.., pairing each with the previous position's residual,
// so the MTP layer's KV cache tracks the sequence. b.res holds this pass's residuals.
void Engine::mtp_catch_up(Seq &s, int T, int pos0) {
    // h_prev rows: h_last, then the residuals of tokens 0..T-2 (b.mtp_h staging lives in b.gate, free here)
    float *h = b.gate;
    CK(hipMemcpyAsync(h, s.h_last, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
    if (T > 1) CK(hipMemcpyAsync(h + HCD, b.res, (size_t)(T - 1) * HCD * 4, hipMemcpyDeviceToDevice, st));
    // the MTP block overwrites b.gate inside hc_mix_run: copy h out of the way first
    CK(hipMemcpyAsync(b.key, h, (size_t)T * HCD * 4, hipMemcpyDeviceToDevice, st));
    mtp_forward(s, T, pos0, b.key, b.tokens);
}

// ---- prefill and speculative generation --------------------------------------------------------

void Engine::prefill(int id, const int *tokens, int n) {
    if (!embd.q) FAIL("prefill before warmup(): the token embeddings are not loaded");
    Seq &s = seqs.at(id);
    s.pending = -1;
    cpu.arm();
    for (int done = 0; done < n;) {
        const int T = std::min(umax, n - done);
        forward(s, tokens + done, T, false);
        mtp_catch_up(s, T, s.n);
        if (done + T == n) {
            head(*this, hc_head, b.res + (size_t)(T - 1) * HCD, 1, VOCAB);
            topk(b.logits, 1, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
            s.top_v.resize(TOPK); s.top_i.resize(TOPK);
            CK(hipMemcpyAsync(s.top_v.data(), b.topv, TOPK * 4, hipMemcpyDeviceToHost, st));
            CK(hipMemcpyAsync(s.top_i.data(), b.topi, TOPK * 4, hipMemcpyDeviceToHost, st));
        }
        CK(hipMemcpyAsync(s.h_last, b.res + (size_t)(T - 1) * HCD, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
        s.hist[0] = T >= 2 ? tokens[done + T - 2] : s.hist[1];
        s.hist[1] = tokens[done + T - 1];
        s.n += T;
        done += T;
    }
    CK(hipStreamSynchronize(st));
    cpu.disarm();
}

namespace {
struct Dist { int n = 0; int id[TOPK]; float p[TOPK]; };

Dist make_dist(const float *v, const int *ids, const Sampling &sp, const std::vector<int> &seen) {
    std::pair<float, int> c[TOPK];
    for (int i = 0; i < TOPK; i++) {
        float x = v[i];
        if (sp.presence_penalty != 0 && std::find(seen.begin(), seen.end(), ids[i]) != seen.end()) x -= sp.presence_penalty;
        c[i] = {x, ids[i]};
    }
    std::sort(c, c + TOPK, [](auto &a, auto &b) { return a.first > b.first; });
    Dist d;
    if (sp.temperature <= 0) { d.n = 1; d.id[0] = c[0].second; d.p[0] = 1; return d; }
    const int k = std::max(1, std::min(sp.top_k > 0 ? sp.top_k : TOPK, TOPK));
    double sum = 0;
    for (int i = 0; i < k; i++) { d.p[i] = expf((c[i].first - c[0].first) / sp.temperature); d.id[i] = c[i].second; sum += d.p[i]; }
    d.n = k;
    double cum = 0;
    for (int i = 0; i < k; i++) {
        d.p[i] /= sum; cum += d.p[i];
        if (cum >= sp.top_p) { d.n = i + 1; break; }
    }
    int m = 0;
    for (int i = 0; i < d.n; i++) if (d.p[i] >= sp.min_p * d.p[0]) { d.p[m] = d.p[i]; d.id[m++] = d.id[i]; }
    d.n = m;
    sum = 0;
    for (int i = 0; i < d.n; i++) sum += d.p[i];
    for (int i = 0; i < d.n; i++) d.p[i] /= sum;
    return d;
}

int draw(const Dist &d, std::mt19937_64 &rng) {
    double u = std::uniform_real_distribution<double>(0, 1)(rng), c = 0;
    for (int i = 0; i < d.n; i++) { c += d.p[i]; if (u < c) return d.id[i]; }
    return d.id[d.n - 1];
}

float prob(const Dist &d, int tok) {
    for (int i = 0; i < d.n; i++) if (d.id[i] == tok) return d.p[i];
    return 0;
}

int draw_without(Dist d, int tok, std::mt19937_64 &rng) {
    double sum = 0;
    for (int i = 0; i < d.n; i++) { if (d.id[i] == tok) d.p[i] = 0; sum += d.p[i]; }
    if (sum <= 0) return d.id[0];
    for (int i = 0; i < d.n; i++) d.p[i] /= sum;
    return draw(d, rng);
}
}  // namespace

// One speculative step. Emits the pending token (if newly sampled), accepted drafts and one more token.
int Engine::generate(int id, const Sampling &sp, int *out, int max_out) {
    Seq &s = seqs.at(id);
    int nout = 0;
    if (s.pending < 0) {
        if (s.top_v.empty()) FAIL("generate: nothing to continue from");
        s.pending = draw(make_dist(s.top_v.data(), s.top_i.data(), sp, s.seen), s.rng);
        s.seen.push_back(s.pending);
        out[nout++] = s.pending;
        if (is_eos(s.pending)) return nout;
    }
    {   // its n-gram rows load while the MTP head drafts (normally already started at the end of the last step)
        size_t rows[PLE_HEADS];
        ple_ids(s.hist[0], s.hist[1], &s.pending, 1, rows);
        ple_prefetch(rows, PLE_HEADS);
    }
    cpu.arm();
    const int k = std::max(0, std::min(sp.n_draft, std::min(3, max_out - nout - 1)));
    const int q = s.n;
    ensure_pages(s, q + k + 1, q);

    // 1. draft k tokens with the MTP head (greedy, on the GPU)
    CK(hipMemcpyAsync(b.draft, &s.pending, 4, hipMemcpyHostToDevice, st));
    for (int i = 0; i < k; i++) {
        if (i == 0) mtp_forward(s, 1, q, s.h_last, b.draft);
        else {
            CK(hipMemcpyAsync(b.value, b.mtp_res, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
            mtp_forward(s, 1, q + i, b.value, b.draft + i);
        }
        hc_mix_run(*this, mtp_head, b.mtp_res, 1, nullptr);
        lm_head(b.hq, 1, DRAFT_VOCAB);
        argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1, st);
    }
    // 2. verify pending + drafts with the full model (the n-gram rows need the drafts on the host)
    const int T = k + 1;
    if (k > 0) {                                  // the drafts' n-gram rows need them on the host
        CK(hipMemcpyAsync(b.h_draft, b.draft, T * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
    } else b.h_draft[0] = s.pending;
    std::vector<int> drafts(b.h_draft, b.h_draft + T);
    forward(s, drafts.data(), T, k > 0);
    head(*this, hc_head, b.res, T, VOCAB);
    topk(b.logits, T, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
    CK(hipMemcpyAsync(b.h_topv, b.topv, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(b.h_topi, b.topi, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    // 3. MTP catch-up on the verified pairs (rejected rows get overwritten later)
    CK(hipMemcpyAsync(b.tokens, b.draft, T * 4, hipMemcpyDeviceToDevice, st));
    mtp_catch_up(s, T, q);
    CK(hipStreamSynchronize(st));
    cpu.disarm();
    const float *tv = b.h_topv;
    const int *ti = b.h_topi;

    // 4. accept drafts (rejection sampling against greedy drafts is lossless) and draw the next token
    int a = 0, next = -1;
    for (; a < k; a++) {
        const Dist d = make_dist(&tv[a * TOPK], &ti[a * TOPK], sp, s.seen);
        const int tok = drafts[a + 1];
        const float pa = prob(d, tok);
        if (std::uniform_real_distribution<float>(0, 1)(s.rng) < pa) {
            if (is_eos(tok)) { next = tok; break; }
            out[nout++] = tok; s.seen.push_back(tok);
            continue;
        }
        next = draw_without(d, tok, s.rng);
        break;
    }
    if (next < 0) next = draw(make_dist(&tv[a * TOPK], &ti[a * TOPK], sp, s.seen), s.rng);
    out[nout++] = next;
    s.seen.push_back(next);

    // 5. commit the recurrent states to the accepted prefix (pending + a drafts)
    if (k > 0) {
        GdnScanArgs sa{};
        sa.S = s.ssm; sa.S_layer = (size_t)GV * GD * GD;
        sa.kn = b.v_kn; sa.in_layer = (size_t)8 * GK * GD;
        sa.vc = b.v_vc; sa.v_layer = (size_t)8 * GV * GD;
        sa.ab = b.v_ab; sa.ab_layer = (size_t)8 * 2 * GV;
        sa.A = gdn_A; sa.dtb = gdn_dt; sa.T = a + 1; sa.write_state = 1; sa.out = nullptr;
        gdn_scan(sa, NGDN, st);
        gdn_conv_commit(s.conv, b.v_raw, a + 1, 8 * QKV, NGDN, st);
        ple_commit(s.ple, b.xconv, a + 1, st);
    }
    CK(hipMemcpyAsync(s.h_last, b.res + (size_t)a * HCD, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
    // history: the last two tokens before position q + a + 1
    if (a >= 1) { s.hist[0] = drafts[a - 1]; s.hist[1] = drafts[a]; }
    else { s.hist[0] = s.hist[1]; s.hist[1] = drafts[0]; }
    s.n = q + a + 1;
    s.pending = next;
    {   // the next step reads this token's n-gram rows first: start fetching them now
        size_t rows[PLE_HEADS];
        ple_ids(s.hist[0], s.hist[1], &next, 1, rows);
        ple_prefetch(rows, PLE_HEADS);
    }
    return nout;
}
