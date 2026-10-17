// Engine implementation: sequences, snapshots, forward passes, MTP drafting, speculative steps.
#include "model.h"
#include <chrono>
#include <thread>
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

// Pinned host memory for a snapshot slot: hugepage anonymous memory registered with the GPU (hipHostMalloc pins page by
// page, ~0.3 s for a slot while the cold experts hold most of RAM; this takes a few ms)
void *Engine::snap_buffer() {
    const size_t n = (state_bytes() + (2 << 20) - 1) & ~size_t((2 << 20) - 1);
    const auto t0 = std::chrono::steady_clock::now();
    void *p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) FAIL("cannot allocate a snapshot buffer");
    madvise(p, n, MADV_HUGEPAGE);
    memset(p, 0, n);
    const auto t1 = std::chrono::steady_clock::now();
    CK(hipHostRegister(p, n, hipHostRegisterDefault));
    const auto t2 = std::chrono::steady_clock::now();
    if (getenv("KILN_TRACE"))
        fprintf(stderr, "kiln-fn: snapshot buffer %zu MB: fault-in %.0f ms, register %.0f ms\n", n >> 20,
                std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
    return p;
}

void *Engine::take_snap_buffer() {
    if (snap_spare.empty()) return snap_buffer();
    void *p = snap_spare.back();
    snap_spare.pop_back();
    return p;
}

// Buffers for n snapshots in all: the missing ones in one region, faulted in by a few threads and pinned in one call.
// Small pages: with MADV_HUGEPAGE, faulting in 1.9 GB right after a load took ~1 s of compaction (pinning more pages costs
// ~0.2 s instead).
void Engine::reserve_snapshots(int n) {
    snap_target = std::max(snap_target, n);
    if (snap_warm) { snap_reserve_start(); snap_reserve_finish(); }
}

void Engine::snap_reserve_start() {
    int have = (int)snap_spare.size();
    for (const Snap &sn : snaps) have += sn.host != nullptr;
    const int k = snap_target - have;
    if (k <= 0 || snap_pending) return;
    const size_t one = (state_bytes() + (2 << 20) - 1) & ~size_t((2 << 20) - 1), all = one * k;
    void *p = mmap(nullptr, all, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) FAIL("cannot allocate %d snapshot buffers", k);
    madvise(p, all, MADV_NOHUGEPAGE);
    snap_pending = p; snap_pending_n = k;
    const int nt = std::min(k, 4);
    for (int t = 0; t < nt; t++)
        snap_fill.emplace_back([=] { const size_t a = (size_t)k * t / nt * one, b = (size_t)k * (t + 1) / nt * one; memset((char *)p + a, 0, b - a); });
}

void Engine::snap_reserve_finish() {
    const auto t0 = std::chrono::steady_clock::now();
    for (std::thread &t : snap_fill) t.join();
    snap_fill.clear();
    if (!snap_pending) return;
    const size_t one = (state_bytes() + (2 << 20) - 1) & ~size_t((2 << 20) - 1), all = one * snap_pending_n;
    const auto t1 = std::chrono::steady_clock::now();
    CK(hipHostRegister(snap_pending, all, hipHostRegisterDefault));
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < snap_pending_n; i++) snap_spare.push_back((char *)snap_pending + (size_t)i * one);
    snap_regions.push_back({snap_pending, all});
    if (getenv("KILN_TRACE"))
        fprintf(stderr, "kiln-fn: %d snapshot buffers (%zu MB): fault-in wait %.0f ms, register %.0f ms\n", snap_pending_n,
                all >> 20, std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
    snap_pending = nullptr; snap_pending_n = 0;
}

int Engine::snapshot(int id) {
    Seq &s = seqs.at(id);
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    if (!sn.host) sn.host = take_snap_buffer();
    sn.used = true;
    sn.n = s.n;
    sn.hist[0] = s.hist[0]; sn.hist[1] = s.hist[1];
    if (s.pending < 0) { sn.top_v = s.top_v; sn.top_i = s.top_i; } else { sn.top_v.clear(); sn.top_i.clear(); }
    sn.pages.assign(s.pages.begin(), s.pages.begin() + (s.n + PAGE - 1) / PAGE);
    for (int p : sn.pages) page_ref[p]++;
    CK(hipStreamWaitEvent(st, copied, 0));
    size_t o = 0;
    CK(hipMemcpyAsync(stage_state + o, s.ssm, ssm_bytes(), hipMemcpyDeviceToDevice, st)); o += ssm_bytes();
    CK(hipMemcpyAsync(stage_state + o, s.conv, conv_bytes(), hipMemcpyDeviceToDevice, st)); o += conv_bytes();
    CK(hipMemcpyAsync(stage_state + o, s.ple, ple_bytes(), hipMemcpyDeviceToDevice, st)); o += ple_bytes();
    CK(hipMemcpyAsync(stage_state + o, s.h_last, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st)); o += (size_t)HCD * 4;
    CK(hipMemcpyAsync(stage_state + o, s.idx_tail, tail_bytes(), hipMemcpyDeviceToDevice, st));
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
    s.top_v = sn.top_v; s.top_i = sn.top_i; s.seen.clear();
    const uint8_t *h = (const uint8_t *)sn.host;
    size_t o = 0;
    CK(hipMemcpyAsync(s.ssm, h + o, ssm_bytes(), hipMemcpyHostToDevice, st)); o += ssm_bytes();
    CK(hipMemcpyAsync(s.conv, h + o, conv_bytes(), hipMemcpyHostToDevice, st)); o += conv_bytes();
    CK(hipMemcpyAsync(s.ple, h + o, ple_bytes(), hipMemcpyHostToDevice, st)); o += ple_bytes();
    CK(hipMemcpyAsync(s.h_last, h + o, (size_t)HCD * 4, hipMemcpyHostToDevice, st)); o += (size_t)HCD * 4;
    CK(hipMemcpyAsync(s.idx_tail, h + o, tail_bytes(), hipMemcpyHostToDevice, st));
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
    } else if (W.type == F16) {
        gemm((const half *)W.f, W.rows, W.cols, hin, ldin, T, out, ldo, out_f16, mode == GEMV_ADD);
    } else {                                      // dequantized to fp16 in row blocks that fit b.deq
        const int rb = std::max(32, (int)(DEQ_ELEMS / W.cols) / 32 * 32);
        for (int r0 = 0; r0 < W.rows; r0 += rb) {
            QTensor Wb = W;
            Wb.rows = std::min(rb, W.rows - r0);
            qtensor_rows(Wb, W, r0);
            dequant(Wb, b.deq, st);
            void *o = out_f16 ? (void *)((half *)out + r0) : (void *)((float *)out + r0);
            gemm(b.deq, Wb.rows, W.cols, hin, ldin, T, o, ldo, out_f16, mode == GEMV_ADD);
        }
    }
}

// hyper-connection mixer: residual res [T][HC][D] -> block input b.mx (fp32), b.hq (int8, decode) / b.hh (fp16),
// and the stream weights `inject` [T][HC] when the mixer has them
static void hc_mix_run(Engine &e, const HcMix &m, const float *res, int T, float *inject) {
    auto &b = e.b;
    if (T <= 4) {                                 // decode: two fused kernels
        hc_mix_fused(res, m.norm, m.down, m.up, inject ? &m.inject : nullptr, T, b.lo, inject, HC, b.inv, b.mx, b.hh, b.hq, e.st);
        return;
    }
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

void Engine::attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *res, float *inject, bool spec) {
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
    idx_prep(b.kidx, b.qidx, w.idx_qn, w.idx_kn, T, pos0, kv(s), kvl, s.idx_tail + (size_t)kvl * IDX_TAIL,
             b.idx_raw + (size_t)kvl * umax * IDX_D, !spec, b.qidx16, st);
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
            idx_select(b.qidx16 + (size_t)g0 * IDX_H * IDX_D, pos0 + g0, G, nblk, kv(s), kvl, b.score, b.sel_hist,
                       b.sel + (size_t)g0 * (SEL_BLOCKS + 1), st);
        }
        attention_sel(a, b.sel, small ? nullptr : b.att_h, small ? b.mq : NO_Q, st);
    }
    if (small) gemv_hc(w.o, b.mq, T, res, inject, st);
    else {
        linear(w.o, T, NO_Q, b.att_h, NH * HD, b.blk, D, GEMV_STORE);
        hc_combine(res, inject, HC, b.blk, nullptr, T, st);
    }
}

void Engine::gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *res, float *inject) {
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
        gemv_hc(w.ssm_out, b.mq, T, res, inject, st);
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
    linear(w.ssm_out, T, NO_Q, b.att_h, GV * GD, b.blk, D, GEMV_STORE);
    hc_combine(res, inject, HC, b.blk, nullptr, T, st);
}

// Routed experts + shared expert for T tokens of b.mx / b.hq. Decode path, also used (8 tokens at a time) by
// prefill until the grouped GEMM path exists: hot picks on the GPU, cold picks on the CPU (cpu_cold).
void Engine::moe_block(const Layer &w, int layer, int T, bool cpu_cold, float *res, float *inject) {
    if (T > 8) {                                  // a pass of one sub-chunk (the MTP block's catch-up)
        moe_front(w, T, 0);
        moe_back(w, layer, T);
        hc_combine(res, inject, HC, b.moe, nullptr, T, st);
        return;
    }
    const float gf = cpu_cold && w.n_cold ? gfrac : 0.0f;
    router_route_dec(w.router, w.sg_w, b.mx, T, b.logits_e, b.sg, w.slot_of, b.ids, b.wts, b.hot, cpu_cold ? d_sh : nullptr, gf,
                     layer, gen + 1, b.rr_counter, st, route_dec);
    if (cpu_cold) gen++;
    if (cpu_cold && layer + 1 < NL && pf_on) {    // the next layer's weights into the Infinity Cache meanwhile (2-3x faster reads)
        CK(hipEventRecord(routed, st));
        CK(hipStreamWaitEvent(pst, routed, 0));
        prefetch(pf_ranges + pf_off[layer + 1], pf_n[layer + 1], pf_sink, pst);
    }
    moe_dec(w.hot_gate, w.hot_up, w.hot_down, w.sh_gate, w.sh_up, w.sh_down, gf > 0 ? &cold_dev[layer] : nullptr, b.hot, T, b.hq,
            b.hq_e, b.wts, b.sg, b.y, st);                // hot picks, shared expert and the GPU's share of the cold picks
    moe_finish(cpu_cold ? &d_sh->done : nullptr, gen, b.y, b.hot, b.y + (size_t)T * TOPK_E * D, nullptr, cpu_cold ? d_sh->y : nullptr,
               inject, res, T, st);
}

// Cost model of a pass's cold experts (per layer), with this machine's rates measured at load (Engine::calibrate): the
// CPU computes an expert in place (bandwidth-bound for experts with few tokens, compute-bound for busy ones), the GPU
// first copies it to VRAM (DMA) and then computes it for nothing. The rates are fixed for the session, so the split (and
// with it the output) depends only on the routing.
namespace {
constexpr double GPU_PER_EXPERT = 4e-6;          // the GPU's own cost per staged expert (relayout and GEMM launches)
constexpr double EXPERT_MAC = 3.0 * FE * D;
}

static double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void Engine::pmark(int phase) {
    if (!prof) return;
    CK(hipStreamSynchronize(st));
    const double t = wall();
    if (phase >= 0) prof_t[phase] += t - prof_last;
    prof_last = t;
}
static const char *PROF_NAMES[12] = {"embed+ple rows", "token mixers", "moe front", "moe host", "moe hot", "moe cold gpu",
                                     "moe cpu wait", "moe combine", "mtp catch-up", "head", "", ""};

// MoE of a pass, part 1, for its sub-chunk of n tokens at c0 (after the sub-chunk's ffn mixer: inputs in b.mx / b.hh):
// router picks into b.idsT / b.wtsT and the shared-expert gate into b.sgT at row c0, the inputs kept in b.hmoe, and b.moe
// started with the gated shared expert.
void Engine::moe_front(const Layer &w, int n, int c0) {
    half *x = b.hmoe + (size_t)c0 * D;
    CK(hipMemcpyAsync(x, b.hh, (size_t)n * D * 2, hipMemcpyDeviceToDevice, st));
    gemm(w.router, NE, D, x, D, n, b.rlog, NE, false, false);
    route_many(b.rlog, w.sg_w, b.mx, n, b.idsT + (size_t)c0 * TOPK_E, b.wtsT + (size_t)c0 * TOPK_E, b.sgT + c0, st);
    // the sub-chunk's picks (and inputs, for the CPU's experts) go to the host on the copy stream while the next
    // sub-chunks run, so moe_back only waits for the last one
    CK(hipEventRecord(front_ready, st));
    CK(hipStreamWaitEvent(cst, front_ready, 0));
    CK(hipMemcpyAsync(b.h_ids + (size_t)c0 * TOPK_E, b.idsT + (size_t)c0 * TOPK_E, (size_t)n * TOPK_E * 4, hipMemcpyDeviceToHost, cst));
    CK(hipMemcpyAsync(b.h_wts + (size_t)c0 * TOPK_E, b.wtsT + (size_t)c0 * TOPK_E, (size_t)n * TOPK_E * 4, hipMemcpyDeviceToHost, cst));
    if (w.n_cold) CK(hipMemcpyAsync(b.h_x16 + (size_t)c0 * D, x, (size_t)n * D * 2, hipMemcpyDeviceToHost, cst));
    half *g = b.att_h, *u = b.att_h + (size_t)n * FS;           // shared expert as GEMMs: gate | up -> silu * up -> down
    linear(w.sh_gate, n, NO_Q, x, D, g, FS, GEMV_STORE, true);
    linear(w.sh_up, n, NO_Q, x, D, u, FS, GEMV_STORE, true);
    swiglu_h(g, u, (int64_t)n * FS, st);
    linear(w.sh_down, n, NO_Q, g, FS, b.blk, D, GEMV_STORE);
    moe_shared(b.blk, b.sgT + c0, n, b.moe + (size_t)c0 * D, st);
}

// MoE of a pass, part 2: the routed experts of all T tokens, added into b.moe. Picks are sorted by expert on the host (one
// sync per layer), so each expert's weights are read once per pass, however long the pass. Hot experts run as grouped
// WMMA GEMMs from VRAM. Of the cold experts with picks, the CPU workers take the ones with the fewest tokens and compute
// them where they are, while the GPU copies the rest into a two-slot VRAM staging ring (copy stream), relayouts them and
// runs the same GEMMs. Picks go in batches of at most smax (the gather / output buffers); a batch's outputs are added to
// its tokens' rows of b.moe in router order, so the result does not depend on how the work was split.
void Engine::moe_back(const Layer &w, int layer, int T) {
    const int N = T * TOPK_E, P = smax;
    CK(hipEventRecord(front_copied, cst));        // the copies moe_front queued for this pass's sub-chunks
    CK(hipEventSynchronize(front_copied));
    const std::vector<int> &slot_of = w.slot_h;
    std::vector<int> cnt(NE, 0);
    for (int j = 0; j < N; j++) cnt[b.h_ids[j]]++;
    if (!route_counts.empty() && layer < NL) for (int e = 0; e < NE; e++) route_counts[(size_t)layer * NE + e] += cnt[e];
    // cold experts with picks, fewest tokens first: the CPU takes a prefix (at most P picks, its output buffer), the GPU the rest
    std::vector<int> ce, hot;
    for (int e = 0; e < NE; e++) if (cnt[e]) (slot_of[e] >= 0 ? hot : ce).push_back(e);
    std::stable_sort(ce.begin(), ce.end(), [&](int a, int c) { return cnt[a] < cnt[c]; });
    const ColdLayer *cl = w.n_cold ? &cold[layer] : nullptr;
    auto cslot = [&](int e) { return -1 - slot_of[e]; };
    auto ebytes = [&](int e) { return 2.0 * FE * cl->row_gu[cl->grp(cslot(e))] + (double)D * cl->row_dn; };
    int kc = 0, kmax = 0;
    {
        double best = 1e30, macs = 0, cb = 0, gb = 0;
        for (int e : ce) gb += ebytes(e);
        const int K = (int)ce.size();
        for (int k = 0, picks = 0; k <= K; k++) {
            const double ta = cb / cal_cpu_bw, tm = macs / cal_cpu_mac;
            const double t = std::max(cpu_bias * (std::max(ta, tm) + 0.25 * std::min(ta, tm)), gb / cal_dma_bw + (K - k) * GPU_PER_EXPERT);
            if (t < best) { best = t; kc = k; }
            kmax = k;
            if (k == K || picks + cnt[ce[k]] > P) break;
            picks += cnt[ce[k]]; macs += cnt[ce[k]] * EXPERT_MAC; cb += ebytes(ce[k]); gb -= ebytes(ce[k]);
        }
    }
    if (split >= 0) kc = split ? kmax : 0;
    std::vector<int> gpu_cold(ce.begin() + kc, ce.end()), cpu_cold(ce.begin(), ce.begin() + kc);
    std::sort(hot.begin(), hot.end(), [&](int a, int c) { return slot_of[a] < slot_of[c]; });
    std::sort(gpu_cold.begin(), gpu_cold.end(), [&](int a, int c) { return slot_of[a] > slot_of[c]; });   // cold slots ascending:
                                                                                                         // format group 0 first
    std::stable_sort(cpu_cold.begin(), cpu_cold.end(), [&](int a, int c) { return cnt[a] > cnt[c]; });   // largest first
    // pick order: hot experts, the GPU's cold experts, the CPU's; tokens ascending within an expert
    std::vector<int> off(NE, 0);
    {
        int o = 0;
        for (const auto *v : {&hot, &gpu_cold, &cpu_cold}) for (int e : *v) { off[e] = o; o += cnt[e]; }
        std::vector<int> cur = off;
        for (int j = 0; j < N; j++) {
            const int e = b.h_ids[j], p = cur[e]++;
            b.h_tok[p] = j / TOPK_E; b.h_entw[p] = b.h_wts[j]; b.h_pos[j] = p;
        }
    }
    // Picks go through the gather / output buffers (P rows) in ranges: a range is gathered once and its outputs added to
    // b.moe once. A range holds groups that run together: all its hot experts, or staging groups of the GPU's cold
    // experts (at most STAGE_G, one format: one DMA + relayout each). Work items are tiles of at most MOE_ITEM of an expert's
    // tokens (rows relative to the range), so a busy expert spreads over many workgroups.
    struct Range { int p0, p1, t0, t1; };
    struct Group { int range, i0, i1, w0, w1; };
    std::vector<Range> ranges;
    std::vector<Group> hot_g, gpu_g;
    int nw = 0, nt = 0;
    std::vector<int> mark(T, -1);
    auto close_range = [&](Range &r) {
        r.t0 = nt;
        const int id = (int)ranges.size();
        for (int p = r.p0; p < r.p1; p++) { const int t = b.h_tok[p]; if (mark[t] != id) { mark[t] = id; b.h_btok[nt++] = t; } }
        r.t1 = nt;
        ranges.push_back(r);
    };
    // experts ex[i0..i1) in groups; max_g experts per group, a new group when the format group changes (cold)
    auto plan = [&](std::vector<Group> &gs, const std::vector<int> &ex, bool hot_slots, int max_g, int cap) {
        Range r{-1, -1, 0, 0};
        Group g{-1, 0, 0, 0, 0};
        auto close_group = [&](int i) { if (g.i1 > g.i0) gs.push_back(g); g = Group{(int)ranges.size(), i, i, nw, nw}; };
        for (int i = 0; i < (int)ex.size(); i++) {
            const int e = ex[i], p0 = off[e], n = cnt[e];
            const bool new_range = r.p0 < 0 || p0 + n - r.p0 > cap;
            const bool new_group = new_range || g.i1 - g.i0 >= max_g || (!hot_slots && cl->grp(cslot(e)) != cl->grp(cslot(ex[g.i0])));
            if (new_range && r.p0 >= 0) { close_group(i); close_range(r); }
            if (new_range) { r = Range{p0, p0, 0, 0}; g = Group{(int)ranges.size(), i, i, nw, nw}; }
            else if (new_group) close_group(i);
            const int sl = hot_slots ? slot_of[e] : i - g.i0;
            for (int t0 = 0; t0 < n; t0 += MOE_ITEM) b.h_work[nw++] = {sl, p0 - r.p0 + t0, std::min(MOE_ITEM, n - t0), 0};
            g.i1 = i + 1; g.w1 = nw; r.p1 = p0 + n;
        }
        if (r.p0 >= 0) { close_group((int)ex.size()); close_range(r); }
    };
    // Hot and GPU-cold ranges in separate halves of the buffers, so the hot experts' GEMMs can run between the cold
    // groups while their DMA streams (the copy engine is the prefill's bottleneck); one expert per range at most P / 2
    int max_cnt = 0;
    for (const auto *v : {&hot, &gpu_cold}) for (int e : *v) max_cnt = std::max(max_cnt, cnt[e]);
    const bool halves = max_cnt <= P / 2 && !hot.empty() && !gpu_cold.empty();
    plan(hot_g, hot, true, NE, halves ? P / 2 : P);
    plan(gpu_g, gpu_cold, false, STAGE_G, halves ? P / 2 : P);
    half *X0 = b.Xs, *H0 = b.Hs, *Y0 = b.Ys;                  // hot
    half *X1 = halves ? b.Xs + (size_t)(P / 2) * D : b.Xs, *H1 = halves ? b.Hs + (size_t)(P / 2) * FE : b.Hs;
    half *Y1 = halves ? b.Ys + (size_t)(P / 2) * D : b.Ys;   // GPU cold
    const int pc0 = cpu_cold.empty() ? N : off[cpu_cold[0]];
    if (!cpu_cold.empty()) { Range r{pc0, N, 0, 0}; close_range(r); }   // the CPU's picks: one range (at most P)
    if (!cpu_cold.empty()) {                      // the CPU starts right away
        j_slot.clear(); j_first.assign(1, 0);
        j_tok.assign(b.h_tok + pc0, b.h_tok + N); j_w.assign(b.h_entw + pc0, b.h_entw + N);
        for (int e : cpu_cold) { j_slot.push_back(cslot(e)); j_first.push_back(off[e] - pc0 + cnt[e]); }
        MoeJob job;
        job.layer = layer; job.T = T; job.n_exp = (int)cpu_cold.size(); job.x = b.h_x16;
        job.slot = j_slot.data(); job.first = j_first.data(); job.tok = j_tok.data(); job.w = j_w.data(); job.out = b.h_cout;
        cpu.submit(job);
    }
    pmark(3);
    CK(hipMemcpyAsync(b.ent_tok, b.h_tok, (size_t)pc0 * 4, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(b.ent_w, b.h_entw, (size_t)pc0 * 4, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(b.pos, b.h_pos, (size_t)N * 4, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(b.work, b.h_work, (size_t)nw * sizeof(MoeWork), hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(b.btok, b.h_btok, (size_t)nt * 4, hipMemcpyHostToDevice, st));
    QTensor none{};
    auto gemms = [&](const Group &g, int w0, int w1, const QTensor &G, const QTensor &U, const QTensor &Dn, half *X, half *H, half *Y) {
        moe_gemm_gateup(G, U, none, none, b.work + w0, w1 - w0, X, H, st);
        moe_gemm_down(Dn, none, b.work + w0, w1 - w0, H, b.ent_w + ranges[g.range].p0, Y, st);
    };
    auto gather = [&](int ri, half *X) { const Range &r = ranges[ri]; gather_rows(b.hmoe, b.ent_tok + r.p0, r.p1 - r.p0, X, st); };
    auto accum = [&](int ri, half *Y) { const Range &r = ranges[ri]; moe_accum(Y, b.pos, r.p0, r.p1, b.btok + r.t0, r.t1 - r.t0, b.moe, st); };
    // staging slot k: gate | up regions of stage_gu bytes each (the group's packed GGUF experts), then down
    auto dma = [&](int gi) {                      // runs of consecutive slots in one copy
        const Group &G = gpu_g[gi];
        const int k = gi % NSLOT, fg = cl->grp(cslot(gpu_cold[G.i0])), base = fg ? cl->n0 : 0;
        const size_t gsz = (size_t)FE * cl->row_gu[fg], dsz = (size_t)D * cl->row_dn;
        CK(hipStreamWaitEvent(cst, raw_free[k], 0));
        uint8_t *r = b.raw[k];
        for (int c = 0; c < G.i1 - G.i0;) {
            const int s0 = cslot(gpu_cold[G.i0 + c]);
            int run_n = 1;
            while (c + run_n < G.i1 - G.i0 && cslot(gpu_cold[G.i0 + c + run_n]) == s0 + run_n) run_n++;
            CK(hipMemcpyAsync(r + c * gsz, cl->gate[fg] + (s0 - base) * gsz, run_n * gsz, hipMemcpyHostToDevice, cst));
            CK(hipMemcpyAsync(r + stage_gu + c * gsz, cl->up[fg] + (s0 - base) * gsz, run_n * gsz, hipMemcpyHostToDevice, cst));
            CK(hipMemcpyAsync(r + 2 * stage_gu + c * dsz, cl->down + s0 * dsz, run_n * dsz, hipMemcpyHostToDevice, cst));
            c += run_n;
        }
        CK(hipEventRecord(raw_ready[k], cst));
    };
    // the hot work items in slices, one before each cold group (with halves) or all before the first
    int nwh = 0;
    for (const Group &g : hot_g) nwh += g.w1 - g.w0;
    const int slots = halves ? (int)gpu_g.size() : 1;
    auto hot_slice = [&](int s) {                 // hot items [nwh * s / slots, nwh * (s + 1) / slots) in item order
        const int a = (int)((int64_t)nwh * s / slots), z = (int)((int64_t)nwh * (s + 1) / slots);
        for (int i = 0, base = 0; i < (int)hot_g.size(); base += hot_g[i].w1 - hot_g[i].w0, i++) {
            const Group &g = hot_g[i];
            const int lo = std::max(a, base), hi = std::min(z, base + g.w1 - g.w0);
            if (lo >= hi) continue;
            if (lo == base) gather(g.range, X0);
            gemms(g, g.w0 + lo - base, g.w0 + hi - base, w.hot_gate, w.hot_up, w.hot_down, X0, H0, Y0);
            if (hi == base + g.w1 - g.w0) accum(g.range, Y0);
        }
    };
    for (int gi = 0; gi < std::min((int)gpu_g.size(), NSLOT - 1); gi++) dma(gi);
    if (gpu_g.empty()) hot_slice(0);
    for (size_t gi = 0; gi < gpu_g.size(); gi++) {
        const Group &g = gpu_g[gi];
        if (gi + NSLOT - 1 < gpu_g.size()) dma((int)gi + NSLOT - 1);   // the copy engine runs ahead while the GPU computes
        if ((int)gi < slots) hot_slice((int)gi);
        if (gi == 0) pmark(4);
        if (gi == 0 || gpu_g[gi - 1].range != g.range) gather(g.range, X1);
        const int k = gi % NSLOT, ns = g.i1 - g.i0, fmt = cl->type_gu[cl->grp(cslot(gpu_cold[g.i0]))];
        CK(hipStreamWaitEvent(st, raw_ready[k], 0));
        Arena ag, au;                             // the SoA arrays of this format over the slot's staging regions
        ag.base = b.sgu[k]; au.base = b.sgu[k] + stage_soa; ag.cap = au.cap = stage_soa;
        const QTensor G = make_qtensor(ag, fmt, ns * FE, D), U = make_qtensor(au, fmt, ns * FE, D);
        QTensor Dn = b.sd_[k];
        Dn.rows = ns * D;
        relayout(G, b.raw[k], st, 0, (int64_t)ns * FE * D / 256);
        relayout(U, b.raw[k] + stage_gu, st, 0, (int64_t)ns * FE * D / 256);
        relayout(Dn, b.raw[k] + 2 * stage_gu, st, 0, (int64_t)ns * D * FE / 32);
        CK(hipEventRecord(raw_free[k], st));
        gemms(g, g.w0, g.w1, G, U, Dn, X1, H1, Y1);
        if (gi + 1 == gpu_g.size() || gpu_g[gi + 1].range != g.range) accum(g.range, Y1);
    }
    pmark(5);
    if (!cpu_cold.empty()) {                      // the CPU's outputs into Ys, then the same accumulation (the last range)
        if (!gpu_cold.empty()) CK(hipEventRecord(moe_gpu_done, st));
        cpu.wait();
        // balance feedback: if the GPU's share is already done when the CPU's is, the CPU got too much (and the reverse);
        // the split's CPU estimates scale by cpu_bias, which settles where both finish together. It runs for the first
        // passes after a load only: a split that keeps following timing would make results depend on history (the CPU
        // and the GPU round differently), and a settled bias is as fast (1.5-2x the estimate: within 2% everywhere).
        if (!gpu_cold.empty() && split < 0 && adapt && adapt_left > 0) {
            cpu_bias = std::clamp(cpu_bias * (hipEventQuery(moe_gpu_done) == hipSuccess ? 1.03 : 1 / 1.03), 0.1, 10.0);
            if (--adapt_left == 0) fprintf(stderr, "kiln-fn: prefill split settled (CPU cost x%.2f)\n", cpu_bias);
        }
        pmark(6);
        CK(hipMemcpyAsync(b.Ys, b.h_cout, (size_t)(N - pc0) * D * 2, hipMemcpyHostToDevice, st));
        accum((int)ranges.size() - 1, b.Ys);
    }
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
    auto copy = [&](size_t i0, size_t i1) {
        ple_prefetch(rows.data() + i0, i1 - i0);
        for (size_t i = i0; i < i1; i++) memcpy(b.ple_rows + i * RB, ngram + rows[i] * RB, RB);
    };
    if (T <= 8) { copy(0, rows.size()); return; }
    // long passes: 4 threads, so the reads of rows that are not in the page cache overlap
    std::thread th[4];
    for (int k = 0; k < 4; k++) th[k] = std::thread(copy, rows.size() * k / 4, rows.size() * (k + 1) / 4);
    for (auto &t : th) t.join();
}

// the n-gram block for the n tokens of a pass at c0 (their rows from ple_rows); res: their residual streams
void Engine::ple_block_run(Seq &s, int c0, int n, bool verify, float *res) {
    const Layer &w = L[PLE_LAYER];
    constexpr size_t RB = PLE_DIM / 32 * 34;
    CK(hipMemcpyAsync(b.d_ple_rows, b.ple_rows + (size_t)c0 * PLE_HEADS * RB, (size_t)n * PLE_HEADS * RB, hipMemcpyHostToDevice, st));
    if (n <= 8) {
        ple_embed(b.d_ple_rows, n, nullptr, nullptr, b.emb_q, st);
        gemv(w.ple_key, b.emb_q, n, b.key, HCD, GEMV_STORE, st);
        gemv(w.ple_value, b.emb_q, n, b.value, D, GEMV_STORE, st);
    } else {
        ple_embed(b.d_ple_rows, n, nullptr, b.emb_h, NO_Q, st);
        linear(w.ple_key, n, NO_Q, b.emb_h, D, b.key, HCD, GEMV_STORE);
        linear(w.ple_value, n, NO_Q, b.emb_h, D, b.value, D, GEMV_STORE);
    }
    ple_block(b.key, b.value, res, w.ple_nk, w.ple_nq, w.ple_nc, w.ple_conv, s.ple, b.xconv, n, st);
    if (!verify) ple_commit(s.ple, b.xconv, n, st);
}

// A pass: tokens (host) at positions s.n .. s.n+T-1 (T <= smax) through the 48 layers, layer by layer. Within a layer,
// the token mixer and the mixers run over sub-chunks of at most umax tokens in order (attention and the recurrent states
// see every earlier token), then the routed experts once for all T. Leaves the final residual in b.res.
void Engine::forward(Seq &s, const int *tokens, int T, bool verify) {
    ensure_pages(s, s.n + T, s.n);
    if (T > 8) pmark(-1);
    // the n-gram rows are needed at layer 1: long passes gather them while layer 0 runs
    std::thread ple_th;
    if (T > 8) ple_th = std::thread([this, &s, tokens, T] { ple_rows(s, tokens, T); });
    else ple_rows(s, tokens, T);
    CK(hipMemcpyAsync(b.tokens, tokens, T * 4, hipMemcpyHostToDevice, st));
    const int nc = (T + umax - 1) / umax;         // sub-chunks as even as possible: none is decode-sized when T > 8
    auto cb = [&](int c) { return (int)((int64_t)T * c / nc); };
    for (int c = 0; c < nc; c++) {
        const int c0 = cb(c), n = cb(c + 1) - c0;
        embed(embd, b.tokens + c0, n, b.x, st);
        hc_init(b.x, b.res + (size_t)c0 * HCD, n, st);
    }
    if (T > 8) pmark(0);
    for (int l = 0, gl = 0; l < NL; l++) {
        const Layer &w = L[l];
        for (int c = 0; c < nc; c++) {
            const int c0 = cb(c), n = cb(c + 1) - c0;
            float *res = b.res + (size_t)c0 * HCD, *inj = b.inj_f + (size_t)c0 * HC;
            if (l == PLE_LAYER) {
                if (ple_th.joinable()) ple_th.join();
                ple_block_run(s, c0, n, verify, res);
            }
            hc_mix_run(*this, w.hca, res, n, b.inj_a);
            if (is_attn_layer(l)) attn_block(w, l / 4, s, n, s.n + c0, res, b.inj_a, verify);
            else gdn_block(w, gl, s, n, verify, res, b.inj_a);
            hc_mix_run(*this, w.hcf, res, n, inj);
            if (T <= 8) moe_block(w, l, n, true, res, inj);
            else { if (prof) pmark(1); moe_front(w, n, c0); if (prof) pmark(2); }
        }
        if (!is_attn_layer(l)) gl++;
        if (T > 8) {
            moe_back(w, l, T);
            hc_combine(b.res, b.inj_f, HC, b.moe, nullptr, T, st);
            pmark(7);
        }
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
void Engine::mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *d_tokens, bool spec) {
    // eh_proj input per (token, stream): [rmsnorm(e) * enorm ; rmsnorm(h_s) * hnorm_s]
    embed(embd, d_tokens, T, b.mtp_e, st);
    rmsnorm(b.mtp_e, enorm, T, D, D, D, b.mtp_e, nullptr, NO_Q, st);
    hc_norm(h_prev, hnorm, T, b.xn, nullptr, NO_Q, st);
    const int R = T * HC;
    if (R <= 8) {
        mtp_pack(b.mtp_e, b.xn, T, b.mtp_in, nullptr, st);
        quant_act(b.mtp_in, (int64_t)R * 2 * D, b.mtp_in_q, st);
        gemv(eh_proj, b.mtp_in_q, R, b.mtp_res, D, GEMV_STORE, st);
    } else {
        mtp_pack(b.mtp_e, b.xn, T, nullptr, b.mtp_in_h, st);
        linear(eh_proj, R, NO_Q, b.mtp_in_h, 2 * D, b.mtp_res, D, GEMV_STORE);
    }
    const Layer &w = M;
    hc_mix_run(*this, w.hca, b.mtp_res, T, b.inj_a);
    attn_block(w, NATT, s, T, pos0, b.mtp_res, b.inj_a, spec);
    hc_mix_run(*this, w.hcf, b.mtp_res, T, b.inj_f);
    moe_block(w, NL, T, false, b.mtp_res, b.inj_f);
}

// Run the MTP block over the T tokens just processed at pos0.., pairing each with the previous position's residual,
// so the MTP layer's KV cache tracks the sequence. b.res holds this pass's residuals.
void Engine::mtp_catch_up(Seq &s, int T, int pos0, bool spec) {
    // in pieces of MTP_CHUNK tokens, so the MTP buffers need not grow with the prefill chunk; h_prev rows (each token's
    // previous residual: h_last for the pass's first) staged in b.key, free here
    for (int t0 = 0; t0 < T; t0 += MTP_CHUNK) {
        const int n = std::min(MTP_CHUNK, T - t0);
        float *h = b.key;
        if (t0 == 0) {
            CK(hipMemcpyAsync(h, s.h_last, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
            if (n > 1) CK(hipMemcpyAsync(h + HCD, b.res, (size_t)(n - 1) * HCD * 4, hipMemcpyDeviceToDevice, st));
        } else {
            CK(hipMemcpyAsync(h, b.res + (size_t)(t0 - 1) * HCD, (size_t)n * HCD * 4, hipMemcpyDeviceToDevice, st));
        }
        mtp_forward(s, n, pos0 + t0, h, b.tokens + t0, spec);
    }
}

// ---- prefill and speculative generation --------------------------------------------------------

void Engine::prefill(int id, const int *tokens, int n) {
    if (!embd.q) FAIL("prefill before warmup(): the token embeddings are not loaded");
    Seq &s = seqs.at(id);
    s.pending = -1;
    for (int done = 0; done < n;) {
        const int T = std::min(smax, n - done);
        if (T <= 8) cpu.arm();                    // a decode-sized pass hands its cold picks over like decode does
        forward(s, tokens + done, T, false);
        mtp_catch_up(s, T, s.n, false);
        if (T > 8) pmark(8);
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
    if (prof && n > 8) {
        pmark(9);
        double tot = 0;
        for (double v : prof_t) tot += v;
        fprintf(stderr, "prefill %d tokens: %.3f s:", n, tot);
        for (int i = 0; i < 10; i++) fprintf(stderr, "  %s %.3f", PROF_NAMES[i], prof_t[i]);
        fprintf(stderr, "\n");
        for (double &v : prof_t) v = 0;
    }
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
    if (route_dec_file && !route_dec) {
        CK(hipMalloc(&route_dec, (size_t)NL * NE * 4));
        CK(hipMemsetAsync(route_dec, 0, (size_t)NL * NE * 4, st));
    }
    if (route_dec && ++route_dec_steps % 256 == 0) {
        std::vector<unsigned> c((size_t)NL * NE);
        CK(hipMemcpyAsync(c.data(), route_dec, c.size() * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        std::vector<int64_t> c64(c.begin(), c.end());
        if (FILE *f = fopen(route_dec_file, "wb")) { fwrite(c64.data(), 8, c64.size(), f); fclose(f); }
    }
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
        if (i == 0) mtp_forward(s, 1, q, s.h_last, b.draft, true);
        else {
            CK(hipMemcpyAsync(b.value, b.mtp_res, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
            mtp_forward(s, 1, q + i, b.value, b.draft + i, true);
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
    CK(hipEventRecord(topk_ready, st));
    // 3. MTP catch-up on the verified pairs (rejected rows get overwritten later); it has no cold experts, so the CPU
    // workers can stop, and the host samples while the GPU runs it (the commit below queues after it)
    CK(hipMemcpyAsync(b.tokens, b.draft, T * 4, hipMemcpyDeviceToDevice, st));
    mtp_catch_up(s, T, q, k > 0);
    CK(hipEventSynchronize(topk_ready));
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
        idx_tail_commit(s.idx_tail, b.idx_raw, (size_t)umax * IDX_D, NKVL, q, q + a + 1, st);
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
