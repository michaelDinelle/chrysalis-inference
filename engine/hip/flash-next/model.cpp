#include "model.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

static const QAct NO_Q{nullptr, nullptr};
static QAct qoff(QAct a, size_t elems) { return {a.q + elems, a.d + elems / 32}; }
static Pos3 cell_pos(const Seq &s, int p0, const int *pos3, int cell);

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
    s.n = 0; s.pos = 0; s.tail_pos = {0, 0, 0}; s.pending = -1; s.pages.clear(); s.seen.clear(); s.top_v.clear(); s.top_i.clear();
    s.hist[0] = s.hist[1] = -1;
    s.look.clear();
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
    if (--page_ref[p] == 0) free_pages.insert(std::upper_bound(free_pages.begin(), free_pages.end(), p), p);
}

void Engine::ensure_pages(Seq &s, int n_tokens, int from) {
    const int need = (n_tokens + PAGE - 1) / PAGE;
    int dirty = (int)s.pages.size();
    int shared = 0;
    for (int i = from / PAGE; i < dirty && i < need; i++) shared += page_ref[s.pages[i]] > 1;
    kv_need(need - dirty + shared);
    if ((int)free_pages.size() < need - dirty + shared) throw std::runtime_error("out of KV pages");
    while ((int)s.pages.size() < need) {
        const int p = page_alloc();
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
    if (getenv("CHRYSALIS_TRACE"))
        fprintf(stderr, "chrysalis-fn: snapshot buffer %zu MB: fault-in %.0f ms, register %.0f ms\n", n >> 20,
                std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
    return p;
}

void *Engine::take_snap_buffer() {
    if (snap_spare.empty()) return snap_buffer();
    void *p = snap_spare.back();
    snap_spare.pop_back();
    return p;
}

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
    if (getenv("CHRYSALIS_TRACE"))
        fprintf(stderr, "chrysalis-fn: %d snapshot buffers (%zu MB): fault-in wait %.0f ms, register %.0f ms\n", snap_pending_n,
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
    sn.pos = s.pos; sn.tail_pos = s.tail_pos;
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
    if (!sn.used) throw std::runtime_error("restore of a freed snapshot");
    CK(hipStreamWaitEvent(st, copied, 0));
    for (int p : s.pages) page_release(p);
    s.pages = sn.pages;
    for (int p : s.pages) page_ref[p]++;
    if (!s.pages.empty()) CK(hipMemcpyAsync(s.d_pages, s.pages.data(), s.pages.size() * 4, hipMemcpyHostToDevice, st));
    s.n = sn.n;
    s.pos = sn.pos; s.tail_pos = sn.tail_pos;
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

struct SnapHeader { char magic[8]; uint64_t kv_page, state; int32_t n, pos, tail[3], hist[2], n_top, n_pages; };
static const char SNAP_MAGIC[8] = {'K', 'I', 'L', 'N', 'S', 'N', 'F', '1'};

size_t Engine::snap_bytes(int snap) const {
    const Snap &sn = snaps.at(snap);
    return sizeof(SnapHeader) + sn.top_v.size() * 8 + state_bytes() + sn.pages.size() * KV_PAGE;
}

void Engine::snap_export(int snap, uint8_t *dst) {
    const Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("export of a freed snapshot");
    CK(hipStreamSynchronize(st));
    CK(hipStreamSynchronize(cst));
    SnapHeader h;
    memcpy(h.magic, SNAP_MAGIC, 8);
    h.kv_page = KV_PAGE; h.state = state_bytes(); h.n = sn.n; h.pos = sn.pos;
    h.tail[0] = sn.tail_pos.t; h.tail[1] = sn.tail_pos.h; h.tail[2] = sn.tail_pos.w;
    h.hist[0] = sn.hist[0]; h.hist[1] = sn.hist[1];
    h.n_top = (int32_t)sn.top_v.size(); h.n_pages = (int32_t)sn.pages.size();
    uint8_t *p = dst;
    memcpy(p, &h, sizeof h); p += sizeof h;
    memcpy(p, sn.top_v.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
    memcpy(p, sn.top_i.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
    memcpy(p, sn.host, h.state); p += h.state;
    for (int pg : sn.pages) {
        CK(hipMemcpyAsync(p, kv_pool + (size_t)pg * KV_PAGE, KV_PAGE, hipMemcpyDeviceToHost, cst));
        p += KV_PAGE;
    }
    CK(hipStreamSynchronize(cst));
}

int Engine::snap_import(int fd, int64_t off, size_t bytes) {
    auto get = [&](void *dst, size_t n) {
        for (size_t got = 0; got < n;) {
            const ssize_t r = pread(fd, (uint8_t *)dst + got, n - got, off + (int64_t)got);
            if (r <= 0) return false;
            got += (size_t)r;
        }
        off += (int64_t)n;
        return true;
    };
    SnapHeader h;
    if (bytes < sizeof h || !get(&h, sizeof h)) return -1;
    if (memcmp(h.magic, SNAP_MAGIC, 8) || h.kv_page != KV_PAGE || h.state != state_bytes() || h.n <= 0 || h.n_top < 0 ||
        h.n_pages != (h.n + PAGE - 1) / PAGE || bytes != sizeof h + (size_t)h.n_top * 8 + h.state + (size_t)h.n_pages * KV_PAGE)
        return -1;
    kv_need(h.n_pages);
    if ((int)free_pages.size() < h.n_pages) return -2;
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    if (!sn.host) sn.host = take_snap_buffer();
    sn.top_v.resize(h.n_top);
    sn.top_i.resize(h.n_top);
    bool ok = get(sn.top_v.data(), (size_t)h.n_top * 4) && get(sn.top_i.data(), (size_t)h.n_top * 4);
    const int64_t state_at = off;
    off += (int64_t)h.state;
    const int per = (int)(h.state / 2 / KV_PAGE);
    hipEvent_t ev[2];
    for (hipEvent_t &e : ev) CK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
    sn.pages.clear();
    for (int i = 0; ok && i < h.n_pages; i += per) {
        const int half = i / per % 2, n = std::min(per, h.n_pages - i);
        uint8_t *buf = (uint8_t *)sn.host + (size_t)half * per * KV_PAGE;
        if (i >= 2 * per) CK(hipEventSynchronize(ev[half]));
        ok = get(buf, (size_t)n * KV_PAGE);
        for (int k = 0; ok && k < n; k++) {
            sn.pages.push_back(page_alloc());
            CK(hipMemcpyAsync(kv_pool + (size_t)sn.pages.back() * KV_PAGE, buf + (size_t)k * KV_PAGE, KV_PAGE, hipMemcpyHostToDevice, st));
        }
        CK(hipEventRecord(ev[half], st));
    }
    CK(hipStreamSynchronize(st));
    for (hipEvent_t e : ev) CK(hipEventDestroy(e));
    off = state_at;
    if (!ok || !get(sn.host, h.state)) {
        for (int p : sn.pages) page_release(p);
        sn.pages.clear();
        return -1;
    }
    sn.n = h.n;
    sn.pos = h.pos;
    sn.tail_pos = {h.tail[0], h.tail[1], h.tail[2]};
    sn.hist[0] = h.hist[0]; sn.hist[1] = h.hist[1];
    sn.used = true;
    return (int)si;
}

void Engine::snap_free(int snap) {
    Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("double free of a snapshot");
    CK(hipStreamSynchronize(st));
    CK(hipStreamSynchronize(cst));
    for (int p : sn.pages) page_release(p);
    sn.pages.clear();
    sn.used = false;
}

Engine::LtPlan &Engine::lt_plan(int N, int K, int T, int ldx, int ldd, bool c32, const half *W, const half *x, void *d) {
    const auto key = std::make_tuple(N, K, T, ldx, ldd, (int)c32);
    auto it = lt_plans.find(key);
    if (it == lt_plans.end()) it = lt_plans.emplace(key, lt_tune::plan(lt, N, K, T, ldx, ldd, c32, W, x, d, lt_ws, lt_ws_size, st)).first;
    return it->second;
}

void Engine::gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate) {
    const float alpha = 1.0f, beta = 0.0f;
    half *dst = y_f16 && !accumulate ? (half *)y : b.gemm16;
    const int ldd = dst == y ? ldy : N;
    const int Tp = std::min(umax * HC, (T + 63) / 64 * 64);
    LtPlan &p = lt_plan(N, K, Tp, ldx, ldd, false, W, x, dst);
    if (hipblasLtMatmul(lt, p.desc, &alpha, W, p.a, x, p.b, &beta, dst, p.c, dst, p.c, &p.algo, lt_ws, lt_ws_size, st) != HIPBLAS_STATUS_SUCCESS)
        FAIL("hipblasLtMatmul failed (%dx%dx%d)", N, K, T);
    if (dst != y) store_f16(b.gemm16, N, T, (float *)y, ldy, accumulate, st);
}

void Engine::gemm32(const half *W, int N, int K, const half *x, int ldx, int T, float *y, int ldy) {
    const float alpha = 1.0f, beta = 0.0f;
    const int Tp = std::min(umax * HC, (T + 63) / 64 * 64);
    LtPlan &p = lt_plan(N, K, Tp, ldx, ldy, true, W, x, y);
    if (hipblasLtMatmul(lt, p.desc, &alpha, W, p.a, x, p.b, &beta, y, p.c, y, p.c, &p.algo, lt_ws, lt_ws_size, st) != HIPBLAS_STATUS_SUCCESS)
        FAIL("hipblasLtMatmul failed (%dx%dx%d)", N, K, T);
}

void Engine::linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16) {
    if (T <= 8) {
        gemv(W, qin, T, (float *)out, ldo, mode, st);
    } else if (T <= GEMM_Q_MAX && W.type != F16 && W.rows % 32 == 0 && W.cols % 256 == 0) {
        gemm_q(W, hin, ldin, T, out, ldo, mode, out_f16, st);
    } else if (W.type == F16) {
        gemm((const half *)W.f, W.rows, W.cols, hin, ldin, T, out, ldo, out_f16, mode == GEMV_ADD);
    } else {
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

static void hc_mix_run(Engine &e, const HcMix &m, const float *res, int T, float *inject, bool pieces = false) {
    auto &b = e.b;
    if (T <= 4 || (pieces && T <= 8)) {
        for (int t0 = 0; t0 < T; t0 += 4) {
            const int n = std::min(4, T - t0);
            hc_mix_fused(res + (size_t)t0 * HCD, m.norm, m.down, m.up, inject ? &m.inject : nullptr, n, b.lo + (size_t)t0 * HCR,
                         inject ? inject + (size_t)t0 * HC : nullptr, HC, b.inv + t0 * HC, b.mx + (size_t)t0 * D,
                         b.hh + (size_t)t0 * D, qoff(b.hq, (size_t)t0 * D), e.st);
        }
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

void Engine::attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *res, float *inject, bool spec, int rope0,
                        const int *rpos, Pos3 tail_pos, int praw_off) {
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
    attn_core(w, kvl, s, T, 0, praw_off, pos0, spec, rope0, rpos, tail_pos);
    if (small) gemv_hc(w.o, b.mq, T, res, inject, st);
    else {
        linear(w.o, T, NO_Q, b.att_h, NH * HD, b.blk, D, GEMV_STORE);
        hc_combine(res, inject, HC, b.blk, nullptr, T, st);
    }
}

void Engine::attn_core(const Layer &w, int kvl, Seq &s, int T, int off, int praw_off, int pos0, bool spec, int rope0,
                       const int *rpos, Pos3 tail_pos) {
    const bool small = T <= 8;
    float *proj = b.proj + (size_t)off * 2 * NH * HD;
    half *Q16 = b.Q16 + (size_t)off * NH * HD, *q16 = b.qidx16 + (size_t)off * IDX_H * IDX_D;
    int *sel = b.sel + (size_t)off * (SEL_BLOCKS + 1);
    const QAct mq = small ? qoff(b.mq, (size_t)off * NH * HD) : NO_Q;
    idx_prep(b.kidx + (size_t)off * IDX_D, b.qidx + (size_t)off * IDX_H * IDX_D, w.idx_qn, w.idx_kn, T, pos0, rope0, rpos, tail_pos,
             kv(s), kvl, s.idx_tail + (size_t)kvl * IDX_TAIL, b.idx_raw + ((size_t)kvl * umax + praw_off) * IDX_D, !spec, q16, st);
    attn_prep(proj, b.kb + (size_t)off * NKV * HD, b.vb + (size_t)off * NKV * HD, w.q_norm, w.k_norm, T, pos0, rope0, rpos, kv(s),
              kvl, Q16, st);
    AttnArgs a;
    a.Q16 = Q16; a.kv = kv(s); a.layer = kvl; a.T = T; a.pos0 = pos0; a.qg = proj;
    a.part_o = b.part_o; a.part_ml = b.part_ml;
    const int last_pos = pos0 + T - 1;
    if ((last_pos + 1) / KPOOL <= SEL_BLOCKS) {
        attention(a, small ? nullptr : b.att_h, mq, st);
    } else {
        const int nblk = (last_pos + 1) / KPOOL;
        for (int g0 = 0; g0 < T; g0 += SEL_G) {
            const int G = std::min(SEL_G, T - g0);
            idx_select(q16 + (size_t)g0 * IDX_H * IDX_D, pos0 + g0, G, nblk, kv(s), kvl, b.score, b.sel_hist,
                       sel + (size_t)g0 * (SEL_BLOCKS + 1), st);
        }
        attention_sel(a, sel, small ? nullptr : b.att_h, mq, st);
    }
}

void Engine::attn_parts(const Layer &w, int kvl, const Part *p, int np, int T, float *res, float *inject) {
    GemvArgs a;
    a.seg[0] = {w.q, b.proj, 2 * NH * HD}; a.seg[1] = {w.k, b.kb, NKV * HD}; a.seg[2] = {w.v, b.vb, NKV * HD};
    a.nseg = 3; a.x = b.hq; a.T = T;
    gemv(a, st);
    GemvArgs ix;
    ix.seg[0] = {w.idx_q, b.qidx, IDX_H * IDX_D}; ix.seg[1] = {w.idx_k, b.kidx, IDX_D}; ix.nseg = 2; ix.x = b.hq; ix.T = T;
    gemv(ix, st);
    for (int i = 0; i < np; i++) {
        Seq &s = *p[i].s;
        attn_core(w, kvl, s, p[i].T, p[i].off, p[i].off, s.n, p[i].verify, s.pos, nullptr, cell_pos(s, s.n, nullptr, s.n / KPOOL * KPOOL));
    }
    gemv_hc(w.o, b.mq, T, res, inject, st);
}

void Engine::gdn_parts(const Layer &w, int gl, const Part *p, int np, int T, float *res, float *inject) {
    float *raw = b.v_raw + (size_t)gl * 8 * QKV;
    GemvArgs a;
    a.seg[0] = {w.qkv, raw, QKV}; a.seg[1] = {w.z, b.zb, GV * GD}; a.nseg = 2; a.x = b.hq; a.T = T;
    gemv(a, st);
    for (int i = 0; i < np; i++) {
        const Part &q = p[i];
        GdnStepArgs g;
        g.raw = raw + (size_t)q.off * QKV; g.z = b.zb + (size_t)q.off * GV * GD; g.hh = b.hh + (size_t)q.off * D; g.Wab = w.Wab;
        g.conv_w = w.conv; g.A = gdn_A + gl * GV; g.dt = gdn_dt + gl * GV; g.norm_w = w.ssm_norm;
        g.conv_state = q.s->conv + (size_t)gl * QKV * 3; g.S = q.s->ssm + (size_t)gl * GV * GD * GD;
        g.T = q.T; g.verify = q.verify;
        g.kn_out = b.v_kn + ((size_t)gl * 8 + q.off) * GK * GD; g.vc_out = b.v_vc + ((size_t)gl * 8 + q.off) * GV * GD;
        g.ab_out = b.v_ab + ((size_t)gl * 8 + q.off) * 2 * GV; g.out = qoff(b.mq, (size_t)q.off * GV * GD);
        gdn_step(g, st);
    }
    gemv_hc(w.ssm_out, b.mq, T, res, inject, st);
}

void Engine::gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *res, float *inject) {
    if (T <= 8) {
        const Part p{&s, T, 0, verify};
        gdn_parts(w, gl, &p, 1, T, res, inject);
        return;
    }
    float *raw = b.proj, *z = b.zb, *ab = b.kb;
    linear(w.qkv, T, NO_Q, b.hh, D, raw, QKV, GEMV_STORE);
    linear(w.z, T, NO_Q, b.hh, D, z, GV * GD, GEMV_STORE);
    gemm32(w.Wab, 2 * GV, D, b.hh, D, T, ab, 2 * GV);
    gdn_conv(raw, T, w.conv, s.conv + (size_t)gl * QKV * 3, true, b.qn, b.kn, b.vc, st);
    GdnScanArgs sa{};
    sa.S = s.ssm + (size_t)gl * GV * GD * GD; sa.qn = b.qn; sa.kn = b.kn; sa.vc = b.vc; sa.ab = ab;
    sa.A = gdn_A + gl * GV; sa.dtb = gdn_dt + gl * GV; sa.T = T; sa.write_state = 1; sa.out = b.vc;
    gdn_scan(sa, 1, st);
    gdn_out(b.vc, z, w.ssm_norm, T, b.att_h, NO_Q, st);
    linear(w.ssm_out, T, NO_Q, b.att_h, GV * GD, b.blk, D, GEMV_STORE);
    hc_combine(res, inject, HC, b.blk, nullptr, T, st);
}

void Engine::moe_block(const Layer &w, int layer, int T, bool cpu_cold, float *res, float *inject, unsigned starts) {
    if (T > 8) {
        moe_front(w, T, 0);
        moe_back(w, layer, T);
        hc_combine(res, inject, HC, b.moe, nullptr, T, st);
        return;
    }
    const float gf = cpu_cold && w.n_cold ? gfrac : 0.0f;
    const bool pooled = layer < NL;
    const float xf = cpu_cold ? xfrac : 1.0f;
    router_route_dec(w.router, w.sg_w, b.mx, T, b.logits_e, b.sg, w.slot_of, pooled ? cold[layer].n : 0, b.ids, b.wts, b.hot,
                     cpu_cold ? d_sh : nullptr, gf, layer, gen + 1, b.rr_counter, st, pooled ? xc.d_counts : nullptr, starts, xf);
    if (cpu_cold) gen++;
    moe_dec(w.hot_gate, w.hot_up, w.hot_down, pooled ? d_fmt : nullptr, w.sh_gate, w.sh_up, w.sh_down, gf > 0 ? &cold_dev[layer] : nullptr,
            b.hot, T, b.hq, b.hq_e, b.wts, b.sg, b.y, st);
    if (gf > 0 && xf < 1.0f) emu_glu(b.hot, d_me, gen, T, st);
    moe_finish(cpu_cold ? &d_sh->done : nullptr, gen, b.y, b.hot, b.y + (size_t)T * TOPK_E * D, nullptr, cpu_cold ? d_sh->y : nullptr,
               inject, res, T, st, d_me ? d_me->y[0] : nullptr);
}

namespace {
constexpr double GPU_PER_EXPERT = 4e-6;
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

void Engine::moe_front(const Layer &w, int n, int c0) {
    half *x = b.hmoe + (size_t)c0 * D;
    CK(hipMemcpyAsync(x, b.hh, (size_t)n * D * 2, hipMemcpyDeviceToDevice, st));
    gemm(w.router, NE, D, x, D, n, b.rlog, NE, false, false);
    route_many(b.rlog, w.sg_w, b.mx, n, b.idsT + (size_t)c0 * TOPK_E, b.wtsT + (size_t)c0 * TOPK_E, b.sgT + c0, st);
    CK(hipEventRecord(front_ready, st));
    CK(hipStreamWaitEvent(cst, front_ready, 0));
    CK(hipMemcpyAsync(b.h_ids + (size_t)c0 * TOPK_E, b.idsT + (size_t)c0 * TOPK_E, (size_t)n * TOPK_E * 4, hipMemcpyDeviceToHost, cst));
    CK(hipMemcpyAsync(b.h_wts + (size_t)c0 * TOPK_E, b.wtsT + (size_t)c0 * TOPK_E, (size_t)n * TOPK_E * 4, hipMemcpyDeviceToHost, cst));
    if (&w != &M) CK(hipMemcpyAsync(b.h_x16 + (size_t)c0 * D, x, (size_t)n * D * 2, hipMemcpyDeviceToHost, cst));
    half *g = b.att_h, *u = b.att_h + (size_t)n * FS;
    linear(w.sh_gate, n, NO_Q, x, D, g, FS, GEMV_STORE, true);
    linear(w.sh_up, n, NO_Q, x, D, u, FS, GEMV_STORE, true);
    swiglu_h(g, u, (int64_t)n * FS, st);
    linear(w.sh_down, n, NO_Q, g, FS, b.blk, D, GEMV_STORE);
    moe_shared(b.blk, b.sgT + c0, n, b.moe + (size_t)c0 * D, st);
}

void Engine::moe_back(const Layer &w, int layer, int T) {
    const int N = T * TOPK_E, P = smax;
    CK(hipEventRecord(front_copied, cst));
    CK(hipEventSynchronize(front_copied));
    const std::vector<int> &slot_of = w.slot_h;
    std::vector<int> cnt(NE, 0);
    for (int j = 0; j < N; j++) cnt[b.h_ids[j]]++;
    if (!route_counts.empty() && layer < NL) for (int e = 0; e < NE; e++) route_counts[(size_t)layer * NE + e] += cnt[e];
    std::vector<int> ce, hot;
    for (int e = 0; e < NE; e++) if (cnt[e]) (slot_of[e] >= 0 ? hot : ce).push_back(e);
    std::stable_sort(ce.begin(), ce.end(), [&](int a, int c) { return cnt[a] < cnt[c]; });
    const ColdLayer *cl = layer < NL ? &cold[layer] : nullptr;
    auto cslot = [&](int e) { return -1 - slot_of[e]; };
    auto ebytes = [&](int e) { return (double)cl->bytes(cslot(e)); };
    int kc = 0, kmax = 0;
    {
        const double bias = split_bias(T);
        double best = 1e30, macs = 0, cb = 0, gb = 0;
        for (int e : ce) gb += ebytes(e);
        const int K = (int)ce.size();
        for (int k = 0, picks = 0; k <= K; k++) {
            const double ta = cb / cal_cpu_bw, tm = macs / cal_cpu_mac;
            const double t = std::max(bias * (std::max(ta, tm) + 0.25 * std::min(ta, tm)), gb / cal_dma_bw + (K - k) * GPU_PER_EXPERT);
            if (t < best) { best = t; kc = k; }
            kmax = k;
            if (k == K || picks + cnt[ce[k]] > P) break;
            picks += cnt[ce[k]]; macs += cnt[ce[k]] * EXPERT_MAC; cb += ebytes(ce[k]); gb -= ebytes(ce[k]);
        }
    }
    if (split >= 0) kc = split ? kmax : 0;
    std::vector<int> gpu_cold(ce.begin() + kc, ce.end()), cpu_cold(ce.begin(), ce.begin() + kc);
    auto fmt_of = [&](int e) { return layer < NL ? (int)xc.fmt[slot_of[e]] : w.hot_gate.type; };
    std::sort(hot.begin(), hot.end(), [&](int a, int c) { return std::make_pair(fmt_of(a), slot_of[a]) < std::make_pair(fmt_of(c), slot_of[c]); });
    std::sort(gpu_cold.begin(), gpu_cold.end(), [&](int a, int c) { return slot_of[a] > slot_of[c]; });
    std::stable_sort(cpu_cold.begin(), cpu_cold.end(), [&](int a, int c) { return cnt[a] > cnt[c]; });
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
    struct Range { int p0, p1, t0, t1; };
    struct Group { int range, i0, i1, w0, w1; };
    std::vector<Range> ranges;
    std::vector<Group> hot_g, gpu_g;
    int nw = 0, nt = 0;
    std::vector<int> item_fmt(N / 64 + 2 * NE);
    std::vector<int> mark(T, -1);
    auto close_range = [&](Range &r) {
        r.t0 = nt;
        const int id = (int)ranges.size();
        for (int p = r.p0; p < r.p1; p++) { const int t = b.h_tok[p]; if (mark[t] != id) { mark[t] = id; b.h_btok[nt++] = t; } }
        r.t1 = nt;
        ranges.push_back(r);
    };
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
            for (int t0 = 0; t0 < n; t0 += MOE_ITEM) {
                item_fmt[nw] = hot_slots ? fmt_of(e) : -1;
                b.h_work[nw++] = {sl, p0 - r.p0 + t0, std::min(MOE_ITEM, n - t0), 0};
            }
            g.i1 = i + 1; g.w1 = nw; r.p1 = p0 + n;
        }
        if (r.p0 >= 0) { close_group((int)ex.size()); close_range(r); }
    };
    int max_cnt = 0;
    for (const auto *v : {&hot, &gpu_cold}) for (int e : *v) max_cnt = std::max(max_cnt, cnt[e]);
    const bool halves = max_cnt <= P / 2 && !hot.empty() && !gpu_cold.empty();
    plan(hot_g, hot, true, NE, halves ? P / 2 : P);
    plan(gpu_g, gpu_cold, false, STAGE_G, halves ? P / 2 : P);
    half *X0 = b.Xs, *H0 = b.Hs, *Y0 = b.Ys;
    half *X1 = halves ? b.Xs + (size_t)(P / 2) * D : b.Xs, *H1 = halves ? b.Hs + (size_t)(P / 2) * FE : b.Hs;
    half *Y1 = halves ? b.Ys + (size_t)(P / 2) * D : b.Ys;
    const int pc0 = cpu_cold.empty() ? N : off[cpu_cold[0]];
    if (!cpu_cold.empty()) { Range r{pc0, N, 0, 0}; close_range(r); }
    if (!cpu_cold.empty()) {
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
    auto hot_gemms = [&](const Group &g, int w0, int w1, half *X, half *H, half *Y) {
        for (int a = w0; a < w1;) {
            int z = a + 1;
            while (z < w1 && item_fmt[z] == item_fmt[a]) z++;
            moe_gemm_gateup(w.hot_gate, w.hot_up, none, none, b.work + a, z - a, X, H, st, layer < NL ? item_fmt[a] : -1);
            a = z;
        }
        moe_gemm_down(w.hot_down, none, b.work + w0, w1 - w0, H, b.ent_w + ranges[g.range].p0, Y, st);
    };
    auto gather = [&](int ri, half *X) { const Range &r = ranges[ri]; gather_rows(b.hmoe, b.ent_tok + r.p0, r.p1 - r.p0, X, st); };
    auto accum = [&](int ri, half *Y) { const Range &r = ranges[ri]; moe_accum(Y, b.pos, r.p0, r.p1, b.btok + r.t0, r.t1 - r.t0, b.moe, st); };
    auto dma = [&](int gi) {
        const Group &G = gpu_g[gi];
        const int k = gi % NSLOT, fg = cl->grp(cslot(gpu_cold[G.i0])), ng = G.i1 - G.i0;
        const size_t gsz = (size_t)FE * cl->row_gu[fg], dsz = (size_t)D * cl->row_dn;
        CK(hipStreamWaitEvent(cst, raw_free[k], 0));
        uint8_t *r = b.raw[k];
        if (fg == 2) CK(hipEventSynchronize(raw_ready[k]));
        for (int c = 0; c < ng;) {
            const int s0 = cslot(gpu_cold[G.i0 + c]);
            int run_n = 1;
            while (c + run_n < ng && cslot(gpu_cold[G.i0 + c + run_n]) == s0 + run_n) run_n++;
            const uint8_t *src[3] = {cl->gu_row(false, s0, 0), cl->gu_row(true, s0, 0), cl->dn_row(s0, 0)};
            const size_t at[3] = {c * gsz, stage_gu + c * gsz, 2 * stage_gu + c * dsz}, len[3] = {run_n * gsz, run_n * gsz, run_n * dsz};
            for (int m = 0; m < 3; m++) {
                if (fg == 2) memcpy(b.bounce[k] + at[m], src[m], len[m]);
                else CK(hipMemcpyAsync(r + at[m], src[m], len[m], hipMemcpyHostToDevice, cst));
            }
            c += run_n;
        }
        if (fg == 2) {
            CK(hipMemcpyAsync(r, b.bounce[k], ng * gsz, hipMemcpyHostToDevice, cst));
            CK(hipMemcpyAsync(r + stage_gu, b.bounce[k] + stage_gu, ng * gsz, hipMemcpyHostToDevice, cst));
            CK(hipMemcpyAsync(r + 2 * stage_gu, b.bounce[k] + 2 * stage_gu, ng * dsz, hipMemcpyHostToDevice, cst));
        }
        CK(hipEventRecord(raw_ready[k], cst));
    };
    int nwh = 0;
    for (const Group &g : hot_g) nwh += g.w1 - g.w0;
    const int slots = halves ? (int)gpu_g.size() : 1;
    auto hot_slice = [&](int s) {
        const int a = (int)((int64_t)nwh * s / slots), z = (int)((int64_t)nwh * (s + 1) / slots);
        for (int i = 0, base = 0; i < (int)hot_g.size(); base += hot_g[i].w1 - hot_g[i].w0, i++) {
            const Group &g = hot_g[i];
            const int lo = std::max(a, base), hi = std::min(z, base + g.w1 - g.w0);
            if (lo >= hi) continue;
            if (lo == base) gather(g.range, X0);
            hot_gemms(g, g.w0 + lo - base, g.w0 + hi - base, X0, H0, Y0);
            if (hi == base + g.w1 - g.w0) accum(g.range, Y0);
        }
    };
    for (int gi = 0; gi < std::min((int)gpu_g.size(), NSLOT - 1); gi++) dma(gi);
    if (gpu_g.empty()) hot_slice(0);
    for (size_t gi = 0; gi < gpu_g.size(); gi++) {
        const Group &g = gpu_g[gi];
        if (gi + NSLOT - 1 < gpu_g.size()) dma((int)gi + NSLOT - 1);
        if ((int)gi < slots) hot_slice((int)gi);
        if (gi == 0) pmark(4);
        if (gi == 0 || gpu_g[gi - 1].range != g.range) gather(g.range, X1);
        const int k = gi % NSLOT, ns = g.i1 - g.i0, fmt = cl->type_gu[cl->grp(cslot(gpu_cold[g.i0]))];
        CK(hipStreamWaitEvent(st, raw_ready[k], 0));
        Arena ag, au;
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
    if (!cpu_cold.empty()) {
        cpu.wait();
        pmark(6);
        CK(hipMemcpyAsync(b.Ys, b.h_cout, (size_t)(N - pc0) * D * 2, hipMemcpyHostToDevice, st));
        accum((int)ranges.size() - 1, b.Ys);
    }
}

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

void Engine::ple_prefetch(const size_t *rows, size_t n) const {
    constexpr size_t RB = PLE_DIM / 32 * 34;
    for (size_t i = 0; i < n; i++) {
        const uintptr_t a = (uintptr_t)(ngram + rows[i] * RB), p0 = a & ~uintptr_t(4095), p1 = (a + RB - 1) & ~uintptr_t(4095);
        madvise((void *)p0, p1 - p0 + 4096, MADV_WILLNEED);
    }
}

void Engine::ple_rows(const Seq &s, const int *tokens, int T, int off) {
    constexpr size_t RB = PLE_DIM / 32 * 34;
    std::vector<size_t> rows((size_t)T * PLE_HEADS);
    ple_ids(s.hist[0], s.hist[1], tokens, T, rows.data());
    uint8_t *dst = b.ple_rows + (size_t)off * PLE_HEADS * RB;
    auto copy = [&](size_t i0, size_t i1) {
        ple_prefetch(rows.data() + i0, i1 - i0);
        for (size_t i = i0; i < i1; i++) memcpy(dst + i * RB, ngram + rows[i] * RB, RB);
    };
    if (T <= 8) { copy(0, rows.size()); return; }
    std::thread th[4];
    for (int k = 0; k < 4; k++) th[k] = std::thread(copy, rows.size() * k / 4, rows.size() * (k + 1) / 4);
    for (auto &t : th) t.join();
}

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

void Engine::lookahead(const Layer &w, int layer, int T, const float *res, unsigned starts) {
    for (int t0 = 0; t0 < T; t0 += 4)
        hc_mix_fused(res + (size_t)t0 * HCD, w.hcf.norm, w.hcf.down, w.hcf.up, nullptr, std::min(4, T - t0), la.lo + (size_t)t0 * HCR,
                     nullptr, HC, la.inv + t0 * HC, la.mx + (size_t)t0 * D, la.hh + (size_t)t0 * D, qoff(la.hq, (size_t)t0 * D), st);
    router_route_dec(w.router, w.sg_w, la.mx, T, la.logits, la.sg, w.slot_of, cold[layer].n, la.ids, la.wts, la.hot, d_sh_pred, gfrac,
                     layer, gen + 1, la.counter, st, nullptr, starts, xfrac);
}

static Pos3 cell_pos(const Seq &s, int p0, const int *pos3, int cell) {
    if (cell < p0) return s.tail_pos;
    const int i = cell - p0;
    if (pos3) return {pos3[3 * i], pos3[3 * i + 1], pos3[3 * i + 2]};
    return {s.pos + i, s.pos + i, s.pos + i};
}

void Engine::forward(Seq &s, const int *tokens, int T, bool verify, const int *pos3, bool img) {
    ensure_pages(s, s.n + T, s.n);
    experts_place(T);
    if (T > 8) pmark(-1);
    std::vector<int> grams;
    const int *hashed = tokens;
    if (img) {
        grams.assign(tokens, tokens + T);
        for (int &t : grams) if (t >= IMG_BASE) t = IMAGE_PAD;
        hashed = grams.data();
    }
    std::thread ple_th;
    if (T > 8) ple_th = std::thread([this, &s, hashed, T] { ple_rows(s, hashed, T); });
    else ple_rows(s, hashed, T);
    CK(hipMemcpyAsync(b.tokens, tokens, T * 4, hipMemcpyHostToDevice, st));
    if (pos3) CK(hipMemcpyAsync(b.rpos, pos3, (size_t)T * 12, hipMemcpyHostToDevice, st));
    const int nc = (T + umax - 1) / umax;
    auto cb = [&](int c) { return (int)((int64_t)T * c / nc); };
    for (int c = 0; c < nc; c++) {
        const int c0 = cb(c), n = cb(c + 1) - c0;
        embed(embd, b.tokens + c0, img ? b.img + c0 : nullptr, n, b.x, st);
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
            if (T <= 4 && la_on && xc.recent_hits < Experts::CPU_HEAVY && w.n_cold) lookahead(w, l, n, res);
            hc_mix_run(*this, w.hca, res, n, b.inj_a);
            if (is_attn_layer(l))
                attn_block(w, l / 4, s, n, s.n + c0, res, b.inj_a, verify, s.pos + c0, pos3 ? b.rpos + 3 * c0 : nullptr,
                           cell_pos(s, s.n, pos3, (s.n + c0) / KPOOL * KPOOL));
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
    experts_plan();
}

void Engine::ple_parts(const Part *p, int np, int T, float *res) {
    const Layer &w = L[PLE_LAYER];
    constexpr size_t RB = PLE_DIM / 32 * 34;
    CK(hipMemcpyAsync(b.d_ple_rows, b.ple_rows, (size_t)T * PLE_HEADS * RB, hipMemcpyHostToDevice, st));
    ple_embed(b.d_ple_rows, T, nullptr, nullptr, b.emb_q, st);
    gemv(w.ple_key, b.emb_q, T, b.key, HCD, GEMV_STORE, st);
    gemv(w.ple_value, b.emb_q, T, b.value, D, GEMV_STORE, st);
    for (int i = 0; i < np; i++) {
        const Part &q = p[i];
        float *xconv = b.xconv + (size_t)q.off * HCD;
        ple_block(b.key + (size_t)q.off * HCD, b.value + (size_t)q.off * D, res + (size_t)q.off * HCD, w.ple_nk, w.ple_nq, w.ple_nc,
                  w.ple_conv, q.s->ple, xconv, q.T, st);
        if (!q.verify) ple_commit(q.s->ple, xconv, q.T, st);
    }
}

void Engine::forward_parts(const Part *p, int np, const int *tokens, int T) {
    unsigned starts = 0;
    for (int i = 0; i < np; i++) {
        ensure_pages(*p[i].s, p[i].s->n + p[i].T, p[i].s->n);
        starts |= 1u << p[i].off;
    }
    experts_place(T);
    for (int i = 0; i < np; i++) ple_rows(*p[i].s, tokens + p[i].off, p[i].T, p[i].off);
    CK(hipMemcpyAsync(b.tokens, tokens, T * 4, hipMemcpyHostToDevice, st));
    embed(embd, b.tokens, nullptr, T, b.x, st);
    hc_init(b.x, b.res, T, st);
    for (int l = 0, gl = 0; l < NL; l++) {
        const Layer &w = L[l];
        if (l == PLE_LAYER) ple_parts(p, np, T, b.res);
        if (la_on && xc.recent_hits < Experts::CPU_HEAVY && w.n_cold) lookahead(w, l, T, b.res, starts);
        hc_mix_run(*this, w.hca, b.res, T, b.inj_a, true);
        if (is_attn_layer(l)) attn_parts(w, l / 4, p, np, T, b.res, b.inj_a);
        else gdn_parts(w, gl++, p, np, T, b.res, b.inj_a);
        hc_mix_run(*this, w.hcf, b.res, T, b.inj_f, true);
        moe_block(w, l, T, true, b.res, b.inj_f, starts);
    }
    for (int i = 0; i < np; i++)
        if (!p[i].verify) gdn_conv_commit(p[i].s->conv, b.v_raw + (size_t)p[i].off * QKV, p[i].T, 8 * QKV, NGDN, st);
    experts_plan();
}

static void head(Engine &e, const HcMix &hm, const float *res, int T, int rows, bool pieces = false) {
    hc_mix_run(e, hm, res, T, nullptr, pieces);
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

void Engine::mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *d_tokens, bool spec, int rope0, const int *rpos,
                         Pos3 tail_pos, const half *const *img, int praw_off) {
    embed(embd, d_tokens, img, T, b.mtp_e, st);
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
    attn_block(w, NATT, s, T, pos0, b.mtp_res, b.inj_a, spec, rope0, rpos, tail_pos, praw_off);
    hc_mix_run(*this, w.hcf, b.mtp_res, T, b.inj_f);
    moe_block(w, NL, T, false, b.mtp_res, b.inj_f);
}

void Engine::mtp_catch_up(Seq &s, int T, int pos0, bool spec, const int *pos3, bool img, int off) {
    const float *res = b.res + (size_t)off * HCD;
    for (int t0 = 0; t0 < T; t0 += MTP_CHUNK) {
        const int n = std::min(MTP_CHUNK, T - t0);
        float *h = b.key;
        if (t0 == 0) {
            CK(hipMemcpyAsync(h, s.h_last, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
            if (n > 1) CK(hipMemcpyAsync(h + HCD, res, (size_t)(n - 1) * HCD * 4, hipMemcpyDeviceToDevice, st));
        } else {
            CK(hipMemcpyAsync(h, res + (size_t)(t0 - 1) * HCD, (size_t)n * HCD * 4, hipMemcpyDeviceToDevice, st));
        }
        mtp_forward(s, n, pos0 + t0, h, b.tokens + off + t0, spec, s.pos + t0, pos3 ? b.rpos + 3 * t0 : nullptr,
                    cell_pos(s, pos0, pos3, (pos0 + t0) / KPOOL * KPOOL), img ? b.img + t0 : nullptr, off);
    }
}

void Engine::prefill(int id, const int *tokens, const int *pos3, int n) {
    if (!embd.q) FAIL("prefill before warmup(): the token embeddings are not loaded");
    Seq &s = seqs.at(id);
    std::vector<const half *> rows(n, nullptr);
    bool any_img = false;
    for (int i = 0; i < n; i++) {
        const int t = tokens[i];
        if (t >= IMG_BASE) {
            const int pic = (t - IMG_BASE) / IMG_ROWS, row = (t - IMG_BASE) % IMG_ROWS;
            if (pic >= (int)images.size() || !images[pic].rows || row >= images[pic].n)
                throw std::invalid_argument("an image token names a picture that is not encoded");
            rows[i] = images[pic].rows + (size_t)row * D;
            any_img = true;
        } else if (t < 0 || t >= VOCAB) {
            throw std::invalid_argument("token id out of range");
        }
    }
    s.pending = -1;
    for (int done = 0; done < n;) {
        const int T = std::min(smax, n - done);
        if (T <= 8) cpu.arm();
        const int *p3 = pos3 ? pos3 + (size_t)done * 3 : nullptr;
        const bool img = any_img && std::any_of(rows.begin() + done, rows.begin() + done + T, [](const half *r) { return r; });
        if (img) CK(hipMemcpyAsync(b.img, rows.data() + done, T * sizeof(half *), hipMemcpyHostToDevice, st));
        forward(s, tokens + done, T, false, p3, img);
        mtp_catch_up(s, T, s.n, false, p3, img);
        if (T > 8) pmark(8);
        if (done + T == n) {
            head(*this, hc_head, b.res + (size_t)(T - 1) * HCD, 1, VOCAB);
            topk(b.logits, 1, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
            logit_stats(b.logits, 1, VOCAB, VOCAB, b.tops, st);
            s.top_v.resize(TOPK + NSTAT); s.top_i.assign(TOPK + NSTAT, 0);
            CK(hipMemcpyAsync(s.top_v.data(), b.topv, TOPK * 4, hipMemcpyDeviceToHost, st));
            CK(hipMemcpyAsync(s.top_v.data() + TOPK, b.tops, NSTAT * 4, hipMemcpyDeviceToHost, st));
            CK(hipMemcpyAsync(s.top_i.data(), b.topi, TOPK * 4, hipMemcpyDeviceToHost, st));
        }
        CK(hipMemcpyAsync(s.h_last, b.res + (size_t)(T - 1) * HCD, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
        auto gram = [](int t) { return t >= IMG_BASE ? IMAGE_PAD : t; };
        s.hist[0] = T >= 2 ? gram(tokens[done + T - 2]) : s.hist[1];
        s.hist[1] = gram(tokens[done + T - 1]);
        const int end = s.n + T, blk = end / KPOOL * KPOOL;
        if (end % KPOOL && blk >= s.n) s.tail_pos = cell_pos(s, s.n, p3, blk);
        if (p3)
            for (int i = 0; i < T * 3; i++) s.pos = std::max(s.pos, p3[i] + 1);
        else
            s.pos += T;
        s.n = end;
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


static void keep_row(const float *v, const int *i, float *rows_v, int *rows_i, int at) {
    std::copy(v, v + TOPK, rows_v + (size_t)at * TOPK);
    std::copy(i, i + TOPK, rows_i + (size_t)at * TOPK);
}

static const float *top_stats(const Seq &s) {
    static const float unknown[NSTAT] = {(float)VOCAB, -1, 0, 0, 0, 0};
    return s.top_v.size() >= (size_t)(TOPK + NSTAT) ? s.top_v.data() + TOPK : unknown;
}

int Engine::image_encode(int slot, const float *patches, int gh, int gw, int frames) {
    const int rows = gh * gw / 4;
    if (slot < 0 || slot >= (INT32_MAX - IMG_BASE) / IMG_ROWS || rows > IMG_ROWS || vision.scratch_need(gh * gw) > barena.cap) return -1;
    if ((int)images.size() <= slot) images.resize(slot + 1);
    Image &im = images[slot];
    if (im.rows && im.n != rows) { CK(hipHostFree(im.rows)); im.rows = nullptr; }
    if (!im.rows) CK(hipHostMalloc(&im.rows, (size_t)rows * D * 2));
    im.n = rows;
    for (hipStream_t s : {st, cst}) if (s) CK(hipStreamSynchronize(s));
    const bool ok = vision.encode(patches, gh, gw, im.rows, barena.base, barena.cap, st, frames);
    CK(hipMemsetAsync(b.rr_counter, 0, 4, st));
    CK(hipMemsetAsync(la.counter, 0, 4, st));
    CK(hipStreamSynchronize(st));
    return ok ? 0 : -1;
}

int Engine::generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v, int *rows_i) {
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    int nout = 0;
    if (route_dec_file && ++route_dec_steps % 256 == 0)
        if (FILE *f = fopen(route_dec_file, "wb")) { fwrite(xc.total.data(), 8, xc.total.size(), f); fclose(f); }
    if (s.pending < 0) {
        if (s.top_v.empty()) FAIL("generate: nothing to continue from");
        s.pending = sample(s.seen, s.top_v.data(), s.top_i.data(), top_stats(s), sp, s.rng);
        if (rows_v) keep_row(s.top_v.data(), s.top_i.data(), rows_v, rows_i, nout);
        out[nout++] = s.pending;
        if (sp.lookup > 0) s.look.push(s.pending);
        if (is_eos(s.pending) || s.pending == sp.halt || nout == max_out) return nout;
    }
    {
        size_t rows[PLE_HEADS];
        ple_ids(s.hist[0], s.hist[1], &s.pending, 1, rows);
        ple_prefetch(rows, PLE_HEADS);
    }
    cpu.arm();
    const int room = std::min(3, max_out - nout - 1);
    int copied[4] = {s.pending};
    const int nl = sp.lookup > 0 ? s.look.find(std::min(sp.lookup, room), VOCAB, copied + 1) : 0;
    const int k = nl > 0 ? nl : std::max(0, std::min(sp.n_draft, room));
    const int q = s.n;
    ensure_pages(s, q + k + 1, q);

    CK(hipMemcpyAsync(b.draft, copied, (nl + 1) * 4, hipMemcpyHostToDevice, st));
    for (int i = 0; i < k && nl == 0; i++) {
        const Pos3 tail = cell_pos(s, q, nullptr, (q + i) / KPOOL * KPOOL);
        if (i == 0) mtp_forward(s, 1, q, s.h_last, b.draft, true, s.pos, nullptr, tail, nullptr);
        else {
            CK(hipMemcpyAsync(b.value, b.mtp_res, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
            mtp_forward(s, 1, q + i, b.value, b.draft + i, true, s.pos + i, nullptr, tail, nullptr);
        }
        hc_mix_run(*this, mtp_head, b.mtp_res, 1, nullptr);
        lm_head(b.hq, 1, DRAFT_VOCAB);
        argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1, st);
    }
    if (k > 0) {
        CK(hipMemcpyAsync(b.h_draft, b.draft, (k + 1) * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
    } else b.h_draft[0] = s.pending;
    return verify(s, sp, std::vector<int>(b.h_draft, b.h_draft + k + 1), false, out, nout, rows_v, rows_i);
}

int Engine::verify(Seq &s, const Sampling &sp, const std::vector<int> &drafts, bool masked, int *out, int nout, float *rows_v,
                   int *rows_i) {
    const int q = s.n, T = (int)drafts.size(), k = T - 1;
    forward(s, drafts.data(), T, k > 0);
    head(*this, hc_head, b.res, T, VOCAB);
    if (masked) mask_logits(b.logits, d_mask, VOCAB, T, st);
    topk(b.logits, T, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
    logit_stats(b.logits, T, VOCAB, VOCAB, b.tops, st);
    CK(hipMemcpyAsync(b.h_topv, b.topv, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(b.h_topi, b.topi, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(b.h_tops, b.tops, T * NSTAT * 4, hipMemcpyDeviceToHost, st));
    CK(hipEventRecord(topk_ready, st));
    CK(hipMemcpyAsync(b.tokens, b.draft, T * 4, hipMemcpyDeviceToDevice, st));
    mtp_catch_up(s, T, q, k > 0);
    CK(hipEventSynchronize(topk_ready));
    cpu.disarm();
    const float *tv = b.h_topv;
    const int *ti = b.h_topi;

    const int a = verify_drafts(s.seen, tv, ti, b.h_tops, drafts.data(), k, sp, s.rng,
                                [&](int t) { return is_eos(t) || t == sp.halt; },
                                [&](int r, int t) {
                                    if (rows_v) keep_row(&tv[r * TOPK], &ti[r * TOPK], rows_v, rows_i, nout);
                                    out[nout++] = t;
                                    if (sp.lookup > 0) s.look.push(t);
                                });
    const int next = out[nout - 1];

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
    if (a >= 1) { s.hist[0] = drafts[a - 1]; s.hist[1] = drafts[a]; }
    else { s.hist[0] = s.hist[1]; s.hist[1] = drafts[0]; }
    const int end = q + a + 1, blk = end / KPOOL * KPOOL;
    if (end % KPOOL && blk >= q) s.tail_pos = cell_pos(s, q, nullptr, blk);
    s.pos += a + 1;
    s.n = end;
    s.pending = next;
    {
        size_t rows[PLE_HEADS];
        ple_ids(s.hist[0], s.hist[1], &next, 1, rows);
        ple_prefetch(rows, PLE_HEADS);
    }
    return nout;
}

uint32_t *Engine::mask_rows() {
    if (!d_mask) CK(hipMalloc(&d_mask, (size_t)4 * (VOCAB / 32) * 4));
    return d_mask;
}

int Engine::draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out) {
    Seq &s = seqs.at(id);
    if (s.pending < 0) FAIL("draft: no pending token");
    const int room = std::min(3, max_k);
    if (room < 1) return 0;
    int copied[4] = {s.pending};
    const int nf = std::min(nfixed, room);
    std::copy(fixed, fixed + nf, copied + 1);
    const int nl = nf == 0 && sp.lookup > 0 ? s.look.find(std::min(sp.lookup, room), VOCAB, copied + 1) : 0;
    const int k = nl > 0 ? nl : std::min(room, nf + std::max(0, sp.n_draft));
    if (k > nf + nl) {
        const int q = s.n;
        ensure_pages(s, q + k + 1, q);
        CK(hipMemcpyAsync(b.draft, copied, (nf + 1) * 4, hipMemcpyHostToDevice, st));
        for (int i = 0; i < k; i++) {
            const Pos3 tail = cell_pos(s, q, nullptr, (q + i) / KPOOL * KPOOL);
            if (i == 0) mtp_forward(s, 1, q, s.h_last, b.draft, true, s.pos, nullptr, tail, nullptr);
            else {
                CK(hipMemcpyAsync(b.value, b.mtp_res, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
                mtp_forward(s, 1, q + i, b.value, b.draft + i, true, s.pos + i, nullptr, tail, nullptr);
            }
            if (i < nf) continue;
            hc_mix_run(*this, mtp_head, b.mtp_res, 1, nullptr);
            lm_head(b.hq, 1, DRAFT_VOCAB);
            argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1, st);
        }
        CK(hipMemcpyAsync(copied + 1 + nf, b.draft + 1 + nf, (k - nf) * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
    }
    std::copy(copied + 1, copied + 1 + k, out);
    return k;
}

int Engine::generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out) {
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    if (s.pending < 0) FAIL("generate_masked: no pending token");
    k = std::max(0, std::min({k, 3, max_out - 1}));
    ensure_pages(s, s.n + k + 1, s.n);
    std::vector<int> toks(k + 1, s.pending);
    std::copy(drafts, drafts + k, toks.begin() + 1);
    CK(hipMemcpyAsync(b.draft, toks.data(), (k + 1) * 4, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(mask_rows(), masks, (size_t)(k + 1) * (VOCAB / 32) * 4, hipMemcpyHostToDevice, st));
    cpu.arm();
    return verify(s, sp, toks, true, out, 0, nullptr, nullptr);
}

void Engine::generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v,
                            int *rows_i) {
    int need = 0;
    for (int i = 0; i < n; i++) {
        const Seq &s = seqs.at(ids[i]);
        const int last = (s.n + 2 + PAGE - 1) / PAGE;
        need += std::max(0, last - (int)s.pages.size());
        for (int pg = s.n / PAGE; pg < (int)s.pages.size() && pg < last; pg++) need += page_ref[s.pages[pg]] > 1;
    }
    kv_need(need);
    if (need > (int)free_pages.size()) throw std::runtime_error("out of KV pages");
    std::vector<int> act;
    for (int i = 0; i < n; i++) {
        Seq &s = seqs.at(ids[i]);
        const Sampling &sp = sps[i];
        int *o = out + (size_t)i * max_out;
        counts[i] = 0;
        if (max_out < 1) continue;
        if (s.pending < 0) {
            if (s.top_v.empty()) FAIL("generate: nothing to continue from");
            s.pending = sample(s.seen, s.top_v.data(), s.top_i.data(), top_stats(s), sp, s.rng);
            if (rows_v) keep_row(s.top_v.data(), s.top_i.data(), rows_v, rows_i, i * max_out + counts[i]);
            o[counts[i]++] = s.pending;
            if (sp.lookup > 0) s.look.push(s.pending);
            if (is_eos(s.pending) || s.pending == sp.halt || counts[i] == max_out) continue;
        }
        act.push_back(i);
    }
    if (act.empty()) return;
    if (act.size() > 8) FAIL("generate_batch: %zu sequences in one pass of 8 rows", act.size());
    const int per = std::min(2, 8 / (int)act.size());
    cpu.arm();
    std::vector<Part> parts;
    std::vector<int> k(n, 0), q(n, 0);
    int T = 0;
    for (int i : act) {
        Seq &s = seqs.at(ids[i]);
        k[i] = std::max(0, std::min({sps[i].n_draft, per - 1, max_out - counts[i] - 1}));
        q[i] = s.n;
        ensure_pages(s, q[i] + k[i] + 1, q[i]);
        CK(hipMemcpyAsync(b.draft + T, &s.pending, 4, hipMemcpyHostToDevice, st));
        for (int d = 0; d < k[i]; d++) {
            const Pos3 tail = cell_pos(s, q[i], nullptr, (q[i] + d) / KPOOL * KPOOL);
            if (d == 0) mtp_forward(s, 1, q[i], s.h_last, b.draft + T, true, s.pos, nullptr, tail, nullptr);
            else {
                CK(hipMemcpyAsync(b.value, b.mtp_res, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
                mtp_forward(s, 1, q[i] + d, b.value, b.draft + T + d, true, s.pos + d, nullptr, tail, nullptr);
            }
            hc_mix_run(*this, mtp_head, b.mtp_res, 1, nullptr);
            lm_head(b.hq, 1, DRAFT_VOCAB);
            argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + T + d + 1, st);
        }
        parts.push_back({&s, k[i] + 1, T, k[i] > 0});
        T += k[i] + 1;
    }
    std::vector<int> drafts(T);
    CK(hipMemcpyAsync(drafts.data(), b.draft, T * 4, hipMemcpyDeviceToHost, st));
    CK(hipStreamSynchronize(st));
    forward_parts(parts.data(), (int)parts.size(), drafts.data(), T);
    head(*this, hc_head, b.res, T, VOCAB, true);
    topk(b.logits, T, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
    logit_stats(b.logits, T, VOCAB, VOCAB, b.tops, st);
    CK(hipMemcpyAsync(b.h_topv, b.topv, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(b.h_topi, b.topi, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(b.h_tops, b.tops, T * NSTAT * 4, hipMemcpyDeviceToHost, st));
    CK(hipEventRecord(topk_ready, st));
    for (const Part &p : parts) mtp_catch_up(*p.s, p.T, p.s->n, p.verify, nullptr, false, p.off);
    CK(hipEventSynchronize(topk_ready));
    cpu.disarm();
    const float *tv = b.h_topv;
    const int *ti = b.h_topi;
    for (size_t j = 0; j < parts.size(); j++) {
        const int i = act[j];
        const Part &p = parts[j];
        Seq &s = *p.s;
        const Sampling &sp = sps[i];
        int *o = out + (size_t)i * max_out;
        const int *dr = drafts.data() + p.off;
        const int a = verify_drafts(s.seen, &tv[p.off * TOPK], &ti[p.off * TOPK], &b.h_tops[p.off * NSTAT], dr, k[i], sp, s.rng,
                                    [&](int t) { return is_eos(t) || t == sp.halt; },
                                    [&](int r, int t) {
                                        if (rows_v) keep_row(&tv[(p.off + r) * TOPK], &ti[(p.off + r) * TOPK], rows_v, rows_i,
                                                             i * max_out + counts[i]);
                                        o[counts[i]++] = t;
                                        if (sp.lookup > 0) s.look.push(t);
                                    });
        const int next = o[counts[i] - 1];
        if (k[i] > 0) {
            GdnScanArgs sa{};
            sa.S = s.ssm; sa.S_layer = (size_t)GV * GD * GD;
            sa.kn = b.v_kn + (size_t)p.off * GK * GD; sa.in_layer = (size_t)8 * GK * GD;
            sa.vc = b.v_vc + (size_t)p.off * GV * GD; sa.v_layer = (size_t)8 * GV * GD;
            sa.ab = b.v_ab + (size_t)p.off * 2 * GV; sa.ab_layer = (size_t)8 * 2 * GV;
            sa.A = gdn_A; sa.dtb = gdn_dt; sa.T = a + 1; sa.write_state = 1; sa.out = nullptr;
            gdn_scan(sa, NGDN, st);
            gdn_conv_commit(s.conv, b.v_raw + (size_t)p.off * QKV, a + 1, 8 * QKV, NGDN, st);
            ple_commit(s.ple, b.xconv + (size_t)p.off * HCD, a + 1, st);
            idx_tail_commit(s.idx_tail, b.idx_raw + (size_t)p.off * IDX_D, (size_t)umax * IDX_D, NKVL, q[i], q[i] + a + 1, st);
        }
        CK(hipMemcpyAsync(s.h_last, b.res + (size_t)(p.off + a) * HCD, (size_t)HCD * 4, hipMemcpyDeviceToDevice, st));
        if (a >= 1) { s.hist[0] = dr[a - 1]; s.hist[1] = dr[a]; }
        else { s.hist[0] = s.hist[1]; s.hist[1] = dr[0]; }
        const int end = q[i] + a + 1, blk = end / KPOOL * KPOOL;
        if (end % KPOOL && blk >= q[i]) s.tail_pos = cell_pos(s, q[i], nullptr, blk);
        s.pos += a + 1;
        s.n = end;
        s.pending = next;
        size_t rows[PLE_HEADS];
        ple_ids(s.hist[0], s.hist[1], &next, 1, rows);
        ple_prefetch(rows, PLE_HEADS);
    }
    CK(hipStreamSynchronize(st));
}
