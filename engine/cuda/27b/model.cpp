#include "model.h"
#include "../../gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>


int Engine::seq_new() {
    for (size_t i = 0; i < seqs.size(); i++) {
        if (seqs[i].used) continue;
        seqs[i].used = true;
        seq_reset(seqs[i]);
        return (int)i;
    }
    return -1;
}

void Engine::settle(Seq &s) {
    if (s.restoring) CK(cudaStreamWaitEvent(st, s.ev[11], 0));
    s.restoring = false;
}

void Engine::seq_reset(Seq &s) {
    settle(s);
    for (int p : s.pages) page_release(p);
    s.n = 0; s.pos = 0; s.pending = -1; s.pages.clear(); s.seen.clear(); s.top_v.clear(); s.top_i.clear();
    s.look.clear();
    s.rng.seed(std::random_device{}());
    CK(cudaMemsetAsync(s.ssm, 0, (size_t)48 * GV * GD * GD * 4, st));
    CK(cudaMemsetAsync(s.conv, 0, (size_t)48 * QKV * 3 * 4, st));
    CK(cudaMemsetAsync(s.h_last, 0, D * 4, st));
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
    reclaim(1);
    if (free_pages.empty()) throw std::runtime_error("out of KV pages");
    const int p = free_pages.back();
    free_pages.pop_back();
    page_ref[p] = 1;
    return p;
}

void Engine::page_release(int p) {
    if (--page_ref[p] == 0) free_pages.push_back(p);
}

void Engine::ensure_pages(Seq &s, int n_tokens, int from) {
    const int need = (n_tokens + PAGE - 1) / PAGE;
    int dirty = (int)s.pages.size();
    int shared = 0;
    for (int i = from / PAGE; i < dirty && i < need; i++) shared += page_ref[s.pages[i]] > 1;
    reclaim(need - dirty + shared);
    if ((int)free_pages.size() < need - dirty + shared) throw std::runtime_error("out of KV pages");
    while ((int)s.pages.size() < need) {
        const int p = page_alloc();
        CK(cudaMemsetAsync(kv_pool + (size_t)p * KV_PAGE, 0, KV_PAGE, st));
        s.pages.push_back(p);
    }
    for (int i = from / PAGE; i < dirty && i < need; i++) {
        if (page_ref[s.pages[i]] == 1) continue;
        const int np = page_alloc();
        CK(cudaMemcpyAsync(kv_pool + (size_t)np * KV_PAGE, kv_pool + (size_t)s.pages[i] * KV_PAGE, KV_PAGE, cudaMemcpyDeviceToDevice, st));
        page_release(s.pages[i]);
        s.pages[i] = np;
        dirty = std::min(dirty, i);
    }
    if (dirty < need)
        CK(cudaMemcpyAsync(s.d_pages + dirty, s.pages.data() + dirty, (need - dirty) * 4, cudaMemcpyHostToDevice, st));
    lend();
}


int Engine::snapshot(int id) {
    Seq &s = seqs.at(id);
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    if (!sn.host) CK(cudaMallocHost(&sn.host, state_bytes()));
    sn.used = true;
    sn.n = s.n;
    sn.pos = s.pos;
    if (s.pending < 0) { sn.top_v = s.top_v; sn.top_i = s.top_i; } else { sn.top_v.clear(); sn.top_i.clear(); }
    sn.pages.assign(s.pages.begin(), s.pages.begin() + (s.n + PAGE - 1) / PAGE);
    for (int p : sn.pages) page_ref[p]++;
    const size_t ssm = (size_t)48 * GV * GD * GD * 4, conv = (size_t)48 * QKV * 3 * 4;
    settle(s);
    CK(cudaStreamWaitEvent(st, copied, 0));
    CK(cudaMemcpyAsync(stage_state, s.ssm, ssm, cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(stage_state + ssm, s.conv, conv, cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(stage_state + ssm + conv, s.h_last, D * 4, cudaMemcpyDeviceToDevice, st));
    CK(cudaEventRecord(staged, st));
    CK(cudaStreamWaitEvent(cst, staged, 0));
    CK(cudaMemcpyAsync(sn.host, stage_state, state_bytes(), cudaMemcpyDeviceToHost, cst));
    CK(cudaEventRecord(copied, cst));
    return (int)si;
}

void Engine::restore(int id, int snap) {
    Seq &s = seqs.at(id);
    Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("restore of a freed snapshot");
    CK(cudaStreamWaitEvent(st, copied, 0));
    for (int p : s.pages) page_release(p);
    s.pages = sn.pages;
    for (int p : s.pages) page_ref[p]++;
    if (!s.pages.empty()) CK(cudaMemcpyAsync(s.d_pages, s.pages.data(), s.pages.size() * 4, cudaMemcpyHostToDevice, st));
    s.n = sn.n;
    s.pos = sn.pos;
    s.pending = -1;
    s.top_v = sn.top_v; s.top_i = sn.top_i; s.seen.clear();
    const uint8_t *h = (const uint8_t *)sn.host;
    const size_t ssm = (size_t)GV * GD * GD * 4, conv = (size_t)QKV * 3 * 4;
    if (!s.ev[0]) for (cudaEvent_t &e : s.ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CK(cudaEventRecord(staged, st));
    CK(cudaStreamWaitEvent(cst, staged, 0));
    for (int gl = 0; gl < 48; gl += 4) {
        CK(cudaMemcpyAsync((uint8_t *)s.ssm + gl * ssm, h + gl * ssm, 4 * ssm, cudaMemcpyHostToDevice, cst));
        CK(cudaMemcpyAsync((uint8_t *)s.conv + gl * conv, h + 48 * ssm + gl * conv, 4 * conv, cudaMemcpyHostToDevice, cst));
        CK(cudaEventRecord(s.ev[gl / 4], cst));
    }
    s.restoring = true;
    CK(cudaMemcpyAsync(s.h_last, h + 48 * (ssm + conv), D * 4, cudaMemcpyHostToDevice, st));
}

struct SnapHeader { char magic[8]; uint64_t kv_page, state; int32_t n, pos, n_top, n_pages; };
static const char SNAP_MAGIC[8] = {'K', 'I', 'L', 'N', 'S', 'N', 'P', '1'};

size_t Engine::snap_bytes(int snap) const {
    const Snap &sn = snaps.at(snap);
    return sizeof(SnapHeader) + sn.top_v.size() * 8 + state_bytes() + sn.pages.size() * KV_PAGE;
}

void Engine::snap_export(int snap, uint8_t *dst) {
    const Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("export of a freed snapshot");
    CK(cudaStreamSynchronize(st));
    CK(cudaStreamSynchronize(cst));
    SnapHeader h;
    memcpy(h.magic, SNAP_MAGIC, 8);
    h.kv_page = KV_PAGE; h.state = state_bytes(); h.n = sn.n; h.pos = sn.pos;
    h.n_top = (int32_t)sn.top_v.size(); h.n_pages = (int32_t)sn.pages.size();
    uint8_t *p = dst;
    memcpy(p, &h, sizeof h); p += sizeof h;
    memcpy(p, sn.top_v.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
    memcpy(p, sn.top_i.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
    memcpy(p, sn.host, h.state); p += h.state;
    for (int pg : sn.pages) {
        CK(cudaMemcpyAsync(p, kv_pool + (size_t)pg * KV_PAGE, KV_PAGE, cudaMemcpyDeviceToHost, cst));
        p += KV_PAGE;
    }
    CK(cudaStreamSynchronize(cst));
}

int Engine::snap_import(int64_t from, int64_t off, size_t bytes) {
    const os::File file = os::from_fd(from);
    auto get = [&](void *dst, size_t n) {
        for (size_t got = 0; got < n;) {
            const int64_t r = os::read_at(file, (uint8_t *)dst + got, n - got, (uint64_t)(off + (int64_t)got));
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
    reclaim(h.n_pages);
    if ((int)free_pages.size() < h.n_pages) return -2;
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    if (!sn.host) CK(cudaMallocHost(&sn.host, state_bytes()));
    sn.top_v.resize(h.n_top);
    sn.top_i.resize(h.n_top);
    bool ok = get(sn.top_v.data(), (size_t)h.n_top * 4) && get(sn.top_i.data(), (size_t)h.n_top * 4);
    const int64_t state_at = off;
    off += (int64_t)h.state;
    const int per = (int)(h.state / 2 / KV_PAGE);
    cudaEvent_t ev[2];
    for (cudaEvent_t &e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    sn.pages.clear();
    for (int i = 0; ok && i < h.n_pages; i += per) {
        const int half = i / per % 2, n = std::min(per, h.n_pages - i);
        uint8_t *buf = (uint8_t *)sn.host + (size_t)half * per * KV_PAGE;
        if (i >= 2 * per) CK(cudaEventSynchronize(ev[half]));
        ok = get(buf, (size_t)n * KV_PAGE);
        for (int k = 0; ok && k < n; k++) {
            sn.pages.push_back(page_alloc());
            CK(cudaMemcpyAsync(kv_pool + (size_t)sn.pages.back() * KV_PAGE, buf + (size_t)k * KV_PAGE, KV_PAGE, cudaMemcpyHostToDevice, st));
        }
        CK(cudaEventRecord(ev[half], st));
    }
    CK(cudaStreamSynchronize(st));
    for (cudaEvent_t e : ev) CK(cudaEventDestroy(e));
    off = state_at;
    if (!ok || !get(sn.host, h.state)) {
        for (int p : sn.pages) page_release(p);
        sn.pages.clear();
        return -1;
    }
    sn.n = h.n;
    sn.pos = h.pos;
    sn.used = true;
    return (int)si;
}

void Engine::snap_free(int snap) {
    Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("double free of a snapshot");
    CK(cudaStreamSynchronize(st));
    CK(cudaStreamSynchronize(cst));
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
    const int Tp = std::min(umax, (T + 63) / 64 * 64);
    LtPlan &p = lt_plan(N, K, Tp, ldx, ldd, false, W, x, dst);
    if (cublasLtMatmul(lt, p.desc, &alpha, W, p.a, x, p.b, &beta, dst, p.c, dst, p.c, &p.algo, lt_ws, lt_ws_size, st) != CUBLAS_STATUS_SUCCESS)
        FAIL("cublasLtMatmul failed (%dx%dx%d)", N, K, T);
    if (dst != y) store_f16(b.gemm16, N, T, (float *)y, ldy, accumulate, st);
}

void Engine::gemm32(const half *W, int N, int K, const half *x, int ldx, int T, float *y, int ldy) {
    const float alpha = 1.0f, beta = 0.0f;
    const int Tp = std::min(umax, (T + 63) / 64 * 64);
    LtPlan &p = lt_plan(N, K, Tp, ldx, ldy, true, W, x, y);
    if (cublasLtMatmul(lt, p.desc, &alpha, W, p.a, x, p.b, &beta, y, p.c, y, p.c, &p.algo, lt_ws, lt_ws_size, st) != CUBLAS_STATUS_SUCCESS)
        FAIL("cublasLtMatmul failed (%dx%dx%d)", N, K, T);
}

void Engine::linear(const QTensor &W, int T, const QAct &qin, const half *, int, void *out, int ldo, int mode, bool out_f16) {
    if (W.type == F16 || W.type == F32) FAIL("linear: %d is not a quantized type", W.type);
    if (T <= 8) gemv(W, qin, T, (float *)out, ldo, mode, st);
    else gemm_q(W, qin, T, out, ldo, mode, out_f16, st);
}

void Engine::residual(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, float *x, const float *next, float *nf) {
    if (T <= 8) {
        gemv(W, qin, T, x, D, GEMV_ADD, st);
        if (next) norm_rows(x, next, T, nf, b.hh, b.hq, st);
    } else {
        linear(W, T, qin, hin, ldin, x, D, GEMV_ADD);
        if (next) rmsnorm(x, next, T, D, D, D, nf, b.hh, b.hq, st);
    }
}

void Engine::attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, int rope0, const int *rpos, float *x) {
    if (T <= 8) {
        const Part p{&s, T, 0, pos0, rope0, rpos, false};
        attn_parts(w, kvl, &p, 1, T, x);
        return;
    }
    linear(w.q, T, b.hq, nullptr, D, b.proj, 2 * NH * HD, GEMV_STORE);
    linear(w.k, T, b.hq, nullptr, D, b.k, NKV * HD, GEMV_STORE);
    linear(w.v, T, b.hq, nullptr, D, b.v, NKV * HD, GEMV_STORE);
    attn_prep(b.proj, b.k, b.v, w.q_norm, w.k_norm, T, pos0, rope0, rpos, kv(s), kvl, b.Q16, st);
    AttnArgs a;
    a.Q16 = b.Q16; a.kv = kv(s); a.layer = kvl; a.T = T; a.pos0 = pos0; a.qg = b.proj;
    a.part_o = b.part_o; a.part_ml = b.part_ml;
    attention(a, nullptr, b.mq, st);
    residual(w.o, T, b.mq, nullptr, NH * HD, x, w.post_norm, nullptr);
}

void Engine::attn_parts(const Layer &w, int kvl, const Part *p, int np, int T, float *x) {
    GemvArgs g;
    g.seg[0] = {w.q, b.proj, 2 * NH * HD}; g.seg[1] = {w.k, b.k, NKV * HD}; g.seg[2] = {w.v, b.v, NKV * HD};
    g.nseg = 3; g.x = b.hq; g.T = T;
    gemv(g, st);
    for (int i = 0; i < np; i++) {
        const Part &q = p[i];
        float *proj = b.proj + (size_t)q.off * 2 * NH * HD;
        half *Q16 = b.Q16 + (size_t)q.off * NH * HD;
        attn_prep(proj, b.k + (size_t)q.off * NKV * HD, b.v + (size_t)q.off * NKV * HD, w.q_norm, w.k_norm, q.T, q.pos0,
                  q.rope0, q.rpos, kv(*q.s), kvl, Q16, st);
        AttnArgs a;
        a.Q16 = Q16; a.kv = kv(*q.s); a.layer = kvl; a.T = q.T; a.pos0 = q.pos0; a.qg = proj;
        a.part_o = b.part_o; a.part_ml = b.part_ml;
        attention(a, nullptr, QAct{b.mq.q + (size_t)q.off * NH * HD, b.mq.d + (size_t)q.off * NH * HD / 32}, st);
    }
    residual(w.o, T, b.mq, b.mix, NH * HD, x, w.post_norm, nullptr);
}

void Engine::gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *x) {
    if (T <= 8) {
        const Part p{&s, T, 0, 0, 0, nullptr, verify};
        gdn_parts(w, gl, &p, 1, T, x);
        return;
    }
    float *raw = b.proj, *z = b.proj + (size_t)T * QKV, *ab = b.k;
    linear(w.qkv, T, b.hq, nullptr, D, raw, QKV, GEMV_STORE);
    linear(w.z, T, b.hq, nullptr, D, z, GV * GD, GEMV_STORE);
    gemm32((const half *)w.ab.f, 2 * GV, D, b.hh, D, T, ab, 2 * GV);
    if (s.restoring) CK(cudaStreamWaitEvent(st, s.ev[gl / 4], 0));
    gdn_conv(raw, T, w.conv, s.conv + (size_t)gl * QKV * 3, true, b.qn, b.kn, b.vc, st);
    GdnScanArgs sa{};
    sa.S = s.ssm + (size_t)gl * GV * GD * GD; sa.qn = b.qn; sa.kn = b.kn; sa.vc = b.vc; sa.ab = ab;
    sa.A = gdn_A + gl * GV; sa.dtb = gdn_dt + gl * GV; sa.T = T; sa.write_state = 1; sa.out = b.vc;
    gdn_scan(sa, 1, st);
    gdn_out(b.vc, z, w.ssm_norm, T, nullptr, b.mq, st);
    residual(w.ssm_out, T, b.mq, nullptr, GV * GD, x, w.post_norm, nullptr);
}

void Engine::gdn_parts(const Layer &w, int gl, const Part *p, int np, int T, float *x) {
    float *raw = b.v_raw + (size_t)gl * 8 * QKV;
    GemvArgs a;
    a.seg[0] = {w.qkv, raw, QKV}; a.seg[1] = {w.z, b.proj, GV * GD}; a.nseg = 2; a.x = b.hq; a.T = T;
    gemv(a, st);
    for (int i = 0; i < np; i++) {
        const Part &q = p[i];
        GdnStepArgs g;
        g.raw = raw + (size_t)q.off * QKV; g.z = b.proj + (size_t)q.off * GV * GD; g.hh = b.hh + (size_t)q.off * D;
        g.Wab = (const half *)w.ab.f;
        g.conv_w = w.conv; g.A = gdn_A + gl * GV; g.dt = gdn_dt + gl * GV; g.norm_w = w.ssm_norm;
        g.conv_state = q.s->conv + (size_t)gl * QKV * 3; g.S = q.s->ssm + (size_t)gl * GV * GD * GD;
        g.T = q.T; g.verify = q.verify;
        g.kn_out = b.v_kn + ((size_t)gl * 8 + q.off) * GK * GD; g.vc_out = b.v_vc + ((size_t)gl * 8 + q.off) * GV * GD;
        g.ab_out = b.v_ab + ((size_t)gl * 8 + q.off) * 2 * GV;
        g.out = QAct{b.mq.q + (size_t)q.off * GV * GD, b.mq.d + (size_t)q.off * GV * GD / 32};
        if (q.s->restoring) CK(cudaStreamWaitEvent(st, q.s->ev[gl / 4], 0));
        gdn_step(g, st);
    }
    residual(w.ssm_out, T, b.mq, nullptr, GV * GD, x, w.post_norm, nullptr);
}

void Engine::ffn_block(const Layer &w, int T, float *x, const float *next, float *nf) {
    if (T <= 8) {
        GemvArgs a;
        a.seg[0] = {w.ffn_gate, nullptr, 0}; a.up = w.ffn_up; a.x = b.hq; a.T = T; a.mode = GEMV_GLU; a.gq = b.aq;
        gemv(a, st);
        residual(w.ffn_down, T, b.aq, nullptr, FF, x, next, nf);
    } else {
        gemm_q_glu(w.ffn_gate, w.ffn_up, b.hq, T, b.aq, st);
        residual(w.ffn_down, T, b.aq, nullptr, FF, x, next, nf);
    }
}

void Engine::forward(Seq &s, int T, bool verify, int rope0, bool rpos, bool img) {
    ensure_pages(s, s.n + T, s.n);
    embed(embd, b.tokens, img ? b.img : nullptr, T, b.x, st);
    if (T <= 8) norm_rows(b.x, L[0].attn_norm, T, nullptr, b.hh, b.hq, st);
    else rmsnorm(b.x, L[0].attn_norm, T, D, D, D, nullptr, b.hh, b.hq, st);
    const int sf = spill_from();
    const bool stage = T > 8 && sf < NL;
    if (stage) for (int l = sf; l < std::min(NL, sf + 2); l++) stage_layer(l);
    for (int l = 0, gl = 0; l < NL; l++) {
        const Layer &w = layer_for(l, stage);
        if (is_attn_layer(l)) attn_block(w, l / 4, s, T, s.n, rope0, rpos ? b.rpos : nullptr, b.x);
        else gdn_block(w, gl++, s, T, verify, b.x);
        const bool last = l == NL - 1;
        ffn_block(w, T, b.x, last ? out_norm : L[l + 1].attn_norm, last ? b.hfin : nullptr);
        layer_done(l, stage);
    }
    s.restoring = false;
    if (T <= 8 && !verify) gdn_conv_commit(s.conv, b.v_raw, T, 8 * QKV, 48, st);
}

void Engine::forward_parts(const Part *p, int np, int T) {
    for (int i = 0; i < np; i++) ensure_pages(*p[i].s, p[i].s->n + p[i].T, p[i].s->n);
    embed(embd, b.tokens, nullptr, T, b.x, st);
    norm_rows(b.x, L[0].attn_norm, T, nullptr, b.hh, b.hq, st);
    for (int l = 0, gl = 0; l < NL; l++) {
        if (is_attn_layer(l)) attn_parts(L[l], l / 4, p, np, T, b.x);
        else gdn_parts(L[l], gl++, p, np, T, b.x);
        const bool last = l == NL - 1;
        ffn_block(L[l], T, b.x, last ? out_norm : L[l + 1].attn_norm, last ? b.hfin : nullptr);
    }
    for (int i = 0; i < np; i++) {
        p[i].s->restoring = false;
        if (!p[i].verify) gdn_conv_commit(p[i].s->conv, b.v_raw + (size_t)p[i].off * QKV, p[i].T, 8 * QKV, 48, st);
    }
}

void Engine::mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *tokens, int rope0, bool rpos, bool img) {
    const bool small = T <= 8;
    float *xm = small ? b.mtp_x : b.mtp_h;
    embed(embd, tokens, img ? b.img : nullptr, T, b.x, st);
    rmsnorm(b.x, enorm, T, D, D, 2 * D, nullptr, nullptr, b.hq, st);
    rmsnorm(h_prev, hnorm, T, D, D, 2 * D, nullptr, nullptr, QAct{b.hq.q + D, b.hq.d + D / 32, b.hq.r ? b.hq.r + D : nullptr}, st);
    if (small) {
        gemv(eh_proj, b.hq, T, xm, D, GEMV_STORE, st);
        norm_rows(xm, M.attn_norm, T, nullptr, b.hh, b.hq, st);
    } else {
        linear(eh_proj, T, b.hq, nullptr, 2 * D, xm, D, GEMV_STORE);
        rmsnorm(xm, M.attn_norm, T, D, D, D, nullptr, b.hh, b.hq, st);
    }
    attn_block(M, NATT, s, T, pos0, rope0, rpos ? b.rpos : nullptr, xm);
    ffn_block(M, T, xm, small ? head_norm : nullptr, small ? b.mtp_x : nullptr);
}

static void rebase(QTensor &t, const uint8_t *from, size_t n, uint8_t *to) {
    auto move = [&](auto *&p) {
        if (p && (const uint8_t *)p >= from && (const uint8_t *)p < from + n) p = (std::remove_reference_t<decltype(p)>)(to + ((const uint8_t *)p - from));
    };
    move(t.q); move(t.qh); move(t.sc); move(t.sh); move(t.d); move(t.f);
}
static void rebase(float *&p, const uint8_t *from, size_t n, uint8_t *to) {
    if (p && (const uint8_t *)p >= from && (const uint8_t *)p < from + n) p = (float *)(to + ((const uint8_t *)p - from));
}

static Layer rebased(Layer w, const uint8_t *from, size_t n, uint8_t *to) {
    for (QTensor *t : {&w.ffn_gate, &w.ffn_up, &w.ffn_down, &w.qkv, &w.z, &w.ab, &w.ssm_out, &w.q, &w.k, &w.v, &w.o}) rebase(*t, from, n, to);
    for (float **p : {&w.attn_norm, &w.post_norm, &w.conv, &w.ssm_norm, &w.q_norm, &w.k_norm}) rebase(*p, from, n, to);
    return w;
}

int Engine::lent_pages() const {
    int n = 0;
    for (int l = first_spill; l < spill_from(); l++) n += lent[l].pages;
    return n;
}

void Engine::lend() {
    constexpr int MARGIN = 64;
    if (!lend_on) return;
    std::vector<uint8_t> free_at(kv_seg.size(), 0);
    for (int p : free_pages) free_at[p] = 1;
    while (spill_from() < NL) {
        const int l = spill_from(), need = (int)((spill[l].bytes + KV_PAGE - 1) / KV_PAGE);
        if ((int)free_pages.size() - need < MARGIN) break;
        int at = -1;
        for (int p = (int)kv_seg.size() - 1, run = 0; p >= 0; p--) {
            run = free_at[p] ? (run && kv_seg[p] == kv_seg[p + 1] ? run + 1 : 1) : 0;
            if (run == need) { at = p; break; }
        }
        if (at < 0) break;
        for (int p = at; p < at + need; p++) free_at[p] = 0;
        free_pages.erase(std::remove_if(free_pages.begin(), free_pages.end(), [&](int p) { return p >= at && p < at + need; }), free_pages.end());
        uint8_t *dst = kv_pool + (size_t)at * KV_PAGE;
        CK(cudaMemcpyAsync(dst, spill[l].host, spill[l].bytes, cudaMemcpyHostToDevice, st));
        L[l] = rebased(spill_L[l], spill[l].host, spill[l].bytes, dst);
        lent[l] = {at, need};
        n_lent++;
    }
}

void Engine::reclaim(int pages) {
    bool synced = false;
    while ((int)free_pages.size() < pages && n_lent > 0) {
        if (!synced) { CK(cudaStreamSynchronize(st)); CK(cudaStreamSynchronize(cst)); synced = true; }
        const int l = spill_from() - 1;
        L[l] = spill_L[l];
        for (int p = lent[l].page; p < lent[l].page + lent[l].pages; p++) free_pages.push_back(p);
        lent[l] = {};
        n_lent--;
    }
}

void Engine::stage_layer(int l) {
    const int k = (l - spill_from()) % 2;
    if (!stage_L[k][l].attn_norm) stage_L[k][l] = rebased(spill_L[l], spill[l].host, spill[l].bytes, stage_slot[k]);
    CK(cudaStreamWaitEvent(cst, stage_used[k], 0));
    CK(cudaMemcpyAsync(stage_slot[k], spill[l].host, spill[l].bytes, cudaMemcpyHostToDevice, cst));
    CK(cudaEventRecord(stage_copied[k], cst));
}

const Layer &Engine::layer_for(int l, bool stage) {
    if (!stage || l < spill_from()) return L[l];
    const int k = (l - spill_from()) % 2;
    CK(cudaStreamWaitEvent(st, stage_copied[k], 0));
    return stage_L[k][l];
}

void Engine::layer_done(int l, bool stage) {
    if (!stage || l < spill_from()) return;
    CK(cudaEventRecord(stage_used[(l - spill_from()) % 2], st));
    if (l + 2 < NL) stage_layer(l + 2);
}

void Engine::lm_head(const QAct &in, int T, int rows) {
    QTensor w = out;
    w.rows = rows;
    gemv(w, in, T, b.logits, VOCAB, GEMV_STORE, st);
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
        const int T = std::min(umax, n - done);
        CK(cudaMemcpyAsync(b.tokens, tokens + done, T * 4, cudaMemcpyHostToDevice, st));
        const bool img = any_img && std::any_of(rows.begin() + done, rows.begin() + done + T, [](const half *r) { return r; });
        if (img) CK(cudaMemcpyAsync(b.img, rows.data() + done, T * sizeof(half *), cudaMemcpyHostToDevice, st));
        if (pos3) CK(cudaMemcpyAsync(b.rpos, pos3 + (size_t)done * 3, T * 12, cudaMemcpyHostToDevice, st));
        const int rope0 = s.pos;
        forward(s, T, false, rope0, pos3, img);
        mtp_catch_up(s, T, s.n, rope0, pos3, img);
        if (done + T == n) {
            quant_act(b.hfin + (size_t)(T - 1) * D, D, b.hq, st);
            lm_head(b.hq, 1, VOCAB);
            topk(b.logits, 1, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
            logit_stats(b.logits, 1, VOCAB, VOCAB, b.tops, st);
            s.top_v.resize(TOPK + NSTAT); s.top_i.assign(TOPK + NSTAT, 0);
            CK(cudaMemcpyAsync(s.top_v.data(), b.topv, TOPK * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(s.top_v.data() + TOPK, b.tops, NSTAT * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(s.top_i.data(), b.topi, TOPK * 4, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaMemcpyAsync(s.h_last, b.hfin + (size_t)(T - 1) * D, D * 4, cudaMemcpyDeviceToDevice, st));
        s.n += T;
        if (pos3)
            for (int i = done * 3; i < (done + T) * 3; i++) s.pos = std::max(s.pos, pos3[i] + 1);
        else
            s.pos += T;
        done += T;
    }
    CK(cudaStreamSynchronize(st));
}

int Engine::image_encode(int slot, const float *patches, int gh, int gw, int frames) {
    const int rows = gh * gw / 4;
    if (slot < 0 || slot >= (INT32_MAX - IMG_BASE) / IMG_ROWS || rows > IMG_ROWS || vision.scratch_need(gh * gw) > barena.cap) return -1;
    if ((int)images.size() <= slot) images.resize(slot + 1);
    Image &im = images[slot];
    if (im.rows && im.n != rows) { CK(cudaFreeHost(im.rows)); im.rows = nullptr; }
    if (!im.rows) CK(cudaMallocHost(&im.rows, (size_t)rows * D * 2));
    im.n = rows;
    CK(cudaStreamSynchronize(st));
    return vision.encode(patches, gh, gw, im.rows, barena.base, barena.cap, st, frames) ? 0 : -1;
}

void Engine::mtp_catch_up(Seq &s, int T, int pos0, int rope0, bool rpos, bool img, int off) {
    CK(cudaMemcpyAsync(b.mtp_h, s.h_last, D * 4, cudaMemcpyDeviceToDevice, st));
    if (T > 1) CK(cudaMemcpyAsync(b.mtp_h + D, b.hfin + (size_t)off * D, (size_t)(T - 1) * D * 4, cudaMemcpyDeviceToDevice, st));
    mtp_forward(s, T, pos0, b.mtp_h, b.tokens + off, rope0, rpos, img);
}

static void keep_row(const float *v, const int *i, float *rows_v, int *rows_i, int at) {
    std::copy(v, v + TOPK, rows_v + (size_t)at * TOPK);
    std::copy(i, i + TOPK, rows_i + (size_t)at * TOPK);
}

static const float *top_stats(const Seq &s) {
    static const float unknown[NSTAT] = {(float)VOCAB, -1, 0, 0, 0, 0};
    return s.top_v.size() >= (size_t)(TOPK + NSTAT) ? s.top_v.data() + TOPK : unknown;
}

void Engine::generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v,
                            int *rows_i) {
    int need = 0;
    for (int i = 0; i < n; i++) {
        const Seq &s = seqs.at(ids[i]);
        const int last = (s.n + 8 + PAGE - 1) / PAGE;
        need += std::max(0, last - (int)s.pages.size());
        for (int pg = s.n / PAGE; pg < (int)s.pages.size() && pg < last; pg++) need += page_ref[s.pages[pg]] > 1;
    }
    reclaim(need);
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
    const int per = 8 / (int)act.size();
    std::vector<Part> parts;
    std::vector<int> k(n, 0), q(n, 0);
    int T = 0;
    for (int i : act) {
        Seq &s = seqs.at(ids[i]);
        k[i] = std::max(0, std::min({sps[i].n_draft, per - 1, max_out - counts[i] - 1}));
        q[i] = s.n;
        ensure_pages(s, q[i] + k[i] + 1, q[i]);
        CK(cudaMemcpyAsync(b.draft + T, &s.pending, 4, cudaMemcpyHostToDevice, st));
        for (int d = 0; d < k[i]; d++) {
            mtp_forward(s, 1, q[i] + d, d == 0 ? s.h_last : b.mtp_x, b.draft + T + d, s.pos + d);
            lm_head(b.hq, 1, DRAFT_VOCAB);
            argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + T + d + 1, st);
        }
        parts.push_back({&s, k[i] + 1, T, q[i], s.pos, nullptr, k[i] > 0});
        T += k[i] + 1;
    }
    CK(cudaMemcpyAsync(b.tokens, b.draft, T * 4, cudaMemcpyDeviceToDevice, st));
    forward_parts(parts.data(), (int)parts.size(), T);
    lm_head(b.hq, T, VOCAB);
    topk(b.logits, T, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
    logit_stats(b.logits, T, VOCAB, VOCAB, b.tops, st);
    std::vector<float> tv(T * TOPK), ts(T * NSTAT);
    std::vector<int> ti(T * TOPK), drafts(T);
    CK(cudaMemcpyAsync(tv.data(), b.topv, T * TOPK * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(ti.data(), b.topi, T * TOPK * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(ts.data(), b.tops, T * NSTAT * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(drafts.data(), b.draft, T * 4, cudaMemcpyDeviceToHost, st));
    for (const Part &p : parts) mtp_catch_up(*p.s, p.T, p.pos0, p.rope0, false, false, p.off);
    CK(cudaStreamSynchronize(st));
    for (size_t j = 0; j < parts.size(); j++) {
        const int i = act[j];
        const Part &p = parts[j];
        Seq &s = *p.s;
        const Sampling &sp = sps[i];
        int *o = out + (size_t)i * max_out;
        const int a = verify_drafts(s.seen, &tv[p.off * TOPK], &ti[p.off * TOPK], &ts[p.off * NSTAT], &drafts[p.off], k[i], sp, s.rng,
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
            gdn_scan(sa, 48, st);
            gdn_conv_commit(s.conv, b.v_raw + (size_t)p.off * QKV, a + 1, 8 * QKV, 48, st);
        }
        CK(cudaMemcpyAsync(s.h_last, b.hfin + (size_t)(p.off + a) * D, D * 4, cudaMemcpyDeviceToDevice, st));
        s.n = q[i] + a + 1;
        s.pos += a + 1;
        s.pending = next;
    }
}

int Engine::generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v, int *rows_i) {
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    int nout = 0;
    if (s.pending < 0) {
        if (s.top_v.empty()) FAIL("generate: nothing to continue from");
        s.pending = sample(s.seen, s.top_v.data(), s.top_i.data(), top_stats(s), sp, s.rng);
        if (rows_v) keep_row(s.top_v.data(), s.top_i.data(), rows_v, rows_i, nout);
        out[nout++] = s.pending;
        if (sp.lookup > 0) s.look.push(s.pending);
        if (is_eos(s.pending) || s.pending == sp.halt || nout == max_out) return nout;
    }
    const int room = std::min(7, max_out - nout - 1);
    int copied[8] = {s.pending};
    const int nl = sp.lookup > 0 ? s.look.find(std::min(sp.lookup, room), VOCAB, copied + 1) : 0;
    const int k = nl > 0 ? nl : std::max(0, std::min(sp.n_draft, room));
    const int q = s.n;
    ensure_pages(s, q + k + 1, q);

    CK(cudaMemcpyAsync(b.draft, copied, (nl + 1) * 4, cudaMemcpyHostToDevice, st));
    for (int i = 0; i < k && nl == 0; i++) {
        mtp_forward(s, 1, q + i, i == 0 ? s.h_last : b.mtp_x, b.draft + i, s.pos + i);
        lm_head(b.hq, 1, DRAFT_VOCAB);
        argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1, st);
    }
    return verify(s, sp, k, false, out, nout, rows_v, rows_i);
}

int Engine::verify(Seq &s, const Sampling &sp, int k, bool masked, int *out, int nout, float *rows_v, int *rows_i) {
    const int q = s.n;
    const int T = k + 1;
    CK(cudaMemcpyAsync(b.tokens, b.draft, T * 4, cudaMemcpyDeviceToDevice, st));
    forward(s, T, k > 0, s.pos);
    lm_head(b.hq, T, VOCAB);
    if (masked) mask_logits(b.logits, d_mask, VOCAB, T, st);
    topk(b.logits, T, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
    logit_stats(b.logits, T, VOCAB, VOCAB, b.tops, st);
    std::vector<float> tv(T * TOPK), ts(T * NSTAT);
    std::vector<int> ti(T * TOPK), drafts(T);
    CK(cudaMemcpyAsync(tv.data(), b.topv, T * TOPK * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(ti.data(), b.topi, T * TOPK * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(ts.data(), b.tops, T * NSTAT * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(drafts.data(), b.draft, T * 4, cudaMemcpyDeviceToHost, st));
    mtp_catch_up(s, T, q, s.pos);
    CK(cudaStreamSynchronize(st));

    const int a = verify_drafts(s.seen, tv.data(), ti.data(), ts.data(), drafts.data(), k, sp, s.rng,
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
        gdn_scan(sa, 48, st);
        gdn_conv_commit(s.conv, b.v_raw, a + 1, 8 * QKV, 48, st);
    }
    CK(cudaMemcpyAsync(s.h_last, b.hfin + (size_t)a * D, D * 4, cudaMemcpyDeviceToDevice, st));
    s.n = q + a + 1;
    s.pos += a + 1;
    s.pending = next;
    return nout;
}

uint32_t *Engine::mask_rows() {
    if (!d_mask) CK(cudaMalloc(&d_mask, (size_t)8 * (VOCAB / 32) * 4));
    return d_mask;
}

int Engine::draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out) {
    Seq &s = seqs.at(id);
    if (s.pending < 0) FAIL("draft: no pending token");
    const int room = std::min(7, max_k);
    if (room < 1) return 0;
    int copied[8] = {s.pending};
    const int nf = std::min(nfixed, room);
    std::copy(fixed, fixed + nf, copied + 1);
    const int nl = nf == 0 && sp.lookup > 0 ? s.look.find(std::min(sp.lookup, room), VOCAB, copied + 1) : 0;
    const int k = nl > 0 ? nl : std::min(room, nf + std::max(0, sp.n_draft));
    if (k > nf + nl) {
        const int q = s.n;
        ensure_pages(s, q + k + 1, q);
        CK(cudaMemcpyAsync(b.draft, copied, (nf + 1) * 4, cudaMemcpyHostToDevice, st));
        for (int i = 0; i < k; i++) {
            mtp_forward(s, 1, q + i, i == 0 ? s.h_last : b.mtp_x, b.draft + i, s.pos + i);
            if (i < nf) continue;
            lm_head(b.hq, 1, DRAFT_VOCAB);
            argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1, st);
        }
        CK(cudaMemcpyAsync(copied + 1 + nf, b.draft + 1 + nf, (k - nf) * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
    std::copy(copied + 1, copied + 1 + k, out);
    return k;
}

int Engine::generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out) {
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    if (s.pending < 0) FAIL("generate_masked: no pending token");
    k = std::max(0, std::min({k, 7, max_out - 1}));
    ensure_pages(s, s.n + k + 1, s.n);
    int toks[8] = {s.pending};
    std::copy(drafts, drafts + k, toks + 1);
    CK(cudaMemcpyAsync(b.draft, toks, (k + 1) * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(mask_rows(), masks, (size_t)(k + 1) * (VOCAB / 32) * 4, cudaMemcpyHostToDevice, st));
    return verify(s, sp, k, true, out, 0, nullptr, nullptr);
}
