#include "model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <unistd.h>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static WArgs wargs(const QTensor &t) { return {t.q.g, t.qh.g, t.sc.g, t.d.g}; }
static int cdiv(long a, long b) { return (int)((a + b - 1) / b); }

static int gemv_rt(const QTensor &W) {
    if (W.type != IQ4_XS) return 2;
    return W.cols > 8192 ? 4 : W.rows > 16384 ? 1 : 8;
}

static bool gemv1k(Mtl &m, const QTensor &W, const GemvA &a, int mode) {
    if ((W.type != IQ4_XS && W.type != Q6_K) || W.cols % 512) return false;
    const bool deep = W.type == IQ4_XS && W.cols > 8192;
    const int p = deep ? 1 : 4, ks = deep ? 8 : 2;
    if (W.rows % (4 * p)) return false;
    char name[48];
    snprintf(name, sizeof name, "gemv1k_%d_%d_%d_%d", W.type, p, ks, mode);
    m.run(m.pipeline(name), a, W.rows / (4 * p), 1, 1, 32 * ks * p);
    return true;
}

void Engine::gemv(const QTensor &W, DP<float> x, DP<half_t> x16, int ldx, int T, DP<float> y, int ldy, int mode, DP<float> nw) {
    if (W.type != Q8_0 && W.type != Q6_K && W.type != IQ4_XS) FAIL("gemv: type %d", W.type);
    if (nw && T != 1) FAIL("gemv: a fused norm with %d tokens", T);
    char name[48];
    GemvA a{wargs(W), {}, x.g, y.g, W.rows, W.cols, ldx, ldy, T, mode == GEMV_ADD, x16.g, 0, nw.g};
    if (T == 1 && gemv1k(m, W, a, mode)) {
    } else if (T == 1) {
        const int rt = gemv_rt(W);
        snprintf(name, sizeof name, "gemv1_%d_%d_%d", W.type, rt, mode);
        m.run(m.pipeline(name), a, W.rows / (8 * rt), 1, 1, 128);
    } else {
        snprintf(name, sizeof name, "gemv_mma_%d_4_0_1", W.type);
        m.run(m.pipeline(name), a, W.rows / 32, 1, 1, 128);
    }
}

void Engine::gemv_glu(const QTensor &G, const QTensor &U, DP<float> x, DP<half_t> x16, int T, DP<float> nw) {
    if (G.type != U.type) FAIL("gemv_glu: gate and up of different types");
    char name[48];
    GemvA a{wargs(G), wargs(U), x.g, b.proj.g, G.rows, G.cols, G.cols, FF, T, 0, x16.g, b.ffn_g.g, nw.g};
    if (T == 1 && gemv1k(m, G, a, 2)) {
    } else if (T == 1) {
        const int rt = gemv_rt(G);
        snprintf(name, sizeof name, "gemv1_%d_%d_2", G.type, rt);
        m.run(m.pipeline(name), a, G.rows / (8 * rt), 1, 1, 128);
    } else {
        snprintf(name, sizeof name, "gemv_mma_%d_4_1_1", G.type);
        m.run(m.pipeline(name), a, G.rows / 32, 1, 1, 128);
    }
}

void Engine::linear(const QTensor &W, int T, DP<half_t> hin, int ldin, DP<float> xin, int ldx, DP<uint8_t> out, int ldo,
                    bool add, bool out_f16) {
    if (T <= 8) {
        if (out_f16) FAIL("linear: fp16 output from a GEMV");
        gemv(W, xin, hin, ldx, T, out.as<float>(), ldo, add ? GEMV_ADD : GEMV_STORE);
        return;
    }
    if (T <= 48) {
        for (int t0 = 0; t0 < T; t0 += 16) {
            const int n = std::min(16, T - t0);
            char name[48];
            snprintf(name, sizeof name, "gemv_mma_%d_4_0_%d", W.type, n <= 8 ? 1 : 2);
            const size_t ob = (size_t)t0 * ldo * (out_f16 ? 2 : 4);
            GemvA g{wargs(W), {}, 0, (out + ob).g, W.rows, W.cols, ldin, ldo, n, add, (hin + (size_t)t0 * ldin).g,
                    out_f16 ? (out + ob).g : 0, 0};
            m.run(m.pipeline(name), g, W.rows / 32, 1, 1, 128);
            m.par();
        }
        m.barrier();
        return;
    }
    if (T >= mps_min) {
        const int c[1] = {W.type};
        DeqA d{wargs(W), b.deq.g, W.rows, W.cols};
        m.run(m.pipeline("dequant", c, 1), d, cdiv((long)W.rows * W.cols / 32, 256), 1, 1, 256);
        m.gemm(hin.g, ldin, b.deq.g, out.g, ldo, !out_f16, T, W.rows, W.cols, add ? 1.0f : 0.0f);
        return;
    }
    const int c[1] = {W.type};
    GemmA g{wargs(W), hin.g, out.g, T, W.rows, W.cols, ldin, ldo, add, out_f16};
    m.run(m.pipeline("gemm_q", c, 1), g, W.rows / 64, cdiv(T, 64), 1, 128);
}

void Engine::rmsnorm(DP<float> x, DP<float> w, int T, int n, int ldx, int ldo, DP<float> of, DP<half_t> oh) {
    NormA a{x.g, w.g, of.g, oh.g, n, ldx, ldo};
    m.run(m.pipeline("rmsnorm"), a, T, 1, 1, 256);
}

void Engine::norm1(DP<float> x, DP<float> w, DP<float> out) {
    NormA a{x.g, w.g, out.g, 0, D, D, D};
    m.run(m.pipeline("rmsnorm1"), a, 1, 1, 1, 1024);
}

void Engine::copy(DP<uint8_t> dst, DP<uint8_t> src, size_t bytes) {
    if (bytes % 16) FAIL("copy: %zu bytes", bytes);
    CopyA a{src.g, dst.g, bytes};
    m.run(m.pipeline("copy16"), a, std::min(cdiv(bytes / 16, 256), 4096), 1, 1, 256);
}

void Engine::fill0(DP<uint8_t> dst, size_t bytes) {
    FillA a{dst.g, bytes};
    m.run(m.pipeline("fill16"), a, std::min(cdiv(bytes / 16, 256), 4096), 1, 1, 256);
}

void Engine::topk(DP<float> logits, int T, int n, int ld) {
    const int nb = cdiv(n, 4096);
    TopkA a{logits.g, b.tsv.g, b.tsi.g, b.topv.g, b.topi.g, n, ld, nb * TOPK};
    m.run(m.pipeline("topk1"), a, nb, T, 1, 512);
    m.run(m.pipeline("topk2"), a, T, 1, 1, 512);
}

void Engine::logit_stats(DP<float> logits, int T, int n, int ld) {
    StatsA a{logits.g, b.tops.g, n, ld};
    m.run(m.pipeline("logit_stats"), a, T, 1, 1, 1024);
}

void Engine::argmax(DP<float> logits, int T, int n, int ld, DP<int> out) {
    ArgmaxA a{logits.g, out.g, n, ld};
    m.run(m.pipeline("argmax"), a, T, 1, 1, 1024);
}

void Engine::embed(DP<int> tokens, int T, DP<float> out) {
    EmbedA a{wargs(embd), tokens.g, out.g, T};
    m.run(m.pipeline("embed"), a, cdiv(D / 32, 32), T, 1, 32);
}

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
    flush();
    for (int p : s.pages) page_release(p);
    s.n = 0; s.pos = 0; s.pending = -1; s.pages.clear(); s.seen.clear(); s.top_v.clear(); s.top_i.clear();
    s.look.clear();
    s.rng.seed(std::random_device{}());
    fill0(s.ssm.as<uint8_t>(), SSM_BYTES);
    fill0(s.conv.as<uint8_t>(), CONV_BYTES);
    fill0(s.h_last.as<uint8_t>(), D * 4);
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

void Engine::init_kv(int max_pages) {
    n_pages = max_pages;
    page_mem.clear(); free_pages.clear(); page_ref.clear();
    if (!grow_kv()) FAIL("no room for a KV cache");
}

bool Engine::grow_kv() {
    const int n = std::min(SEG_PAGES, n_pages - (int)page_mem.size());
    if (n <= 0) return false;
    const DP<uint8_t> seg = m.try_alloc((size_t)n * KV_PAGE);
    if (!seg) return false;
    if (!kv_pool) kv_pool = seg;
    for (int i = n; i-- > 0;) free_pages.push_back((int)page_mem.size() + i);
    for (int i = 0; i < n; i++) page_mem.push_back(seg + (size_t)i * KV_PAGE);
    page_ref.resize(page_mem.size(), 0);
    return true;
}

int Engine::page_alloc() {
    if (free_pages.empty() && !grow_kv()) throw std::runtime_error("out of KV pages");
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
    int shared = 0;
    for (int i = from / PAGE; i < (int)s.pages.size() && i < need; i++) shared += page_ref[s.pages[i]] > 1;
    if (free_page_count() < need - (int)s.pages.size() + shared) throw std::runtime_error("out of KV pages");
    while ((int)s.pages.size() < need) s.pages.push_back(page_alloc());
    for (int i = from / PAGE; i < (int)s.pages.size() && i < need; i++) {
        if (page_ref[s.pages[i]] == 1) continue;
        const int np = page_alloc();
        copy(page_mem[np], page_mem[s.pages[i]], KV_PAGE);
        page_release(s.pages[i]);
        s.pages[i] = np;
    }
    for (size_t i = 0; i < s.pages.size(); i++) s.d_pages.h[i] = (int64_t)(page_mem[s.pages[i]].g - kv_pool.g);
}

int Engine::snapshot(int id) {
    flush();
    Seq &s = seqs.at(id);
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    if (!sn.host) sn.host = m.alloc(state_bytes(), "a snapshot");
    sn.used = true;
    sn.n = s.n;
    sn.pos = s.pos;
    if (s.pending < 0) { sn.top_v = s.top_v; sn.top_i = s.top_i; } else { sn.top_v.clear(); sn.top_i.clear(); }
    sn.pages.assign(s.pages.begin(), s.pages.begin() + (s.n + PAGE - 1) / PAGE);
    for (int p : sn.pages) page_ref[p]++;
    copy(sn.host, s.ssm.as<uint8_t>(), state_bytes());
    m.sync();
    return (int)si;
}

void Engine::restore(int id, int snap) {
    flush();
    Seq &s = seqs.at(id);
    Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("restore of a freed snapshot");
    for (int p : s.pages) page_release(p);
    s.pages = sn.pages;
    for (int p : s.pages) page_ref[p]++;
    for (size_t i = 0; i < s.pages.size(); i++) s.d_pages.h[i] = (int64_t)(page_mem[s.pages[i]].g - kv_pool.g);
    s.n = sn.n;
    s.pos = sn.pos;
    s.pending = -1;
    s.top_v = sn.top_v; s.top_i = sn.top_i; s.seen.clear();
    copy(s.ssm.as<uint8_t>(), sn.host, state_bytes());
    m.sync();
}

struct SnapHeader { char magic[8]; uint64_t kv_page, state; int32_t n, pos, n_top, n_pages; };
static const char SNAP_MAGIC[8] = {'K', 'I', 'L', 'N', 'M', 'T', 'L', '1'};

size_t Engine::snap_bytes(int snap) const {
    const Snap &sn = snaps.at(snap);
    return sizeof(SnapHeader) + sn.top_v.size() * 8 + state_bytes() + sn.pages.size() * KV_PAGE;
}

void Engine::snap_export(int snap, uint8_t *dst) {
    const Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("export of a freed snapshot");
    m.sync();
    SnapHeader h;
    memcpy(h.magic, SNAP_MAGIC, 8);
    h.kv_page = KV_PAGE; h.state = state_bytes(); h.n = sn.n; h.pos = sn.pos;
    h.n_top = (int32_t)sn.top_v.size(); h.n_pages = (int32_t)sn.pages.size();
    uint8_t *p = dst;
    memcpy(p, &h, sizeof h); p += sizeof h;
    memcpy(p, sn.top_v.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
    memcpy(p, sn.top_i.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
    memcpy(p, sn.host.h, h.state); p += h.state;
    for (int pg : sn.pages) { memcpy(p, page_mem[pg].h, KV_PAGE); p += KV_PAGE; }
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
    if (free_page_count() < h.n_pages) return -2;
    m.sync();
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    if (!sn.host) sn.host = m.alloc(state_bytes(), "a snapshot");
    sn.top_v.resize(h.n_top);
    sn.top_i.resize(h.n_top);
    bool ok = get(sn.top_v.data(), (size_t)h.n_top * 4) && get(sn.top_i.data(), (size_t)h.n_top * 4) && get(sn.host.h, h.state);
    sn.pages.clear();
    for (int i = 0; ok && i < h.n_pages; i++) {
        sn.pages.push_back(page_alloc());
        ok = get(page_mem[sn.pages.back()].h, KV_PAGE);
    }
    if (!ok) {
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
    for (int p : sn.pages) page_release(p);
    sn.pages.clear();
    sn.used = false;
}

void Engine::attend(const Layer &w, int kvl, const Part &q, bool small) {
    const DP<float> qg = b.proj + (size_t)q.off * 2 * NH * HD;
    const int nrows = q.T * 6;
    const bool dec = q.T <= 8;
    const int nsg = dec ? cdiv(nrows, 8) : 8;
    const int qrows = dec ? cdiv(nrows, nsg * 8) * nsg * 8 : cdiv(nrows, 64) * 64;
    PrepA pa{qg.g, (b.k + (size_t)q.off * NKV * HD).g, (b.v + (size_t)q.off * NKV * HD).g, w.q_norm.g, w.k_norm.g,
             q.rpos ? (b.rpos + (size_t)q.off * 3).g : 0, kv_pool.g, q.s->d_pages.g, b.Qg.g, q.T, q.pos0, q.rope0, kvl, qrows, 0};
    m.run(m.pipeline("attn_prep"), pa, NH + 2 * NKV, q.T, 1, 256);
    const int kend = q.pos0 + q.T;
    const int max_splits = q.T <= 2 ? 4 * MAX_SPLITS : MAX_SPLITS;
    const int min_len = q.T <= 2 ? std::min(64, std::max(16, cdiv(kend, 40 * 16) * 16)) : 256;
    const int split_len = dec ? std::max(min_len, cdiv(cdiv(kend, max_splits), 16) * 16) : kend;
    const int nsplit = cdiv(kend, split_len);
    if (!dec && kend >= attn_gemm_min) {
        const int M = q.T * 6, R = std::max(8, std::min(M, (int)(ATT_S / kend) / 8 * 8));
        for (int g = 0; g < NKV; g++) {
            UnpackA u{kv_pool.g, q.s->d_pages.g, b.kc.g, b.vt.g, kvl, g, kend};
            m.run(m.pipeline("kv_unpack"), u, cdiv(kend, 8), 1, 1, 256);
            for (int r0 = 0; r0 < M; r0 += R) {
                const int rows = std::min(R, M - r0);
                const DP<float> O = b.part_o + ((size_t)g * qrows + r0) * HD, ml = b.part_ml + ((size_t)g * qrows + r0) * 2;
                m.gemm((b.Qg + ((size_t)g * qrows + r0) * HD).g, HD, b.kc.g, b.st.g, kend, true, rows, kend, HD, 0.0f);
                SmaxA sm{b.st.g, b.pt.g, O.g, ml.g, rows, kend, 0, q.pos0, r0};
                m.run(m.pipeline("attn_smax"), sm, rows, 1, 1, 256);
                m.gemm(b.pt.g, kend, b.vt.g, O.g, HD, true, rows, HD, kend, 1.0f, kend);
            }
        }
        CombineA ca{b.part_o.g, b.part_ml.g, qg.g, 0, (b.mix + (size_t)q.off * NH * HD).g, q.T, qrows, 1};
        m.run(m.pipeline("attn_combine"), ca, q.T * NH, 1, 1, 256);
        return;
    }
    AttnA aa{b.Qg.g, kv_pool.g, q.s->d_pages.g, b.part_o.g, b.part_ml.g, q.T, q.pos0, kvl, qrows, split_len, nsplit};
    if (dec) m.run(m.pipeline("attn_vec"), &aa, sizeof aa, nsplit, NKV, 1, q.T * 2 * 32);
    else m.run(m.pipeline("attn_pf"), &aa, sizeof aa, nsplit, NKV, cdiv(nrows, 64), 512);
    CombineA ca{b.part_o.g, b.part_ml.g, qg.g, small ? (b.o + (size_t)q.off * NH * HD).g : 0, (b.mix + (size_t)q.off * NH * HD).g, q.T,
                qrows, nsplit};
    m.run(m.pipeline("attn_combine"), ca, q.T * NH, 1, 1, 256);
}

void Engine::attn_layer(const Layer &w, int kvl, const Part *p, int np, int T, DP<float> x, DP<float> xn, DP<half_t> hh) {
    const bool small = T <= 8;
    if (T == 1) {
        norm1(x, w.attn_norm, xn);
        gemv(w.q, xn, {}, D, 1, b.proj, 2 * NH * HD, GEMV_STORE);
        m.par();
        gemv(w.k, xn, {}, D, 1, b.k, NKV * HD, GEMV_STORE);
        m.par();
        gemv(w.v, xn, {}, D, 1, b.v, NKV * HD, GEMV_STORE);
    } else {
        linear(w.q, T, hh, D, xn, D, b.proj.as<uint8_t>(), 2 * NH * HD, false, false);
        m.par();
        linear(w.k, T, hh, D, xn, D, b.k.as<uint8_t>(), NKV * HD, false, false);
        m.par();
        linear(w.v, T, hh, D, xn, D, b.v.as<uint8_t>(), NKV * HD, false, false);
    }
    for (int i = 0; i < np; i++) attend(w, kvl, p[i], small);
    linear(w.o, T, b.mix, NH * HD, b.o, NH * HD, x.as<uint8_t>(), D, true, false);
    if (T > 1) rmsnorm(x, w.post_norm, T, D, D, D, small ? xn : DP<float>{}, hh);
}

void Engine::gdn_layer(const Layer &w, int gl, const Part *p, int np, int T, DP<float> x, DP<float> xn, DP<half_t> hh) {
    if (T <= 8) {
        const DP<float> raw = b.v_raw + (size_t)gl * 8 * QKV;
        const bool f = T == 1;
        if (f) norm1(x, w.attn_norm, xn);
        gemv(w.qkv, xn, hh, D, T, raw, QKV, GEMV_STORE);
        m.par();
        gemv(w.z, xn, hh, D, T, b.proj, GV * GD, GEMV_STORE);
        if (f && !p[0].verify) {
            StepA s{};
            s.Wab = w.ab.g; s.x = xn.g; s.ab_out = b.ab.g;
            m.par();
            m.run(m.pipeline("gdn_ab1"), s, 2 * GV, 1, 1, 128);
        }
        for (int i = 0; i < np; i++) {
            const Part &q = p[i];
            StepA s{(raw + (size_t)q.off * QKV).g, (b.proj + (size_t)q.off * GV * GD).g, ((f ? x : xn) + (size_t)q.off * D).g, w.ab.g,
                    w.conv.g, (gdn_A + gl * GV).g, (gdn_dt + gl * GV).g, w.ssm_norm.g,
                    (q.s->conv + (size_t)gl * QKV * 3).g, (q.s->ssm + (size_t)gl * GV * GD * GD).g,
                    (b.v_kn + ((size_t)gl * 8 + q.off) * GK * GD).g, (b.v_vc + ((size_t)gl * 8 + q.off) * GV * GD).g,
                    (b.v_ab + ((size_t)gl * 8 + q.off) * 2 * GV).g, (b.o + (size_t)q.off * GV * GD).g,
                    (b.mix + (size_t)q.off * GV * GD).g, f ? w.attn_norm.g : 0, 0, q.T, q.verify};
            if (i) m.par();
            if (f && !q.verify) {
                s.ab_in = b.ab.g;
                m.run(m.pipeline("gdn_step1s"), s, GV, 4, 1, 128);
                GdnOutA go{b.o.g, b.proj.g, w.ssm_norm.g, b.o.g, 0};
                m.run(m.pipeline("gdn_out"), go, GV, 1, 1, 128);
            } else {
                m.run(m.pipeline(q.T == 1 ? "gdn_step1" : "gdn_step"), s, GV, 1, 1, 512);
            }
        }
        gemv(w.ssm_out, b.o, b.mix, GV * GD, T, x, D, GEMV_ADD);
        if (!f) rmsnorm(x, w.post_norm, T, D, D, D, xn, hh);
        return;
    }
    Seq &s = *p[0].s;
    const DP<float> raw = b.proj, z = b.proj + (size_t)T * QKV;
    linear(w.qkv, T, hh, D, {}, 0, raw.as<uint8_t>(), QKV, false, false);
    linear(w.z, T, hh, D, {}, 0, z.as<uint8_t>(), GV * GD, false, false);
    AbA ab{w.ab.g, hh.g, b.ab.g, T};
    m.run(m.pipeline("gdn_ab"), ab, 2 * GV, T, 1, 256);
    const DP<float> cst = s.conv + (size_t)gl * QKV * 3;
    ConvA ca{raw.g, w.conv.g, cst.g, b.qn.g, b.kn.g, b.vc.g, T, 1};
    m.run(m.pipeline("gdn_conv"), ca, 2 * GK + GV, cdiv(T, 16), 1, 128);
    ConvStateA cs{raw.g, cst.g, T};
    m.run(m.pipeline("gdn_conv_state"), cs, cdiv(QKV, 256), 1, 1, 256);
    ScanA sa{(s.ssm + (size_t)gl * GV * GD * GD).g, 0, b.qn.g, b.kn.g, 0, b.vc.g, 0, b.ab.g, 0, (gdn_A + gl * GV).g,
             (gdn_dt + gl * GV).g, b.o.g, T, 1};
    m.run(m.pipeline("gdn_scan"), sa, GV, GD / 32, 1, 128);
    GdnOutA go{b.o.g, z.g, w.ssm_norm.g, 0, b.mix.g};
    m.run(m.pipeline("gdn_out"), go, GV, T, 1, 128);
    linear(w.ssm_out, T, b.mix, GV * GD, {}, 0, x.as<uint8_t>(), D, true, false);
    rmsnorm(x, w.post_norm, T, D, D, D, {}, hh);
}

void Engine::ffn_layer(const Layer &w, int T, DP<float> x, DP<float> xn, DP<half_t> hh, DP<float> next, DP<float> nf) {
    const bool small = T <= 8;
    if (small) {
        gemv_glu(w.ffn_gate, w.ffn_up, T == 1 ? x : xn, hh, T, T == 1 ? w.post_norm : DP<float>{});
        gemv(w.ffn_down, b.proj, b.ffn_g, FF, T, x, D, GEMV_ADD);
    } else {
        linear(w.ffn_gate, T, hh, D, {}, 0, b.ffn_g.as<uint8_t>(), FF, false, true);
        linear(w.ffn_up, T, hh, D, {}, 0, b.ffn_u.as<uint8_t>(), FF, false, true);
        SwigluA sw{b.ffn_g.g, b.ffn_u.g, T * FF};
        m.run(m.pipeline("swiglu_h"), sw, cdiv((long)T * FF, 256), 1, 1, 256);
        linear(w.ffn_down, T, b.ffn_g, FF, {}, 0, x.as<uint8_t>(), D, true, false);
    }
    if (next && (T > 1 || nf)) rmsnorm(x, next, T, D, D, D, nf ? nf : small ? xn : DP<float>{}, hh);
}

void Engine::forward_parts(const Part *p, int np, int T) {
    for (int i = 0; i < np; i++) ensure_pages(*p[i].s, p[i].s->n + p[i].T, p[i].s->n);
    const bool small = T <= 8;
    embed(b.tokens, T, b.x);
    if (T > 1) rmsnorm(b.x, L[0].attn_norm, T, D, D, D, small ? b.xn : DP<float>{}, b.hh);
    for (int l = 0, gl = 0; l < NL; l++) {
        if (is_attn_layer(l)) attn_layer(L[l], l / 4, p, np, T, b.x, b.xn, b.hh);
        else gdn_layer(L[l], gl++, p, np, T, b.x, b.xn, b.hh);
        const bool last = l == NL - 1;
        ffn_layer(L[l], T, b.x, b.xn, b.hh, last ? out_norm : L[l + 1].attn_norm, last ? b.hfin : DP<float>{});
    }
    if (!small) return;
    for (int i = 0; i < np; i++)
        if (!p[i].verify) {
            ConvCommitA cc{p[i].s->conv.g, (b.v_raw + (size_t)p[i].off * QKV).g, p[i].T, 8 * QKV};
            m.run(m.pipeline("gdn_conv_commit"), cc, cdiv(QKV, 256), 48, 1, 256);
        }
}

void Engine::forward(Seq &s, int T, bool verify, int rope0, bool rpos) {
    const Part p{&s, T, 0, s.n, rope0, rpos, verify};
    forward_parts(&p, 1, T);
}

void Engine::mtp_forward(Seq &s, int T, int pos0, DP<float> h_prev, DP<int> tokens, int rope0, bool rpos) {
    embed(tokens, T, b.x);
    rmsnorm(b.x, enorm, T, D, D, 2 * D, b.mtp_xn, b.hh);
    m.par();
    rmsnorm(h_prev, hnorm, T, D, D, 2 * D, b.mtp_xn + D, b.hh + D);
    gemv(eh_proj, b.mtp_xn, b.hh, 2 * D, T, b.mtp_x, D, GEMV_STORE);
    if (T > 1) rmsnorm(b.mtp_x, M.attn_norm, T, D, D, D, b.xn, b.hh);
    const Part p{&s, T, 0, pos0, rope0, rpos, false};
    attn_layer(M, NATT, &p, 1, T, b.mtp_x, b.xn, b.hh);
    ffn_layer(M, T, b.mtp_x, b.xn, b.hh, head_norm, b.mtp_x);
}

void Engine::mtp_catch_up(Seq &s, int T, int pos0, int rope0, bool rpos, int off) {
    copy(b.mtp_h.as<uint8_t>(), s.h_last.as<uint8_t>(), D * 4);
    m.par();
    if (T > 1) copy((b.mtp_h + D).as<uint8_t>(), (b.hfin + (size_t)off * D).as<uint8_t>(), (size_t)(T - 1) * D * 4);
    m.par();
    embed(b.tokens + off, T, b.x);
    const bool small = T <= 8;
    if (small) {
        rmsnorm(b.x, enorm, T, D, D, 2 * D, b.mtp_xn, b.hh);
        m.par();
        rmsnorm(b.mtp_h, hnorm, T, D, D, 2 * D, b.mtp_xn + D, b.hh + D);
        gemv(eh_proj, b.mtp_xn, b.hh, 2 * D, T, b.mtp_x, D, GEMV_STORE);
        rmsnorm(b.mtp_x, M.attn_norm, T, D, D, D, b.xn, b.hh);
        gemv(M.k, b.xn, b.hh, D, T, b.k, NKV * HD, GEMV_STORE);
        m.par();
        gemv(M.v, b.xn, b.hh, D, T, b.v, NKV * HD, GEMV_STORE);
    } else {
        rmsnorm(b.x, enorm, T, D, D, 2 * D, {}, b.hh);
        m.par();
        rmsnorm(b.mtp_h, hnorm, T, D, D, 2 * D, {}, b.hh + D);
        linear(eh_proj, T, b.hh, 2 * D, {}, 0, b.x.as<uint8_t>(), D, false, false);
        rmsnorm(b.x, M.attn_norm, T, D, D, D, {}, b.hh);
        linear(M.k, T, b.hh, D, {}, 0, b.k.as<uint8_t>(), NKV * HD, false, false);
        linear(M.v, T, b.hh, D, {}, 0, b.v.as<uint8_t>(), NKV * HD, false, false);
    }
    PrepA pa{0, b.k.g, b.v.g, M.q_norm.g, M.k_norm.g, rpos ? (b.rpos + (size_t)off * 3).g : 0, kv_pool.g, s.d_pages.g, 0, T, pos0,
             rope0, NATT, 0, NH};
    m.run(m.pipeline("attn_prep"), pa, 2 * NKV, T, 1, 256);
}

void Engine::lm_head(DP<float> in, int T, int rows) {
    QTensor w = out;
    w.rows = rows;
    gemv(w, in, b.hh, D, T, b.logits, VOCAB, GEMV_STORE);
}

void Engine::prefill(int id, const int *tokens, const int *pos3, int n) {
    flush();
    Seq &s = seqs.at(id);
    for (int i = 0; i < n; i++)
        if (tokens[i] >= IMG_BASE) throw std::invalid_argument("an image token: pictures are not supported by this engine");
        else if (tokens[i] < 0 || tokens[i] >= VOCAB) throw std::invalid_argument("token id out of range");
    if (free_page_count() * PAGE < (s.n + n + PAGE - 1) / PAGE * PAGE - (int)s.pages.size() * PAGE)
        throw std::runtime_error("out of KV pages");
    s.pending = -1;
    for (int done = 0; done < n;) {
        const int T = std::min(umax, n - done);
        std::copy(tokens + done, tokens + done + T, b.tokens.h);
        if (pos3) std::copy(pos3 + (size_t)done * 3, pos3 + (size_t)(done + T) * 3, b.rpos.h);
        const int rope0 = s.pos;
        forward(s, T, false, rope0, pos3);
        mtp_catch_up(s, T, s.n, rope0, pos3);
        if (done + T == n) {
            lm_head(b.hfin + (size_t)(T - 1) * D, 1, VOCAB);
            topk(b.logits, 1, VOCAB, VOCAB);
            logit_stats(b.logits, 1, VOCAB, VOCAB);
        }
        m.par();
        copy(s.h_last.as<uint8_t>(), (b.hfin + (size_t)(T - 1) * D).as<uint8_t>(), D * 4);
        const double t0 = now_ms();
        m.sync();
        static const bool trace = getenv("CHRYSALIS_TRACE");
        if (trace) fprintf(stderr, "chrysalis: prefill chunk of %d at %d: GPU %.1f ms, wait %.1f ms\n", T, s.n, m.last_gpu_ms, now_ms() - t0);
        if (done + T == n) {
            s.top_v.assign(b.topv.h, b.topv.h + TOPK);
            s.top_v.insert(s.top_v.end(), b.tops.h, b.tops.h + NSTAT);
            s.top_i.assign(b.topi.h, b.topi.h + TOPK);
            s.top_i.resize(TOPK + NSTAT, 0);
        }
        s.n += T;
        if (pos3)
            for (int i = done * 3; i < (done + T) * 3; i++) s.pos = std::max(s.pos, pos3[i] + 1);
        else
            s.pos += T;
        done += T;
    }
}

static void keep_row(const float *v, const int *i, float *rows_v, int *rows_i, int at) {
    std::copy(v, v + TOPK, rows_v + (size_t)at * TOPK);
    std::copy(i, i + TOPK, rows_i + (size_t)at * TOPK);
}

static const float *top_stats(const Seq &s) {
    static const float unknown[NSTAT] = {(float)VOCAB, -1, 0, 0, 0, 0};
    return s.top_v.size() >= (size_t)(TOPK + NSTAT) ? s.top_v.data() + TOPK : unknown;
}

void Engine::defer_commit(Seq &s, int off, int n, int hrow) { deferred.push_back({&s, off, n, hrow}); }

void Engine::flush() {
    for (const Deferred &d : deferred) {
        Seq &s = *d.s;
        if (d.n > 0) {
            ScanA sa{s.ssm.g, (ulong)GV * GD * GD, 0, (b.v_kn + (size_t)d.off * GK * GD).g, (ulong)8 * GK * GD,
                     (b.v_vc + (size_t)d.off * GV * GD).g, (ulong)8 * GV * GD, (b.v_ab + (size_t)d.off * 2 * GV).g, (ulong)8 * 2 * GV,
                     gdn_A.g, gdn_dt.g, 0, d.n, 1};
            m.run(m.pipeline("gdn_scan"), sa, GV, GD / 32, 48, 128);
            m.par();
            ConvCommitA cc{s.conv.g, (b.v_raw + (size_t)d.off * QKV).g, d.n, 8 * QKV};
            m.run(m.pipeline("gdn_conv_commit"), cc, cdiv(QKV, 256), 48, 1, 256);
            m.par();
        }
        copy(s.h_last.as<uint8_t>(), (b.hfin + (size_t)d.hrow * D).as<uint8_t>(), D * 4);
        m.par();
    }
    if (!deferred.empty()) m.barrier();
    deferred.clear();
}

void Engine::generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v,
                            int *rows_i) {
    flush();
    int need = 0;
    for (int i = 0; i < n; i++) {
        const Seq &s = seqs.at(ids[i]);
        const int last = (s.n + 8 + PAGE - 1) / PAGE;
        need += std::max(0, last - (int)s.pages.size());
        for (int pg = s.n / PAGE; pg < (int)s.pages.size() && pg < last; pg++) need += page_ref[s.pages[pg]] > 1;
    }
    if (need > free_page_count()) throw std::runtime_error("out of KV pages");
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
            continue;
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
        b.draft.h[T] = s.pending;
        for (int d = 0; d < k[i]; d++) {
            mtp_forward(s, 1, q[i] + d, d == 0 ? s.h_last : b.mtp_x, b.draft + T + d, s.pos + d);
            lm_head(b.mtp_x, 1, DRAFT_VOCAB);
            argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + T + d + 1);
        }
        parts.push_back({&s, k[i] + 1, T, q[i], s.pos, false, k[i] > 0});
        T += k[i] + 1;
    }
    copy(b.tokens.as<uint8_t>(), b.draft.as<uint8_t>(), 16 * 4);
    forward_parts(parts.data(), (int)parts.size(), T);
    lm_head(b.hfin, T, VOCAB);
    topk(b.logits, T, VOCAB, VOCAB);
    logit_stats(b.logits, T, VOCAB, VOCAB);
    for (const Part &p : parts) mtp_catch_up(*p.s, p.T, p.pos0, p.rope0, false, p.off);
    m.sync();
    const std::vector<float> tv(b.topv.h, b.topv.h + T * TOPK), ts(b.tops.h, b.tops.h + T * NSTAT);
    const std::vector<int> ti(b.topi.h, b.topi.h + T * TOPK), drafts(b.draft.h, b.draft.h + T);
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
        defer_commit(s, p.off, k[i] > 0 ? a + 1 : 0, p.off + a);
        s.n = q[i] + a + 1;
        s.pos += a + 1;
        s.pending = next;
    }
}

int Engine::generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v, int *rows_i) {
    flush();
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    int nout = 0;
    if (s.pending < 0) {
        if (s.top_v.empty()) FAIL("generate: nothing to continue from");
        s.pending = sample(s.seen, s.top_v.data(), s.top_i.data(), top_stats(s), sp, s.rng);
        if (rows_v) keep_row(s.top_v.data(), s.top_i.data(), rows_v, rows_i, nout);
        out[nout++] = s.pending;
        if (sp.lookup > 0) s.look.push(s.pending);
        return nout;
    }
    const int room = std::min(7, max_out - nout - 1);
    int copied[8] = {s.pending};
    const int nl = sp.lookup > 0 ? s.look.find(std::min(sp.lookup, room), VOCAB, copied + 1) : 0;
    const int k = nl > 0 ? nl : std::max(0, std::min(sp.n_draft, room));
    const int q = s.n;
    ensure_pages(s, q + k + 1, q);
    std::copy(copied, copied + nl + 1, b.draft.h);
    for (int i = 0; i < k && nl == 0; i++) {
        mtp_forward(s, 1, q + i, i == 0 ? s.h_last : b.mtp_x, b.draft + i, s.pos + i);
        lm_head(b.mtp_x, 1, DRAFT_VOCAB);
        argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1);
    }
    return verify(s, sp, k, false, out, nout, rows_v, rows_i);
}

int Engine::verify(Seq &s, const Sampling &sp, int k, bool masked, int *out, int nout, float *rows_v, int *rows_i) {
    const double t0 = now_ms();
    const int q = s.n;
    const int T = k + 1;
    copy(b.tokens.as<uint8_t>(), b.draft.as<uint8_t>(), 16 * 4);
    forward(s, T, k > 0, s.pos);
    lm_head(b.hfin, T, VOCAB);
    if (masked) {
        MaskA ma{b.logits.g, b.mask.g, VOCAB};
        m.run(m.pipeline("mask_logits"), ma, cdiv(VOCAB, 256), T, 1, 256);
    }
    topk(b.logits, T, VOCAB, VOCAB);
    logit_stats(b.logits, T, VOCAB, VOCAB);
    if (sp.n_draft > 0 || sp.lookup > 0 || masked) mtp_catch_up(s, T, q, s.pos);
    static const bool trace = getenv("CHRYSALIS_TRACE");
    const int nd = m.dispatches;
    const double t1 = now_ms();
    m.sync();
    if (trace) fprintf(stderr, "chrysalis: step T=%d: %d dispatches, GPU %.2f ms, encode %.2f ms, wait %.2f ms (start %.2f, wake %.2f)\n", T, nd,
                       m.last_gpu_ms, t1 - t0, now_ms() - t1, m.last_start_ms, m.last_wake_ms);
    m.dispatches = 0;
    const float *tv = b.topv.h, *ts = b.tops.h;
    const int *ti = b.topi.h;
    int drafts[8];
    std::copy(b.draft.h, b.draft.h + T, drafts);

    const int a = verify_drafts(s.seen, tv, ti, ts, drafts, k, sp, s.rng, [&](int t) { return is_eos(t) || t == sp.halt; },
                                [&](int r, int t) {
                                    if (rows_v) keep_row(&tv[r * TOPK], &ti[r * TOPK], rows_v, rows_i, nout);
                                    out[nout++] = t;
                                    if (sp.lookup > 0) s.look.push(t);
                                });
    const int next = out[nout - 1];
    defer_commit(s, 0, k > 0 ? a + 1 : 0, a);
    s.n = q + a + 1;
    s.pos += a + 1;
    s.pending = next;
    return nout;
}

int Engine::draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out) {
    flush();
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
        std::copy(copied, copied + nf + 1, b.draft.h);
        for (int i = 0; i < k; i++) {
            mtp_forward(s, 1, q + i, i == 0 ? s.h_last : b.mtp_x, b.draft + i, s.pos + i);
            if (i < nf) continue;
            lm_head(b.mtp_x, 1, DRAFT_VOCAB);
            argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1);
        }
        m.sync();
        std::copy(b.draft.h + 1 + nf, b.draft.h + 1 + k, copied + 1 + nf);
    }
    std::copy(copied + 1, copied + 1 + k, out);
    return k;
}

int Engine::generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out) {
    flush();
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    if (s.pending < 0) FAIL("generate_masked: no pending token");
    k = std::max(0, std::min({k, 7, max_out - 1}));
    ensure_pages(s, s.n + k + 1, s.n);
    b.draft.h[0] = s.pending;
    std::copy(drafts, drafts + k, b.draft.h + 1);
    memcpy(b.mask.h, masks, (size_t)(k + 1) * (VOCAB / 32) * 4);
    return verify(s, sp, k, true, out, 0, nullptr, nullptr);
}
