// Engine implementation: loading, sequences, forward passes, MTP drafting, speculative steps.
#include "model.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

static const QAct NO_Q{nullptr, nullptr};
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
    s.rng.seed(std::random_device{}());
    CK(hipMemsetAsync(s.ssm, 0, (size_t)48 * GV * GD * GD * 4, st));
    CK(hipMemsetAsync(s.conv, 0, (size_t)48 * QKV * 3 * 4, st));
    CK(hipMemsetAsync(s.h_last, 0, D * 4, st));
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
    int shared = 0;                         // check capacity first so a failure leaves s untouched
    for (int i = from / PAGE; i < dirty && i < need; i++) shared += page_ref[s.pages[i]] > 1;
    if ((int)free_pages.size() < need - dirty + shared) throw std::runtime_error("out of KV pages");
    while ((int)s.pages.size() < need) {
        // Fresh pages start zeroed: masked slots are multiplied by zero, but WMMA rounding still depends on
        // their values (1 ulp), so stale data from earlier sequences would make results history-dependent.
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
    if (s.pending < 0) { sn.top_v = s.top_v; sn.top_i = s.top_i; } else { sn.top_v.clear(); sn.top_i.clear(); }
    sn.pages.assign(s.pages.begin(), s.pages.begin() + (s.n + PAGE - 1) / PAGE);
    for (int p : sn.pages) page_ref[p]++;
    // Copy the state into a device staging buffer (fast, keeps the compute stream moving), then to host
    // memory on the copy stream. The staging buffer is reused once the previous host copy has finished.
    const size_t ssm = (size_t)48 * GV * GD * GD * 4, conv = (size_t)48 * QKV * 3 * 4;
    CK(hipStreamWaitEvent(st, copied, 0));
    CK(hipMemcpyAsync(stage_state, s.ssm, ssm, hipMemcpyDeviceToDevice, st));
    CK(hipMemcpyAsync(stage_state + ssm, s.conv, conv, hipMemcpyDeviceToDevice, st));
    CK(hipMemcpyAsync(stage_state + ssm + conv, s.h_last, D * 4, hipMemcpyDeviceToDevice, st));
    CK(hipEventRecord(staged, st));
    CK(hipStreamWaitEvent(cst, staged, 0));
    CK(hipMemcpyAsync(sn.host, stage_state, state_bytes(), hipMemcpyDeviceToHost, cst));
    CK(hipEventRecord(copied, cst));
    return (int)si;
}

void Engine::restore(int id, int snap) {
    Seq &s = seqs.at(id);
    Snap &sn = snaps.at(snap);
    CK(hipStreamWaitEvent(st, copied, 0));      // the snapshot's host copy may still be in flight
    for (int p : s.pages) page_release(p);
    s.pages = sn.pages;
    for (int p : s.pages) page_ref[p]++;
    if (!s.pages.empty()) CK(hipMemcpyAsync(s.d_pages, s.pages.data(), s.pages.size() * 4, hipMemcpyHostToDevice, st));
    s.n = sn.n;
    s.pending = -1;
    s.top_v = sn.top_v; s.top_i = sn.top_i; s.seen.clear();
    const uint8_t *h = (const uint8_t *)sn.host;
    const size_t ssm = (size_t)48 * GV * GD * GD * 4, conv = (size_t)48 * QKV * 3 * 4;
    CK(hipMemcpyAsync(s.ssm, h, ssm, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(s.conv, h + ssm, conv, hipMemcpyHostToDevice, st));
    CK(hipMemcpyAsync(s.h_last, h + ssm + conv, D * 4, hipMemcpyHostToDevice, st));
}

void Engine::snap_free(int snap) {
    Snap &sn = snaps.at(snap);
    CK(hipStreamSynchronize(st));            // pending copies into sn.host must finish before reuse
    CK(hipStreamSynchronize(cst));
    for (int p : sn.pages) page_release(p);
    sn.pages.clear();
    sn.used = false;
}

// ---- building blocks --------------------------------------------------------------------------

// y = x W^T for prefill (x fp16 [T][ldx], W fp16 [N][K]) with hipBLASLt (~95 TFLOPS on gfx1100 vs ~60
// for rocBLAS). Its fp16-output kernels are the fast ones, so fp32 results go through an fp16 scratch and
// a convert/accumulate pass. Plans (descriptors + heuristic algorithm) are cached per shape.
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

void Engine::gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate) {
    const float alpha = 1.0f, beta = 0.0f;
    half *dst = y_f16 && !accumulate ? (half *)y : b.gemm16;
    const int ldd = dst == y ? ldy : N;
    const int Tp = std::min(umax, (T + 63) / 64 * 64);   // padded rows (ignored downstream) let plans be reused
    LtPlan &p = lt_plan(N, K, Tp, ldx, ldd);
    if (hipblasLtMatmul(lt, p.desc, &alpha, W, p.a, x, p.b, &beta, dst, p.c, dst, p.c, &p.algo, lt_ws, lt_ws_size, st) != HIPBLAS_STATUS_SUCCESS)
        FAIL("hipblasLtMatmul failed (%dx%dx%d)", N, K, T);
    if (dst != y) store_f16(b.gemm16, N, T, (float *)y, ldy, accumulate, st);
}

void Engine::linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16) {
    if (T <= 8) {
        gemv(W, qin, T, (float *)out, ldo, mode, st);
    } else if (T <= GEMM_Q_MAX && W.type != F16) {
        gemm_q(W, hin, ldin, T, out, ldo, mode, out_f16, st);
    } else {
        const half *w16 = (const half *)W.f;
        if (W.type != F16) { dequant(W, b.deq, st); w16 = b.deq; }
        gemm(w16, W.rows, W.cols, hin, ldin, T, out, ldo, out_f16, mode == GEMV_ADD);
    }
}

// Residual projection x += W.in, then normalize x with `next` (weights of the following norm) into
// b.hq/b.hh for the next block; `nf` additionally receives fp32 output. Fused into one GEMV when decoding.
void Engine::residual(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, float *x, const float *next, float *nf) {
    if (T <= 8) {
        gemv(W, qin, T, x, D, GEMV_ADD, st);
        if (next) norm_rows(x, next, T, nf, b.hh, b.hq, st);
    } else {
        linear(W, T, NO_Q, hin, ldin, x, D, GEMV_ADD);
        if (next) rmsnorm(x, next, T, D, D, D, nf, b.hh, NO_Q, st);
    }
}

void Engine::attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *x) {
    const bool small = T <= 8;
    if (small) {
        GemvArgs a;
        a.seg[0] = {w.q, b.proj, 2 * NH * HD}; a.seg[1] = {w.k, b.k, NKV * HD}; a.seg[2] = {w.v, b.v, NKV * HD};
        a.nseg = 3; a.x = b.hq; a.T = T;
        gemv(a, st);
    } else {
        linear(w.q, T, NO_Q, b.hh, D, b.proj, 2 * NH * HD, GEMV_STORE);
        linear(w.k, T, NO_Q, b.hh, D, b.k, NKV * HD, GEMV_STORE);
        linear(w.v, T, NO_Q, b.hh, D, b.v, NKV * HD, GEMV_STORE);
    }
    attn_prep(b.proj, b.k, b.v, w.q_norm, w.k_norm, T, pos0, kv(s), kvl, b.Q16, st);
    AttnArgs a;
    a.Q16 = b.Q16; a.kv = kv(s); a.layer = kvl; a.T = T; a.pos0 = pos0; a.qg = b.proj;
    a.part_o = b.part_o; a.part_ml = b.part_ml;
    attention(a, small ? nullptr : b.mix, small ? b.mq : NO_Q, st);
    residual(w.o, T, b.mq, b.mix, NH * HD, x, w.post_norm, nullptr);
}

void Engine::gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *x) {
    if (T <= 8) {
        float *raw = b.v_raw + (size_t)gl * 8 * QKV;
        GemvArgs a;
        a.seg[0] = {w.qkv, raw, QKV}; a.seg[1] = {w.z, b.proj, GV * GD}; a.nseg = 2; a.x = b.hq; a.T = T;
        gemv(a, st);
        GdnStepArgs g;
        g.raw = raw; g.z = b.proj; g.hh = b.hh; g.Wab = (const half *)w.ab.f;
        g.conv_w = w.conv; g.A = gdn_A + gl * GV; g.dt = gdn_dt + gl * GV; g.norm_w = w.ssm_norm;
        g.conv_state = s.conv + (size_t)gl * QKV * 3; g.S = s.ssm + (size_t)gl * GV * GD * GD;
        g.T = T; g.verify = verify;
        g.kn_out = b.v_kn + (size_t)gl * 8 * GK * GD; g.vc_out = b.v_vc + (size_t)gl * 8 * GV * GD;
        g.ab_out = b.v_ab + (size_t)gl * 8 * 2 * GV; g.out = b.mq;
        gdn_step(g, st);
        residual(w.ssm_out, T, b.mq, nullptr, GV * GD, x, w.post_norm, nullptr);
        return;
    }
    float *raw = b.proj, *z = b.proj + (size_t)T * QKV, *ab = b.k;
    linear(w.qkv, T, NO_Q, b.hh, D, raw, QKV, GEMV_STORE);
    linear(w.z, T, NO_Q, b.hh, D, z, GV * GD, GEMV_STORE);
    gemm((const half *)w.ab.f, 2 * GV, D, b.hh, D, T, ab, 2 * GV, false, false);
    gdn_conv(raw, T, w.conv, s.conv + (size_t)gl * QKV * 3, true, b.qn, b.kn, b.vc, st);
    GdnScanArgs sa{};
    sa.S = s.ssm + (size_t)gl * GV * GD * GD; sa.qn = b.qn; sa.kn = b.kn; sa.vc = b.vc; sa.ab = ab;
    sa.A = gdn_A + gl * GV; sa.dtb = gdn_dt + gl * GV; sa.T = T; sa.write_state = 1; sa.out = b.vc;
    gdn_scan(sa, 1, st);
    gdn_out(b.vc, z, w.ssm_norm, T, b.mix, NO_Q, st);
    residual(w.ssm_out, T, NO_Q, b.mix, GV * GD, x, w.post_norm, nullptr);
}

void Engine::ffn_block(const Layer &w, int T, float *x, const float *next, float *nf) {
    if (T <= 8) {
        GemvArgs a;
        a.seg[0] = {w.ffn_gate, nullptr, 0}; a.up = w.ffn_up; a.x = b.hq; a.T = T; a.mode = GEMV_GLU; a.gq = b.aq;
        gemv(a, st);
        residual(w.ffn_down, T, b.aq, nullptr, FF, x, next, nf);
    } else {
        linear(w.ffn_gate, T, NO_Q, b.hh, D, b.ffn_g, FF, GEMV_STORE, true);
        linear(w.ffn_up, T, NO_Q, b.hh, D, b.ffn_u, FF, GEMV_STORE, true);
        swiglu_h(b.ffn_g, b.ffn_u, (int64_t)T * FF, st);
        residual(w.ffn_down, T, NO_Q, b.ffn_g, FF, x, next, nf);
    }
}

// The residual stream stays normalized for the next block: every block's last projection also
// produces the normalized input of the following one (b.hq for GEMVs, b.hh for GEMMs).
void Engine::forward(Seq &s, int T, bool verify) {
    ensure_pages(s, s.n + T, s.n);
    embed(embd, b.tokens, T, b.x, st);
    if (T <= 8) norm_rows(b.x, L[0].attn_norm, T, nullptr, b.hh, b.hq, st);
    else rmsnorm(b.x, L[0].attn_norm, T, D, D, D, nullptr, b.hh, NO_Q, st);
    for (int l = 0, gl = 0; l < NL; l++) {
        if (is_attn_layer(l)) attn_block(L[l], l / 4, s, T, s.n, b.x);
        else gdn_block(L[l], gl++, s, T, verify, b.x);
        const bool last = l == NL - 1;
        ffn_block(L[l], T, b.x, last ? out_norm : L[l + 1].attn_norm, last ? b.hfin : nullptr);
    }
    if (T <= 8 && !verify) gdn_conv_commit(s.conv, b.v_raw, T, 8 * QKV, 48, st);   // decode: advance conv states
}

// MTP block over pairs (token at pos0+t, final hidden of pos0+t-1). For T <= 8 its output hidden goes
// to b.mtp_x (fp32, chained into the next draft) and b.hq (input of the LM head).
void Engine::mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *tokens) {
    const bool small = T <= 8;
    float *xm = small ? b.mtp_x : b.mtp_h;          // large T: h_prev (= b.mtp_h) is consumed before xm is written
    embed(embd, tokens, T, b.x, st);
    rmsnorm(b.x, enorm, T, D, D, 2 * D, nullptr, b.hh, small ? b.hq : NO_Q, st);
    rmsnorm(h_prev, hnorm, T, D, D, 2 * D, nullptr, b.hh + D, small ? QAct{b.hq.q + D, b.hq.d + D / 32} : NO_Q, st);
    if (small) {
        gemv(eh_proj, b.hq, T, xm, D, GEMV_STORE, st);
        norm_rows(xm, M.attn_norm, T, nullptr, b.hh, b.hq, st);
    } else {
        linear(eh_proj, T, NO_Q, b.hh, 2 * D, xm, D, GEMV_STORE);
        rmsnorm(xm, M.attn_norm, T, D, D, D, nullptr, b.hh, NO_Q, st);
    }
    attn_block(M, NATT, s, T, pos0, xm);
    ffn_block(M, T, xm, small ? head_norm : nullptr, small ? b.mtp_x : nullptr);
}

void Engine::lm_head(const QAct &in, int T, int rows) {
    QTensor w = out;
    w.rows = rows;
    gemv(w, in, T, b.logits, VOCAB, GEMV_STORE, st);
}

// ---- prefill and speculative generation --------------------------------------------------------

void Engine::prefill(int id, const int *tokens, int n) {
    if (!embd.q) FAIL("prefill before warmup(): the token embeddings are not loaded");
    Seq &s = seqs.at(id);
    s.pending = -1;
    for (int done = 0; done < n;) {
        const int T = std::min(umax, n - done);
        CK(hipMemcpyAsync(b.tokens, tokens + done, T * 4, hipMemcpyHostToDevice, st));
        forward(s, T, false);
        mtp_catch_up(s, T, s.n);
        if (done + T == n) {   // top-k logits for the next token
            quant_act(b.hfin + (size_t)(T - 1) * D, D, b.hq, st);
            lm_head(b.hq, 1, VOCAB);
            topk(b.logits, 1, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
            s.top_v.resize(TOPK); s.top_i.resize(TOPK);
            CK(hipMemcpyAsync(s.top_v.data(), b.topv, TOPK * 4, hipMemcpyDeviceToHost, st));
            CK(hipMemcpyAsync(s.top_i.data(), b.topi, TOPK * 4, hipMemcpyDeviceToHost, st));
        }
        CK(hipMemcpyAsync(s.h_last, b.hfin + (size_t)(T - 1) * D, D * 4, hipMemcpyDeviceToDevice, st));
        s.n += T;
        done += T;
    }
    CK(hipStreamSynchronize(st));
}

// Run the MTP block over the T tokens just processed at pos0.., pairing each with the previous
// position's final hidden state, so the MTP layer's KV cache tracks the sequence.
void Engine::mtp_catch_up(Seq &s, int T, int pos0) {
    CK(hipMemcpyAsync(b.mtp_h, s.h_last, D * 4, hipMemcpyDeviceToDevice, st));
    if (T > 1) CK(hipMemcpyAsync(b.mtp_h + D, b.hfin, (size_t)(T - 1) * D * 4, hipMemcpyDeviceToDevice, st));
    mtp_forward(s, T, pos0, b.mtp_h, b.tokens);
}

namespace {
struct Dist { int n = 0; int id[TOPK]; float p[TOPK]; };

Dist make_dist(const float *v, const int *ids, const Sampling &sp, const std::vector<int> &seen) {
    std::pair<float, int> c[TOPK];
    int nc = 0;
    for (int i = 0; i < TOPK; i++) {
        if (ids[i] == sp.ban) continue;                 // not a candidate at all (so not a fallback either)
        float x = v[i];
        if (sp.presence_penalty != 0 && std::find(seen.begin(), seen.end(), ids[i]) != seen.end()) x -= sp.presence_penalty;
        c[nc++] = {x, ids[i]};
    }
    std::sort(c, c + nc, [](auto &a, auto &b) { return a.first > b.first; });
    Dist d;
    if (sp.temperature <= 0) { d.n = 1; d.id[0] = c[0].second; d.p[0] = 1; return d; }
    const int k = std::max(1, std::min(sp.top_k > 0 ? sp.top_k : nc, nc));
    double sum = 0;
    for (int i = 0; i < k; i++) { d.p[i] = expf((c[i].first - c[0].first) / sp.temperature); d.id[i] = c[i].second; sum += d.p[i]; }
    d.n = k;
    double cum = 0;
    for (int i = 0; i < k; i++) {                       // top-p, then min-p
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

int draw_without(Dist d, int tok, std::mt19937_64 &rng) {   // residual distribution after rejecting tok
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
    const int k = std::max(0, std::min(sp.n_draft, std::min(7, max_out - nout - 1)));
    const int q = s.n;
    ensure_pages(s, q + k + 1, q);

    // 1. draft k tokens with the MTP head (greedy, on the GPU, no host round trips)
    CK(hipMemcpyAsync(b.draft, &s.pending, 4, hipMemcpyHostToDevice, st));
    for (int i = 0; i < k; i++) {
        mtp_forward(s, 1, q + i, i == 0 ? s.h_last : b.mtp_x, b.draft + i);
        lm_head(b.hq, 1, DRAFT_VOCAB);
        argmax(b.logits, 1, DRAFT_VOCAB, VOCAB, b.draft + i + 1, st);
    }
    // 2. verify pending + drafts with the full model
    const int T = k + 1;
    CK(hipMemcpyAsync(b.tokens, b.draft, T * 4, hipMemcpyDeviceToDevice, st));
    forward(s, T, k > 0);
    lm_head(b.hq, T, VOCAB);
    topk(b.logits, T, VOCAB, VOCAB, b.tsv, b.tsi, b.topv, b.topi, st);
    std::vector<float> tv(T * TOPK);
    std::vector<int> ti(T * TOPK), drafts(T);
    CK(hipMemcpyAsync(tv.data(), b.topv, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(ti.data(), b.topi, T * TOPK * 4, hipMemcpyDeviceToHost, st));
    CK(hipMemcpyAsync(drafts.data(), b.draft, T * 4, hipMemcpyDeviceToHost, st));
    // 3. MTP catch-up on the verified pairs (independent of acceptance; rejected rows get overwritten)
    mtp_catch_up(s, T, q);
    CK(hipStreamSynchronize(st));

    // 4. accept drafts (rejection sampling against greedy drafts is lossless) and draw the next token
    int a = 0, next = -1;
    for (; a < k; a++) {
        const Dist d = make_dist(&tv[a * TOPK], &ti[a * TOPK], sp, s.seen);
        const int tok = drafts[a + 1];
        const float pa = prob(d, tok);
        if (std::uniform_real_distribution<float>(0, 1)(s.rng) < pa) {
            if (is_eos(tok)) { next = tok; break; }        // an accepted end-of-turn stays pending, like a sampled one
            out[nout++] = tok; s.seen.push_back(tok);
            continue;
        }
        next = draw_without(d, tok, s.rng);
        break;
    }
    if (next < 0) next = draw(make_dist(&tv[a * TOPK], &ti[a * TOPK], sp, s.seen), s.rng);
    out[nout++] = next;
    s.seen.push_back(next);

    // 5. commit the recurrent state to the accepted prefix (pending + a drafts)
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
    CK(hipMemcpyAsync(s.h_last, b.hfin + (size_t)a * D, D * 4, hipMemcpyDeviceToDevice, st));
    s.n = q + a + 1;
    s.pending = next;
    return nout;
}
