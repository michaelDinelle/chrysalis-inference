#pragma once
#include "common.h"
#include "args.h"
#include "sampler.h"
#include "lookup.h"
#include <random>
#include <string>
#include <vector>

constexpr int TOPK = 64;
static_assert(TOPK == SAMPLE_N, "the sampler takes the engine's rows as they are");
constexpr int EOS_IM_END = 248046, EOS_TEXT = 248044;
constexpr int IMG_BASE = 1 << 30, IMG_ROWS = 1 << 14;
constexpr size_t SSM_BYTES = (size_t)48 * GV * GD * GD * 4, CONV_BYTES = (size_t)48 * QKV * 3 * 4;
inline size_t state_bytes() { return SSM_BYTES + CONV_BYTES + D * 4; }
inline bool is_eos(int t) { return t == EOS_IM_END || t == EOS_TEXT; }

void relayout(const QTensor &t, const uint8_t *raw, size_t b0, size_t b1);

struct Layer {
    DP<float> attn_norm, post_norm;
    QTensor ffn_gate, ffn_up, ffn_down;
    QTensor qkv, z, ssm_out;
    DP<uint16_t> ab;
    DP<float> conv, ssm_norm;
    QTensor q, k, v, o;
    DP<float> q_norm, k_norm;
};

struct Seq {
    bool used = false;
    int n = 0;
    int pos = 0;
    int pending = -1;
    std::vector<int> pages;
    DP<int64_t> d_pages;
    DP<float> ssm, conv;
    DP<float> h_last;
    std::vector<float> top_v;
    std::vector<int> top_i;
    Seen seen;
    std::vector<int> ctx;
    std::mt19937_64 rng;
    Lookup look;
};

struct Engine {
    Mtl m;
    size_t weight_bytes = 0;
    QTensor embd, out;
    DP<float> out_norm;
    Layer L[NL], M;
    QTensor eh_proj;
    DP<float> enorm, hnorm, head_norm;
    DP<float> gdn_A, gdn_dt;
    static constexpr size_t ATT_S = (size_t)64 << 20;
    int attn_gemm_min = 4096;
    int mps_min = 512;
    int umax = 2048;
    static constexpr int DRAFT_VOCAB = 65536;
    static constexpr int MAX_SPLITS = 256;
    struct {
        DP<int> tokens; DP<float> x, hfin, xn, proj, k, v, qn, kn, vc, o, logits, part_o, part_ml, ab;
        DP<half_t> hh, deq, Qg, mix, ffn_g, ffn_u, kc, vt, pt;
        DP<float> st;
        DP<float> topv, tsv, tops; DP<int> topi, tsi;
        DP<float> v_raw, v_kn, v_vc, v_ab;
        DP<float> mtp_h, mtp_x, mtp_xn; DP<int> draft;
        DP<int> rpos;
        DP<uint32_t> mask;
    } b{};
    int qrows_max = 0;
    static constexpr int SEG_PAGES = 128;
    DP<uint8_t> kv_pool;
    std::vector<DP<uint8_t>> page_mem;
    int n_pages = 0;
    std::vector<int> free_pages;
    void init_kv(int max_pages);
    bool grow_kv();
    int free_page_count() const { return (int)free_pages.size() + n_pages - (int)page_mem.size(); }
    std::vector<Seq> seqs;
    std::vector<int> page_ref;
    struct Snap { bool used = false; int n = 0, pos = 0; std::vector<int> pages; DP<uint8_t> host;
                  std::vector<float> top_v; std::vector<int> top_i; };
    std::vector<Snap> snaps;

    void load(const std::string &path, int max_seqs, size_t reserve_bytes);
    void warmup();
    void alloc_buffers(int max_seqs);
    int seq_new();
    void seq_reset(Seq &s);
    void seq_free(int id);
    void seq_truncate(int id, int n);
    int snapshot(int id);
    void restore(int id, int snap);
    void snap_free(int snap);
    size_t snap_bytes(int snap) const;
    void snap_export(int snap, uint8_t *dst);
    int snap_import(const uint8_t *src, size_t bytes);
    void prefill(int id, const int *tokens, const int *pos3, int n);
    int generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v = nullptr, int *rows_i = nullptr);
    int draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out);
    int generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out);
    int verify(Seq &s, const Sampling &sp, int k, bool masked, int *out, int nout, float *rows_v, int *rows_i);
    void generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v = nullptr,
                        int *rows_i = nullptr);

    struct Part { Seq *s; int T, off, pos0, rope0; bool rpos; bool verify; };
    void forward_parts(const Part *p, int np, int T);
    void forward(Seq &s, int T, bool verify, int rope0, bool rpos = false);
    void mtp_forward(Seq &s, int T, int pos0, DP<float> h_prev, DP<int> tokens, int rope0, bool rpos = false);
    void mtp_catch_up(Seq &s, int T, int pos0, int rope0, bool rpos = false, int off = 0);
    void lm_head(DP<float> in, int T, int rows);
    void ensure_pages(Seq &s, int n_tokens, int from);
    struct Deferred { Seq *s; int off, n, hrow; };
    std::vector<Deferred> deferred;
    void defer_commit(Seq &s, int off, int n, int hrow);
    void flush();
    int page_alloc();
    void page_release(int p);

    void gemv(const QTensor &W, DP<float> x, DP<half_t> x16, int ldx, int T, DP<float> y, int ldy, int mode, DP<float> nw = {});
    void gemv_glu(const QTensor &G, const QTensor &U, DP<float> x, DP<half_t> x16, int T, DP<float> nw = {});
    void linear(const QTensor &W, int T, DP<half_t> hin, int ldin, DP<float> xin, int ldx, DP<uint8_t> out, int ldo, bool add, bool out_f16);
    void norm1(DP<float> x, DP<float> w, DP<float> out);
    void rmsnorm(DP<float> x, DP<float> w, int T, int n, int ldx, int ldo, DP<float> of, DP<half_t> oh);
    void copy(DP<uint8_t> dst, DP<uint8_t> src, size_t bytes);
    void fill0(DP<uint8_t> dst, size_t bytes);
    void topk(DP<float> logits, int T, int n, int ld);
    void argmax(DP<float> logits, int T, int n, int ld, DP<int> out);
    void logit_stats(DP<float> logits, int T, int n, int ld);
    void embed(DP<int> tokens, int T, DP<float> out);
    void attend(const Layer &w, int kvl, const Part &q, bool small);
    void attn_layer(const Layer &w, int kvl, const Part *p, int np, int T, DP<float> x, DP<float> xn, DP<half_t> hh);
    void gdn_layer(const Layer &w, int gl, const Part *p, int np, int T, DP<float> x, DP<float> xn, DP<half_t> hh);
    void ffn_layer(const Layer &w, int T, DP<float> x, DP<float> xn, DP<half_t> hh, DP<float> next, DP<float> nf);
};
