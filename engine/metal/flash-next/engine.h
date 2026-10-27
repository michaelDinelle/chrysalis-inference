#pragma once
#include "mtl.h"
#include "engine/sampler.h"
#include "engine/lookup.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <random>
#include <thread>

constexpr int TOPN = 64;
static_assert(TOPN == SAMPLE_N, "topk64 writes the sampler's candidate count");

inline bool is_eos(int t) { return t == EOS_IM_END || t == EOS_TEXT; }
inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct WT { int type = -1; int rows = 0, cols = 0; size_t rb = 0; View v; };
struct Mix { View norm; WT down, up, inject; };

struct Layer {
    Mix hca, hcf;
    WT qkv, z, ab, ssm_out;
    View conv_w, ssm_norm, A, dt;
    WT q, k, v, o, iq, ik;
    View qn, kn, iqn, ikn;
    WT ple_key, ple_value;
    View ple_nk, ple_nq, ple_nc, ple_conv;
    WT router, sg, sh_gate, sh_up, sh_down;
};

struct Seq {
    bool used = false;
    int n = 0;
    int pending = -1;
    std::vector<int> pages;
    View d_pages;
    View ssm, conv, ple, tail, h_last;
    int hist[2] = {-1, -1};
    std::vector<float> top_v;
    std::vector<int> top_i;
    Seen seen;
    std::vector<int> ctx;
    std::mt19937_64 rng;
    Lookup look;
};

inline size_t ssm_bytes() { return (size_t)NGDN * GV * GD * GD * 4; }
inline size_t conv_bytes() { return (size_t)NGDN * QKV * 3 * 4; }
inline size_t ple_bytes() { return (size_t)PLE_HIST * HCD * 4; }
inline size_t tail_bytes() { return (size_t)NKVL * KPOOL * IDX_D * 2; }
inline size_t state_bytes() { return ssm_bytes() + conv_bytes() + ple_bytes() + tail_bytes() + (size_t)HCD * 4; }

struct Experts {
    int n_slots = 0, n_stage = 0;
    int ring0 = 0, ring = 0;
    std::vector<int> ring_keys[2];
    std::vector<View> bufs;
    std::vector<uint8_t *> slot_host;
    std::vector<uint64_t> slot_gpu;
    std::vector<int> owner;
    std::vector<int> slot_of;
    std::vector<float> score;
    std::vector<float> counts;
    std::vector<uint32_t> last_use;
    uint32_t pass = 0;
    double decay = 0.997;
    double boost = 1.0;
    int min_layer = 0;
    std::vector<int> per_layer;
    std::vector<int> stage_free;
    View table;
    View fmt;
    std::vector<XSrc> src;
    int fd = -1;
    std::mt19937 rng{7};
    uint64_t picks = 0, hits = 0, reads = 0, staged = 0;
    uint64_t pred_misses = 0, pred_caught = 0;
    double read_s = 0, read_bytes = 0;
    struct Job { int fd; size_t off, n; uint8_t *dst; std::atomic<int> *left = nullptr; };
    struct Ahead { int key, slot; std::atomic<int> left{0}; };
    std::vector<std::unique_ptr<Ahead>> ahead;
    std::vector<int> ahead_of;
    std::vector<uint8_t> reserved;
    uint64_t ahead_reads = 0, ahead_used = 0;
    std::vector<std::thread> io;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    std::deque<Job> jobs;
    int inflight = 0;
    bool quit = false;
};

struct Engine {
    Mtl m;
    GGUF g;
    WT embd, out;
    Mix hc_head;
    Layer L[NL];
    Layer M;
    WT eh_e, eh_h, dout;
    View enorm, hnorm, mtp_x;
    Mix mtp_head;
    bool has_mtp = false;
    id<MTLCommandBuffer> pending = nil;
    int npass = 0;
    void settle();
    Arena warena, barena;
    View kv;
    int n_pages = 0, max_pages_seq = 0;
    std::vector<int> free_pages, page_ref;
    std::vector<Seq> seqs;
    Experts xc;
    id<MTLSharedEvent> ev_g = nil, ev_c = nil;
    uint64_t evg = 0, evc = 0;
    const uint8_t *ngram = nullptr; size_t ngram_bytes = 0; void *ngram_map = nullptr;
    int ngram_fd = -1; size_t ngram_off = 0;
    uint64_t ple_mult[3] = {};
    uint32_t ple_off[PLE_HEADS] = {}, ple_voc[PLE_HEADS] = {};
    int TM = 8;
    int umax = 2048;
    int max_ctx = 65536;
    int max_blocks = 0;
    int prof = getenv("CHRYSALIS_FN_PROF") ? std::max(1, atoi(getenv("CHRYSALIS_FN_PROF"))) : 0;
    double t_wait = 0, t_io = 0;
    struct {
        View res, xn, lo, loh, gate, inj, inj2, mx, mxh;
        View raw, z, ab, go, blk;
        View qg, kb, vb, qir, kir, Q, Qi, score, sel, nsel, part_o, part_ml, att, atth;
        View rlog, ids, wts, sgv, shg, shu, shh, shout, xh, xy;
        View emb, embh, key, value, xconv, ple_rows;
        View logits, topv, topi, tops, topv1, topi1, tokens, mask, inv, done, ind_args, ind_grids, desc, ids_pred, wts_pred, rlog_pred;
        View vraw, vab, vkir, memb, mhp, mres, dlog, dtok;
    } b;
    int last_row = 0;
    int U = 1024, PMAX = 8192;
    struct {
        View res, xn, lo, gate, inj, inj2, mx, mxh, raw, z, ab, qn, kn, vc, blk, qg, kb, vb, qir, kir, Q, Qi, score, sel, nsel,
             part_o, part_ml, att, atth, rlog, ids, wts, sgv, shg, shu, shh, shout, Y, emb, embh, key, value, xconv, ple_rows, moe, tokens, tokens2;
    } pb;
    void alloc_prompt_buffers();
    void prompt_pass(Seq &s, const int *tokens, int P);
    struct Snap { bool used = false; int n = 0, hist[2] = {-1, -1}; std::vector<int> pages; std::vector<uint8_t> state;
                  std::vector<float> top_v; std::vector<int> top_i; };
    std::vector<Snap> snaps;

    void load(const std::string &path, int max_seqs);
    void shutdown();
    WT weight(const std::string &name, bool required = true);
    View small(const std::string &name);
    void alloc_buffers(int max_seqs);

    int seq_new();
    void seq_reset(Seq &s);
    void seq_free(int id);
    void ensure_pages(Seq &s, int n_tokens, int from);
    int page_alloc();
    void page_release(int p);
    int snapshot(int id);
    void restore(int id, int snap);
    void snap_free(int snap);

    void prefill(int id, const int *tokens, int n);
    int generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v, int *rows_i);
    int draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out);
    int generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out);
    int verify(Seq &s, const Sampling &sp, const int *drafts, int k, bool masked, int *out, int nout, float *rows_v, int *rows_i);
    void commit(Seq &s, int q, int n);
    void mtp_layer(Enc &e, Seq &s, int T, uint32_t pos0, View hp, View tok, bool spec);
    void mtp_catch_up(Seq &s, int T, int pos0, bool spec);
    void mtp_draft(Seq &s, int k, int *out, int nfixed = 0);
    void load_mtp();
    void forward_small(Seq &s, const int *tokens, int T, bool all_logits, bool verify = false);
    void ple_ids(int h0, int h1, const int *tokens, int T, size_t *rows) const;
    void ple_gather(const Seq &s, const int *tokens, int T, uint8_t *dst) const;
    void advance(Seq &s, const int *tokens, int T);

    void gemv(Enc &e, const WT &w, View x, int ldx, View y, int ldy, int T, int add = 0, float scale = 1.0f, View inj = View{});
    void gemv_combine(Enc &e, const WT &w, View x, int ldx, int T, View inj) { gemv(e, w, x, ldx, b.res, HCD, T, 2, 1.0f, inj); }
    void mixer(Enc &e, const Mix &mx, View res, int T, View inj);
    void topk(Enc &e, int rows);
    void keep_top(Seq &s, int row = 0);
    void wait_event(id<MTLSharedEvent> ev, uint64_t v, id<MTLCommandBuffer> cb);

    void experts_init(size_t cache_bytes, int n_stage);
    void experts_stop();
    int experts_resolve(int layer, const int *ids, int n, const std::function<void()> &publish_ok);
    void experts_read(int layer, int e, uint8_t *dst);
    void expert_jobs(int layer, int e, uint8_t *dst, std::atomic<int> *left, std::vector<Experts::Job> &jobs) const;
    void io_submit(const std::vector<Experts::Job> &jobs, bool warm = false);
    void keep_warm();
    View warm_buf;
    void experts_fill(double seconds_budget);
    void experts_tick(int T);
    void experts_stage_layer(int layer, int half);
    void experts_unstage(int half);
    void experts_ahead(int layer, const int *ids, int n);
};
const float *top_stats(const Seq &s);

const char *tname(int type);
