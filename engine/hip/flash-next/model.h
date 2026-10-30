#pragma once
#include "kernels.h"
#include "../../sampler.h"
#include "../../lookup.h"
static_assert(TOPK == SAMPLE_N, "the sampler takes the engine's rows as they are");
static_assert(LOGIT_STATS == NSTAT, "the sampler reads the statistics the engine writes");
#include "cpu_moe.h"
#include "../vision.h"
#include <hipblaslt/hipblaslt.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

extern std::atomic<int> warm_done, warm_total;

inline bool is_eos(int t) { return t == EOS_IM_END || t == EOS_TEXT; }
inline size_t ssm_bytes() { return (size_t)NGDN * GV * GD * GD * 4; }
inline size_t conv_bytes() { return (size_t)NGDN * QKV * 3 * 4; }
inline size_t ple_bytes() { return (size_t)PLE_HIST * HCD * 4; }
inline size_t tail_bytes() { return (size_t)NKVL * IDX_TAIL * 2; }
inline size_t state_bytes() { return ssm_bytes() + conv_bytes() + ple_bytes() + (size_t)HCD * 4 + tail_bytes(); }

struct HcMix { float *norm = nullptr; QTensor down, up, inject; };

struct Layer {
    HcMix hca, hcf;
    QTensor qkv, z, ssm_out;
    half *Wab = nullptr;
    float *conv = nullptr, *ssm_norm = nullptr;
    QTensor q, k, v, o, idx_q, idx_k;
    float *q_norm = nullptr, *k_norm = nullptr, *idx_qn = nullptr, *idx_kn = nullptr;
    QTensor ple_key, ple_value;
    float *ple_nk = nullptr, *ple_nq = nullptr, *ple_nc = nullptr, *ple_conv = nullptr;
    half *router = nullptr;
    float *sg_w = nullptr;
    QTensor sh_gate, sh_up, sh_down;
    QTensor hot_gate, hot_up, hot_down;
    int *slot_of = nullptr;
    std::vector<int> slot_h;
    int n_hot = 0, n_cold = 0;
};

struct Seq {
    bool used = false;
    int n = 0;
    int pos = 0;
    Pos3 tail_pos{0, 0, 0};
    int pending = -1;
    std::vector<int> pages;
    int *d_pages = nullptr;
    float *ssm = nullptr, *conv = nullptr, *ple = nullptr;
    float *h_last = nullptr;
    half *idx_tail = nullptr;
    int hist[2] = {-1, -1};
    std::vector<float> top_v;
    std::vector<int> top_i;
    Seen seen;
    std::vector<int> ctx;
    std::mt19937_64 rng;
    Lookup look;
};

struct Engine {
    QTensor embd;
    void *embd_host = nullptr;
    QTensor out;
    HcMix hc_head;
    Layer L[NL], M;
    QTensor eh_proj;
    float *enorm = nullptr, *hnorm = nullptr;
    HcMix mtp_head;
    float *gdn_A = nullptr, *gdn_dt = nullptr;
    const uint8_t *ngram = nullptr; size_t ngram_map_bytes = 0; uint8_t *ngram_map = nullptr;
    uint64_t ple_mult[3] = {};
    uint32_t ple_off[PLE_HEADS] = {}, ple_voc[PLE_HEADS] = {};
    QTensor pool_gate, pool_up, pool_down;
    int n_slots = 0, n_extra = 0, n_extra_max = 0, extra_pages = 0;
    int *d_slot_of = nullptr;
    uint8_t *d_fmt = nullptr;
    int file_fd = -1; const uint8_t *file_base = nullptr;
    uint8_t *spill_mem = nullptr; size_t spill_bytes = 0;
    struct Experts {
        bool on = true;
        double half_life = 256;
        std::vector<int> owner;
        std::vector<uint8_t> fmt;
        std::vector<int> slot, home, type;
        std::vector<float> score, mb;
        std::vector<uint32_t> fresh;
        std::vector<uint64_t> total;
        std::vector<uint8_t> busy;
        std::vector<uint8_t> coming;
        int spilled = 0, spill_cap = 0;
        double spare = 4e9;
        double avail = 0;
        std::chrono::steady_clock::time_point avail_at{};
        double tokens = 0;
        unsigned *d_counts = nullptr;
        uint32_t *h_counts = nullptr;
        hipEvent_t counted = nullptr;
        bool counts_pending = false;
        int placed = 0, loaded_file = 0;
        double copied_bytes = 0, copied_ms = 0;
        uint64_t picks = 0, hits = 0, mem_limited = 0, step = 0, waited_steps = 0;
        double waited_s = 0;
        bool decoding = false, reading = false;
        int decode_loads = 16;
        enum { FREE, QUEUED, COPYING };
        struct Load {
            std::atomic<int> state{FREE};
            int key = -1, slot = -1;
            int out = -1;
            uint64_t due = 0;
            uint8_t *stage = nullptr;
            uint8_t *bounce = nullptr;
            hipEvent_t start = nullptr, copied = nullptr, done = nullptr;
        };
        Load loads[PLACE_MAX];
        hipStream_t xst = nullptr;
        std::thread loader;
        std::mutex mu;
        std::condition_variable cv;
        struct Job { int what, after; };
        std::deque<Job> jobs;
        bool quit = false;
    } xc;
    void experts_init();
    void extra_init();
    void extra_grow();
    void kv_need(int need);
    int free_tokens() const { return (int)(free_pages.size() + extra_pages) * PAGE; }
    void experts_place(int T);
    void experts_plan();
    void experts_loader();
    void experts_stop();
    bool pool_save(const char *path) const;
    int pool_warm(const char *path);
    uint64_t model_bytes() const;
    void file_read(uint8_t *dst, const uint8_t *src, size_t n) const;
    void file_read3(uint8_t *const dst[3], const uint8_t *const src[3], const size_t n[3]) const;
    std::vector<ColdLayer> cold;
    std::vector<ColdDec> cold_dev;
    uint8_t *cold_mem = nullptr; size_t cold_bytes = 0;
    MoeShared *sh = nullptr, *d_sh = nullptr;
    CpuMoe cpu;
    float gfrac = 0.3f, xfrac = 1.0f;
    bool gfrac_pinned = false, xfrac_pinned = false;
    MoeEmu *me = nullptr, *d_me = nullptr;
    double measure_dma();
    double cal_cpu_bw = 50e9, cal_cpu_mac = 150e9, cal_dma_bw = 25e9;
    static constexpr int SPLIT_AT[3] = {32, 256, 4096};
    double cpu_bias[3] = {1.0, 1.0, 1.0}, restore_s = 4e-3;
    double split_bias(int T) const;
    std::string cal_path;
    bool calibrated = false;
    std::string cal_key() const;
    bool cal_load();
    void cal_split(int id);
    void cal_save();
    int n_workers = 0;
    int split = -1;
    bool prof = getenv("CHRYSALIS_FN_PROF") != nullptr;
    std::vector<int64_t> route_counts;
    const char *route_dec_file = getenv("CHRYSALIS_FN_DECODE_COUNTS");
    int route_dec_steps = 0;
    double prof_t[12] = {}, prof_last = 0;
    void pmark(int phase);
    uint32_t gen = 0;

    hipStream_t st, cst, lst;
    unsigned la_gen = 0, la_pend = 0;
    MoeShared *sh_pred = nullptr, *d_sh_pred = nullptr;
    struct { float *lo, *inv, *mx, *logits, *sg, *wts; half *hh; QAct hq; int *ids, *hot; unsigned *counter; } la{};
    bool la_on = getenv("CHRYSALIS_FN_NO_LOOKAHEAD") == nullptr;
    void lookahead_flags(const Layer &w, int layer, int T, const float *res, unsigned starts = 1);
    void la_sync() {
        if (la_pend) { wait_flag(la.counter + 17, la_pend, st); la_pend = 0; }
    }
    hipEvent_t staged, copied, topk_ready;
    hipEvent_t front_ready, front_copied;
    uint8_t *stage_state = nullptr;
    hipblasLtHandle_t lt = nullptr;
    void *lt_ws = nullptr;
    size_t lt_ws_size = 32 << 20;
    using LtPlan = lt_tune::Plan;
    std::map<std::tuple<int, int, int, int, int, int>, LtPlan> lt_plans;
    LtPlan &lt_plan(int N, int K, int T, int ldx, int ldd, bool c32, const half *W, const half *x, void *d);
    int smax = 8192;
    int umax = 1024;
    int max_ctx = 262144;
    static constexpr int DRAFT_VOCAB = 65536;
    static constexpr int GEMM_Q_MAX = 384;
    static constexpr size_t DEQ_ELEMS = (size_t)6144 * 2560;
    static constexpr int SEL_G = 64;
    static constexpr int MTP_CHUNK = 256;
    Arena warena, barena;
    struct {
        int *tokens; float *x, *res, *xn, *lo, *gate, *inj_a, *inj_f, *mx, *proj, *zb, *kb, *vb, *qidx, *kidx, *blk;
        half *xn_h, *lo_h, *hh, *Q16, *qidx16, *att_h, *deq, *gemm16;
        QAct xn_q, lo_q, hq, mq;
        float *part_o, *part_ml, *score; int *sel; uint32_t *sel_hist;
        float *qn, *kn, *vc;
        float *v_raw, *v_kn, *v_vc, *v_ab;
        float *logits_e, *sg, *wts, *y, *moe; int *ids, *hot;
        QAct hq_e;
        uint8_t *ple_rows, *d_ple_rows; float *emb, *key, *value, *xconv; half *emb_h; QAct emb_q;
        float *logits, *topv, *tsv, *tops; int *topi, *tsi;
        float *mtp_res, *mtp_e, *mtp_in; half *mtp_in_h; QAct mtp_in_q; int *draft;
        float *fin; QAct fin_q; half *fin_h;
        float *inv;
        half *idx_raw;
        unsigned *rr_counter;
        float *rlog, *sgT, *wtsT, *ent_w; int *idsT, *ent_tok, *pos, *btok; MoeWork *work;
        half *Xs, *Hs, *Ys, *hmoe;
        int *h_ids; float *h_wts; int *h_tok; float *h_entw; int *h_pos, *h_btok; MoeWork *h_work;
        uint16_t *h_x16, *h_cout;
        float *h_topv, *h_tops; int *h_topi, *h_draft;
        uint8_t *raw[4], *sgu[4]; QTensor sd_[4];
        uint8_t *bounce[4];
        const half **img;
        int *rpos;
    } b{};
    static constexpr int STAGE_G = 8;
    static constexpr int NSLOT = 2;
    size_t stage_gu = 0, stage_soa = 0;
    std::vector<int> j_slot, j_first, j_tok;
    std::vector<float> j_w;
    uint8_t *d_cold = nullptr;
    hipEvent_t raw_free[NSLOT], raw_ready[NSLOT];
    uint8_t *kv_pool = nullptr;
    int n_pages = 0;
    std::vector<int> free_pages;
    std::vector<Seq> seqs;
    std::vector<int> page_ref;
    struct Snap { bool used = false; int n = 0, pos = 0; Pos3 tail_pos{0, 0, 0}; int hist[2] = {-1, -1}; std::vector<int> pages;
                  void *host = nullptr; std::vector<float> top_v; std::vector<int> top_i; };
    Vision vision;
    struct Image { half *rows = nullptr; int n = 0; };
    std::vector<Image> images;
    std::vector<Snap> snaps;
    uint32_t *d_mask = nullptr;
    float *ssm_pool = nullptr;
    std::vector<void *> snap_spare;
    std::vector<std::pair<void *, size_t>> snap_regions;
    void *snap_pending = nullptr;
    int snap_target = 2, snap_pending_n = 0;
    bool snap_warm = false;
    std::vector<std::thread> snap_fill;
    void snap_reserve_start();
    void snap_reserve_finish();

    void load(const std::string &path, int max_seqs, size_t reserve_bytes);
    void calibrate();
    void warmup();
    void alloc_buffers(int max_seqs);
    int seq_new();
    void seq_reset(Seq &s);
    void seq_free(int id);
    void seq_truncate(int id, int n);
    int snapshot(int id);
    void *snap_buffer();
    void *take_snap_buffer();
    void reserve_snapshots(int n);
    void restore(int id, int snap);
    void snap_free(int snap);
    void prefill(int id, const int *tokens, const int *pos3, int n);
    int image_encode(int slot, const float *patches, int gh, int gw, int frames = 1);
    int generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v = nullptr, int *rows_i = nullptr);
    int draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out);
    int generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out);
    int verify(Seq &s, const Sampling &sp, const std::vector<int> &drafts, bool masked, int *out, int nout, float *rows_v, int *rows_i);
    uint32_t *mask_rows();
    size_t snap_bytes(int snap) const;
    void snap_export(int snap, uint8_t *dst);
    int snap_import(int fd, int64_t off, size_t bytes);
    void shutdown();

    void forward(Seq &s, const int *tokens, int T, bool verify, const int *pos3 = nullptr, bool img = false);
    void ple_rows(const Seq &s, const int *tokens, int T, int off = 0);
    void ple_ids(int h0, int h1, const int *tokens, int T, size_t *rows) const;
    void ple_prefetch(const size_t *rows, size_t n) const;
    void mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *d_tokens, bool spec, int rope0, const int *rpos,
                     Pos3 tail_pos, const half *const *img, int praw_off = 0);
    void mtp_catch_up(Seq &s, int T, int pos0, bool spec, const int *pos3 = nullptr, bool img = false, int off = 0);
    void lm_head(const QAct &in, int T, int rows);
    void ensure_pages(Seq &s, int n_tokens, int from);
    int page_alloc();
    void page_release(int p);
    KVRef kv(const Seq &s) const { return {kv_pool, s.d_pages}; }
    void linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16 = false);
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate);
    void gemm32(const half *W, int N, int K, const half *x, int ldx, int T, float *y, int ldy);
    void attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *res, float *inject, bool spec, int rope0,
                    const int *rpos, Pos3 tail_pos, int praw_off = 0);
    void attn_core(const Layer &w, int kvl, Seq &s, int T, int off, int praw_off, int pos0, bool spec, int rope0, const int *rpos,
                   Pos3 tail_pos);
    void gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *res, float *inject);
    void moe_block(const Layer &w, int layer, int T, bool cpu_cold, float *res, float *inject, unsigned starts = 1);
    struct Part { Seq *s; int T, off; bool verify; };
    void attn_parts(const Layer &w, int kvl, const Part *p, int np, int T, float *res, float *inject);
    void gdn_parts(const Layer &w, int gl, const Part *p, int np, int T, float *res, float *inject);
    void ple_parts(const Part *p, int np, int T, float *res);
    void forward_parts(const Part *p, int np, const int *tokens, int T);
    void generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v = nullptr,
                        int *rows_i = nullptr);
    void moe_front(const Layer &w, int n, int c0);
    void moe_back(const Layer &w, int layer, int T);
    void ple_block_run(Seq &s, int c0, int n, bool verify, float *res);
};
