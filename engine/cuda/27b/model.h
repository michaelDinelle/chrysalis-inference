#pragma once
#include "kernels.h"
#include "../../sampler.h"
#include "../../lookup.h"
#include "../vision.h"
static_assert(TOPK == SAMPLE_N, "the sampler takes the engine's rows as they are");
static_assert(LOGIT_STATS == NSTAT, "the sampler reads the statistics the engine writes");
#include <random>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <string>
#include <vector>

constexpr int EOS_IM_END = 248046, EOS_TEXT = 248044;
constexpr int IMG_BASE = 1 << 30, IMG_ROWS = 1 << 14;
inline size_t state_bytes() { return (size_t)48 * GV * GD * GD * 4 + (size_t)48 * QKV * 3 * 4 + D * 4; }
inline bool is_eos(int t) { return t == EOS_IM_END || t == EOS_TEXT; }

struct Layer {
    float *attn_norm = nullptr, *post_norm = nullptr;
    QTensor ffn_gate, ffn_up, ffn_down;
    QTensor qkv, z, ab, ssm_out;
    float *conv = nullptr, *ssm_norm = nullptr;
    QTensor q, k, v, o;
    float *q_norm = nullptr, *k_norm = nullptr;
};

struct Seq {
    bool used = false;
    int n = 0;
    int pos = 0;
    int pending = -1;
    std::vector<int> pages;
    int *d_pages = nullptr;
    float *ssm = nullptr, *conv = nullptr;
    float *h_last = nullptr;
    cudaEvent_t ev[12] = {};
    bool restoring = false;
    std::vector<float> top_v;
    std::vector<int> top_i;
    Seen seen;
    std::vector<int> ctx;
    std::mt19937_64 rng;
    Lookup look;
};

struct Engine {
    size_t weight_bytes = 0;
    QTensor embd, out;
    std::string model_path;
    size_t embd_offset = 0;
    float *out_norm = nullptr;
    Layer L[NL], M;
    QTensor eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    float *gdn_A = nullptr, *gdn_dt = nullptr;

    cudaStream_t st, cst;
    cudaEvent_t staged, copied;
    uint8_t *stage_state = nullptr;
    cublasLtHandle_t lt = nullptr;
    void *lt_ws = nullptr;
    size_t lt_ws_size = 32 << 20;
    using LtPlan = lt_tune::Plan;
    std::map<std::tuple<int, int, int, int, int, int>, LtPlan> lt_plans;
    LtPlan &lt_plan(int N, int K, int T, int ldx, int ldd, bool c32, const half *W, const half *x, void *d);
    int umax = 2048;
    static constexpr int DRAFT_VOCAB = 65536;
    static constexpr int GEMM_Q_MAX = 384;
    Arena barena;
    struct {
        int *tokens; float *x, *hfin, *proj, *k, *v, *qn, *kn, *vc, *o, *act, *logits, *part_o, *part_ml;
        half *hh, *deq, *Q16, *mix, *ffn_g, *ffn_u, *gemm16;
        QAct hq, mq, aq;
        float *topv, *tsv, *tops; int *topi, *tsi;
        float *v_raw, *v_kn, *v_vc, *v_ab;
        float *mtp_h, *mtp_x; int *draft;
        const half **img;
        int *rpos;
    } b{};
    uint8_t *kv_pool = nullptr;
    int n_pages = 0;
    std::vector<int> free_pages;
    std::vector<Seq> seqs;
    std::vector<int> page_ref;
    struct Snap { bool used = false; int n = 0, pos = 0; std::vector<int> pages; void *host = nullptr;
                  std::vector<float> top_v; std::vector<int> top_i; };
    std::vector<Snap> snaps;
    uint32_t *d_mask = nullptr;
    float *ssm_pool = nullptr, *conv_pool = nullptr;
    Vision vision;
    struct Spill { uint8_t *host = nullptr; size_t bytes = 0; };
    Spill spill[NL];
    int first_spill = NL;
    size_t stage_bytes = 0;
    uint8_t *stage_slot[2] = {};
    cudaEvent_t stage_copied[2] = {}, stage_used[2] = {};
    Layer stage_L[2][NL];
    Layer spill_L[NL];
    struct Lent { int page = -1, pages = 0; };
    Lent lent[NL];
    int n_lent = 0, kv_segs = 0;
    bool lend_on = true;
    std::vector<int> kv_seg;
    int spill_from() const { return first_spill + n_lent; }
    int lent_pages() const;
    void lend();
    void reclaim(int pages);
    void stage_layer(int l);
    const Layer &layer_for(int l, bool stage);
    void layer_done(int l, bool stage);
    struct Image { half *rows = nullptr; int n = 0; };
    std::vector<Image> images;

    void load(const std::string &path, int max_seqs, size_t reserve_bytes);
    void warmup();
    void alloc_buffers(int max_seqs);
    size_t buffer_need(int max_seqs) const;
    void add_kv_memory(uint8_t *p, size_t bytes);
    int seq_new();
    void seq_reset(Seq &s);
    void settle(Seq &s);
    void seq_free(int id);
    void seq_truncate(int id, int n);
    int snapshot(int id);
    void restore(int id, int snap);
    void snap_free(int snap);
    size_t snap_bytes(int snap) const;
    void snap_export(int snap, uint8_t *dst);
    int snap_import(int64_t from, int64_t off, size_t bytes);
    void prefill(int id, const int *tokens, const int *pos3, int n);
    int image_encode(int slot, const float *patches, int gh, int gw, int frames = 1);
    int generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v = nullptr, int *rows_i = nullptr);
    int draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out);
    int generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out);
    int verify(Seq &s, const Sampling &sp, int k, bool masked, int *out, int nout, float *rows_v, int *rows_i);
    uint32_t *mask_rows();

    void forward(Seq &s, int T, bool verify, int rope0, bool rpos = false, bool img = false);
    void mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *tokens, int rope0, bool rpos = false, bool img = false);
    void mtp_catch_up(Seq &s, int T, int pos0, int rope0, bool rpos = false, bool img = false, int off = 0);
    void lm_head(const QAct &in, int T, int rows);
    void ensure_pages(Seq &s, int n_tokens, int from);
    int page_alloc();
    void page_release(int p);
    KVRef kv(const Seq &s) const { return {kv_pool, s.d_pages}; }
    void linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16 = false);
    void attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, int rope0, const int *rpos, float *x);
    void gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *x);
    struct Part { Seq *s; int T, off, pos0, rope0; const int *rpos; bool verify; };
    void attn_parts(const Layer &w, int kvl, const Part *p, int np, int T, float *x);
    void gdn_parts(const Layer &w, int gl, const Part *p, int np, int T, float *x);
    void forward_parts(const Part *p, int np, int T);
    void generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v = nullptr,
                        int *rows_i = nullptr);
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate);
    void gemm32(const half *W, int N, int K, const half *x, int ldx, int T, float *y, int ldy);
    void ffn_block(const Layer &w, int T, float *x, const float *next, float *nf);
    void residual(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, float *x, const float *next, float *nf);
};
