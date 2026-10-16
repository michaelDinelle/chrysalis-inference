// The Qwen3.8-Flash-Next engine: weights (dense + hot experts in VRAM, cold experts in system RAM, the n-gram
// table memory-mapped), sequences (paged KV + recurrent and conv states), forward passes, MTP drafting and
// speculative decoding steps. Same structure and C API as the 27B engine (engine/).
#pragma once
#include "kernels.h"
#include "cpu_moe.h"
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

inline bool is_eos(int t) { return t == EOS_IM_END || t == EOS_TEXT; }
// recurrent state of one sequence: GDN states, GDN conv states, PLE conv history, last residual (MTP input)
inline size_t ssm_bytes() { return (size_t)NGDN * GV * GD * GD * 4; }
inline size_t conv_bytes() { return (size_t)NGDN * QKV * 3 * 4; }
inline size_t ple_bytes() { return (size_t)PLE_HIST * HCD * 4; }
inline size_t state_bytes() { return ssm_bytes() + conv_bytes() + ple_bytes() + (size_t)HCD * 4; }

struct HcMix { float *norm = nullptr; QTensor down, up, inject; };   // inject unused for the head mixers

struct Layer {
    HcMix hca, hcf;                              // before the token mixer, before the MoE
    QTensor qkv, z, ssm_out;                     // gated delta net
    half *Wab = nullptr;                         // alpha | beta rows [2*GV][D] fp16
    float *conv = nullptr, *ssm_norm = nullptr;
    QTensor q, k, v, o, idx_q, idx_k;            // QSA attention (q rows: query|gate per head)
    float *q_norm = nullptr, *k_norm = nullptr, *idx_qn = nullptr, *idx_kn = nullptr;
    QTensor ple_key, ple_value;                  // PLE (layer 1)
    float *ple_nk = nullptr, *ple_nq = nullptr, *ple_nc = nullptr, *ple_conv = nullptr;
    half *router = nullptr;                      // [NE][D] fp16
    float *sg_w = nullptr;                       // shared expert gate [D]
    QTensor sh_gate, sh_up, sh_down;             // shared expert
    QTensor hot_gate, hot_up, hot_down;          // experts in VRAM: [n_hot*FE][D], [n_hot*D][FE]
    int *slot_of = nullptr;                      // device [NE]: hot slot >= 0, cold slot -1 - s
    std::vector<int> slot_h;                     // the same, host copy (prefill sorts its picks on the host)
    int n_hot = 0, n_cold = 0;
};

struct Sampling {                                // mirrors KilnSampling in api.cpp
    float temperature = 1.0f, top_p = 0.95f, min_p = 0.0f, presence_penalty = 0.0f;
    int top_k = 20, n_draft = 3;
};

struct Seq {
    bool used = false;
    int n = 0;                       // tokens in the KV cache / recurrent state
    int pending = -1;                // sampled but not yet processed token
    std::vector<int> pages;          // KV pages, one per PAGE tokens
    int *d_pages = nullptr;          // device copy of the page table
    float *ssm = nullptr, *conv = nullptr, *ple = nullptr;   // GDN states, GDN conv states, PLE conv history
    float *h_last = nullptr;         // residual [HC][D] after the last layer for token n-1 (MTP pairs it with token n)
    int hist[2] = {-1, -1};          // the two tokens before position n (n-gram hash), -1 = none
    std::vector<float> top_v;        // host top-k logits for the next position (valid when pending < 0)
    std::vector<int> top_i;
    std::vector<int> seen;           // tokens of the current reply (presence penalty)
    std::mt19937_64 rng;
};

struct Engine {
    // weights
    QTensor embd;                    // Q8_0 rows in pinned host memory
    void *embd_host = nullptr;
    QTensor out;                     // LM head (Q8_0, VRAM)
    HcMix hc_head;                   // final mixer (the model's output norm)
    Layer L[NL], M;                  // M = the MTP block (QSA attention, its own 512 experts all in VRAM)
    QTensor eh_proj;
    float *enorm = nullptr, *hnorm = nullptr;
    HcMix mtp_head;
    float *gdn_A = nullptr, *gdn_dt = nullptr;   // [NGDN][GV], contiguous for the all-layer commit
    // n-gram table (memory-mapped source file) and its hash
    const uint8_t *ngram = nullptr; size_t ngram_map_bytes = 0; uint8_t *ngram_map = nullptr;
    uint64_t ple_mult[3] = {};
    uint32_t ple_off[PLE_HEADS] = {}, ple_voc[PLE_HEADS] = {};
    // cold experts in system RAM and the CPU that computes them
    std::vector<ColdLayer> cold;
    uint8_t *cold_mem = nullptr; size_t cold_bytes = 0;
    MoeShared *sh = nullptr, *d_sh = nullptr;
    CpuMoe cpu;
    uint32_t gen = 0;

    // runtime
    hipStream_t st, cst;
    hipEvent_t staged, copied;
    uint8_t *stage_state = nullptr;
    hipblasLtHandle_t lt = nullptr;
    void *lt_ws = nullptr;
    size_t lt_ws_size = 32 << 20;
    struct LtPlan { hipblasLtMatmulDesc_t desc; hipblasLtMatrixLayout_t a, b, c; hipblasLtMatmulAlgo_t algo; };
    std::map<std::tuple<int, int, int, int, int>, LtPlan> lt_plans;
    LtPlan &lt_plan(int N, int K, int T, int ldx, int ldd);
    int umax = 512;                  // prefill chunk size
    int max_ctx = 262144;
    static constexpr int DRAFT_VOCAB = 65536;
    static constexpr int GEMM_Q_MAX = 384;
    static constexpr int SEL_G = 64;  // indexer selection runs over groups of this many tokens
    Arena warena, barena;            // weights, activation buffers
    struct {
        int *tokens; float *x, *res, *xn, *lo, *gate, *inj_a, *inj_f, *mx, *proj, *zb, *kb, *vb, *qidx, *kidx, *blk;
        half *xn_h, *lo_h, *hh, *Q16, *qidx16, *att_h, *deq, *gemm16;
        QAct xn_q, lo_q, hq, mq;
        float *part_o, *part_ml, *score; int *sel;
        float *qn, *kn, *vc;                     // GDN prefill
        float *v_raw, *v_kn, *v_vc, *v_ab;       // GDN verify inputs per layer (commit)
        float *logits_e, *sg, *wts, *y, *shared, *moe; int *ids, *hot;   // MoE decode
        QAct hq_e, hs_q;
        uint8_t *ple_rows, *d_ple_rows; float *emb, *key, *value, *xconv; half *emb_h; QAct emb_q;
        float *logits, *topv, *tsv; int *topi, *tsi;
        float *mtp_res, *mtp_e, *mtp_in; half *mtp_in_h; QAct mtp_in_q; int *draft;
        float *fin; QAct fin_q; half *fin_h;     // final mixer output (LM head input)
        // MoE prefill: router logits, picks sorted by expert, per-pick outputs, staging of cold experts
        float *rlog, *sgT, *wtsT, *ent_w; int *idsT, *ent_tok, *pos; MoeWork *work;
        half *Xs, *Hs, *Ys;
        int *h_ids; float *h_wts; int *h_tok; float *h_entw; int *h_pos; MoeWork *h_work;   // pinned host
        float *h_x; uint16_t *h_cout;            // pinned host: MoE inputs for the CPU, its per-pick outputs (fp16)
        float *h_topv; int *h_topi, *h_draft;    // pinned host: a step's top-k and drafts
        uint8_t *raw[2]; QTensor sg_[2], su_[2], sd_[2];
    } b{};
    static constexpr int STAGE_G = 32;           // cold experts staged per group
    std::vector<int> j_slot, j_first, j_tok;     // the CPU's share of a prefill chunk's cold experts (MoeJob arrays)
    std::vector<float> j_w;
    uint8_t *d_cold = nullptr;                   // device view of cold_mem (registered)
    hipEvent_t raw_free[2], raw_ready[2];
    uint8_t *kv_pool = nullptr;
    int n_pages = 0;
    std::vector<int> free_pages;
    std::vector<Seq> seqs;
    std::vector<int> page_ref;
    struct Snap { bool used = false; int n = 0; int hist[2] = {-1, -1}; std::vector<int> pages; void *host = nullptr; };
    std::vector<Snap> snaps;
    float *ssm_pool = nullptr;

    void load(const std::string &path, int max_seqs, size_t reserve_bytes);   // load.cpp
    void warmup();
    void alloc_buffers(int max_seqs);
    int seq_new();
    void seq_reset(Seq &s);
    void seq_free(int id);
    void seq_truncate(int id, int n);
    int snapshot(int id);
    void restore(int id, int snap);
    void snap_free(int snap);
    void prefill(int id, const int *tokens, int n);
    int generate(int id, const Sampling &sp, int *out, int max_out);
    void shutdown(bool keep_host);              // frees VRAM; keep_host keeps the cold experts in RAM

    // internals
    void forward(Seq &s, const int *tokens, int T, bool verify);       // tokens (host) at positions s.n..s.n+T-1
    void ple_rows(const Seq &s, const int *tokens, int T);              // n-gram rows for these tokens -> b.ple_rows
    void ple_ids(int h0, int h1, const int *tokens, int T, size_t *rows) const;   // their table rows (hash)
    void ple_prefetch(const size_t *rows, size_t n) const;              // start reading their pages (async)
    void mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *d_tokens);
    void mtp_catch_up(Seq &s, int T, int pos0);
    void lm_head(const QAct &in, int T, int rows);
    void ensure_pages(Seq &s, int n_tokens, int from);
    int page_alloc();
    void page_release(int p);
    KVRef kv(const Seq &s) const { return {kv_pool, s.d_pages}; }
    void linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16 = false);
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate);
    void hc_mix_block(const HcMix &m, int T, float *inject, bool want_f16);   // b.res -> b.mx (+ b.hq or b.hh), inject
    void attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *out);
    void gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *out);
    void moe_block(const Layer &w, int layer, int T, bool cpu_cold, float *out);
    void moe_prefill(const Layer &w, int layer, int T, float *out);
    void ple_block_run(Seq &s, int T, bool verify);
};
