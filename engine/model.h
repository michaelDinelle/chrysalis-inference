// The Qwen3.8-27B engine: weights, sequences (paged KV + recurrent state), forward passes,
// MTP drafting and speculative decoding steps.
#pragma once
#include "kernels.h"
#include <random>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>
#include <string>
#include <vector>

constexpr int EOS_IM_END = 248046, EOS_TEXT = 248044;   // <|im_end|>, <|endoftext|>
inline size_t state_bytes() { return (size_t)48 * GV * GD * GD * 4 + (size_t)48 * QKV * 3 * 4 + D * 4; }   // one sequence
inline bool is_eos(int t) { return t == EOS_IM_END || t == EOS_TEXT; }

struct Layer {
    float *attn_norm = nullptr, *post_norm = nullptr;
    QTensor ffn_gate, ffn_up, ffn_down;
    QTensor qkv, z, ab, ssm_out;                 // gated delta net (ab: alpha|beta rows, fp16)
    float *conv = nullptr, *ssm_norm = nullptr;  // conv [QKV][4], norm [GD]
    QTensor q, k, v, o;                          // full attention (q rows: query|gate per head)
    float *q_norm = nullptr, *k_norm = nullptr;
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
    float *ssm = nullptr, *conv = nullptr;   // recurrent state [48][GV][GD][GD], conv state [48][QKV][3]
    float *h_last = nullptr;         // final hidden of token n-1 (MTP pairs it with token n)
    std::vector<float> top_v;        // host top-k logits for the next position (valid when pending < 0)
    std::vector<int> top_i;
    std::vector<int> seen;           // tokens of the current reply (presence penalty)
    std::mt19937_64 rng;             // reseeded for every reply, see kiln_seq_begin
};

struct Engine {
    // weights (allocated in slabs, see load.cpp)
    size_t weight_bytes = 0;
    QTensor embd, out;               // embd lives in pinned host memory, read by the GPU over PCIe (filled by warmup())
    std::string model_path;
    size_t embd_offset = 0;          // of the token embeddings in the model file
    float *out_norm = nullptr;
    Layer L[NL], M;                  // M = the MTP block
    QTensor eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    float *gdn_A = nullptr, *gdn_dt = nullptr;   // [48][GV], contiguous for the all-layer commit

    // runtime
    hipStream_t st, cst;                // compute stream, snapshot copy stream
    hipEvent_t staged, copied;
    uint8_t *stage_state = nullptr;     // device staging copy of one sequence's recurrent state
    hipblasLtHandle_t lt = nullptr;      // created by warmup()
    void *lt_ws = nullptr;
    size_t lt_ws_size = 32 << 20;
    struct LtPlan { hipblasLtMatmulDesc_t desc; hipblasLtMatrixLayout_t a, b, c; hipblasLtMatmulAlgo_t algo; };
    std::map<std::tuple<int, int, int, int, int>, LtPlan> lt_plans;
    LtPlan &lt_plan(int N, int K, int T, int ldx, int ldd);
    int umax = 2048;                 // prefill chunk size
    static constexpr int DRAFT_VOCAB = 65536;   // MTP drafts only consider the first (most frequent) token ids
    static constexpr int GEMM_Q_MAX = 384;      // up to this many tokens, projections run on quantized weights (gemm_q)
    Arena barena;                    // activation buffers
    struct {
        int *tokens; float *x, *hfin, *proj, *k, *v, *qn, *kn, *vc, *o, *act, *logits, *part_o, *part_ml;
        half *hh, *deq, *Q16, *mix, *ffn_g, *ffn_u, *gemm16;
        QAct hq, mq, aq;
        float *topv, *tsv; int *topi, *tsi;
        float *v_raw, *v_kn, *v_vc, *v_ab;       // verify-batch GDN inputs per layer (for commit)
        float *mtp_h, *mtp_x; int *draft;
    } b{};
    uint8_t *kv_pool = nullptr;
    int n_pages = 0;
    std::vector<int> free_pages;
    std::vector<Seq> seqs;
    std::vector<int> page_ref;
    struct Snap { bool used = false; int n = 0; std::vector<int> pages; void *host = nullptr; };
    std::vector<Snap> snaps;
    float *ssm_pool = nullptr, *conv_pool = nullptr;

    void load(const std::string &path, int max_seqs, size_t reserve_bytes);   // load.cpp
    void warmup();
    void alloc_buffers(int max_seqs);
    void add_kv_memory(uint8_t *p, size_t bytes);
    int seq_new();
    void seq_reset(Seq &s);
    void seq_free(int id);
    void seq_truncate(int id, int n);
    int snapshot(int id);
    void restore(int id, int snap);
    void snap_free(int snap);
    void prefill(int id, const int *tokens, int n);
    int generate(int id, const Sampling &sp, int *out, int max_out);

    // internals
    void forward(Seq &s, int T, bool verify);          // tokens in b.tokens, positions s.n..s.n+T-1
    void mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *tokens);
    void mtp_catch_up(Seq &s, int T, int pos0);
    void lm_head(const QAct &in, int T, int rows);
    void ensure_pages(Seq &s, int n_tokens, int from);
    int page_alloc();
    void page_release(int p);
    KVRef kv(const Seq &s) const { return {kv_pool, s.d_pages}; }
    void linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16 = false);
    void attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *x);
    void gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *x);
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate);
    void ffn_block(const Layer &w, int T, float *x, const float *next, float *nf);
    void residual(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, float *x, const float *next, float *nf);
};
