#pragma once
#include "kernels.h"
#include "sampler.h"
#include "lookup.h"
#include "vision.h"
static_assert(TOPK == SAMPLE_N, "the sampler takes the engine's rows as they are");
#include <random>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>
#include <string>
#include <vector>

constexpr int EOS_IM_END = 248046, EOS_TEXT = 248044;   // <|im_end|>, <|endoftext|>
// Image tokens: IMG_BASE + picture * IMG_ROWS + row names row `row` of a picture's embeddings (kiln_image_encode), so a
// picture's tokens are its own wherever it appears and the prompt cache tells pictures apart by their tokens.
constexpr int IMG_BASE = 1 << 30, IMG_ROWS = 1 << 14;
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

struct Seq {
    bool used = false;
    int n = 0;                       // tokens in the KV cache / recurrent state
    int pos = 0;                     // M-RoPE position of the next text token: n, until a picture (its tokens take fewer)
    int pending = -1;                // sampled but not yet processed token
    std::vector<int> pages;          // KV pages, one per PAGE tokens
    int *d_pages = nullptr;          // device copy of the page table
    float *ssm = nullptr, *conv = nullptr;   // recurrent state [48][GV][GD][GD], conv state [48][QKV][3]
    float *h_last = nullptr;         // final hidden of token n-1 (MTP pairs it with token n)
    std::vector<float> top_v;        // host top-k logits for the next position (valid when pending < 0)
    std::vector<int> top_i;
    Seen seen;                       // tokens of the current reply (presence penalty, DRY)
    std::vector<int> ctx;            // the prompt's tail, for DRY (set per reply when DRY is on)
    std::mt19937_64 rng;             // reseeded for every reply, see kiln_seq_begin
    Lookup look;                     // prompt lookup's view of the conversation (engine/lookup.h)
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

    hipStream_t st, cst;                // compute stream, snapshot copy stream
    hipEvent_t staged, copied;
    uint8_t *stage_state = nullptr;     // device staging copy of one sequence's recurrent state
    hipblasLtHandle_t lt = nullptr;      // created by warmup()
    void *lt_ws = nullptr;
    size_t lt_ws_size = 32 << 20;
    using LtPlan = lt_tune::Plan;
    std::map<std::tuple<int, int, int, int, int, int>, LtPlan> lt_plans;
    LtPlan &lt_plan(int N, int K, int T, int ldx, int ldd, bool c32, const half *W, const half *x, void *d);
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
        const half **img;            // per token of a prefill chunk: its picture's embedding row, or null
        int *rpos;                   // per token of a prefill chunk with pictures: M-RoPE (t, h, w)
    } b{};
    uint8_t *kv_pool = nullptr;
    int n_pages = 0;
    std::vector<int> free_pages;
    std::vector<Seq> seqs;
    std::vector<int> page_ref;
    // top_v / top_i: the next token's top-k logits when the snapshot ends a prompt (empty otherwise), so a restore
    // there samples at once
    struct Snap { bool used = false; int n = 0, pos = 0; std::vector<int> pages; void *host = nullptr;
                  std::vector<float> top_v; std::vector<int> top_i; };
    std::vector<Snap> snaps;
    uint32_t *d_mask = nullptr;      // held answers' token bitmasks, a row per position of a step (mask_rows())
    float *ssm_pool = nullptr, *conv_pool = nullptr;
    Vision vision;                   // opened by kiln_vision_open
    struct Image { half *rows = nullptr; int n = 0; };   // a picture's embeddings [n][D] fp16, pinned host memory
    std::vector<Image> images;

    void load(const std::string &path, int max_seqs, size_t reserve_bytes);
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
    size_t snap_bytes(int snap) const;
    void snap_export(int snap, uint8_t *dst);
    int snap_import(const uint8_t *src, size_t bytes);
    void prefill(int id, const int *tokens, const int *pos3, int n);   // pos3: [n][3] M-RoPE positions, or null (text)
    int image_encode(int slot, const float *patches, int gh, int gw, int frames = 1);
    int generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v = nullptr, int *rows_i = nullptr);
    int draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out);
    int generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out);
    int verify(Seq &s, const Sampling &sp, int k, bool masked, int *out, int nout, float *rows_v, int *rows_i);
    uint32_t *mask_rows();

    // tokens in b.tokens at KV slots s.n..s.n+T-1, rotated at rope0.. (or b.rpos when rpos); img: b.img holds their rows
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
    // One sequence's rows of a decode or verify pass (up to 8 rows in all): several sequences share a pass's projections
    // (generate_batch), each keeping its own attention and recurrent state.
    struct Part { Seq *s; int T, off, pos0, rope0; const int *rpos; bool verify; };
    void attn_parts(const Layer &w, int kvl, const Part *p, int np, int T, float *x);
    void gdn_parts(const Layer &w, int gl, const Part *p, int np, int T, float *x);
    void forward_parts(const Part *p, int np, int T);
    void generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts);
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate);
    void gemm32(const half *W, int N, int K, const half *x, int ldx, int T, float *y, int ldy);
    void ffn_block(const Layer &w, int T, float *x, const float *next, float *nf);
    void residual(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, float *x, const float *next, float *nf);
};
