// The Qwen3.8-Flash-Next engine: weights (dense + hot experts in VRAM, cold experts in system RAM, the n-gram
// table memory-mapped), sequences (paged KV + recurrent and conv states), forward passes, MTP drafting and
// speculative decoding steps. Same structure and C API as the 27B engine (engine/).
#pragma once
#include "kernels.h"
#include "../engine/sampler.h"
static_assert(TOPK == SAMPLE_N, "the sampler takes the engine's rows as they are");
#include "cpu_moe.h"
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

inline bool is_eos(int t) { return t == EOS_IM_END || t == EOS_TEXT; }
// recurrent state of one sequence: GDN states, GDN conv states, PLE conv history, last residual (MTP input)
inline size_t ssm_bytes() { return (size_t)NGDN * GV * GD * GD * 4; }
inline size_t conv_bytes() { return (size_t)NGDN * QKV * 3 * 4; }
inline size_t ple_bytes() { return (size_t)PLE_HIST * HCD * 4; }
inline size_t tail_bytes() { return (size_t)NKVL * IDX_TAIL * 2; }   // raw indexer keys of the incomplete block
inline size_t state_bytes() { return ssm_bytes() + conv_bytes() + ple_bytes() + (size_t)HCD * 4 + tail_bytes(); }

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

struct Seq {
    bool used = false;
    int n = 0;                       // tokens in the KV cache / recurrent state
    int pending = -1;                // sampled but not yet processed token
    std::vector<int> pages;          // KV pages, one per PAGE tokens
    int *d_pages = nullptr;          // device copy of the page table
    float *ssm = nullptr, *conv = nullptr, *ple = nullptr;   // GDN states, GDN conv states, PLE conv history
    float *h_last = nullptr;         // residual [HC][D] after the last layer for token n-1 (MTP pairs it with token n)
    half *idx_tail = nullptr;        // [NKVL][KPOOL][IDX_D] raw indexer keys of the cells after the last complete block
    int hist[2] = {-1, -1};          // the two tokens before position n (n-gram hash), -1 = none
    std::vector<float> top_v;        // host top-k logits for the next position (valid when pending < 0)
    std::vector<int> top_i;
    Seen seen;                       // tokens of the current reply (presence penalty, DRY)
    std::vector<int> ctx;            // the prompt's tail, for DRY (set per reply when DRY is on)
    std::mt19937_64 rng;
};

struct Engine {
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
    std::vector<ColdDec> cold_dev;               // the same as the GPU sees them
    uint8_t *cold_mem = nullptr; size_t cold_bytes = 0;
    MoeShared *sh = nullptr, *d_sh = nullptr;
    CpuMoe cpu;
    float gfrac = 0.3f;                          // share of the decode cold picks the GPU reads over PCIe itself (calibrate())
    double cal_cpu_bw = 50e9, cal_cpu_mac = 150e9, cal_dma_bw = 25e9;   // measured at load: CPU read and VNNI rates, DMA
    double cpu_bias = 1.0;                       // learned correction of the CPU's estimated cost in the prefill split
    int adapt_left = 480;                        // balance-feedback steps left (10 passes' worth): then cpu_bias stays,
                                                 // so a prompt prefills to the same result however often it is read
    hipEvent_t moe_gpu_done;
    int n_workers = 0;                           // CPU workers: one per physical core
    int split = -1;                              // experiments: prefill cold experts by the cost model (-1), all on the GPU (0), all on the CPU (1)
    bool adapt = true;                           // experiments: false freezes cpu_bias (no balance feedback)
    // KILN_FN_PROF: time prefill phases (syncing the stream at each boundary, so phases do not overlap) and print them
    bool prof = getenv("KILN_FN_PROF") != nullptr;
    std::vector<int64_t> route_counts;           // picks per (layer, expert) in prefill passes, when enabled (kiln_route_counts)
    // KILN_FN_DECODE_COUNTS=file: picks per (layer, expert) at decode, counted on the GPU and written to the file (int64
    // [NL][NE]) every 256 steps, for tuning which experts live in VRAM to what the model generates
    unsigned *route_dec = nullptr;
    const char *route_dec_file = getenv("KILN_FN_DECODE_COUNTS");
    int route_dec_steps = 0;
    double prof_t[12] = {}, prof_last = 0;
    void pmark(int phase);
    uint32_t gen = 0;

    hipStream_t st, cst, pst;                    // compute, copies, decode prefetch
    // Expert lookahead (decode): at a layer's start a fourth stream runs the layer's own MoE mixer and router on the
    // residual before its token mixer, and publishes the predicted CPU share of the cold picks in sh_pred (go = the
    // generation the real handoff will carry). The CPU workers load those experts' weights into their caches while the
    // GPU runs the token mixer, a window in which the memory bus would idle. It finds ~70% of the cold picks and changes
    // no result.
    hipStream_t lst = nullptr;
    hipEvent_t la_ready = nullptr, la_done = nullptr;
    MoeShared *sh_pred = nullptr, *d_sh_pred = nullptr;
    struct { float *res, *lo, *inv, *mx, *logits, *sg, *wts; half *hh; QAct hq; int *ids, *hot; unsigned *counter; } la{};
    bool la_on = getenv("KILN_FN_NO_LOOKAHEAD") == nullptr;   // experiments: tune key "lookahead"
    void lookahead(const Layer &w, int layer, int T, const float *res);
    hipEvent_t staged, copied, topk_ready, routed;
    hipEvent_t front_ready, front_copied;        // a sub-chunk's MoE inputs and picks are ready / on the host (prefill)
    PrefetchRange *pf_ranges = nullptr;          // per decoder layer: its dense weights, read into the Infinity Cache while
    std::vector<int> pf_off, pf_n;               // the CPU computes the previous layer's cold experts (decode)
    unsigned *pf_sink = nullptr;
    bool pf_on = true;                           // experiments (kiln_tune "prefetch")
    uint8_t *stage_state = nullptr;
    hipblasLtHandle_t lt = nullptr;
    void *lt_ws = nullptr;
    size_t lt_ws_size = 32 << 20;
    struct LtPlan { hipblasLtMatmulDesc_t desc; hipblasLtMatrixLayout_t a, b, c; hipblasLtMatmulAlgo_t algo; };
    std::map<std::tuple<int, int, int, int, int>, LtPlan> lt_plans;
    LtPlan &lt_plan(int N, int K, int T, int ldx, int ldd);
    int smax = 8192;                 // tokens per pass (a prefill is split into passes): the MoE reads each expert once per pass
    int umax = 1024;                 // tokens per sub-chunk of a pass for everything else (attention, GDN, mixers)
    int max_ctx = 262144;
    static constexpr int DRAFT_VOCAB = 65536;
    static constexpr int GEMM_Q_MAX = 384;
    static constexpr size_t DEQ_ELEMS = (size_t)6144 * 2560;   // fp16 scratch for dequantized weights (row blocks)
    static constexpr int SEL_G = 64;  // indexer selection runs over groups of this many tokens
    static constexpr int MTP_CHUNK = 256;   // the MTP catch-up of a prefill chunk runs in pieces of this size
    Arena warena, barena;            // weights, activation buffers
    struct {
        int *tokens; float *x, *res, *xn, *lo, *gate, *inj_a, *inj_f, *mx, *proj, *zb, *kb, *vb, *qidx, *kidx, *blk;
        half *xn_h, *lo_h, *hh, *Q16, *qidx16, *att_h, *deq, *gemm16;
        QAct xn_q, lo_q, hq, mq;
        float *part_o, *part_ml, *score; int *sel; uint32_t *sel_hist;
        float *qn, *kn, *vc;                     // GDN prefill
        float *v_raw, *v_kn, *v_vc, *v_ab;       // GDN verify inputs per layer (commit)
        float *logits_e, *sg, *wts, *y, *moe; int *ids, *hot;   // MoE decode (y: picks, then the shared expert per token)
        QAct hq_e;
        uint8_t *ple_rows, *d_ple_rows; float *emb, *key, *value, *xconv; half *emb_h; QAct emb_q;
        float *logits, *topv, *tsv; int *topi, *tsi;
        float *mtp_res, *mtp_e, *mtp_in; half *mtp_in_h; QAct mtp_in_q; int *draft;
        float *fin; QAct fin_q; half *fin_h;     // final mixer output (LM head input)
        float *inv;                              // fused decode mixer: per token and stream norm factors
        half *idx_raw;                           // [NKVL][umax][IDX_D] this pass's raw indexer keys, per attention layer
        unsigned *rr_counter;                    // fused router + routing: workgroups done (reset by the kernel)
        // MoE prefill: router logits, picks sorted by expert, per-pick outputs, staging of cold experts
        float *rlog, *sgT, *wtsT, *ent_w; int *idsT, *ent_tok, *pos, *btok; MoeWork *work;
        half *Xs, *Hs, *Ys, *hmoe;               // a batch's gathered inputs, hidden, outputs; the pass's MoE inputs
        int *h_ids; float *h_wts; int *h_tok; float *h_entw; int *h_pos, *h_btok; MoeWork *h_work;   // pinned host
        uint16_t *h_x16, *h_cout;                // pinned host: MoE inputs for the CPU, its per-pick outputs (fp16)
        float *h_topv; int *h_topi, *h_draft;    // pinned host: a step's top-k and drafts
        uint8_t *raw[4], *sgu[4]; QTensor sd_[4];   // cold expert staging ring: GGUF blocks, then gate | up SoA arrays, down
    } b{};
    static constexpr int STAGE_G = 8;            // cold experts staged per group
    static constexpr int NSLOT = 2;              // staging ring slots: copies run up to NSLOT - 1 groups ahead of the GPU
    size_t stage_gu = 0, stage_soa = 0;          // bytes of a staged gate (or up) matrix: GGUF blocks, SoA arrays
    std::vector<int> j_slot, j_first, j_tok;     // the CPU's share of a prefill chunk's cold experts (MoeJob arrays)
    std::vector<float> j_w;
    uint8_t *d_cold = nullptr;                   // device view of cold_mem (registered)
    hipEvent_t raw_free[NSLOT], raw_ready[NSLOT];
    uint8_t *kv_pool = nullptr;
    int n_pages = 0;
    std::vector<int> free_pages;
    std::vector<Seq> seqs;
    std::vector<int> page_ref;
    // top_v / top_i: the next token's top-k logits when the snapshot ends a prompt (empty otherwise), so a restore
    // there samples at once
    struct Snap { bool used = false; int n = 0; int hist[2] = {-1, -1}; std::vector<int> pages; void *host = nullptr;
                  std::vector<float> top_v; std::vector<int> top_i; };
    std::vector<Snap> snaps;
    float *ssm_pool = nullptr;
    // Snapshot host buffers, made ahead of need in one pinned region (reserve_snapshots): made one at a time on demand,
    // each took 0.1-0.4 s in the minutes after a load (direct reclaim and compaction), paid by a request.
    std::vector<void *> snap_spare;
    std::vector<std::pair<void *, size_t>> snap_regions;   // pinned regions the buffers were carved from
    void *snap_pending = nullptr;                // a region being faulted in, of snap_pending_n buffers
    int snap_target = 2, snap_pending_n = 0;
    bool snap_warm = false;
    std::vector<std::thread> snap_fill;          // faulting the region in (started early in warmup())
    void snap_reserve_start();                   // map the missing buffers' region, fault it in on a few threads
    void snap_reserve_finish();                  // wait for them, pin the region, hand the buffers out

    void load(const std::string &path, int max_seqs, size_t reserve_bytes);
    void calibrate();                            // this machine's CPU / DMA rates for the cold expert split
    void warmup();
    void alloc_buffers(int max_seqs);
    int seq_new();
    void seq_reset(Seq &s);
    void seq_free(int id);
    void seq_truncate(int id, int n);
    int snapshot(int id);
    void *snap_buffer();
    void *take_snap_buffer();                   // a spare one, else a new one
    void reserve_snapshots(int n);              // buffers for n snapshots in all (made at warmup(), or now if it ran)
    void restore(int id, int snap);
    void snap_free(int snap);
    void prefill(int id, const int *tokens, int n);
    int generate(int id, const Sampling &sp, int *out, int max_out, float *rows_v = nullptr, int *rows_i = nullptr);
    void shutdown();                            // frees everything

    void forward(Seq &s, const int *tokens, int T, bool verify);       // tokens (host) at positions s.n..s.n+T-1
    void ple_rows(const Seq &s, const int *tokens, int T);              // n-gram rows for these tokens -> b.ple_rows
    void ple_ids(int h0, int h1, const int *tokens, int T, size_t *rows) const;   // their table rows (hash)
    void ple_prefetch(const size_t *rows, size_t n) const;              // start reading their pages (async)
    void mtp_forward(Seq &s, int T, int pos0, const float *h_prev, const int *d_tokens, bool spec);
    void mtp_catch_up(Seq &s, int T, int pos0, bool spec);   // spec: the pass may still be rejected (drafts)
    void lm_head(const QAct &in, int T, int rows);
    void ensure_pages(Seq &s, int n_tokens, int from);
    int page_alloc();
    void page_release(int p);
    KVRef kv(const Seq &s) const { return {kv_pool, s.d_pages}; }
    void linear(const QTensor &W, int T, const QAct &qin, const half *hin, int ldin, void *out, int ldo, int mode, bool out_f16 = false);
    void gemm(const half *W, int N, int K, const half *x, int ldx, int T, void *y, int ldy, bool y_f16, bool accumulate);
    void hc_mix_block(const HcMix &m, int T, float *inject, bool want_f16);   // b.res -> b.mx (+ b.hq or b.hh), inject
    void attn_block(const Layer &w, int kvl, Seq &s, int T, int pos0, float *res, float *inject, bool spec);   // res += output (HC)
    void gdn_block(const Layer &w, int gl, Seq &s, int T, bool verify, float *res, float *inject);
    void moe_block(const Layer &w, int layer, int T, bool cpu_cold, float *res, float *inject);   // res += its output (HC)
    void moe_front(const Layer &w, int n, int c0);   // a sub-chunk's router, shared expert, MoE inputs (passes of T > 8)
    void moe_back(const Layer &w, int layer, int T); // the pass's routed experts -> b.moe
    void ple_block_run(Seq &s, int c0, int n, bool verify, float *res);
};
