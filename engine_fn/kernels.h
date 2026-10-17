// Host-side entry points for the Flash-Next engine's GPU kernels.
#pragma once
#include "common.h"
#include <utility>
#include <vector>

// quant.hip ---------------------------------------------------------------------------------------
enum { GEMV_STORE = 0, GEMV_ADD = 1, GEMV_GLU = 2, GEMV_HC = 3 };   // HC: res[t][s][row] += 2 sigmoid(inj[t][s] / HC) * y
QTensor make_qtensor(Arena &a, int type, int rows, int cols);
void qtensor_arrays(const QTensor &t, std::vector<std::pair<const void *, size_t>> &out);   // its arrays (pointer, bytes)
void qtensor_rows(QTensor &t, const QTensor &W, int r0);   // t (rows set by the caller) = rows r0.. of W
struct PrefetchRange { const int4 *p; size_t n16; };   // 16-byte aligned VRAM range, in 16-byte units
// read the ranges so they land in the Infinity Cache (results discarded; sink: a device word never written in practice)
void prefetch(const PrefetchRange *r, int n, unsigned *sink, hipStream_t st);
// relayout.hip: blocks b0 .. b0 + nb (nb < 0: all) of a tensor from raw GGUF data starting at block b0
void relayout(const QTensor &t, const uint8_t *raw_dev, hipStream_t st, int64_t b0 = 0, int64_t nb = -1);
void f32_to_f16(const float *in, half *out, int64_t n, hipStream_t st);
struct GemvSeg { QTensor W; float *y = nullptr; int ldy = 0; };
struct GemvArgs {
    GemvSeg seg[3]; int nseg = 1;
    QAct x{}; int T = 1, mode = GEMV_STORE;
    QTensor up; QAct gq{};                      // GLU: up matrix (seg[0] is the gate) and int8 output
    const float *inj = nullptr; float *res = nullptr;   // HC: the hyper-connection update instead of storing y
};
void gemv(const GemvArgs &a, hipStream_t st);
void gemv(const QTensor &W, QAct x, int T, float *y, int ldy, int mode, hipStream_t st);
// a block's output projection fused with its hyper-connection update (W has D rows)
void gemv_hc(const QTensor &W, QAct x, int T, float *res, const float *inj, hipStream_t st);
// y[t] (+)= W x[t] for 9..384 fp16 inputs, fp32 or fp16 (y_f16) output, weights dequantized in registers
void gemm_q(const QTensor &W, const half *x, int ldx, int T, void *y, int ldy, int mode, bool y_f16, hipStream_t st);
void dequant(const QTensor &W, half *out, hipStream_t st);
// token embeddings: IQ4_XS (SoA) or Q8_0 rows in GGUF block order, e.g. in pinned host memory
void embed(const QTensor &W, const int *tokens, int T, float *out, hipStream_t st);
void quant_act(const float *x, int64_t n, QAct out, hipStream_t st);

// ops.hip -----------------------------------------------------------------------------------------
constexpr int TOPK = 64;   // candidates per row handed to the host sampler
void rmsnorm(const float *x, const float *w, int T, int n, int ldx, int ldo, float *of, half *oh, QAct oq, hipStream_t st);
void norm_rows(const float *x, const float *w, int T, float *of, half *oh, QAct oq, hipStream_t st);   // rows of D
void swiglu_h(half *g, const half *u, int64_t n, hipStream_t st);
void to_half(const float *x, half *y, int64_t n, hipStream_t st);
void store_f16(const half *x, int n, int T, float *y, int ldy, bool accumulate, hipStream_t st);
void topk(const float *logits, int T, int n, int ld, float *scratch_v, int *scratch_i, float *ov, int *oi, hipStream_t st);
void argmax(const float *logits, int T, int n, int ld, int *out, hipStream_t st);
// hyper-connections: res [T][HC][D]
void hc_init(const float *x, float *res, int T, hipStream_t st);
void hc_norm(const float *res, const float *gamma, int T, float *of, half *oh, QAct oq, hipStream_t st);
void hc_lo(float *lo, int T, half *oh, QAct oq, hipStream_t st);
void hc_mix(const float *xn, const float *gate, int T, float *of, half *oh, QAct oq, hipStream_t st);
void hc_combine(float *res, const float *inject, int ld_inject, const float *out, const float *out2, int T, hipStream_t st);
// the whole HC mixer at decode for T <= 4 (norms, W_down + W_inject rows, silu, W_up rows, the gated mean): two kernels
void hc_mix_fused(const float *res, const float *gamma, const QTensor &down, const QTensor &up, const QTensor *inj, int T,
                  float *lo, float *inject, int ld_inject, float *inv, float *mx, half *hh, QAct hq, hipStream_t st);
void mtp_pack(const float *e, const float *xn, int T, float *out, half *out_h, hipStream_t st);   // [T][HC][2D] = [e ; xn_s], fp32 or fp16

// gdn.hip -----------------------------------------------------------------------------------------
struct GdnScanArgs {
    float *S; size_t S_layer;              // states [layers][GV][GD][GD]
    const float *qn, *kn; size_t in_layer; // normalized q/k [T][GK][GD] per layer
    const float *vc; size_t v_layer;       // conv'd v [T][GV][GD] per layer
    const float *ab; size_t ab_layer;      // alpha | beta projections [T][2*GV] per layer
    const float *A, *dtb;                  // [layers][GV]
    int T, write_state;
    float *out;                            // [T][GV][GD] or nullptr
};
void gdn_conv(const float *raw, int T, const float *w, float *state, bool update, float *qn, float *kn, float *vc, hipStream_t st);
void gdn_conv_commit(float *state, const float *raw, int n, int raw_layer_stride, int nlayers, hipStream_t st);
void gdn_scan(const GdnScanArgs &a, int nlayers, hipStream_t st);
void gdn_out(const float *o, const float *z, const float *w, int T, half *oh, QAct oq, hipStream_t st);
struct GdnStepArgs {
    const float *raw, *z;                  // projections [T][QKV] and [T][GV*GD]
    const half *hh, *Wab;                  // normed input [T][D] fp16, alpha|beta weights [2*GV][D]
    const float *conv_w, *A, *dt, *norm_w;
    float *conv_state, *S;
    int T, verify;
    float *kn_out, *vc_out, *ab_out;       // verify: inputs saved for the commit
    QAct out;                              // [T][GV*GD] int8 for ssm_out
};
void gdn_step(const GdnStepArgs &a, hipStream_t st);

// moe.hip / quant.hip: routed experts at decode ----------------------------------------------------
struct MoeShared;
void router_dec(const half *W, const float *wsg, const float *x, int T, float *logits, float *sg, hipStream_t st);
// router_dec + route_dec in one launch (counter: a zeroed device word the kernel resets itself); counts: if set, picks per
// (layer, expert) are added to it ([NL][NE])
void router_route_dec(const half *W, const float *wsg, const float *x, int T, float *logits, float *sg, const int *slot_of,
                      int *ids, float *wts, int *hot_slot, MoeShared *sh, float gfrac, int layer, uint32_t gen, unsigned *counter,
                      hipStream_t st, unsigned *counts = nullptr);
void route_dec(const float *logits, int T, const int *slot_of, int *ids, float *wts, int *hot_slot, MoeShared *sh, float gfrac,
               const float *x, int layer, uint32_t gen, hipStream_t st);
// the GPU's routed picks (hot_slot >= 0) and, if sgate is set, each token's shared expert as pick T*TOPK_E + t (y rows
// weighted by sg[t]); hq / y have T * (TOPK_E + 1) rows
void moe_experts_dec(const QTensor &gate, const QTensor &up, const QTensor &down, const QTensor &sgate, const QTensor &sup,
                     const QTensor &sdown, const int *hot_slot, int T, QAct x, QAct hq, const float *wts, const float *sg, float *y,
                     hipStream_t st);
// the GPU's share of the cold picks (hot_slot <= -2), GGUF blocks read from registered host memory
struct ColdDec {                       // a layer's RAM experts as the GPU sees them (device pointers)
    const uint8_t *gate[2] = {}, *up[2] = {}, *down = nullptr;
    int type[2] = {-1, -1};            // gate / up format of each group (IQ4_XS, Q3_K, Q2_K); down is IQ4_NL
    int n0 = 0;                        // cold slots below n0 are group 0's, the rest group 1's (slot - n0)
};
void moe_cold_dec(const ColdDec &c, const int *hot_slot, int T, QAct x, QAct hq, const float *wts, float *y, hipStream_t st);
// both in two launches, the cold picks (c, may be nullptr) side by side with the hot ones and the shared expert
void moe_dec(const QTensor &gate, const QTensor &up, const QTensor &down, const QTensor &sgate, const QTensor &sup, const QTensor &sdown,
             const ColdDec *c, const int *hot_slot, int T, QAct x, QAct hq, const float *wts, const float *sg, float *y, hipStream_t st);
void moe_wait(const uint32_t *done, uint32_t gen, hipStream_t st);
// moe_wait + moe_sum + hc_combine (res += 2 sigmoid(inject / HC) * out) in one launch; done = nullptr: no CPU part;
// sg = nullptr: the shared rows are already weighted
void moe_finish(const uint32_t *done, uint32_t gen, const float *y, const int *hot_slot, const float *shared, const float *sg,
                const float *cold, const float *inject, float *res, int T, hipStream_t st);
void moe_sum(const float *y, const int *hot_slot, const float *shared, const float *sg, const float *cold, int T, float *out, hipStream_t st);

// attn.hip ----------------------------------------------------------------------------------------
constexpr int MAX_SPLITS = 256;          // decode attention key splits
constexpr int PART_ROWS = 1024;          // split partials: key splits x tokens (decode / verify passes use up to 1024)
struct AttnArgs {
    const half *Q16;                       // [T][NH][HD], normed + roped + pre-scaled
    KVRef kv; int layer;
    int T, pos0;
    int split_len = 0, nsplit = 1, partial = 0;
    const float *qg;                       // projections [T][NH][2][HD]; the gate is the second half
    half *out_h = nullptr;                 // [T][NH*HD] when not partial
    float *part_o = nullptr, *part_ml = nullptr;
};
void attn_prep(const float *qg, const float *k, const float *v, const float *qnw, const float *knw, int T, int pos0,
               KVRef kv, int layer, half *Q16, hipStream_t st);
void attention(AttnArgs a, half *oh, QAct oq, hipStream_t st);           // dense causal (all cells up to each query)
// QSA indexer: raw keys / pooled keys into the cache, query heads normed + rotated (q16 [T][IDX_H][IDX_D])
// indexer: raw keys -> praw (this pass, fp16 [T][IDX_D]), queries normed + rotated -> q16, pooled keys of the blocks this
// pass completes (cells before pos0 from tail = the sequence's incomplete block); keep_tail: the pass is final, so the
// tail takes its new incomplete block (speculative passes leave that to idx_tail_commit for the accepted cells)
void idx_prep(const float *kraw, const float *qraw, const float *qnw, const float *knw, int T, int pos0, KVRef kv, int layer,
              half *tail, half *praw, bool keep_tail, half *q16, hipStream_t st);
void idx_tail_commit(half *tails, const half *praw, size_t praw_layer, int layers, int pos0, int n_end, hipStream_t st);
// block selection for G tokens at pos0..: score scratch [G][nblk], sel [G][SEL_BLOCKS + 1]
void idx_select(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, uint32_t *hist, int *sel,
                hipStream_t st);   // hist: scratch [G][1024]
void idx_select_ref(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, int *sel, hipStream_t st);
void attention_sel(AttnArgs a, const int *sel, half *oh, QAct oq, hipStream_t st);   // sparse, per-token block lists

// ple.hip -----------------------------------------------------------------------------------------
void ple_embed(const uint8_t *rows, int T, float *of, half *oh, QAct oq, hipStream_t st);
void ple_block(const float *key, const float *value, float *res, const float *nk, const float *nq, const float *nc,
               const float *conv_w, const float *state, float *xconv, int T, hipStream_t st);
void ple_commit(float *state, const float *xconv, int n, hipStream_t st);

// routed experts at prefill (quant.hip / moe.hip) ------------------------------------------------------
struct MoeWork { int slot, m0, n, tier; };
constexpr int MOE_ITEM = 256;                     // tokens per work item at most (an expert's tokens split into items)
void moe_gemm_gateup(const QTensor &g0, const QTensor &u0, const QTensor &g1, const QTensor &u1, const MoeWork *work, int nwork,
                     const half *X, half *H, hipStream_t st);
void moe_gemm_down(const QTensor &d0, const QTensor &d1, const MoeWork *work, int nwork, const half *H, const float *wts, half *Y, hipStream_t st);
void route_many(const float *logits, const float *wsg, const float *x, int T, int *ids, float *wts, float *sg, hipStream_t st);
void gather_rows(const half *src, const int *idx, int n, half *dst, hipStream_t st);
void moe_combine(const half *Y, const int *pos, const float *shared, const float *sg, int T, float *out, hipStream_t st);
// the same in batches over a long pass: moe = sg * shared, then per batch of picks [p0, p1) (outputs in Y rows from 0)
// moe[t] += its picks in the batch, for the batch's tokens toks[0..ntok)
void moe_shared(const float *shared, const float *sg, int T, float *moe, hipStream_t st);
void moe_accum(const half *Y, const int *pos, int p0, int p1, const int *toks, int ntok, float *moe, hipStream_t st);
