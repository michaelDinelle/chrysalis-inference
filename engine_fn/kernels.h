// Host-side entry points for the Flash-Next engine's GPU kernels.
#pragma once
#include "common.h"

// quant.hip ---------------------------------------------------------------------------------------
enum { GEMV_STORE = 0, GEMV_ADD = 1, GEMV_GLU = 2 };
QTensor make_qtensor(Arena &a, int type, int rows, int cols);
// relayout.hip: blocks b0 .. b0 + nb (nb < 0: all) of a tensor from raw GGUF data starting at block b0
void relayout(const QTensor &t, const uint8_t *raw_dev, hipStream_t st, int64_t b0 = 0, int64_t nb = -1);
void f32_to_f16(const float *in, half *out, int64_t n, hipStream_t st);
struct GemvSeg { QTensor W; float *y = nullptr; int ldy = 0; };
struct GemvArgs {
    GemvSeg seg[3]; int nseg = 1;
    QAct x{}; int T = 1, mode = GEMV_STORE;
    QTensor up; QAct gq{};                      // GLU: up matrix (seg[0] is the gate) and int8 output
};
void gemv(const GemvArgs &a, hipStream_t st);
void gemv(const QTensor &W, QAct x, int T, float *y, int ldy, int mode, hipStream_t st);
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
void mtp_pack(const float *e, const float *xn, int T, float *out, hipStream_t st);   // [T][HC][2D] = [e ; xn_s]

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
void route_dec(const float *logits, int T, const int *slot_of, int *ids, float *wts, int *hot_slot, MoeShared *sh,
               const float *x, int layer, uint32_t gen, hipStream_t st);
void moe_experts_dec(const QTensor &gate, const QTensor &up, const QTensor &down, const int *hot_slot, int T, QAct x,
                     QAct hq, const float *wts, float *y, hipStream_t st);
void moe_wait(const uint32_t *done, uint32_t gen, hipStream_t st);
void moe_sum(const float *y, const int *hot_slot, const float *shared, const float *sg, const float *cold, int T, float *out, hipStream_t st);

// attn.hip ----------------------------------------------------------------------------------------
constexpr int MAX_SPLITS = 256;          // decode attention key splits (sizes the partial buffers)
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
void idx_prep(const float *kraw, const float *qraw, const float *qnw, const float *knw, int T, int pos0, KVRef kv, int layer,
              half *q16, hipStream_t st);
// block selection for G tokens at pos0..: score scratch [G][nblk], sel [G][SEL_BLOCKS + 1]
void idx_select(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, int *sel, hipStream_t st);
void attention_sel(AttnArgs a, const int *sel, half *oh, QAct oq, hipStream_t st);   // sparse, per-token block lists

// ple.hip -----------------------------------------------------------------------------------------
void ple_embed(const uint8_t *rows, int T, float *of, half *oh, QAct oq, hipStream_t st);
void ple_block(const float *key, const float *value, float *res, const float *nk, const float *nq, const float *nc,
               const float *conv_w, const float *state, float *xconv, int T, hipStream_t st);
void ple_commit(float *state, const float *xconv, int n, hipStream_t st);

// routed experts at prefill (quant.hip / moe.hip) ------------------------------------------------------
struct MoeWork { int slot, m0, n, tier; };
void moe_gemm_gateup(const QTensor &g0, const QTensor &u0, const QTensor &g1, const QTensor &u1, const MoeWork *work, int nwork,
                     const half *X, half *H, hipStream_t st);
void moe_gemm_down(const QTensor &d0, const QTensor &d1, const MoeWork *work, int nwork, const half *H, const float *wts, half *Y, hipStream_t st);
void route_many(const float *logits, const float *wsg, const float *x, int T, int *ids, float *wts, float *sg, hipStream_t st);
void gather_rows(const half *src, const int *idx, int n, half *dst, hipStream_t st);
void moe_combine(const half *Y, const int *pos, const float *shared, const float *sg, int T, float *out, hipStream_t st);
