// Host-side entry points for all GPU kernels.
#pragma once
#include "common.h"

// quant.hip
enum { GEMV_STORE = 0, GEMV_ADD = 1, GEMV_GLU = 2 };
QTensor make_qtensor(Arena &a, int type, int rows, int cols);
void relayout(const QTensor &t, const uint8_t *raw_dev, hipStream_t st);
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
void embed(const QTensor &W, const int *tokens, int T, float *out, hipStream_t st);
void quant_act(const float *x, int64_t n, QAct out, hipStream_t st);

// ops.hip
constexpr int TOPK = 64;   // candidates per row handed to the host sampler
void rmsnorm(const float *x, const float *w, int T, int n, int ldx, int ldo, float *of, half *oh, QAct oq, hipStream_t st);
void norm_rows(const float *x, const float *w, int T, float *of, half *oh, QAct oq, hipStream_t st);
void swiglu_h(half *g, const half *u, int64_t n, hipStream_t st);
void to_half(const float *x, half *y, int64_t n, hipStream_t st);
void store_f16(const half *x, int n, int T, float *y, int ldy, bool accumulate, hipStream_t st);
void topk(const float *logits, int T, int n, int ld, float *scratch_v, int *scratch_i, float *ov, int *oi, hipStream_t st);
void argmax(const float *logits, int T, int n, int ld, int *out, hipStream_t st);

// gdn.hip
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

// attn.hip
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
void attention(AttnArgs a, half *oh, QAct oq, hipStream_t st);
