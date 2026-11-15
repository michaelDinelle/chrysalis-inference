#pragma once
#include "common.h"
#include <utility>
#include <vector>

enum { GEMV_STORE = 0, GEMV_ADD = 1, GEMV_HC = 3 };
QTensor make_qtensor(Arena &a, int type, int rows, int cols);
void qtensor_rows(QTensor &t, const QTensor &W, int r0);
void relayout(const QTensor &t, const uint8_t *raw_dev, hipStream_t st, int64_t b0 = 0, int64_t nb = -1);
struct Place { const uint8_t *src = nullptr; int slot = 0, type = 0; };
struct Patch { int slot_of = -1, fmt = -1, value = 0; };
constexpr int PLACE_MAX = 32;
struct PlaceArgs {
    QTensor gate, up, down;
    int *slot_of = nullptr; uint8_t *fmt = nullptr;
    Place p[PLACE_MAX]; int n = 0;
    Patch patch[3 * PLACE_MAX]; int npatch = 0;
};
void place_experts(const PlaceArgs &a, hipStream_t st);
struct GemvSeg { QTensor W; float *y = nullptr; int ldy = 0; };
struct GemvArgs {
    GemvSeg seg[3]; int nseg = 1;
    QAct x{}; int T = 1, mode = GEMV_STORE;
    const float *inj = nullptr; float *res = nullptr;
};
void gemv(const GemvArgs &a, hipStream_t st);
void gemv(const QTensor &W, QAct x, int T, float *y, int ldy, int mode, hipStream_t st);
void gemv_hc(const QTensor &W, QAct x, int T, float *res, const float *inj, hipStream_t st);
void gemm_q(const QTensor &W, const half *x, int ldx, int T, void *y, int ldy, int mode, bool y_f16, hipStream_t st);
void dequant(const QTensor &W, half *out, hipStream_t st);
void quant_act(const float *x, int64_t n, QAct out, hipStream_t st);

constexpr int TOPK = 64;
void rmsnorm(const float *x, const float *w, int T, int n, int ldx, int ldo, float *of, half *oh, QAct oq, hipStream_t st);
void swiglu_h(half *g, const half *u, int64_t n, hipStream_t st);
void store_f16(const half *x, int n, int T, float *y, int ldy, bool accumulate, hipStream_t st);
void topk(const float *logits, int T, int n, int ld, float *scratch_v, int *scratch_i, float *ov, int *oi, hipStream_t st);
constexpr int LOGIT_STATS = 6;
void logit_stats(const float *logits, int T, int n, int ld, float *out, hipStream_t st);
void mask_logits(float *logits, const uint32_t *mask, int n, int rows, hipStream_t st);
void argmax(const float *logits, int T, int n, int ld, int *out, hipStream_t st);
void hc_init(const float *x, float *res, int T, hipStream_t st);
void hc_norm(const float *res, const float *gamma, int T, float *of, half *oh, QAct oq, hipStream_t st);
void hc_lo(float *lo, int T, half *oh, QAct oq, hipStream_t st);
void hc_mix(const float *xn, const float *gate, int T, float *of, half *oh, QAct oq, hipStream_t st);
void hc_combine(float *res, const float *inject, int ld_inject, const float *out, const float *out2, int T, hipStream_t st);
void wait_flag(const unsigned *flag, unsigned gen, unsigned *lost, hipStream_t st);
void hc_mix_fused(const float *res, const float *gamma, const QTensor &down, const QTensor &up, const QTensor *inj, int T,
                  float *lo, float *inject, int ld_inject, float *inv, float *mx, half *hh, QAct hq, QAct xq, int slot, hipStream_t st,
                  unsigned *flags = nullptr, unsigned gen = 0, int role = 0);
void mtp_pack(const float *e, const float *xn, int T, float *out, half *out_h, hipStream_t st);

struct GdnScanArgs {
    float *S; size_t S_layer;
    const float *qn, *kn; size_t in_layer;
    const float *vc; size_t v_layer;
    const float *ab; size_t ab_layer;
    const float *A, *dtb;
    int T, write_state;
    float *out;
};
void gdn_conv(const float *raw, int T, const float *w, float *state, bool update, float *qn, float *kn, float *vc, hipStream_t st);
void gdn_conv_commit(float *state, const float *raw, int n, int raw_layer_stride, int nlayers, hipStream_t st);
void gdn_scan(const GdnScanArgs &a, int nlayers, hipStream_t st);
void gdn_out(const float *o, const float *z, const float *w, int T, half *oh, QAct oq, hipStream_t st);
struct GdnStepArgs {
    const float *raw, *z;
    const half *hh, *Wab;
    const float *conv_w, *A, *dt, *norm_w;
    float *conv_state, *S;
    int T, verify;
    float *kn_out, *vc_out, *ab_out;
    QAct out;
};
void gdn_step(const GdnStepArgs &a, hipStream_t st);

struct MoeShared;
struct MoeEmu;
constexpr int EMU_PICK = -(1 << 30);
void router_dec(const half *W, const float *wsg, const float *x, int T, float *logits, float *sg, hipStream_t st);
void router_route_dec(const half *W, const float *wsg, const float *x, int T, float *logits, float *sg, const int *slot_of,
                      int n_ram, int *ids, float *wts, int *hot_slot, MoeShared *sh, float gfrac, int layer, uint32_t gen,
                      unsigned *counter, hipStream_t st, unsigned *counts = nullptr, unsigned starts = 1, float xfrac = 1.0f, bool hand_x = true);
void route_dec(const float *logits, int T, const int *slot_of, int n_ram, int *ids, float *wts, int *hot_slot, MoeShared *sh,
               float gfrac, const float *x, int layer, uint32_t gen, hipStream_t st);
void moe_experts_dec(const QTensor &gate, const QTensor &up, const QTensor &down, const uint8_t *fmt, const QTensor &sgate,
                     const QTensor &sup, const QTensor &sdown, const int *hot_slot, int T, QAct x, QAct hq, const float *wts,
                     const float *sg, float *y, hipStream_t st);
struct ColdDec {
    const uint8_t *gate[2] = {}, *up[2] = {}, *down = nullptr;
    int type[2] = {-1, -1};
    int n0 = 0;
};
void moe_cold_dec(const ColdDec &c, const int *hot_slot, int T, QAct x, QAct hq, const float *wts, float *y, hipStream_t st);
void moe_dec(const QTensor &gate, const QTensor &up, const QTensor &down, const uint8_t *fmt, const QTensor &sgate, const QTensor &sup,
             const QTensor &sdown, const ColdDec *c, const int *hot_slot, int T, QAct x, QAct hq, const float *wts, const float *sg,
             float *y, hipStream_t st);
void moe_finish(const uint32_t *done, uint32_t gen, const float *y, const int *hot_slot, const float *shared, const float *sg,
                const float *cold, const float *inject, float *res, int T, hipStream_t st, const float *ye = nullptr);
void emu_glu(const int *hot_slot, MoeEmu *me, uint32_t gen, int T, hipStream_t st);

constexpr int MAX_SPLITS = 256;
constexpr int PART_ROWS = 1024;
struct AttnArgs {
    const half *Q16;
    KVRef kv; int layer;
    int T, pos0;
    int split_len = 0, nsplit = 1, partial = 0;
    const float *qg;
    half *out_h = nullptr;
    float *part_o = nullptr, *part_ml = nullptr;
};
void attn_prep(const float *qg, const float *k, const float *v, const float *qnw, const float *knw, int T, int pos0,
               int rope0, const int *rpos, KVRef kv, int layer, half *Q16, hipStream_t st);
void attention(AttnArgs a, half *oh, QAct oq, hipStream_t st);
void idx_prep(const float *kraw, const float *qraw, const float *qnw, const float *knw, int T, int pos0, int rope0, const int *rpos,
              Pos3 tail_pos, KVRef kv, int layer, half *tail, half *praw, bool keep_tail, half *q16, hipStream_t st);
void idx_tail_commit(half *tails, const half *praw, size_t praw_layer, int layers, int pos0, int n_end, hipStream_t st);
void idx_select(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, uint32_t *hist, int *sel,
                hipStream_t st);
void idx_select_ref(const half *q16, int pos0, int G, int nblk, KVRef kv, int layer, float *score, int *sel, hipStream_t st);
void attention_sel(AttnArgs a, const int *sel, half *oh, QAct oq, hipStream_t st);

void ple_embed(const uint8_t *rows, int T, float *of, half *oh, QAct oq, hipStream_t st);
void embed(const QTensor &W, const int *tokens, const half *const *rows, int T, float *out, hipStream_t st);
void ple_block(const float *key, const float *value, float *res, const float *nk, const float *nq, const float *nc,
               const float *conv_w, const float *state, float *xconv, int T, hipStream_t st);
void ple_commit(float *state, const float *xconv, int n, hipStream_t st);

struct MoeWork { int slot, m0, n, tier; };
constexpr int MOE_ITEM = 256;
void moe_gemm_gateup(const QTensor &g0, const QTensor &u0, const QTensor &g1, const QTensor &u1, const MoeWork *work, int nwork,
                     const half *X, half *H, hipStream_t st, int slot_type = -1);
void moe_gemm_down(const QTensor &d0, const QTensor &d1, const MoeWork *work, int nwork, const half *H, const float *wts, half *Y, hipStream_t st);
void route_many(const float *logits, const float *wsg, const float *x, int T, int *ids, float *wts, float *sg, hipStream_t st);
void gather_rows(const half *src, const int *idx, int n, half *dst, hipStream_t st);
void moe_shared(const float *shared, const float *sg, int T, float *moe, hipStream_t st);
void moe_accum(const half *Y, const int *pos, int p0, int p1, const int *toks, int ntok, float *moe, hipStream_t st);
