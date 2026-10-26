#pragma once
#ifdef __METAL_VERSION__
#define PTR(T) device T *
#else
#include <cstdint>
#define PTR(T) uint64_t
typedef uint64_t ulong;
typedef uint32_t uint;
#endif

enum { GEMV_STORE = 0, GEMV_ADD = 1 };

struct WArgs {
    PTR(const uchar) q; PTR(const uchar) qh; PTR(const char) sc; PTR(const half) d;
};

struct GemvA {
    WArgs w, w2;
    PTR(const float) x; PTR(float) y;
    int rows, cols, ldx, ldy;
    int T, add;
    PTR(const half) x16; PTR(half) y16;
    PTR(const float) nw;
};

struct DeqA { WArgs w; PTR(half) out; int rows, cols; };

struct EmbedA { WArgs w; PTR(const int) tokens; PTR(float) out; int T; };

struct NormA {
    PTR(const float) x; PTR(const float) w; PTR(float) of; PTR(half) oh;
    int n, ldx, ldo;
};

struct SwigluA { PTR(half) g; PTR(const half) u; int n; };

struct CopyA { PTR(const uchar) src; PTR(uchar) dst; ulong n; };
struct FillA { PTR(uchar) dst; ulong n; };

struct TopkA { PTR(const float) logits; PTR(float) cv; PTR(int) ci; PTR(float) ov; PTR(int) oi; int n, ld, ncand; };
struct ArgmaxA { PTR(const float) logits; PTR(int) out; int n, ld; };
struct MaskA { PTR(float) logits; PTR(const uint) mask; int n; };
struct StatsA { PTR(const float) logits; PTR(float) out; int n, ld; };

struct PrepA {
    PTR(const float) qg; PTR(const float) k; PTR(const float) v;
    PTR(const float) qnw; PTR(const float) knw;
    PTR(const int) rpos;
    PTR(uchar) pool; PTR(const long) pages;
    PTR(half) Qg;
    int T, pos0, rope0, layer, qrows;
    int job0;
};
struct AttnA {
    PTR(const half) Qg; PTR(const uchar) pool; PTR(const long) pages;
    PTR(float) part_o; PTR(float) part_ml;
    int T, pos0, layer, qrows, split_len, nsplit;
};
struct UnpackA {
    PTR(const uchar) pool; PTR(const long) pages; PTR(half) kc; PTR(half) vt;
    int layer, head, L;
};
struct SmaxA {
    PTR(const float) S; PTR(half) P; PTR(float) O; PTR(float) ml;
    int M, n, k0, pos0, r0;
};
struct CombineA {
    PTR(const float) part_o; PTR(const float) part_ml; PTR(const float) qg;
    PTR(float) of; PTR(half) oh;
    int T, qrows, nsplit;
};

struct ConvA {
    PTR(const float) raw; PTR(const float) w; PTR(float) state; PTR(float) qn; PTR(float) kn; PTR(float) vc;
    int T, update;
};
struct ConvStateA { PTR(const float) raw; PTR(float) state; int T; };
struct ConvCommitA { PTR(float) state; PTR(const float) raw; int n, raw_layer; };
struct ScanA {
    PTR(float) S; ulong S_layer;
    PTR(const float) qn; PTR(const float) kn; ulong in_layer;
    PTR(const float) vc; ulong v_layer;
    PTR(const float) ab; ulong ab_layer;
    PTR(const float) A; PTR(const float) dtb;
    PTR(float) out;
    int T, write_state;
};
struct GdnOutA { PTR(const float) o; PTR(const float) z; PTR(const float) w; PTR(float) of; PTR(half) oh; };
struct StepA {
    PTR(const float) raw; PTR(const float) z; PTR(const float) x;
    PTR(const bfloat) Wab;
    PTR(const float) conv_w; PTR(const float) A; PTR(const float) dt; PTR(const float) norm_w;
    PTR(float) conv_state; PTR(float) S;
    PTR(float) kn_out; PTR(float) vc_out; PTR(float) ab_out;
    PTR(float) out; PTR(half) out16;
    PTR(const float) nw;
    PTR(const float) ab_in;
    int T, verify;
};
struct AbA { PTR(const bfloat) W; PTR(const half) x; PTR(float) out; int T; };

struct GemmA {
    WArgs w;
    PTR(const half) x; PTR(uchar) y;
    int T, rows, cols, ldx, ldy, add, y_f16;
};
