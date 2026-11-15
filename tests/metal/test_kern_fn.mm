#include "engine/metal/flash-next/prefill.h"
#include <cmath>
#include <functional>
#include <random>

static const int8_t IQ4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static float h2f(const uint8_t *p) { __fp16 h; memcpy(&h, p, 2); return (float)h; }

enum { F16 = 1, Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q6_K = 14, IQ4_NL = 20, IQ4_XS = 23 };
static int block_vals(int t) { return t == F16 ? 1 : t == Q8_0 || t == IQ4_NL ? 32 : 256; }
static int block_bytes(int t) { return (int)ggml_row_bytes(t, block_vals(t)); }
static const char *name_of(int t) {
    switch (t) { case F16: return "f16"; case Q8_0: return "q8_0"; case Q2_K: return "q2_k"; case Q3_K: return "q3_k";
                 case Q6_K: return "q6_k"; case IQ4_NL: return "iq4_nl"; case IQ4_XS: return "iq4_xs"; }
    return "?";
}

static void ref_dequant(int type, const uint8_t *b, float *y) {
    switch (type) {
    case F16: y[0] = h2f(b); return;
    case Q8_0: { const float d = h2f(b); for (int j = 0; j < 32; j++) y[j] = d * (int8_t)b[2 + j]; return; }
    case IQ4_NL: {
        const float d = h2f(b);
        for (int j = 0; j < 16; j++) { y[j] = d * IQ4[b[2 + j] & 0xf]; y[j + 16] = d * IQ4[b[2 + j] >> 4]; }
        return;
    }
    case IQ4_XS: {
        const float d = h2f(b);
        const uint16_t sh = b[2] | (b[3] << 8);
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; j++) {
                y[32 * ib + j] = dl * IQ4[b[8 + 16 * ib + j] & 0xf];
                y[32 * ib + j + 16] = dl * IQ4[b[8 + 16 * ib + j] >> 4];
            }
        }
        return;
    }
    case Q6_K: {
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        const float d = h2f(b + 208);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[n + l] = d * sc[is] * q1;
                y[n + l + 32] = d * sc[is + 2] * q2;
                y[n + l + 64] = d * sc[is + 4] * q3;
                y[n + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
        return;
    }
    case Q3_K: {
        const uint8_t *hm = b, *q = b + 32;
        const float d = h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, b + 96, 12);
        const uint32_t k1 = 0x03030303, k2 = 0x0f0f0f0f, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
        aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
        aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
        aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
        const int8_t *scales = (const int8_t *)aux;
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        return;
    }
    case Q2_K: {
        const float d = h2f(b + 80), mn = h2f(b + 82);
        const uint8_t *q = b + 16;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = b[is++];
                float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = b[is++];
                dl = d * (sc & 0xF); ml = mn * (sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
        return;
    }
    }
    FAIL("ref_dequant: type %d", type);
}

static void random_blocks(int type, uint8_t *raw, size_t nb, std::mt19937 &rng) {
    const int bb = block_bytes(type);
    for (size_t i = 0; i < nb * bb; i++) raw[i] = (uint8_t)rng();
    auto put = [&](uint8_t *p, float v) { const __fp16 h = (__fp16)v; memcpy(p, &h, 2); };
    for (size_t b = 0; b < nb; b++) {
        uint8_t *p = raw + b * bb;
        const float d = ((rng() % 1000) + 1) * 2e-6f;
        switch (type) {
            case F16: put(p, (int)(rng() % 2001 - 1000) * 1e-3f); break;
            case Q8_0: put(p, d * 4); break;
            case Q6_K: put(p + 208, d); break;
            case Q3_K: put(p + 108, d * 4); break;
            case Q2_K: put(p + 80, d * 8); put(p + 82, d * 4); break;
            default: put(p, d);
        }
    }
}

struct Mat { int type, rows, cols; size_t rb; std::vector<uint8_t> raw; std::vector<float> w; };
static Mat make_mat(int type, int rows, int cols, std::mt19937 &rng) {
    Mat m{type, rows, cols, ggml_row_bytes(type, cols)};
    const size_t nb = (size_t)rows * cols / block_vals(type);
    m.raw.resize(nb * block_bytes(type));
    random_blocks(type, m.raw.data(), nb, rng);
    m.w.resize((size_t)rows * cols);
    for (size_t b = 0; b < nb; b++) ref_dequant(type, m.raw.data() + b * block_bytes(type), m.w.data() + b * block_vals(type));
    return m;
}
static View upload(Mtl &m, const void *p, size_t n) { View v = m.alloc(n); memcpy(v.host(), p, n); return v; }
static View upload(Mtl &m, const std::vector<float> &v) { return upload(m, v.data(), v.size() * 4); }
static std::vector<float> randv(size_t n, std::mt19937 &rng, float a = 1.f) {
    std::uniform_real_distribution<float> U(-a, a);
    std::vector<float> v(n);
    for (auto &x : v) x = U(rng);
    return v;
}
static void run(Mtl &m, const std::function<void(Enc &)> &f) {
    id<MTLCommandBuffer> cb = [m.q commandBuffer];
    Enc e;
    e.begin(m, cb);
    f(e);
    e.end();
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
}

static int fails = 0, checks = 0;
static void check(const std::string &what, const float *got, const double *want, const double *scale, size_t n, double tol) {
    double worst = 0;
    bool finite = n > 0;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(got[i])) finite = false;
        worst = std::max(worst, std::fabs(got[i] - want[i]) / std::max(scale[i], 1e-30));
    }
    const bool ok = finite && worst <= tol;
    if (finite) printf("%-4s %-58s max |err| / scale %.3g (tol %.0e, n %zu)\n", ok ? "ok" : "FAIL", what.c_str(), worst, tol, n);
    else printf("FAIL %-58s non-finite values (n %zu)\n", what.c_str(), n);
    fails += !ok; checks++;
}
static void ref_gemv(const Mat &W, const float *x, int ldx, int T, std::vector<double> &y, std::vector<double> &s) {
    y.assign((size_t)T * W.rows, 0); s.assign((size_t)T * W.rows, 0);
    for (int t = 0; t < T; t++)
        for (int r = 0; r < W.rows; r++) {
            double a = 0, b = 0;
            const float *w = &W.w[(size_t)r * W.cols], *xt = x + (size_t)t * ldx;
            for (int k = 0; k < W.cols; k++) { a += (double)w[k] * xt[k]; b += std::fabs((double)w[k] * xt[k]); }
            y[(size_t)t * W.rows + r] = a; s[(size_t)t * W.rows + r] = b;
        }
}

static void deq_checks(Mtl &m, std::mt19937 &rng) {
    for (int type : {IQ4_XS, IQ4_NL, Q3_K, Q2_K, Q6_K, Q8_0, F16}) {
        const int rows = 7, cols = 2560;
        Mat W = make_mat(type, rows, cols, rng);
        View w = upload(m, W.raw.data(), W.raw.size()), out = m.alloc((size_t)rows * cols * 4);
        const uint32_t a[4] = {(uint32_t)cols, (uint32_t)W.rb, (uint32_t)rows, 0};
        run(m, [&](Enc &e) { e.k(std::string("deq_rows_") + name_of(type)).c(0, a).b(1, w).b(2, out).grid((cols / 32 + 31) / 32, rows, 1, 32); });
        std::vector<double> want(W.w.begin(), W.w.end()), scale(want.size());
        double mag = 0;
        for (double v : want) mag = std::max(mag, std::fabs(v));
        std::fill(scale.begin(), scale.end(), mag);
        check(std::string("deq_rows_") + name_of(type), out.host<float>(), want.data(), scale.data(), want.size(), 1e-6);
    }
}

struct GemvArgs { uint32_t rows, cols, rb, T, ldx, ldy, add; float scale; uint32_t ggml; };
static void gemv_checks(Mtl &m, std::mt19937 &rng) {
    struct V { const char *k; int lpr, r, cols, rows, T; };
    const V vs[] = {{"gemvr2_", 32, 2, 2560, 515, 1}, {"gemv32_", 32, 4, 2560, 10250, 1}, {"gemv32t_", 32, 4, 6144, 517, 5},
                    {"gemv8_", 8, 2, 640, 301, 1}, {"gemv8t_", 8, 2, 640, 301, 5}};
    for (int type : {Q8_0, Q6_K, F16})
        for (const V &v : vs) {
            if (type == Q6_K && v.cols % 256) continue;
            Mat W = make_mat(type, v.rows, v.cols, rng);
            std::vector<float> x = randv((size_t)v.T * v.cols, rng), y0 = randv((size_t)v.T * v.rows, rng);
            std::vector<double> ref, s;
            ref_gemv(W, x.data(), v.cols, v.T, ref, s);
            View w = upload(m, W.raw.data(), W.raw.size()), xv = upload(m, x);
            for (int add : {0, 1}) {
                View yv = upload(m, y0);
                const float sc = add ? 0.5f : 1.f;
                GemvArgs a{(uint32_t)v.rows, (uint32_t)v.cols, (uint32_t)W.rb, (uint32_t)v.T, (uint32_t)v.cols, (uint32_t)v.rows, (uint32_t)add, sc, 0};
                const int per_tg = 4 * (32 / v.lpr) * v.r;
                run(m, [&](Enc &e) { e.k(std::string(v.k) + name_of(type)).c(0, a).b(1, w).b(2, xv).b(3, yv).b(4, yv).grid((v.rows + per_tg - 1) / per_tg, 128); });
                std::vector<double> want(ref.size()), scale(ref.size());
                for (size_t i = 0; i < ref.size(); i++) { want[i] = sc * ref[i] + (add ? y0[i] : 0); scale[i] = sc * s[i] + (add ? std::fabs(y0[i]) : 0); }
                char nm[96];
                snprintf(nm, sizeof nm, "%s%s %d x %d T=%d%s", v.k, name_of(type), v.rows, v.cols, v.T, add ? " add, scale 0.5" : "");
                check(nm, yv.host<float>(), want.data(), scale.data(), want.size(), 1e-5);
            }
        }
    const int T = 3, cols = 6144;
    Mat W = make_mat(Q8_0, D, cols, rng);
    std::vector<float> x = randv((size_t)T * cols, rng), res = randv((size_t)T * HCD, rng), inj = randv((size_t)T * HC, rng, 4);
    std::vector<double> ref, s;
    ref_gemv(W, x.data(), cols, T, ref, s);
    View w = upload(m, W.raw.data(), W.raw.size()), xv = upload(m, x), rv = upload(m, res), iv = upload(m, inj);
    GemvArgs a{(uint32_t)D, (uint32_t)cols, (uint32_t)W.rb, (uint32_t)T, (uint32_t)cols, (uint32_t)HCD, 2, 1.f, 0};
    run(m, [&](Enc &e) { e.k("gemv32t_q8_0").c(0, a).b(1, w).b(2, xv).b(3, rv).b(4, iv).grid((D + 15) / 16, 128); });
    std::vector<double> want((size_t)T * HCD), scale(want.size());
    for (int t = 0; t < T; t++)
        for (int c = 0; c < HC; c++)
            for (int r = 0; r < D; r++) {
                const double g = 2 / (1 + std::exp(-inj[t * HC + c] / 4.0)), i = (size_t)t * HCD + c * D + r;
                want[i] = res[i] + ref[(size_t)t * D + r] * g;
                scale[i] = std::fabs(res[i]) + s[(size_t)t * D + r] * g;
            }
    check("gemv32t_q8_0 residual combine T=3", rv.host<float>(), want.data(), scale.data(), want.size(), 1e-5);
}

struct HcArgs { uint32_t T, inject, ggml; };
static void mixer_checks(Mtl &m, std::mt19937 &rng) {
    Mat Wd = make_mat(Q8_0, HCR, HCD, rng), Wi = make_mat(Q8_0, HC, HCD, rng), Wu = make_mat(Q8_0, HCD, HCR, rng);
    for (auto &[T, inject] : {std::pair<int, bool>{3, true}, {1, false}, {8, true}}) {
        std::vector<float> res = randv((size_t)T * HCD, rng, 2), gamma = randv(HCD, rng);
        for (auto &g : gamma) g += 1.5f;
        View rv = upload(m, res), gv = upload(m, gamma), wd = upload(m, Wd.raw.data(), Wd.raw.size()), wi = upload(m, Wi.raw.data(), Wi.raw.size()),
             wu = upload(m, Wu.raw.data(), Wu.raw.size());
        View lo = m.alloc((size_t)T * HCR * 4), inj = m.alloc((size_t)T * HC * 4), inv = m.alloc((size_t)T * HC * 4), mx = m.alloc((size_t)T * D * 4),
             mh = m.alloc((size_t)T * D * 2);
        HcArgs a{(uint32_t)T, inject ? 1u : 0u, 0};
        run(m, [&](Enc &e) {
            e.k("hc_pre").c(0, a).b(1, rv).b(2, gv).b(3, wd).b(4, inject ? wi : wd).b(5, lo).b(6, inj).b(7, inv).grid((HCR + (inject ? HC : 0) + 1) / 2, 128);
            e.barrier();
            e.k("hc_post").c(0, a).b(1, rv).b(2, gv).b(3, inv).b(4, lo).b(5, wu).b(6, mx).b(7, mh).grid(D / 4, 128);
        });
        std::vector<float> xn((size_t)T * HCD);
        std::vector<double> ivr((size_t)T * HC);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < HC; c++) {
                double ss = 0;
                for (int d = 0; d < D; d++) ss += (double)res[(size_t)t * HCD + c * D + d] * res[(size_t)t * HCD + c * D + d];
                ivr[t * HC + c] = 1 / std::sqrt(ss / D + 1e-6);
                for (int d = 0; d < D; d++) xn[(size_t)t * HCD + c * D + d] = (float)(res[(size_t)t * HCD + c * D + d] * ivr[t * HC + c] * gamma[c * D + d]);
            }
        std::vector<double> dn, dns, in, ins;
        ref_gemv(Wd, xn.data(), HCD, T, dn, dns);
        ref_gemv(Wi, xn.data(), HCD, T, in, ins);
        std::vector<double> lo_want(dn.size()), lo_scale(dn.size());
        for (size_t i = 0; i < dn.size(); i++) { const double z = dn[i] / 4; lo_want[i] = z / (1 + std::exp(-z)); lo_scale[i] = 1.1 * dns[i] / 4; }
        check("hc_pre q8_0 lo T=" + std::to_string(T), lo.host<float>(), lo_want.data(), lo_scale.data(), lo_want.size(), 1e-5);
        if (inject) check("hc_pre q8_0 inject T=" + std::to_string(T), inj.host<float>(), in.data(), ins.data(), in.size(), 1e-5);
        std::vector<float> lof(lo_want.begin(), lo_want.end());
        std::vector<double> up, ups;
        ref_gemv(Wu, lof.data(), HCR, T, up, ups);
        std::vector<double> mx_want((size_t)T * D), mx_scale((size_t)T * D);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++) {
                double v = 0, s = 0;
                for (int c = 0; c < HC; c++) {
                    const double x = xn[(size_t)t * HCD + c * D + d], z = up[(size_t)t * HCD + c * D + d];
                    v += x / (1 + std::exp(-z));
                    s += std::fabs(x) * (1 / (1 + std::exp(-z)) + 0.25 * ups[(size_t)t * HCD + c * D + d]);
                }
                mx_want[(size_t)t * D + d] = v / 4; mx_scale[(size_t)t * D + d] = s / 4;
            }
        check("hc_post q8_0 mix T=" + std::to_string(T), mx.host<float>(), mx_want.data(), mx_scale.data(), mx_want.size(), 1e-5);
    }
}

static void embed_checks(Mtl &m, std::mt19937 &rng) {
    {
        const int V = 100, T = 4, toks[T] = {5, 99, 0, 5};
        Mat E = make_mat(Q8_0, V, D, rng);
        View ev = upload(m, E.raw.data(), E.raw.size()), tv = upload(m, toks, sizeof toks), res = m.alloc((size_t)T * HCD * 4);
        const uint32_t rb = (uint32_t)E.rb;
        run(m, [&](Enc &e) { e.k("embed_q8_0").b(0, tv).b(1, ev).c(2, rb).b(3, res).b(4, tv).grid(T, 80); });
        std::vector<double> want((size_t)T * HCD), scale(want.size(), 1.0);
        for (int t = 0; t < T; t++) for (int c = 0; c < HC; c++) for (int d = 0; d < D; d++) want[(size_t)t * HCD + c * D + d] = E.w[(size_t)toks[t] * D + d];
        check("embed_q8_0 (exact, every stream)", res.host<float>(), want.data(), scale.data(), want.size(), 1e-7);
    }
    {
        const int T = 3;
        Mat R = make_mat(Q8_0, T * PLE_HEADS, PLE_DIM, rng);
        if (R.rb != PLE_ROW) FAIL("PLE_ROW %d, a Q8_0 row of %d is %zu bytes", PLE_ROW, PLE_DIM, R.rb);
        View rv = upload(m, R.raw.data(), R.raw.size()), emb = m.alloc((size_t)T * D * 4), eh = m.alloc((size_t)T * D * 2);
        const uint32_t nb = T * PLE_HEADS * 5;
        run(m, [&](Enc &e) { e.k("ple_embed").b(0, rv).b(1, emb).b(2, eh).c(3, nb).grid((nb + 255) / 256, 256); });
        std::vector<double> want(R.w.begin(), R.w.end()), scale(want.size(), 1.0);
        check("ple_embed q8_0 rows (exact)", emb.host<float>(), want.data(), scale.data(), want.size(), 1e-7);
    }
}

struct Experts_ {
    int n = 0;
    std::vector<int> type;
    std::vector<Mat> gate, up, down;
    View slots;
};
static Experts_ make_experts(Mtl &m, int n, std::mt19937 &rng) {
    Experts_ x;
    x.n = n;
    x.slots = m.alloc((size_t)n * XSLOT);
    const int types[3] = {IQ4_XS, Q3_K, Q2_K};
    for (int e = 0; e < n; e++) {
        const int t = types[e % 3];
        x.type.push_back(t);
        x.down.push_back(make_mat(IQ4_NL, D, FE, rng));
        x.gate.push_back(make_mat(t, FE, D, rng));
        x.up.push_back(make_mat(t, FE, D, rng));
        uint8_t *s = x.slots.host<uint8_t>() + (size_t)e * XSLOT;
        if (x.down[e].raw.size() != XDN || x.gate[e].raw.size() != xgu_bytes(t)) FAIL("expert sizes");
        memcpy(s, x.down[e].raw.data(), XDN);
        memcpy(s + XDN, x.gate[e].raw.data(), xgu_bytes(t));
        memcpy(s + XDN + xgu_bytes(t), x.up[e].raw.data(), xgu_bytes(t));
    }
    return x;
}
static void ref_expert(const Experts_ &x, int e, const float *xt, std::vector<double> &h, std::vector<double> &y, std::vector<double> &ys) {
    h.assign(FE, 0); y.assign(D, 0); ys.assign(D, 0);
    for (int j = 0; j < FE; j++) {
        double g = 0, u = 0;
        for (int k = 0; k < D; k++) { g += (double)x.gate[e].w[(size_t)j * D + k] * xt[k]; u += (double)x.up[e].w[(size_t)j * D + k] * xt[k]; }
        h[j] = g / (1 + std::exp(-g)) * u;
    }
    for (int r = 0; r < D; r++)
        for (int j = 0; j < FE; j++) { y[r] += x.down[e].w[(size_t)r * FE + j] * h[j]; ys[r] += std::fabs(x.down[e].w[(size_t)r * FE + j] * h[j]); }
}

struct MoeArgs { uint32_t T, layer, phase, ggml; };
static void moe_checks(Mtl &m, std::mt19937 &rng) {
    const int NX = 12, T = 2, P = T * TOPK_E, layer = 5;
    Experts_ x = make_experts(m, NX, rng);
    std::vector<int> ids(P);
    for (int t = 0; t < T; t++) {
        std::vector<int> all(NX);
        for (int i = 0; i < NX; i++) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        for (int k = 0; k < TOPK_E; k++) ids[t * TOPK_E + k] = all[k];
    }
    const std::vector<int> missing = {1, 5, 9};
    View table = m.alloc((size_t)NL * NE * 8), fmt = m.alloc((size_t)NL * NE), idv = upload(m, ids.data(), P * 4);
    memset(table.host(), 0, (size_t)NL * NE * 8);
    memset(fmt.host(), 0, (size_t)NL * NE);
    for (int e = 0; e < NX; e++) {
        fmt.host<uint8_t>()[layer * NE + e] = (uint8_t)x.type[e];
        if (std::find(missing.begin(), missing.end(), e) == missing.end()) table.host<uint64_t>()[layer * NE + e] = x.slots.gpu() + (size_t)e * XSLOT;
    }
    std::vector<float> xin = randv((size_t)T * D, rng);
    View xv = upload(m, xin), h = m.alloc((size_t)P * FE * 4), y = m.alloc((size_t)P * D * 4), done = m.alloc((size_t)P * 4);
    memset(h.host(), 0xff, (size_t)P * FE * 4);
    memset(y.host(), 0xff, (size_t)P * D * 4);
    auto phase = [&](uint32_t ph) {
        MoeArgs a{(uint32_t)T, (uint32_t)layer, ph, 0};
        run(m, [&](Enc &e) {
            e.k("moe_gu").c(0, a).b(1, table).b(2, idv).b(3, xv).b(4, h).b(5, done).b(6, fmt).grid(T * TOPK_E * FE / 16, 128);
            e.barrier();
            e.k("moe_down").c(0, a).b(1, table).b(2, idv).b(3, h).b(4, y).b(5, done).grid(T * TOPK_E * D / 16, 128);
        });
    };
    phase(0);
    int wrong_done = 0, n_missing = 0;
    for (int p = 0; p < P; p++) {
        const bool miss = std::find(missing.begin(), missing.end(), ids[p]) != missing.end();
        n_missing += miss;
        wrong_done += done.host<uint32_t>()[p] != (miss ? 0u : 1u);
    }
    printf("%-4s %-58s %d picks marked wrong (%d of %d missing)\n", wrong_done || !n_missing ? "FAIL" : "ok", "moe_gu phase 0: done[] = expert in memory", wrong_done, n_missing, P);
    fails += wrong_done || !n_missing; checks++;
    for (int e : missing) table.host<uint64_t>()[layer * NE + e] = x.slots.gpu() + (size_t)e * XSLOT;
    phase(1);
    std::vector<double> hw((size_t)P * FE), yw((size_t)P * D), ysc((size_t)P * D);
    for (int p = 0; p < P; p++) {
        std::vector<double> hp, yp, ys;
        ref_expert(x, ids[p], &xin[(size_t)(p / TOPK_E) * D], hp, yp, ys);
        std::copy(hp.begin(), hp.end(), hw.begin() + (size_t)p * FE);
        std::copy(yp.begin(), yp.end(), yw.begin() + (size_t)p * D);
        std::copy(ys.begin(), ys.end(), ysc.begin() + (size_t)p * D);
    }
    for (int f : {IQ4_XS, Q3_K, Q2_K}) {
        std::vector<float> hg, yg;
        std::vector<double> hr, hs, yr, ys;
        for (int p = 0; p < P; p++) {
            if (x.type[ids[p]] != f) continue;
            double mag = 0;
            for (int j = 0; j < FE; j++) mag = std::max(mag, std::fabs(hw[(size_t)p * FE + j]));
            for (int j = 0; j < FE; j++) { hg.push_back(h.host<float>()[(size_t)p * FE + j]); hr.push_back(hw[(size_t)p * FE + j]); hs.push_back(mag); }
            for (int r = 0; r < D; r++) { yg.push_back(y.host<float>()[(size_t)p * D + r]); yr.push_back(yw[(size_t)p * D + r]); ys.push_back(ysc[(size_t)p * D + r]); }
        }
        check(std::string("moe_gu ") + name_of(f) + " gate / up, both phases", hg.data(), hr.data(), hs.data(), hg.size(), 1e-4);
        check(std::string("moe_down iq4_nl after ") + name_of(f) + ", both phases", yg.data(), yr.data(), ys.data(), yg.size(), 1e-4);
    }
}

static void pf_checks(Mtl &m, std::mt19937 &rng) {
    for (int type : {Q8_0, Q6_K, F16}) {
        const int rows = 200, cols = 2560, T = 133;
        Mat W = make_mat(type, rows, cols, rng);
        std::vector<float> xin = randv((size_t)T * cols, rng);
        std::vector<double> ref, s;
        ref_gemv(W, xin.data(), cols, T, ref, s);
        View w = upload(m, W.raw.data(), W.raw.size()), xv = upload(m, xin), yv = m.alloc((size_t)T * rows * 4);
        run(m, [&](Enc &e) { pf_mm(e, type, w, rows, cols, W.rb, xv, cols, yv, rows, T); });
        check(std::string("pf_mm_") + name_of(type) + " 200 x 2560 T=133", yv.host<float>(), ref.data(), s.data(), ref.size(), 2e-3);
    }
    const int NX = 12, T = 70, P = T * TOPK_E, layer = 9;
    Experts_ x = make_experts(m, NX, rng);
    std::vector<int> ids(P);
    std::vector<float> wts(P);
    for (int t = 0; t < T; t++) {
        std::vector<int> all(NX);
        for (int i = 0; i < NX; i++) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        for (int k = 0; k < TOPK_E; k++) { ids[t * TOPK_E + k] = all[k]; wts[t * TOPK_E + k] = 0.05f + 0.1f * (rng() % 10) / 10.f; }
    }
    View table = m.alloc((size_t)NL * NE * 8), fmt = m.alloc((size_t)NL * NE), idv = upload(m, ids.data(), P * 4), wv = upload(m, wts);
    memset(table.host(), 0, (size_t)NL * NE * 8);
    memset(fmt.host(), 0, (size_t)NL * NE);
    for (int e = 0; e < NX; e++) {
        fmt.host<uint8_t>()[layer * NE + e] = (uint8_t)x.type[e];
        table.host<uint64_t>()[layer * NE + e] = x.slots.gpu() + (size_t)e * XSLOT;
    }
    std::vector<float> xin = randv((size_t)T * D, rng);
    std::vector<double> want((size_t)T * D, 0);
    for (int p = 0; p < P; p++) {
        std::vector<double> hp, yp, ys;
        ref_expert(x, ids[p], &xin[(size_t)(p / TOPK_E) * D], hp, yp, ys);
        for (int r = 0; r < D; r++) want[(size_t)(p / TOPK_E) * D + r] += wts[p] * yp[r];
    }
    View xv = upload(m, xin);
    for (int bt : {16, 32}) {
        View scratch = m.alloc(pf_moe_scratch(P, bt)), Y = m.alloc((size_t)T * D * 4);
        const PfMoeBufs s = pf_moe_bufs(scratch, P, bt);
        run(m, [&](Enc &e) { pf_moe(e, table, fmt, layer, idv, wv, xv, D, T, s, Y, bt); });
        double worst = 0;
        bool finite = true;
        for (int t = 0; t < T; t++) {
            double en = 0, rn = 0;
            for (int r = 0; r < D; r++) {
                const double g = Y.host<float>()[(size_t)t * D + r], w = want[(size_t)t * D + r];
                if (!std::isfinite(g)) finite = false;
                en += (g - w) * (g - w); rn += w * w;
            }
            worst = std::max(worst, std::sqrt(en / rn));
        }
        const bool ok = finite && worst < 5e-3;
        printf("%-4s %-58s worst token |err|_2 / |ref|_2 %.3g (tol 5e-03)\n", ok ? "ok" : "FAIL",
               ("pf_moe mixed formats T=70 bt=" + std::to_string(bt)).c_str(), worst);
        fails += !ok; checks++;
    }
}

int main() {
    Mtl m;
    m.init();
    std::mt19937 rng(1234);
    deq_checks(m, rng);
    gemv_checks(m, rng);
    mixer_checks(m, rng);
    embed_checks(m, rng);
    moe_checks(m, rng);
    pf_checks(m, rng);
    printf("%d of %d checks failed\n", fails, checks);
    return fails ? 1 : 0;
}
