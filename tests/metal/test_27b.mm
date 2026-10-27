#include "model.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

static const int8_t IQ4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static float h2f(const uint8_t *p) { __fp16 h; memcpy(&h, p, 2); return (float)h; }

static int block_bytes(int type) { return type == Q8_0 ? 34 : type == Q6_K ? 210 : 136; }
static int block_vals(int type) { return type == Q8_0 ? 32 : 256; }

static void ref_dequant(int type, const uint8_t *b, float *y) {
    if (type == Q8_0) {
        const float d = h2f(b);
        for (int j = 0; j < 32; j++) y[j] = d * (int8_t)b[2 + j];
    } else if (type == IQ4_XS) {
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
    } else {
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
    }
}

static std::vector<uint8_t> random_blocks(int type, size_t nb, std::mt19937 &rng) {
    std::vector<uint8_t> raw(nb * block_bytes(type));
    for (auto &x : raw) x = rng();
    for (size_t b = 0; b < nb; b++) {
        uint8_t *p = raw.data() + b * block_bytes(type) + (type == Q6_K ? 208 : 0);
        const __fp16 d = (__fp16)(((rng() % 1000) + 1) * (type == Q8_0 ? 1e-5f : 1e-6f));
        memcpy(p, &d, 2);
    }
    return raw;
}

struct Mat { QTensor t; std::vector<float> ref; };
static Mat make_mat(Engine &e, int type, int rows, int cols, std::mt19937 &rng, bool want_ref) {
    Mat m;
    const size_t n = (size_t)rows * cols, nb = n / block_vals(type);
    const std::vector<uint8_t> raw = random_blocks(type, nb, rng);
    Arena dry; dry.cap = SIZE_MAX;
    make_qtensor(dry, type, rows, cols);
    Arena a; a.base = e.m.alloc(dry.used, "test weights"); a.cap = dry.used;
    m.t = make_qtensor(a, type, rows, cols);
    relayout(m.t, raw.data(), 0, nb);
    if (want_ref) {
        m.ref.resize(n);
        for (size_t b = 0; b < nb; b++) ref_dequant(type, raw.data() + b * block_bytes(type), m.ref.data() + b * block_vals(type));
    }
    return m;
}

static int fails = 0;
static void check(const char *what, const float *got, const double *want, size_t n, double tol) {
    double err = 0, mag = 0;
    for (size_t i = 0; i < n; i++) { err = std::max(err, std::fabs(got[i] - want[i])); mag = std::max(mag, std::fabs(want[i])); }
    const bool ok = err <= tol * mag && mag > 0;
    printf("%-44s max err %.3g of %.3g  %s\n", what, err, mag, ok ? "ok" : "FAIL");
    fails += !ok;
}

static void attention_checks(Engine &e, std::mt19937 &rng) {
    e.init_kv(16);
    Layer w;
    DP<float> nw = e.m.alloc(2 * HD * 4, "norms").as<float>();
    w.q_norm = nw; w.k_norm = nw + HD;
    std::uniform_real_distribution<float> U(-1, 1);
    for (int i = 0; i < 2 * HD; i++) nw.h[i] = 1.0f + 0.3f * U(rng);
    const int n = 300;
    std::vector<float> qg((size_t)n * NH * 2 * HD), kk((size_t)n * NKV * HD), vv((size_t)n * NKV * HD);
    for (auto &x : qg) x = U(rng) * 2;
    for (auto &x : kk) x = U(rng) * 2;
    for (auto &x : vv) x = U(rng);
    for (int i = 0; i < n * NKV * HD; i += 37) kk[i] *= 8;
    auto normed = [&](const float *x, const float *wt, int pos, double *out) {
        double ss = 0;
        for (int i = 0; i < HD; i++) ss += (double)x[i] * x[i];
        const double inv = 1 / std::sqrt(ss / HD + 1e-6);
        for (int i = 0; i < HD; i++) out[i] = x[i] * inv * wt[i];
        double tmp[ROT];
        for (int i = 0; i < ROT; i++) tmp[i] = out[i];
        for (int i = 0; i < ROT; i++) {
            const int f = i % (ROT / 2);
            const double ang = pos * std::pow(1e7, -2.0 * f / ROT), c = std::cos(ang), sn = std::sin(ang);
            out[i] = i < ROT / 2 ? tmp[i] * c - tmp[i + ROT / 2] * sn : tmp[i] * c + tmp[i - ROT / 2] * sn;
        }
    };
    std::vector<double> Kn((size_t)n * NKV * HD);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < NKV; h++) normed(&kk[((size_t)t * NKV + h) * HD], nw.h + HD, t, &Kn[((size_t)t * NKV + h) * HD]);
    auto ref_row = [&](int t, int h, double *o) {
        double q[HD];
        normed(&qg[((size_t)t * NH + h) * 2 * HD], nw.h, t, q);
        const int g = h / 6;
        std::vector<double> sc(t + 1);
        double mx = -1e300;
        for (int j = 0; j <= t; j++) {
            double s = 0;
            for (int i = 0; i < HD; i++) s += q[i] * Kn[((size_t)j * NKV + g) * HD + i];
            sc[j] = s / 16; mx = std::max(mx, sc[j]);
        }
        double L = 0;
        for (int j = 0; j <= t; j++) { sc[j] = std::exp(sc[j] - mx); L += sc[j]; }
        for (int i = 0; i < HD; i++) {
            double acc = 0;
            for (int j = 0; j <= t; j++) acc += sc[j] * vv[((size_t)j * NKV + g) * HD + i];
            o[i] = acc / L / (1 + std::exp(-(double)qg[((size_t)t * NH + h) * 2 * HD + HD + i]));
        }
    };
    Seq &s = e.seqs[0];
    s.used = true;
    auto run = [&](int t0, int T, const char *what) {
        e.ensure_pages(s, t0 + T, t0);
        std::copy(qg.begin() + (size_t)t0 * NH * 2 * HD, qg.begin() + (size_t)(t0 + T) * NH * 2 * HD, e.b.proj.h);
        std::copy(kk.begin() + (size_t)t0 * NKV * HD, kk.begin() + (size_t)(t0 + T) * NKV * HD, e.b.k.h);
        std::copy(vv.begin() + (size_t)t0 * NKV * HD, vv.begin() + (size_t)(t0 + T) * NKV * HD, e.b.v.h);
        const bool small = T <= 8;
        const Engine::Part p{&s, T, 0, t0, t0, false, false};
        e.attend(w, 3, p, small);
        e.m.sync();
        std::vector<float> got((size_t)T * NH * HD);
        std::vector<double> want((size_t)T * NH * HD);
        for (size_t i = 0; i < got.size(); i++) got[i] = small ? e.b.o.h[i] : (float)e.b.mix.h[i];
        for (int t = 0; t < T; t++)
            for (int h = 0; h < NH; h++) ref_row(t0 + t, h, &want[((size_t)t * NH + h) * HD]);
        check(what, got.data(), want.data(), got.size(), 0.03);
    };
    run(0, 200, "attention prefill, 200 tokens");
    run(200, 1, "attention decode at 200");
    run(201, 4, "attention verify, 4 rows at 201");
    run(205, 8, "attention verify, 8 rows at 205");
    run(213, 87, "attention prefill, 87 tokens at 213");
    e.attn_gemm_min = 0;
    run(213, 87, "attention prefill as GEMMs, 87 tokens at 213");
    e.attn_gemm_min = 4096;
}

static void gemv_dram(Engine &e, std::mt19937 &rng) {
    struct Shape { const char *name; int type, rows, cols; } shapes[] = {
        {"ffn_gate IQ4_XS", IQ4_XS, FF, D}, {"ffn_down IQ4_XS", IQ4_XS, D, FF}, {"attn_qkv IQ4_XS", IQ4_XS, QKV, D},
        {"attn_gate IQ4_XS", IQ4_XS, GV * GD, D}, {"ssm_out Q6_K", Q6_K, D, GV * GD}, {"output Q6_K", Q6_K, VOCAB, D},
        {"attn_k Q8_0", Q8_0, NKV * HD, D}, {"big IQ4_XS", IQ4_XS, VOCAB / 2, D}};
    DP<float> x = e.m.alloc(8 * FF * 4 * 2, "x").as<float>(), y = e.m.alloc((size_t)8 * VOCAB * 4, "y").as<float>();
    DP<half_t> x16 = e.m.alloc(8 * FF * 2 * 2, "x16").as<half_t>();
    for (int i = 0; i < 8 * FF * 2; i++) { x.h[i] = 0.01f * (i % 13); x16.h[i] = (half_t)x.h[i]; }
    for (auto &s : shapes) {
        const double bytes = (double)s.rows * s.cols * (s.type == Q8_0 ? 34.0 / 32 : s.type == Q6_K ? 210.0 / 256 : 136.0 / 256);
        const int copies = std::max(2, (int)(600e6 / bytes));
        std::vector<Mat> ms;
        for (int c = 0; c < copies; c++) ms.push_back(make_mat(e, s.type, s.rows, s.cols, rng, false));
        for (int T : {1, 2, 4, 8}) {
            const int reps = std::max(copies, 24);
            for (int it = 0; it < copies; it++) e.gemv(ms[it].t, x, x16, s.cols, T, y, s.rows, GEMV_STORE);
            e.m.sync();
            static const bool nobar = getenv("NOBAR");
            for (int it = 0; it < reps; it++) { if (nobar) e.m.par(); e.gemv(ms[it % copies].t, x, x16, s.cols, T, y, s.rows, GEMV_STORE); }
            e.m.sync();
            const double ms_ = e.m.last_gpu_ms / reps;
            printf("%-18s T=%d  %7.3f ms  %6.1f GB/s from memory\n", s.name, T, ms_, bytes / ms_ / 1e6);
        }
    }
}

int main(int argc, char **argv) {
    const bool bench = argc > 1 && !strcmp(argv[1], "bench");
    if (argc > 1 && !strcmp(argv[1], "gemv")) {
        Engine e;
        e.m.init(std::string(getenv("CHRYSALIS_METALLIB") ? getenv("CHRYSALIS_METALLIB") : "build/chrysalis.metallib"));
        std::mt19937 rng(7);
        e.umax = 256;
        e.alloc_buffers(1);
        gemv_dram(e, rng);
        return 0;
    }
    Engine e;
    e.m.init(std::string(getenv("CHRYSALIS_METALLIB") ? getenv("CHRYSALIS_METALLIB") : "build/chrysalis.metallib"));
    std::mt19937 rng(7);
    const char *names[] = {"Q8_0", "Q6_K", "IQ4_XS"};
    const int types[] = {Q8_0, Q6_K, IQ4_XS};
    e.umax = 256;
    e.alloc_buffers(1);
    if (!bench) {
        const int R = 64, K = 1024;
        DP<float> x = e.m.alloc(8 * K * 4, "x").as<float>(), y = e.m.alloc(8 * R * 4, "y").as<float>();
        DP<half_t> x16 = e.m.alloc(8 * K * 2, "x16").as<half_t>();
        for (int i = 0; i < 8 * K; i++) { x.h[i] = std::uniform_real_distribution<float>(-1, 1)(rng); x16.h[i] = (half_t)x.h[i]; }
        for (int ti = 0; ti < 3; ti++) {
            Mat m = make_mat(e, types[ti], R, K, rng, true), m2 = make_mat(e, types[ti], R, K, rng, true);
            for (int T : {1, 2, 3, 4, 8})
                for (int mode : {(int)GEMV_STORE, (int)GEMV_ADD, 2}) {
                    if (mode == 2 && types[ti] == Q6_K) continue;
                    const bool f32 = T == 1;
                    std::vector<double> want(T * R);
                    for (int t = 0; t < T; t++)
                        for (int r = 0; r < R; r++) {
                            double s = 0, s2 = 0;
                            for (int k = 0; k < K; k++) {
                                const double xv = f32 ? (double)x.h[t * K + k] : (double)x16.h[t * K + k];
                                s += (double)m.ref[(size_t)r * K + k] * xv; s2 += (double)m2.ref[(size_t)r * K + k] * xv;
                            }
                            const double y0 = 0.5 * (t + 1);
                            want[t * R + r] = mode == GEMV_ADD ? y0 + s : mode == 2 ? s / (1 + std::exp(-s)) * s2 : s;
                            y.h[t * R + r] = (float)y0;
                        }
                    std::vector<float> got(T * R);
                    if (mode == 2) {
                        e.gemv_glu(m.t, m2.t, x, x16, T);
                        e.m.sync();
                        for (int t = 0; t < T; t++)
                            for (int r = 0; r < R; r++) got[t * R + r] = f32 ? e.b.proj.h[(size_t)t * FF + r] : (float)e.b.ffn_g.h[(size_t)t * FF + r];
                    } else {
                        e.gemv(m.t, x, x16, K, T, y, R, mode);
                        e.m.sync();
                        std::copy(y.h, y.h + T * R, got.begin());
                    }
                    char what[96];
                    snprintf(what, sizeof what, "gemv %s T=%d %s", names[ti], T, mode == 0 ? "store" : mode == 1 ? "add" : "swiglu");
                    check(what, got.data(), want.data(), T * R, f32 ? 1e-4 : 2e-3);
                }
            {
                DP<float> nw = e.m.alloc(K * 4, "nw").as<float>();
                double ss = 0;
                for (int k = 0; k < K; k++) { nw.h[k] = 0.5f + (k % 7) * 0.1f; ss += (double)x.h[k] * x.h[k]; }
                const double inv = 1 / std::sqrt(ss / K + 1e-6);
                std::vector<double> want(R);
                for (int r = 0; r < R; r++) {
                    double sum = 0;
                    for (int k = 0; k < K; k++) sum += (double)m.ref[(size_t)r * K + k] * x.h[k] * inv * nw.h[k];
                    want[r] = sum;
                }
                e.gemv(m.t, x, {}, K, 1, y, R, GEMV_STORE, nw);
                e.m.sync();
                char what[96];
                snprintf(what, sizeof what, "gemv %s T=1 with the input's norm", names[ti]);
                check(what, y.h, want.data(), R, 1e-4);
            }
            for (int pass = 0; pass < 3; pass++) {
            const int T = pass ? 40 : 20;
            e.mps_min = pass == 2 ? 0 : 512;
            DP<half_t> xh = e.m.alloc(T * K * 2, "xh").as<half_t>();
            DP<float> yy = e.m.alloc(T * R * 4, "yy").as<float>();

            std::vector<double> want(T * R);
            for (int i = 0; i < T * K; i++) xh.h[i] = (half_t)x.h[i % (8 * K)];
            for (int t = 0; t < T; t++)
                for (int r = 0; r < R; r++) {
                    double s = 1.0;
                    for (int k = 0; k < K; k++) s += (double)(half_t)m.ref[(size_t)r * K + k] * (double)xh.h[t * K + k];
                    want[t * R + r] = s;
                    yy.h[t * R + r] = 1.0f;
                }
            e.linear(m.t, T, xh, K, {}, 0, yy.as<uint8_t>(), R, true, false);
            e.m.sync();
            char what[96];
            snprintf(what, sizeof what, "%s %s T=%d add", T <= 32 ? "prefill (matrix GEMV)" : e.mps_min ? "prefill (gemm_q)" : "prefill (dequant + MPS)", names[ti], T);
            check(what, yy.h, want.data(), T * R, 2e-3);
            }
        }
        {
            const int RD = 64, KD = 8704;
            Mat m = make_mat(e, IQ4_XS, RD, KD, rng, true);
            DP<float> xd = e.m.alloc(8 * KD * 4, "xd").as<float>(), yd = e.m.alloc(8 * RD * 4, "yd").as<float>();
            DP<half_t> xd16 = e.m.alloc(8 * KD * 2, "xd16").as<half_t>();
            for (int i = 0; i < 8 * KD; i++) { xd.h[i] = std::uniform_real_distribution<float>(-1, 1)(rng); xd16.h[i] = (half_t)xd.h[i]; }
            for (int T : {1, 2, 4, 8}) {
                const bool f32 = T == 1;
                std::vector<double> want(T * RD);
                for (int t = 0; t < T; t++)
                    for (int r = 0; r < RD; r++) {
                        double s = 0;
                        for (int k = 0; k < KD; k++) s += (double)m.ref[(size_t)r * KD + k] * (f32 ? (double)xd.h[t * KD + k] : (double)xd16.h[t * KD + k]);
                        want[t * RD + r] = s;
                    }
                e.gemv(m.t, xd, xd16, KD, T, yd, RD, GEMV_STORE);
                e.m.sync();
                char what[96];
                snprintf(what, sizeof what, "gemv IQ4_XS %d columns T=%d store", KD, T);
                check(what, yd.h, want.data(), T * RD, f32 ? 1e-4 : 2e-3);
            }
        }
        for (int ti = 0; ti < 3; ti++)
            for (const int K : {1024, 8704}) {
                const int R = 64;
                Mat m = make_mat(e, types[ti], R, K, rng, false), m2 = make_mat(e, types[ti], R, K, rng, false);
                DP<half_t> xa = e.m.alloc(8 * K * 2, "xa").as<half_t>();
                DP<float> ya = e.m.alloc(8 * R * 4, "ya").as<float>();
                for (int i = 0; i < 8 * K; i++) xa.h[i] = (half_t)std::uniform_real_distribution<float>(-1, 1)(rng);
                for (int glu = 0; glu < 2; glu++) {
                    if (glu && types[ti] == Q6_K) continue;
                    std::vector<float> all;
                    int same = 0, compared = 0;
                    for (int T = 8; T >= 2; T--) {
                        if (glu) e.gemv_glu(m.t, m2.t, {}, xa, T);
                        else e.gemv(m.t, {}, xa, K, T, ya, R, GEMV_STORE);
                        e.m.sync();
                        std::vector<float> got((size_t)T * R);
                        for (int t = 0; t < T; t++)
                            for (int r = 0; r < R; r++) got[(size_t)t * R + r] = glu ? (float)e.b.ffn_g.h[(size_t)t * FF + r] : ya.h[(size_t)t * R + r];
                        if (T == 8) { all = got; continue; }
                        for (size_t i = 0; i < got.size(); i++, compared++) same += !memcmp(&got[i], &all[i], 4);
                    }
                    const bool ok = compared > 0 && same == compared;
                    printf("gemv %s %d columns%s: rows alone as in 8 (T=2-7, %d/%d)  %s\n", names[ti], K, glu ? " swiglu" : "", same, compared,
                           ok ? "ok" : "FAIL");
                    fails += !ok;
                }
            }
        attention_checks(e, rng);
        printf(fails ? "%d FAILURES\n" : "all ok\n", fails);
        return fails != 0;
    }
    {
        DP<uint8_t> z = e.m.alloc(1 << 20, "z");
        for (int pass = 0; pass < 2; pass++)
            for (int bar = 1; bar >= 0; bar--) {
                for (int i = 0; i < 500; i++) { if (!bar) e.m.par(); e.fill0(z, 4096); }
                e.m.sync();
                if (pass) printf("500 tiny dispatches, %s: GPU %.3f ms\n", bar ? "barrier before each" : "no barriers", e.m.last_gpu_ms);
            }
    }
    {
        e.umax = 2048;
        e.alloc_buffers(1);
        e.init_kv(80);
        Seq &sq = e.seqs[0];
        sq.used = true;
        e.ensure_pages(sq, 4096, 0);
        Layer w;
        DP<float> nw = e.m.alloc(2 * HD * 4, "norms").as<float>();
        for (int i = 0; i < 2 * HD; i++) nw.h[i] = 1.0f;
        w.q_norm = nw; w.k_norm = nw + HD;
        for (size_t i = 0; i < (size_t)2048 * NH * 2 * HD; i++) e.b.proj.h[i] = (float)((i * 2654435761u) % 1000) / 500.0f - 1.0f;
        for (size_t i = 0; i < (size_t)2048 * NKV * HD; i++) { e.b.k.h[i] = (float)((i * 40503u) % 1000) / 500.0f - 1.0f; e.b.v.h[i] = e.b.k.h[i]; }
        for (int rep = 0; rep < 2; rep++) {
            const Engine::Part p{&sq, 2048, 0, 2048, 2048, false, false};
            e.attend(w, 3, p, false);
            e.m.sync();
            if (rep) printf("prefill attention, 2048 rows at 2048: GPU %.1f ms\n", e.m.last_gpu_ms);
        }
        e.init_kv(1100);
        Seq &sd = e.seqs[0];
        for (int p : sd.pages) e.page_release(p);
        sd.pages.clear();
        e.ensure_pages(sd, 65536 + 8, 0);
        for (int pg : sd.pages) e.fill0(e.page_mem[pg], KV_PAGE);
        e.m.make_resident();
        e.m.sync();
        for (int T : {1, 4}) {
            for (int rep = 0; rep < 2; rep++) {
                const Engine::Part p{&sd, T, 0, 65536, 65536, false, false};
                e.attend(w, 3, p, true);
                e.m.sync();
                if (rep) printf("decode attention, %d rows at 65536: GPU %.2f ms (%.0f GB/s of KV)\n", T, e.m.last_gpu_ms,
                                65536.0 * NKV * KV_HEAD / PAGE / e.m.last_gpu_ms / 1e6);
            }
        }
        if (argc > 2) return 0;
    }
    struct Shape { const char *name; int type, rows, cols; } shapes[] = {
        {"ffn_gate IQ4_XS", IQ4_XS, FF, D}, {"ffn_down IQ4_XS", IQ4_XS, D, FF}, {"attn_qkv IQ4_XS", IQ4_XS, QKV, D},
        {"ssm_out Q6_K", Q6_K, D, GV * GD}, {"output Q6_K", Q6_K, VOCAB, D}, {"attn_k Q8_0", Q8_0, NKV * HD, D},
        {"eh_proj Q8_0", Q8_0, D, 2 * D}};
    DP<float> x = e.m.alloc(8 * FF * 4 * 2, "x").as<float>(), y = e.m.alloc((size_t)8 * VOCAB * 4, "y").as<float>();
    DP<half_t> x16 = e.m.alloc(8 * FF * 2 * 2, "x16").as<half_t>();
    for (auto &s : shapes) {
        Mat m = make_mat(e, s.type, s.rows, s.cols, rng, false);
        const double bytes = (double)s.rows * s.cols * (s.type == Q8_0 ? 34.0 / 32 : s.type == Q6_K ? 210.0 / 256 : 136.0 / 256);
        for (int T : {1, 4}) {
            const int reps = 20;
            for (int it = 0; it < 3; it++) e.gemv(m.t, x, x16, s.cols, T, y, s.rows, GEMV_STORE);
            e.m.sync();
            for (int it = 0; it < reps; it++) e.gemv(m.t, x, x16, s.cols, T, y, s.rows, GEMV_STORE);
            e.m.sync();
            const double ms = e.m.last_gpu_ms / reps;
            printf("%-18s T=%d  %7.3f ms  %6.1f GB/s\n", s.name, T, ms, bytes / ms / 1e6);
        }
    }
    {
        Mat m = make_mat(e, IQ4_XS, FF, D, rng, false);
        const int U = 2048;
        DP<half_t> xh = e.m.alloc((size_t)U * D * 2, "xh").as<half_t>();
        DP<float> yy = e.m.alloc((size_t)U * FF * 4, "yy").as<float>();
        for (int T : {16, 32, 64, 128, 256, 512, 1024, 2048}) {
                e.linear(m.t, T, xh, D, {}, 0, yy.as<uint8_t>(), FF, false, false);
                e.m.sync();
                const int reps = 5;
                for (int it = 0; it < reps; it++) e.linear(m.t, T, xh, D, {}, 0, yy.as<uint8_t>(), FF, false, false);
                e.m.sync();
                const double ms = e.m.last_gpu_ms / reps;
                printf("ffn_gate T=%-5d %8.2f ms  %5.2f TFLOPS\n", T, ms, 2.0 * T * FF * D / ms / 1e9);
            }
    }
    return 0;
}
