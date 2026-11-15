#include "../engine/cpu_exact.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

constexpr int D = 2560, FE = 640, NB = D / 32, NH = FE / 32;
constexpr int Q3K = 11, Q2K = 10, IQ4XS = 23, IQ4NL = 20;

static uint8_t KV[16];
static std::mt19937 rng(7);

static float h2f(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return ggml_fp16_to_fp32(h); }
static void put_h(uint8_t *p, float v) { const uint16_t h = ggml_fp32_to_fp16(v); memcpy(p, &h, 2); }
static size_t row_bytes(int type, int n) { return ggml_row_size((ggml_type)type, n); }
static bool same(const void *a, const void *b, size_t n) { return !memcmp(a, b, n); }

static float sum16(const float *v) {
    float t3[8], t6[4];
    for (int i = 0; i < 8; i++) t3[i] = v[i] + v[i + 8];
    for (int i = 0; i < 4; i++) t6[i] = t3[i] + t3[i + 4];
    return (t6[0] + t6[2]) + (t6[1] + t6[3]);
}

struct Gu {
    uint8_t w[D];
    float a[D / 16], b[D / 16];
};

static void dec_iq4(const uint8_t *qs, uint8_t *w) {
    for (int j = 0; j < 16; j++) { w[j] = KV[qs[j] & 15]; w[j + 16] = KV[qs[j] >> 4]; }
}

static void dec_iq4xs(const uint8_t *row, uint8_t *w, float *ws) {
    for (int sb = 0; sb < D / 256; sb++) {
        const uint8_t *blk = row + 136 * sb;
        const float d = h2f(blk);
        const int sh = blk[2] | blk[3] << 8;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((blk[4 + ib / 2] >> 4 * (ib % 2)) & 15) | (((sh >> 2 * ib) & 3) << 4);
            ws[sb * 8 + ib] = d * (float)(ls - 32);
            dec_iq4(blk + 8 + 16 * ib, w + (sb * 8 + ib) * 32);
        }
    }
}

static void dec_iq4nl(const uint8_t *row, int K, uint8_t *w, float *ws) {
    for (int b = 0; b < K / 32; b++) { ws[b] = h2f(row + 18 * b); dec_iq4(row + 18 * b + 2, w + 32 * b); }
}

static void q3k_scales(const uint8_t *blk, int *sc) {
    uint32_t a[3];
    memcpy(a, blk + 96, 12);
    const uint32_t s[4] = {(a[0] & 0x0f0f0f0f) | ((a[2] & 0x03030303) << 4), (a[1] & 0x0f0f0f0f) | (((a[2] >> 2) & 0x03030303) << 4),
                           ((a[0] >> 4) & 0x0f0f0f0f) | (((a[2] >> 4) & 0x03030303) << 4), ((a[1] >> 4) & 0x0f0f0f0f) | (((a[2] >> 6) & 0x03030303) << 4)};
    for (int l = 0; l < 16; l++) sc[l] = (int)((s[l / 4] >> 8 * (l % 4)) & 0xff) - 32;
}

static void dec_q3k(const uint8_t *row, Gu &g) {
    for (int sb = 0; sb < D / 256; sb++) {
        const uint8_t *blk = row + 110 * sb;
        int sc[16];
        q3k_scales(blk, sc);
        const float d = h2f(blk + 108);
        for (int l = 0; l < 16; l++) { g.a[sb * 16 + l] = d * (float)sc[l]; g.b[sb * 16 + l] = g.a[sb * 16 + l] * 4.0f; }
        for (int n = 0; n < 2; n++)
            for (int j = 0; j < 4; j++)
                for (int l = 0; l < 32; l++)
                    g.w[sb * 256 + 128 * n + 32 * j + l] = (uint8_t)(((blk[32 + 32 * n + l] >> 2 * j) & 3) | (((blk[l] >> (4 * n + j)) & 1) << 2));
    }
}

static void dec_q2k(const uint8_t *row, Gu &g) {
    for (int sb = 0; sb < D / 256; sb++) {
        const uint8_t *blk = row + 84 * sb;
        const float d = h2f(blk + 80), dm = h2f(blk + 82);
        for (int l = 0; l < 16; l++) { g.a[sb * 16 + l] = d * (float)(blk[l] & 15); g.b[sb * 16 + l] = dm * (float)(blk[l] >> 4); }
        for (int n = 0; n < 2; n++)
            for (int j = 0; j < 4; j++)
                for (int l = 0; l < 32; l++) g.w[sb * 256 + 128 * n + 32 * j + l] = (uint8_t)((blk[16 + 32 * n + l] >> 2 * j) & 3);
    }
}

static int dot_bytes(const uint8_t *w, const int8_t *x, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += (int)w[i] * (int)x[i];
    return s;
}

struct Act {
    std::vector<int8_t> q;
    std::vector<float> s, s16, f16, xf;
    std::vector<int32_t> sum;
    std::string name;
};

static Act act_from(const std::vector<int8_t> &q, const std::vector<float> &s, const std::string &name) {
    Act a{q, s, std::vector<float>(D / 16), std::vector<float>(D / 16), std::vector<float>(NB), std::vector<int32_t>(D / 16), name};
    for (int g = 0; g < D / 16; g++) {
        int t = 0;
        for (int i = 0; i < 16; i++) t += q[16 * g + i];
        a.sum[g] = t;
        a.s16[g] = s[g / 2];
        a.f16[g] = s[g / 2] * (float)t;
    }
    for (int b = 0; b < NB; b++) a.xf[b] = (float)(a.sum[2 * b] + a.sum[2 * b + 1]);
    return a;
}

static void ref_quant(const float *x, int n, int8_t *q, float *s, int32_t *sum, float *c) {
    for (int b = 0; b < n / 32; b++) {
        float m = 0;
        for (int i = 0; i < 32; i++) m = std::max(m, fabsf(x[32 * b + i]));
        const float d = m / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        int t[2] = {0, 0};
        for (int i = 0; i < 32; i++) {
            const int v = (int)nearbyintf(x[32 * b + i] * id);
            q[32 * b + i] = (int8_t)std::clamp(v, -128, 127);
            t[i / 16] += v;
        }
        s[b] = d;
        if (sum) { sum[2 * b] = t[0]; sum[2 * b + 1] = t[1]; }
        if (c) c[b] = 128.0f * d * (float)(t[0] + t[1]);
    }
}

static void ref_emu_quant(const float *x, EmuAct &a) {
    for (int b = 0; b < NB; b++) {
        float m = 0;
        for (int i = 0; i < 32; i++) m = std::max(m, fabsf(x[32 * b + i]));
        const float d = m / 127.0f;
        int t[2] = {0, 0};
        for (int i = 0; i < 32; i++) {
            const int v = d > 0 ? (int)nearbyintf(x[32 * b + i] / d) : 0;
            a.q[32 * b + i] = (int8_t)std::clamp(v, -128, 127);
            t[i / 16] += v;
        }
        a.d[b] = d;
        a.s16[2 * b] = t[0];
        a.s16[2 * b + 1] = t[1];
        a.c32[b] = 128 * (t[0] + t[1]);
    }
}

static float ref_gu_small(const uint8_t *row, int type, const Act &x) {
    Gu g;
    if (type == Q3K) dec_q3k(row, g);
    else dec_q2k(row, g);
    float acc[16] = {}, cor[16] = {};
    for (int sb = 0; sb < D / 256; sb++) {
        float cs[16];
        for (int l = 0; l < 16; l++) {
            const float av = g.a[sb * 16 + l], bv = type == Q3K ? av : g.b[sb * 16 + l];
            cs[l] = av * x.s16[sb * 16 + l];
            cor[l] = std::fma(bv, x.f16[sb * 16 + l], cor[l]);
        }
        for (int k = 0; k < 4; k++)
            for (int l = 0; l < 16; l++) {
                const int o = sb * 256 + 64 * k + 4 * l;
                acc[l] = std::fma((float)dot_bytes(g.w + o, x.q.data() + o, 4), cs[4 * k + l / 4], acc[l]);
            }
    }
    return type == Q3K ? sum16(acc) - 4.0f * sum16(cor) : sum16(acc) - sum16(cor);
}

static float ref_gu_iq4xs(const uint8_t *row, const Act &x) {
    uint8_t w[D];
    float ws[NB], acc[16] = {}, cor = 0;
    dec_iq4xs(row, w, ws);
    for (int b = 0; b < NB; b += 2) {
        const float c[2] = {ws[b] * x.s[b], ws[b + 1] * x.s[b + 1]};
        for (int l = 0; l < 16; l++) acc[l] = std::fma((float)dot_bytes(w + 32 * b + 4 * l, x.q.data() + 32 * b + 4 * l, 4), c[l / 8], acc[l]);
        cor = std::fma(128.0f, std::fma(c[0], x.xf[b], c[1] * x.xf[b + 1]), cor);
    }
    return sum16(acc) - cor;
}

static float ref_gu(const uint8_t *row, int type, const Act &x) { return type == IQ4XS ? ref_gu_iq4xs(row, x) : ref_gu_small(row, type, x); }

static float ref_dn(const uint8_t *row, const int8_t *hq, const float *hs, const float *hc) {
    uint8_t w[FE];
    float ws[NH], acc[16] = {}, cor = 0;
    dec_iq4nl(row, FE, w, ws);
    for (int b = 0; b < NH; b += 2) {
        const float c[2] = {ws[b] * hs[b], ws[b + 1] * hs[b + 1]};
        for (int l = 0; l < 16; l++) acc[l] = std::fma((float)dot_bytes(w + 32 * b + 4 * l, hq + 32 * b + 4 * l, 4), c[l / 8], acc[l]);
        cor = cor + std::fma(ws[b], hc[b], ws[b + 1] * hc[b + 1]);
    }
    return sum16(acc) - cor;
}

static float chunk(const float s[5][16], const float i[5][16]) {
    float v[16];
    for (int l = 0; l < 16; l++) {
        const float a = std::fma(s[4][l], i[4][l], std::fma(s[2][l], i[2][l], std::fma(s[0][l], i[0][l], 0.0f)));
        const float b = std::fma(s[3][l], i[3][l], std::fma(s[1][l], i[1][l], 0.0f));
        v[l] = a + b;
    }
    return sum16(v);
}

static float ref_emu_gu(const uint8_t *row, int type, const EmuAct &a) {
    float sv[5][16], iv[5][16];
    if (type == IQ4XS) {
        uint8_t w[D];
        float ws[NB];
        dec_iq4xs(row, w, ws);
        for (int B = 0; B < NB; B++) {
            iv[B / 16][B % 16] = (float)(dot_bytes(w + 32 * B, a.q + 32 * B, 32) - a.c32[B]);
            sv[B / 16][B % 16] = ws[B] * a.d[B];
        }
        return chunk(sv, iv);
    }
    Gu g;
    int sc[D / 16];
    float d[10], dm[10];
    for (int sb = 0; sb < 10; sb++) {
        if (type == Q3K) {
            const uint8_t *blk = row + 110 * sb;
            q3k_scales(blk, sc + 16 * sb);
            d[sb] = h2f(blk + 108);
        } else {
            const uint8_t *blk = row + 84 * sb;
            for (int l = 0; l < 16; l++) sc[16 * sb + l] = blk[l];
            d[sb] = h2f(blk + 80);
            dm[sb] = h2f(blk + 82);
        }
    }
    if (type == Q3K) dec_q3k(row, g);
    else dec_q2k(row, g);
    for (int B = 0; B < NB; B++) {
        int t = 0, u = 0;
        for (int h = 0; h < 2; h++) {
            const int G = 2 * B + h, gs = dot_bytes(g.w + 16 * G, a.q + 16 * G, 16);
            if (type == Q3K) t += (gs - 4 * a.s16[G]) * sc[G];
            else { t += gs * (sc[G] & 15); u += a.s16[G] * (sc[G] >> 4); }
        }
        const float dv = d[B / 8], xd = a.d[B];
        if (type == Q3K) { sv[B / 16][B % 16] = dv * xd; iv[B / 16][B % 16] = (float)t; }
        else { sv[B / 16][B % 16] = xd; iv[B / 16][B % 16] = std::fma(dv, (float)t, -(dm[B / 8] * (float)u)); }
    }
    return chunk(sv, iv);
}

static float ref_emu_dn(const uint8_t *row, const int8_t *hq, const float *hd, const int32_t *hc) {
    uint8_t w[FE];
    float ws[NH], v[16];
    dec_iq4nl(row, FE, w, ws);
    for (int l = 0; l < 16; l++) {
        const float a = (ws[l] * hd[l]) * (float)(dot_bytes(w + 32 * l, hq + 32 * l, 32) - hc[l]);
        const float b = l < 4 ? (ws[16 + l] * hd[16 + l]) * (float)(dot_bytes(w + 32 * (16 + l), hq + 32 * (16 + l), 32) - hc[16 + l]) : 0.0f;
        v[l] = a + b;
    }
    return sum16(v);
}

static float ref_row(const uint8_t *row, int fmt, const Act &x, const int32_t *xc) {
    float a[2] = {0, 0};
    if (fmt == IQ4XS) {
        uint8_t w[D];
        float ws[NB];
        dec_iq4xs(row, w, ws);
        for (int b = 0; b < NB; b++)
            a[b % 2] = std::fma((float)(dot_bytes(w + 32 * b, x.q.data() + 32 * b, 32) - xc[b]), ws[b] * x.s[b], a[b % 2]);
        return a[0] + a[1];
    }
    Gu g;
    if (fmt == Q3K) dec_q3k(row, g);
    else dec_q2k(row, g);
    for (int b = 0; b < NB; b++) {
        const float s0 = (float)dot_bytes(g.w + 32 * b, x.q.data() + 32 * b, 16), s1 = (float)dot_bytes(g.w + 32 * b + 16, x.q.data() + 32 * b + 16, 16);
        a[0] = std::fma(std::fma(s1, g.a[2 * b + 1], s0 * g.a[2 * b]), x.s[b], a[0]);
        const float f0 = x.s[b] * (float)x.sum[2 * b], f1 = x.s[b] * (float)x.sum[2 * b + 1];
        a[1] = std::fma(-g.b[2 * b + 1], f1, std::fma(-g.b[2 * b], f0, a[1]));
    }
    return a[0] + a[1];
}

static void ref_expert(const ColdLayer &c, int slot, const Act &x, float *y) {
    const int fmt = c.type_gu[c.grp(slot)];
    int32_t xc[NB];
    for (int b = 0; b < NB; b++) xc[b] = 128 * (x.sum[2 * b] + x.sum[2 * b + 1]);
    std::vector<float> h(FE);
    for (int r = 0; r < FE; r++) {
        const float g = ref_row(c.gu_row(false, slot, r), fmt, x, xc), u = ref_row(c.gu_row(true, slot, r), fmt, x, xc);
        h[r] = g / (1.0f + expf(-g)) * u;
    }
    int8_t hq[FE];
    float hs[NH];
    int32_t hc[NH];
    for (int b = 0; b < NH; b++) {
        float m = 0;
        for (int r = 0; r < 32; r++) m = std::max(m, fabsf(h[32 * b + r]));
        const float d = m / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        int s = 0;
        for (int r = 0; r < 32; r++) {
            const int q = (int)lrintf(h[32 * b + r] * id);
            s += q;
            hq[32 * b + r] = (int8_t)q;
        }
        hs[b] = d;
        hc[b] = 128 * s;
    }
    for (int r = 0; r < D; r++) {
        uint8_t w[FE];
        float ws[NH], a[2] = {0, 0};
        dec_iq4nl(c.dn_row(slot, r), FE, w, ws);
        for (int b = 0; b < NH; b++) a[b % 2] = std::fma((float)(dot_bytes(w + 32 * b, hq + 32 * b, 32) - hc[b]), ws[b] * hs[b], a[b % 2]);
        y[r] = a[0] + a[1];
    }
}

static void random_row(int type, int n, uint8_t *row) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(-0.03f, 0.03f);
    const size_t len = row_bytes(type, n);
    for (size_t i = 0; i < len; i++) row[i] = (uint8_t)byte(rng);
    if (type == Q3K) for (int sb = 0; sb < n / 256; sb++) put_h(row + 110 * sb + 108, scale(rng));
    if (type == Q2K) for (int sb = 0; sb < n / 256; sb++) { put_h(row + 84 * sb + 80, scale(rng)); put_h(row + 84 * sb + 82, scale(rng)); }
    if (type == IQ4XS) for (int sb = 0; sb < n / 256; sb++) put_h(row + 136 * sb, scale(rng));
    if (type == IQ4NL) for (int b = 0; b < n / 32; b++) put_h(row + 18 * b, scale(rng));
}

static void extreme_row(int type, int n, uint8_t *row, int hi) {
    random_row(type, n, row);
    if (type == Q3K) for (int sb = 0; sb < n / 256; sb++) memset(row + 110 * sb, hi ? 0xff : 0, 96);
    if (type == Q2K) for (int sb = 0; sb < n / 256; sb++) memset(row + 84 * sb + 16, hi ? 0xff : 0, 64);
    if (type == IQ4XS) for (int sb = 0; sb < n / 256; sb++) memset(row + 136 * sb + 8, hi ? 0xff : 0, 128);
    if (type == IQ4NL) for (int b = 0; b < n / 32; b++) memset(row + 18 * b + 2, hi ? 0xff : 0, 16);
}

static std::vector<uint8_t> rows_of(int type, int n, int count) {
    const size_t rb = row_bytes(type, n);
    std::vector<uint8_t> out(rb * count);
    std::normal_distribution<float> nd(0.f, 0.03f);
    std::vector<float> w((size_t)n), im(n, 1.0f);
    for (int r = 0; r < count; r++) {
        uint8_t *row = out.data() + rb * r;
        if (r % 4 == 0) {
            for (auto &v : w) v = nd(rng);
            ggml_quantize_chunk((ggml_type)type, w.data(), row, 0, 1, n, im.data());
        } else if (r % 4 == 1) extreme_row(type, n, row, r % 8 == 1);
        else random_row(type, n, row);
    }
    return out;
}

static std::vector<Act> activations() {
    std::vector<Act> acts;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> sd(0.001f, 0.05f);
    std::uniform_int_distribution<int> any(-128, 127);
    std::vector<float> s(NB);
    for (int k = 0; k < 3; k++) {
        std::vector<float> x(D);
        for (auto &v : x) v = nd(rng) * (k == 2 ? 30.0f : 1.0f);
        std::vector<int8_t> q(D);
        ref_quant(x.data(), D, q.data(), s.data(), nullptr, nullptr);
        acts.push_back(act_from(q, s, "quantized activations"));
    }
    const int8_t fill[3] = {127, -128, -127};
    for (int8_t f : fill) {
        for (auto &v : s) v = sd(rng);
        acts.push_back(act_from(std::vector<int8_t>(D, f), s, "every activation " + std::to_string(f)));
    }
    std::vector<int8_t> q(D);
    for (auto &v : q) v = (int8_t)any(rng);
    for (auto &v : s) v = sd(rng);
    acts.push_back(act_from(q, s, "random bytes"));
    return acts;
}

struct Tally {
    int cases = 0, bad = 0;
    void add(bool ok) { cases++; bad += !ok; }
};

static int fails = 0;
static void report(const char *what, const char *path, const Tally &t) {
    const bool ok = t.cases > 0 && t.bad == 0;
    char res[64];
    if (t.bad) snprintf(res, sizeof res, "%d differ from", t.bad);
    else snprintf(res, sizeof res, "all the same as");
    printf("%-60s %-13s %4d cases, %s the scalar reference %s\n", what, path, t.cases, res, ok ? "ok" : "FAIL");
    fails += !ok;
}

struct Expert {
    std::vector<uint8_t> gate, up, down;
    ColdLayer c;
};

static void make_expert(Expert &e, int type) {
    e.gate = rows_of(type, D, FE);
    e.up = rows_of(type, D, FE);
    e.down = rows_of(IQ4NL, FE, D);
    e.c.gate[0] = e.gate.data(); e.c.up[0] = e.up.data(); e.c.down = e.down.data();
    e.c.type_gu[0] = type; e.c.row_gu[0] = row_bytes(type, D);
    e.c.type_dn = IQ4NL; e.c.row_dn = row_bytes(IQ4NL, FE);
    e.c.n0 = 1; e.c.n = 1;
}

static uint64_t mix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

int main() {
    {
        uint8_t blk[18] = {};
        put_h(blk, 1.0f);
        for (int j = 0; j < 8; j++) blk[2 + j] = (uint8_t)(2 * j | (2 * j + 1) << 4);
        float y[32];
        ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(blk, y, 32);
        for (int j = 0; j < 8; j++) { KV[2 * j] = (uint8_t)((int)y[j] + 128); KV[2 * j + 1] = (uint8_t)((int)y[j + 16] + 128); }
    }
    const int types[3] = {Q3K, Q2K, IQ4XS};
    const char *tname[3] = {"Q3_K", "Q2_K", "IQ4_XS"};
    constexpr int R = 48;
    std::vector<uint8_t> gu_rows[3], dn_rows = rows_of(IQ4NL, FE, R);
    for (int t = 0; t < 3; t++) gu_rows[t] = rows_of(types[t], D, R);
    {
        int bad = 0, n = 0;
        auto check = [&](int type, const uint8_t *row, int K) {
            std::vector<float> y(K), v(K);
            ggml_get_type_traits((ggml_type)type)->to_float(row, y.data(), K);
            uint8_t w[D];
            float ws[NB];
            Gu g;
            if (type == IQ4NL) { dec_iq4nl(row, K, w, ws); for (int i = 0; i < K; i++) v[i] = ws[i / 32] * (float)(w[i] - 128); }
            else if (type == IQ4XS) { dec_iq4xs(row, w, ws); for (int i = 0; i < K; i++) v[i] = ws[i / 32] * (float)(w[i] - 128); }
            else if (type == Q3K) { dec_q3k(row, g); for (int i = 0; i < K; i++) v[i] = g.a[i / 16] * (float)(g.w[i] - 4); }
            else { dec_q2k(row, g); for (int i = 0; i < K; i++) v[i] = g.a[i / 16] * (float)g.w[i] - g.b[i / 16]; }
            for (int i = 0; i < K; i++, n++) bad += fabsf(v[i] - y[i]) > 1e-6f * std::max(1.0f, fabsf(y[i]));
        };
        for (int t = 0; t < 3; t++) for (int r = 0; r < R; r++) check(types[t], gu_rows[t].data() + row_bytes(types[t], D) * r, D);
        for (int r = 0; r < R; r++) check(IQ4NL, dn_rows.data() + row_bytes(IQ4NL, FE) * r, FE);
        printf("%-60s %-13s %5d weights, %s\n", "the reference decodes the weights as ggml does", "", n, bad ? "FAIL" : "ok");
        fails += bad != 0 || n == 0;
    }
    int pmax;
    {
        uint8_t blk[136];
        extreme_row(IQ4XS, 256, blk, 1);
        uint8_t w[32];
        dec_iq4(blk + 8, w);
        pmax = 2 * w[0] * 127;
    }
    printf("IQ4 weights of %d against activations of 127 sum in pairs to %d, past the 32767 a 16-bit intermediate holds\n", KV[15], pmax);
    const std::vector<Act> acts = activations();
    const int A = (int)acts.size();
    std::vector<EmuAct> eacts(A);
    for (int a = 0; a < A; a++) {
        const Act &x = acts[a];
        memcpy(eacts[a].q, x.q.data(), D);
        for (int b = 0; b < NB; b++) { eacts[a].d[b] = x.s[b]; eacts[a].c32[b] = 128 * (x.sum[2 * b] + x.sum[2 * b + 1]); }
        memcpy(eacts[a].s16, x.sum.data(), sizeof eacts[a].s16);
    }
    constexpr int NX = 4;
    std::vector<float> xf((size_t)NX * D);
    {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (int i = 0; i < NX * D; i++) xf[i] = nd(rng) * (i < D ? 1.0f : i < 2 * D ? 1e-3f : 40.0f);
        for (int i = 3 * D; i < 3 * D + 64; i++) xf[i] = 0.0f;
        for (int i = 3 * D + 64; i < 3 * D + 96; i++) xf[i] = (float)(i % 7 - 3) * 0.5f;
    }
    struct QRef { int8_t q[D], hq[FE]; float s[NB], hs[NH], hc[NH]; int32_t sum[D / 16]; EmuAct e; };
    std::vector<QRef> qref(NX);
    for (int k = 0; k < NX; k++) {
        const float *x = xf.data() + (size_t)k * D;
        ref_quant(x, D, qref[k].q, qref[k].s, qref[k].sum, nullptr);
        ref_quant(x, FE, qref[k].hq, qref[k].hs, nullptr, qref[k].hc);
        ref_emu_quant(x, qref[k].e);
    }
    std::vector<float> want_gu[3], want_emu[3];
    for (int t = 0; t < 3; t++)
        for (int r = 0; r < R; r++)
            for (int a = 0; a < A; a++) {
                const uint8_t *row = gu_rows[t].data() + row_bytes(types[t], D) * r;
                want_gu[t].push_back(ref_gu(row, types[t], acts[a]));
                want_emu[t].push_back(ref_emu_gu(row, types[t], eacts[a]));
            }
    struct DnIn { float hs[NH], hc[NH]; int32_t hci[NH]; const int8_t *hq; };
    std::vector<DnIn> dn_in;
    std::vector<float> want_dn, want_edn;
    for (int r = 0; r < R; r++)
        for (int a = 0; a < A; a++) {
            DnIn in;
            in.hq = acts[a].q.data() + FE * (r % 4);
            for (int b = 0; b < NH; b++) {
                int s = 0;
                for (int i = 0; i < 32; i++) s += in.hq[32 * b + i];
                in.hs[b] = acts[a].s[b + NH * (r % 4)];
                in.hc[b] = 128.0f * in.hs[b] * (float)s;
                in.hci[b] = 128 * s;
            }
            const uint8_t *row = dn_rows.data() + row_bytes(IQ4NL, FE) * r;
            want_dn.push_back(ref_dn(row, in.hq, in.hs, in.hc));
            want_edn.push_back(ref_emu_dn(row, in.hq, in.hs, in.hci));
            dn_in.push_back(in);
        }
    std::vector<Expert> experts(3);
    for (int t = 0; t < 3; t++) make_expert(experts[t], types[t]);
    const int counts[3] = {5, 13, 21};
    std::vector<std::vector<float>> want_y(3 * A, std::vector<float>(D));
    for (int t = 0; t < 3; t++)
        for (int a = 0; a < A; a++) ref_expert(experts[t].c, 0, acts[a], want_y[t * A + a].data());
    uint64_t hashes[EXACT_PATHS] = {};
    int ran = 0;
    for (int p = 0; p < EXACT_PATHS; p++) {
        const char *pn = exact_path_name(p);
        if (!exact_use_path(p)) { printf("%-60s %-13s not run: this CPU lacks it\n", "every exact CPU function", pn); continue; }
        ran++;
        ExactScratch S;
        uint64_t h = 1469598103934665603ull;
        Tally tq, th, te;
        for (int k = 0; k < NX; k++) {
            const float *x = xf.data() + (size_t)k * D;
            int8_t q[D];
            float s[NB], c[NH];
            int32_t sum[D / 16];
            exact_quant_x(x, q, s, sum);
            tq.add(same(q, qref[k].q, D) && same(s, qref[k].s, sizeof s) && same(sum, qref[k].sum, sizeof sum));
            h = mix(mix(mix(h, q, D), s, sizeof s), sum, sizeof sum);
            exact_quant_h(x, q, s, c);
            th.add(same(q, qref[k].hq, FE) && same(s, qref[k].hs, NH * 4) && same(c, qref[k].hc, sizeof c));
            h = mix(mix(mix(h, q, FE), s, NH * 4), c, sizeof c);
            EmuAct e;
            emu_quant_x(x, e);
            te.add(same(e.q, qref[k].e.q, D) && same(e.d, qref[k].e.d, sizeof e.d) && same(e.s16, qref[k].e.s16, sizeof e.s16) &&
                   same(e.c32, qref[k].e.c32, sizeof e.c32));
            h = mix(h, &e, sizeof e);
        }
        report("activations quantized for gate/up", pn, tq);
        report("SiLU output quantized for down", pn, th);
        report("activations quantized for the GPU's arithmetic", pn, te);
        for (int t = 0; t < 3; t++) {
            Tally tg, tp, tm;
            const size_t rb = row_bytes(types[t], D);
            for (int r = 0; r < R; r++)
                for (int a = 0; a < A; a++) {
                    const uint8_t *row = gu_rows[t].data() + rb * r;
                    const Act &x = acts[a];
                    const float want = want_gu[t][r * A + a];
                    const float got = exact_dot_gu1(row, types[t], x.q.data(), x.s.data(), x.sum.data(), x.s16.data(), x.f16.data(), S);
                    tg.add(same(&got, &want, 4));
                    h = mix(h, &got, 4);
                    if (types[t] == IQ4XS) {
                        float o[2];
                        exact_dot_gu2(row, gu_rows[t].data() + rb * ((r + 1) % R), x.q.data(), x.s.data(), x.xf.data(), o);
                        tp.add(same(&o[0], &want, 4) && same(&o[1], &want_gu[t][((r + 1) % R) * A + a], 4));
                        h = mix(h, o, 8);
                    }
                    const float e = emu_gu_row(row, types[t], eacts[a], S);
                    tm.add(same(&e, &want_emu[t][r * A + a], 4));
                    h = mix(h, &e, 4);
                }
            std::string what = std::string("gate/up row dot, ") + tname[t] + ", one token";
            report(what.c_str(), pn, tg);
            if (types[t] == IQ4XS) report("gate/up row dots, IQ4_XS, two rows at once", pn, tp);
            what = std::string("gate/up row dot in the GPU's arithmetic, ") + tname[t];
            report(what.c_str(), pn, tm);
        }
        {
            Tally tn, tm;
            for (int r = 0; r < R; r++)
                for (int a = 0; a < A; a++) {
                    const DnIn &in = dn_in[r * A + a];
                    const uint8_t *row = dn_rows.data() + row_bytes(IQ4NL, FE) * r;
                    const float got = exact_dot_dn1(row, in.hq, in.hs, in.hc), e = emu_dn_row(row, in.hq, in.hs, in.hci, S);
                    tn.add(same(&got, &want_dn[r * A + a], 4));
                    tm.add(same(&e, &want_edn[r * A + a], 4));
                    h = mix(mix(h, &got, 4), &e, 4);
                }
            report("down row dot, IQ4_NL, one token", pn, tn);
            report("down row dot in the GPU's arithmetic, IQ4_NL", pn, tm);
        }
        for (int t = 0; t < 3; t++) {
            Tally tx;
            for (int n : counts) {
                std::vector<const int8_t *> pq(n);
                std::vector<const float *> ps(n);
                std::vector<const int32_t *> pc(n);
                std::vector<float> w(n);
                std::vector<uint16_t> out((size_t)n * D), ref(D);
                std::vector<uint16_t *> po(n);
                for (int i = 0; i < n; i++) {
                    const Act &x = acts[(i + t) % A];
                    pq[i] = x.q.data(); ps[i] = x.s.data(); pc[i] = x.sum.data();
                    w[i] = 0.05f + 0.01f * (float)i;
                    po[i] = out.data() + (size_t)i * D;
                }
                expert_exact(experts[t].c, 0, n, pq.data(), ps.data(), pc.data(), w.data(), po.data(), S);
                h = mix(h, out.data(), out.size() * 2);
                for (int i = 0; i < n; i++) {
                    const float *y = want_y[t * A + (i + t) % A].data();
                    for (int r = 0; r < D; r++) ref[r] = ggml_fp32_to_fp16(y[r] * w[i]);
                    tx.add(same(po[i], ref.data(), D * 2));
                }
            }
            const std::string what = std::string("prefill expert, ") + tname[t] + " gate/up, IQ4_NL down, 5/13/21 tokens";
            report(what.c_str(), pn, tx);
        }
        hashes[p] = h;
    }
    int first = -1, differ = 0;
    for (int p = 0; p < EXACT_PATHS; p++) {
        if (!hashes[p]) continue;
        if (first < 0) first = p;
        else differ += hashes[p] != hashes[first];
    }
    const bool ok = exact_path_supported(EXACT_AVX2) && ran > 0 && differ == 0;
    printf("%-60s %-13s %d paths, %s %s\n", "every path's outputs against the others'", "", ran, differ ? "not all the same" : "the same to the bit",
           ok ? "ok" : "FAIL");
    fails += !ok;
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
