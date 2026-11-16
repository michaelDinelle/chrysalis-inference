#include "../../engine/gpu/27b/kernels.h"
#include <cmath>
#include <random>
#include <vector>
using std::vector;

static float q8(float x, float d) { return d > 0 ? d * std::nearbyint(x / d) : 0.0f; }
static void fwht(float *x) {
    for (int h = 1; h < HD; h <<= 1)
        for (int i = 0; i < HD; i++) if (!(i & h)) { float a = x[i], b = x[i | h]; x[i] = a + b; x[i | h] = a - b; }
    for (int i = 0; i < HD; i++) x[i] *= 0.0625f;
}
static float round_f16(float x) { return (float)(_Float16)x; }

static vector<int> mrope_sections() {
    vector<int> sec(ROT / 2, 0);
    const int section[3] = {11, 11, 10};
    for (int dim = 1; dim <= 2; dim++)
        for (int f = dim; f < section[dim] * 3; f += 3) sec[f] = dim;
    return sec;
}

struct Ref {
    vector<float> K, V;
    void prep(const float *qg, const float *k, const float *v, const float *qw, const float *kw, int T, int pos0,
              const int *rpos, vector<float> &Q) {
        static const vector<int> sec = mrope_sections();
        auto rope_norm = [&](const float *x, const float *w, int t, float *o) {
            double ss = 0; for (int i = 0; i < HD; i++) ss += x[i] * x[i];
            float inv = 1.0f / sqrtf(ss / HD + EPS), y[HD];
            for (int i = 0; i < HD; i++) y[i] = x[i] * inv * w[i];
            for (int i = 0; i < HD; i++) o[i] = y[i];
            for (int i = 0; i < ROT / 2; i++) {
                const int pos = rpos ? rpos[t * 3 + sec[i]] : pos0 + t;
                const float turn = pos * powf(ROPE_BASE, -2.0f * i / ROT), s = sinf(turn), c = cosf(turn);
                o[i] = y[i] * c - y[i + ROT / 2] * s; o[i + ROT / 2] = y[i + ROT / 2] * c + y[i] * s;
            }
        };
        K.resize((size_t)(pos0 + T) * NKV * HD); V.resize(K.size()); Q.resize((size_t)T * NH * HD);
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float o[HD]; rope_norm(qg + ((size_t)t * NH + h) * 2 * HD, qw, t, o);
                for (int i = 0; i < HD; i++) Q[((size_t)t * NH + h) * HD + i] = round_f16(o[i] * 0.0625f);
            }
            for (int h = 0; h < NKV; h++) {
                float o[HD]; rope_norm(k + ((size_t)t * NKV + h) * HD, kw, t, o);
                float vv[HD]; for (int i = 0; i < HD; i++) vv[i] = v[((size_t)t * NKV + h) * HD + i];
                fwht(vv);
                for (int b = 0; b < HD / 32; b++) {
                    float mk = 0, mv = 0;
                    for (int i = 0; i < 32; i++) { mk = fmaxf(mk, fabsf(o[b * 32 + i])); mv = fmaxf(mv, fabsf(vv[b * 32 + i])); }
                    float dk = round_f16(mk / 127), dv = round_f16(mv / 15), dk0 = mk / 127, dv0 = mv / 15;
                    for (int i = 0; i < 32; i++) {
                        size_t at = ((size_t)(pos0 + t) * NKV + h) * HD + b * 32 + i;
                        K[at] = dk * (dk0 > 0 ? std::nearbyint(o[b * 32 + i] / dk0) : 0);
                        V[at] = dv * (dv0 > 0 ? std::nearbyint(vv[b * 32 + i] / dv0) : 0);
                    }
                }
            }
        }
    }
    double score(const vector<float> &Q, int t, int h, int j) const {
        double d = 0;
        for (int i = 0; i < HD; i++) d += Q[((size_t)t * NH + h) * HD + i] * round_f16(K[((size_t)j * NKV + h / 6) * HD + i]);
        return d;
    }
    void attend(const vector<float> &Q, const float *qg, int T, int pos0, vector<float> &out) {
        out.assign((size_t)T * NH * HD, 0);
        for (int t = 0; t < T; t++) for (int h = 0; h < NH; h++) {
            const int g = h / 6, n = pos0 + t + 1;
            vector<double> s(n); double mx = -1e30;
            for (int j = 0; j < n; j++) {
                double d = 0;
                for (int i = 0; i < HD; i++) d += Q[((size_t)t * NH + h) * HD + i] * round_f16(K[((size_t)j * NKV + g) * HD + i]);
                s[j] = d; mx = fmax(mx, d);
            }
            double L = 0; for (auto &x : s) { x = exp(x - mx); L += x; }
            float o[HD];
            for (int i = 0; i < HD; i++) {
                double acc = 0; for (int j = 0; j < n; j++) acc += s[j] * round_f16(V[((size_t)j * NKV + g) * HD + i]);
                o[i] = acc / L;
            }
            fwht(o);
            for (int i = 0; i < HD; i++) {
                float gate = qg[((size_t)t * NH + h) * 2 * HD + HD + i];
                out[((size_t)t * NH + h) * HD + i] = o[i] / (1 + expf(-gate));
            }
        }
    }
};

int main() {
    hipStream_t st; CK(hipStreamCreate(&st));
    std::mt19937 rng(7); std::normal_distribution<float> nd;
    const int maxpos = 36864, npages = maxpos / PAGE;
    uint8_t *pool; CK(hipMalloc(&pool, KV_PAGE * npages));
    vector<int> pages(npages); for (int i = 0; i < npages; i++) pages[i] = npages - 1 - i;
    int *dpages; CK(hipMalloc(&dpages, npages * 4)); CK(hipMemcpy(dpages, pages.data(), npages * 4, hipMemcpyHostToDevice));
    KVRef kv{pool, dpages};
    vector<float> qw(HD), kw(HD); for (auto &x : qw) x = 1 + 0.1f * nd(rng); for (auto &x : kw) x = 1 + 0.1f * nd(rng);
    float *dqw, *dkw; CK(hipMalloc(&dqw, HD * 4)); CK(hipMalloc(&dkw, HD * 4));
    CK(hipMemcpy(dqw, qw.data(), HD * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dkw, kw.data(), HD * 4, hipMemcpyHostToDevice));
    Ref ref; int fails = 0, pos = 0;
    struct Case { int T; const char *name; bool check = true; bool mrope = false; };
    vector<Case> cases = {{100, "prefill 100"}, {22, "prefill +22"}};
    for (int i = 0; i < 12; i++) cases.push_back({1, "decode into a split"});
    for (Case c : {Case{16, "prefill +16"}, Case{1, "decode"}, Case{4, "verify 4"}, Case{8, "verify 8"}, Case{1, "decode"},
                   Case{24, "M-RoPE picture", true, true}, Case{1, "decode after it"}}) cases.push_back(c);
    for (int i = 0; i < 16; i++) cases.push_back({2048, "prefill 2048", false});
    for (int t : {1, 2, 3, 4, 8, 5, 1}) cases.push_back({t, "deep decode/verify"});
    cases.push_back({9, "deep append"});
    float *po, *pml;
    CK(hipMalloc(&po, (size_t)MAX_SPLITS * 8 * NH * HD * 4)); CK(hipMalloc(&pml, (size_t)MAX_SPLITS * 8 * NH * 2 * 4));
    for (auto c : cases) {
        const int T = c.T;
        vector<float> qg((size_t)T * NH * 2 * HD), k((size_t)T * NKV * HD), v(k.size());
        for (auto &x : qg) x = nd(rng); for (auto &x : k) x = nd(rng); for (auto &x : v) x = nd(rng);
        float *dqg, *dk, *dv; half *Q16, *oh;
        CK(hipMalloc(&dqg, qg.size() * 4)); CK(hipMalloc(&dk, k.size() * 4)); CK(hipMalloc(&dv, v.size() * 4));
        CK(hipMalloc(&Q16, (size_t)T * NH * HD * 2)); CK(hipMalloc(&oh, (size_t)T * NH * HD * 2));
        CK(hipMemcpy(dqg, qg.data(), qg.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice));
        vector<int> rp;
        int *drp = nullptr;
        if (c.mrope) {
            for (int t = 0; t < T; t++) {
                const int p = pos + t, img = t - 2;
                if (img < 0) rp.insert(rp.end(), {p, p, p});
                else rp.insert(rp.end(), {pos + 2, pos + 2 + 300 * (img / 5), pos + 2 + 700 * (img % 5)});
            }
            CK(hipMalloc(&drp, rp.size() * 4)); CK(hipMemcpy(drp, rp.data(), rp.size() * 4, hipMemcpyHostToDevice));
        }
        attn_prep(dqg, dk, dv, dqw, dkw, T, pos, pos, drp, kv, 5, Q16, st);
        AttnArgs a; a.Q16 = Q16; a.kv = kv; a.layer = 5; a.T = T; a.pos0 = pos; a.qg = dqg; a.part_o = po; a.part_ml = pml;
        QAct none{nullptr, nullptr};
        attention(a, oh, none, st);
        CK(hipStreamSynchronize(st));
        vector<_Float16> got((size_t)T * NH * HD); CK(hipMemcpy(got.data(), oh, got.size() * 2, hipMemcpyDeviceToHost));
        vector<float> Q, want; ref.prep(qg.data(), k.data(), v.data(), qw.data(), kw.data(), T, pos, c.mrope ? rp.data() : nullptr, Q);
        if (drp) CK(hipFree(drp));
        if (!c.check) { pos += T; CK(hipFree(dqg)); CK(hipFree(dk)); CK(hipFree(dv)); CK(hipFree(Q16)); CK(hipFree(oh)); continue; }
        ref.attend(Q, qg.data(), T, pos, want);
        double e2 = 0, r2 = 0, emax = 0;
        for (size_t i = 0; i < want.size(); i++) { double e = (double)got[i] - want[i]; e2 += e * e; r2 += want[i] * want[i]; emax = fmax(emax, fabs(e)); }
        double peak = 0;
        if (T <= 8) {
            const int len = decode_split(pos + T), ns = (pos + T + len - 1) / len;
            vector<float> ml((size_t)ns * T * NH * 2); CK(hipMemcpy(ml.data(), pml, ml.size() * 4, hipMemcpyDeviceToHost));
            for (int sp = 0; sp < ns; sp++) for (int t = 0; t < T; t++) for (int h = 0; h < NH; h++) {
                double mx = -INFINITY;
                for (int j = sp * len; j < std::min(pos + t + 1, (sp + 1) * len); j++) mx = fmax(mx, ref.score(Q, t, h, j));
                const double m = ml[((size_t)sp * T * NH + t * NH + h) * 2];
                peak = fmax(peak, mx == -INFINITY ? (m == -INFINITY ? 0.0 : 1e9) : fabs(m - mx));
            }
        }
        const double rel = sqrt(e2 / r2); const bool ok = rel < 5e-3 && peak < 0.02;
        printf("%-16s T=%-3d pos0=%-3d rel %.2e max %.2e row maxima off by %.1e %s\n", c.name, T, pos, rel, emax, peak, ok ? "ok" : "FAIL");
        fails += !ok; pos += T;
        CK(hipFree(dqg)); CK(hipFree(dk)); CK(hipFree(dv)); CK(hipFree(Q16)); CK(hipFree(oh));
    }
    printf(fails ? "%d FAILURES\n" : "all ok\n", fails);
    return fails != 0;
}
