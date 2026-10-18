// The host sampler, shared by both engines. A row holds the likeliest SAMPLE_N tokens with their logits; penalties
// (presence, DRY) adjust them, then top K, temperature, top P and min P act in the order a request picks.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

constexpr int SAMPLE_N = 64;                         // = TOPK of both engines: candidates per row
enum Stage : int { STAGE_TOP_K, STAGE_TEMPERATURE, STAGE_TOP_P, STAGE_MIN_P, STAGES };

// Also the C API's layout (ctypes mirrors it in kiln/engine.py): plain fields only, in this order.
struct Sampling {
    float temperature = 1.0f, top_p = 0.95f, min_p = 0.0f, presence_penalty = 0.0f;
    int top_k = 20, n_draft = 3;
    int ban = -1;                                    // a token never drawn (-1: none): <tool_call> when a request offers no tools
    float dry_multiplier = 0.0f, dry_base = 1.75f;   // DRY, off at 0
    int dry_allowed_length = 2;
    int order[STAGES] = {STAGE_TOP_K, STAGE_TEMPERATURE, STAGE_TOP_P, STAGE_MIN_P};
    int lookup = 0;                                  // drafts copied from earlier in the conversation, at most (0: off)
    int halt = -1;                                   // a token that ends a step once emitted, as an end of turn does (-1: none):
                                                     // </think> before an answer kiln constrains token by token (kiln_draw)
};

// The current reply's tokens, in order (DRY) and as a set (presence penalty: a lookup, not a scan of the reply).
struct Seen {
    std::vector<int> list;
    std::vector<uint8_t> mask;
    void push(int t) {
        list.push_back(t);
        if ((size_t)t >= mask.size()) mask.resize((size_t)t + 4097);
        mask[t] = 1;
    }
    void clear() {
        for (int t : list) mask[t] = 0;
        list.clear();
    }
    bool has(int t) const { return (size_t)t < mask.size() && mask[t]; }
};

// DRY ("don't repeat yourself", p-e-w): a candidate that would continue a run of tokens the context already holds is
// penalized by multiplier * base^(run - allowed length), where run counts the tokens before it that match the end
// of the context. Sequence breakers (tokens holding a newline, ':', '"' or '*', set by the server) end a run. The
// context is the prompt's tail (Seq::ctx) followed by the reply so far.
constexpr size_t DRY_RANGE = 32768;                  // tokens looked back over
constexpr int DRY_MAX_RUN = 48;                      // longer runs are penalized like this one (already ~1e11)
inline std::vector<uint8_t> dry_breakers;            // by token id

inline void dry_runs(const std::vector<int> &ctx, const std::vector<int> &reply, std::vector<std::pair<int, int>> &runs) {
    runs.clear();
    const size_t nc = ctx.size(), n = nc + reply.size();
    if (n < 2) return;
    auto at = [&](size_t i) { return i < nc ? ctx[i] : reply[i - nc]; };
    auto breaker = [](int t) { return (size_t)t < dry_breakers.size() && dry_breakers[t]; };
    const size_t lo = n > DRY_RANGE ? n - DRY_RANGE : 0;
    const int last = at(n - 1);
    if (breaker(last)) return;
    for (size_t i = n - 1; i-- > lo;) {              // each earlier occurrence of the last token
        if (at(i) != last) continue;
        const int next = at(i + 1);
        if (breaker(next)) continue;
        int run = 1;
        while (run < DRY_MAX_RUN && i >= lo + run) {
            const int a = at(i - run);
            if (a != at(n - 1 - run) || breaker(a)) break;
            run++;
        }
        runs.push_back({next, run});
    }
}

struct Dist { int n = 0; int id[SAMPLE_N]; float p[SAMPLE_N]; };

inline Dist make_dist(const float *v, const int *ids, const Sampling &sp, const Seen &seen, const std::vector<int> &ctx) {
    float pen[SAMPLE_N] = {};
    if (sp.dry_multiplier > 0) {
        thread_local std::vector<std::pair<int, int>> runs;
        dry_runs(ctx, seen.list, runs);
        for (auto [tok, run] : runs)
            if (run >= sp.dry_allowed_length)
                for (int i = 0; i < SAMPLE_N; i++)
                    if (ids[i] == tok) pen[i] = std::max(pen[i], sp.dry_multiplier * powf(sp.dry_base, (float)(run - sp.dry_allowed_length)));
    }
    std::pair<float, int> c[SAMPLE_N];
    int nc = 0;
    for (int i = 0; i < SAMPLE_N; i++) {
        if (ids[i] == sp.ban) continue;                 // not a candidate at all (so not a fallback either)
        float x = v[i] - pen[i];
        if (sp.presence_penalty != 0 && seen.has(ids[i])) x -= sp.presence_penalty;
        c[nc++] = {x, ids[i]};
    }
    std::sort(c, c + nc, [](auto &a, auto &b) { return a.first > b.first; });
    Dist d;
    if (sp.temperature <= 0) { d.n = 1; d.id[0] = c[0].second; d.p[0] = 1; return d; }
    // Every stage keeps a prefix of the sorted candidates, so `n` says what is left. Top P and min P judge the
    // probabilities at the temperature applied so far (1 until the temperature stage); the reply is drawn at the
    // temperature. P holds the probabilities at temperature pt, normalized over the first pn candidates.
    int n = nc, pn = 0;
    float t = 1.0f, pt = -1.0f, P[SAMPLE_N];
    auto probs = [&](float temp) {
        double sum = 0;
        for (int i = 0; i < n; i++) { P[i] = expf((c[i].first - c[0].first) / temp); sum += P[i]; }
        for (int i = 0; i < n; i++) P[i] /= sum;
        pt = temp, pn = n;
    };
    for (int s = 0; s < STAGES; s++) switch (sp.order[s]) {
        case STAGE_TOP_K: n = std::max(1, std::min(sp.top_k > 0 ? sp.top_k : n, n)); break;
        case STAGE_TEMPERATURE: t = sp.temperature; break;
        case STAGE_TOP_P: {
            if (pt != t || pn != n) probs(t);
            double cum = 0;
            for (int i = 0; i < n; i++) { cum += P[i]; if (cum >= sp.top_p) { n = i + 1; break; } }
            break;
        }
        case STAGE_MIN_P: {                             // a ratio to the likeliest: any normalization will do
            if (pt != t) probs(t);
            int m = 1;
            while (m < n && P[m] >= sp.min_p * P[0]) m++;
            n = m;
            break;
        }
    }
    if (pt != sp.temperature) probs(sp.temperature);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += P[i];
    for (int i = 0; i < n; i++) { d.p[i] = P[i] / sum; d.id[i] = c[i].second; }
    d.n = n;
    return d;
}

inline int draw(const Dist &d, std::mt19937_64 &rng) {
    double u = std::uniform_real_distribution<double>(0, 1)(rng), c = 0;
    for (int i = 0; i < d.n; i++) { c += d.p[i]; if (u < c) return d.id[i]; }
    return d.id[d.n - 1];
}

inline float prob(const Dist &d, int tok) {
    for (int i = 0; i < d.n; i++) if (d.id[i] == tok) return d.p[i];
    return 0;
}

inline int draw_without(Dist d, int tok, std::mt19937_64 &rng) {   // residual distribution after rejecting tok
    double sum = 0;
    for (int i = 0; i < d.n; i++) { if (d.id[i] == tok) d.p[i] = 0; sum += d.p[i]; }
    if (sum <= 0) return d.id[0];
    for (int i = 0; i < d.n; i++) d.p[i] /= sum;
    return draw(d, rng);
}
