#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

constexpr int SAMPLE_N = 64;
enum Stage : int { STAGE_TOP_K, STAGE_TEMPERATURE, STAGE_TOP_P, STAGE_MIN_P, STAGES };

struct Sampling {
    float temperature = 1.0f, top_p = 0.95f, min_p = 0.0f, presence_penalty = 0.0f;
    int top_k = 20, n_draft = 3;
    int ban = -1;
    float dry_multiplier = 0.0f, dry_base = 1.75f;
    int dry_allowed_length = 2;
    int order[STAGES] = {STAGE_TOP_K, STAGE_TEMPERATURE, STAGE_TOP_P, STAGE_MIN_P};
    int lookup = 0;
    int halt = -1;
};

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

constexpr size_t DRY_RANGE = 32768;
constexpr int DRY_MAX_RUN = 48;
inline std::vector<uint8_t> dry_breakers;

inline void dry_runs(const std::vector<int> &ctx, const std::vector<int> &reply, std::vector<std::pair<int, int>> &runs) {
    runs.clear();
    const size_t nc = ctx.size(), n = nc + reply.size();
    if (n < 2) return;
    auto at = [&](size_t i) { return i < nc ? ctx[i] : reply[i - nc]; };
    auto breaker = [](int t) { return (size_t)t < dry_breakers.size() && dry_breakers[t]; };
    const size_t lo = n > DRY_RANGE ? n - DRY_RANGE : 0;
    const int last = at(n - 1);
    if (breaker(last)) return;
    for (size_t i = n - 1; i-- > lo;) {
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
        if (ids[i] == sp.ban) continue;
        float x = v[i] - pen[i];
        if (sp.presence_penalty != 0 && seen.has(ids[i])) x -= sp.presence_penalty;
        c[nc++] = {x, ids[i]};
    }
    std::sort(c, c + nc, [](auto &a, auto &b) { return a.first > b.first; });
    Dist d;
    if (sp.temperature <= 0) { d.n = 1; d.id[0] = c[0].second; d.p[0] = 1; return d; }
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
        case STAGE_MIN_P: {
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

inline int draw_without(Dist d, int tok, std::mt19937_64 &rng) {
    double sum = 0;
    for (int i = 0; i < d.n; i++) { if (d.id[i] == tok) d.p[i] = 0; sum += d.p[i]; }
    if (sum <= 0) return d.id[0];
    for (int i = 0; i < d.n; i++) d.p[i] /= sum;
    return draw(d, rng);
}
