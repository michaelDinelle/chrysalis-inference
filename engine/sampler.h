#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

constexpr int SAMPLE_N = 64, NSTAT = 6, MAX_ORDER = 32;
enum Stage : int { STAGE_PENALTIES, STAGE_DRY, STAGE_TOP_N_SIGMA, STAGE_TOP_K, STAGE_TYPICAL_P, STAGE_TOP_P, STAGE_MIN_P, STAGE_XTC,
                   STAGE_TEMPERATURE, STAGES };
enum Stat : int { STAT_VOCAB, STAT_FINITE, STAT_SUM, STAT_SQUARES, STAT_MASS, STAT_XMASS };

struct Sampling {
    float temperature = 1.0f, dynatemp_range = 0.0f, dynatemp_exponent = 1.0f;
    int top_k = 20;
    float top_p = 0.95f, min_p = 0.0f, typical_p = 1.0f, xtc_probability = 0.0f, xtc_threshold = 0.1f, top_n_sigma = -1.0f;
    int repeat_last_n = 64;
    float repeat_penalty = 1.0f, frequency_penalty = 0.0f, presence_penalty = 0.0f, dry_multiplier = 0.0f, dry_base = 1.75f;
    int dry_allowed_length = 2, dry_penalty_last_n = -1, mirostat = 0;
    float mirostat_tau = 5.0f, mirostat_eta = 0.1f;
    int min_keep = 0, n_order = STAGES;
    int order[MAX_ORDER] = {STAGE_PENALTIES, STAGE_DRY, STAGE_TOP_N_SIGMA, STAGE_TOP_K, STAGE_TYPICAL_P, STAGE_TOP_P, STAGE_MIN_P,
                            STAGE_XTC, STAGE_TEMPERATURE};
    int n_draft = 3, lookup = 0, ban = -1, halt = -1;
};

inline bool mirostat_on(const Sampling &sp) { return sp.mirostat == 1 || sp.mirostat == 2; }
inline bool in_chain(const Sampling &sp, int stage) {
    if (mirostat_on(sp)) return false;
    for (int i = 0; i < sp.n_order && i < MAX_ORDER; i++)
        if (sp.order[i] == stage) return true;
    return false;
}
inline int penalty_window(const Sampling &sp) {
    if (!in_chain(sp, STAGE_PENALTIES) || sp.repeat_last_n == 0 ||
        (sp.repeat_penalty == 1.0f && sp.frequency_penalty == 0.0f && sp.presence_penalty == 0.0f))
        return 0;
    return sp.repeat_last_n < 0 ? INT_MAX : sp.repeat_last_n;
}
inline int dry_window(const Sampling &sp) {
    if (!in_chain(sp, STAGE_DRY) || sp.dry_multiplier == 0.0f || sp.dry_base < 1.0f || sp.dry_penalty_last_n == 0) return 0;
    return sp.dry_penalty_last_n < 0 ? INT_MAX : sp.dry_penalty_last_n;
}
inline bool xtc_draws(const Sampling &sp) { return in_chain(sp, STAGE_XTC) && sp.xtc_probability > 0.0f && sp.xtc_threshold <= 0.5f; }

template <class T> T &grown(std::vector<T> &v, int i, T fill) {
    if ((size_t)i >= v.size()) v.resize((size_t)i + 4097, fill);
    return v[i];
}

struct Cand { float l; int id; };

inline void sort_desc(Cand *c, int n) {
    for (int i = 1; i < n; i++) {
        const Cand x = c[i];
        int j = i;
        for (; j > 0 && c[j - 1].l < x.l; j--) c[j] = c[j - 1];
        c[j] = x;
    }
}

struct Seen {
    std::vector<int> hist, count, prev, last, run, cur, seqs;
    std::vector<std::pair<int, float>> bias;
    std::vector<uint8_t> brk;
    int pen_n = 0, dry_n = 0, mirostat = 0;
    float mu = 0, tau = 0, eta = 0;

    void clear() {
        for (int t : hist) {
            if ((size_t)t < count.size()) count[t] = 0;
            if ((size_t)t < last.size()) last[t] = -1;
        }
        hist.clear(); prev.clear(); run.clear(); cur.clear(); seqs.clear(); bias.clear();
        std::fill(brk.begin(), brk.end(), 0);
        pen_n = dry_n = mirostat = 0;
    }

    void begin(const std::vector<int> &ctx, const Sampling &sp) {
        clear();
        pen_n = penalty_window(sp);
        dry_n = dry_window(sp);
        mirostat = mirostat_on(sp) ? sp.mirostat : 0;
        tau = sp.mirostat_tau;
        eta = sp.mirostat_eta;
        mu = 2.0f * tau;
        const size_t take = std::min((size_t)std::max(pen_n, dry_n), ctx.size());
        for (size_t i = ctx.size() - take; i < ctx.size(); i++)
            if (ctx[i] >= 0) hist.push_back(ctx[i]);
        const int n = (int)hist.size();
        for (int i = std::max(0, n - pen_n); pen_n && i < n; i++) grown(count, hist[i], 0)++;
        if (dry_n) index();
    }

    void index() {
        const int n = (int)hist.size(), N = n - std::max(0, n - dry_n);
        prev.resize(n);
        run.assign(n, 0);
        cur.clear();
        for (int p = 0; p < n; p++) {
            int &l = grown(last, hist[p], -1);
            prev[p] = l;
            l = p;
        }
        auto back = [&](int i) { return hist[n - 1 - i]; };
        for (int k = 1, lt = 0, rt = 0; k < N; k++) {
            int z = 0;
            if (k > rt) {
                while (z + k < N && back(z) == back(z + k)) z++;
                if (z > 0) lt = k, rt = k + z - 1;
            } else if (run[n - 1 - (k - lt)] < rt - k + 1) {
                z = run[n - 1 - (k - lt)];
            } else {
                int i = rt + 1;
                while (i < N && back(i) == back(i - k)) i++;
                z = i - k;
                lt = k;
                rt = i - 1;
            }
            run[n - 1 - k] = z;
            if (z) cur.push_back(n - 1 - k);
        }
    }

    void push(int t, float p) {
        if (t < 0) return;
        if (mirostat && p > 0) mu = mu - eta * (-log2f(p) - tau);
        const int n = (int)hist.size();
        if (pen_n) {
            grown(count, t, 0)++;
            if (n >= pen_n) count[hist[n - pen_n]]--;
        }
        if (dry_n) {
            int &l = grown(last, t, -1);
            const int lo = std::max(0, n + 1 - dry_n);
            thread_local std::vector<std::pair<int, int>> fresh;
            fresh.clear();
            for (int q = l; q >= lo; q = prev[q]) fresh.push_back({q, 1 + (q > 0 ? run[q - 1] : 0)});
            for (int q : cur) run[q] = 0;
            cur.clear();
            for (auto [q, z] : fresh) {
                run[q] = z;
                cur.push_back(q);
            }
            prev.push_back(l);
            run.push_back(0);
            l = n;
        }
        hist.push_back(t);
    }

    void set_bias(const int *ids, const float *vals, int n) {
        bias.clear();
        for (int i = 0; i < n; i++) bias.emplace_back(ids[i], vals[i]);
        std::stable_sort(bias.begin(), bias.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    }

    float biased(int id, float x) const {
        auto it = std::lower_bound(bias.begin(), bias.end(), id, [](const auto &a, int v) { return a.first < v; });
        for (; it != bias.end() && it->first == id; ++it) x += it->second;
        return x;
    }

    void set_breakers(const int *flat, int n) {
        std::fill(brk.begin(), brk.end(), 0);
        seqs.clear();
        for (int i = 0; i + 1 < n && flat[i] >= 1 && i + 1 + flat[i] <= n; i += 1 + flat[i]) {
            const int len = flat[i], head = flat[i + 1];
            if (head < 0) continue;
            grown(brk, head, (uint8_t)0) |= len == 1 ? 1 : 2;
            if (len > 1) seqs.insert(seqs.end(), flat + i, flat + i + 1 + len);
        }
    }

    int count_of(int t) const { return (size_t)t < count.size() ? count[t] : 0; }
    bool lone_breaker(int t) const { return (size_t)t < brk.size() && (brk[t] & 1); }

    int restart(int N) const {
        const int n = (int)hist.size();
        for (int i = 0; i < N; i++) {
            const int t = hist[n - 1 - i], f = (size_t)t < brk.size() ? brk[t] : 0;
            if (!f) continue;
            int longest = f & 1 ? 0 : -1;
            for (size_t j = 0; (f & 2) && j < seqs.size(); j += 1 + seqs[j]) {
                const int tail = seqs[j] - 1;
                if (seqs[j + 1] != t || tail <= longest || tail > i) continue;
                int o = 0;
                while (o < tail && seqs[j + 2 + o] == hist[n - i + o]) o++;
                if (o == tail) longest = tail;
            }
            if (longest >= 0) return i - longest;
        }
        return N;
    }

    void dry(Cand *c, int n, const Sampling &sp) const {
        const int total = (int)hist.size(), N = std::min(total, dry_n), A = sp.dry_allowed_length;
        if (!dry_n || N <= A) return;
        const int limit = restart(N);
        if (limit < A) return;
        int slot[2 * SAMPLE_N], rep[SAMPLE_N];
        std::fill(slot, slot + 2 * SAMPLE_N, -1);
        auto home = [](int id) { return (int)(((uint32_t)id * 2654435761u) >> 25); };
        for (int i = 0; i < n; i++) {
            rep[i] = -1;
            int h = home(c[i].id);
            while (slot[h] >= 0) h = (h + 1) & (2 * SAMPLE_N - 1);
            slot[h] = i;
        }
        auto find = [&](int id) {
            for (int h = home(id); slot[h] >= 0; h = (h + 1) & (2 * SAMPLE_N - 1))
                if (c[slot[h]].id == id) return slot[h];
            return -1;
        };
        const int lo = total - N;
        for (int q : cur) {
            if (q < lo || q > total - 2) continue;
            const int len = std::min({run[q], q - lo + 1, limit});
            if (len < A) continue;
            const int i = find(hist[q + 1]);
            if (i >= 0) rep[i] = std::max(rep[i], len);
        }
        if (A == 0)
            for (int i = 0; i < n; i++)
                if ((size_t)c[i].id < last.size() && last[c[i].id] > lo) rep[i] = std::max(rep[i], 0);
        int max_exp = 0;
        if (sp.dry_base > 1.000001f) max_exp = 88.7228391f / std::log(sp.dry_base);
        for (int i = 0; i < n; i++) {
            if (rep[i] < 0 || lone_breaker(c[i].id)) continue;
            int e = rep[i] - A;
            if (max_exp > 0 && e > max_exp) e = max_exp;
            const float penalty = sp.dry_multiplier * std::pow(sp.dry_base, e);
            c[i].l -= penalty;
        }
    }
};

struct Tail {
    double entries = 0, finite = 0, sum = 0, squares = 0, mass = 0, xmass = 0;
    float ref = 0, top = -INFINITY;
    bool exact = false;
    bool live() const { return finite > 0 && exact; }
    void forget() { finite = sum = squares = mass = xmass = 0; exact = false; }
};

inline Tail tail_of(const float *v, const float *full) {
    Tail t;
    if (!full) return t;
    t.entries = std::max(0.0, (double)full[STAT_VOCAB] - SAMPLE_N);
    if (full[STAT_FINITE] < 0 || !std::isfinite(v[0])) return t;
    double finite = full[STAT_FINITE], sum = full[STAT_SUM], squares = full[STAT_SQUARES], mass = full[STAT_MASS], xmass = full[STAT_XMASS];
    for (int i = 0; i < SAMPLE_N; i++) {
        if (!std::isfinite(v[i])) continue;
        const double x = v[i], e = std::exp(x - v[0]);
        finite -= 1; sum -= x; squares -= x * x; mass -= e; xmass -= (x - v[0]) * e;
    }
    if (finite < 0.5) return t;
    t.finite = finite; t.sum = sum; t.squares = std::max(squares, 0.0); t.mass = std::max(mass, 0.0); t.xmass = std::min(xmass, 0.0);
    t.ref = v[0];
    t.top = v[SAMPLE_N - 1];
    t.exact = true;
    return t;
}

inline float softmax(const Cand *c, int n, const Tail &t, float *p) {
    const float m = c[0].l;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        p[i] = expf(c[i].l - m);
        cum += p[i];
    }
    if (t.live()) cum = (float)(cum + t.mass * std::exp((double)t.ref - m));
    for (int i = 0; i < n; i++) p[i] /= cum;
    return cum;
}

inline float entropy(const float *p, int n, const Tail &t, float m, float cum) {
    float h = 0.0f;
    for (int i = 0; i < n; i++)
        if (p[i] > 0.0f) h -= p[i] * logf(p[i]);
    if (t.live()) {
        const double w = std::exp((double)t.ref - m);
        h = (float)(h - (w / cum) * (t.xmass + ((double)t.ref - m - std::log((double)cum)) * t.mass));
    }
    return h;
}

inline void scale(Cand *c, int n, Tail &t, float temp) {
    if (temp <= 0.0f) {
        for (int i = 1; i < n; i++) c[i].l = -INFINITY;
        t.forget();
        return;
    }
    for (int i = 0; i < n; i++) c[i].l /= temp;
    t.sum /= temp;
    t.squares /= (double)temp * temp;
    t.top /= temp;
    t.mass = t.xmass = 0;
    t.exact = false;
}

struct Dist { int n = 0; int id[SAMPLE_N]; float p[SAMPLE_N]; };

inline Dist make_dist(const float *v, const int *ids, const float *full, const Sampling &sp, const Seen &st, float chance) {
    Cand c[SAMPLE_N];
    int n = SAMPLE_N;
    for (int i = 0; i < SAMPLE_N; i++) c[i] = {ids[i] == sp.ban ? -INFINITY : st.biased(ids[i], v[i]), ids[i]};
    sort_desc(c, n);
    Tail t;
    bool known = !full;
    auto tail = [&]() -> Tail & {
        if (!known) t = tail_of(v, full), known = true;
        return t;
    };
    auto drop = [&] { t = Tail{}, known = true; };
    float p[SAMPLE_N];
    Dist d;
    if (mirostat_on(sp)) {
        if (sp.temperature != 1.0f) scale(c, n, tail(), sp.temperature);
        softmax(c, n, tail(), p);
        int finite = 0;
        while (finite < n && c[finite].l != -INFINITY) finite++;
        if (sp.mirostat == 1) {
            float tb = 0.0f, tt = 0.0f;
            for (int i = 0; i < 99 && i < finite - 1; i++) {
                const float ti = logf(float(i + 2) / float(i + 1)), bi = logf(p[i] / p[i + 1]);
                tb += ti * bi;
                tt += ti * ti;
            }
            const float s_hat = tb / tt, eps = s_hat - 1;
            const float vocab = full ? full[STAT_VOCAB] : (float)SAMPLE_N;
            const float k = powf((eps * powf(2, st.mu)) / (1 - powf(vocab, -eps)), 1 / s_hat);
            n = std::min(n, k >= 1.0f && k < 2147483648.0f ? (int)k : 1);
        } else {
            int m = 0;
            while (m < n && !(-log2f(p[m]) > st.mu)) m++;
            n = std::max(m, 1);
        }
        drop();
        softmax(c, n, t, p);
        for (int i = 0; i < n; i++)
            if (p[i] > 0.0f) { d.id[d.n] = c[i].id; d.p[d.n++] = p[i]; }
        return d.n ? d : Dist{1, {c[0].id}, {1.0f}};
    }
    for (int s = 0; s < sp.n_order && s < MAX_ORDER; s++) switch (sp.order[s]) {
        case STAGE_PENALTIES:
            if (!st.pen_n) break;
            for (int i = 0; i < n; i++) {
                const int k = st.count_of(c[i].id);
                if (!k) continue;
                c[i].l = c[i].l <= 0 ? c[i].l * sp.repeat_penalty : c[i].l / sp.repeat_penalty;
                c[i].l -= float(k) * sp.frequency_penalty + float(k > 0) * sp.presence_penalty;
            }
            sort_desc(c, n);
            break;
        case STAGE_DRY:
            if (!st.dry_n) break;
            st.dry(c, n, sp);
            sort_desc(c, n);
            break;
        case STAGE_TOP_N_SIGMA: {
            if (sp.top_n_sigma <= 0.0f || n + tail().entries <= 1) break;
            float mx = c[0].l, sum = 0.0f, acc = 0.0f;
            size_t cnt = 0;
            for (int i = 0; i < n; i++)
                if (c[i].l != -INFINITY) { mx = std::max(mx, c[i].l); sum += c[i].l; cnt++; }
            const double all = cnt + t.finite;
            const float mean = t.finite > 0 ? (float)((sum + t.sum) / all) : cnt > 0 ? sum / cnt : 0.0f;
            for (int i = 0; i < n; i++)
                if (c[i].l != -INFINITY) acc += std::pow(c[i].l - mean, 2);
            float sd = 0.0f;
            if (t.finite > 0) sd = (float)std::sqrt((acc + t.squares - 2.0 * mean * t.sum + t.finite * (double)mean * mean) / all);
            else if (cnt > 0) sd = (float)std::sqrt((double)(acc / cnt));
            const float thr = mx - (sp.top_n_sigma * sd);
            for (int i = 0; i < n; i++)
                if (c[i].l < thr) c[i].l = -INFINITY;
            if (t.finite > 0 && t.top < thr) t.forget();
            break;
        }
        case STAGE_TOP_K: {
            if (sp.top_k <= 0) break;
            if (sp.top_k < n) {
                n = sp.top_k;
                drop();
                break;
            }
            const int k = (int)std::min<double>(sp.top_k, n + tail().entries);
            if (k - n < t.entries) {
                if (k - n < t.finite) t.forget();
                t.entries = k - n;
            }
            break;
        }
        case STAGE_TYPICAL_P: {
            if (sp.typical_p >= 1.0f) break;
            const float cum = softmax(c, n, tail(), p);
            const float h = entropy(p, n, t, c[0].l, cum);
            float score[SAMPLE_N];
            int idx[SAMPLE_N];
            for (int i = 0; i < n; i++) {
                score[i] = fabsf(-logf(p[i]) - h);
                int j = i;
                for (; j > 0 && score[idx[j - 1]] > score[i]; j--) idx[j] = idx[j - 1];
                idx[j] = i;
            }
            float acc = 0.0f;
            int last = n;
            for (int i = 0; i < n; i++) {
                acc += p[idx[i]];
                if (acc > sp.typical_p && (sp.min_keep == 0 || i >= sp.min_keep - 1)) { last = i + 1; break; }
            }
            Cand kept[SAMPLE_N];
            for (int i = 0; i < last; i++) kept[i] = c[idx[i]];
            std::copy(kept, kept + last, c);
            n = last;
            sort_desc(c, n);
            drop();
            break;
        }
        case STAGE_TOP_P: {
            if (sp.top_p >= 1.0f) break;
            softmax(c, n, tail(), p);
            float acc = 0.0f;
            int last = -1;
            for (int i = 0; i < n; i++) {
                acc += p[i];
                if (acc >= sp.top_p && i + 1 >= sp.min_keep) { last = i + 1; break; }
            }
            if (last >= 0) {
                n = last;
                drop();
            } else if (t.finite > 0) t.forget();
            break;
        }
        case STAGE_MIN_P: {
            if (sp.min_p <= 0.0f || n + tail().entries == 0) break;
            const float thr = c[0].l + logf(sp.min_p);
            int keep = 0;
            while (keep < n && c[keep].l >= thr) keep++;
            n = std::min(n, std::max({keep, 1, sp.min_keep}));
            if (t.finite > 0 && t.top >= thr && keep == n) t.forget();
            else drop();
            break;
        }
        case STAGE_XTC: {
            if (sp.xtc_probability <= 0.0f || sp.xtc_threshold > 0.5f || chance > sp.xtc_probability || n + tail().entries < 2) break;
            softmax(c, n, t, p);
            int last = 0;
            for (int i = 0; i < n && p[i] >= sp.xtc_threshold; i++) last = i;
            if (n + t.entries - last >= sp.min_keep && last > 0) {
                std::copy(c + last, c + n, c);
                n -= last;
            }
            break;
        }
        case STAGE_TEMPERATURE:
            if (sp.dynatemp_range > 0.0f) {
                const double size = n + tail().entries;
                if (size <= 1) break;
                const float max_ent = -logf(1.0f / (float)size);
                const float cum = softmax(c, n, t, p);
                const float h = entropy(p, n, t, c[0].l, cum);
                const float lo = std::max(0.0f, sp.temperature - sp.dynatemp_range), hi = sp.temperature + sp.dynatemp_range;
                scale(c, n, t, lo + (hi - lo) * powf(h / max_ent, sp.dynatemp_exponent));
            } else if (sp.temperature != 1.0f) scale(c, n, tail(), sp.temperature);
            break;
    }
    const float m = c[0].l;
    double sum = 0;
    for (int i = 0; i < n; i++) {
        p[i] = expf(c[i].l - m);
        sum += p[i];
    }
    for (int i = 0; i < n; i++)
        if (p[i] > 0.0f) { d.id[d.n] = c[i].id; d.p[d.n++] = p[i] / sum; }
    return d.n ? d : Dist{1, {c[0].id}, {1.0f}};
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

inline float xtc_chance(const Sampling &sp, std::mt19937_64 &rng) {
    return xtc_draws(sp) ? std::uniform_real_distribution<float>(0, 1)(rng) : 2.0f;
}

inline Dist dist_at(const Seen &st, const float *v, const int *ids, const float *full, const Sampling &sp, std::mt19937_64 &rng) {
    return make_dist(v, ids, full, sp, st, xtc_chance(sp, rng));
}

inline int sample(Seen &st, const float *v, const int *ids, const float *full, const Sampling &sp, std::mt19937_64 &rng) {
    const Dist d = dist_at(st, v, ids, full, sp, rng);
    const int t = draw(d, rng);
    st.push(t, prob(d, t));
    return t;
}

template <class Stop, class Emit>
int verify_drafts(Seen &st, const float *tv, const int *ti, const float *ts, const int *drafts, int k, const Sampling &sp,
                  std::mt19937_64 &rng, Stop stop, Emit emit) {
    for (int a = 0; a < k; a++) {
        const Dist d = dist_at(st, tv + (size_t)a * SAMPLE_N, ti + (size_t)a * SAMPLE_N, ts ? ts + (size_t)a * NSTAT : nullptr, sp, rng);
        const int tok = drafts[a + 1];
        const float pa = prob(d, tok);
        if (std::uniform_real_distribution<float>(0, 1)(rng) < pa) {
            st.push(tok, pa);
            emit(a, tok);
            if (stop(tok)) return a;
            continue;
        }
        const int t = draw_without(d, tok, rng);
        st.push(t, prob(d, t));
        emit(a, t);
        return a;
    }
    emit(k, sample(st, tv + (size_t)k * SAMPLE_N, ti + (size_t)k * SAMPLE_N, ts ? ts + (size_t)k * NSTAT : nullptr, sp, rng));
    return k;
}
