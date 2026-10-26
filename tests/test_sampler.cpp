#include "../engine/sampler.h"
#include "llama.h"
#include <cstdio>
#include <map>
#include <set>
#include <string>

struct llama_sampler *llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length,
                                                     int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>> &seq_breakers);

using Rng = std::mt19937_64;
static int fails = 0;
static void check(bool ok, const std::string &what) {
    printf("%-104s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    fails += !ok;
}
static float uni(Rng &r, float a, float b) { return std::uniform_real_distribution<float>(a, b)(r); }
static int pick(Rng &r, int n) { return (int)(r() % (uint64_t)n); }
static bool one_in(Rng &r, int n) { return r() % (uint64_t)n == 0; }
static std::string num(long n) { return std::to_string(n); }

struct Setup {
    Sampling sp;
    std::vector<int> ctx;
    std::vector<std::vector<int>> breakers;
    std::vector<std::pair<int, float>> bias;
    uint32_t seed = 1;
};

static Sampling random_sampling(Rng &r, bool sure_xtc, bool allow_mirostat) {
    Sampling sp;
    sp.temperature = one_in(r, 8) ? 0.0f : one_in(r, 6) ? 1.0f : uni(r, 0.1f, 2.0f);
    sp.dynatemp_range = one_in(r, 4) ? uni(r, 0.0f, 1.5f) : 0.0f;
    sp.dynatemp_exponent = uni(r, 0.5f, 2.0f);
    sp.top_k = one_in(r, 4) ? 0 : 1 + pick(r, 80);
    sp.top_p = one_in(r, 3) ? 1.0f : uni(r, 0.05f, 1.0f);
    sp.min_p = one_in(r, 2) ? 0.0f : uni(r, 0.0f, 0.5f);
    sp.typical_p = one_in(r, 3) ? uni(r, 0.1f, 1.0f) : 1.0f;
    sp.xtc_probability = one_in(r, 3) ? (sure_xtc ? 1.0f : uni(r, 0.0f, 1.0f)) : 0.0f;
    sp.xtc_threshold = uni(r, 0.0f, 0.6f);
    sp.top_n_sigma = one_in(r, 3) ? uni(r, 0.1f, 4.0f) : -1.0f;
    const int windows[] = {0, 1, 8, 64, -1};
    sp.repeat_last_n = windows[pick(r, 5)];
    sp.repeat_penalty = one_in(r, 2) ? uni(r, 0.5f, 2.0f) : 1.0f;
    sp.frequency_penalty = one_in(r, 3) ? uni(r, -1.0f, 1.0f) : 0.0f;
    sp.presence_penalty = one_in(r, 2) ? uni(r, -1.0f, 2.0f) : 0.0f;
    if (one_in(r, 2)) {
        sp.dry_multiplier = uni(r, 0.1f, 3.0f);
        sp.dry_base = uni(r, 0.9f, 3.0f);
        sp.dry_allowed_length = pick(r, 5);
        const int dry_windows[] = {1, 4, 16, 64, -1};
        sp.dry_penalty_last_n = dry_windows[pick(r, 5)];
    }
    sp.mirostat = allow_mirostat && one_in(r, 4) ? 1 + pick(r, 2) : 0;
    sp.mirostat_tau = uni(r, 1.0f, 8.0f);
    sp.mirostat_eta = uni(r, 0.01f, 0.5f);
    sp.min_keep = one_in(r, 3) ? 1 + pick(r, 10) : 0;
    if (!one_in(r, 3)) {
        sp.n_order = 1 + pick(r, 12);
        for (int i = 0; i < sp.n_order; i++) sp.order[i] = pick(r, STAGES);
    }
    return sp;
}

static void random_context(Rng &r, Setup &s, const std::vector<int> &pool) {
    std::vector<int> pattern;
    for (int i = 0, n = 2 + pick(r, 6); i < n; i++) pattern.push_back(pool[pick(r, std::min<int>(16, pool.size()))]);
    for (int i = 0, n = pick(r, 300); i < n; i++) s.ctx.push_back(one_in(r, 5) ? pool[pick(r, pool.size())] : pattern[i % pattern.size()]);
    for (int i = 0; i < (int)pool.size(); i++)
        if (one_in(r, 10)) s.breakers.push_back({pool[i]});
    for (int i = 0, n = pick(r, 4); i < n; i++) {
        std::vector<int> seq;
        for (int j = 0, m = 2 + pick(r, 2); j < m; j++) seq.push_back(pattern[pick(r, pattern.size())]);
        s.breakers.push_back(seq);
    }
}

struct Result { std::map<int, float> p; int selected = -1; bool nan = false, exact = true; };

struct Llama {
    std::vector<llama_sampler *> stages;
    std::vector<int> kind;
    const Setup &s;

    Llama(const Setup &s, int n_vocab) : s(s) {
        const Sampling &sp = s.sp;
        const int all = (int)s.ctx.size() + 1000;
        auto add = [&](llama_sampler *x, int k) { stages.push_back(x); kind.push_back(k); };
        std::vector<llama_logit_bias> lb;
        if (sp.ban >= 0) lb.push_back({sp.ban, -INFINITY});
        for (auto [t, b] : s.bias) lb.push_back({t, b});
        if (!lb.empty()) add(llama_sampler_init_logit_bias(n_vocab, (int32_t)lb.size(), lb.data()), -1);
        if (sp.mirostat == 1 || sp.mirostat == 2) {
            add(llama_sampler_init_temp(sp.temperature), STAGE_TEMPERATURE);
            add(sp.mirostat == 1 ? llama_sampler_init_mirostat(n_vocab, s.seed, sp.mirostat_tau, sp.mirostat_eta, 100)
                                 : llama_sampler_init_mirostat_v2(s.seed, sp.mirostat_tau, sp.mirostat_eta), -2);
        } else {
            for (int i = 0; i < sp.n_order; i++) switch (sp.order[i]) {
                case STAGE_PENALTIES:
                    add(llama_sampler_init_penalties(n_vocab, sp.repeat_last_n < 0 ? all : sp.repeat_last_n, sp.repeat_penalty,
                                                     sp.frequency_penalty, sp.presence_penalty), STAGE_PENALTIES);
                    break;
                case STAGE_DRY:
                    if (sp.dry_multiplier != 0 && sp.dry_base >= 1 && sp.dry_penalty_last_n != 0)
                        add(llama_sampler_init_dry_testing(sp.dry_multiplier, sp.dry_base, sp.dry_allowed_length,
                                                           sp.dry_penalty_last_n < 0 ? all : sp.dry_penalty_last_n, s.breakers), STAGE_DRY);
                    break;
                case STAGE_TOP_N_SIGMA: add(llama_sampler_init_top_n_sigma(sp.top_n_sigma), STAGE_TOP_N_SIGMA); break;
                case STAGE_TOP_K: add(llama_sampler_init_top_k(sp.top_k), STAGE_TOP_K); break;
                case STAGE_TYPICAL_P: add(llama_sampler_init_typical(sp.typical_p, sp.min_keep), STAGE_TYPICAL_P); break;
                case STAGE_TOP_P: add(llama_sampler_init_top_p(sp.top_p, sp.min_keep), STAGE_TOP_P); break;
                case STAGE_MIN_P: add(llama_sampler_init_min_p(sp.min_p, sp.min_keep), STAGE_MIN_P); break;
                case STAGE_XTC: add(llama_sampler_init_xtc(sp.xtc_probability, sp.xtc_threshold, sp.min_keep, s.seed), STAGE_XTC); break;
                case STAGE_TEMPERATURE:
                    add(llama_sampler_init_temp_ext(sp.temperature, sp.dynatemp_range, sp.dynatemp_exponent), STAGE_TEMPERATURE);
                    break;
            }
            add(llama_sampler_init_dist(s.seed), -2);
        }
        for (int t : s.ctx) accept(t);
    }
    ~Llama() {
        for (llama_sampler *x : stages) llama_sampler_free(x);
    }
    void accept(int t) {
        for (llama_sampler *x : stages) llama_sampler_accept(x, t);
    }

    Result apply(std::vector<llama_token_data> cur, const std::set<int> *cands = nullptr) {
        llama_token_data_array a = {cur.data(), cur.size(), -1, false};
        Result r;
        bool scaled = false;
        int tail = cands ? (int)(cur.size() - cands->size()) : 0;
        for (size_t j = 0; j < stages.size(); j++) {
            const int k = kind[j];
            if (k == STAGE_TYPICAL_P && s.sp.typical_p < 1 && !s.sp.mirostat && a.size) {
                float mx = -INFINITY;
                for (size_t i = 0; i < a.size; i++) mx = std::max(mx, a.data[i].logit);
                for (size_t i = 0; i < a.size; i++) r.nan |= expf(a.data[i].logit - mx) < 1e-37f;
            }
            const bool mass = k == STAGE_TOP_P || k == STAGE_XTC || k == STAGE_TYPICAL_P || k == -2 ||
                              (k == STAGE_TEMPERATURE && s.sp.dynatemp_range > 0);
            if (cands && mass && scaled && tail > 0) r.exact = false;
            llama_sampler_apply(stages[j], &a);
            if (!cands) continue;
            int now = 0, kept = 0;
            for (size_t i = 0; i < a.size; i++)
                if (a.data[i].logit != -INFINITY) (cands->count(a.data[i].id) ? kept : now)++;
            if ((now > 0 && now < tail) || (now > 0 && kept == 0)) r.exact = false;
            if (k == STAGE_TEMPERATURE && now > 0 && (s.sp.temperature != 1.0f || s.sp.dynatemp_range > 0)) scaled = true;
            tail = now;
        }
        for (size_t i = 0; i < a.size; i++)
            if (a.data[i].p > 0) r.p[a.data[i].id] = a.data[i].p;
        r.selected = a.selected >= 0 ? a.data[a.selected].id : -1;
        return r;
    }
};

static bool miro_banned(const Setup &s) {
    bool zero = s.sp.ban >= 0;
    for (auto [t, b] : s.bias) zero |= b == -INFINITY;
    return s.sp.mirostat == 1 && zero;
}

static float xtc_draw(uint32_t seed) {
    std::mt19937 x(seed);
    return std::uniform_real_distribution<float>(0.0f, 1.0f)(x);
}

static void prepare(Seen &st, const Setup &s) {
    st.begin(s.ctx, s.sp);
    std::vector<int> flat, ids;
    std::vector<float> vals;
    for (auto &b : s.breakers) {
        flat.push_back((int)b.size());
        flat.insert(flat.end(), b.begin(), b.end());
    }
    st.set_breakers(flat.data(), (int)flat.size());
    for (auto [t, b] : s.bias) ids.push_back(t), vals.push_back(b);
    st.set_bias(ids.data(), vals.data(), (int)ids.size());
}

static double gap(const Result &l, const Dist &d) {
    std::map<int, float> ours;
    for (int i = 0; i < d.n; i++) ours[d.id[i]] = d.p[i];
    double worst = 0;
    for (auto [id, p] : l.p) worst = std::max(worst, (double)std::fabs(p - (ours.count(id) ? ours[id] : 0.0f)));
    for (auto [id, p] : ours)
        if (!l.p.count(id)) worst = std::max(worst, (double)p);
    return worst;
}

static std::string shown(const Result &l, const Dist &d) {
    std::string s = "\n    llama.cpp:";
    for (auto [id, p] : l.p) s += " " + num(id) + ":" + std::to_string(p);
    s += "\n    chrysalis:";
    for (int i = 0; i < d.n; i++) s += " " + num(d.id[i]) + ":" + std::to_string(d.p[i]);
    return s;
}

static const char *stage_name[STAGES] = {"penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"};

static std::string describe(const Sampling &sp) {
    char b[512];
    snprintf(b, sizeof b, "temp %g dyn %g/%g k %d p %g min %g typ %g xtc %g/%g sigma %g pen %d %g/%g/%g dry %g/%g/%d/%d miro %d %g/%g keep %d order",
             sp.temperature, sp.dynatemp_range, sp.dynatemp_exponent, sp.top_k, sp.top_p, sp.min_p, sp.typical_p, sp.xtc_probability,
             sp.xtc_threshold, sp.top_n_sigma, sp.repeat_last_n, sp.repeat_penalty, sp.frequency_penalty, sp.presence_penalty,
             sp.dry_multiplier, sp.dry_base, sp.dry_allowed_length, sp.dry_penalty_last_n, sp.mirostat, sp.mirostat_tau,
             sp.mirostat_eta, sp.min_keep);
    std::string s = b;
    for (int i = 0; i < sp.n_order; i++) s += std::string(" ") + stage_name[sp.order[i]];
    return s;
}

static void row(Rng &r, const std::vector<int> &pool, float *v, int *ids, std::vector<llama_token_data> &cands) {
    std::vector<int> order = pool;
    std::shuffle(order.begin(), order.end(), r);
    std::normal_distribution<float> logit(0, 3);
    for (int i = 0; i < SAMPLE_N; i++) v[i] = logit(r);
    std::sort(v, v + SAMPLE_N, std::greater<float>());
    cands.clear();
    for (int i = 0; i < SAMPLE_N; i++) {
        ids[i] = order[i];
        cands.push_back({ids[i], v[i], 0.0f});
    }
}

static void single_steps() {
    Rng r(1);
    int same = 0, cases = 0, left = 0, used[STAGES] = {}, miro = 0;
    std::string first;
    for (int c = 0; c < 30000; c++) {
        float v[SAMPLE_N];
        int ids[SAMPLE_N];
        std::set<int> distinct;
        while ((int)distinct.size() < SAMPLE_N) distinct.insert(pick(r, 3000));
        const std::vector<int> pool(distinct.begin(), distinct.end());
        std::vector<llama_token_data> cands;
        row(r, pool, v, ids, cands);
        Setup s;
        s.sp = random_sampling(r, false, true);
        s.seed = (uint32_t)(r() % 1000000007u);
        if (one_in(r, 4)) s.sp.ban = ids[pick(r, SAMPLE_N)];
        for (int i = 0, n = one_in(r, 3) ? 1 + pick(r, 5) : 0; i < n; i++) {
            const int t = ids[pick(r, SAMPLE_N)];
            bool dup = t == s.sp.ban;
            for (auto [u, b] : s.bias) dup |= u == t;
            if (!dup) s.bias.push_back({t, one_in(r, 4) ? -INFINITY : uni(r, -5.0f, 5.0f)});
        }
        random_context(r, s, pool);
        Llama llama(s, SAMPLE_N);
        const Result want = llama.apply(cands);
        if (want.nan || want.p.empty() || miro_banned(s)) { left++; continue; }
        Seen st;
        prepare(st, s);
        const Dist got = make_dist(v, ids, nullptr, s.sp, st, xtc_draw(s.seed));
        const bool ok = gap(want, got) < 2e-6;
        same += ok;
        cases++;
        if (!ok && first.empty()) first = "case " + num(c) + ": " + describe(s.sp) + shown(want, got);
        if (ok) {
            miro += s.sp.mirostat != 0;
            for (int i = 0; i < s.sp.n_order && !s.sp.mirostat; i++) used[s.sp.order[i]]++;
        }
    }
    bool all_used = miro > 100;
    for (int u : used) all_used &= u > 1000;
    check(all_used, "the random cases cover every sampler and both mirostats");
    check(same == cases, "make_dist gives llama.cpp's chain (" + num(same) + "/" + num(cases) + "; " + num(left) +
                             " left out where llama.cpp itself gives NaN or mirostat 1 counts a banned token)");
    if (!first.empty()) printf("  first difference: %s\n", first.c_str());
}

static void sequences() {
    Rng r(2);
    int same = 0, steps = 0;
    std::string first;
    std::vector<int> vocab;
    for (int i = 0; i < 200; i++) vocab.push_back(i * 7 + 3);
    for (int c = 0; c < 3000; c++) {
        Setup s;
        s.sp = random_sampling(r, true, true);
        s.seed = (uint32_t)(r() % 1000000007u);
        random_context(r, s, vocab);
        Llama llama(s, SAMPLE_N);
        Seen st;
        prepare(st, s);
        for (int step = 0; step < 24; step++) {
            float v[SAMPLE_N];
            int ids[SAMPLE_N];
            std::vector<llama_token_data> cands;
            row(r, vocab, v, ids, cands);
            const Result want = llama.apply(cands);
            if (want.nan || want.p.empty()) break;
            const Dist got = make_dist(v, ids, nullptr, s.sp, st, 0.0f);
            const bool ok = gap(want, got) < 2e-6;
            same += ok;
            steps++;
            if (!ok && first.empty()) first = "case " + num(c) + " step " + num(step) + ": " + describe(s.sp) + shown(want, got);
            if (!ok) break;
            llama.accept(want.selected);
            st.push(want.selected, prob(got, want.selected));
        }
    }
    check(steps > 50000, "replies run long enough to wrap the windows (" + num(steps) + " steps)");
    check(same == steps, "replies of 24 tokens keep penalties, DRY and mirostat's mu in step with llama.cpp (" + num(same) + "/" +
                             num(steps) + ")");
    if (!first.empty()) printf("  first difference: %s\n", first.c_str());
}

static void full_vocabulary() {
    Rng r(3);
    const int V = 4096;
    int same = 0, compared = 0, partial = 0;
    std::string first;
    for (int c = 0; c < 6000; c++) {
        std::normal_distribution<float> base(0, 2);
        std::vector<float> logits(V);
        for (float &x : logits) x = base(r);
        for (int i = 0; i < 24; i++) logits[pick(r, V)] += uni(r, 4.0f, 14.0f);
        std::vector<int> rank(V);
        for (int i = 0; i < V; i++) rank[i] = i;
        std::sort(rank.begin(), rank.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        float v[SAMPLE_N];
        int ids[SAMPLE_N];
        std::vector<int> pool;
        for (int i = 0; i < SAMPLE_N; i++) { ids[i] = rank[i]; v[i] = logits[rank[i]]; pool.push_back(ids[i]); }
        double sum = 0, sq = 0, mass = 0, xmass = 0;
        for (float x : logits) {
            sum += x;
            sq += (double)x * x;
            const double e = std::exp((double)x - v[0]);
            mass += e;
            xmass += (x - (double)v[0]) * e;
        }
        const float full[NSTAT] = {(float)V, (float)V, (float)sum, (float)sq, (float)mass, (float)xmass};
        Setup s;
        s.seed = (uint32_t)(r() % 1000000007u);
        s.sp = random_sampling(r, false, false);
        if (one_in(r, 2)) {
            const int fronts[] = {STAGE_PENALTIES, STAGE_DRY, STAGE_TOP_N_SIGMA, STAGE_TOP_K, STAGE_TOP_P, STAGE_MIN_P, STAGE_XTC, STAGE_TYPICAL_P};
            s.sp.n_order = 0;
            for (int i = 0, n = 1 + pick(r, 7); i < n; i++) s.sp.order[s.sp.n_order++] = fronts[pick(r, 8)];
            s.sp.order[s.sp.n_order++] = STAGE_TEMPERATURE;
        } else if (one_in(r, 3)) {
            s.sp.n_order = 2;
            s.sp.order[0] = STAGE_TEMPERATURE;
            s.sp.order[1] = STAGE_TOP_K;
            s.sp.top_k = 1 + pick(r, SAMPLE_N);
            s.sp.dynatemp_range = uni(r, 0.1f, 1.5f);
        } else if (one_in(r, 4)) {
            s.sp.mirostat = 2;
            s.sp.temperature = 1.0f;
        } else {
            s.sp.n_order = STAGES;
            for (int i = 0; i < STAGES; i++) s.sp.order[i] = i;
        }
        if (s.sp.temperature <= 0) s.sp.temperature = 0.5f;
        if (one_in(r, 3)) s.sp.top_k = 0;
        if (one_in(r, 4)) s.sp.ban = ids[pick(r, SAMPLE_N)];
        for (int i = 0, n = one_in(r, 3) ? 1 + pick(r, 5) : 0; i < n; i++) s.bias.push_back({ids[pick(r, SAMPLE_N)], uni(r, -4.0f, 4.0f)});
        random_context(r, s, pool);
        std::vector<llama_token_data> cur;
        for (int i = 0; i < V; i++) cur.push_back({i, logits[i], 0.0f});
        const std::set<int> cands(ids, ids + SAMPLE_N);
        Llama llama(s, V);
        const Result want = llama.apply(cur, &cands);
        bool inside = (int)want.p.size() <= SAMPLE_N;
        for (auto [id, p] : want.p) inside &= cands.count(id) > 0;
        if (want.nan || !inside || !want.exact || s.sp.typical_p < 1) { partial++; continue; }
        Seen st;
        prepare(st, s);
        const Dist got = make_dist(v, ids, full, s.sp, st, xtc_draw(s.seed));
        const bool ok = gap(want, got) < 2e-5;
        same += ok;
        compared++;
        if (!ok && first.empty()) first = "case " + num(c) + ": " + describe(s.sp) + shown(want, got);
    }
    check(compared > 2500, "most full-vocabulary cases keep the whole tail or none of it (" + num(compared) + " compared, " + num(partial) +
                               " keep part of it or run typical_p over it)");
    check(same == compared, "given the vocabulary's statistics, make_dist gives llama.cpp's chain over all " + num(V) + " tokens (" +
                                num(same) + "/" + num(compared) + ")");
    if (!first.empty()) printf("  first difference: %s\n", first.c_str());
}

static void speculation() {
    Rng r(4);
    int same = 0, cases = 0, possible = 0, tokens = 0;
    std::string first;
    std::vector<int> vocab;
    for (int i = 0; i < 100; i++) vocab.push_back(i * 5 + 1);
    for (int c = 0; c < 3000; c++) {
        Setup s;
        s.sp = random_sampling(r, true, false);
        s.sp.presence_penalty = uni(r, 2.0f, 40.0f);
        s.sp.repeat_last_n = one_in(r, 2) ? -1 : 8 + pick(r, 60);
        if (one_in(r, 2)) s.sp.top_k = 1 + pick(r, 3);
        s.seed = (uint32_t)(r() % 1000000007u);
        random_context(r, s, vocab);
        const int k = 1 + pick(r, 7), repeat = vocab[pick(r, 100)];
        std::vector<float> tv((k + 2) * SAMPLE_N);
        std::vector<int> ti((k + 2) * SAMPLE_N), drafts(k + 1, vocab[0]);
        std::vector<std::vector<llama_token_data>> rows(k + 2);
        for (int a = 0; a <= k + 1; a++) {
            row(r, vocab, &tv[a * SAMPLE_N], &ti[a * SAMPLE_N], rows[a]);
            if (a > k) continue;
            int *at = std::find(&ti[a * SAMPLE_N], &ti[(a + 1) * SAMPLE_N], repeat);
            if (at == &ti[(a + 1) * SAMPLE_N]) at = &ti[(a + 1) * SAMPLE_N - 1];
            *at = ti[a * SAMPLE_N];
            ti[a * SAMPLE_N] = repeat;
            tv[a * SAMPLE_N] += 30;
            for (int i = 0; i < SAMPLE_N; i++) rows[a][i] = {ti[a * SAMPLE_N + i], tv[a * SAMPLE_N + i], 0.0f};
            if (a < k) drafts[a + 1] = one_in(r, 4) ? ti[a * SAMPLE_N + pick(r, SAMPLE_N)] : repeat;
        }
        Seen st;
        prepare(st, s);
        Rng rng(s.seed);
        std::vector<int> out;
        verify_drafts(st, tv.data(), ti.data(), nullptr, drafts.data(), k, s.sp, rng, [](int) { return false; },
                      [&](int, int t) { out.push_back(t); });
        Llama llama(s, 1000);
        bool nan = false;
        for (size_t j = 0; j < out.size(); j++) {
            const Result at = llama.apply(rows[j]);
            nan |= at.nan;
            possible += at.p.count(out[j]) > 0 || at.nan;
            tokens++;
            if (!at.p.count(out[j]) && !at.nan && first.empty())
                first = "case " + num(c) + ": token " + num(out[j]) + " at " + num(j) + " is impossible: " + describe(s.sp);
            llama.accept(out[j]);
        }
        const Result next = llama.apply(rows[out.size()]);
        if (nan || next.nan || next.p.empty()) continue;
        const Dist got = make_dist(&tv[out.size() * SAMPLE_N], &ti[out.size() * SAMPLE_N], nullptr, s.sp, st, 0.0f);
        const bool ok = gap(next, got) < 2e-6;
        same += ok;
        cases++;
        if (!ok && first.empty()) first = "case " + num(c) + ": " + describe(s.sp) + shown(next, got);
    }
    check(possible == tokens, "every token speculation emits has a chance at its position (" + num(possible) + "/" + num(tokens) + ")");
    check(same == cases, "after speculation the state holds the emitted tokens, no rejected draft (" + num(same) + "/" + num(cases) + ")");
    if (!first.empty()) printf("  first difference: %s\n", first.c_str());
}

int main() {
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    single_steps();
    sequences();
    full_vocabulary();
    speculation();
    puts(fails ? "FAILURES" : "all ok");
    return fails != 0;
}
