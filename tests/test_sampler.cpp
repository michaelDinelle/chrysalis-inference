// The host sampler (engine/sampler.h): the default order draws exactly as the sampler before it (fixed stages, a scan
// of the reply for the presence penalty), stage order and DRY do what they say.
//   build/test_sampler
#include "../engine/sampler.h"
#include <cstdio>
#include <cstring>

namespace old {   // make_dist as it was (commit 9b7b298)
Dist make_dist(const float *v, const int *ids, const Sampling &sp, const std::vector<int> &seen) {
    std::pair<float, int> c[SAMPLE_N];
    int nc = 0;
    for (int i = 0; i < SAMPLE_N; i++) {
        if (ids[i] == sp.ban) continue;
        float x = v[i];
        if (sp.presence_penalty != 0 && std::find(seen.begin(), seen.end(), ids[i]) != seen.end()) x -= sp.presence_penalty;
        c[nc++] = {x, ids[i]};
    }
    std::sort(c, c + nc, [](auto &a, auto &b) { return a.first > b.first; });
    Dist d;
    if (sp.temperature <= 0) { d.n = 1; d.id[0] = c[0].second; d.p[0] = 1; return d; }
    const int k = std::max(1, std::min(sp.top_k > 0 ? sp.top_k : nc, nc));
    double sum = 0;
    for (int i = 0; i < k; i++) { d.p[i] = expf((c[i].first - c[0].first) / sp.temperature); d.id[i] = c[i].second; sum += d.p[i]; }
    d.n = k;
    double cum = 0;
    for (int i = 0; i < k; i++) {
        d.p[i] /= sum; cum += d.p[i];
        if (cum >= sp.top_p) { d.n = i + 1; break; }
    }
    int m = 0;
    for (int i = 0; i < d.n; i++) if (d.p[i] >= sp.min_p * d.p[0]) { d.p[m] = d.p[i]; d.id[m++] = d.id[i]; }
    d.n = m;
    sum = 0;
    for (int i = 0; i < d.n; i++) sum += d.p[i];
    for (int i = 0; i < d.n; i++) d.p[i] /= sum;
    return d;
}
}  // namespace old

static int fails = 0;
static void check(bool ok, const char *what) {
    printf("%-58s %s\n", what, ok ? "ok" : "FAIL");
    fails += !ok;
}

int main() {
    std::mt19937_64 rng(1);
    std::normal_distribution<float> logit(0, 3);
    // 1. default order = the old sampler, bit for bit, over random rows, settings and replies
    int same = 0, cases = 20000;
    for (int t = 0; t < cases; t++) {
        float v[SAMPLE_N];
        int ids[SAMPLE_N];
        for (int i = 0; i < SAMPLE_N; i++) { v[i] = logit(rng); ids[i] = (int)(rng() % 5000); }
        std::sort(v, v + SAMPLE_N, std::greater<float>());
        Sampling sp;
        sp.temperature = std::uniform_real_distribution<float>(0, 2)(rng);
        if (t % 17 == 0) sp.temperature = 0;
        sp.top_p = std::uniform_real_distribution<float>(0.05f, 1)(rng);
        sp.min_p = t % 3 ? 0 : std::uniform_real_distribution<float>(0, 0.4f)(rng);
        sp.top_k = (int)(rng() % 70);
        sp.presence_penalty = t % 2 ? 1.5f : 0;
        sp.ban = t % 5 ? -1 : ids[rng() % SAMPLE_N];
        std::vector<int> reply;
        Seen seen;
        for (int i = 0; i < (int)(rng() % 40); i++) { int x = ids[rng() % SAMPLE_N]; reply.push_back(x); seen.push(x); }
        const Dist a = old::make_dist(v, ids, sp, reply), b = make_dist(v, ids, sp, seen, {});
        same += a.n == b.n && !memcmp(a.id, b.id, a.n * 4) && !memcmp(a.p, b.p, a.n * 4);
    }
    char line[96];
    snprintf(line, sizeof line, "default order = previous sampler (%d/%d bit-identical)", same, cases);
    check(same == cases, line);

    // 2. order: top P before the temperature judges the untempered odds
    float v[SAMPLE_N];
    int ids[SAMPLE_N];
    for (int i = 0; i < SAMPLE_N; i++) { v[i] = -1.0f * i; ids[i] = i; }
    Sampling sp;
    sp.temperature = 3; sp.top_p = 0.9f; sp.top_k = 0;
    const int first = make_dist(v, ids, sp, {}, {}).n;
    sp.order[0] = STAGE_TOP_K; sp.order[1] = STAGE_TOP_P; sp.order[2] = STAGE_MIN_P; sp.order[3] = STAGE_TEMPERATURE;
    const int last = make_dist(v, ids, sp, {}, {}).n;
    snprintf(line, sizeof line, "temperature last keeps fewer at a high temperature (%d < %d)", last, first);
    check(last < first, line);

    // 3. DRY: the token that would repeat "a b c" after "a b" is penalized; nothing without a repeat
    std::vector<int> ctx = {7, 1, 2, 3, 9, 9, 1, 2};    // "... a b c ... a b" -> c would repeat a run of 2
    Sampling dry;
    dry.top_k = 0; dry.top_p = 1; dry.dry_multiplier = 2; dry.dry_allowed_length = 2;
    for (int i = 0; i < SAMPLE_N; i++) v[i] = 0;
    const Dist d0 = make_dist(v, ids, Sampling{.top_p = 1, .top_k = 0}, {}, ctx), d1 = make_dist(v, ids, dry, {}, ctx);
    check(prob(d1, 3) < prob(d0, 3) * 0.2f && prob(d1, 4) > prob(d0, 4), "DRY penalizes the continuation of a repeated run");
    dry_breakers.assign(16, 0);
    dry_breakers[2] = 1;                                 // a breaker inside the run ends it
    const Dist d2 = make_dist(v, ids, dry, {}, ctx);
    check(std::abs(prob(d2, 3) - prob(d0, 3)) < 1e-6f, "a sequence breaker ends the run");
    dry_breakers.clear();
    Seen reply;                                          // the reply counts as context too
    for (int x : {1, 2}) reply.push(x);
    const Dist d3 = make_dist(v, ids, dry, reply, {7, 1, 2, 3, 9});
    check(prob(d3, 3) < prob(d0, 3) * 0.2f, "DRY sees the prompt's tail followed by the reply");
    puts(fails ? "FAILURES" : "all ok");
    return fails != 0;
}
