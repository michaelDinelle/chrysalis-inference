// make_dist (engine/sampler.h) on cases from stdin, for web/sampler.test.ts: the web UI's copy of the sampler must
// give the odds the engine draws from. Per case: SAMPLE_N logits (descending) and their ids, the reply's ids, the DRY
// breaker ids, then temperature top_p min_p presence_penalty top_k dry_multiplier dry_base dry_allowed_length and the
// four stage ids in order. Writes each case's distribution as "n id p id p ...".
//   build/sampler_cases < cases.txt
#include "../engine/sampler.h"
#include <cstdio>

int main() {
    int cases;
    if (scanf("%d", &cases) != 1) return 1;
    for (int c = 0; c < cases; c++) {
        float v[SAMPLE_N];
        int ids[SAMPLE_N], n;
        for (float &x : v) if (scanf("%f", &x) != 1) return 1;
        for (int &x : ids) if (scanf("%d", &x) != 1) return 1;
        Seen seen;
        if (scanf("%d", &n) != 1) return 1;
        for (int i = 0, t; i < n; i++) { if (scanf("%d", &t) != 1) return 1; seen.push(t); }
        dry_breakers.assign(1 << 16, 0);
        if (scanf("%d", &n) != 1) return 1;
        for (int i = 0, t; i < n; i++) { if (scanf("%d", &t) != 1 || t < 0 || t >= 1 << 16) return 1; dry_breakers[t] = 1; }
        Sampling sp;
        if (scanf("%f %f %f %f %d %f %f %d", &sp.temperature, &sp.top_p, &sp.min_p, &sp.presence_penalty, &sp.top_k,
                  &sp.dry_multiplier, &sp.dry_base, &sp.dry_allowed_length) != 8) return 1;
        for (int &s : sp.order) if (scanf("%d", &s) != 1) return 1;
        const Dist d = make_dist(v, ids, sp, seen, {});
        printf("%d", d.n);
        for (int i = 0; i < d.n; i++) printf(" %d %.9g", d.id[i], d.p[i]);
        printf("\n");
    }
    return 0;
}
