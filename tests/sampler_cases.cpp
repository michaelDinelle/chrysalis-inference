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
        std::vector<int> ctx, brk;
        if (scanf("%d", &n) != 1) return 1;
        ctx.resize(n);
        for (int &t : ctx) if (scanf("%d", &t) != 1) return 1;
        if (scanf("%d", &n) != 1) return 1;
        for (int i = 0, t; i < n; i++) {
            if (scanf("%d", &t) != 1) return 1;
            brk.push_back(1);
            brk.push_back(t);
        }
        Sampling sp;
        int xtc;
        float mu;
        if (scanf("%f %f %f %d %f %f %f %f %f %f %d %f %f %f %f %f %d %d %d %f %f %d %d %f %d", &sp.temperature, &sp.dynatemp_range,
                  &sp.dynatemp_exponent, &sp.top_k, &sp.top_p, &sp.min_p, &sp.typical_p, &sp.xtc_probability, &sp.xtc_threshold,
                  &sp.top_n_sigma, &sp.repeat_last_n, &sp.repeat_penalty, &sp.presence_penalty, &sp.frequency_penalty, &sp.dry_multiplier,
                  &sp.dry_base, &sp.dry_allowed_length, &sp.dry_penalty_last_n, &sp.mirostat, &sp.mirostat_tau, &sp.mirostat_eta,
                  &sp.min_keep, &xtc, &mu, &sp.n_order) != 25 || sp.n_order < 0 || sp.n_order > MAX_ORDER)
            return 1;
        for (int i = 0; i < sp.n_order; i++) if (scanf("%d", &sp.order[i]) != 1) return 1;
        Seen st;
        st.begin(ctx, sp);
        st.mu = mu;
        st.set_breakers(brk.data(), (int)brk.size());
        const Dist d = make_dist(v, ids, nullptr, sp, st, xtc ? 0.0f : 2.0f);
        printf("%d", d.n);
        for (int i = 0; i < d.n; i++) printf(" %d %.9g", d.id[i], d.p[i]);
        printf("\n");
    }
    return 0;
}
