#include "../../engine/gpu/flash-next/kernels.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

static float h2f(half h) { return __half2float(h); }

static std::vector<int> mrope_sections() {
    std::vector<int> sec(ROT / 2, 0);
    const int section[3] = {11, 11, 10};
    for (int dim = 1; dim <= 2; dim++)
        for (int f = dim; f < section[dim] * 3; f += 3) sec[f] = dim;
    return sec;
}

int main() {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    const int N = 6000, pages = (N + PAGE - 1) / PAGE;
    uint8_t *pool; CK(cudaMalloc(&pool, (size_t)pages * KV_PAGE)); CK(cudaMemset(pool, 0, (size_t)pages * KV_PAGE));
    std::vector<int> pg(pages); for (int i = 0; i < pages; i++) pg[i] = i;
    int *dpg; CK(cudaMalloc(&dpg, pages * 4)); CK(cudaMemcpy(dpg, pg.data(), pages * 4, cudaMemcpyHostToDevice));
    KVRef kv{pool, dpg};
    const int layer = 3;
    const int C = 256;
    std::vector<float> qg((size_t)C * NH * 2 * HD), k((size_t)C * NKV * HD), v((size_t)C * NKV * HD), kr((size_t)C * IDX_D), qr((size_t)C * IDX_H * IDX_D);
    std::vector<float> ones(HD, 1.0f);
    float *dqg, *dk, *dv, *dkr, *dqr, *dones; half *dQ16, *dq16;
    CK(cudaMalloc(&dqg, qg.size() * 4)); CK(cudaMalloc(&dk, k.size() * 4)); CK(cudaMalloc(&dv, v.size() * 4));
    CK(cudaMalloc(&dkr, kr.size() * 4)); CK(cudaMalloc(&dqr, qr.size() * 4)); CK(cudaMalloc(&dones, HD * 4));
    CK(cudaMalloc(&dQ16, (size_t)C * NH * HD * 2)); CK(cudaMalloc(&dq16, (size_t)C * IDX_H * IDX_D * 2));
    CK(cudaMemcpy(dones, ones.data(), HD * 4, cudaMemcpyHostToDevice));
    half *tail, *praw;
    CK(cudaMalloc(&tail, IDX_TAIL * 2)); CK(cudaMalloc(&praw, (size_t)C * IDX_D * 2));
    std::vector<float> all_kr((size_t)N * IDX_D);
    std::vector<int> P3((size_t)N * 3);
    for (int c = 0; c < N; c++) {
        const int img = c - 1500;
        const bool in_img = img >= 0 && img < 1200;
        const int p = c < 1500 ? c : c - 600;
        P3[c * 3] = in_img ? 1500 : p;
        P3[c * 3 + 1] = in_img ? 1500 + 3 * (img / 40) : p;
        P3[c * 3 + 2] = in_img ? 1500 + 7 * (img % 40) : p;
    }
    int *dP3; CK(cudaMalloc(&dP3, P3.size() * 4)); CK(cudaMemcpy(dP3, P3.data(), P3.size() * 4, cudaMemcpyHostToDevice));
    auto tail_of = [&](int p0) { const int c = p0 / KPOOL * KPOOL; return Pos3{P3[c * 3], P3[c * 3 + 1], P3[c * 3 + 2]}; };
    std::vector<int> sizes;
    for (int p0 = 0, i = 0; p0 < N; i++) { const int T = std::min({C, N - p0, (int)(37 + 61 * (i % 4) + (i % 3))}); sizes.push_back(T); p0 += T; }
    for (int p0 = 0, si = 0; p0 < N; p0 += sizes[si++]) {
        const int T = sizes[si];
        for (auto &x : qg) x = nd(rng); for (auto &x : k) x = nd(rng); for (auto &x : v) x = nd(rng);
        for (auto &x : kr) x = nd(rng); for (auto &x : qr) x = nd(rng);
        CK(cudaMemcpy(dqg, qg.data(), qg.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dkr, kr.data(), kr.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dqr, qr.data(), qr.size() * 4, cudaMemcpyHostToDevice));
        std::copy(kr.begin(), kr.begin() + (size_t)T * IDX_D, all_kr.begin() + (size_t)p0 * IDX_D);
        attn_prep(dqg, dk, dv, dones, dones, T, p0, 0, dP3 + 3 * p0, kv, layer, dQ16, st);
        idx_prep(dkr, dqr, dones, dones, T, p0, 0, dP3 + 3 * p0, tail_of(p0), kv, layer, tail, praw, true, dq16, st);
    }
    CK(cudaStreamSynchronize(st));
    int fails = 0;
    {
        const int c0 = 1537, T = 200;
        for (auto &x : qg) x = nd(rng);
        CK(cudaMemcpy(dqg, qg.data(), qg.size() * 4, cudaMemcpyHostToDevice));
        attn_prep(dqg, dk, dv, dones, dones, T, c0, 0, dP3 + 3 * c0, kv, 0, dQ16, st);
        std::vector<half> got((size_t)T * NH * HD);
        CK(cudaMemcpy(got.data(), dQ16, got.size() * 2, cudaMemcpyDeviceToHost));
        const std::vector<int> sec = mrope_sections();
        double worst = 0;
        for (int t = 0; t < T; t++)
            for (int h = 0; h < NH; h++) {
                const float *x = &qg[((size_t)t * NH + h) * 2 * HD];
                double ss = 0;
                for (int i = 0; i < HD; i++) ss += (double)x[i] * x[i];
                const float r = 1.0f / sqrtf((float)(ss / HD) + EPS);
                for (int i = 0; i < HD; i++) {
                    float y = x[i] * r;
                    if (i < ROT) {
                        const int f = i % (ROT / 2);
                        const float th = P3[(c0 + t) * 3 + sec[f]] * powf(ROPE_BASE, -2.0f * f / ROT), c = cosf(th), sn = sinf(th);
                        y = i < ROT / 2 ? x[i] * r * c - x[i + ROT / 2] * r * sn : x[i] * r * c + x[i - ROT / 2] * r * sn;
                    }
                    worst = std::max(worst, (double)fabsf(h2f(got[((size_t)t * NH + h) * HD + i]) - y * 0.0625f));
                }
            }
        printf("M-RoPE queries of %d picture cells vs the host: max diff %.2e %s\n", T, worst, worst < 1e-3 ? "ok" : "FAIL");
        fails += worst >= 1e-3;
    }
    for (const int P : {1999, 1997}) {
        for (auto &x : qg) x = nd(rng); for (auto &x : k) x = nd(rng); for (auto &x : v) x = nd(rng);
        CK(cudaMemcpy(dqg, qg.data(), (size_t)NH * 2 * HD * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), (size_t)NKV * HD * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), (size_t)NKV * HD * 4, cudaMemcpyHostToDevice));
        attn_prep(dqg, dk, dv, dones, dones, 1, P, P, nullptr, kv, layer, dQ16, st);
        float *po, *pml; half *o1, *o2;
        CK(cudaMalloc(&po, (size_t)PART_ROWS * NH * HD * 4)); CK(cudaMalloc(&pml, (size_t)PART_ROWS * NH * 2 * 4));
        CK(cudaMalloc(&o1, NH * HD * 2)); CK(cudaMalloc(&o2, NH * HD * 2));
        AttnArgs a; a.Q16 = dQ16; a.kv = kv; a.layer = layer; a.T = 1; a.pos0 = P; a.qg = dqg; a.part_o = po; a.part_ml = pml;
        attention(a, o1, QAct{nullptr, nullptr}, st);
        std::vector<int> sel(SEL_BLOCKS + 1, -1);
        const int B = (P + 1) / KPOOL;
        for (int b = 0; b < B; b++) sel[b] = b;
        if ((P + 1) % KPOOL) sel[B] = B;
        int *dsel; CK(cudaMalloc(&dsel, sel.size() * 4)); CK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
        AttnArgs a2 = a;
        attention_sel(a2, dsel, o2, QAct{nullptr, nullptr}, st);
        std::vector<half> h1(NH * HD), h2(NH * HD);
        CK(cudaMemcpy(h1.data(), o1, NH * HD * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(), o2, NH * HD * 2, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (int i = 0; i < NH * HD; i++) { const double d = h2f(h1[i]) - h2f(h2[i]); num += d * d; den += h2f(h1[i]) * h2f(h1[i]); }
        const double rel = sqrt(num / den);
        printf("dense vs sparse (all %d blocks%s): rel diff %.2e %s\n", B, (P + 1) % KPOOL ? " + tail" : "", rel, rel < 5e-3 ? "ok" : "FAIL");
        fails += rel >= 5e-3;
    }
    {
        const int T = 37, P0 = 5900;
        for (auto &x : qg) x = nd(rng); for (auto &x : k) x = nd(rng); for (auto &x : v) x = nd(rng);
        CK(cudaMemcpy(dqg, qg.data(), qg.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        attn_prep(dqg, dk, dv, dones, dones, T, P0, P0, nullptr, kv, layer, dQ16, st);
        std::vector<int> sel((size_t)T * (SEL_BLOCKS + 1), -1);
        for (int t = 0; t < T; t++) {
            const int p = P0 + t, vis = (p + 1) / KPOOL;
            std::vector<int> blocks(vis);
            for (int b = 0; b < vis; b++) blocks[b] = b;
            std::shuffle(blocks.begin(), blocks.end(), rng);
            const int n = std::min({vis, SEL_BLOCKS, 100 + 13 * t});
            std::sort(blocks.begin(), blocks.begin() + n);
            for (int i = 0; i < n; i++) sel[(size_t)t * (SEL_BLOCKS + 1) + i] = blocks[i];
            if ((p + 1) % KPOOL) sel[(size_t)t * (SEL_BLOCKS + 1) + n] = vis;
        }
        float *po, *pml; half *oa, *o1;
        CK(cudaMalloc(&po, (size_t)PART_ROWS * NH * HD * 4)); CK(cudaMalloc(&pml, (size_t)PART_ROWS * NH * 2 * 4));
        CK(cudaMalloc(&oa, (size_t)T * NH * HD * 2)); CK(cudaMalloc(&o1, NH * HD * 2));
        int *dsel; CK(cudaMalloc(&dsel, sel.size() * 4)); CK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
        AttnArgs a; a.Q16 = dQ16; a.kv = kv; a.layer = layer; a.T = T; a.pos0 = P0; a.qg = dqg; a.part_o = po; a.part_ml = pml;
        attention_sel(a, dsel, oa, QAct{nullptr, nullptr}, st);
        std::vector<half> ha((size_t)T * NH * HD), h1(NH * HD);
        CK(cudaMemcpy(ha.data(), oa, ha.size() * 2, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (int t = 0; t < T; t++) {
            AttnArgs a1 = a;
            a1.Q16 = dQ16 + (size_t)t * NH * HD; a1.qg = dqg + (size_t)t * NH * 2 * HD; a1.T = 1; a1.pos0 = P0 + t;
            attention_sel(a1, dsel + (size_t)t * (SEL_BLOCKS + 1), o1, QAct{nullptr, nullptr}, st);
            CK(cudaMemcpy(h1.data(), o1, NH * HD * 2, cudaMemcpyDeviceToHost));
            for (int i = 0; i < NH * HD; i++) {
                const double d = h2f(ha[(size_t)t * NH * HD + i]) - h2f(h1[i]);
                num += d * d; den += h2f(h1[i]) * h2f(h1[i]);
            }
        }
        const double rel = sqrt(num / den);
        printf("sparse, %d tokens at once vs one at a time (random block lists, tails): rel diff %.2e %s\n", T, rel, rel < 2e-3 ? "ok" : "FAIL");
        fails += rel >= 2e-3;
    }
    {
        const int G = 4, P0 = N - G;
        for (auto &x : qr) x = nd(rng);
        CK(cudaMemcpy(dqr, qr.data(), (size_t)G * IDX_H * IDX_D * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dkr, all_kr.data() + (size_t)P0 * IDX_D, (size_t)G * IDX_D * 4, cudaMemcpyHostToDevice));
        idx_prep(dkr, dqr, dones, dones, G, P0, 0, dP3 + 3 * P0, tail_of(P0), kv, layer, tail, praw, false, dq16, st);
        const int nblk = N / KPOOL;
        float *score; int *dsel;
        CK(cudaMalloc(&score, (size_t)G * nblk * 4)); CK(cudaMalloc(&dsel, (size_t)G * (SEL_BLOCKS + 1) * 4));
        uint32_t *dhist; CK(cudaMalloc(&dhist, (size_t)G * 1024 * 4));
        idx_select(dq16, P0, G, nblk, kv, layer, score, dhist, dsel, st);
        {
            int *rsel; CK(cudaMalloc(&rsel, (size_t)G * (SEL_BLOCKS + 1) * 4));
            int bad = 0;
            for (int pos_end : {300, 2047, 2049, 4000, N}) {
                const int p0 = pos_end - G, nb = pos_end / KPOOL;
                idx_select(dq16, p0, G, nb, kv, layer, score, dhist, dsel, st);
                std::vector<int> a1((size_t)G * (SEL_BLOCKS + 1)), a2(a1.size());
                CK(cudaMemcpy(a1.data(), dsel, a1.size() * 4, cudaMemcpyDeviceToHost));
                idx_select_ref(dq16, p0, G, nb, kv, layer, score, rsel, st);
                CK(cudaMemcpy(a2.data(), rsel, a2.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t k = 0; k < a1.size(); k++) bad += a1[k] != a2[k];
                for (int t = 0; t < G; t++) {
                    idx_select(dq16 + (size_t)t * IDX_H * IDX_D, p0 + t, 1, nb, kv, layer, score, dhist, dsel, st);
                    CK(cudaMemcpy(a1.data(), dsel, (SEL_BLOCKS + 1) * 4, cudaMemcpyDeviceToHost));
                    for (int k = 0; k <= SEL_BLOCKS; k++) bad += a1[k] != a2[(size_t)t * (SEL_BLOCKS + 1) + k];
                }
            }
            {
                half *zq; CK(cudaMalloc(&zq, (size_t)G * IDX_H * IDX_D * 2)); CK(cudaMemset(zq, 0, (size_t)G * IDX_H * IDX_D * 2));
                idx_select(zq, P0, G, nblk, kv, layer, score, dhist, dsel, st);
                std::vector<int> a1((size_t)G * (SEL_BLOCKS + 1)), a2(a1.size());
                CK(cudaMemcpy(a1.data(), dsel, a1.size() * 4, cudaMemcpyDeviceToHost));
                idx_select_ref(zq, P0, G, nblk, kv, layer, score, rsel, st);
                CK(cudaMemcpy(a2.data(), rsel, a2.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t k = 0; k < a1.size(); k++) bad += a1[k] != a2[k];
                bad += a1[0] != 0 || a1[SEL_BLOCKS - 1] != SEL_BLOCKS - 1;
            }
            printf("new selection vs the reference kernels (5 context lengths x %d queries, at once and one at a time, plus all ties): %d entries differ %s\n", G, bad, bad ? "FAIL" : "ok");
            fails += bad != 0;
            idx_select(dq16, P0, G, nblk, kv, layer, score, dhist, dsel, st);
        }
        std::vector<int> sel((size_t)G * (SEL_BLOCKS + 1));
        CK(cudaMemcpy(sel.data(), dsel, sel.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<half> q16((size_t)G * IDX_H * IDX_D);
        CK(cudaMemcpy(q16.data(), dq16, q16.size() * 2, cudaMemcpyDeviceToHost));
        const std::vector<int> sec = mrope_sections();
        std::vector<half> pk((size_t)nblk * IDX_D);
        int pool_bad = 0;
        for (int b = 0; b < nblk; b++) {
            const int cell = b * KPOOL;
            const size_t off = (size_t)pg[cell / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + NKV * KV_HEAD + (size_t)((cell % PAGE) / KPOOL) * IDX_D * 2;
            CK(cudaMemcpy(&pk[(size_t)b * IDX_D], pool + off, IDX_D * 2, cudaMemcpyDeviceToHost));
            float x[IDX_D];
            double ss = 0;
            for (int i = 0; i < IDX_D; i++) {
                float v = 0;
                for (int c = 0; c < KPOOL; c++) v += __half2float(__float2half(all_kr[(size_t)(cell + c) * IDX_D + i]));
                x[i] = v / KPOOL; ss += x[i] * x[i];
            }
            const float inv = 1.0f / sqrtf((float)(ss / IDX_D) + EPS);
            for (int i = 0; i < IDX_D; i++) x[i] *= inv;
            for (int i = 0; i < ROT / 2; i++) {
                const float th = P3[cell * 3 + sec[i]] * powf(ROPE_BASE, -2.0f * i / ROT), c0 = cosf(th), s0 = sinf(th);
                const float a = x[i], bb = x[i + ROT / 2];
                x[i] = a * c0 - bb * s0; x[i + ROT / 2] = bb * c0 + a * s0;
            }
            for (int i = 0; i < IDX_D; i++) pool_bad += fabsf(__half2float(pk[(size_t)b * IDX_D + i]) - x[i]) > 2e-3f * (1 + fabsf(x[i]));
        }
        printf("pooled keys of %d blocks (passes of uneven sizes, tail carried over) vs the host: %d values off %s\n", nblk, pool_bad,
               pool_bad ? "FAIL" : "ok");
        fails += pool_bad != 0;
        for (int g = 0; g < G; g++) {
            const int pos = P0 + g, nvis = (pos + 1) / KPOOL;
            std::vector<std::pair<float, int>> sc(nvis);
            for (int b = 0; b < nvis; b++) {
                float s = 0;
                for (int h = 0; h < IDX_H; h++) {
                    float d = 0;
                    for (int i = 0; i < IDX_D; i++) d += h2f(q16[((size_t)g * IDX_H + h) * IDX_D + i]) * h2f(pk[(size_t)b * IDX_D + i]);
                    s += std::max(d, 0.0f);
                }
                sc[b] = {s * 0.08838834764831845f, b};
            }
            std::stable_sort(sc.begin(), sc.end(), [](auto &x, auto &y) { return x.first > y.first; });
            std::vector<int> want;
            for (int i = 0; i < std::min(nvis, SEL_BLOCKS); i++) want.push_back(sc[i].second);
            std::sort(want.begin(), want.end());
            const int *got = &sel[(size_t)g * (SEL_BLOCKS + 1)];
            int same = 0, agree_thr = 0;
            std::vector<int> gotv(got, got + SEL_BLOCKS);
            std::vector<int> inter;
            std::set_intersection(want.begin(), want.end(), gotv.begin(), gotv.end(), std::back_inserter(inter));
            same = (int)inter.size();
            const float thr = sc[std::min(nvis, SEL_BLOCKS) - 1].first;
            for (int i = 0; i < SEL_BLOCKS; i++) {
                if (std::binary_search(want.begin(), want.end(), gotv[i])) continue;
                for (auto &e : sc) if (e.second == gotv[i]) { agree_thr += fabsf(e.first - thr) < 1e-3f * fabsf(thr) + 1e-6f; break; }
            }
            const bool ascending = std::is_sorted(gotv.begin(), gotv.end());
            const int tail = got[SEL_BLOCKS];
            const bool tail_ok = ((pos + 1) % KPOOL) ? tail == (pos + 1) / KPOOL : tail == -1;
            const bool ok = ascending && tail_ok && same + agree_thr >= SEL_BLOCKS;
            printf("selection pos %d: %d/%d blocks agree (+%d threshold ties), ascending %d, tail %d %s\n", pos, same, SEL_BLOCKS,
                   agree_thr, ascending, tail, ok ? "ok" : "FAIL");
            fails += !ok;
        }
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
