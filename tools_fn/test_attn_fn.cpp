// QSA checks on random data: (1) sparse attention over every visible block equals dense attention (context <= 2048
// cells, where the top-512 selection is everything); (2) the GPU's top-512 block selection equals a CPU reference
// computed from the same pooled keys and indexer queries.
#include "../engine_fn/kernels.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

static float h2f(half h) { return __half2float(h); }

int main() {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    hipStream_t st; CK(hipStreamCreate(&st));
    const int N = 6000, pages = (N + PAGE - 1) / PAGE;
    uint8_t *pool; CK(hipMalloc(&pool, (size_t)pages * KV_PAGE)); CK(hipMemset(pool, 0, (size_t)pages * KV_PAGE));
    std::vector<int> pg(pages); for (int i = 0; i < pages; i++) pg[i] = i;
    int *dpg; CK(hipMalloc(&dpg, pages * 4)); CK(hipMemcpy(dpg, pg.data(), pages * 4, hipMemcpyHostToDevice));
    KVRef kv{pool, dpg};
    const int layer = 3;
    // fill the cache: K/V and the indexer keys of every cell, in chunks
    const int C = 256;
    std::vector<float> qg((size_t)C * NH * 2 * HD), k((size_t)C * NKV * HD), v((size_t)C * NKV * HD), kr((size_t)C * IDX_D), qr((size_t)C * IDX_H * IDX_D);
    std::vector<float> ones(HD, 1.0f);
    float *dqg, *dk, *dv, *dkr, *dqr, *dones; half *dQ16, *dq16;
    CK(hipMalloc(&dqg, qg.size() * 4)); CK(hipMalloc(&dk, k.size() * 4)); CK(hipMalloc(&dv, v.size() * 4));
    CK(hipMalloc(&dkr, kr.size() * 4)); CK(hipMalloc(&dqr, qr.size() * 4)); CK(hipMalloc(&dones, HD * 4));
    CK(hipMalloc(&dQ16, (size_t)C * NH * HD * 2)); CK(hipMalloc(&dq16, (size_t)C * IDX_H * IDX_D * 2));
    CK(hipMemcpy(dones, ones.data(), HD * 4, hipMemcpyHostToDevice));
    for (int p0 = 0; p0 < N; p0 += C) {
        const int T = std::min(C, N - p0);
        for (auto &x : qg) x = nd(rng); for (auto &x : k) x = nd(rng); for (auto &x : v) x = nd(rng);
        for (auto &x : kr) x = nd(rng); for (auto &x : qr) x = nd(rng);
        CK(hipMemcpy(dqg, qg.data(), qg.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dkr, kr.data(), kr.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dqr, qr.data(), qr.size() * 4, hipMemcpyHostToDevice));
        attn_prep(dqg, dk, dv, dones, dones, T, p0, kv, layer, dQ16, st);
        idx_prep(dkr, dqr, dones, dones, T, p0, kv, layer, dq16, st);
    }
    CK(hipStreamSynchronize(st));
    int fails = 0;
    // (1) dense vs sparse-all at position P (context P+1 <= 2048 cells)
    {
        const int P = 1999;
        for (auto &x : qg) x = nd(rng); for (auto &x : k) x = nd(rng); for (auto &x : v) x = nd(rng);
        CK(hipMemcpy(dqg, qg.data(), (size_t)NH * 2 * HD * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dk, k.data(), (size_t)NKV * HD * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dv, v.data(), (size_t)NKV * HD * 4, hipMemcpyHostToDevice));
        attn_prep(dqg, dk, dv, dones, dones, 1, P, kv, layer, dQ16, st);   // overwrite cell P as the query token
        float *po, *pml; half *o1, *o2;
        CK(hipMalloc(&po, (size_t)MAX_SPLITS * 8 * NH * HD * 4)); CK(hipMalloc(&pml, (size_t)MAX_SPLITS * 8 * NH * 2 * 4));
        CK(hipMalloc(&o1, NH * HD * 2)); CK(hipMalloc(&o2, NH * HD * 2));
        AttnArgs a; a.Q16 = dQ16; a.kv = kv; a.layer = layer; a.T = 1; a.pos0 = P; a.qg = dqg; a.part_o = po; a.part_ml = pml;
        attention(a, o1, QAct{nullptr, nullptr}, st);
        std::vector<int> sel(SEL_BLOCKS + 1, -1);
        const int B = (P + 1) / KPOOL;
        for (int b = 0; b < B; b++) sel[b] = b;
        if ((P + 1) % KPOOL) sel[B] = B;
        int *dsel; CK(hipMalloc(&dsel, sel.size() * 4)); CK(hipMemcpy(dsel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice));
        AttnArgs a2 = a;
        attention_sel(a2, dsel, o2, QAct{nullptr, nullptr}, st);
        std::vector<half> h1(NH * HD), h2(NH * HD);
        CK(hipMemcpy(h1.data(), o1, NH * HD * 2, hipMemcpyDeviceToHost)); CK(hipMemcpy(h2.data(), o2, NH * HD * 2, hipMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (int i = 0; i < NH * HD; i++) { const double d = h2f(h1[i]) - h2f(h2[i]); num += d * d; den += h2f(h1[i]) * h2f(h1[i]); }
        const double rel = sqrt(num / den);
        printf("dense vs sparse (all %d blocks): rel diff %.2e %s\n", B, rel, rel < 5e-3 ? "ok" : "FAIL");
        fails += rel >= 5e-3;
    }
    // (2) selection at a long context: GPU vs CPU over the same pooled keys
    {
        const int G = 4, P0 = N - G;                     // queries at the last G positions
        for (auto &x : qr) x = nd(rng);
        CK(hipMemcpy(dqr, qr.data(), (size_t)G * IDX_H * IDX_D * 4, hipMemcpyHostToDevice));
        // indexer queries via idx_prep (also rewrites these cells' raw keys; pooled keys of complete blocks stay)
        CK(hipMemcpy(dkr, kr.data(), (size_t)G * IDX_D * 4, hipMemcpyHostToDevice));
        idx_prep(dkr, dqr, dones, dones, G, P0, kv, layer, dq16, st);
        const int nblk = N / KPOOL;
        float *score; int *dsel;
        CK(hipMalloc(&score, (size_t)G * nblk * 4)); CK(hipMalloc(&dsel, (size_t)G * (SEL_BLOCKS + 1) * 4));
        idx_select(dq16, P0, G, nblk, kv, layer, score, dsel, st);
        std::vector<int> sel((size_t)G * (SEL_BLOCKS + 1));
        CK(hipMemcpy(sel.data(), dsel, sel.size() * 4, hipMemcpyDeviceToHost));
        std::vector<half> q16((size_t)G * IDX_H * IDX_D);
        CK(hipMemcpy(q16.data(), dq16, q16.size() * 2, hipMemcpyDeviceToHost));
        // pooled keys back from the pages
        std::vector<half> pk((size_t)nblk * IDX_D);
        for (int b = 0; b < nblk; b++) {
            const int cell = b * KPOOL;
            const size_t off = (size_t)pg[cell / PAGE] * KV_PAGE + (size_t)layer * KV_LAYER + NKV * KV_HEAD + IDX_RAW + (size_t)((cell % PAGE) / KPOOL) * IDX_D * 2;
            CK(hipMemcpy(&pk[(size_t)b * IDX_D], pool + off, IDX_D * 2, hipMemcpyDeviceToHost));
        }
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
            // blocks that differ must tie with the threshold score (float rounding between CPU and GPU sums)
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
