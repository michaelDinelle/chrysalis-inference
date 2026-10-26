#include "engine/metal/flash-next/mtl.h"
#include <functional>
#include <random>

struct KVArgs { uint32_t T, pos0, layer, nsplit; };

int main(int argc, char **argv) {
    const uint32_t U = 1024, pos0 = argc > 1 ? atoi(argv[1]) : 30720;
    Mtl m; m.init();
    std::mt19937 rng(1);
    auto buf = [&](size_t floats, float scale) {
        View v = m.alloc(floats * 4);
        for (size_t i = 0; i < floats; i++) v.host<float>()[i] = std::uniform_real_distribution<float>(-scale, scale)(rng);
        return v;
    };
    auto time = [&](const char *name, int reps, const std::function<void(Enc &)> &f) {
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            id<MTLCommandBuffer> cb = [m.q commandBuffer];
            Enc e; e.begin(m, cb);
            for (int i = 0; i < reps; i++) { f(e); e.barrier(); }
            e.end();
            [cb commit]; [cb waitUntilCompleted];
            best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / reps);
        }
        printf("%-34s %9.3f ms\n", name, best * 1e3);
    };
    const uint32_t QKV = 10240;
    View raw = buf((size_t)U * QKV, 1), convw = buf(QKV * 4, 0.5), state = buf(QKV * 3, 1), qn = buf((size_t)U * 2048, 0.1),
         kn = buf((size_t)U * 2048, 0.1), vc = buf((size_t)U * 6144, 1), ab = buf(U * 96, 1), A = buf(48, 0.5), dt = buf(48, 0.5),
         S = buf((size_t)48 * 128 * 128, 0.01), z = buf((size_t)U * 6144, 1), w = buf(128, 1);
    time("gdn_conv_pre", 3, [&](Enc &e) { e.k("gdn_conv_pre").b(0, raw).c(1, U).b(2, convw).b(3, state).b(4, qn).b(5, kn).b(6, vc).grid(80, U / 16, 1, 128); });
    time("gdn_scan", 3, [&](Enc &e) { e.k("gdn_scan").b(0, qn).b(1, kn).b(2, vc).b(3, ab).b(4, A).b(5, dt).b(6, S).c(7, U).grid(48, 4, 1, 128); });
    time("gdn_out", 3, [&](Enc &e) { e.k("gdn_out").b(0, vc).b(1, z).b(2, w).grid(48, U, 1, 128); });
    const uint32_t cells = pos0 + U, pages = (cells + 63) / 64, nblk = cells / 4;
    const size_t KV_PAGE = (size_t)(2 * 2 * 64 * 256 * 2 + 16 * 128 * 2) * 12;
    View kv = m.alloc((size_t)pages * KV_PAGE), pt = m.alloc(pages * 4);
    for (uint32_t p = 0; p < pages; p++) pt.host<int>()[p] = p;
    for (size_t i = 0; i < (size_t)pages * KV_PAGE / 2; i++) kv.host<__fp16>()[i] = (__fp16)std::uniform_real_distribution<float>(-1, 1)(rng);
    View Q = buf((size_t)U * 24 * 256, 0.06), Qi = buf((size_t)U * 4 * 128, 0.1), score = m.alloc((size_t)U * nblk * 4),
         sel = m.alloc((size_t)U * 513 * 4), nsel = m.alloc(U * 4), po = m.alloc((size_t)U * 24 * 256 * 4), pml = m.alloc((size_t)U * 24 * 2 * 4),
         qg = buf((size_t)U * 24 * 512, 1), att = m.alloc((size_t)U * 6144 * 4), atth = m.alloc((size_t)U * 6144 * 2);
    KVArgs ka{U, pos0, 0, 1};
    time("idx_score", 2, [&](Enc &e) { e.k("idx_score").c(0, ka).b(1, Qi).b(2, kv).b(3, pt).b(4, score).c(5, nblk).grid((nblk + 255) / 256, U, 1, 256); });
    time("idx_select", 2, [&](Enc &e) { e.k("idx_select").c(0, ka).b(1, score).c(2, nblk).b(3, sel).b(4, nsel).grid(U, 1024); });
    time("attn_dec (nsplit 1)", 2, [&](Enc &e) { e.k("attn_dec").c(0, ka).b(1, Q).b(2, kv).b(3, pt).b(4, sel).b(5, nsel).b(6, po).b(7, pml).grid(1, U, 2, 384); });
    time("attn_combine", 2, [&](Enc &e) { e.k("attn_combine").c(0, ka).b(1, po).b(2, pml).b(3, qg).b(4, att).b(5, atth).grid(24, U, 1, 256); });
    View res = buf((size_t)U * 10240, 1), gam = buf(10240, 1), xn = m.alloc((size_t)U * 10240 * 4), xh = m.alloc((size_t)U * 10240 * 2),
         gate = buf((size_t)U * 10240, 1), mx = m.alloc((size_t)U * 2560 * 4), mh = m.alloc((size_t)U * 2560 * 2), inj = buf(U * 4, 1), blk = buf(U * 2560, 1);
    uint32_t mode = 1, Tn = U;
    time("hc_norm", 3, [&](Enc &e) { e.k("hc_norm").b(0, res).b(1, gam).b(2, xn).b(3, xh).c(4, mode).grid(4, U, 1, 256); });
    time("hc_mix", 3, [&](Enc &e) { e.k("hc_mix").b(0, xn).b(1, gate).b(2, mx).b(3, mh).c(4, Tn).grid((U * 2560 + 255) / 256, 256); });
    struct HcArgs { uint32_t T, inject; } ha{1, 1};
    View wd = buf(320 * 10240 / 2, 0.01), wi = buf(4 * 10240 / 2, 0.01), wu = buf(10240 * 320 / 2, 0.01), lo = buf(320, 1), injd = m.alloc(64), inv = buf(4, 1);
    for (auto [nm, R] : {std::pair<const char *, int>{"hc_pre1", 1}, {"hc_pre", 2}, {"hc_pre4", 4}})
        time((std::string(nm) + " (T=1, 3.5 MB Q8_0)").c_str(), 20, [&](Enc &e) { e.k(nm).c(0, ha).b(1, res).b(2, gam).b(3, wd).b(4, wi).b(5, lo).b(6, injd).b(7, inv).grid((324 + R - 1) / R, 128); });
    time("hc_post (T=1, 3.5 MB Q8_0)", 20, [&](Enc &e) { e.k("hc_post").c(0, ha).b(1, res).b(2, gam).b(3, inv).b(4, lo).b(5, wu).b(6, mx).b(7, mh).grid(640, 128); });
    time("hc_combine", 3, [&](Enc &e) { e.k("hc_combine").b(0, res).b(1, inj).b(2, blk).c(3, Tn).grid((U * 10240 + 255) / 256, 256); });
    struct MoeArgs { uint32_t T, layer, phase, ggml; };
    const int NX = 60;
    View slots = m.alloc((size_t)NX * XSLOT), table = m.alloc((size_t)NL * NE * 8), fmt = m.alloc((size_t)NL * NE), ids = m.alloc(10 * 4),
         xm = buf(2560, 1), hb = buf(10 * 640, 1), xy = m.alloc(10 * 2560 * 4), done = m.alloc(10 * 4);
    for (size_t i = 0; i < (size_t)NX * XSLOT / 2; i++) slots.host<uint16_t>()[i] = 0x2000 | (rng() & 0x13ff);
    for (int k = 0; k < 10; k++) ids.host<int>()[k] = k;
    for (int l = 0; l < 6; l++) for (int k = 0; k < 10; k++) table.host<uint64_t>()[l * NE + k] = slots.gpu() + (size_t)(l * 10 + k) * XSLOT;
    for (int f : {T_IQ4_XS, T_Q3_K, T_Q2_K}) {
        memset(fmt.host(), f, (size_t)NL * NE);
        double gu = 0, dn = 0;
        for (int l = 0; l < 6; l++) {
            const MoeArgs a{1, (uint32_t)l, 0, 0};
            double best = 1e9, bestd = 1e9;
            for (int r = 0; r < 3; r++) {
                id<MTLCommandBuffer> cb = [m.q commandBuffer];
                Enc e; e.begin(m, cb);
                e.k("moe_gu").c(0, a).b(1, table).b(2, ids).b(3, xm).b(4, hb).b(5, done).b(6, fmt).grid(10 * 640 / 16, 128);
                e.end(); [cb commit]; [cb waitUntilCompleted];
                best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
                cb = [m.q commandBuffer];
                Enc d; d.begin(m, cb);
                d.k("moe_down").c(0, a).b(1, table).b(2, ids).b(3, hb).b(4, xy).b(5, done).grid(10 * 2560 / 16, 128);
                d.end(); [cb commit]; [cb waitUntilCompleted];
                bestd = std::min(bestd, cb.GPUEndTime - cb.GPUStartTime);
            }
            gu += best / 6; dn += bestd / 6;
        }
        printf("moe_gu %-6s (10 picks)            %9.3f ms %6.0f GB/s | moe_down iq4_nl %7.3f ms %6.0f GB/s\n", f == T_IQ4_XS ? "iq4_xs" : f == T_Q3_K ? "q3_k" : "q2_k",
               gu * 1e3, 10 * 2 * xgu_bytes(f) / gu / 1e9, dn * 1e3, 10 * XDN / dn / 1e9);
    }
}
