#include "engine.h"
#include "prefill.h"
#include <algorithm>
#include <cmath>

struct KVArgs { uint32_t T, pos0, layer, nsplit; };
struct HcArgs2 { uint32_t T, inject; };

void Engine::alloc_prompt_buffers() {
    PMAX = getenv("CHRYSALIS_FN_PMAX") ? atoi(getenv("CHRYSALIS_FN_PMAX")) : std::max(umax, 512);
    U = std::min(1024, PMAX);
    auto f = [&](size_t n) { return (size_t)U * n * 4; };
    const int bt = pf_moe_bt(U);
    std::vector<std::pair<View *, size_t>> parts = {
        {&pb.res, (size_t)PMAX * HCD * 4}, {&pb.tokens, (size_t)PMAX * 4}, {&pb.tokens2, (size_t)PMAX * 4},
        {&pb.ple_rows, (size_t)PMAX * PLE_HEADS * PLE_ROW}, {&pb.xn, f(HCD)}, {&pb.lo, f(HCR)},
        {&pb.gate, f(HCD)}, {&pb.inj, f(HC)}, {&pb.inj2, f(HC)}, {&pb.mx, f(D)}, {&pb.mxh, f(D)}, {&pb.raw, f(QKV)},
        {&pb.z, f(GV * GD)}, {&pb.ab, f(96)}, {&pb.qn, f(GK * GD)}, {&pb.kn, f(GK * GD)}, {&pb.vc, f(GV * GD)}, {&pb.blk, f(D)},
        {&pb.qg, f(2 * NH * HD)}, {&pb.kb, f(NKV * HD)}, {&pb.vb, f(NKV * HD)}, {&pb.qir, f(IDX_H * IDX_D)}, {&pb.kir, f(IDX_D)},
        {&pb.Q, f(NH * HD)}, {&pb.Qi, f(IDX_H * IDX_D)}, {&pb.score, f(max_blocks)}, {&pb.sel, f(SEL_BLOCKS + 1)}, {&pb.nsel, f(1)},
        {&pb.part_o, f(NH * HD)}, {&pb.part_ml, f(NH * 2)}, {&pb.att, f(NH * HD)}, {&pb.atth, f(NH * HD)}, {&pb.rlog, f(NE)},
        {&pb.ids, f(TOPK_E)}, {&pb.wts, f(TOPK_E)}, {&pb.sgv, f(1)}, {&pb.shg, f(FS)}, {&pb.shu, f(FS)}, {&pb.shh, f(FS)},
        {&pb.shout, f(D)}, {&pb.Y, f(D)}, {&pb.emb, f(D)}, {&pb.embh, f(D)}, {&pb.key, f(HCD)}, {&pb.value, f(D)}, {&pb.xconv, f(HCD)},
        {&pb.moe, pf_moe_scratch(U * TOPK_E, bt)}};
    size_t need = 0;
    for (auto &[v, n] : parts) need += (n + 255) & ~size_t(255);
    Arena a;
    a.init(m, need);
    for (auto &[v, n] : parts) *v = a.get(n);
}

void Engine::prompt_pass(Seq &s, const int *tokens, int P) {
    const double t0 = now_s();
    memcpy(b.tokens.host(), tokens, std::min(P, TM) * 4);
    std::thread gather([&] { ple_gather(s, tokens, P, pb.ple_rows.host<uint8_t>()); });
    uint64_t ple_ready = 0;
    View tok = npass++ & 1 ? pb.tokens2 : pb.tokens;
    memcpy(tok.host(), tokens, (size_t)P * 4);
    const bool all = P * TOPK_E >= 2048;
    id<MTLCommandBuffer> cb = [m.q commandBufferWithUnretainedReferences];
    Enc e;
    e.begin(m, cb);
    const uint32_t erb = (uint32_t)embd.rb;
    e.k("embed_q8_0").b(0, tok).b(1, embd.v).c(2, erb).b(3, pb.res).grid(P, 80);
    e.barrier();
    std::vector<uint64_t> routed(NL), done(NL), staged(NL);
    int gl = 0, al = 0;
    const int bt = pf_moe_bt(U);
    static const std::string skips = getenv("CHRYSALIS_FN_SKIP") ? getenv("CHRYSALIS_FN_SKIP") : "";
    auto skip = [&](const char *k) { return skips.find(k) != std::string::npos; };
    for (int l = 0; l < NL; l++) {
        const Layer &w = L[l];
        for (int c0 = 0; c0 < P; c0 += U) {
            const int n = std::min(U, P - c0);
            const uint32_t nn = n, pos0 = s.n + c0;
            View res = pb.res.at((size_t)c0 * HCD * 4);
            auto mm = [&](const WT &wt, View x, int ldx, View y, int ldy) { if (!skip("proj")) pf_mm(e, wt.type, wt.v, wt.rows, wt.cols, wt.rb, x, ldx, y, ldy, n); };
            auto mix = [&](const Mix &mx, View inj) {
                if (skip("mix")) return;
                uint32_t mode = 1;
                e.k("hc_norm").b(0, res).b(1, mx.norm).b(2, pb.xn).b(3, pb.embh).c(4, mode).grid(HC, n, 1, 256);
                e.barrier();
                mm(mx.down, pb.xn, HCD, pb.lo, HCR);
                if (mx.inject.rows) mm(mx.inject, pb.xn, HCD, inj, HC);
                e.barrier();
                const uint32_t nl = n * HCR;
                e.k("hc_lo").b(0, pb.lo).b(1, pb.embh).c(2, nl).grid((nl + 255) / 256, 256);
                e.barrier();
                mm(mx.up, pb.lo, HCR, pb.gate, HCD);
                e.barrier();
                e.k("hc_mix").b(0, pb.xn).b(1, pb.gate).b(2, pb.mx).b(3, pb.mxh).c(4, nn).grid((n * D + 255) / 256, 256);
                e.barrier();
            };
            auto combine = [&](View inj) {
                e.k("hc_combine").b(0, res).b(1, inj).b(2, pb.blk).c(3, nn).grid((n * HCD + 255) / 256, 256);
                e.barrier();
            };
            if (l == PLE_LAYER) {
                if (c0 == 0) {
                    e.end();
                    ple_ready = ++evc;
                    [cb encodeWaitForEvent:ev_c value:ple_ready];
                    e.begin(m, cb);
                }
                const uint32_t nb = n * PLE_HEADS * 5;
                e.k("ple_embed").b(0, pb.ple_rows.at((size_t)c0 * PLE_HEADS * PLE_ROW)).b(1, pb.emb).b(2, pb.embh).c(3, nb).grid((nb + 255) / 256, 256);
                e.barrier();
                mm(w.ple_key, pb.emb, D, pb.key, HCD);
                mm(w.ple_value, pb.emb, D, pb.value, D);
                e.barrier();
                e.k("ple_gate").b(0, pb.key).b(1, pb.value).b(2, res).b(3, w.ple_nk).b(4, w.ple_nq).b(5, w.ple_nc).b(6, pb.xconv).grid(HC, n, 1, 256);
                e.barrier();
                e.k("ple_conv").b(0, pb.xconv).b(1, s.ple).b(2, w.ple_conv).b(3, res).c(4, nn).grid((n * HCD + 255) / 256, 256);
                e.barrier();
                e.k("ple_commit").b(0, s.ple).b(1, pb.xconv).c(2, nn).grid(HCD / 256, 256);
                e.barrier();
            }
            mix(w.hca, pb.inj);
            if (is_attn_layer(l) && skip("attn")) {
                mm(w.o, pb.att, NH * HD, pb.blk, D);
            } else if (!is_attn_layer(l) && skip("gdn")) {
                mm(w.ssm_out, pb.vc, GV * GD, pb.blk, D);
            } else if (is_attn_layer(l)) {
                mm(w.q, pb.mx, D, pb.qg, 2 * NH * HD);
                mm(w.k, pb.mx, D, pb.kb, NKV * HD);
                mm(w.v, pb.mx, D, pb.vb, NKV * HD);
                mm(w.iq, pb.mx, D, pb.qir, IDX_H * IDX_D);
                mm(w.ik, pb.mx, D, pb.kir, IDX_D);
                e.barrier();
                const int nb_end = (pos0 + n) / KPOOL;
                KVArgs ka{nn, pos0, (uint32_t)al, 1};
                View tail = s.tail.at((size_t)al * KPOOL * IDX_D * 2);
                e.k("attn_prep").c(0, ka).b(1, pb.qg).b(2, pb.kb).b(3, pb.vb).b(4, w.qn).b(5, w.kn).b(6, kv).b(7, s.d_pages).b(8, pb.Q).grid(NH + 2 * NKV, n, 1, 256);
                e.k("idx_qprep").c(0, ka).b(1, pb.qir).b(2, w.iqn).b(3, pb.Qi).grid(IDX_H, n, 1, 128);
                const int first = pos0 / KPOOL, last = (pos0 + n) / KPOOL - 1;
                if (last >= first) e.k("idx_pool").c(0, ka).b(1, pb.kir).b(2, tail).b(3, w.ikn).b(4, kv).b(5, s.d_pages).grid(last - first + 1, 128);
                e.barrier();
                e.k("idx_tail").c(0, ka).b(1, pb.kir).b(2, tail).grid(1, 128);
                const uint32_t ld = std::max(1, nb_end);
                if (nb_end > SEL_BLOCKS) e.k("idx_score").c(0, ka).b(1, pb.Qi).b(2, kv).b(3, s.d_pages).b(4, pb.score).c(5, ld).grid((ld + 255) / 256, n, 1, 256);
                e.barrier();
                e.k("idx_select").c(0, ka).b(1, pb.score).c(2, ld).b(3, pb.sel).b(4, pb.nsel).grid(n, 1024);
                e.barrier();
                e.k("attn_dec").c(0, ka).b(1, pb.Q).b(2, kv).b(3, s.d_pages).b(4, pb.sel).b(5, pb.nsel).b(6, pb.part_o).b(7, pb.part_ml).grid(1, n, NKV, 32 * QPK);
                e.barrier();
                e.k("attn_combine").c(0, ka).b(1, pb.part_o).b(2, pb.part_ml).b(3, pb.qg).b(4, pb.att).b(5, pb.atth).grid(NH, n, 1, 256);
                e.barrier();
                mm(w.o, pb.att, NH * HD, pb.blk, D);
            } else {
                mm(w.qkv, pb.mx, D, pb.raw, QKV);
                mm(w.z, pb.mx, D, pb.z, GV * GD);
                mm(w.ab, pb.mx, D, pb.ab, 2 * GV);
                e.barrier();
                View conv = s.conv.at((size_t)gl * QKV * 3 * 4), S = s.ssm.at((size_t)gl * GV * GD * GD * 4);
                e.k("gdn_conv_pre").b(0, pb.raw).c(1, nn).b(2, w.conv_w).b(3, conv).b(4, pb.qn).b(5, pb.kn).b(6, pb.vc).grid(2 * GK + GV, (n + 15) / 16, 1, 128);
                e.barrier();
                e.k("gdn_conv_commit").b(0, conv).b(1, pb.raw).c(2, nn).grid(QKV / 256, 256);
                e.k("gdn_scan").b(0, pb.qn).b(1, pb.kn).b(2, pb.vc).b(3, pb.ab).b(4, w.A).b(5, w.dt).b(6, S).c(7, nn).grid(GV, GD / 32, 1, 128);
                e.barrier();
                e.k("gdn_out").b(0, pb.vc).b(1, pb.z).b(2, w.ssm_norm).grid(GV, n, 1, 128);
                e.barrier();
                mm(w.ssm_out, pb.vc, GV * GD, pb.blk, D);
            }
            e.barrier();
            combine(pb.inj);
            mix(w.hcf, pb.inj2);
            mm(w.router, pb.mx, D, pb.rlog, NE);
            mm(w.sg, pb.mx, D, pb.sgv, 1);
            mm(w.sh_gate, pb.mx, D, pb.shg, FS);
            mm(w.sh_up, pb.mx, D, pb.shu, FS);
            e.barrier();
            e.k("route_topk").b(0, pb.rlog).b(1, pb.ids).b(2, pb.wts).grid(n, 32);
            const uint32_t nsh = n * FS;
            e.k("swiglu").b(0, pb.shg).b(1, pb.shu).b(2, pb.shh).c(3, nsh).grid((nsh + 255) / 256, 256);
            e.barrier();
            mm(w.sh_down, pb.shg, FS, pb.shout, D);
            if (c0 == 0) {
                e.end();
                if (!all) { evg++; [cb encodeSignalEvent:ev_g value:evg]; routed[l] = evg; }
                staged[l] = ++evc;
                [cb encodeWaitForEvent:ev_c value:staged[l]];
                e.begin(m, cb);
            }
            if (!skip("moe")) pf_moe(e, xc.table, xc.fmt, l, pb.ids, pb.wts, pb.mx, D, n, pf_moe_bufs(pb.moe, n * TOPK_E, bt), pb.Y, bt);
            e.barrier();
            e.k("moe_combine_sum").c(0, nn).b(1, pb.Y).b(2, pb.sgv).b(3, pb.shout).b(4, pb.inj2).b(5, res).grid((n * D + 255) / 256, 256);
            e.barrier();
        }
        e.end();
        evg++;
        [cb encodeSignalEvent:ev_g value:evg];
        done[l] = evg;
        e.begin(m, cb);
        if (is_attn_layer(l)) al++; else gl++;
    }
    mixer(e, hc_head, pb.res.at((size_t)(P - 1) * HCD * 4), 1, View{});
    gemv(e, out, b.mx, D, b.logits, VOCAB, 1);
    e.barrier();
    topk(e, 1);
    uint64_t ready = 0;
    if (has_mtp) {
        e.end();
        ready = ++evg;
        [cb encodeSignalEvent:ev_g value:ready];
        e.begin(m, cb);
        const Layer &w = M;
        for (int c0 = 0; c0 < P; c0 += U) {
            const int n = std::min(U, P - c0);
            const uint32_t nn = n, pos0 = s.n + c0, nh = HCD, nr = n * HCD;
            const View res = pb.xconv;
            auto mm = [&](const WT &wt, View x, int ldx, View y, int ldy, int rows) { pf_mm(e, wt.type, wt.v, wt.rows, wt.cols, wt.rb, x, ldx, y, ldy, rows); };
            auto mix = [&](const Mix &mx, View inj) {
                const uint32_t mode = 1, nl = n * HCR;
                e.k("hc_norm").b(0, res).b(1, mx.norm).b(2, pb.xn).b(3, pb.embh).c(4, mode).grid(HC, n, 1, 256);
                e.barrier();
                mm(mx.down, pb.xn, HCD, pb.lo, HCR, n);
                mm(mx.inject, pb.xn, HCD, inj, HC, n);
                e.barrier();
                e.k("hc_lo").b(0, pb.lo).b(1, pb.embh).c(2, nl).grid((nl + 255) / 256, 256);
                e.barrier();
                mm(mx.up, pb.lo, HCR, pb.gate, HCD, n);
                e.barrier();
                e.k("hc_mix").b(0, pb.xn).b(1, pb.gate).b(2, pb.mx).b(3, pb.mxh).c(4, nn).grid((n * D + 255) / 256, 256);
                e.barrier();
            };
            if (c0 == 0) {
                e.k("copy_f32").b(0, s.h_last).b(1, pb.gate).c(2, nh).grid(HCD / 256, 256);
                if (n > 1) {
                    const uint32_t rest = (n - 1) * HCD;
                    e.k("copy_f32").b(0, pb.res).b(1, pb.gate.at((size_t)HCD * 4)).c(2, rest).grid((rest + 255) / 256, 256);
                }
            } else {
                e.k("copy_f32").b(0, pb.res.at((size_t)(c0 - 1) * HCD * 4)).b(1, pb.gate).c(2, nr).grid((nr + 255) / 256, 256);
            }
            e.k("embed_q8_0").b(0, tok.at((size_t)c0 * 4)).b(1, embd.v).c(2, erb).b(3, pb.xn).grid(n, 80);
            e.barrier();
            e.k("mtp_norm").b(0, pb.xn).b(1, pb.gate).b(2, enorm).b(3, hnorm).b(4, pb.emb).b(5, pb.key).grid(HC, n, 1, 256);
            e.barrier();
            mm(eh_e, pb.emb, D, pb.blk, D, n);
            mm(eh_h, pb.key, D, res, D, n * HC);
            e.barrier();
            e.k("hc_add_row").b(0, res).b(1, pb.blk).c(2, nn).grid((nr + 255) / 256, 256);
            e.barrier();
            mix(w.hca, pb.inj);
            mm(w.q, pb.mx, D, pb.qg, 2 * NH * HD, n);
            mm(w.k, pb.mx, D, pb.kb, NKV * HD, n);
            mm(w.v, pb.mx, D, pb.vb, NKV * HD, n);
            mm(w.iq, pb.mx, D, pb.qir, IDX_H * IDX_D, n);
            mm(w.ik, pb.mx, D, pb.kir, IDX_D, n);
            e.barrier();
            const int nb_end = (pos0 + n) / KPOOL;
            const KVArgs ka{nn, pos0, (uint32_t)NATT, 1};
            const View tail = s.tail.at((size_t)NATT * KPOOL * IDX_D * 2);
            e.k("attn_prep").c(0, ka).b(1, pb.qg).b(2, pb.kb).b(3, pb.vb).b(4, w.qn).b(5, w.kn).b(6, kv).b(7, s.d_pages).b(8, pb.Q).grid(NH + 2 * NKV, n, 1, 256);
            e.k("idx_qprep").c(0, ka).b(1, pb.qir).b(2, w.iqn).b(3, pb.Qi).grid(IDX_H, n, 1, 128);
            const int first = pos0 / KPOOL, last = (pos0 + n) / KPOOL - 1;
            if (last >= first) e.k("idx_pool").c(0, ka).b(1, pb.kir).b(2, tail).b(3, w.ikn).b(4, kv).b(5, s.d_pages).grid(last - first + 1, 128);
            e.barrier();
            e.k("idx_tail").c(0, ka).b(1, pb.kir).b(2, tail).grid(1, 128);
            const uint32_t ld = std::max(1, nb_end);
            if (nb_end > SEL_BLOCKS) e.k("idx_score").c(0, ka).b(1, pb.Qi).b(2, kv).b(3, s.d_pages).b(4, pb.score).c(5, ld).grid((ld + 255) / 256, n, 1, 256);
            e.barrier();
            e.k("idx_select").c(0, ka).b(1, pb.score).c(2, ld).b(3, pb.sel).b(4, pb.nsel).grid(n, 1024);
            e.barrier();
            e.k("attn_dec").c(0, ka).b(1, pb.Q).b(2, kv).b(3, s.d_pages).b(4, pb.sel).b(5, pb.nsel).b(6, pb.part_o).b(7, pb.part_ml).grid(1, n, NKV, 32 * QPK);
            e.barrier();
            e.k("attn_combine").c(0, ka).b(1, pb.part_o).b(2, pb.part_ml).b(3, pb.qg).b(4, pb.att).b(5, pb.atth).grid(NH, n, 1, 256);
            e.barrier();
            mm(w.o, pb.att, NH * HD, pb.blk, D, n);
            e.barrier();
            e.k("hc_combine").b(0, res).b(1, pb.inj).b(2, pb.blk).c(3, nn).grid((nr + 255) / 256, 256);
            e.barrier();
            mix(w.hcf, pb.inj2);
            mm(w.router, pb.mx, D, pb.rlog, NE, n);
            mm(w.sg, pb.mx, D, pb.sgv, 1, n);
            mm(w.sh_gate, pb.mx, D, pb.shg, FS, n);
            mm(w.sh_up, pb.mx, D, pb.shu, FS, n);
            e.barrier();
            e.k("route_topk").b(0, pb.rlog).b(1, pb.ids).b(2, pb.wts).grid(n, 32);
            const uint32_t nsh = n * FS;
            e.k("swiglu").b(0, pb.shg).b(1, pb.shu).b(2, pb.shh).c(3, nsh).grid((nsh + 255) / 256, 256);
            e.barrier();
            mm(w.sh_down, pb.shg, FS, pb.shout, D, n);
            pf_moe(e, xc.table, xc.fmt, NL, pb.ids, pb.wts, pb.mx, D, n, pf_moe_bufs(pb.moe, n * TOPK_E, bt), pb.Y, bt);
            e.barrier();
            e.k("moe_combine_sum").c(0, nn).b(1, pb.Y).b(2, pb.sgv).b(3, pb.shout).b(4, pb.inj2).b(5, res).grid((n * D + 255) / 256, 256);
            e.barrier();
        }
        const uint32_t nh = HCD;
        e.k("copy_f32").b(0, pb.res.at((size_t)(P - 1) * HCD * 4)).b(1, s.h_last).c(2, nh).grid(HCD / 256, 256);
        e.barrier();
    }
    e.end();
    [cb commit];
    auto wait_g = [&](uint64_t v) { wait_event(ev_g, v, cb); };
    double t_stage = 0, t_ple = 0;
    auto rows_ready = [&] { const double a = now_s(); gather.join(); t_ple = now_s() - a; ev_c.signaledValue = ple_ready;
                            if (getenv("CHRYSALIS_FN_TRACE")) fprintf(stderr, "  rows ready at %.3f s (c %llu)\n", now_s() - t0, (unsigned long long)ple_ready); };
    if (all) {
        auto stage = [&](int l) { const double a = now_s(); experts_stage_layer(l, l % 2); t_stage += now_s() - a; ev_c.signaledValue = staged[l];
                                  if (getenv("CHRYSALIS_FN_TRACE")) fprintf(stderr, "  layer %d staged at %.3f s (c %llu, GPU %llu / %llu)\n", l, now_s() - t0,
                                      (unsigned long long)staged[l], (unsigned long long)ev_g.signaledValue, (unsigned long long)done[NL - 1]); };
        stage(0);
        {
            const double a = now_s();
            experts_stage_layer(1, 1);
            t_stage += now_s() - a;
        }
        rows_ready();
        ev_c.signaledValue = staged[1];
        for (int l = 0; l < NL; l++) {
            wait_g(done[l]);
            experts_unstage(l % 2);
            if (l + 2 < NL) stage(l + 2);
        }
    } else {
        for (int l = 0; l < NL; l++) {
            if (l == PLE_LAYER) rows_ready();
            wait_g(routed[l]);
            const double a = now_s();
            experts_resolve(l, pb.ids.host<int>(), P * TOPK_E, [] {});
            t_stage += now_s() - a;
            ev_c.signaledValue = staged[l];
        }
    }
    if (has_mtp) {
        wait_event(ev_g, ready, cb);
        pending = cb;
        if (prof) settle();
    } else {
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
    }
    if (prof)
        fprintf(stderr, "prompt pass P=%d: %.1f ms (waiting for the n-gram rows %.1f ms, GPU %.1f ms, expert reads %.1f ms, %s)\n", P, (now_s() - t0) * 1e3,
                t_ple * 1e3, (cb.GPUEndTime - cb.GPUStartTime) * 1e3, t_stage * 1e3, all ? "every missing expert" : "routed misses");
}
