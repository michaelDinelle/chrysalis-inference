// Loading kiln's Flash-Next file (tools_fn/fn_quantize.py): dense weights and hot experts to VRAM (relayout to
// structure-of-arrays), cold experts to system RAM in GGUF block order for the CPU workers, token embeddings to pinned
// host memory, the n-gram table memory-mapped from the source file; then buffers, sequences and the KV pool.
#include "model.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int per_block(int type) { return (type == Q8_0 || type == Q5_0 || type == IQ4_NL) ? 32 : 256; }
int block_bytes(int type) {
    switch (type) { case Q8_0: return 34; case Q5_0: return 22; case IQ4_NL: return 18; case Q6_K: return 210; case IQ4_XS: return 136; }
    return 0;
}
}

void Engine::load(const std::string &path, int max_seqs, size_t reserve_bytes) {
    const double t0 = now();
    static GGUF g;                                  // the mapping stays alive: the n-gram path and cold copies use it
    if (!g.base && !g.open(path.c_str())) FAIL("cannot open %s", path.c_str());
    for (Layer *w = L; w <= &M; w++) *w = Layer{};  // a reload (after shutdown(keep_host)) starts from clean pointers
    hc_head = HcMix{}; mtp_head = HcMix{};
    if (g.strs["general.architecture"] != "kilnfn") FAIL("%s is not a kiln Flash-Next file (tools_fn/fn_quantize.py)", path.c_str());
    auto get = [&](const std::string &name) {
        const GGUFTensor *t = g.get(name);
        if (!t) FAIL("missing tensor %s", name.c_str());
        return t;
    };

    CK(hipInit(0));
    CK(hipSetDevice(0));
    CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    CK(hipStreamCreateWithFlags(&cst, hipStreamNonBlocking));
    CK(hipEventCreateWithFlags(&staged, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&copied, hipEventDisableTiming));
    CK(hipEventRecord(copied, cst));

    // ---- plan the VRAM weights: every dense tensor and the hot / MTP expert tiers --------------------------------
    struct Up { const GGUFTensor *t; QTensor *q = nullptr; float **f = nullptr; half **h = nullptr; int **i = nullptr; };
    std::vector<Up> ups;
    auto quant = [&](const std::string &name, QTensor &q) { ups.push_back({get(name), &q}); };
    auto vec = [&](const std::string &name, float *&f) { ups.push_back({get(name), nullptr, &f}); };
    auto f16 = [&](const std::string &name, half *&h) { ups.push_back({get(name), nullptr, nullptr, &h}); };
    auto i32 = [&](const std::string &name, int *&i) { ups.push_back({get(name), nullptr, nullptr, nullptr, &i}); };
    auto hcmix = [&](const std::string &p, HcMix &m, bool inject) {
        vec(p + "norm.weight", m.norm); quant(p + "down.weight", m.down); quant(p + "up.weight", m.up);
        if (inject) quant(p + "inject.weight", m.inject);
    };
    quant("output.weight", out);
    hcmix("output_hc_", hc_head, false);
    std::vector<float *> ssm_a(NGDN), ssm_dt(NGDN);
    for (int l = 0, gl = 0; l <= NL; l++) {
        Layer &w = l < NL ? L[l] : M;
        const std::string p = "blk." + std::to_string(l) + ".";
        hcmix(p + "hc_attn_", w.hca, true);
        hcmix(p + "hc_ffn_", w.hcf, true);
        if (l == NL || is_attn_layer(l)) {
            quant(p + "attn_q.weight", w.q); quant(p + "attn_k.weight", w.k); quant(p + "attn_v.weight", w.v);
            quant(p + "attn_output.weight", w.o);
            vec(p + "attn_q_norm.weight", w.q_norm); vec(p + "attn_k_norm.weight", w.k_norm);
            quant(p + "indexer.q_proj.weight", w.idx_q); quant(p + "indexer.k_proj.weight", w.idx_k);
            vec(p + "indexer.q_norm.weight", w.idx_qn); vec(p + "indexer.k_norm.weight", w.idx_kn);
        } else {
            quant(p + "attn_qkv.weight", w.qkv); quant(p + "attn_gate.weight", w.z); quant(p + "ssm_out.weight", w.ssm_out);
            f16(p + "ssm_ab.weight", w.Wab);
            vec(p + "ssm_conv1d.weight", w.conv); vec(p + "ssm_norm.weight", w.ssm_norm);
            vec(p + "ssm_a", ssm_a[gl]); vec(p + "ssm_dt.bias", ssm_dt[gl]);
            gl++;
            if (l == PLE_LAYER) {
                quant(p + "ple_key.weight", w.ple_key); quant(p + "ple_value.weight", w.ple_value);
                vec(p + "ple_norm_key.weight", w.ple_nk); vec(p + "ple_norm_query.weight", w.ple_nq);
                vec(p + "ple_norm_conv.weight", w.ple_nc); vec(p + "ple_conv1d.weight", w.ple_conv);
            }
        }
        f16(p + "ffn_router.weight", w.router);
        vec(p + "ffn_gate_inp_shexp.weight", w.sg_w);
        quant(p + "ffn_gate_shexp.weight", w.sh_gate); quant(p + "ffn_up_shexp.weight", w.sh_up); quant(p + "ffn_down_shexp.weight", w.sh_down);
        const std::string tier = l < NL ? "hot" : "mtp";
        if (g.get(p + "ffn_gate_" + tier + ".weight")) {
            quant(p + "ffn_gate_" + tier + ".weight", w.hot_gate); quant(p + "ffn_up_" + tier + ".weight", w.hot_up);
            quant(p + "ffn_down_" + tier + ".weight", w.hot_down);
            w.n_hot = (int)get(p + "ffn_gate_" + tier + ".weight")->ne[2];
        }
        if (l < NL) {
            i32(p + "ffn_exp_slot", w.slot_of);
            const GGUFTensor *c = g.get(p + "ffn_gate_cold.weight");
            w.n_cold = c ? (int)c->ne[2] : 0;
        }
        if (l == NL) {
            quant(p + "nextn.eh_proj.weight", eh_proj);
            vec(p + "nextn.enorm.weight", enorm); vec(p + "nextn.hnorm.weight", hnorm);
            hcmix(p + "nextn.hc_head_", mtp_head, false);
        }
    }
    // sizes: quant tensors as make_qtensor carves them, vectors as they are
    size_t wbytes = 0;
    auto shape = [](const GGUFTensor *t, int &rows, int &cols) { cols = (int)t->ne[0]; rows = (int)(t->ne[1] * t->ne[2] * t->ne[3]); };
    for (Up &u : ups) {
        if (u.q) {
            int rows, cols; shape(u.t, rows, cols);
            Arena dry; dry.cap = SIZE_MAX; make_qtensor(dry, u.t->type, rows, cols); wbytes += dry.used;
        } else wbytes += (u.t->nbytes + 255) & ~size_t(255);
    }
    wbytes += (size_t)NE * 4 + 2 * (size_t)NGDN * GV * 4 + (1 << 20);
    warena.init(wbytes);

    // ---- upload through pinned staging + a device scratch, relayout on the GPU --------------------------------------
    constexpr size_t STAGE = 64 << 20;
    uint8_t *pin[2], *dev[2];
    hipEvent_t ev[2];
    for (int i = 0; i < 2; i++) {
        CK(hipHostMalloc(&pin[i], STAGE));
        CK(hipMalloc(&dev[i], STAGE));
        CK(hipEventCreateWithFlags(&ev[i], hipEventDisableTiming));
        CK(hipEventRecord(ev[i], st));
    }
    int slot = 0;
    size_t total = 0;
    auto put = [&](const uint8_t *src, size_t n, auto &&consume) {   // copy src[0..n) in pieces; consume(dev, off, len)
        for (size_t off = 0; off < n; off += STAGE) {
            const size_t len = std::min(STAGE, n - off);
            CK(hipEventSynchronize(ev[slot]));
            memcpy(pin[slot], src + off, len);
            CK(hipMemcpyAsync(dev[slot], pin[slot], len, hipMemcpyHostToDevice, st));
            consume(dev[slot], off, len);
            CK(hipEventRecord(ev[slot], st));
            slot ^= 1;
        }
        total += n;
    };
    for (Up &u : ups) {
        if (u.q) {
            int rows, cols; shape(u.t, rows, cols);
            *u.q = make_qtensor(warena, u.t->type, rows, cols);
            const int bb = block_bytes(u.t->type);
            if (!bb) FAIL("unsupported type %d for %s", u.t->type, "tensor");
            const size_t piece = STAGE / bb * bb;
            for (size_t off = 0; off < u.t->nbytes; off += piece) {
                const size_t len = std::min(piece, u.t->nbytes - off);
                CK(hipEventSynchronize(ev[slot]));
                memcpy(pin[slot], u.t->data + off, len);
                CK(hipMemcpyAsync(dev[slot], pin[slot], len, hipMemcpyHostToDevice, st));
                relayout(*u.q, dev[slot], st, off / bb, len / bb);
                CK(hipEventRecord(ev[slot], st));
                slot ^= 1;
            }
            total += u.t->nbytes;
        } else {
            uint8_t *d = warena.get(u.t->nbytes);
            put(u.t->data, u.t->nbytes, [&](uint8_t *src, size_t off, size_t len) {
                CK(hipMemcpyAsync(d + off, src, len, hipMemcpyDeviceToDevice, st));
            });
            if (u.f) *u.f = (float *)d;
            if (u.h) *u.h = (half *)d;
            if (u.i) *u.i = (int *)d;
        }
    }
    // GDN A / dt_bias contiguous for the all-layer commit
    gdn_A = warena.get<float>((size_t)NGDN * GV);
    gdn_dt = warena.get<float>((size_t)NGDN * GV);
    for (int gl = 0; gl < NGDN; gl++) {
        CK(hipMemcpyAsync(gdn_A + gl * GV, ssm_a[gl], GV * 4, hipMemcpyDeviceToDevice, st));
        CK(hipMemcpyAsync(gdn_dt + gl * GV, ssm_dt[gl], GV * 4, hipMemcpyDeviceToDevice, st));
    }
    {   // MTP experts are all in VRAM: identity slot map
        std::vector<int> id(NE);
        for (int e = 0; e < NE; e++) id[e] = e;
        M.slot_of = warena.get<int>(NE);
        CK(hipMemcpyAsync(M.slot_of, id.data(), NE * 4, hipMemcpyHostToDevice, st));
        CK(hipStreamSynchronize(st));
    }
    for (int i = 0; i < 2; i++) { CK(hipHostFree(pin[i])); CK(hipFree(dev[i])); CK(hipEventDestroy(ev[i])); }
    const double t_vram = now();

    // ---- cold experts -> system RAM (hugepages), copied by several threads ----------------------------------------
    if (!cold_mem) {                                // kept across shutdown(keep_host): a model switch skips this
    cold.assign(NL, ColdLayer{});
    std::vector<std::tuple<const uint8_t *, size_t, size_t>> copies;   // src, bytes, offset
    cold_bytes = 0;
    for (int l = 0; l < NL; l++) {
        if (!L[l].n_cold) continue;
        const std::string p = "blk." + std::to_string(l) + ".";
        for (const char *k : {"ffn_gate_cold.weight", "ffn_up_cold.weight", "ffn_down_cold.weight"}) {
            const GGUFTensor *t = get(p + k);
            copies.emplace_back(t->data, t->nbytes, cold_bytes);
            cold_bytes += (t->nbytes + 4095) & ~size_t(4095);
        }
    }
    if (cold_bytes) {
        cold_mem = (uint8_t *)mmap(nullptr, cold_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (cold_mem == MAP_FAILED) FAIL("cannot allocate %.1f GB of system RAM for the cold experts", cold_bytes / 1e9);
        madvise(cold_mem, cold_bytes, MADV_HUGEPAGE);
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int w = 0; w < 8; w++) th.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < copies.size();) {
                auto [src, n, off] = copies[i];
                memcpy(cold_mem + off, src, n);
                g.drop(src, n);                     // and the file pages: they would push the n-gram table out
            }
        });
        for (auto &t : th) t.join();
        int ci = 0;
        for (int l = 0; l < NL; l++) {
            if (!L[l].n_cold) continue;
            ColdLayer &c = cold[l];
            c.gate = cold_mem + std::get<2>(copies[ci]); c.up = cold_mem + std::get<2>(copies[ci + 1]); c.down = cold_mem + std::get<2>(copies[ci + 2]);
            const std::string p = "blk." + std::to_string(l) + ".";
            c.type_gu = get(p + "ffn_gate_cold.weight")->type; c.type_dn = get(p + "ffn_down_cold.weight")->type;
            c.row_gu = (size_t)D / per_block(c.type_gu) * block_bytes(c.type_gu);
            c.row_dn = (size_t)FE / per_block(c.type_dn) * block_bytes(c.type_dn);
            ci += 3;
        }
    }
    }
    if (cold_bytes) {                               // the GPU streams cold experts during prefill
        (void)hipHostUnregister(cold_mem);          // a device reset by the other engine may have dropped it
        (void)hipGetLastError();
        CK(hipHostRegister(cold_mem, cold_bytes, hipHostRegisterMapped));
        void *dp;
        CK(hipHostGetDevicePointer(&dp, cold_mem, 0));
        d_cold = (uint8_t *)dp;
    }
    const double t_cold = now();

    // ---- token embeddings -> pinned host memory (the GPU reads its rows over PCIe) --------------------------------
    {
        const GGUFTensor *t = get("token_embd.weight");
        if (t->type != Q8_0) FAIL("token embeddings must be Q8_0");
        void *host;
        CK(hipHostMalloc(&host, t->nbytes, hipHostMallocMapped));
        memcpy(host, t->data, t->nbytes);
        embd_host = host;
        void *devp;
        CK(hipHostGetDevicePointer(&devp, host, 0));
        embd.type = Q8_0; embd.rows = (int)t->ne[1]; embd.cols = (int)t->ne[0]; embd.q = (uint8_t *)devp;
    }
    // ---- n-gram table: memory-map the source file ------------------------------------------------------------------
    if (!ngram_map) {
        std::string f = g.strs["kilnfn.ngram_file"];
        if (f.empty()) FAIL("no n-gram table path in %s", path.c_str());
        if (f[0] != '/') f = path.substr(0, path.find_last_of('/') + 1) + f;
        const int fd = ::open(f.c_str(), O_RDONLY);
        if (fd < 0) FAIL("cannot open the n-gram table %s", f.c_str());
        struct stat sb; fstat(fd, &sb);
        ngram_map_bytes = sb.st_size;
        ngram_map = (uint8_t *)mmap(nullptr, ngram_map_bytes, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (ngram_map == MAP_FAILED) FAIL("cannot map %s", f.c_str());
        madvise(ngram_map, ngram_map_bytes, MADV_RANDOM);
        ngram = ngram_map + g.ints["kilnfn.ngram_offset"];
        const auto &m = g.arrs["kilnfn.ple.layer_multipliers"], &o = g.arrs["kilnfn.ple.head_offsets"], &v = g.arrs["kilnfn.ple.head_vocab_sizes"];
        if (m.size() < 3 || o.size() < PLE_HEADS || v.size() < PLE_HEADS) FAIL("missing n-gram hash constants");
        for (int i = 0; i < 3; i++) ple_mult[i] = (uint64_t)m[i];
        for (int h = 0; h < PLE_HEADS; h++) { ple_off[h] = (uint32_t)o[h]; ple_voc[h] = (uint32_t)v[h]; }
    }
    // ---- CPU workers for the cold experts ---------------------------------------------------------------------------
    {
        void *p;
        CK(hipHostMalloc(&p, sizeof(MoeShared), hipHostMallocCoherent | hipHostMallocMapped));
        memset(p, 0, sizeof(MoeShared));
        sh = (MoeShared *)p;
        void *dp;
        CK(hipHostGetDevicePointer(&dp, p, 0));
        d_sh = (MoeShared *)dp;
        cpu.start(sh, cold, 8);
    }
    // ---- hipBLASLt, buffers, sequences, KV pool -----------------------------------------------------------------
    hipblasLtCreate(&lt);
    CK(hipMalloc(&lt_ws, lt_ws_size));
    alloc_buffers(max_seqs);
    size_t free_b, total_b;
    CK(hipMemGetInfo(&free_b, &total_b));
    const size_t want = free_b > reserve_bytes ? free_b - reserve_bytes : 0;
    const size_t max_pages = ((size_t)max_ctx * 3 / 2 + PAGE - 1) / PAGE;            // context plus snapshots' pages
    n_pages = (int)std::min(max_pages, want / KV_PAGE);
    if (n_pages < 64) FAIL("not enough VRAM left for the KV cache");
    CK(hipMalloc(&kv_pool, (size_t)n_pages * KV_PAGE));
    free_pages.clear();
    for (int p = n_pages - 1; p >= 0; p--) free_pages.push_back(p);
    page_ref.assign(n_pages, 0);
    fprintf(stderr, "kiln-fn: weights %.2f GB VRAM (%.1f s), cold experts %.2f GB RAM (%.1f s), buffers %.2f GB, KV pool %d pages = %d tokens (%.2f GB), load %.1f s\n",
            warena.used / 1e9, t_vram - t0, cold_bytes / 1e9, t_cold - t_vram, barena.used / 1e9, n_pages, n_pages * PAGE,
            (double)n_pages * KV_PAGE / 1e9, now() - t0);
    (void)total;
}

void Engine::alloc_buffers(int max_seqs) {
    const int U = umax, T8 = 8;
    const size_t big = (size_t)U * HCD;
    size_t bytes = 0;
    auto plan = [&](size_t n) { bytes += (n + 255) & ~size_t(255); };
    // generous upper bound, computed from the same list as below
    plan((size_t)U * 4);                                   // tokens
    for (size_t n : {(size_t)U * D, big, big, (size_t)U * HCR, big, (size_t)U * HC, (size_t)U * HC, (size_t)U * D, (size_t)U * 2 * NH * HD,
                     (size_t)U * GV * GD, (size_t)U * NKV * HD * 2, (size_t)U * NKV * HD, (size_t)U * IDX_H * IDX_D, (size_t)U * IDX_D, (size_t)U * D})
        plan(n * 4);
    for (size_t n : {big, (size_t)U * HCR, (size_t)U * D, (size_t)U * NH * HD, (size_t)U * IDX_H * IDX_D, (size_t)U * NH * HD, (size_t)12288 * 2560,
                     (size_t)U * HC * 2 * D})
        plan(n * 2);
    plan(big + big / 32 * 4); plan((size_t)U * HCR + U * HCR / 32 * 4); plan((size_t)U * D + U * D / 32 * 4); plan((size_t)U * NH * HD + U * NH * HD / 32 * 4);
    plan((size_t)MAX_SPLITS * 8 * NH * HD * 4); plan((size_t)MAX_SPLITS * 8 * NH * 2 * 4);
    plan((size_t)SEL_G * (max_ctx / KPOOL + 1) * 4); plan((size_t)U * (SEL_BLOCKS + 1) * 4);
    plan((size_t)U * GK * GD * 4 * 2); plan((size_t)U * GV * GD * 4);
    plan((size_t)NGDN * 8 * QKV * 4); plan((size_t)NGDN * 8 * GK * GD * 4); plan((size_t)NGDN * 8 * GV * GD * 4); plan((size_t)NGDN * 8 * 2 * GV * 4);
    plan((size_t)T8 * NE * 4); plan(T8 * 4); plan((size_t)T8 * TOPK_E * 4 * 3); plan((size_t)T8 * TOPK_E * D * 4); plan((size_t)T8 * D * 4 * 2);
    plan((size_t)T8 * TOPK_E * FE * 2); plan((size_t)T8 * FS * 2);
    plan((size_t)U * PLE_HEADS * 170); plan((size_t)U * D * 4); plan(big * 4); plan((size_t)U * D * 4); plan(big * 4); plan((size_t)U * D * 2); plan((size_t)U * D * 2);
    plan((size_t)T8 * VOCAB * 4); plan((size_t)T8 * 64 * TOPK * 4 * 2); plan((size_t)T8 * TOPK * 8);
    plan(big * 4); plan((size_t)U * D * 4); plan((size_t)U * HC * 2 * D * 4); plan((size_t)U * HC * 2 * D * 2); plan((size_t)U * HC * 2 * D * 2); plan(64);
    plan((size_t)U * D * 4 * 2); plan((size_t)U * D * 2);
    const size_t N = (size_t)U * TOPK_E;                   // MoE prefill
    plan((size_t)U * NE * 4); plan((size_t)U * 4); plan(N * 4 * 5); plan(1024 * sizeof(MoeWork));
    plan(N * D * 2); plan(N * FE * 2); plan(N * D * 2);
    const size_t raw_bytes = (size_t)STAGE_G * (2 * (size_t)FE * (D / 256 * 136) + (size_t)D * (FE / 32 * 18));
    plan(2 * raw_bytes); plan(2 * raw_bytes + (64 << 10));
    barena.init(bytes + (64 << 20));
    auto qact = [&](size_t n) { QAct a; a.q = barena.get<int8_t>(n); a.d = barena.get<float>(n / 32 + 1); return a; };
    b.tokens = barena.get<int>(U);
    b.x = barena.get<float>((size_t)U * D);
    b.res = barena.get<float>(big);
    b.xn = barena.get<float>(big);
    b.lo = barena.get<float>((size_t)U * HCR);
    b.gate = barena.get<float>(big);
    b.inj_a = barena.get<float>((size_t)U * HC);
    b.inj_f = barena.get<float>((size_t)U * HC);
    b.mx = barena.get<float>((size_t)U * D);
    b.proj = barena.get<float>((size_t)U * std::max(2 * NH * HD, QKV));
    b.zb = barena.get<float>((size_t)U * GV * GD);
    b.kb = barena.get<float>((size_t)U * std::max(NKV * HD, 2 * GV));
    b.vb = barena.get<float>((size_t)U * NKV * HD);
    b.qidx = barena.get<float>((size_t)U * IDX_H * IDX_D);
    b.kidx = barena.get<float>((size_t)U * IDX_D);
    b.blk = barena.get<float>((size_t)U * D);
    b.xn_h = barena.get<half>(big);
    b.lo_h = barena.get<half>((size_t)U * HCR);
    b.hh = barena.get<half>((size_t)U * D);
    b.Q16 = barena.get<half>((size_t)U * NH * HD);
    b.qidx16 = barena.get<half>((size_t)U * IDX_H * IDX_D);
    b.att_h = barena.get<half>((size_t)U * NH * HD);
    b.deq = barena.get<half>((size_t)12288 * 2560);
    b.gemm16 = barena.get<half>((size_t)U * HC * 2 * D);
    b.xn_q = qact(big);
    b.lo_q = qact((size_t)U * HCR);
    b.hq = qact((size_t)U * D);
    b.mq = qact((size_t)U * NH * HD);
    b.part_o = barena.get<float>((size_t)MAX_SPLITS * 8 * NH * HD);
    b.part_ml = barena.get<float>((size_t)MAX_SPLITS * 8 * NH * 2);
    b.score = barena.get<float>((size_t)SEL_G * (max_ctx / KPOOL + 1));
    b.sel = barena.get<int>((size_t)U * (SEL_BLOCKS + 1));
    b.qn = barena.get<float>((size_t)U * GK * GD);
    b.kn = barena.get<float>((size_t)U * GK * GD);
    b.vc = barena.get<float>((size_t)U * GV * GD);
    b.v_raw = barena.get<float>((size_t)NGDN * 8 * QKV);
    b.v_kn = barena.get<float>((size_t)NGDN * 8 * GK * GD);
    b.v_vc = barena.get<float>((size_t)NGDN * 8 * GV * GD);
    b.v_ab = barena.get<float>((size_t)NGDN * 8 * 2 * GV);
    b.logits_e = barena.get<float>((size_t)T8 * NE);
    b.sg = barena.get<float>(T8);
    b.wts = barena.get<float>((size_t)T8 * TOPK_E);
    b.ids = barena.get<int>((size_t)T8 * TOPK_E);
    b.hot = barena.get<int>((size_t)T8 * TOPK_E);
    b.y = barena.get<float>((size_t)T8 * TOPK_E * D);
    b.shared = barena.get<float>((size_t)T8 * D);
    b.moe = barena.get<float>((size_t)U * D);
    b.hq_e = qact((size_t)T8 * TOPK_E * FE);
    b.hs_q = qact((size_t)T8 * FS);
    CK(hipHostMalloc(&b.ple_rows, (size_t)U * PLE_HEADS * 170));
    b.d_ple_rows = barena.get<uint8_t>((size_t)U * PLE_HEADS * 170);
    b.emb = barena.get<float>((size_t)U * D);
    b.key = barena.get<float>(big);
    b.value = barena.get<float>((size_t)U * D);
    b.xconv = barena.get<float>(big);
    b.emb_h = barena.get<half>((size_t)U * D);
    b.emb_q = qact((size_t)U * D);
    b.logits = barena.get<float>((size_t)T8 * VOCAB);
    const int nb = (VOCAB + 4095) / 4096;
    b.tsv = barena.get<float>((size_t)T8 * nb * TOPK);
    b.tsi = barena.get<int>((size_t)T8 * nb * TOPK);
    b.topv = barena.get<float>((size_t)T8 * TOPK);
    b.topi = barena.get<int>((size_t)T8 * TOPK);
    b.mtp_res = barena.get<float>(big);
    b.mtp_e = barena.get<float>((size_t)U * D);
    b.mtp_in = barena.get<float>((size_t)U * HC * 2 * D);
    b.mtp_in_h = barena.get<half>((size_t)U * HC * 2 * D);
    b.mtp_in_q = qact((size_t)8 * 2 * D);
    b.draft = barena.get<int>(16);
    b.rlog = barena.get<float>((size_t)U * NE);
    b.sgT = barena.get<float>(U);
    b.wtsT = barena.get<float>(N); b.ent_w = barena.get<float>(N);
    b.idsT = barena.get<int>(N); b.ent_tok = barena.get<int>(N); b.pos = barena.get<int>(N);
    b.work = barena.get<MoeWork>(1024);
    b.Xs = barena.get<half>(N * D); b.Hs = barena.get<half>(N * FE); b.Ys = barena.get<half>(N * D);
    CK(hipHostMalloc(&b.h_ids, N * 4)); CK(hipHostMalloc(&b.h_wts, N * 4)); CK(hipHostMalloc(&b.h_tok, N * 4));
    CK(hipHostMalloc(&b.h_entw, N * 4)); CK(hipHostMalloc(&b.h_pos, N * 4)); CK(hipHostMalloc(&b.h_work, 1024 * sizeof(MoeWork)));
    for (int k = 0; k < 2; k++) {
        b.raw[k] = barena.get<uint8_t>(raw_bytes);
        b.sg_[k] = make_qtensor(barena, IQ4_XS, STAGE_G * FE, D);
        b.su_[k] = make_qtensor(barena, IQ4_XS, STAGE_G * FE, D);
        b.sd_[k] = make_qtensor(barena, IQ4_NL, STAGE_G * D, FE);
        CK(hipEventCreateWithFlags(&raw_free[k], hipEventDisableTiming));
        CK(hipEventCreateWithFlags(&raw_ready[k], hipEventDisableTiming));
        CK(hipEventRecord(raw_free[k], st));
    }
    // sequences
    seqs.assign(max_seqs, Seq{});
    for (Seq &s : seqs) {
        CK(hipMalloc(&s.d_pages, ((size_t)max_ctx * 2 / PAGE + 8) * 4));
        CK(hipMalloc(&s.ssm, ssm_bytes()));
        CK(hipMalloc(&s.conv, conv_bytes()));
        CK(hipMalloc(&s.ple, ple_bytes()));
        CK(hipMalloc(&s.h_last, (size_t)HCD * 4));
    }
    CK(hipMalloc(&stage_state, state_bytes()));
}

void Engine::warmup() {
    // first launches load the code objects; keep them out of the first request
    QAct none{nullptr, nullptr};
    (void)none;
    CK(hipStreamSynchronize(st));
}

void Engine::shutdown(bool keep_host) {
    cpu.stop();
    CK(hipDeviceSynchronize());
    if (kv_pool) (void)hipFree(kv_pool);
    for (Seq &s : seqs) { (void)hipFree(s.d_pages); (void)hipFree(s.ssm); (void)hipFree(s.conv); (void)hipFree(s.ple); (void)hipFree(s.h_last); }
    seqs.clear();
    for (Snap &sn : snaps) if (sn.host) (void)hipHostFree(sn.host);
    snaps.clear();
    if (warena.base) (void)hipFree(warena.base);
    if (barena.base) (void)hipFree(barena.base);
    warena = Arena{}; barena = Arena{};
    for (void *p : {(void *)b.ple_rows, (void *)b.h_ids, (void *)b.h_wts, (void *)b.h_tok, (void *)b.h_entw, (void *)b.h_pos, (void *)b.h_work, (void *)sh})
        if (p) (void)hipHostFree(p);
    if (embd.q) { void *h = nullptr; (void)hipHostGetDevicePointer(&h, embd.q, 0); }
    if (embd_host) (void)hipHostFree(embd_host);
    embd_host = nullptr; embd.q = nullptr; sh = d_sh = nullptr;
    if (stage_state) (void)hipFree(stage_state);
    if (lt_ws) (void)hipFree(lt_ws);
    if (lt) hipblasLtDestroy(lt);
    lt = nullptr; lt_ws = nullptr; stage_state = nullptr; kv_pool = nullptr;
    lt_plans.clear();
    free_pages.clear(); page_ref.clear();
    if (cold_mem) (void)hipHostUnregister(cold_mem);
    (void)hipGetLastError();
    (void)hipStreamDestroy(st); (void)hipStreamDestroy(cst);
    if (!keep_host) {
        if (cold_mem) munmap(cold_mem, cold_bytes);
        if (ngram_map) munmap(ngram_map, ngram_map_bytes);
        cold_mem = nullptr; ngram_map = nullptr; cold_bytes = 0;
    }
}
