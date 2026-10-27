#include "engine.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>

std::atomic<float> g_progress{0};

const char *tname(int t) {
    switch (t) {
        case 0: return "f32"; case 1: return "f16"; case 30: return "bf16"; case 42: return "q2_0"; case 2: return "q4_0";
        case 6: return "q5_0"; case 8: return "q8_0"; case 20: return "iq4_nl"; case 23: return "iq4_xs"; case 11: return "q3_k";
        case 12: return "q4_k"; case 13: return "q5_k"; case 14: return "q6_k";
    }
    FAIL("no kernels for ggml type %d", t);
}

struct GemvArgs { uint32_t rows, cols, rb, T, ldx, ldy, add; float scale; uint32_t ggml; };
static const uint32_t GGML_ACT = getenv("CHRYSALIS_FN_GGML_ACT") ? 1 : 0;
struct KVArgs { uint32_t T, pos0, layer, nsplit; };
struct MoeArgs { uint32_t T, layer, phase, ggml; };
struct GdnStepArgs { uint32_t T, write_state; };

static double mem_available() {
    vm_statistics64_data_t vs;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &n) != KERN_SUCCESS) return 0;
    return (double)(vs.free_count + vs.inactive_count + vs.purgeable_count + vs.speculative_count) * vm_page_size;
}
static double mem_total() { uint64_t v = 0; size_t n = sizeof v; sysctlbyname("hw.memsize", &v, &n, nullptr, 0); return (double)v; }
static double env_d(const char *k, double def) { const char *v = getenv(k); return v ? atof(v) : def; }

static void read_all(const std::vector<std::tuple<int, size_t, size_t, uint8_t *>> &jobs, int threads) {
    std::vector<std::tuple<int, size_t, size_t, uint8_t *>> pieces;
    for (auto &[fd, off, n, dst] : jobs)
        for (size_t o = 0; o < n; o += 32 << 20) pieces.push_back({fd, off + o, std::min(n - o, (size_t)32 << 20), dst + o});
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int k = 0; k < threads; k++) th.emplace_back([&] {
        for (size_t i; (i = next++) < pieces.size();) {
            auto [fd, off, n, dst] = pieces[i];
            for (size_t done = 0; done < n;) {
                const ssize_t r = pread(fd, dst + done, n - done, off + done);
                if (r <= 0) FAIL("read of the model file failed at %zu", off + done);
                done += r;
            }
        }
    });
    for (auto &t : th) t.join();
}

static std::vector<std::tuple<int, size_t, size_t, uint8_t *>> g_reads;
static std::vector<int> g_nocache;

WT Engine::weight(const std::string &name, bool required) {
    const GGUFTensor *t = g.get(name);
    if (!t) { if (required) FAIL("missing tensor %s", name.c_str()); return {}; }
    WT w;
    w.type = t->type; w.cols = (int)t->ne[0]; w.rows = (int)(t->ne[1] * t->ne[2]); w.rb = ggml_row_bytes(t->type, t->ne[0]);
    if (w.cols % 32) FAIL("tensor %s: %d columns", name.c_str(), w.cols);
    tname(w.type);
    w.v = warena.get(t->nbytes);
    g_reads.push_back({g_nocache[t->file], t->off, t->nbytes, w.v.host<uint8_t>()});
    return w;
}

View Engine::small(const std::string &name) {
    const GGUFTensor *t = g.get(name);
    if (!t) FAIL("missing tensor %s", name.c_str());
    const size_t n = t->ne[0] * t->ne[1] * t->ne[2];
    View v = warena.get(n * 4);
    if (t->type == 0) { g_reads.push_back({g_nocache[t->file], t->off, t->nbytes, v.host<uint8_t>()}); return v; }
    if (t->type != 1) FAIL("tensor %s: type %d where fp32 / fp16 was expected", name.c_str(), t->type);
    std::vector<__fp16> h(n);
    if (pread(g.fds[t->file], h.data(), n * 2, t->off) != (ssize_t)(n * 2)) FAIL("read %s", name.c_str());
    for (size_t i = 0; i < n; i++) v.host<float>()[i] = (float)h[i];
    return v;
}

void Engine::load(const std::string &path, int max_seqs) {
    const double t0 = now_s();
    if (!g.open(path)) FAIL("cannot open %s", path.c_str());
    if (g.strs["general.architecture"] != "chrysalisfn") FAIL("%s is not a Chrysalis Flash-Next file (tools_fn/fn_quantize.py)", path.c_str());
    if (g.fds.size() != 1) FAIL("%s: one file expected (the experts are read through one descriptor)", path.c_str());
    for (int fd : g.fds) { (void)fd; }
    for (size_t i = 0; i < g.fds.size(); i++) {
        const std::string p = i == 0 ? path : "";
        int fd = dup(g.fds[i]);
        fcntl(fd, F_NOCACHE, 1);
        g_nocache.push_back(fd);
    }
    m.init();
    const std::string mtp = "blk." + std::to_string(NL) + ".";
    size_t dense = 0, mtp_dense = 0;
    for (auto &[n, t] : g.tensors) {
        if (n.find("_hot.") != std::string::npos || n.find("_cold") != std::string::npos || n.find("ffn_exp_slot") != std::string::npos ||
            n.find("_mtp.") != std::string::npos) continue;
        const size_t b = ((t.type == 1 && t.ne[1] == 1 ? t.nbytes * 2 : t.nbytes) + 255) & ~size_t(255);
        dense += b;
        if (n.rfind(mtp, 0) == 0) mtp_dense += b;
    }
    warena.init(m, dense + (1 << 20));
    embd = weight("token_embd.weight");
    if (embd.type != 8) FAIL("token embeddings: type %d (the embed kernel reads Q8_0)", embd.type);
    out = weight("output.weight");
    auto mix = [&](const std::string &p, Mix &mx, bool inject) {
        mx.norm = small(p + "norm.weight");
        mx.down = weight(p + "down.weight");
        mx.up = weight(p + "up.weight");
        if (inject) mx.inject = weight(p + "inject.weight");
        if (mx.down.type != 8 || mx.up.type != 8 || (inject && mx.inject.type != 8)) FAIL("%s*: not Q8_0", p.c_str());
    };
    mix("output_hc_", hc_head, false);
    xc.src.assign((size_t)NL * NE, XSrc{});
    int gl = 0;
    for (int l = 0; l < NL; l++) {
        Layer &w = L[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        mix(p + "hc_attn_", w.hca, true);
        mix(p + "hc_ffn_", w.hcf, true);
        if (is_attn_layer(l)) {
            w.q = weight(p + "attn_q.weight"); w.k = weight(p + "attn_k.weight"); w.v = weight(p + "attn_v.weight");
            w.o = weight(p + "attn_output.weight");
            w.qn = small(p + "attn_q_norm.weight"); w.kn = small(p + "attn_k_norm.weight");
            w.iq = weight(p + "indexer.q_proj.weight"); w.ik = weight(p + "indexer.k_proj.weight");
            w.iqn = small(p + "indexer.q_norm.weight"); w.ikn = small(p + "indexer.k_norm.weight");
            if (w.q.rows != 2 * NH * HD || w.k.rows != NKV * HD || w.iq.rows != IDX_H * IDX_D) FAIL("layer %d: attention shapes", l);
        } else {
            w.qkv = weight(p + "attn_qkv.weight"); w.z = weight(p + "attn_gate.weight");
            w.ab = weight(p + "ssm_ab.weight");
            w.ssm_out = weight(p + "ssm_out.weight");
            w.conv_w = small(p + "ssm_conv1d.weight"); w.ssm_norm = small(p + "ssm_norm.weight");
            w.A = small(p + "ssm_a"); w.dt = small(p + "ssm_dt.bias");
            if (w.qkv.rows != QKV || w.z.rows != GV * GD || w.ab.rows != 2 * GV) FAIL("layer %d: GDN shapes", l);
            gl++;
        }
        if (l == PLE_LAYER) {
            w.ple_key = weight(p + "ple_key.weight"); w.ple_value = weight(p + "ple_value.weight");
            w.ple_nk = small(p + "ple_norm_key.weight"); w.ple_nq = small(p + "ple_norm_query.weight");
            w.ple_nc = small(p + "ple_norm_conv.weight"); w.ple_conv = small(p + "ple_conv1d.weight");
        }
        w.router = weight(p + "ffn_router.weight");
        w.sg = weight(p + "ffn_gate_inp_shexp.weight");
        w.sh_gate = weight(p + "ffn_gate_shexp.weight"); w.sh_up = weight(p + "ffn_up_shexp.weight");
        w.sh_down = weight(p + "ffn_down_shexp.weight");
        expert_sources(g, l, &xc.src[(size_t)l * NE]);
    }
    if (gl != NGDN) FAIL("%d GDN layers", gl);
    if (g.get(mtp + "nextn.eh_proj.weight")) {
        Layer &w = M;
        mix(mtp + "hc_attn_", w.hca, true);
        mix(mtp + "hc_ffn_", w.hcf, true);
        w.q = weight(mtp + "attn_q.weight"); w.k = weight(mtp + "attn_k.weight"); w.v = weight(mtp + "attn_v.weight");
        w.o = weight(mtp + "attn_output.weight");
        w.qn = small(mtp + "attn_q_norm.weight"); w.kn = small(mtp + "attn_k_norm.weight");
        w.iq = weight(mtp + "indexer.q_proj.weight"); w.ik = weight(mtp + "indexer.k_proj.weight");
        w.iqn = small(mtp + "indexer.q_norm.weight"); w.ikn = small(mtp + "indexer.k_norm.weight");
        if (w.q.rows != 2 * NH * HD || w.k.rows != NKV * HD || w.iq.rows != IDX_H * IDX_D) FAIL("MTP layer: attention shapes");
        w.router = weight(mtp + "ffn_router.weight");
        w.sg = weight(mtp + "ffn_gate_inp_shexp.weight");
        w.sh_gate = weight(mtp + "ffn_gate_shexp.weight"); w.sh_up = weight(mtp + "ffn_up_shexp.weight");
        w.sh_down = weight(mtp + "ffn_down_shexp.weight");
        const WT eh = weight(mtp + "nextn.eh_proj.weight");
        if (eh.rows != D || eh.cols != 2 * D || eh.type != 8) FAIL("MTP layer: eh_proj is not Q8_0 [%d, %d]", 2 * D, D);
        eh_e = eh; eh_e.cols = D;
        eh_h = eh; eh_h.cols = D; eh_h.v = eh.v.at(ggml_row_bytes(eh.type, D));
        enorm = small(mtp + "nextn.enorm.weight"); hnorm = small(mtp + "nextn.hnorm.weight");
        mix(mtp + "nextn.hc_head_", mtp_head, false);
        dout = out; dout.rows = DRAFT_VOCAB;
        has_mtp = true;
    }
    read_all(g_reads, 8);
    g_reads.clear();
    const double t_dense = now_s();
    g_progress = 0.2f;
    {
        std::string f = g.strs["chrysalisfn.ngram_file"];
        if (f.empty()) FAIL("no n-gram table path (chrysalisfn.ngram_file) in %s", path.c_str());
        if (f[0] != '/') f = path.substr(0, path.find_last_of('/') + 1) + f;
        ngram_fd = ::open(f.c_str(), O_RDONLY);
        if (ngram_fd < 0) FAIL("cannot open the n-gram table %s", f.c_str());
        struct stat sb;
        fstat(ngram_fd, &sb);
        ngram_off = (size_t)g.ints["chrysalisfn.ngram_offset"];
        const size_t rows = (size_t)g.ints["chrysalisfn.ngram_rows"];
        if (!rows || ngram_off + rows * PLE_ROW > (size_t)sb.st_size) FAIL("%s ends before its %zu-row n-gram table (a partial copy?)", f.c_str(), rows);
        ngram_bytes = sb.st_size;
        ngram_map = mmap(nullptr, ngram_bytes, PROT_READ, MAP_SHARED, ngram_fd, 0);
        if (ngram_map == MAP_FAILED) FAIL("cannot map %s", f.c_str());
        madvise(ngram_map, ngram_bytes, MADV_RANDOM);
        ngram = (const uint8_t *)ngram_map + ngram_off;
        const auto &mu = g.arrs["chrysalisfn.ple.layer_multipliers"], &o = g.arrs["chrysalisfn.ple.head_offsets"], &v = g.arrs["chrysalisfn.ple.head_vocab_sizes"];
        if (mu.size() < 3 || o.size() < PLE_HEADS || v.size() < PLE_HEADS) FAIL("missing n-gram hash constants");
        for (int i = 0; i < 3; i++) ple_mult[i] = (uint64_t)mu[i];
        for (int h = 0; h < PLE_HEADS; h++) {
            ple_off[h] = (uint32_t)o[h]; ple_voc[h] = (uint32_t)v[h];
            if ((size_t)o[h] + v[h] > rows) FAIL("n-gram head %d runs past the table's %zu rows", h, rows);
        }
    }
    ev_g = [m.dev newSharedEvent];
    ev_c = [m.dev newSharedEvent];
    max_ctx = (int)env_d("CHRYSALIS_FN_CTX", 65536);
    alloc_buffers(max_seqs);
    alloc_prompt_buffers();
    mlock(warena.base.host(), warena.cap);
    if (has_mtp) {
        mtp_x = m.alloc((size_t)NE * XSLOT_MTP);
        if (mlock(mtp_x.host(), (size_t)NE * XSLOT_MTP)) fprintf(stderr, "chrysalis-fn: cannot wire the MTP experts (%s)\n", strerror(errno));
    }
    const double ws = m.dev.recommendedMaxWorkingSetSize, avail = mem_available(), total = mem_total();
    const double keep = env_d("CHRYSALIS_FN_KEEP_GB", 6.5) * 1e9 + env_d("CHRYSALIS_FN_SNAPSHOTS", 8) * state_bytes();
    double cache = std::min(ws - m.allocated - 0.75e9, avail - keep);
    cache = std::min(cache, total - keep - m.allocated - 1.0e9);
    if (getenv("CHRYSALIS_FN_CACHE_GB")) {
        cache = env_d("CHRYSALIS_FN_CACHE_GB", 0) * 1e9;
        if (has_mtp) cache -= (double)NE * XSLOT_MTP + mtp_dense + (double)n_pages * KV_LAYER;
    }
    const int stage = (int)env_d("CHRYSALIS_FN_STAGE", 256);
    if (cache < 2e9) FAIL("only %.1f GB of memory for the expert cache (%.1f GB available, %.1f GB kept for macOS)", cache / 1e9, avail / 1e9, keep / 1e9);
    fprintf(stderr, "chrysalis-fn: dense %.2f GB (%.1f s), buffers %.2f GB, expert cache %.2f GB (%.0f%% of the experts); %.1f GB available, working set %.1f GB\n",
            warena.cap / 1e9, t_dense - t0, (m.allocated - warena.cap) / 1e9, cache / 1e9, 100.0 * cache / (XSLOT * NL * NE), avail / 1e9, ws / 1e9);
    experts_init((size_t)cache, stage);
    if (has_mtp) load_mtp();
    experts_fill(60);
    g_progress = 1.0f;
    fprintf(stderr, "chrysalis-fn: loaded in %.1f s\n", now_s() - t0);
}

void Engine::alloc_buffers(int max_seqs) {
    max_pages_seq = (max_ctx + PAGE - 1) / PAGE;
    max_blocks = max_ctx / KPOOL + 1;
    n_pages = (int)env_d("CHRYSALIS_FN_KV_PAGES", max_pages_seq + max_pages_seq / 4);
    kv = m.alloc((size_t)n_pages * KV_PAGE);
    page_ref.assign(n_pages, 0);
    free_pages.clear();
    for (int p = n_pages - 1; p >= 0; p--) free_pages.push_back(p);
    const int T = TM;
    auto f = [&](size_t n) { return (size_t)T * n * 4; };
    size_t need = 0;
    const size_t sizes[] = {f(HCD), f(HCD), f(HCR), f(HCR), f(HCD), f(HC), f(HC), f(D), f(D), f(QKV), f(GV * GD), f(96), f(GV * GD), f(D),
                            f(2 * NH * HD), f(NKV * HD), f(NKV * HD), f(IDX_H * IDX_D), f(IDX_D), f(NH * HD), f(IDX_H * IDX_D),
                            f(max_blocks), f(SEL_BLOCKS + 1), f(1), f(NH * 16 * HD), f(NH * 16 * 2), f(NH * HD), f(NH * HD),
                            f(NE), f((NL + 1) * TOPK_E), f((NL + 1) * TOPK_E), f(1), f(FS), f(FS), f(FS), f(D), f(TOPK_E * FE), f(TOPK_E * D),
                            f(D), f(D), f(HCD), f(D), f(HCD), (size_t)T * PLE_HEADS * PLE_ROW, f(VOCAB), f(SAMPLE_N), f(SAMPLE_N), f(NSTAT),
                            f(128 * SAMPLE_N), f(128 * SAMPLE_N), f(1), (size_t)T * VOCAB / 8, f(HC), f((NL + 1) * TOPK_E),
                            (size_t)12 * 8192, (size_t)12 * 8192, (size_t)16 * NL, f(NL * TOPK_E), f(NL * TOPK_E), f(NE),
                            f((size_t)NGDN * QKV), f((size_t)NGDN * 96), f((size_t)NKVL * IDX_D), f(HCD), f(HCD), f(HCD), (size_t)DRAFT_VOCAB * 4, f(1)};
    for (size_t s : sizes) need += (s + 255) & ~size_t(255);
    barena.init(m, need);
    View *dst[] = {&b.res, &b.xn, &b.lo, &b.loh, &b.gate, &b.inj, &b.inj2, &b.mx, &b.mxh, &b.raw, &b.z, &b.ab, &b.go, &b.blk,
                   &b.qg, &b.kb, &b.vb, &b.qir, &b.kir, &b.Q, &b.Qi, &b.score, &b.sel, &b.nsel, &b.part_o, &b.part_ml, &b.att, &b.atth,
                   &b.rlog, &b.ids, &b.wts, &b.sgv, &b.shg, &b.shu, &b.shh, &b.shout, &b.xh, &b.xy,
                   &b.emb, &b.embh, &b.key, &b.value, &b.xconv, &b.ple_rows, &b.logits, &b.topv, &b.topi, &b.tops, &b.topv1, &b.topi1,
                   &b.tokens, &b.mask, &b.inv, &b.done, &b.ind_args, &b.ind_grids, &b.desc, &b.ids_pred, &b.wts_pred, &b.rlog_pred,
                   &b.vraw, &b.vab, &b.vkir, &b.memb, &b.mhp, &b.mres, &b.dlog, &b.dtok};
    static_assert(sizeof(sizes) / sizeof(sizes[0]) == sizeof(dst) / sizeof(dst[0]), "a size per buffer");
    for (size_t i = 0; i < sizeof(dst) / sizeof(dst[0]); i++) *dst[i] = barena.get(sizes[i]);
    seqs.assign(max_seqs, Seq{});
    for (Seq &s : seqs) {
        s.d_pages = m.alloc((size_t)max_pages_seq * 4);
        View st = m.alloc(state_bytes());
        s.ssm = st; s.conv = st.at(ssm_bytes()); s.ple = s.conv.at(conv_bytes()); s.tail = s.ple.at(ple_bytes()); s.h_last = s.tail.at(tail_bytes());
    }
}

void Engine::settle() {
    if (!pending) return;
    id<MTLCommandBuffer> cb = pending;
    pending = nil;
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
}

void Engine::shutdown() {
    settle();
    experts_stop();
    if (ngram_map) munmap(ngram_map, ngram_bytes);
    ngram_map = nullptr;
    if (ngram_fd >= 0) ::close(ngram_fd);
    ngram_fd = -1;
    for (int fd : g_nocache) ::close(fd);
    g_nocache.clear();
    seqs.clear();
    m.release_all();
}

int Engine::seq_new() {
    for (size_t i = 0; i < seqs.size(); i++) {
        if (seqs[i].used) continue;
        seqs[i].used = true;
        seq_reset(seqs[i]);
        return (int)i;
    }
    return -1;
}

void Engine::seq_reset(Seq &s) {
    settle();
    for (int p : s.pages) page_release(p);
    s.n = 0; s.pending = -1; s.pages.clear(); s.seen.clear(); s.top_v.clear(); s.top_i.clear();
    s.hist[0] = s.hist[1] = -1;
    s.look.clear();
    s.rng.seed(std::random_device{}());
    memset(s.ssm.host(), 0, state_bytes());
}

void Engine::seq_free(int id) {
    Seq &s = seqs.at(id);
    for (int p : s.pages) page_release(p);
    s.pages.clear();
    s.used = false;
}

int Engine::page_alloc() {
    if (free_pages.empty()) throw std::runtime_error("out of KV pages");
    const int p = free_pages.back();
    free_pages.pop_back();
    page_ref[p] = 1;
    return p;
}
void Engine::page_release(int p) { if (--page_ref[p] == 0) free_pages.push_back(p); }

void Engine::ensure_pages(Seq &s, int n_tokens, int from) {
    settle();
    if (n_tokens > max_ctx) throw std::runtime_error("past the context size");
    const int need = (n_tokens + PAGE - 1) / PAGE;
    int shared = 0;
    for (int i = from / PAGE; i < (int)s.pages.size() && i < need; i++) shared += page_ref[s.pages[i]] > 1;
    if ((int)free_pages.size() < need - (int)s.pages.size() + shared) throw std::runtime_error("out of KV pages");
    while ((int)s.pages.size() < need) s.pages.push_back(page_alloc());
    for (int i = from / PAGE; i < (int)s.pages.size() && i < need; i++) {
        if (page_ref[s.pages[i]] == 1) continue;
        const int np = page_alloc();
        memcpy(kv.host<uint8_t>() + (size_t)np * KV_PAGE, kv.host<uint8_t>() + (size_t)s.pages[i] * KV_PAGE, KV_PAGE);
        page_release(s.pages[i]);
        s.pages[i] = np;
    }
    memcpy(s.d_pages.host(), s.pages.data(), s.pages.size() * 4);
}

int Engine::snapshot(int id) {
    settle();
    Seq &s = seqs.at(id);
    size_t si = 0;
    while (si < snaps.size() && snaps[si].used) si++;
    if (si == snaps.size()) snaps.emplace_back();
    Snap &sn = snaps[si];
    sn.used = true;
    sn.n = s.n; sn.hist[0] = s.hist[0]; sn.hist[1] = s.hist[1];
    if (s.pending < 0) { sn.top_v = s.top_v; sn.top_i = s.top_i; } else { sn.top_v.clear(); sn.top_i.clear(); }
    sn.pages.assign(s.pages.begin(), s.pages.begin() + (s.n + PAGE - 1) / PAGE);
    for (int p : sn.pages) page_ref[p]++;
    sn.state.resize(state_bytes());
    memcpy(sn.state.data(), s.ssm.host(), state_bytes());
    return (int)si;
}

void Engine::restore(int id, int snap) {
    settle();
    Seq &s = seqs.at(id);
    Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("restore of a freed snapshot");
    for (int p : s.pages) page_release(p);
    s.pages = sn.pages;
    for (int p : s.pages) page_ref[p]++;
    if (!s.pages.empty()) memcpy(s.d_pages.host(), s.pages.data(), s.pages.size() * 4);
    s.n = sn.n; s.hist[0] = sn.hist[0]; s.hist[1] = sn.hist[1];
    s.pending = -1;
    s.top_v = sn.top_v; s.top_i = sn.top_i; s.seen.clear();
    memcpy(s.ssm.host(), sn.state.data(), state_bytes());
}

void Engine::snap_free(int snap) {
    Snap &sn = snaps.at(snap);
    if (!sn.used) throw std::runtime_error("double free of a snapshot");
    for (int p : sn.pages) page_release(p);
    sn.pages.clear();
    sn.used = false;
}

void Engine::ple_ids(int h0, int h1, const int *tokens, int T, size_t *rows) const {
    constexpr int64_t EOS = EOS_TEXT;
    for (int t = 0; t < T; t++) {
        const int64_t p1 = t >= 1 ? tokens[t - 1] : h1;
        const int64_t p2 = t >= 2 ? tokens[t - 2] : t == 1 ? h1 : h0;
        bool cut = p1 < 0 || p1 == EOS;
        const int64_t c1 = cut ? EOS : p1;
        cut = cut || p2 < 0 || p2 == EOS;
        const int64_t c2 = cut ? EOS : p2;
        const uint64_t m2 = (uint64_t)tokens[t] * ple_mult[0] ^ (uint64_t)c1 * ple_mult[1];
        const uint64_t m3 = m2 ^ (uint64_t)c2 * ple_mult[2];
        for (int g = 0; g < 8; g++) rows[(size_t)t * PLE_HEADS + g] = m2 % ple_voc[g] + ple_off[g];
        for (int g = 8; g < 16; g++) rows[(size_t)t * PLE_HEADS + g] = m3 % ple_voc[g] + ple_off[g];
    }
}

void Engine::ple_gather(const Seq &s, const int *tokens, int T, uint8_t *dst) const {
    std::vector<size_t> rows((size_t)T * PLE_HEADS);
    ple_ids(s.hist[0], s.hist[1], tokens, T, rows.data());
    if (T <= 8) {
        for (size_t i = 0; i < rows.size(); i++) {
            const uintptr_t a = (uintptr_t)(ngram + rows[i] * PLE_ROW), p0 = a & ~uintptr_t(16383);
            madvise((void *)p0, ((a + PLE_ROW - 1) & ~uintptr_t(16383)) - p0 + 16384, MADV_WILLNEED);
        }
        for (size_t i = 0; i < rows.size(); i++) memcpy(dst + i * PLE_ROW, ngram + rows[i] * PLE_ROW, PLE_ROW);
        return;
    }
    const int nt = 32;
    std::thread th[nt];
    for (int k = 0; k < nt; k++)
        th[k] = std::thread([&, k] {
            for (size_t i = rows.size() * k / nt; i < rows.size() * (k + 1) / nt; i++)
                if (pread(ngram_fd, dst + i * PLE_ROW, PLE_ROW, ngram_off + rows[i] * PLE_ROW) != PLE_ROW) FAIL("n-gram table read failed");
        });
    for (auto &t : th) t.join();
}

void Engine::gemv(Enc &e, const WT &w, View x, int ldx, View y, int ldy, int T, int add, float scale, View inj) {
    const int lpr = w.cols >= 2048 ? 32 : 8;
    GemvArgs a{(uint32_t)w.rows, (uint32_t)w.cols, (uint32_t)w.rb, (uint32_t)T, (uint32_t)ldx, (uint32_t)ldy, (uint32_t)add, scale, GGML_ACT};
    const int r = lpr == 8 ? 2 : T > 1 || w.rows >= 10240 ? 4 : 2;
    const int per_tg = 4 * (32 / lpr) * r;
    e.k(std::string(lpr == 32 ? (T > 1 ? "gemv32t_" : r == 4 ? "gemv32_" : "gemvr2_") : (T == 1 ? "gemv8_" : "gemv8t_")) + tname(w.type)).c(0, a).b(1, w.v).b(2, x).b(3, y).b(4, inj.buf ? inj : y)
        .grid((w.rows + per_tg - 1) / per_tg, 128);
}

struct HcArgs { uint32_t T, inject, ggml; };
void Engine::mixer(Enc &e, const Mix &mx, View res, int T, View inj) {
    static const bool old = getenv("CHRYSALIS_FN_OLD_MIXER") != nullptr;
    if (old) {
        uint32_t mode = 1;
        e.k("hc_norm").b(0, res).b(1, mx.norm).b(2, b.xn).b(3, b.embh).c(4, mode).grid(HC, T, 1, 256);
        e.barrier();
        gemv(e, mx.down, b.xn, HCD, b.lo, HCR, T);
        if (mx.inject.rows) gemv(e, mx.inject, b.xn, HCD, inj, HC, T);
        e.barrier();
        const uint32_t n = T * HCR;
        e.k("hc_lo").b(0, b.lo).b(1, b.loh).c(2, n).grid((n + 255) / 256, 256);
        e.barrier();
        gemv(e, mx.up, b.lo, HCR, b.gate, HCD, T);
        e.barrier();
        const uint32_t tt = T;
        e.k("hc_mix").b(0, b.xn).b(1, b.gate).b(2, b.mx).b(3, b.mxh).c(4, tt).grid((T * D + 255) / 256, 256);
        e.barrier();
        return;
    }
    HcArgs a{(uint32_t)T, mx.inject.rows ? 1u : 0u, GGML_ACT};
    e.k("hc_pre").c(0, a).b(1, res).b(2, mx.norm).b(3, mx.down.v).b(4, mx.inject.rows ? mx.inject.v : mx.down.v).b(5, b.lo)
        .b(6, inj.buf ? inj : b.inj).b(7, b.inv).grid((HCR + (mx.inject.rows ? HC : 0) + 1) / 2, 128);
    e.barrier();
    e.k("hc_post").c(0, a).b(1, res).b(2, mx.norm).b(3, b.inv).b(4, b.lo).b(5, mx.up.v).b(6, b.mx).b(7, b.mxh).grid(D / 4, 128);
    e.barrier();
}

void Engine::forward_small(Seq &s, const int *tokens, int T, bool all_logits, bool verify) {
    if (T > TM) FAIL("forward_small: %d tokens", T);
    settle();
    memcpy(b.tokens.host(), tokens, T * 4);
    experts_tick(T);
    ple_gather(s, tokens, T, b.ple_rows.host<uint8_t>());
    const uint32_t pos0 = s.n, tt = T, P = T * TOPK_E;
    id<MTLCommandBuffer> cb = [m.q commandBufferWithUnretainedReferences];
    Enc e;
    static id<MTLCounterSampleBuffer> tsb = nil;
    if (prof && !tsb) {
        for (id<MTLCounterSet> cs in m.dev.counterSets)
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
                d.counterSet = cs; d.storageMode = MTLStorageModeShared; d.sampleCount = 1024;
                NSError *err = nil;
                tsb = [m.dev newCounterSampleBufferWithDescriptor:d error:&err];
            }
        if (![m.dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) tsb = nil;
    }
    e.ts = prof ? tsb : nil;
    e.args = b.ind_args; e.grids = b.ind_grids;
    e.begin(m, cb);
    const uint32_t erb = (uint32_t)embd.rb;
    e.k("embed_q8_0").b(0, b.tokens).b(1, embd.v).c(2, erb).b(3, b.res).grid(T, 80);
    e.barrier();
    auto ids_of = [&](int l) { return b.ids.at((size_t)l * TM * TOPK_E * 4); };
    auto wts_of = [&](int l) { return b.wts.at((size_t)l * TM * TOPK_E * 4); };
    auto done_of = [&](int l) { return b.done.at((size_t)l * TM * TOPK_E * 4); };
    auto gdn_index = [](int l) { return l - (l + 1) / 4; };
    static const std::string skips = getenv("CHRYSALIS_FN_SKIP") ? getenv("CHRYSALIS_FN_SKIP") : "";
    auto skip = [&](const char *k) { return skips.find(k) != std::string::npos; };
    auto pre = [&](int l) {
        const Layer &w = L[l];
        if (l == PLE_LAYER) {
            const uint32_t nb = T * PLE_HEADS * 5;
            e.k("ple_embed").b(0, b.ple_rows).b(1, b.emb).b(2, b.embh).c(3, nb).grid((nb + 255) / 256, 256);
            e.barrier();
            gemv(e, w.ple_key, b.emb, D, b.key, HCD, T);
            gemv(e, w.ple_value, b.emb, D, b.value, D, T);
            e.barrier();
            e.k("ple_gate").b(0, b.key).b(1, b.value).b(2, b.res).b(3, w.ple_nk).b(4, w.ple_nq).b(5, w.ple_nc).b(6, b.xconv).grid(HC, T, 1, 256);
            e.barrier();
            e.k("ple_conv").b(0, b.xconv).b(1, s.ple).b(2, w.ple_conv).b(3, b.res).c(4, tt).grid((T * HCD + 255) / 256, 256);
            e.barrier();
            if (!verify) e.k("ple_commit").b(0, s.ple).b(1, b.xconv).c(2, tt).grid(HCD / 256, 256);
            e.barrier();
        }
        if (!skip("mix")) mixer(e, w.hca, b.res, T, b.inj);
        if (is_attn_layer(l) && skip("attn")) {
        } else if (!is_attn_layer(l) && skip("gdn")) {
        } else if (is_attn_layer(l)) {
            const int al = l / 4;
            const View kir = verify ? b.vkir.at((size_t)al * TM * IDX_D * 4) : b.kir;
            gemv(e, w.q, b.mx, D, b.qg, 2 * NH * HD, T);
            gemv(e, w.k, b.mx, D, b.kb, NKV * HD, T);
            gemv(e, w.v, b.mx, D, b.vb, NKV * HD, T);
            gemv(e, w.iq, b.mx, D, b.qir, IDX_H * IDX_D, T);
            gemv(e, w.ik, b.mx, D, kir, IDX_D, T);
            e.barrier();
            const int nb_end = (pos0 + T) / KPOOL;
            const int nsel_max = std::min(nb_end, SEL_BLOCKS) + 1;
            const uint32_t nsplit = std::max(1, std::min(16, nsel_max / 16));
            KVArgs ka{tt, pos0, (uint32_t)al, nsplit};
            View tail = s.tail.at((size_t)al * KPOOL * IDX_D * 2);
            e.k("attn_prep").c(0, ka).b(1, b.qg).b(2, b.kb).b(3, b.vb).b(4, w.qn).b(5, w.kn).b(6, kv).b(7, s.d_pages).b(8, b.Q).grid(NH + 2 * NKV, T, 1, 256);
            e.k("idx_qprep").c(0, ka).b(1, b.qir).b(2, w.iqn).b(3, b.Qi).grid(IDX_H, T, 1, 128);
            const int first = pos0 / KPOOL, last = (pos0 + T) / KPOOL - 1;
            if (last >= first) e.k("idx_pool").c(0, ka).b(1, kir).b(2, tail).b(3, w.ikn).b(4, kv).b(5, s.d_pages).grid(last - first + 1, 128);
            e.barrier();
            if (!verify) e.k("idx_tail").c(0, ka).b(1, kir).b(2, tail).grid(1, 128);
            const uint32_t ld = std::max(1, nb_end);
            if (nb_end > SEL_BLOCKS) e.k("idx_score").c(0, ka).b(1, b.Qi).b(2, kv).b(3, s.d_pages).b(4, b.score).c(5, ld).grid((ld + 255) / 256, T, 1, 256);
            e.barrier();
            e.k("idx_select").c(0, ka).b(1, b.score).c(2, ld).b(3, b.sel).b(4, b.nsel).grid(T, 1024);
            e.barrier();
            e.k("attn_dec").c(0, ka).b(1, b.Q).b(2, kv).b(3, s.d_pages).b(4, b.sel).b(5, b.nsel).b(6, b.part_o).b(7, b.part_ml).grid(nsplit, T, NKV, 32 * QPK);
            e.barrier();
            e.k("attn_combine").c(0, ka).b(1, b.part_o).b(2, b.part_ml).b(3, b.qg).b(4, b.att).b(5, b.atth).grid(NH, T, 1, 256);
            e.barrier();
            gemv_combine(e, w.o, b.att, NH * HD, T, b.inj);
        } else {
            const int gl = gdn_index(l);
            const View raw = verify ? b.vraw.at((size_t)gl * TM * QKV * 4) : b.raw, ab = verify ? b.vab.at((size_t)gl * TM * 96 * 4) : b.ab;
            gemv(e, w.qkv, b.mx, D, raw, QKV, T);
            gemv(e, w.z, b.mx, D, b.z, GV * GD, T);
            gemv(e, w.ab, b.mx, D, ab, 2 * GV, T);
            e.barrier();
            GdnStepArgs ga{tt, verify ? 0u : 1u};
            View conv = s.conv.at((size_t)gl * QKV * 3 * 4), S = s.ssm.at((size_t)gl * GV * GD * GD * 4);
            e.k("gdn_step").c(0, ga).b(1, raw).b(2, ab).b(3, b.z).b(4, w.conv_w).b(5, conv).b(6, w.A).b(7, w.dt).b(8, S)
                .b(9, w.ssm_norm).b(10, b.go).grid(GV, 512);
            e.barrier();
            if (!verify) e.k("gdn_conv_commit").b(0, conv).b(1, raw).c(2, tt).grid(QKV / 256, 256);
            gemv_combine(e, w.ssm_out, b.go, GV * GD, T, b.inj);
        }
        e.barrier();
        if (!skip("mix")) mixer(e, w.hcf, b.res, T, b.inj2);
        gemv(e, w.router, b.mx, D, b.rlog, NE, T);
        gemv(e, w.sg, b.mx, D, b.sgv, 1, T);
        gemv(e, w.sh_gate, b.mx, D, b.shg, FS, T);
        gemv(e, w.sh_up, b.mx, D, b.shu, FS, T);
        e.barrier();
        e.k("route_topk").b(0, b.rlog).b(1, ids_of(l)).b(2, wts_of(l)).grid(T, 32);
        const uint32_t nsh = T * FS;
        e.k("swiglu").b(0, b.shg).b(1, b.shu).b(2, b.shh).c(3, nsh).grid((nsh + 255) / 256, 256);
        e.barrier();
    };
    auto phase = [&](int l, uint32_t ph) {
        if (skip("moe") && ph == 0) { e.k("zero_u32").b(0, done_of(l)).c(1, P).grid(1, 128); e.barrier(); }
        if (skip("moe")) return;
        MoeArgs ma{tt, (uint32_t)l, ph, GGML_ACT};
        e.k("moe_gu").c(0, ma).b(1, xc.table).b(2, ids_of(l)).b(3, b.mx).b(4, b.xh).b(5, done_of(l)).b(6, xc.fmt).grid(T * TOPK_E * FE / 16, 128);
        if (ph == 0) gemv(e, L[l].sh_down, b.shg, FS, b.shout, D, T);
        static const bool predict = getenv("CHRYSALIS_FN_AHEAD") != nullptr;
        if (ph == 0 && predict && l + 1 < NL) gemv(e, L[l + 1].router, b.mx, D, b.rlog_pred, NE, T);
        e.barrier();
        if (ph == 0 && predict && l + 1 < NL) {
            e.k("route_topk").b(0, b.rlog_pred).b(1, b.ids_pred.at((size_t)(l + 1) * TM * TOPK_E * 4)).b(2, b.wts_pred).grid(T, 32);
        }
        e.k("moe_down").c(0, ma).b(1, xc.table).b(2, ids_of(l)).b(3, b.xh).b(4, b.xy).b(5, done_of(l)).grid(T * TOPK_E * D / 16, 128);
        e.barrier();
    };
    const int rows = all_logits ? T : 1;
    static View layer_dump;
    static const char *dump_path = getenv("CHRYSALIS_FN_DUMP");
    if (dump_path && !layer_dump.buf) layer_dump = m.alloc((size_t)NL * HCD * 4);
    auto rest = [&](int l) {
        const MoeArgs mc{tt, (uint32_t)l, 0, 0};
        e.k("moe_combine").c(0, mc).b(1, b.xy).b(2, wts_of(l)).b(3, b.sgv).b(4, b.shout).b(5, b.inj2).b(6, b.res).grid((T * D + 255) / 256, 256);
        e.barrier();
        if (dump_path) {
            const uint32_t n = HCD;
            e.k("copy_f32").b(0, b.res.at((size_t)(T - 1) * HCD * 4)).b(1, layer_dump.at((size_t)l * HCD * 4)).c(2, n).grid(HCD / 256, 256);
            e.barrier();
        }
        if (l + 1 < NL) { pre(l + 1); return; }
        mixer(e, hc_head, b.res, T, View{});
        if (!skip("head")) gemv(e, out, all_logits ? b.mx : b.mx.at((size_t)(T - 1) * D * 4), D, b.logits, VOCAB, rows);
        e.barrier();
        topk(e, rows);
    };
    std::vector<uint64_t> routed(NL), phase0(NL), resolved(NL);
    static const bool nosync = getenv("CHRYSALIS_FN_NOSYNC") != nullptr;
    uint32_t *desc = b.desc.host<uint32_t>();
    pre(0);
    for (int l = 0; l < NL; l++) {
        e.end();
        routed[l] = ++evg;
        if (!nosync) [cb encodeSignalEvent:ev_g value:routed[l]];
        e.begin(m, cb);
        phase(l, 0);
        const uint32_t lu = l;
        e.k("spec_check").b(0, done_of(l)).c(1, P).b(2, b.desc).c(3, lu).b(4, b.ind_grids).b(5, b.ind_args).grid(1, 256);
        e.barrier();
        desc[4 * l] = e.nind;
        e.ind = true; rest(l); e.ind = false;
        desc[4 * l + 1] = e.nind;
        e.end();
        phase0[l] = ++evg;
        resolved[l] = ++evc;
        if (!nosync) {
            [cb encodeSignalEvent:ev_g value:phase0[l]];
            [cb encodeWaitForEvent:ev_c value:resolved[l]];
        }
        e.begin(m, cb);
        phase(l, 1);
        desc[4 * l + 2] = e.nind;
        e.ind = true; rest(l); e.ind = false;
        desc[4 * l + 3] = e.nind;
    }
    e.end();
    if (e.nind > 8192) FAIL("%u indirect dispatches", e.nind);
    [cb commit];
    const double tw0 = now_s();
    double trc[NL][2] = {};
    int nplaced[NL] = {};
    for (int l = 0; l < NL && !nosync; l++) {
        const double ta = now_s();
        wait_event(ev_g, routed[l], cb);
        const double tb = now_s();
        t_wait += tb - ta;
        if (prof && l > 0) {
            const int *ids = ids_of(l).host<int>(), *pr = b.ids_pred.at((size_t)l * TM * TOPK_E * 4).host<int>();
            for (uint32_t i = 0; i < P; i++) {
                if (xc.slot_of[l * NE + ids[i]] >= 0) continue;
                xc.pred_misses++;
                for (uint32_t j = 0; j < P; j++) if (pr[j] == ids[i]) { xc.pred_caught++; break; }
            }
        }
        const int placed = experts_resolve(l, ids_of(l).host<int>(), P, [&] { wait_event(ev_g, phase0[l], cb); });
        t_io += now_s() - tb;
        nplaced[l] = placed;
        ev_c.signaledValue = resolved[l];
        static const bool ahead = getenv("CHRYSALIS_FN_AHEAD") != nullptr;
        if (ahead && l + 1 < NL) {
            wait_event(ev_g, phase0[l], cb);
            experts_ahead(l + 1, (placed ? b.ids_pred.at((size_t)(l + 1) * TM * TOPK_E * 4) : ids_of(l + 1)).host<int>(), P);
        }
        trc[l][0] = tb; trc[l][1] = now_s();
    }
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
    if (prof) {
        fprintf(stderr, "pass T=%d: %.2f ms (GPU %.2f ms), waiting %.2f ms, experts %.2f ms\n", T, (now_s() - tw0) * 1e3,
                (cb.GPUEndTime - cb.GPUStartTime) * 1e3, t_wait * 1e3, t_io * 1e3);
        if (e.ts && e.ts_n >= 2) {
            NSData *d = [e.ts resolveCounterRange:NSMakeRange(0, e.ts_n)];
            const MTLCounterResultTimestamp *r = (const MTLCounterResultTimestamp *)d.bytes;
            double work = 0, gap = 0;
            for (uint32_t k = 0; k + 1 < e.ts_n; k += 2) {
                work += r[k + 1].timestamp - r[k].timestamp;
                if (k + 2 < e.ts_n) gap += r[k + 2].timestamp - r[k + 1].timestamp;
            }
            fprintf(stderr, "  GPU: encoders %.2f ms, gaps %.2f ms (%u encoders)\n", work / 1e6, gap / 1e6, e.ts_n / 2);
            MTLTimestamp c0t, g0t;
            [m.dev sampleTimestamps:&c0t gpuTimestamp:&g0t];
            mach_timebase_info_data_t tbi; mach_timebase_info(&tbi);
            const double off = now_s() - mach_absolute_time() * 1e-9 * tbi.numer / tbi.denom;
            auto cpu = [&](uint64_t gt) { return ((double)gt - (double)g0t + (double)c0t) * 1e-9 + off; };
            double gm = 0, gh = 0, lag = 0, res_lag = 0; int nm = 0;
            for (int l = 0; l < NL; l++) {
                const double r_end = cpu(r[2 * (2 * l) + 1].timestamp), a_s = cpu(r[2 * (2 * l + 1)].timestamp),
                             a_e = cpu(r[2 * (2 * l + 1) + 1].timestamp), b_s = cpu(r[2 * (2 * l + 2)].timestamp);
                (nplaced[l] ? gm : gh) += (b_s - a_e) + (a_s - r_end);
                nm += nplaced[l] > 0;
                lag += trc[l][0] - r_end;
                res_lag += b_s - std::max(a_e, trc[l][1]);
                if (prof > 1 && (b_s - a_e > 1e-3 || a_s - r_end > 1e-3))
                    fprintf(stderr, "  layer %2d: routed->A %.0f us, A %.0f us, A->B %.0f us; CPU saw routing %.0f us after it, set c %.0f us before A ended\n",
                            l, (a_s - r_end) * 1e6, (a_e - a_s) * 1e6, (b_s - a_e) * 1e6, (trc[l][0] - r_end) * 1e6, (a_e - trc[l][1]) * 1e6);
            }
            fprintf(stderr, "  HS: miss layers %d, gap at miss %.2f ms, gap no miss %.2f ms, CPU sees routing +%.0f us avg, GPU resumes +%.0f us avg\n",
                    nm, gm * 1e3, gh * 1e3, lag / NL * 1e6, res_lag / NL * 1e6);
        }
    }
    t_wait = t_io = 0;
    if (dump_path) if (FILE *f = fopen(dump_path, "wb")) { fwrite(layer_dump.host(), 4, (size_t)NL * HCD, f); fclose(f); }
}

void Engine::wait_event(id<MTLSharedEvent> ev, uint64_t v, id<MTLCommandBuffer> cb) {
    if (ev.signaledValue >= v) return;
    static MTLSharedEventListener *lis = [[MTLSharedEventListener alloc] initWithDispatchQueue:
        dispatch_queue_create_with_target("chrysalis.fn.events", DISPATCH_QUEUE_SERIAL, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0))];
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [ev notifyListener:lis atValue:v block:^(id<MTLSharedEvent>, uint64_t) { dispatch_semaphore_signal(sem); }];
    static const bool warm = !getenv("CHRYSALIS_FN_NOWARM");
    while (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (warm ? 50 : 20000) * NSEC_PER_USEC))) {
        if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
        if (warm) keep_warm();
    }
}

void Engine::topk(Enc &e, int rows) {
    const uint32_t nv = VOCAB;
    e.k("logit_stats").b(0, b.logits).b(1, b.tops).c(2, nv).grid(rows, 1, 1, 1024);
    uint32_t n = VOCAB, ids = 0;
    View v = b.logits, vi = b.logits, ov = b.topv1, oi = b.topi1;
    for (;;) {
        const uint32_t nch = (n + 2047) / 2048;
        if (nch == 1) { ov = b.topv; oi = b.topi; }
        uint32_t a[2] = {n, ids};
        e.k("topk64").b(0, v).b(1, vi).c(2, a).b(3, ov).b(4, oi).grid(nch, rows, 1, 256);
        e.barrier();
        if (nch == 1) return;
        n = nch * SAMPLE_N; ids = 1;
        v = ov; vi = oi;
        ov = v.buf == b.topv1.buf && v.off == b.topv1.off ? b.score : b.topv1;
        oi = vi.off == b.topi1.off ? b.sel : b.topi1;
    }
}

void Engine::advance(Seq &s, const int *tokens, int T) {
    if (T >= 2) { s.hist[0] = tokens[T - 2]; s.hist[1] = tokens[T - 1]; }
    else { s.hist[0] = s.hist[1]; s.hist[1] = tokens[0]; }
    s.n += T;
}

void Engine::prefill(int id, const int *tokens, int n) {
    Seq &s = seqs.at(id);
    for (int i = 0; i < n; i++) if (tokens[i] < 0 || tokens[i] >= VOCAB) throw std::invalid_argument("token id out of range");
    ensure_pages(s, s.n + n, s.n);
    s.pending = -1;
    for (int done = 0; done < n;) {
        const int T = std::min(n - done <= TM ? TM : PMAX, n - done);
        if (T <= TM || getenv("CHRYSALIS_FN_DECODE_PREFILL")) {
            const int t = std::min(TM, T);
            forward_small(s, tokens + done, t, false);
            if (has_mtp) mtp_catch_up(s, t, s.n, false);
            advance(s, tokens + done, t);
            done += t;
            continue;
        }
        prompt_pass(s, tokens + done, T);
        advance(s, tokens + done, T);
        done += T;
    }
    keep_top(s);
}

void Engine::keep_top(Seq &s, int row) {
    const float *v = b.topv.host<float>() + (size_t)row * SAMPLE_N, *st = b.tops.host<float>() + (size_t)row * NSTAT;
    const int *i = b.topi.host<int>() + (size_t)row * SAMPLE_N;
    s.top_v.assign(v, v + SAMPLE_N);
    s.top_v.insert(s.top_v.end(), st, st + NSTAT);
    s.top_i.assign(i, i + SAMPLE_N);
    s.top_i.resize(SAMPLE_N + NSTAT, 0);
    last_row = row;
}

const float *top_stats(const Seq &s) {
    static const float unknown[NSTAT] = {(float)VOCAB, -1, 0, 0, 0, 0};
    return s.top_v.size() >= (size_t)(SAMPLE_N + NSTAT) ? s.top_v.data() + SAMPLE_N : unknown;
}

static void keep_row(const float *v, const int *i, float *rows_v, int *rows_i, int at) {
    std::copy(v, v + SAMPLE_N, rows_v + (size_t)at * SAMPLE_N);
    std::copy(i, i + SAMPLE_N, rows_i + (size_t)at * SAMPLE_N);
}

int Engine::generate(int id, const Sampling &sp, int *out_, int max_out, float *rows_v, int *rows_i) {
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    int nout = 0;
    if (s.pending < 0) {
        if (s.top_v.empty()) FAIL("generate: nothing to continue from");
        s.pending = sample(s.seen, s.top_v.data(), s.top_i.data(), top_stats(s), sp, s.rng);
        if (rows_v) keep_row(s.top_v.data(), s.top_i.data(), rows_v, rows_i, nout);
        out_[nout++] = s.pending;
        if (sp.lookup > 0) s.look.push(s.pending);
        if (is_eos(s.pending) || s.pending == sp.halt || nout == max_out) return nout;
    }
    const int room = std::min(3, max_out - nout - 1);
    int copied[4] = {s.pending};
    int k = sp.lookup > 0 && room > 0 ? s.look.find(std::min(sp.lookup, room), VOCAB, copied + 1) : 0;
    if (k == 0 && has_mtp && room > 0 && sp.n_draft > 0) {
        k = std::min(sp.n_draft, room);
        ensure_pages(s, s.n + k + 1, s.n);
        mtp_draft(s, k, copied + 1);
    }
    return verify(s, sp, copied, k, false, out_, nout, rows_v, rows_i);
}

int Engine::verify(Seq &s, const Sampling &sp, const int *drafts, int k, bool masked, int *out, int nout, float *rows_v, int *rows_i) {
    const int q = s.n, T = k + 1;
    ensure_pages(s, q + T, q);
    forward_small(s, drafts, T, true, k > 0);
    if (has_mtp) mtp_catch_up(s, T, q, k > 0);
    if (masked) {
        id<MTLCommandBuffer> cb = [m.q commandBufferWithUnretainedReferences];
        Enc en; en.begin(m, cb);
        const uint32_t n = VOCAB;
        en.k("mask_logits").b(0, b.logits).b(1, b.mask).c(2, n).grid((VOCAB + 255) / 256, T, 1, 256);
        en.barrier();
        topk(en, T);
        en.end();
        [cb commit]; [cb waitUntilCompleted];
    }
    const float *tv = b.topv.host<float>(), *ts = b.tops.host<float>();
    const int *ti = b.topi.host<int>();
    const int a = verify_drafts(s.seen, tv, ti, ts, drafts, k, sp, s.rng, [&](int t) { return is_eos(t) || t == sp.halt; },
                                [&](int r, int t) {
                                    if (rows_v) keep_row(tv + (size_t)r * SAMPLE_N, ti + (size_t)r * SAMPLE_N, rows_v, rows_i, nout);
                                    out[nout++] = t;
                                    if (sp.lookup > 0) s.look.push(t);
                                });
    if (k > 0) commit(s, q, a + 1);
    advance(s, drafts, a + 1);
    keep_top(s, a);
    s.pending = out[nout - 1];
    return nout;
}

void Engine::commit(Seq &s, int q, int n) {
    id<MTLCommandBuffer> cb = [m.q commandBufferWithUnretainedReferences];
    Enc e; e.begin(m, cb);
    const uint32_t nn = n;
    const GdnStepArgs ga{nn, 1};
    for (int l = 0; l < NL; l++) {
        if (is_attn_layer(l)) continue;
        const int gl = l - (l + 1) / 4;
        const Layer &w = L[l];
        e.k("gdn_step").c(0, ga).b(1, b.vraw.at((size_t)gl * TM * QKV * 4)).b(2, b.vab.at((size_t)gl * TM * 96 * 4)).b(3, b.z).b(4, w.conv_w)
            .b(5, s.conv.at((size_t)gl * QKV * 3 * 4)).b(6, w.A).b(7, w.dt).b(8, s.ssm.at((size_t)gl * GV * GD * GD * 4)).b(9, w.ssm_norm)
            .b(10, b.go).grid(GV, 512);
    }
    e.barrier();
    for (int gl = 0; gl < NGDN; gl++)
        e.k("gdn_conv_commit").b(0, s.conv.at((size_t)gl * QKV * 3 * 4)).b(1, b.vraw.at((size_t)gl * TM * QKV * 4)).c(2, nn).grid(QKV / 256, 256);
    e.k("ple_commit").b(0, s.ple).b(1, b.xconv).c(2, nn).grid(HCD / 256, 256);
    if (has_mtp) {
        const uint32_t nh = HCD;
        e.k("copy_f32").b(0, b.res.at((size_t)(n - 1) * HCD * 4)).b(1, s.h_last).c(2, nh).grid(HCD / 256, 256);
    }
    for (int al = 0; al < (has_mtp ? NKVL : NATT); al++) {
        const KVArgs ka{nn, (uint32_t)q, (uint32_t)al, 1};
        e.k("idx_tail").c(0, ka).b(1, b.vkir.at((size_t)al * TM * IDX_D * 4)).b(2, s.tail.at((size_t)al * KPOOL * IDX_D * 2)).grid(1, 128);
    }
    e.end();
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
}

void Engine::mtp_layer(Enc &e, Seq &s, int T, uint32_t pos0, View hp, View tok, bool spec) {
    const uint32_t tt = T, erb = (uint32_t)embd.rb;
    e.k("embed_q8_0").b(0, tok).b(1, embd.v).c(2, erb).b(3, b.memb).grid(T, 80);
    e.barrier();
    e.k("mtp_norm").b(0, b.memb).b(1, hp).b(2, enorm).b(3, hnorm).b(4, b.emb).b(5, b.key).grid(HC, T, 1, 256);
    e.barrier();
    gemv(e, eh_e, b.emb, D, b.blk, D, T);
    for (int r0 = 0; r0 < T * HC; r0 += 8)
        gemv(e, eh_h, b.key.at((size_t)r0 * D * 4), D, b.mres.at((size_t)r0 * D * 4), D, std::min(8, T * HC - r0));
    e.barrier();
    const uint32_t nr = T * HCD;
    e.k("hc_add_row").b(0, b.mres).b(1, b.blk).c(2, tt).grid((nr + 255) / 256, 256);
    e.barrier();
    const Layer &w = M;
    mixer(e, w.hca, b.mres, T, b.inj);
    const View kir = spec ? b.vkir.at((size_t)NATT * TM * IDX_D * 4) : b.kir;
    gemv(e, w.q, b.mx, D, b.qg, 2 * NH * HD, T);
    gemv(e, w.k, b.mx, D, b.kb, NKV * HD, T);
    gemv(e, w.v, b.mx, D, b.vb, NKV * HD, T);
    gemv(e, w.iq, b.mx, D, b.qir, IDX_H * IDX_D, T);
    gemv(e, w.ik, b.mx, D, kir, IDX_D, T);
    e.barrier();
    const int nb_end = (pos0 + T) / KPOOL;
    const int nsel_max = std::min(nb_end, SEL_BLOCKS) + 1;
    const uint32_t nsplit = std::max(1, std::min(16, nsel_max / 16));
    const KVArgs ka{tt, pos0, (uint32_t)NATT, nsplit};
    const View tail = s.tail.at((size_t)NATT * KPOOL * IDX_D * 2);
    e.k("attn_prep").c(0, ka).b(1, b.qg).b(2, b.kb).b(3, b.vb).b(4, w.qn).b(5, w.kn).b(6, kv).b(7, s.d_pages).b(8, b.Q).grid(NH + 2 * NKV, T, 1, 256);
    e.k("idx_qprep").c(0, ka).b(1, b.qir).b(2, w.iqn).b(3, b.Qi).grid(IDX_H, T, 1, 128);
    const int first = pos0 / KPOOL, last = (pos0 + T) / KPOOL - 1;
    if (last >= first) e.k("idx_pool").c(0, ka).b(1, kir).b(2, tail).b(3, w.ikn).b(4, kv).b(5, s.d_pages).grid(last - first + 1, 128);
    e.barrier();
    if (!spec) e.k("idx_tail").c(0, ka).b(1, kir).b(2, tail).grid(1, 128);
    const uint32_t ld = std::max(1, nb_end);
    if (nb_end > SEL_BLOCKS) e.k("idx_score").c(0, ka).b(1, b.Qi).b(2, kv).b(3, s.d_pages).b(4, b.score).c(5, ld).grid((ld + 255) / 256, T, 1, 256);
    e.barrier();
    e.k("idx_select").c(0, ka).b(1, b.score).c(2, ld).b(3, b.sel).b(4, b.nsel).grid(T, 1024);
    e.barrier();
    e.k("attn_dec").c(0, ka).b(1, b.Q).b(2, kv).b(3, s.d_pages).b(4, b.sel).b(5, b.nsel).b(6, b.part_o).b(7, b.part_ml).grid(nsplit, T, NKV, 32 * QPK);
    e.barrier();
    e.k("attn_combine").c(0, ka).b(1, b.part_o).b(2, b.part_ml).b(3, b.qg).b(4, b.att).b(5, b.atth).grid(NH, T, 1, 256);
    e.barrier();
    gemv(e, w.o, b.att, NH * HD, b.mres, HCD, T, 2, 1.0f, b.inj);
    e.barrier();
    mixer(e, w.hcf, b.mres, T, b.inj2);
    gemv(e, w.router, b.mx, D, b.rlog, NE, T);
    gemv(e, w.sg, b.mx, D, b.sgv, 1, T);
    gemv(e, w.sh_gate, b.mx, D, b.shg, FS, T);
    gemv(e, w.sh_up, b.mx, D, b.shu, FS, T);
    e.barrier();
    const View ids = b.ids.at((size_t)NL * TM * TOPK_E * 4), wts = b.wts.at((size_t)NL * TM * TOPK_E * 4), done = b.done.at((size_t)NL * TM * TOPK_E * 4);
    e.k("route_topk").b(0, b.rlog).b(1, ids).b(2, wts).grid(T, 32);
    const uint32_t nsh = T * FS;
    e.k("swiglu").b(0, b.shg).b(1, b.shu).b(2, b.shh).c(3, nsh).grid((nsh + 255) / 256, 256);
    e.barrier();
    const MoeArgs ma{tt, (uint32_t)NL, 0, GGML_ACT};
    e.k("moe_gu").c(0, ma).b(1, xc.table).b(2, ids).b(3, b.mx).b(4, b.xh).b(5, done).b(6, xc.fmt).grid(T * TOPK_E * FE / 16, 128);
    gemv(e, w.sh_down, b.shg, FS, b.shout, D, T);
    e.barrier();
    e.k("moe_down").c(0, ma).b(1, xc.table).b(2, ids).b(3, b.xh).b(4, b.xy).b(5, done).grid(T * TOPK_E * D / 16, 128);
    e.barrier();
    e.k("moe_combine").c(0, ma).b(1, b.xy).b(2, wts).b(3, b.sgv).b(4, b.shout).b(5, b.inj2).b(6, b.mres).grid((T * D + 255) / 256, 256);
    e.barrier();
}

void Engine::mtp_catch_up(Seq &s, int T, int pos0, bool spec) {
    id<MTLCommandBuffer> cb = [m.q commandBufferWithUnretainedReferences];
    Enc e; e.begin(m, cb);
    const uint32_t nh = HCD, nr = (T - 1) * HCD;
    e.k("copy_f32").b(0, s.h_last).b(1, b.mhp).c(2, nh).grid(HCD / 256, 256);
    if (T > 1) e.k("copy_f32").b(0, b.res).b(1, b.mhp.at((size_t)HCD * 4)).c(2, nr).grid((nr + 255) / 256, 256);
    e.barrier();
    mtp_layer(e, s, T, (uint32_t)pos0, b.mhp, b.tokens, spec);
    if (!spec) e.k("copy_f32").b(0, b.res.at((size_t)(T - 1) * HCD * 4)).b(1, s.h_last).c(2, nh).grid(HCD / 256, 256);
    e.end();
    [cb commit];
    pending = cb;
}

void Engine::mtp_draft(Seq &s, int k, int *out, int nfixed) {
    int *tok = b.dtok.host<int>();
    tok[0] = s.pending;
    for (int i = 0; i < nfixed; i++) tok[i + 1] = out[i];
    id<MTLCommandBuffer> cb = [m.q commandBufferWithUnretainedReferences];
    Enc e; e.begin(m, cb);
    const uint32_t nh = HCD, nv = DRAFT_VOCAB;
    for (int i = 0; i < k; i++) {
        if (i > 0) {
            e.k("copy_f32").b(0, b.mres).b(1, b.mhp).c(2, nh).grid(HCD / 256, 256);
            e.barrier();
        }
        mtp_layer(e, s, 1, (uint32_t)(s.n + i), i == 0 ? s.h_last : b.mhp, b.dtok.at((size_t)i * 4), true);
        if (i < nfixed) continue;
        mixer(e, mtp_head, b.mres, 1, View{});
        gemv(e, dout, b.mx, D, b.dlog, DRAFT_VOCAB, 1);
        e.barrier();
        e.k("argmax").b(0, b.dlog).c(1, nv).b(2, b.dtok.at((size_t)(i + 1) * 4)).grid(1, 1024);
        e.barrier();
    }
    e.end();
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) FAIL("GPU error: %s", cb.error.localizedDescription.UTF8String);
    for (int i = nfixed; i < k; i++) out[i] = tok[i + 1];
}

void Engine::load_mtp() {
    const std::string p = "blk." + std::to_string(NL) + ".";
    const GGUFTensor *gt = g.get(p + "ffn_gate_mtp.weight"), *ut = g.get(p + "ffn_up_mtp.weight"), *dt = g.get(p + "ffn_down_mtp.weight");
    auto shaped = [](const GGUFTensor *t, int type, int64_t cols, int64_t rows) { return t && t->type == type && t->ne[0] == cols && t->ne[1] == rows && t->ne[2] == NE; };
    if (!shaped(gt, T_Q3_K, D, FE) || !shaped(ut, T_Q3_K, D, FE) || !shaped(dt, T_IQ4_NL, FE, D))
        FAIL("MTP layer: its experts are not Q3_K gate / up and IQ4_NL down, %d of them", NE);
    const size_t gu = xgu_bytes(T_Q3_K);
    std::vector<Experts::Job> jobs;
    uint64_t *table = (uint64_t *)xc.table.host();
    for (int e = 0; e < NE; e++) {
        uint8_t *dst = mtp_x.host<uint8_t>() + (size_t)e * XSLOT_MTP;
        jobs.push_back({xc.fd, dt->off + (size_t)e * XDN, XDN, dst, nullptr});
        jobs.push_back({xc.fd, gt->off + (size_t)e * gu, gu, dst + XDN, nullptr});
        jobs.push_back({xc.fd, ut->off + (size_t)e * gu, gu, dst + XDN + gu, nullptr});
        table[(size_t)NL * NE + e] = mtp_x.gpu() + (size_t)e * XSLOT_MTP;
        xc.fmt.host<uint8_t>()[(size_t)NL * NE + e] = T_Q3_K;
    }
    io_submit(jobs);
}

int Engine::draft(int id, const Sampling &sp, const int *fixed, int nfixed, int max_k, int *out) {
    Seq &s = seqs.at(id);
    if (s.pending < 0) FAIL("draft: no pending token");
    const int room = std::min(3, max_k);
    if (room < 1) return 0;
    int copied[4] = {s.pending};
    const int nf = std::min(nfixed, room);
    std::copy(fixed, fixed + nf, copied + 1);
    const int nl = nf == 0 && sp.lookup > 0 ? s.look.find(std::min(sp.lookup, room), VOCAB, copied + 1) : 0;
    const int k = nl > 0 ? nl : has_mtp ? std::min(room, nf + std::max(0, sp.n_draft)) : nf;
    if (k > nf + nl) {
        ensure_pages(s, s.n + k + 1, s.n);
        mtp_draft(s, k, copied + 1, nf);
    }
    std::copy(copied + 1, copied + 1 + k, out);
    return k;
}

int Engine::generate_masked(int id, const Sampling &sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out) {
    if (max_out < 1) return 0;
    Seq &s = seqs.at(id);
    if (s.pending < 0) FAIL("generate_masked: no pending token");
    k = std::max(0, std::min({k, 3, max_out - 1}));
    int toks[4] = {s.pending};
    std::copy(drafts, drafts + k, toks + 1);
    memcpy(b.mask.host(), masks, (size_t)(k + 1) * (VOCAB / 32) * 4);
    return verify(s, sp, toks, k, true, out, 0, nullptr, nullptr);
}
