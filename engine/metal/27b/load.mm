#include "model.h"
#include "gguf.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::atomic<size_t> load_done{0}, load_total{0};
static std::atomic<int> load_phase{0};
float load_progress() {
    if (load_phase >= 3) return 1.0f;
    if (load_phase == 2) return 0.97f;
    return load_total ? 0.95f * (float)load_done / (float)load_total : 0.0f;
}

static inline uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

static inline int q4_pos(int ui, int j) { return ((j & 7) >> 1) * 16 + ui * 4 + (j >> 3) * 2 + (j & 1); }
static inline int qh_pos(int ui, int j) { return (j >> 1) * 8 + ui * 2 + (j & 1); }
static inline int q8_pos(int ui, int j) { return ((j & 7) >> 1) * 32 + ui * 8 + (j >> 3) * 2 + (j & 1); }

void relayout(const QTensor &t, const uint8_t *raw, size_t b0, size_t b1) {
    if (t.type == Q8_0) {
        for (size_t b = b0; b < b1; b++) {
            const uint8_t *s = raw + b * 34;
            memcpy(t.d.h + b * 2, s, 2);
            uint8_t *grp = t.q.h + (b / 4) * 128;
            for (int j = 0; j < 32; j++) grp[q8_pos(b % 4, j)] = s[2 + j];
        }
    } else if (t.type == IQ4_XS) {
        for (size_t b = b0; b < b1; b++) {
            const uint8_t *s = raw + b * 136;
            memcpy(t.d.h + b * 2, s, 2);
            memcpy(t.qh.h + b * 2, s + 2, 2);
            memcpy(t.sc.h + b * 4, s + 4, 4);
            for (int ib = 0; ib < 8; ib++) {
                uint8_t *grp = t.q.h + (b * 8 + ib) / 4 * 64;
                for (int j = 0; j < 16; j++) grp[q4_pos(ib % 4, j)] = s[8 + 16 * ib + j];
            }
        }
    } else if (t.type == Q6_K) {
        for (size_t b = b0; b < b1; b++) {
            const uint8_t *s = raw + b * 210, *ql = s, *qh = s + 128;
            uint8_t u[256];
            for (int n = 0; n < 2; n++)
                for (int l = 0; l < 32; l++) {
                    const uint8_t *L = ql + 64 * n, *H = qh + 32 * n;
                    u[128 * n + l] = (L[l] & 15) | ((H[l] & 3) << 4);
                    u[128 * n + l + 32] = (L[l + 32] & 15) | (((H[l] >> 2) & 3) << 4);
                    u[128 * n + l + 64] = (L[l] >> 4) | (((H[l] >> 4) & 3) << 4);
                    u[128 * n + l + 96] = (L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4);
                }
            for (int un = 0; un < 8; un++) {
                const uint8_t *v = u + 32 * un;
                const size_t gu = b * 8 + un;
                uint8_t *lo = t.q.h + gu / 4 * 64, *hi = t.qh.h + gu / 4 * 32;
                for (int j = 0; j < 16; j++) lo[q4_pos(gu % 4, j)] = (v[j] & 15) | ((v[j + 16] & 15) << 4);
                for (int j = 0; j < 8; j++)
                    hi[qh_pos(gu % 4, j)] = (v[j] >> 4) | ((v[j + 8] >> 4) << 2) | ((v[j + 16] >> 4) << 4) | ((v[j + 24] >> 4) << 6);
            }
            memcpy(t.sc.h + b * 16, s + 192, 16);
            memcpy(t.d.h + b * 2, s + 208, 2);
        }
    } else {
        FAIL("relayout: type %d", t.type);
    }
}

void Engine::load(const std::string &path, int max_seqs, size_t reserve_bytes) {
    load_done = 0; load_total = 0; load_phase = 0;
    const double t0 = now();
    const char *lib = getenv("CHRYSALIS_METALLIB");
    std::string libpath = lib ? lib : "";
    if (libpath.empty()) {
        Dl_info di;
        if (!dladdr((void *)&load_progress, &di)) FAIL("cannot find the library's directory");
        libpath = std::string(di.dli_fname);
        libpath = libpath.substr(0, libpath.rfind('/') + 1) + "chrysalis.metallib";
    }
    m.init(libpath);
    GGUF g;
    if (!g.open(path.c_str())) FAIL("cannot open %s", path.c_str());
    if (g.ints["qwen35.block_count"] != NL + 1 || g.ints["qwen35.embedding_length"] != D)
        FAIL("%s is not a Qwen3.8-27B GGUF with an MTP layer", path.c_str());

    struct Job { const GGUFTensor *t; QTensor *q = nullptr; DP<float> *vec = nullptr; int kind = 0, gl = 0; Layer *w = nullptr; };
    enum { QUANT, VEC, ALPHA, BETA, GDN_A, GDN_DT };
    std::vector<Job> jobs;
    auto get = [&](const std::string &name) {
        const GGUFTensor *t = g.get(name);
        if (!t) FAIL("missing tensor %s", name.c_str());
        if (t->data + t->nbytes > g.base + g.size) FAIL("%s is cut short (%s)", path.c_str(), name.c_str());
        return t;
    };
    auto quant = [&](const std::string &name, QTensor &q) { jobs.push_back({get(name), &q}); };
    auto vec = [&](const std::string &name, DP<float> &v) { Job j{get(name)}; j.vec = &v; j.kind = VEC; jobs.push_back(j); };
    quant("token_embd.weight", embd);
    quant("output.weight", out);
    vec("output_norm.weight", out_norm);
    for (int l = 0, gl = 0; l <= NL; l++) {
        Layer &w = l < NL ? L[l] : M;
        const std::string p = "blk." + std::to_string(l) + ".";
        vec(p + "attn_norm.weight", w.attn_norm);
        vec(p + "post_attention_norm.weight", w.post_norm);
        quant(p + "ffn_gate.weight", w.ffn_gate); quant(p + "ffn_up.weight", w.ffn_up); quant(p + "ffn_down.weight", w.ffn_down);
        if (l == NL || is_attn_layer(l)) {
            quant(p + "attn_q.weight", w.q); quant(p + "attn_k.weight", w.k);
            quant(p + "attn_v.weight", w.v); quant(p + "attn_output.weight", w.o);
            vec(p + "attn_q_norm.weight", w.q_norm); vec(p + "attn_k_norm.weight", w.k_norm);
        } else {
            quant(p + "attn_qkv.weight", w.qkv); quant(p + "attn_gate.weight", w.z); quant(p + "ssm_out.weight", w.ssm_out);
            vec(p + "ssm_conv1d.weight", w.conv); vec(p + "ssm_norm.weight", w.ssm_norm);
            for (auto [name, kind] : {std::pair{"ssm_alpha.weight", (int)ALPHA}, {"ssm_beta.weight", (int)BETA}, {"ssm_a", (int)GDN_A},
                                      {"ssm_dt.bias", (int)GDN_DT}}) {
                Job j{get(p + name)};
                if (j.t->type != F32) FAIL("unexpected type of %s", (p + name).c_str());
                j.kind = kind; j.gl = gl; j.w = &w;
                jobs.push_back(j);
            }
            gl++;
        }
    }
    const std::string p = "blk." + std::to_string(NL) + ".";
    quant(p + "nextn.eh_proj.weight", eh_proj);
    vec(p + "nextn.enorm.weight", enorm); vec(p + "nextn.hnorm.weight", hnorm);
    vec(p + "nextn.shared_head_norm.weight", head_norm);
    std::sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) { return a.t->data < b.t->data; });

    auto carve = [&](Arena &a, Job &j) {
        const int rows = (int)j.t->ne[1], cols = (int)j.t->ne[0];
        if (j.kind == QUANT) {
            if (j.t->type != Q8_0 && j.t->type != Q6_K && j.t->type != IQ4_XS) FAIL("unsupported weight type %d", j.t->type);
            *j.q = make_qtensor(a, j.t->type, rows, cols);
        } else if (j.kind == VEC) {
            *j.vec = a.get<float>(j.t->nbytes / 4);
        } else if (j.kind == ALPHA) {
            j.w->ab = a.get<uint16_t>((size_t)2 * GV * D);
        }
    };
    Arena dry; dry.cap = SIZE_MAX;
    for (Job &j : jobs) carve(dry, j);
    dry.used += 2 * 48 * GV * 4 + 4096;
    Arena wa;
    wa.base = m.alloc(dry.used, "weights");
    wa.cap = dry.used;
    for (Job &j : jobs) carve(wa, j);
    gdn_A = wa.get<float>(48 * GV); gdn_dt = wa.get<float>(48 * GV);
    weight_bytes = wa.used;
    size_t total = 0;
    for (const Job &j : jobs) total += j.t->nbytes;
    load_total = total;

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) FAIL("cannot open %s", path.c_str());
    fcntl(fd, F_NOCACHE, 1);
    constexpr size_t PIECE = 64 << 20;
    uint8_t *scratch = (uint8_t *)aligned_alloc(16384, PIECE + 16384);
    for (const Job &j : jobs) {
        const size_t off = j.t->data - g.base;
        const size_t bb = j.kind == QUANT ? (j.q->type == Q8_0 ? 34 * 4 : j.q->type == Q6_K ? 210 : 136) : j.t->nbytes;
        const size_t step = j.kind == QUANT ? PIECE / bb * bb : PIECE;
        for (size_t p0 = 0; p0 < j.t->nbytes; p0 += step) {
            const size_t len = std::min(step, j.t->nbytes - p0);
            for (size_t got = 0; got < len;) {
                const ssize_t r = pread(fd, scratch + got, len - got, off + p0 + got);
                if (r <= 0) FAIL("cannot read %s", path.c_str());
                got += r;
            }
            const uint8_t *src = scratch;
            switch (j.kind) {
                case QUANT: {
                    const size_t ub = j.q->type == Q8_0 ? 34 : bb, b0 = p0 / ub, b1 = (p0 + len) / ub;
                    const size_t per = 1024;
                    const QTensor q = *j.q;
                    const uint8_t *base = src - b0 * ub;
                    dispatch_apply((b1 - b0 + per - 1) / per, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                        relayout(q, base, b0 + i * per, std::min(b1, b0 + (i + 1) * per));
                    });
                    break;
                }
                case VEC: memcpy((uint8_t *)j.vec->h + p0, src, len); break;
                case ALPHA: case BETA: {
                    const uint32_t *f = (const uint32_t *)src;
                    uint16_t *o = j.w->ab.h + (j.kind == BETA ? (size_t)GV * D : 0) + p0 / 4;
                    for (size_t i = 0; i < len / 4; i++) {
                        if (f[i] & 0xffff) FAIL("ssm_alpha / ssm_beta are not bf16 values: the engine keeps them as bf16");
                        o[i] = (uint16_t)(f[i] >> 16);
                    }
                    break;
                }
                case GDN_A: case GDN_DT: memcpy((uint8_t *)((j.kind == GDN_A ? gdn_A : gdn_dt).h + (size_t)j.gl * GV) + p0, src, len); break;
            }
            load_done += len;
        }
    }
    free(scratch);
    ::close(fd);
    const double t_w = now();
    alloc_buffers(max_seqs);

    const size_t want = (size_t)(262144 / PAGE + 16) * KV_PAGE;
    const size_t avail = m.max_ws > m.allocated + reserve_bytes ? m.max_ws - m.allocated - reserve_bytes : 0;
    const size_t bytes = std::min(want, avail) / KV_PAGE * KV_PAGE;
    if (bytes < KV_PAGE * 64) FAIL("no room for a KV cache (%.2f GB of %.2f GB in use)", m.allocated / 1e9, m.max_ws / 1e9);
    init_kv((int)(bytes / KV_PAGE));
    m.make_resident();

    fprintf(stderr, "chrysalis: weights %.2f GB, KV pool up to %d pages = %d tokens (%.2f GB, allocated as used), %.2f GB of the GPU's "
            "%.2f GB working set with it\n", weight_bytes / 1e9, n_pages, n_pages * PAGE, bytes / 1e9, (m.allocated + bytes) / 1e9, m.max_ws / 1e9);
    fprintf(stderr, "chrysalis: loaded in %.2f s (weights %.2f s)\n", now() - t0, t_w - t0);
}

void Engine::warmup() {
    load_phase = 2;
    struct Done { ~Done() { load_phase = 3; } } done_;
    const double t0 = now();
    struct K { const char *name; int c; };
    std::vector<K> ks = {{"rmsnorm", -1}, {"copy16", -1}, {"fill16", -1}, {"topk1", -1}, {"topk2", -1}, {"argmax", -1},
                         {"embed", -1}, {"mask_logits", -1}, {"swiglu_h", -1}, {"attn_prep", -1}, {"attn_vec", -1}, {"kv_unpack", -1}, {"attn_smax", -1}, {"attn_pf", -1},
                         {"attn_combine", -1}, {"gdn_conv", -1}, {"gdn_conv_state", -1}, {"gdn_conv_commit", -1}, {"gdn_scan", -1},
                         {"gdn_out", -1}, {"gdn_ab", -1}, {"gdn_step", -1}, {"gdn_step1", -1}, {"gemv1_23_1_0", -1}, {"gemv1_23_1_2", -1},
                         {"gemv1_23_4_1", -1}, {"gemv1_23_8_0", -1}, {"gemv1_14_2_0", -1}, {"gemv1_14_2_1", -1}, {"gemv1_8_2_0", -1},
                         {"gemv1_8_2_1", -1}, {"gemv1_8_2_2", -1}, {"gemv_mma_23_4_1_1", -1}, {"gemv_mma_8_4_1_1", -1}};
    for (const char *n : {"gemv_mma_23_4_0_%d", "gemv_mma_14_4_0_%d", "gemv_mma_8_4_0_%d"})
        for (int t : {1, 2}) {
            static char names[9][32];
            static int k = 0;
            snprintf(names[k % 9], 32, n, t);
            ks.push_back({names[k++ % 9], -1});
        }
    for (int t : {(int)IQ4_XS, (int)Q6_K, (int)Q8_0}) { ks.push_back({"gemm_q", t}); ks.push_back({"dequant", t}); }
    Mtl *mp = &m;
    dispatch_apply(ks.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        mp->pipeline(ks[i].name, ks[i].c >= 0 ? &ks[i].c : nullptr, ks[i].c >= 0 ? 1 : 0);
    });
    m.save_archive();
    const double t1 = now();
    int id = seq_new();
    const bool borrowed = id < 0;
    if (borrowed) id = 0;
    const std::vector<int> toks(9, 0);
    prefill(id, toks.data(), nullptr, 1);
    prefill(id, toks.data(), nullptr, 9);
    if (borrowed) seq_truncate(id, 0);
    else seq_free(id);
    fprintf(stderr, "chrysalis: warmed up in %.2f s (pipelines %.2f s)\n", now() - t0, t1 - t0);
}

void Engine::alloc_buffers(int max_seqs) {
    umax = (umax + 63) / 64 * 64;
    const size_t U = umax;
    qrows_max = (int)((U * 6 + 63) / 64 * 64);
    const size_t part = std::max((size_t)qrows_max, (size_t)MAX_SPLITS * 64) * NKV;
    Arena a;
    size_t need = 0;
    {
        Arena dry; dry.cap = SIZE_MAX;
        auto carve = [&](Arena &r) {
            b.tokens = r.get<int>(U + 64);
            b.x = r.get<float>(U * D); b.hfin = r.get<float>(U * D); b.xn = r.get<float>(8 * 2 * D);
            b.mtp_h = r.get<float>(U * D);
            b.proj = r.get<float>(U * 16384); b.k = r.get<float>(U * NKV * HD); b.v = r.get<float>(U * NKV * HD);
            b.qn = r.get<float>(U * GK * GD); b.kn = r.get<float>(U * GK * GD);
            b.vc = r.get<float>(U * GV * GD); b.o = r.get<float>(U * GV * GD); b.ab = r.get<float>(U * 2 * GV);
            b.hh = r.get<half_t>(U * 2 * D); b.deq = r.get<half_t>((size_t)FF * D);
            b.kc = r.get<half_t>((size_t)(262144 + 1024) * HD); b.vt = r.get<half_t>((size_t)(262144 + 1024) * HD);
            b.st = r.get<float>(ATT_S); b.pt = r.get<half_t>(ATT_S);
            b.Qg = r.get<half_t>((size_t)NKV * (qrows_max + 64) * HD); b.mix = r.get<half_t>(U * NH * HD);
            b.ffn_g = r.get<half_t>(U * FF); b.ffn_u = r.get<half_t>(U * FF);
            b.logits = r.get<float>(8 * (size_t)VOCAB);
            b.part_o = r.get<float>(part * HD); b.part_ml = r.get<float>(part * 2);
            b.topv = r.get<float>(8 * TOPK); b.topi = r.get<int>(8 * TOPK); b.tops = r.get<float>(8 * NSTAT);
            b.tsv = r.get<float>(8 * 64 * TOPK); b.tsi = r.get<int>(8 * 64 * TOPK);
            b.v_raw = r.get<float>(48 * 8 * QKV); b.v_kn = r.get<float>(48 * 8 * GK * GD);
            b.v_vc = r.get<float>(48 * 8 * GV * GD); b.v_ab = r.get<float>(48 * 8 * 2 * GV);
            b.mtp_x = r.get<float>(8 * D); b.mtp_xn = r.get<float>(8 * 2 * D); b.draft = r.get<int>(16);
            b.rpos = r.get<int>(U * 3);
            b.mask = r.get<uint32_t>((size_t)8 * (VOCAB / 32));
        };
        carve(dry);
        need = dry.used;
        a.base = m.alloc(need, "activation buffers");
        a.cap = need;
        carve(a);
    }
    seqs.resize(max_seqs);
    DP<uint8_t> st = m.alloc((SSM_BYTES + CONV_BYTES + D * 4 + 262144 / PAGE * 16) * max_seqs, "sequence states");
    for (int i = 0; i < max_seqs; i++) {
        DP<uint8_t> p = st + (SSM_BYTES + CONV_BYTES + D * 4 + 262144 / PAGE * 16) * i;
        seqs[i].ssm = p.as<float>();
        seqs[i].conv = (p + SSM_BYTES).as<float>();
        seqs[i].h_last = (p + SSM_BYTES + CONV_BYTES).as<float>();
        seqs[i].d_pages = (p + SSM_BYTES + CONV_BYTES + D * 4).as<int64_t>();
    }
}
