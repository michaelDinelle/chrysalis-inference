#include "../../api.h"
#include "engine.h"
#include <cmath>
#include <pthread/qos.h>
#include <stdexcept>
#include <unistd.h>

static void interactive() {
    static thread_local bool done = false;
    if (!done) { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); done = true; }
}

static Engine *E = nullptr;
extern std::atomic<float> g_progress;

struct SnapHeader { char magic[8]; uint64_t kv_page, state; int32_t n, hist[2], n_top, n_pages; };
static const char SNAP_MAGIC[8] = {'K', 'I', 'L', 'N', 'M', 'F', 'N', '1'};

extern "C" {

int chrysalis_init(const char *path, int max_seqs, int chunk, long long reserve_mb) {
    @autoreleasepool {
        (void)reserve_mb;
        g_progress = 0;
        E = new Engine;
        if (chunk > 0) E->umax = chunk;
        E->load(path, max_seqs);
        return E->n_pages * PAGE;
    }
}
void chrysalis_warmup() {}
void chrysalis_reserve_snapshots(int) {}
float chrysalis_load_progress() { return g_progress; }
void chrysalis_shutdown(int) { @autoreleasepool { if (E) { E->shutdown(); delete E; E = nullptr; } } }
int chrysalis_seq_new() { return E->seq_new(); }
void chrysalis_seq_free(int id) { E->seq_free(id); }
int chrysalis_seq_len(int id) { return E->seqs.at(id).n; }
void chrysalis_seq_truncate(int id, int n) {
    @autoreleasepool {
        Seq &s = E->seqs.at(id);
        if (n == s.n) { s.pending = -1; return; }
        if (n != 0) FAIL("truncate to %d: the recurrent state can only be rolled back via snapshots", n);
        E->seq_reset(s);
    }
}
void chrysalis_seq_begin(int id, uint64_t seed, const Sampling *sp) {
    Seq &s = E->seqs.at(id);
    s.rng.seed(seed);
    s.seen.begin(s.ctx, sp ? *sp : Sampling{});
}
void chrysalis_seq_context(int id, const int *tokens, int n) { E->seqs.at(id).ctx.assign(tokens, tokens + n); }
void chrysalis_seq_breakers(int id, const int *flat, int n) { E->seqs.at(id).seen.set_breakers(flat, n); }
void chrysalis_seq_bias(int id, const int *ids, const float *vals, int n) { E->seqs.at(id).seen.set_bias(ids, vals, n); }
void chrysalis_seq_push(int id, const int *tokens, int n) {
    Seq &s = E->seqs.at(id);
    for (int i = 0; i < n; i++) s.seen.push(tokens[i], 0);
}
int chrysalis_free_tokens() { return (int)E->free_pages.size() * PAGE; }
void chrysalis_seq_lookup(int id, const int *tokens, int n) { E->seqs.at(id).look.set(tokens, n); }
void chrysalis_masked_top(const uint32_t *mask, float *v, int *ids, float *stats) {
    @autoreleasepool {
        Engine &e = *E;
        memcpy(e.b.mask.host(), mask, VOCAB / 32 * 4);
        id<MTLCommandBuffer> cb = [e.m.q commandBufferWithUnretainedReferences];
        Enc en; en.begin(e.m, cb);
        const uint32_t n = VOCAB;
        en.k("mask_logits").b(0, e.b.logits).b(1, e.b.mask).c(2, n).grid((VOCAB + 255) / 256, 1, 1, 256);
        en.barrier();
        e.topk(en, 1);
        en.end();
        [cb commit]; [cb waitUntilCompleted];
        memcpy(v, e.b.topv.host(), SAMPLE_N * 4);
        memcpy(ids, e.b.topi.host(), SAMPLE_N * 4);
        memcpy(stats, e.b.tops.host(), NSTAT * 4);
    }
}
int chrysalis_draw(int id, const float *v, const int *ids, const float *stats, const Sampling *sp) {
    Seq &s = E->seqs.at(id);
    const int t = sample(s.seen, v, ids, stats, *sp, s.rng);
    if (sp->lookup > 0) s.look.push(t);
    return t;
}
void chrysalis_seq_pending(int id, int tok) { E->seqs.at(id).pending = tok; }
int chrysalis_draft(int id, const Sampling *sp, const int *fixed, int nfixed, int max_k, int *out) {
    @autoreleasepool {
        try { return E->draft(id, *sp, fixed, nfixed, max_k, out); } catch (const std::exception &) { return -1; }
    }
}
int chrysalis_generate_masked(int id, const Sampling *sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out) {
    @autoreleasepool {
        interactive();
        try { return E->generate_masked(id, *sp, drafts, k, masks, out, max_out); } catch (const std::exception &) { return -1; }
    }
}
int chrysalis_generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v, int *rows_i) {
    @autoreleasepool {
        interactive();
        int need = 0;
        for (int i = 0; i < n; i++) {
            const Seq &s = E->seqs.at(ids[i]);
            const int last = (s.n + 2 + PAGE - 1) / PAGE;
            need += std::max(0, last - (int)s.pages.size());
        }
        if (need > (int)E->free_pages.size()) return -1;
        try {
            for (int i = 0; i < n; i++)
                counts[i] = E->generate(ids[i], sps[i], out + (size_t)i * max_out, max_out, rows_v ? rows_v + (size_t)i * max_out * SAMPLE_N : nullptr,
                                        rows_i ? rows_i + (size_t)i * max_out * SAMPLE_N : nullptr);
        } catch (const std::exception &) { return -1; }
        return 0;
    }
}
int chrysalis_snapshot(int id) { @autoreleasepool { return E->snapshot(id); } }
int chrysalis_restore(int id, int snap) { @autoreleasepool { try { E->restore(id, snap); return 0; } catch (const std::exception &) { return -1; } } }
int chrysalis_snap_free(int snap) { try { E->snap_free(snap); return 0; } catch (const std::exception &) { return -1; } }
int64_t chrysalis_snap_bytes(int snap) {
    try {
        const Engine::Snap &sn = E->snaps.at(snap);
        return (int64_t)(sizeof(SnapHeader) + sn.top_v.size() * 8 + state_bytes() + sn.pages.size() * KV_PAGE);
    } catch (const std::exception &) { return -1; }
}
int chrysalis_snap_export(int snap, void *dst) {
    @autoreleasepool {
        E->settle();
        try {
            const Engine::Snap &sn = E->snaps.at(snap);
            if (!sn.used) return -1;
            SnapHeader h;
            memcpy(h.magic, SNAP_MAGIC, 8);
            h.kv_page = KV_PAGE; h.state = state_bytes(); h.n = sn.n; h.hist[0] = sn.hist[0]; h.hist[1] = sn.hist[1];
            h.n_top = (int32_t)sn.top_v.size(); h.n_pages = (int32_t)sn.pages.size();
            uint8_t *p = (uint8_t *)dst;
            memcpy(p, &h, sizeof h); p += sizeof h;
            memcpy(p, sn.top_v.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
            memcpy(p, sn.top_i.data(), (size_t)h.n_top * 4); p += (size_t)h.n_top * 4;
            memcpy(p, sn.state.data(), h.state); p += h.state;
            for (int pg : sn.pages) { memcpy(p, E->kv.host<uint8_t>() + (size_t)pg * KV_PAGE, KV_PAGE); p += KV_PAGE; }
            return 0;
        } catch (const std::exception &) { return -1; }
    }
}
int chrysalis_snap_import(int64_t file, int64_t offset, int64_t bytes) {
    @autoreleasepool {
        const int fd = (int)file;
        E->settle();
        auto get = [&](void *dst, size_t n) {
            for (size_t got = 0; got < n;) {
                const ssize_t r = pread(fd, (uint8_t *)dst + got, n - got, offset + (int64_t)got);
                if (r <= 0) return false;
                got += (size_t)r;
            }
            offset += (int64_t)n;
            return true;
        };
        SnapHeader h;
        if (bytes < (int64_t)sizeof h || !get(&h, sizeof h)) return -1;
        if (memcmp(h.magic, SNAP_MAGIC, 8) || h.kv_page != KV_PAGE || h.state != state_bytes() || h.n <= 0 || h.n_top < 0 ||
            h.n_pages != (h.n + PAGE - 1) / PAGE || (size_t)bytes != sizeof h + (size_t)h.n_top * 8 + h.state + (size_t)h.n_pages * KV_PAGE)
            return -1;
        if ((int)E->free_pages.size() < h.n_pages) return -2;
        size_t si = 0;
        while (si < E->snaps.size() && E->snaps[si].used) si++;
        if (si == E->snaps.size()) E->snaps.emplace_back();
        Engine::Snap &sn = E->snaps[si];
        sn.top_v.resize(h.n_top);
        sn.top_i.resize(h.n_top);
        sn.state.resize(h.state);
        bool ok = get(sn.top_v.data(), (size_t)h.n_top * 4) && get(sn.top_i.data(), (size_t)h.n_top * 4) && get(sn.state.data(), h.state);
        sn.pages.clear();
        for (int i = 0; ok && i < h.n_pages; i++) {
            sn.pages.push_back(E->page_alloc());
            ok = get(E->kv.host<uint8_t>() + (size_t)sn.pages.back() * KV_PAGE, KV_PAGE);
        }
        if (!ok) {
            for (int p : sn.pages) E->page_release(p);
            sn.pages.clear();
            return -1;
        }
        sn.n = h.n; sn.hist[0] = h.hist[0]; sn.hist[1] = h.hist[1];
        sn.used = true;
        return (int)si;
    }
}
void chrysalis_sync() {}
int chrysalis_prefill_pos(int id, const int *tokens, const int *pos3, int n) {
    @autoreleasepool {
        for (int i = 0; i < n; i++) if (tokens[i] >= IMG_BASE) return -2;
        (void)pos3;
        interactive();
        try { E->prefill(id, tokens, n); return 0; }
        catch (const std::invalid_argument &) { return -2; }
        catch (const std::exception &) { return -1; }
    }
}
int chrysalis_prefill(int id, const int *tokens, int n) { return chrysalis_prefill_pos(id, tokens, nullptr, n); }
int chrysalis_generate(int id, const Sampling *sp, int *out, int max_out, float *rows_v, int *rows_i) {
    @autoreleasepool {
        interactive();
        try { return E->generate(id, *sp, out, max_out, rows_v, rows_i); } catch (const std::exception &) { return -1; }
    }
}
double chrysalis_tune(const char *key, double value) {
    const std::string k = key;
    double old = NAN;
    if (k == "experts_half_life") { old = std::log(0.5) / std::log(E->xc.decay); if (value > 0) E->xc.decay = std::pow(0.5, 1.0 / value); }
    return old;
}
void chrysalis_expert_stats(int64_t *out) {
    const Experts &x = E->xc;
    int placed = 0;
    for (int s = 0; s < x.n_slots; s++) placed += x.owner[s] >= 0;
    out[0] = placed; out[1] = x.n_slots; out[2] = (int64_t)x.picks; out[3] = (int64_t)x.hits;
    out[4] = x.read_s > 0 ? (int64_t)(x.read_bytes / x.read_s / 1e6) : 0; out[5] = (int64_t)x.staged; out[6] = (int64_t)x.reads; out[7] = 0;
    if (x.ahead_reads) fprintf(stderr, "chrysalis-fn: %llu experts read ahead, %llu of them picked\n", (unsigned long long)x.ahead_reads, (unsigned long long)x.ahead_used);
    if (x.pred_misses) fprintf(stderr, "chrysalis-fn: %llu misses, %.1f%% predicted by the layer before\n", (unsigned long long)x.pred_misses, 100.0 * x.pred_caught / x.pred_misses);
}
void chrysalis_route_counts(int64_t *out) { if (out) memset(out, 0, (size_t)NL * NE * 8); }
void chrysalis_last_logits(float *out) { memcpy(out, E->b.logits.host<float>() + (size_t)E->last_row * VOCAB, (size_t)VOCAB * 4); }
}
