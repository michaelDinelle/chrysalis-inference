#include "../../api.h"
#include "model.h"
#include <stdexcept>

static Engine *E = nullptr;

float load_progress();

extern "C" {

int chrysalis_init(const char *path, int max_seqs, int umax, long long reserve_mb) {
    @autoreleasepool {
        E = new Engine;
        E->umax = umax;
        E->load(path, max_seqs, (size_t)reserve_mb << 20);
        return E->n_pages * PAGE;
    }
}
void chrysalis_shutdown(int) {
    @autoreleasepool {
        if (!E) return;
        E->m.free_all();
        delete E;
        E = nullptr;
    }
}
void chrysalis_warmup() { @autoreleasepool { E->warmup(); } }
float chrysalis_load_progress() { return load_progress(); }
int chrysalis_seq_new() { return E->seq_new(); }
void chrysalis_seq_free(int id) { E->seq_free(id); }
int chrysalis_seq_len(int id) { return E->seqs.at(id).n; }
void chrysalis_seq_truncate(int id, int n) { @autoreleasepool { E->seq_truncate(id, n); E->m.sync(); } }
void chrysalis_seq_begin(int id, uint64_t seed, const Sampling *sp) {
    Seq &s = E->seqs.at(id);
    s.rng.seed(seed);
    s.seen.begin(s.ctx, sp ? *sp : Sampling{});
}
void chrysalis_seq_lookup(int id, const int *tokens, int n) { E->seqs.at(id).look.set(tokens, n); }
void chrysalis_seq_context(int id, const int *tokens, int n) { E->seqs.at(id).ctx.assign(tokens, tokens + n); }
void chrysalis_seq_breakers(int id, const int *flat, int n) { E->seqs.at(id).seen.set_breakers(flat, n); }
void chrysalis_seq_bias(int id, const int *ids, const float *vals, int n) { E->seqs.at(id).seen.set_bias(ids, vals, n); }
void chrysalis_seq_push(int id, const int *tokens, int n) {
    Seq &s = E->seqs.at(id);
    for (int i = 0; i < n; i++) s.seen.push(tokens[i], 0);
}
int chrysalis_free_tokens() { return E->free_page_count() * PAGE; }
int chrysalis_generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v, int *rows_i) {
    @autoreleasepool {
        try { E->generate_batch(n, ids, sps, out, max_out, counts, rows_v, rows_i); return 0; } catch (const std::exception &) { return -1; }
    }
}
void chrysalis_masked_top(const uint32_t *mask, float *v, int *ids, float *stats) {
    @autoreleasepool {
        memcpy(E->b.mask.h, mask, VOCAB / 32 * 4);
        MaskA ma{E->b.logits.g, E->b.mask.g, VOCAB};
        E->m.run(E->m.pipeline("mask_logits"), ma, (VOCAB + 255) / 256, 1, 1, 256);
        E->topk(E->b.logits, 1, VOCAB, VOCAB);
        E->logit_stats(E->b.logits, 1, VOCAB, VOCAB);
        E->m.sync();
        memcpy(v, E->b.topv.h, SAMPLE_N * 4);
        memcpy(ids, E->b.topi.h, SAMPLE_N * 4);
        memcpy(stats, E->b.tops.h, NSTAT * 4);
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
        try { return E->generate_masked(id, *sp, drafts, k, masks, out, max_out); } catch (const std::exception &) { return -1; }
    }
}
int chrysalis_snapshot(int id) { @autoreleasepool { return E->snapshot(id); } }
int chrysalis_restore(int id, int snap) { @autoreleasepool { try { E->restore(id, snap); return 0; } catch (const std::exception &) { return -1; } } }
int chrysalis_snap_free(int snap) { try { E->snap_free(snap); return 0; } catch (const std::exception &) { return -1; } }
int64_t chrysalis_snap_bytes(int snap) { try { return (int64_t)E->snap_bytes(snap); } catch (const std::exception &) { return -1; } }
int chrysalis_snap_export(int snap, void *dst) { @autoreleasepool { try { E->snap_export(snap, (uint8_t *)dst); return 0; } catch (const std::exception &) { return -1; } } }
int chrysalis_snap_import(int fd, int64_t offset, int64_t bytes) { @autoreleasepool { return E->snap_import(fd, offset, (size_t)bytes); } }
void chrysalis_sync() { @autoreleasepool { E->flush(); E->m.sync(); } }
int chrysalis_prefill_pos(int id, const int *tokens, const int *pos3, int n) {
    @autoreleasepool {
        try { E->prefill(id, tokens, pos3, n); return 0; }
        catch (const std::invalid_argument &) { return -2; }
        catch (const std::exception &) { return -1; }
    }
}
int chrysalis_prefill(int id, const int *tokens, int n) { return chrysalis_prefill_pos(id, tokens, nullptr, n); }
void chrysalis_vision_open(const char *path) { @autoreleasepool { E->vision.open(path, D, E->m.dev, E->m.lib); } }
void chrysalis_vision_release() { @autoreleasepool { E->vision.release(); } }
int chrysalis_image_encode(int slot, const float *patches, int gh, int gw) { @autoreleasepool { return E->image_encode(slot, patches, gh, gw); } }
int chrysalis_video_encode(int slot, const float *patches, int gh, int gw) { @autoreleasepool { return E->image_encode(slot, patches, gh, gw, 2); } }
void chrysalis_image_free(int slot) {
    if (slot >= 0 && slot < (int)E->images.size()) E->images[slot] = {};
}
int chrysalis_image_read(int slot, float *out) {
    if (slot < 0 || slot >= (int)E->images.size() || E->images[slot].rows.empty()) return -1;
    const Engine::Image &im = E->images[slot];
    for (size_t i = 0; i < im.rows.size(); i++) out[i] = (float)im.rows[i];
    return im.n;
}
int chrysalis_generate(int id, const Sampling *sp, int *out, int max_out, float *rows_v, int *rows_i) {
    @autoreleasepool {
        try { return E->generate(id, *sp, out, max_out, rows_v, rows_i); } catch (const std::exception &) { return -1; }
    }
}
int chrysalis_bench_fill(int id, int n) {
    @autoreleasepool {
        Seq &s = E->seqs.at(id);
        if (n <= s.n) return 0;
        E->flush();
        const int had = (int)s.pages.size();
        try { E->ensure_pages(s, n, s.n); } catch (const std::exception &) { return -1; }
        for (int i = had; i < (int)s.pages.size(); i++) E->fill0(E->page_mem[s.pages[i]], KV_PAGE);
        E->m.sync();
        s.pos += n - s.n;
        s.n = n;
        s.pending = -1;
        fprintf(stderr, "chrysalis: sequence filled to %d tokens; %.2f GB of Metal buffers\n", n, E->m.allocated / 1e9);
        return 0;
    }
}
void chrysalis_last_logits(float *out) { @autoreleasepool { E->m.sync(); memcpy(out, E->b.logits.h, (size_t)VOCAB * 4); } }
}
