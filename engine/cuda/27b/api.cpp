#include "../../api.h"
#include "model.h"
#include <climits>
#include <cstring>
#include <stdexcept>

static Engine *E = nullptr;

float load_progress();

extern "C" {

int chrysalis_init(const char *path, int max_seqs, int umax, long long reserve_mb) {
    E = new Engine;
    E->umax = umax;
    E->load(path, max_seqs, (size_t)reserve_mb << 20);
    return E->n_pages * PAGE;
}
void chrysalis_shutdown(int) {
    if (!E) return;
    (void)cudaDeviceSynchronize();
    (void)cudaDeviceReset();
    E = nullptr;
}
void chrysalis_warmup() { E->warmup(); }
float chrysalis_load_progress() { return load_progress(); }
int chrysalis_seq_new() { return E->seq_new(); }
void chrysalis_seq_free(int id) { E->seq_free(id); }
int chrysalis_seq_len(int id) { return E->seqs.at(id).n; }
void chrysalis_seq_truncate(int id, int n) { E->seq_truncate(id, n); }
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
int chrysalis_free_tokens() { return ((int)E->free_pages.size() + E->lent_pages()) * PAGE; }
int chrysalis_generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v, int *rows_i) {
    try { E->generate_batch(n, ids, sps, out, max_out, counts, rows_v, rows_i); return 0; } catch (const std::exception &) { return -1; }
}
void chrysalis_masked_top(const uint32_t *mask, float *v, int *ids, float *stats) {
    CK(cudaMemcpyAsync(E->mask_rows(), mask, VOCAB / 32 * 4, cudaMemcpyHostToDevice, E->st));
    mask_logits(E->b.logits, E->d_mask, VOCAB, 1, E->st);
    topk(E->b.logits, 1, VOCAB, VOCAB, E->b.tsv, E->b.tsi, E->b.topv, E->b.topi, E->st);
    logit_stats(E->b.logits, 1, VOCAB, VOCAB, E->b.tops, E->st);
    CK(cudaMemcpyAsync(v, E->b.topv, SAMPLE_N * 4, cudaMemcpyDeviceToHost, E->st));
    CK(cudaMemcpyAsync(ids, E->b.topi, SAMPLE_N * 4, cudaMemcpyDeviceToHost, E->st));
    CK(cudaMemcpyAsync(stats, E->b.tops, NSTAT * 4, cudaMemcpyDeviceToHost, E->st));
    CK(cudaStreamSynchronize(E->st));
}
int chrysalis_draw(int id, const float *v, const int *ids, const float *stats, const Sampling *sp) {
    Seq &s = E->seqs.at(id);
    const int t = sample(s.seen, v, ids, stats, *sp, s.rng);
    if (sp->lookup > 0) s.look.push(t);
    return t;
}
void chrysalis_seq_pending(int id, int tok) { E->seqs.at(id).pending = tok; }
int chrysalis_draft(int id, const Sampling *sp, const int *fixed, int nfixed, int max_k, int *out) {
    try { return E->draft(id, *sp, fixed, nfixed, max_k, out); } catch (const std::exception &) { return -1; }
}
int chrysalis_generate_masked(int id, const Sampling *sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out) {
    try { return E->generate_masked(id, *sp, drafts, k, masks, out, max_out); } catch (const std::exception &) { return -1; }
}
int chrysalis_snapshot(int id) { return E->snapshot(id); }
int chrysalis_restore(int id, int snap) { try { E->restore(id, snap); return 0; } catch (const std::exception &) { return -1; } }
int chrysalis_snap_free(int snap) { try { E->snap_free(snap); return 0; } catch (const std::exception &) { return -1; } }
int64_t chrysalis_snap_bytes(int snap) { try { return (int64_t)E->snap_bytes(snap); } catch (const std::exception &) { return -1; } }
int chrysalis_snap_export(int snap, void *dst) { try { E->snap_export(snap, (uint8_t *)dst); return 0; } catch (const std::exception &) { return -1; } }
int chrysalis_snap_import(int fd, int64_t offset, int64_t bytes) { return E->snap_import(fd, offset, (size_t)bytes); }
void chrysalis_sync() { CK(cudaStreamSynchronize(E->st)); }
double chrysalis_tune(const char *key, double value) {
    if (strcmp(key, "lend") || E->first_spill >= NL) return -1;
    const double old = E->n_lent;
    E->lend_on = value != 0;
    if (E->lend_on) E->lend();
    else E->reclaim(INT32_MAX);
    return old;
}
int chrysalis_prefill_pos(int id, const int *tokens, const int *pos3, int n) {
    try { E->prefill(id, tokens, pos3, n); return 0; }
    catch (const std::invalid_argument &) { return -2; }
    catch (const std::exception &) { return -1; }
}
int chrysalis_prefill(int id, const int *tokens, int n) { return chrysalis_prefill_pos(id, tokens, nullptr, n); }
void chrysalis_vision_open(const char *path) { E->vision.open(path, D); }
void chrysalis_vision_release() { E->vision.release(); }
int chrysalis_image_encode(int slot, const float *patches, int gh, int gw) { return E->image_encode(slot, patches, gh, gw); }
int chrysalis_video_encode(int slot, const float *patches, int gh, int gw) { return E->image_encode(slot, patches, gh, gw, 2); }
void chrysalis_image_free(int slot) {
    if (slot < 0 || slot >= (int)E->images.size() || !E->images[slot].rows) return;
    CK(cudaFreeHost(E->images[slot].rows));
    E->images[slot] = {};
}
int chrysalis_image_read(int slot, float *out) {
    if (slot < 0 || slot >= (int)E->images.size() || !E->images[slot].rows) return -1;
    const Engine::Image &im = E->images[slot];
    for (size_t i = 0; i < (size_t)im.n * D; i++) out[i] = __half2float(im.rows[i]);
    return im.n;
}
int chrysalis_generate(int id, const Sampling *sp, int *out, int max_out, float *rows_v, int *rows_i) {
    try { return E->generate(id, *sp, out, max_out, rows_v, rows_i); } catch (const std::exception &) { return -1; }
}
void chrysalis_last_logits(float *out) { CK(cudaMemcpy(out, E->b.logits, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost)); }
}
