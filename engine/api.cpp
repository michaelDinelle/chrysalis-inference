// C API used by the Python server through ctypes.
#include "model.h"
#include <stdexcept>

static Engine *E = nullptr;

float load_progress();                          // load.cpp (C++ linkage, so declared outside extern "C")

extern "C" {

// Loads the model; returns KV capacity in tokens.
int kiln_init(const char *path, int max_seqs, int umax, long long reserve_mb) {
    E = new Engine;
    E->umax = umax;
    E->load(path, max_seqs, (size_t)reserve_mb << 20);
    return E->n_pages * PAGE;
}
// Frees the GPU for another engine in this process (model switch): a device reset releases every allocation this
// engine made (weights, KV pool, pinned buffers); the next kiln_init loads from scratch. keep_host is ignored here.
void kiln_shutdown(int) {
    if (!E) return;
    (void)hipDeviceSynchronize();
    (void)hipDeviceReset();
    E = nullptr;                                // its memory went with the reset; the host-side object is left behind
}
// Loads code objects and pins snapshot buffers; call once after kiln_init, before the first request.
void kiln_warmup() { E->warmup(); }
// 0..1 while kiln_init / kiln_warmup run (polled from another thread for a progress bar)
float kiln_load_progress() { return load_progress(); }
int kiln_seq_new() { return E->seq_new(); }
void kiln_seq_free(int id) { E->seq_free(id); }
int kiln_seq_len(int id) { return E->seqs.at(id).n; }
void kiln_seq_truncate(int id, int n) { E->seq_truncate(id, n); }
// Starts a reply: the same seed after the same prompt draws the same tokens, and the presence penalty counts
// this reply's tokens only (not those of earlier replies the sequence still holds).
void kiln_seq_begin(int id, uint64_t seed) {
    Seq &s = E->seqs.at(id);
    s.rng.seed(seed);
    s.seen.clear();
}
// The conversation so far, where the next reply's drafts may be copied from (prompt lookup, Sampling::lookup).
void kiln_seq_lookup(int id, const int *tokens, int n) { E->seqs.at(id).look.set(tokens, n); }
// The prompt's tail that DRY looks back over (with the reply), for the next reply of this sequence.
void kiln_seq_context(int id, const int *tokens, int n) { E->seqs.at(id).ctx.assign(tokens, tokens + n); }
// DRY's sequence breakers: token ids that end a repeated run.
void kiln_dry_breakers(const int *ids, int n) {
    dry_breakers.assign(VOCAB, 0);
    for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < VOCAB) dry_breakers[ids[i]] = 1;
}
int kiln_free_tokens() { return (int)E->free_pages.size() * PAGE; }
// One step for several sequences in one pass (Engine::generate_batch): out holds max_out tokens per sequence, counts what
// each emitted. -1: out of KV pages (the sequences left as they were).
int kiln_generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts) {
    try { E->generate_batch(n, ids, sps, out, max_out, counts); return 0; } catch (const std::exception &) { return -1; }
}
// The SAMPLE_N largest of the last prefill's logits among the tokens a bitmask allows (bit t % 32 of word t / 32), a
// constrained answer's candidates for kiln_draw. The logits are overwritten.
void kiln_masked_top(const uint32_t *mask, float *v, int *ids) {
    CK(hipMemcpyAsync(E->mask_rows(), mask, VOCAB / 32 * 4, hipMemcpyHostToDevice, E->st));
    mask_logits(E->b.logits, E->d_mask, VOCAB, 1, E->st);
    topk(E->b.logits, 1, VOCAB, VOCAB, E->b.tsv, E->b.tsi, E->b.topv, E->b.topi, E->st);
    CK(hipMemcpyAsync(v, E->b.topv, SAMPLE_N * 4, hipMemcpyDeviceToHost, E->st));
    CK(hipMemcpyAsync(ids, E->b.topi, SAMPLE_N * 4, hipMemcpyDeviceToHost, E->st));
    CK(hipStreamSynchronize(E->st));
}
// A token for sequence id drawn from the given SAMPLE_N candidates (logits and ids; -inf for none) with its sampler and
// random state, as generate() draws: kiln/server.py's constrained answers pass the masked logits' largest. It joins the
// reply's tokens (presence penalty, DRY); prefill it to go on.
int kiln_draw(int id, const float *v, const int *ids, const Sampling *sp) {
    Seq &s = E->seqs.at(id);
    const int t = draw(make_dist(v, ids, *sp, s.seen, s.ctx), s.rng);
    s.seen.push(t);
    if (sp->lookup > 0) s.look.push(t);
    return t;
}
// tok becomes sequence id's pending token, the first input of its next step: a held answer's token from kiln_draw goes
// on this way, with drafts, where a prefill would read it alone.
void kiln_seq_pending(int id, int tok) { E->seqs.at(id).pending = tok; }
// Drafts for a held answer's next step (Engine::draft); -1 when the KV pool lacks the pages.
int kiln_draft(int id, const Sampling *sp, const int *fixed, int nfixed, int max_k, int *out) {
    try { return E->draft(id, *sp, fixed, nfixed, max_k, out); } catch (const std::exception &) { return -1; }
}
// One step of a held answer (Engine::generate_masked); -1 when the KV pool lacks the pages.
int kiln_generate_masked(int id, const Sampling *sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out) {
    try { return E->generate_masked(id, *sp, drafts, k, masks, out, max_out); } catch (const std::exception &) { return -1; }
}
int kiln_snapshot(int id) { return E->snapshot(id); }
int kiln_restore(int id, int snap) { try { E->restore(id, snap); return 0; } catch (const std::exception &) { return -1; } }
int kiln_snap_free(int snap) { try { E->snap_free(snap); return 0; } catch (const std::exception &) { return -1; } }
// A snapshot as bytes for a later run of the same model and build (Engine::snap_export), and back: kiln_snap_import
// returns the new snapshot, -1 for bytes it cannot take, -2 when the KV pool lacks the pages.
int64_t kiln_snap_bytes(int snap) { try { return (int64_t)E->snap_bytes(snap); } catch (const std::exception &) { return -1; } }
int kiln_snap_export(int snap, void *dst) { try { E->snap_export(snap, (uint8_t *)dst); return 0; } catch (const std::exception &) { return -1; } }
int kiln_snap_import(const void *src, int64_t bytes) { return E->snap_import((const uint8_t *)src, (size_t)bytes); }
void kiln_sync() { CK(hipStreamSynchronize(E->st)); }
// Capacity errors (out of KV pages) come back as -1 with the sequence left as it was; tokens that name no encoded picture
// (or no token at all) as -2, before anything runs. pos3: null for text, else [n][3] M-RoPE positions (kiln/vision.py).
int kiln_prefill_pos(int id, const int *tokens, const int *pos3, int n) {
    try { E->prefill(id, tokens, pos3, n); return 0; }
    catch (const std::invalid_argument &) { return -2; }
    catch (const std::exception &) { return -1; }
}
int kiln_prefill(int id, const int *tokens, int n) { return kiln_prefill_pos(id, tokens, nullptr, n); }
// The vision tower (engine/vision.h): its GGUF, read when the first picture is encoded.
void kiln_vision_open(const char *path) { E->vision.open(path, D); }
// Its weights out of RAM until the next picture.
void kiln_vision_release() { E->vision.release(); }
// A picture's embeddings under `slot`, for its image tokens: patches float32 [gh * gw][768] (kiln/vision.py). 0, or -1 if
// the picture is too large for the activation arena.
int kiln_image_encode(int slot, const float *patches, int gh, int gw) { return E->image_encode(slot, patches, gh, gw); }
// A video's frame pair under `slot`, as a picture's (kiln_image_encode) but patches [gh * gw][1536]: each its two frames.
int kiln_video_encode(int slot, const float *patches, int gh, int gw) { return E->image_encode(slot, patches, gh, gw, 2); }
void kiln_image_free(int slot) {
    if (slot < 0 || slot >= (int)E->images.size() || !E->images[slot].rows) return;
    CK(hipHostFree(E->images[slot].rows));
    E->images[slot] = {};
}
// A picture's embeddings as fp32 [rows][D] (tests compare them with the reference vision tower's)
int kiln_image_read(int slot, float *out) {
    if (slot < 0 || slot >= (int)E->images.size() || !E->images[slot].rows) return -1;
    const Engine::Image &im = E->images[slot];
    for (size_t i = 0; i < (size_t)im.n * D; i++) out[i] = __half2float(im.rows[i]);
    return im.n;
}
// rows_v / rows_i: null, or room for max_out rows of SAMPLE_N candidates (each emitted token's, for log probs).
int kiln_generate(int id, const Sampling *sp, int *out, int max_out, float *rows_v, int *rows_i) {
    try { return E->generate(id, *sp, out, max_out, rows_v, rows_i); } catch (const std::exception &) { return -1; }
}
// Full logits of the last prefilled position (validation against reference implementations).
void kiln_last_logits(float *out) { CK(hipMemcpy(out, E->b.logits, (size_t)VOCAB * 4, hipMemcpyDeviceToHost)); }
}
