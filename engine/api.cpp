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
// The prompt's tail that DRY looks back over (with the reply), for the next reply of this sequence.
void kiln_seq_context(int id, const int *tokens, int n) { E->seqs.at(id).ctx.assign(tokens, tokens + n); }
// DRY's sequence breakers: token ids that end a repeated run.
void kiln_dry_breakers(const int *ids, int n) {
    dry_breakers.assign(VOCAB, 0);
    for (int i = 0; i < n; i++) if (ids[i] >= 0 && ids[i] < VOCAB) dry_breakers[ids[i]] = 1;
}
int kiln_free_tokens() { return (int)E->free_pages.size() * PAGE; }
int kiln_snapshot(int id) { return E->snapshot(id); }
int kiln_restore(int id, int snap) { try { E->restore(id, snap); return 0; } catch (const std::exception &) { return -1; } }
int kiln_snap_free(int snap) { try { E->snap_free(snap); return 0; } catch (const std::exception &) { return -1; } }
void kiln_sync() { CK(hipStreamSynchronize(E->st)); }
// Capacity errors (out of KV pages) come back as -1 with the sequence left as it was.
int kiln_prefill(int id, const int *tokens, int n) {
    try { E->prefill(id, tokens, n); return 0; } catch (const std::exception &) { return -1; }
}
// rows_v / rows_i: null, or room for max_out rows of SAMPLE_N candidates (each emitted token's, for log probs).
int kiln_generate(int id, const Sampling *sp, int *out, int max_out, float *rows_v, int *rows_i) {
    try { return E->generate(id, *sp, out, max_out, rows_v, rows_i); } catch (const std::exception &) { return -1; }
}
// Full logits of the last prefilled position (validation against reference implementations).
void kiln_last_logits(float *out) { CK(hipMemcpy(out, E->b.logits, (size_t)VOCAB * 4, hipMemcpyDeviceToHost)); }
}
