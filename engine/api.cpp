// C API used by the Python server through ctypes.
#include "model.h"
#include <stdexcept>

static Engine *E = nullptr;

extern "C" {
struct KilnSampling { float temperature, top_p, min_p, presence_penalty; int top_k, n_draft; };

// Loads the model; returns KV capacity in tokens.
int kiln_init(const char *path, int max_seqs, int umax, long long reserve_mb) {
    E = new Engine;
    E->umax = umax;
    E->load(path, max_seqs, (size_t)reserve_mb << 20);
    return E->n_pages * PAGE;
}
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
int kiln_free_tokens() { return (int)E->free_pages.size() * PAGE; }
int kiln_snapshot(int id) { return E->snapshot(id); }
void kiln_restore(int id, int snap) { E->restore(id, snap); }
void kiln_snap_free(int snap) { E->snap_free(snap); }
void kiln_sync() { CK(hipStreamSynchronize(E->st)); }
// Capacity errors (out of KV pages) come back as -1 with the sequence left as it was.
int kiln_prefill(int id, const int *tokens, int n) {
    try { E->prefill(id, tokens, n); return 0; } catch (const std::exception &) { return -1; }
}
int kiln_generate(int id, const KilnSampling *p, int *out, int max_out) {
    Sampling sp;
    sp.temperature = p->temperature; sp.top_p = p->top_p; sp.min_p = p->min_p;
    sp.presence_penalty = p->presence_penalty; sp.top_k = p->top_k; sp.n_draft = p->n_draft;
    try { return E->generate(id, sp, out, max_out); } catch (const std::exception &) { return -1; }
}
// Full logits of the last prefilled position (validation against reference implementations).
void kiln_last_logits(float *out) { CK(hipMemcpy(out, E->b.logits, (size_t)VOCAB * 4, hipMemcpyDeviceToHost)); }
}
