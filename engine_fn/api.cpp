// C API used by the Python server through ctypes (the same functions as engine/api.cpp, in libkiln_fn.so).
#include "model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

static Engine *E = nullptr;

float load_progress();                          // load.cpp (C++ linkage, so declared outside extern "C")

extern "C" {
struct KilnSampling { float temperature, top_p, min_p, presence_penalty; int top_k, n_draft; };

// Loads the model; returns KV capacity in tokens. chunk: tokens per prefill pass (each RAM expert is read once per pass).
int kiln_init(const char *path, int max_seqs, int chunk, long long reserve_mb) {
    E = new Engine;
    if (chunk > 0) E->smax = chunk;
    E->umax = std::min(E->smax, 1024);
    E->load(path, max_seqs, (size_t)reserve_mb << 20);
    return E->n_pages * PAGE;
}
void kiln_warmup() { E->warmup(); }
// Pinned buffers for n snapshots in all: made by warmup(), or now if it already ran.
void kiln_reserve_snapshots(int n) { E->reserve_snapshots(n); }
// 0..1 while kiln_init / kiln_warmup run (polled from another thread for a progress bar)
float kiln_load_progress() { return load_progress(); }
// frees everything the engine made (model switch); the argument is unused (the 27B engine's signature)
void kiln_shutdown(int) { if (E) { E->shutdown(); delete E; E = nullptr; } }
int kiln_seq_new() { return E->seq_new(); }
void kiln_seq_free(int id) { E->seq_free(id); }
int kiln_seq_len(int id) { return E->seqs.at(id).n; }
void kiln_seq_truncate(int id, int n) { E->seq_truncate(id, n); }
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
int kiln_prefill(int id, const int *tokens, int n) {
    try { E->prefill(id, tokens, n); return 0; } catch (const std::exception &) { return -1; }
}
int kiln_generate(int id, const KilnSampling *p, int *out, int max_out) {
    Sampling sp;
    sp.temperature = p->temperature; sp.top_p = p->top_p; sp.min_p = p->min_p;
    sp.presence_penalty = p->presence_penalty; sp.top_k = p->top_k; sp.n_draft = p->n_draft;
    try { return E->generate(id, sp, out, max_out); } catch (const std::exception &) { return -1; }
}
// Tuning knobs for experiments (benchmarks set them between runs); returns the previous value, NAN if unknown.
double kiln_tune(const char *key, double value) {
    const std::string k = key;
    double old = NAN;
    if (k == "gfrac") { old = E->gfrac; E->gfrac = (float)value; }
    if (k == "split") { old = E->split; E->split = (int)value; }
    if (k == "vnni") old = E->cpu.use_vnni((int)value);
    if (k == "prefetch") { old = E->pf_on; E->pf_on = value != 0; }
    if (k == "cpu_bias") { old = E->cpu_bias; if (value > 0) E->cpu_bias = value; }
    if (k == "adapt") { old = E->adapt; E->adapt = value != 0; }
    if (k == "lookahead") { old = E->la_on; E->la_on = value != 0; }
    if (k == "adapt_left") { old = E->adapt_left; if (value >= 0) E->adapt_left = (int)value; }
    return old;
}
// Routing statistics (tools_fn/route_stats.py): out == nullptr starts counting the picks per (layer, expert) of prefill
// passes from zero; otherwise copies the [48][512] counts so far.
void kiln_route_counts(int64_t *out) {
    if (!out) E->route_counts.assign((size_t)NL * NE, 0);
    else std::copy(E->route_counts.begin(), E->route_counts.end(), out);
}
// Full logits of the last prefilled position (validation against reference implementations).
void kiln_last_logits(float *out) { CK(hipMemcpy(out, E->b.logits, (size_t)VOCAB * 4, hipMemcpyDeviceToHost)); }
}
