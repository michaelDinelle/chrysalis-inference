#pragma once
#include "sampler.h"
#include <cstdint>

#ifdef _WIN32
#define CHRYSALIS_API __declspec(dllexport)
#else
#define CHRYSALIS_API
#endif

#pragma GCC visibility push(default)
extern "C" {
CHRYSALIS_API void chrysalis_calibration(const char *path);
CHRYSALIS_API int chrysalis_init(const char *path, int max_seqs, int chunk, long long reserve_mb);
CHRYSALIS_API void chrysalis_warmup();
CHRYSALIS_API void chrysalis_reserve_snapshots(int n);
CHRYSALIS_API float chrysalis_load_progress();
CHRYSALIS_API void chrysalis_shutdown(int);
CHRYSALIS_API void chrysalis_sync();
CHRYSALIS_API int chrysalis_free_tokens();

CHRYSALIS_API int chrysalis_seq_new();
CHRYSALIS_API void chrysalis_seq_free(int id);
CHRYSALIS_API int chrysalis_seq_len(int id);
CHRYSALIS_API void chrysalis_seq_truncate(int id, int n);
CHRYSALIS_API void chrysalis_seq_begin(int id, uint64_t seed, const Sampling *sp);
CHRYSALIS_API void chrysalis_seq_context(int id, const int *tokens, int n);
CHRYSALIS_API void chrysalis_seq_breakers(int id, const int *flat, int n);
CHRYSALIS_API void chrysalis_seq_bias(int id, const int *ids, const float *vals, int n);
CHRYSALIS_API void chrysalis_seq_push(int id, const int *tokens, int n);
CHRYSALIS_API void chrysalis_seq_lookup(int id, const int *tokens, int n);
CHRYSALIS_API void chrysalis_seq_pending(int id, int tok);

CHRYSALIS_API int chrysalis_prefill(int id, const int *tokens, int n);
CHRYSALIS_API int chrysalis_prefill_pos(int id, const int *tokens, const int *pos3, int n);
CHRYSALIS_API int chrysalis_generate(int id, const Sampling *sp, int *out, int max_out, float *rows_v, int *rows_i);
CHRYSALIS_API int chrysalis_generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v, int *rows_i);
CHRYSALIS_API int chrysalis_draft(int id, const Sampling *sp, const int *fixed, int nfixed, int max_k, int *out);
CHRYSALIS_API int chrysalis_generate_masked(int id, const Sampling *sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out);
CHRYSALIS_API void chrysalis_masked_top(const uint32_t *mask, float *v, int *ids, float *stats);
CHRYSALIS_API int chrysalis_draw(int id, const float *v, const int *ids, const float *stats, const Sampling *sp);
CHRYSALIS_API void chrysalis_last_logits(float *out);

CHRYSALIS_API int chrysalis_snapshot(int id);
CHRYSALIS_API int chrysalis_restore(int id, int snap);
CHRYSALIS_API int chrysalis_snap_free(int snap);
CHRYSALIS_API int64_t chrysalis_snap_bytes(int snap);
CHRYSALIS_API int chrysalis_snap_export(int snap, void *dst);
CHRYSALIS_API int chrysalis_snap_import(int64_t file, int64_t offset, int64_t bytes);

CHRYSALIS_API void chrysalis_vision_open(const char *path);
CHRYSALIS_API void chrysalis_vision_release();
CHRYSALIS_API int chrysalis_image_encode(int slot, const float *patches, int gh, int gw);
CHRYSALIS_API int chrysalis_video_encode(int slot, const float *patches, int gh, int gw);
CHRYSALIS_API void chrysalis_image_free(int slot);
CHRYSALIS_API int chrysalis_image_read(int slot, float *out);

CHRYSALIS_API double chrysalis_tune(const char *key, double value);
CHRYSALIS_API void chrysalis_expert_stats(int64_t *out);
CHRYSALIS_API void chrysalis_route_counts(int64_t *out);
CHRYSALIS_API int chrysalis_pool_save(const char *path);
CHRYSALIS_API int chrysalis_pool_warm(const char *path);
CHRYSALIS_API int chrysalis_bench_fill(int id, int n);
}
#pragma GCC visibility pop
