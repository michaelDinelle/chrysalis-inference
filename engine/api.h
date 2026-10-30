#pragma once
#include "sampler.h"
#include <cstdint>

extern "C" {
void chrysalis_calibration(const char *path);
int chrysalis_init(const char *path, int max_seqs, int chunk, long long reserve_mb);
void chrysalis_warmup();
void chrysalis_reserve_snapshots(int n);
float chrysalis_load_progress();
void chrysalis_shutdown(int);
void chrysalis_sync();
int chrysalis_free_tokens();

int chrysalis_seq_new();
void chrysalis_seq_free(int id);
int chrysalis_seq_len(int id);
void chrysalis_seq_truncate(int id, int n);
void chrysalis_seq_begin(int id, uint64_t seed, const Sampling *sp);
void chrysalis_seq_context(int id, const int *tokens, int n);
void chrysalis_seq_breakers(int id, const int *flat, int n);
void chrysalis_seq_bias(int id, const int *ids, const float *vals, int n);
void chrysalis_seq_push(int id, const int *tokens, int n);
void chrysalis_seq_lookup(int id, const int *tokens, int n);
void chrysalis_seq_pending(int id, int tok);

int chrysalis_prefill(int id, const int *tokens, int n);
int chrysalis_prefill_pos(int id, const int *tokens, const int *pos3, int n);
int chrysalis_generate(int id, const Sampling *sp, int *out, int max_out, float *rows_v, int *rows_i);
int chrysalis_generate_batch(int n, const int *ids, const Sampling *sps, int *out, int max_out, int *counts, float *rows_v, int *rows_i);
int chrysalis_draft(int id, const Sampling *sp, const int *fixed, int nfixed, int max_k, int *out);
int chrysalis_generate_masked(int id, const Sampling *sp, const int *drafts, int k, const uint32_t *masks, int *out, int max_out);
void chrysalis_masked_top(const uint32_t *mask, float *v, int *ids, float *stats);
int chrysalis_draw(int id, const float *v, const int *ids, const float *stats, const Sampling *sp);
void chrysalis_last_logits(float *out);

int chrysalis_snapshot(int id);
int chrysalis_restore(int id, int snap);
int chrysalis_snap_free(int snap);
int64_t chrysalis_snap_bytes(int snap);
int chrysalis_snap_export(int snap, void *dst);
int chrysalis_snap_import(int fd, int64_t offset, int64_t bytes);

void chrysalis_vision_open(const char *path);
void chrysalis_vision_release();
int chrysalis_image_encode(int slot, const float *patches, int gh, int gw);
int chrysalis_video_encode(int slot, const float *patches, int gh, int gw);
void chrysalis_image_free(int slot);
int chrysalis_image_read(int slot, float *out);

double chrysalis_tune(const char *key, double value);
void chrysalis_expert_stats(int64_t *out);
void chrysalis_route_counts(int64_t *out);
int chrysalis_pool_save(const char *path);
int chrysalis_pool_warm(const char *path);
int chrysalis_bench_fill(int id, int n);
}
