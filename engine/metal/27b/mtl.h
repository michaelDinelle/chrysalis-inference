#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define FAIL(...) do { fprintf(stderr, "chrysalis: " __VA_ARGS__); fprintf(stderr, "\n"); abort(); } while (0)

template <typename T> struct DP {
    uint64_t g = 0;
    T *h = nullptr;
    DP operator+(size_t n) const { return {g + n * sizeof(T), h + n}; }
    explicit operator bool() const { return g != 0; }
    template <typename U> DP<U> as() const { return {g, (U *)h}; }
};

struct Mtl {
    void *dev = nullptr, *queue = nullptr, *lib = nullptr, *rset = nullptr;
    void *cb = nullptr, *enc = nullptr;
    bool skip_barrier = true, resident = false;
    std::vector<void *> bufs;
    size_t allocated = 0;
    size_t max_ws = 0;
    int dispatches = 0;

    void init(const std::string &metallib);
    DP<uint8_t> alloc(size_t bytes, const char *what);
    DP<uint8_t> try_alloc(size_t bytes);
    void free_all();
    void make_resident();
    void *pipeline(const char *name, const int *consts = nullptr, int nconsts = 0);
    void save_archive();
    void run(void *pso, const void *args, size_t nargs, int gx, int gy, int gz, int tx, int ty = 1, int tz = 1);
    template <typename A> void run(void *pso, const A &a, int gx, int gy, int gz, int tx) { run(pso, &a, sizeof a, gx, gy, gz, tx); }
    void gemm(uint64_t A, int lda, uint64_t B, uint64_t C, int ldc, bool c_f32, int M, int N, int K, float beta, int ldb = 0);
    void barrier() { skip_barrier = false; }
    void par() { skip_barrier = true; }
    void sync();
    double last_gpu_ms = 0, last_start_ms = 0, last_wake_ms = 0, commit_time = 0;
private:
    void *last_cb = nullptr, *archive = nullptr;
    std::string archive_path;
    bool archive_dirty = false;
    void begin();
    std::vector<std::pair<uint64_t, size_t>> ranges;
    void commit();
    void *enc_begin();
    struct Warm;
    Warm *warm = nullptr;
    void warm_start();
};
