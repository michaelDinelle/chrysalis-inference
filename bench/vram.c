#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int hipError_t;
typedef void *hipStream_t;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static void **ptrs;
static size_t *sizes, n, cap, now, peak;
static int out = -2;

static void save(void) {
    if (out == -2) {
        const char *path = getenv("BENCH_VRAM_OUT");
        out = path ? open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644) : -1;
    }
    if (out < 0) return;
    char line[32];
    int len = snprintf(line, sizeof line, "%020zu\n", peak);
    if (pwrite(out, line, (size_t)len, 0) != len) out = -1;
}

static void note(void *p, size_t size) {
    pthread_mutex_lock(&mu);
    if (n == cap) {
        cap = cap ? cap * 2 : 1024;
        ptrs = realloc(ptrs, cap * sizeof *ptrs);
        sizes = realloc(sizes, cap * sizeof *sizes);
    }
    ptrs[n] = p;
    sizes[n++] = size;
    now += size;
    if (now > peak) {
        peak = now;
        save();
    }
    pthread_mutex_unlock(&mu);
}

static void drop(void *p) {
    pthread_mutex_lock(&mu);
    for (size_t i = n; i-- > 0;)
        if (ptrs[i] == p) {
            now -= sizes[i];
            ptrs[i] = ptrs[--n];
            sizes[i] = sizes[n];
            break;
        }
    pthread_mutex_unlock(&mu);
}

static void *hip(const char *name) {
    void *f = dlsym(RTLD_NEXT, name);
    if (!f) {
        void *h = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_NOLOAD);
        f = h ? dlsym(h, name) : NULL;
    }
    if (!f) {
        fprintf(stderr, "vram_peak: no %s in libamdhip64\n", name);
        abort();
    }
    return f;
}

#define REAL(name, type) static __typeof__(type) real; if (!real) real = (__typeof__(type))hip(name)

hipError_t hipMalloc(void **p, size_t size) {
    REAL("hipMalloc", hipError_t (*)(void **, size_t));
    hipError_t e = real(p, size);
    if (!e && *p) note(*p, size);
    return e;
}

hipError_t hipExtMallocWithFlags(void **p, size_t size, unsigned flags) {
    REAL("hipExtMallocWithFlags", hipError_t (*)(void **, size_t, unsigned));
    hipError_t e = real(p, size, flags);
    if (!e && *p) note(*p, size);
    return e;
}

hipError_t hipMallocAsync(void **p, size_t size, hipStream_t s) {
    REAL("hipMallocAsync", hipError_t (*)(void **, size_t, hipStream_t));
    hipError_t e = real(p, size, s);
    if (!e && *p) note(*p, size);
    return e;
}

hipError_t hipFree(void *p) {
    REAL("hipFree", hipError_t (*)(void *));
    hipError_t e = real(p);
    if (!e && p) drop(p);
    return e;
}

hipError_t hipFreeAsync(void *p, hipStream_t s) {
    REAL("hipFreeAsync", hipError_t (*)(void *, hipStream_t));
    hipError_t e = real(p, s);
    if (!e && p) drop(p);
    return e;
}
