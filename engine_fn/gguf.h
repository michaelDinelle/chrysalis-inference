// Minimal read-only GGUF parser: memory-maps the file and indexes tensors by name.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct GGUFTensor {
    int type = -1;
    int64_t ne[4] = {1, 1, 1, 1};
    const uint8_t *data = nullptr;
    size_t nbytes = 0;
};

struct GGUF {
    std::map<std::string, GGUFTensor> tensors;
    std::map<std::string, int64_t> ints;
    std::map<std::string, std::string> strs;
    std::map<std::string, std::vector<int64_t>> arrs;   // integer array metadata
    uint8_t *base = nullptr;
    size_t size = 0;
    int fd = -1;                              // kept open for drop()

    static size_t type_bytes(int type, int64_t n) {
        switch (type) {
            case 0: return n * 4;                // F32
            case 1: return n * 2;                // F16
            case 6: return n / 32 * 22;          // Q5_0
            case 8: return n / 32 * 34;          // Q8_0
            case 20: return n / 32 * 18;         // IQ4_NL
            case 26: return n * 4;               // I32
            case 30: return n * 2;               // BF16
            case 14: return n / 256 * 210;       // Q6_K
            case 23: return n / 256 * 136;       // IQ4_XS
            case 10: return n / 256 * 84;        // Q2_K
            case 11: return n / 256 * 110;       // Q3_K
            default: return 0;
        }
    }

    bool open(const char *path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st); size = st.st_size;
        base = (uint8_t *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED || memcmp(base, "GGUF", 4) != 0) return false;

        const uint8_t *p = base + 4;
        auto rd = [&](auto &v) { memcpy(&v, p, sizeof(v)); p += sizeof(v); };
        auto rs = [&]() { uint64_t n; rd(n); std::string s((const char *)p, n); p += n; return s; };
        uint32_t version; uint64_t n_tensors, n_kv;
        rd(version); rd(n_tensors); rd(n_kv);

        static const int scalar_size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rs();
            uint32_t t; rd(t);
            if (t == 8) { strs[key] = rs(); continue; }           // string
            if (t == 9) {                                         // array
                uint32_t at; uint64_t n; rd(at); rd(n);
                if (at == 8) for (uint64_t j = 0; j < n; j++) rs();
                else if (at == 11 || at == 10) { std::vector<int64_t> v(n); memcpy(v.data(), p, n * 8); arrs[key] = v; p += n * 8; }
                else if (at == 4 || at == 5) { std::vector<int64_t> v(n); for (uint64_t j = 0; j < n; j++) { int32_t x; memcpy(&x, p + 4 * j, 4); v[j] = x; } arrs[key] = v; p += n * 4; }
                else p += n * scalar_size[at];
                continue;
            }
            int64_t v = 0;
            if (t == 4 || t == 5) { int32_t x; rd(x); v = x; }
            else if (t == 10 || t == 11) rd(v);
            else p += scalar_size[t];
            ints[key] = v;
            if (key == "general.alignment") alignment = v;
        }
        std::map<std::string, uint64_t> offsets;
        for (uint64_t i = 0; i < n_tensors; i++) {
            std::string name = rs();
            uint32_t nd; rd(nd);
            GGUFTensor t;
            for (uint32_t d = 0; d < nd; d++) rd(t.ne[d]);
            uint32_t type; rd(type); t.type = type;
            uint64_t off; rd(off); offsets[name] = off;
            tensors[name] = t;
        }
        size_t data_start = ((p - base) + alignment - 1) / alignment * alignment;
        for (auto &[name, t] : tensors) {
            t.data = base + data_start + offsets[name];
            t.nbytes = type_bytes(t.type, t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
            if (!t.nbytes) { fprintf(stderr, "kiln: tensor %s has a type this reader does not know (%d)\n", name.c_str(), t.type); return false; }
        }
        return true;
    }

    const GGUFTensor *get(const std::string &name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }

    // evict a range of the mapping from the page cache once it has been copied out (keeps the cache for the n-gram table)
    void drop(const void *p, size_t n) const {
        const size_t off = (size_t)((const uint8_t *)p - base) & ~size_t(4095);
        madvise(base + off, n + ((const uint8_t *)p - base - off), MADV_DONTNEED);
        posix_fadvise(fd, (off_t)off, (off_t)(n + ((const uint8_t *)p - base - off)), POSIX_FADV_DONTNEED);
    }
    ~GGUF() { if (base && base != MAP_FAILED) munmap(base, size); if (fd >= 0) ::close(fd); }
};
