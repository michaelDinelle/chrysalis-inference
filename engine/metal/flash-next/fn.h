#pragma once
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

[[noreturn, gnu::cold, gnu::noinline, gnu::format(printf, 1, 2)]] inline void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("chrysalis-fn: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}
#define FAIL(...) fail(__VA_ARGS__)

constexpr int D = 2560;
constexpr int HC = 4;
constexpr int HCD = HC * D;
constexpr int HCR = 320;
constexpr int VOCAB = 248320;
constexpr int NL = 48;
constexpr int NH = 24, NKV = 2, HD = 256, ROT = 64;
constexpr int QPK = NH / NKV;
constexpr int GK = 16, GV = 48, GD = 128;
constexpr int QKV = 2 * GK * GD + GV * GD;
constexpr int NGDN = 36, NATT = 12, NKVL = NATT + 1;
constexpr int NE = 512, TOPK_E = 10, FE = 640, FS = 640;
constexpr int IDX_H = 4, IDX_D = 128, KPOOL = 4, SEL_BLOCKS = 512;
constexpr int PLE_LAYER = 1, PLE_HEADS = 16, PLE_DIM = 160, PLE_K = 4, PLE_DIL = 3;
constexpr int PLE_HIST = (PLE_K - 1) * PLE_DIL;
constexpr int PLE_ROW = PLE_DIM / 32 * 34;
constexpr float EPS = 1e-6f;
constexpr int EOS_IM_END = 248046, EOS_TEXT = 248044;
constexpr int IMG_BASE = 1 << 30, IMG_ROWS = 1 << 14, IMAGE_PAD = 248056;
inline bool is_attn_layer(int l) { return l % 4 == 3; }

constexpr int T_IQ4_NL = 20, T_IQ4_XS = 23, T_Q3_K = 11, T_Q2_K = 10;
constexpr size_t XDN = (size_t)D * FE / 32 * 18;
constexpr size_t xgu_bytes(int type) { return (size_t)FE * (D / 256) * (type == T_IQ4_XS ? 136 : type == T_Q3_K ? 110 : 84); }
constexpr size_t XSLOT = XDN + 2 * xgu_bytes(T_IQ4_XS);
constexpr size_t XSLOT_MTP = XDN + 2 * xgu_bytes(T_Q3_K);

constexpr int PAGE = 64;
constexpr size_t KV_K = (size_t)NKV * PAGE * HD * 2;
constexpr size_t KV_IDX = (size_t)(PAGE / KPOOL) * IDX_D * 2;
constexpr size_t KV_LAYER = 2 * KV_K + KV_IDX;
constexpr size_t KV_PAGE = KV_LAYER * NKVL;
constexpr int DRAFT_VOCAB = 65536;

inline size_t ggml_row_bytes(int type, int64_t cols) {
    switch (type) {
        case 0: case 26: return cols * 4;
        case 1: case 30: return cols * 2;
        case 2: case 20: return cols / 32 * 18;
        case 6: return cols / 32 * 22;
        case 8: return cols / 32 * 34;
        case 42: return cols / 64 * 18;
        case 10: return cols / 256 * 84;
        case 11: return cols / 256 * 110;
        case 12: return cols / 256 * 144;
        case 13: return cols / 256 * 176;
        case 14: return cols / 256 * 210;
        case 23: return cols / 256 * 136;
    }
    return 0;
}

struct GGUFTensor {
    int type = -1;
    int64_t ne[4] = {1, 1, 1, 1};
    size_t off = 0;
    size_t nbytes = 0;
    int file = 0;
};

struct GGUF {
    std::map<std::string, GGUFTensor> tensors;
    std::map<std::string, int64_t> ints;
    std::map<std::string, std::string> strs;
    std::map<std::string, std::vector<int64_t>> arrs;
    std::vector<int> fds;
    std::vector<size_t> sizes;

    bool open_one(const std::string &path) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        const int file = (int)fds.size();
        fds.push_back(fd); sizes.push_back(st.st_size);
        std::vector<uint8_t> buf;
        size_t pos = 0;
        auto need = [&](size_t n) {
            while (pos + n > buf.size()) {
                const size_t old = buf.size();
                buf.resize(old + (1 << 20));
                const ssize_t r = pread(fd, buf.data() + old, 1 << 20, old);
                if (r <= 0) FAIL("short GGUF header in %s", path.c_str());
                buf.resize(old + r);
            }
        };
        auto rd = [&](auto &v) { need(sizeof(v)); memcpy(&v, buf.data() + pos, sizeof(v)); pos += sizeof(v); };
        auto rs = [&]() { uint64_t n; rd(n); need(n); std::string s((const char *)buf.data() + pos, n); pos += n; return s; };
        uint32_t magic, version; uint64_t n_tensors, n_kv;
        rd(magic); rd(version); rd(n_tensors); rd(n_kv);
        if (memcmp(&magic, "GGUF", 4)) return false;
        static const int scalar_size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rs();
            uint32_t t; rd(t);
            if (t == 8) { strs[key] = rs(); continue; }
            if (t == 9) {
                uint32_t at; uint64_t n; rd(at); rd(n);
                if (at == 8) { for (uint64_t j = 0; j < n; j++) rs(); continue; }
                std::vector<int64_t> v(n);
                for (uint64_t j = 0; j < n; j++) {
                    if (at == 10 || at == 11) { int64_t x; rd(x); v[j] = x; }
                    else if (at == 4 || at == 5) { int32_t x; rd(x); v[j] = x; }
                    else { need(scalar_size[at]); pos += scalar_size[at]; }
                }
                arrs[key] = v;
                continue;
            }
            int64_t v = 0;
            if (t == 4 || t == 5) { int32_t x; rd(x); v = x; }
            else if (t == 10 || t == 11) rd(v);
            else if (t == 6) { float f; rd(f); v = (int64_t)f; }
            else { need(scalar_size[t]); pos += scalar_size[t]; }
            ints[key] = v;
            if (key == "general.alignment") alignment = v;
        }
        std::vector<std::pair<std::string, uint64_t>> offs;
        for (uint64_t i = 0; i < n_tensors; i++) {
            std::string name = rs();
            uint32_t nd; rd(nd);
            GGUFTensor t;
            for (uint32_t d = 0; d < nd; d++) rd(t.ne[d]);
            uint32_t type; rd(type); t.type = type;
            uint64_t off; rd(off);
            t.file = file;
            const size_t rb = ggml_row_bytes(t.type, t.ne[0]);
            if (!rb) FAIL("tensor %s: unknown type %d", name.c_str(), t.type);
            t.nbytes = rb * t.ne[1] * t.ne[2] * t.ne[3];
            tensors[name] = t;
            offs.push_back({name, off});
        }
        const size_t data = (pos + alignment - 1) / alignment * alignment;
        for (auto &[n, o] : offs) {
            GGUFTensor &t = tensors[n];
            t.off = data + o;
            if (t.off + t.nbytes > (size_t)st.st_size) FAIL("tensor %s runs past the end of %s (a partial download?)", n.c_str(), path.c_str());
        }
        return true;
    }
    bool open(const std::string &path) {
        if (!open_one(path)) return false;
        const auto at = path.rfind("-00001-of-");
        if (at != std::string::npos) {
            const int n = atoi(path.c_str() + at + 10);
            for (int k = 2; k <= n; k++) {
                char part[32];
                snprintf(part, sizeof part, "-%05d-of-%05d", k, n);
                std::string p = path.substr(0, at) + part + path.substr(at + 15);
                if (!open_one(p)) FAIL("cannot open shard %s", p.c_str());
            }
        }
        return true;
    }
    const GGUFTensor *get(const std::string &name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }
    ~GGUF() { for (int fd : fds) ::close(fd); }
};

struct XSrc { size_t off[3]; int type; };
inline void expert_sources(const GGUF &g, int l, XSrc *src) {
    const std::string p = "blk." + std::to_string(l) + ".";
    auto tensor = [&](const char *name, int type, int64_t cols, int64_t rows, int64_t *n) -> const GGUFTensor * {
        const GGUFTensor *t = g.get(p + name);
        if (!t) return nullptr;
        if (t->type != type || t->ne[0] != cols || t->ne[1] != rows) FAIL("layer %d: %s is not type %d [%lld, %lld]", l, name, type, (long long)cols, (long long)rows);
        if (*n >= 0 && t->ne[2] != *n) FAIL("layer %d: %s holds %lld experts, not %lld", l, name, (long long)t->ne[2], (long long)*n);
        *n = t->ne[2];
        return t;
    };
    int64_t nh = -1, n0 = -1, n1 = -1, nc = -1;
    const GGUFTensor *hg = tensor("ffn_gate_hot.weight", T_IQ4_XS, D, FE, &nh), *hu = tensor("ffn_up_hot.weight", T_IQ4_XS, D, FE, &nh),
                     *hd = tensor("ffn_down_hot.weight", T_IQ4_NL, FE, D, &nh);
    const GGUFTensor *cg[2] = {tensor("ffn_gate_cold0.weight", T_Q3_K, D, FE, &n0), tensor("ffn_gate_cold1.weight", T_Q2_K, D, FE, &n1)},
                     *cu[2] = {tensor("ffn_up_cold0.weight", T_Q3_K, D, FE, &n0), tensor("ffn_up_cold1.weight", T_Q2_K, D, FE, &n1)},
                     *cd = tensor("ffn_down_cold.weight", T_IQ4_NL, FE, D, &nc);
    nh = std::max<int64_t>(nh, 0); n0 = std::max<int64_t>(n0, 0); n1 = std::max<int64_t>(n1, 0); nc = std::max<int64_t>(nc, 0);
    if ((nh && (!hg || !hu || !hd)) || (n0 && (!cg[0] || !cu[0])) || (n1 && (!cg[1] || !cu[1])) || nc != n0 + n1 || nh + nc != NE)
        FAIL("layer %d: experts hot %lld + cold %lld + %lld (downs %lld), not %d", l, (long long)nh, (long long)n0, (long long)n1, (long long)nc, NE);
    const GGUFTensor *st = g.get(p + "ffn_exp_slot");
    if (!st || st->type != 26 || st->ne[0] != NE) FAIL("layer %d: no ffn_exp_slot", l);
    int32_t slot[NE];
    if (pread(g.fds[st->file], slot, sizeof slot, st->off) != (ssize_t)sizeof slot) FAIL("layer %d: cannot read ffn_exp_slot", l);
    std::vector<uint8_t> seen(NE, 0);
    for (int e = 0; e < NE; e++) {
        const int s = slot[e];
        XSrc &x = src[e];
        if (s >= 0) {
            if (s >= nh) FAIL("layer %d: expert %d in hot slot %d of %lld", l, e, s, (long long)nh);
            x.type = T_IQ4_XS;
            x.off[0] = hd->off + (size_t)s * XDN; x.off[1] = hg->off + (size_t)s * xgu_bytes(x.type); x.off[2] = hu->off + (size_t)s * xgu_bytes(x.type);
        } else {
            const int c = -1 - s, grp = c >= n0, i = c - grp * (int)n0;
            if (c >= nc) FAIL("layer %d: expert %d in cold slot %d of %lld", l, e, c, (long long)nc);
            x.type = grp ? T_Q2_K : T_Q3_K;
            x.off[0] = cd->off + (size_t)c * XDN; x.off[1] = cg[grp]->off + (size_t)i * xgu_bytes(x.type); x.off[2] = cu[grp]->off + (size_t)i * xgu_bytes(x.type);
        }
        uint8_t &u = seen[s >= 0 ? s : nh + (-1 - s)];
        if (u++) FAIL("layer %d: two experts in slot %d", l, s);
    }
}
