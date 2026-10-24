#include "llama.h"
#include "../engine_fn/gguf.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct Range { int fd; size_t off, len; };
static std::map<int, std::vector<Range>> layer_ranges;
static int prefetched = -1;

static void advise(int layer, int how) {
    auto it = layer_ranges.find(layer);
    if (it == layer_ranges.end()) return;
    for (const Range &r : it->second) posix_fadvise(r.fd, (off_t)r.off, (off_t)r.len, how);
}
static void ahead(int done) {
    for (int l = std::max(prefetched + 1, done + 1); l <= done + 3; l++) { advise(l, POSIX_FADV_WILLNEED); prefetched = l; }
    if (done >= 0) advise(done, POSIX_FADV_DONTNEED);
}

static bool on_eval(struct ggml_tensor *t, bool ask, void *) {
    const char *n = ggml_get_name(t);
    if (strncmp(n, "l_last-", 7) != 0) return false;
    if (!ask) ahead(atoi(n + 7));
    return true;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model tokens.i32 pos.i32 embd.f32 out.f32\n", argv[0]); return 1; }
    auto slurp = [](const char *path, size_t elem) {
        std::vector<uint8_t> v;
        if (FILE *f = fopen(path, "rb")) { fseek(f, 0, SEEK_END); v.resize(ftell(f)); fseek(f, 0, SEEK_SET); fread(v.data(), 1, v.size(), f); fclose(f); }
        v.resize(v.size() / elem * elem);
        return v;
    };
    const auto tb = slurp(argv[2], 4), pb = slurp(argv[3], 4), eb = slurp(argv[4], 4);
    const int32_t *toks = (const int32_t *)tb.data(), *pos = (const int32_t *)pb.data();
    const float *embd = (const float *)eb.data();
    const int n = (int)(tb.size() / 4);
    if ((int)(pb.size() / 4) != 3 * n) { fprintf(stderr, "positions do not match the tokens\n"); return 1; }
    std::string first = argv[1];
    const size_t at = first.find("-00001-of-");
    const int nsplit = at == std::string::npos ? 1 : atoi(first.c_str() + at + 10);
    static std::vector<GGUF> splits(nsplit);
    for (int i = 0; i < nsplit; i++) {
        std::string p = first;
        if (at != std::string::npos) { char num[8]; snprintf(num, sizeof num, "%05d", i + 1); p.replace(at + 1, 5, num); }
        if (!splits[i].open(p.c_str())) { fprintf(stderr, "cannot open %s\n", p.c_str()); return 1; }
        for (auto &[name, t] : splits[i].tensors) {
            if (name.rfind("blk.", 0) != 0) continue;
            const int l = atoi(name.c_str() + 4);
            layer_ranges[l].push_back({splits[i].fd, (size_t)(t.data - splits[i].base), t.nbytes});
        }
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    static ggml_backend_dev_t no_devices[] = {nullptr};
    mp.devices = no_devices;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    for (auto &[l, _] : layer_ranges) advise(l, POSIX_FADV_DONTNEED);
    ahead(-1);
    auto cp = llama_context_default_params();
    cp.n_ctx = (size_t)n + 16;
    cp.n_batch = cp.n_ubatch = (size_t)n;
    cp.cb_eval = on_eval;
    cp.op_offload = false;
    cp.n_threads = cp.n_threads_batch = 16;
    cp.cb_eval_user_data = nullptr;
    llama_context *ctx = llama_init_from_model(model, cp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model)), ne = llama_model_n_embd_inp(model);
    std::vector<float> logits((size_t)n * nv);
    llama_batch_ext *b = llama_batch_ext_init(ctx);
    int used = 0;
    for (int i = 0; i < n;) {
        const bool img = toks[i] >= (1 << 30);
        int j = i;
        while (j < n && (toks[j] >= (1 << 30)) == img) j++;
        llama_batch_ext_clear(b);
        for (int k = i; k < j; k++) {
            int idx;
            if (img) {
                if ((size_t)(used + 1) * ne * 4 > eb.size()) { fprintf(stderr, "not enough embeddings\n"); return 1; }
                idx = llama_batch_ext_add_embd(b, 0, llama_embd{embd + (size_t)used++ * ne, 1, (size_t)ne});
                const llama_pos p[4] = {pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2], 0};
                llama_batch_ext_set_pos(b, idx, p);
            } else {
                idx = llama_batch_ext_add_token(b, 0, toks[k]);
                const llama_pos p = pos[k * 3];
                llama_batch_ext_set_pos(b, idx, &p);
            }
            llama_batch_ext_set_output_logits(b, idx, true);
        }
        prefetched = -1;
        ahead(-1);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int k = i; k < j; k++) {
            const float *l = llama_get_logits_ith(ctx, k - i);
            std::copy(l, l + nv, logits.begin() + (size_t)k * nv);
        }
        i = j;
    }
    FILE *o = fopen(argv[5], "wb");
    fwrite(logits.data(), 4, logits.size(), o);
    fclose(o);
    fprintf(stderr, "wrote %d x %d logits\n", n, nv);
    return 0;
}
