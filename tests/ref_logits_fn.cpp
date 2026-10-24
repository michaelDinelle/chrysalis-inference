#include "llama.h"
#include "../engine/gguf.h"
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
    if (argc < 4) { fprintf(stderr, "usage: %s model tokens.bin out.bin [n_gpu_layers]\n", argv[0]); return 1; }
    FILE *f = fopen(argv[2], "rb");
    std::vector<llama_token> toks;
    for (int t; fread(&t, 4, 1, f) == 1;) toks.push_back(t);
    fclose(f);
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
    mp.n_gpu_layers = argc > 4 ? atoi(argv[4]) : 0;
    static ggml_backend_dev_t no_devices[] = {nullptr};
    if (mp.n_gpu_layers == 0) mp.devices = no_devices;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    for (auto &[l, _] : layer_ranges) advise(l, POSIX_FADV_DONTNEED);
    ahead(-1);
    auto cp = llama_context_default_params();
    cp.n_ctx = toks.size() + 16;
    cp.n_batch = cp.n_ubatch = toks.size();
    cp.cb_eval = on_eval;
    cp.op_offload = false;
    cp.n_threads = cp.n_threads_batch = 16;
    cp.cb_eval_user_data = nullptr;
    llama_context *ctx = llama_init_from_model(model, cp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch b = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = toks.size();
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    FILE *o = fopen(argv[3], "wb");
    for (size_t i = 0; i < toks.size(); i++) fwrite(llama_get_logits_ith(ctx, i), 4, nv, o);
    fclose(o);
    fprintf(stderr, "wrote %zu x %d logits\n", toks.size(), nv);
    return 0;
}
