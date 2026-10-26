#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model tokens.bin out.bin [positions] [ubatch]\n", argv[0]); return 1; }
    FILE *f = fopen(argv[2], "rb");
    std::vector<llama_token> toks;
    for (int t; fread(&t, 4, 1, f) == 1;) toks.push_back(t);
    fclose(f);
    std::vector<int> want;
    if (argc > 4 && strcmp(argv[4], "all") != 0)
        for (char *p = strtok(argv[4], ","); p; p = strtok(nullptr, ",")) want.push_back(atoi(p));
    else
        for (size_t i = 0; i < toks.size(); i++) want.push_back((int)i);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params();
    cp.n_ctx = toks.size() + 16;
    cp.n_batch = toks.size();
    cp.n_ubatch = argc > 5 ? std::min<size_t>(toks.size(), (size_t)atoi(argv[5])) : toks.size();
    llama_context *ctx = llama_init_from_model(model, cp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<char> keep(toks.size(), 0);
    for (int w : want) keep[w] = 1;
    llama_batch b = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = keep[i];
    }
    b.n_tokens = toks.size();
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    FILE *o = fopen(argv[3], "wb");
    for (int w : want) fwrite(llama_get_logits_ith(ctx, w), 4, nv, o);
    fclose(o);
    fprintf(stderr, "wrote %zu x %d logits\n", want.size(), nv);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
