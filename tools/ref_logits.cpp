// Reference logits from llama.cpp for a token sequence: writes float32 [n][vocab] (all positions).
// Build: see tools/compare_ref.py. Usage: ref_logits model.gguf tokens.bin out.bin
#include "llama.h"
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model tokens.bin out.bin\n", argv[0]); return 1; }
    FILE *f = fopen(argv[2], "rb");
    std::vector<llama_token> toks;
    for (int t; fread(&t, 4, 1, f) == 1;) toks.push_back(t);
    fclose(f);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params();
    cp.n_ctx = toks.size() + 16;
    cp.n_batch = cp.n_ubatch = toks.size();
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
