#include "llama.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <typename T> static std::vector<T> slurp(const char *path) {
    std::vector<T> v;
    FILE *f = fopen(path, "rb");
    if (!f) return v;
    for (T x; fread(&x, sizeof(T), 1, f) == 1;) v.push_back(x);
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model tokens.i32 pos.i32 embd.f32 out.f32\n", argv[0]); return 1; }
    const auto toks = slurp<int32_t>(argv[2]), pos = slurp<int32_t>(argv[3]);
    const auto embd = slurp<float>(argv[4]);
    const int n = (int)toks.size();
    if ((int)pos.size() != 3 * n) { fprintf(stderr, "positions do not match the tokens\n"); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s with every layer on the GPU\n", argv[1]); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = n + 16;
    cp.n_batch = n;
    cp.n_ubatch = getenv("REF_UBATCH") ? atoi(getenv("REF_UBATCH")) : n;
    if (getenv("REF_KV_Q")) {
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q5_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
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
                if ((size_t)(used + 1) * ne > embd.size()) { fprintf(stderr, "not enough embeddings\n"); return 1; }
                idx = llama_batch_ext_add_embd(b, 0, llama_embd{embd.data() + (size_t)used++ * ne, 1, (size_t)ne});
                const llama_pos p[4] = {pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2], 0};
                llama_batch_ext_set_pos(b, idx, p);
            } else {
                idx = llama_batch_ext_add_token(b, 0, toks[k]);
                const llama_pos p = pos[k * 3];
                llama_batch_ext_set_pos(b, idx, &p);
            }
            llama_batch_ext_set_output_logits(b, idx, true);
        }
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
