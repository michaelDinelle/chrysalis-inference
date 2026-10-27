#include "engine/metal/flash-next/mtl.h"

struct GemvArgs { uint32_t rows, cols, rb, T, ldx, ldy, add; float scale; uint32_t ggml; };
static const char *tn(int t) {
    switch (t) { case 0: return "f32"; case 1: return "f16"; case 30: return "bf16"; case 42: return "q2_0"; case 2: return "q4_0"; case 6: return "q5_0";
        case 8: return "q8_0"; case 20: return "iq4_nl"; case 23: return "iq4_xs"; case 11: return "q3_k"; case 12: return "q4_k";
        case 13: return "q5_k"; case 14: return "q6_k"; }
    return "?";
}

int main(int argc, char **argv) {
    GGUF g;
    if (argc < 2 || !g.open(argv[1])) FAIL("usage: %s <qwen3.8-flash-next.gguf> [prefix]", argv[0]);
    const std::string pre = argc > 2 ? argv[2] : "gemv";
    Mtl m; m.init();
    const char *names[] = {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.hc_attn_down.weight",
                           "blk.0.hc_attn_up.weight", "blk.0.ffn_router.weight", "blk.0.ssm_ab.weight", "blk.3.attn_q.weight", "output.weight",
                           "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_down_shexp.weight"};
    View x = m.alloc(8 * 10240 * 4), y = m.alloc((size_t)8 * 248320 * 4);
    for (int i = 0; i < 8 * 10240; i++) x.host<float>()[i] = 0.01f * (i % 97);
    for (const char *nm : names) {
        const GGUFTensor *t = g.get(nm);
        const size_t rb = ggml_row_bytes(t->type, t->ne[0]);
        const int rows = (int)t->ne[1], cols = (int)t->ne[0];
        View w = m.alloc(t->nbytes);
        pread(g.fds[t->file], w.host(), t->nbytes, t->off);
        for (int T : {1}) for (int R : {1, 2, 4, 8}) {
            const int lpr = cols >= 2048 ? 32 : 8, per = 4 * 32 / lpr * (lpr == 32 ? R : 2);
            if (lpr == 8 && R != 4) continue;
            const std::string kn = lpr == 8 ? "gemv8_" : R == 4 ? "gemv32_" : "gemvr" + std::to_string(R) + "_";
            GemvArgs a{(uint32_t)rows, (uint32_t)cols, (uint32_t)rb, (uint32_t)T, (uint32_t)cols, (uint32_t)rows, 0, 1.f};
            const int N = std::max(20, (int)(2e9 / t->nbytes));
            double best = 1e9;
            for (int rep = 0; rep < 3; rep++) {
                id<MTLCommandBuffer> cb = [m.q commandBuffer];
                Enc e; e.begin(m, cb);
                for (int i = 0; i < N; i++) {
                    e.k(kn + tn(t->type)).c(0, a).b(1, w).b(2, x).b(3, y).grid((rows + per - 1) / per, 128);
                    e.barrier();
                }
                e.end();
                [cb commit]; [cb waitUntilCompleted];
                best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / N);
            }
            printf("%-30s %-6s %6d x %5d R=%d: %7.1f us  %6.1f GB/s\n", nm, tn(t->type), rows, cols, R, best * 1e6, t->nbytes / best / 1e9);
        }
    }
}
