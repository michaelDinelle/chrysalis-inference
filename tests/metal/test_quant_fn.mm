#include "engine/metal/flash-next/mtl.h"
#include <ggml.h>
#include <cmath>
#include <random>

static const char *tname(int t) {
    switch (t) {
        case 0: return "f32"; case 1: return "f16"; case 30: return "bf16"; case 42: return "q2_0"; case 2: return "q4_0";
        case 6: return "q5_0"; case 8: return "q8_0"; case 20: return "iq4_nl"; case 23: return "iq4_xs"; case 11: return "q3_k";
        case 12: return "q4_k"; case 13: return "q5_k"; case 14: return "q6_k"; case 10: return "q2_k";
    }
    return nullptr;
}

struct GemvArgs { uint32_t rows, cols, rb, T, ldx, ldy, add; float scale; uint32_t ggml; };

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <qwen3.8-flash-next.gguf>\n", argv[0]); return 2; }
    GGUF g;
    if (!g.open(argv[1])) FAIL("cannot open %s", argv[1]);
    Mtl m;
    m.init();
    std::map<int, std::string> pick;
    for (auto &[n, t] : g.tensors) {
        if (!tname(t.type) || t.ne[0] % 32) continue;
        auto it = pick.find(t.type);
        if (it == pick.end() || (t.ne[0] >= 640 && g.tensors[it->second].ne[0] < 640)) pick[t.type] = n;
    }
    int failed = 0, checked = 0;
    std::mt19937 rng(1);
    for (auto &[type, name] : pick) {
        const GGUFTensor &t = g.tensors[name];
        const int cols = (int)t.ne[0];
        const int rows = (int)std::min<int64_t>(64, t.ne[1] * t.ne[2]);
        const size_t rb = ggml_row_bytes(t.type, cols);
        View w = m.alloc(rb * rows), out = m.alloc((size_t)rows * cols * 4);
        if (pread(g.fds[t.file], w.host(), rb * rows, t.off) != (ssize_t)(rb * rows)) FAIL("read %s", name.c_str());
        std::vector<float> ref((size_t)rows * cols);
        const ggml_type_traits *tr = ggml_get_type_traits((ggml_type)t.type);
        if (!tr->to_float && t.type != 0) FAIL("ggml has no to_float for type %d", t.type);
        for (int r = 0; r < rows; r++) {
            const uint8_t *src = w.host<uint8_t>() + r * rb;
            if (t.type == 0) memcpy(&ref[(size_t)r * cols], src, cols * 4);
            else tr->to_float(src, &ref[(size_t)r * cols], cols);
        }
        const int NT = 3;
        View x = m.alloc((size_t)NT * cols * 4), y = m.alloc((size_t)NT * rows * 4);
        for (int i = 0; i < NT * cols; i++) x.host<float>()[i] = std::uniform_real_distribution<float>(-1, 1)(rng);
        id<MTLCommandBuffer> cb = [m.q commandBuffer];
        Enc e; e.begin(m, cb);
        uint32_t a4[4] = {(uint32_t)cols, (uint32_t)rb, (uint32_t)rows, 0};
        e.k(std::string("deq_rows_") + tname(type)).c(0, a4).b(1, w).b(2, out).grid(cols / 32, rows, 1, 1);
        e.end();
        [cb commit]; [cb waitUntilCompleted];
        int bad = 0;
        for (size_t i = 0; i < ref.size(); i++) if (memcmp(&ref[i], &out.host<float>()[i], 4)) bad++;
        checked++;
        if (bad) { failed++; fprintf(stderr, "FAIL deq %-6s %s: %d of %zu values differ (first: %g vs %g)\n", tname(type), name.c_str(), bad, ref.size(), ref[0], out.host<float>()[0]); }
        for (int lpr : {32, 8}) for (int T : {1, 3}) {
            GemvArgs ga{(uint32_t)rows, (uint32_t)cols, (uint32_t)rb, (uint32_t)T, (uint32_t)cols, (uint32_t)rows, 0, 1.0f};
            cb = [m.q commandBuffer];
            e.begin(m, cb);
            const int rows_per_tg = 4 * (32 / lpr) * (lpr == 32 ? 4 : 2);
            e.k(std::string("gemv") + std::to_string(lpr) + (T == 1 ? "_" : "t_") + tname(type)).c(0, ga).b(1, w).b(2, x).b(3, y)
                .grid((rows + rows_per_tg - 1) / rows_per_tg, 128);
            e.end();
            [cb commit]; [cb waitUntilCompleted];
            double worst = 0;
            for (int tt = 0; tt < T; tt++) for (int r = 0; r < rows; r++) {
                double s = 0, mag = 0;
                for (int c = 0; c < cols; c++) { s += (double)ref[(size_t)r * cols + c] * x.host<float>()[tt * cols + c]; mag += fabs(ref[(size_t)r * cols + c] * x.host<float>()[tt * cols + c]); }
                worst = std::max(worst, fabs(y.host<float>()[tt * rows + r] - s) / (mag + 1e-30));
            }
            checked++;
            if (!(worst < 1e-4)) { failed++; fprintf(stderr, "FAIL gemv%d %-6s T=%d: relative error %.3g\n", lpr, tname(type), T, worst); }
        }
        printf("%-6s %-36s %5d cols: dequant %s\n", tname(type), name.c_str(), cols, bad ? "DIFFERS" : "bit-identical");
    }
    if (pick.size() < 8) { fprintf(stderr, "FAIL: only %zu of 8 types found in the file\n", pick.size()); failed++; }
    printf("%d of %d checks failed\n", failed, checked);
    return failed ? 1 : 0;
}
