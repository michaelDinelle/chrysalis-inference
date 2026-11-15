#include "model.h"
#include "../gpu.h"
#include <stdexcept>
#include "../../gguf.h"
#include <algorithm>
#include <array>
#include <set>
#include <chrono>
#include <cstring>
#include <thread>

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::atomic<size_t> load_read{0}, load_total{0};
std::atomic<int> load_phase{0};
void read_file(const GGUF &g, uint8_t *dst, const uint8_t *src, size_t n) {
    uint64_t off = (uint64_t)(src - g.base);
    while (n) {
        const int64_t r = os::read_at(g.fd, dst, std::min<size_t>(n, (size_t)64 << 20), off);
        if (r <= 0) FAIL("reading the model file failed");
        dst += r; off += r; n -= (size_t)r;
        load_read += (size_t)r;
    }
}
}

std::atomic<int> warm_done{0}, warm_total{0};

float load_progress() {
    if (warm_total > 0) return 0.8f + 0.2f * std::min(1.0f, (float)warm_done / (float)warm_total);
    if (load_phase >= 1) return load_phase == 1 ? 0.74f : load_phase == 2 ? 0.77f : 0.8f;
    return load_total ? 0.72f * std::min(1.0f, (float)load_read / (float)load_total) : 0.0f;
}

namespace {
int physical_cores() {
    const size_t n = os::cores().size();
    return n ? (int)n : (int)std::max(1u, std::max(1u, std::thread::hardware_concurrency()) / 2);
}
}

double Engine::measure_dma() {
    const size_t piece = std::min<size_t>(stage_gu ? stage_gu / STAGE_G : (size_t)1 << 20, cold_bytes / 64);
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    float best_ms = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        CK(hipEventRecord(e0, cst));
        for (int i = 0; i < 48; i++) CK(hipMemcpyAsync(b.raw[i % NSLOT], cold_mem + (size_t)i * 7 * piece % (cold_bytes - piece), piece, hipMemcpyHostToDevice, cst));
        CK(hipEventRecord(e1, cst)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        best_ms = std::min(best_ms, ms);
    }
    CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
    return 48.0 * piece / (best_ms * 1e-3);
}

void Engine::calibrate() {
    const size_t slice = (size_t)64 << 20, nw = std::min<size_t>(n_workers, cold_bytes / slice);
    double best = 1e30;
    for (int rep = 0; rep < 3 && nw; rep++) {
        std::vector<std::thread> th;
        std::vector<uint64_t> sink(nw * 8);
        const double t0 = now();
        for (size_t w = 0; w < nw; w++) th.emplace_back([&, w] {
            const uint64_t *p = (const uint64_t *)(cold_mem + w * slice);
            uint64_t a = 0, b2 = 0, c = 0, d = 0;
            for (size_t i = 0; i < slice / 8; i += 4) { a ^= p[i]; b2 ^= p[i + 1]; c ^= p[i + 2]; d ^= p[i + 3]; }
            sink[w * 8] = a ^ b2 ^ c ^ d;
        });
        for (auto &t : th) t.join();
        best = std::min(best, now() - t0);
    }
    if (nw) cal_cpu_bw = nw * slice / best;
    const ColdLayer *c0 = nullptr;
    for (const ColdLayer &c : cold) if (c.n) { c0 = &c; break; }
    if (c0 && vnni_supported() && vnni_formats(*c0)) {
        const int n = 16;
        std::vector<int8_t> xq((size_t)n * D, 1); std::vector<float> xs((size_t)n * (D / 32), 0.01f), w(n, 0.1f);
        std::vector<int32_t> xsum((size_t)n * (D / 16), 16);
        std::vector<const int8_t *> pq(n); std::vector<const float *> ps(n); std::vector<const int32_t *> pc(n);
        for (int i = 0; i < n; i++) { pq[i] = xq.data() + (size_t)i * D; ps[i] = xs.data() + (size_t)i * (D / 32); pc[i] = xsum.data() + (size_t)i * (D / 16); }
        const int per = 2;
        std::vector<std::vector<uint16_t>> outs(n_workers, std::vector<uint16_t>((size_t)n * D));
        double tb = 1e30;
        for (int rep = 0; rep < 2; rep++) {
            std::vector<std::thread> th;
            const double t0 = now();
            for (int wk = 0; wk < n_workers; wk++) th.emplace_back([&, wk] {
                VnniScratch scr;
                std::vector<uint16_t *> o(n);
                for (int i = 0; i < n; i++) o[i] = outs[wk].data() + (size_t)i * D;
                for (int k = 0; k < per; k++) expert_vnni(*c0, (wk * per + k) % c0->n, n, pq.data(), ps.data(), pc.data(), w.data(), o.data(), scr);
            });
            for (auto &t : th) t.join();
            tb = std::min(tb, now() - t0);
        }
        cal_cpu_mac = (double)n_workers * per * n * (3.0 * FE * D) / tb;
    } else cal_cpu_mac = cal_cpu_bw;
    cal_dma_bw = measure_dma();
    gfrac = (float)(cal_dma_bw / (cal_dma_bw + cal_cpu_bw));
    std::vector<const ColdLayer *> tier;
    for (const ColdLayer &c : cold) if (c.n2) tier.push_back(&c);
    if (file_fd != os::NO_FILE && !tier.empty()) {
        const int reps = 8;
        std::vector<uint8_t> scratch(3 * ((size_t)FE * tier[0]->row_gu[2] + (size_t)D * tier[0]->row_dn));
        double spent = 0;
        for (int i = 0; i < reps; i++) {
            const ColdLayer &c = *tier[(size_t)i * 7 % tier.size()];
            const int h = c.n + (int)((size_t)i * 131 % (size_t)c.n2);
            const uint8_t *const mb[3] = {c.gate[2], c.up[2], c.down2}, *const fb[3] = {c.file_gate, c.file_up, c.file_down2};
            const uint8_t *m[3] = {c.gu_row(false, h, 0), c.gu_row(true, h, 0), c.dn_row(h, 0)}, *src[3];
            const size_t n[3] = {(size_t)FE * c.row_gu[2], (size_t)FE * c.row_gu[2], (size_t)D * c.row_dn};
            uint8_t *dst[3] = {scratch.data(), scratch.data() + n[0], scratch.data() + n[0] + n[1]};
            for (int k = 0; k < 3; k++) {
                src[k] = fb[k] + (m[k] - mb[k]);
                os::forget(file_fd, (uint64_t)(src[k] - file_base), n[k]);
            }
            const double t0 = now();
            file_read3(dst, src, n);
            spent += now() - t0;
        }
        restore_s = spent / reps;
    }
}

double Engine::split_bias(int T) const {
    const int i = T < SPLIT_AT[1] ? 0 : 1;
    const double a = std::log2((double)SPLIT_AT[i]), b = std::log2((double)SPLIT_AT[i + 1]);
    const double f = (std::log2((double)std::clamp(T, SPLIT_AT[0], SPLIT_AT[2])) - a) / (b - a);
    return cpu_bias[i] * std::pow(cpu_bias[i + 1] / cpu_bias[i], std::clamp(f, 0.0, 1.0));
}

std::string Engine::cal_key() const {
    std::string cpu = os::cpu_name();
    cpu.erase(cpu.find_last_not_of(" \n") + 1);
    const long long ram = (long long)(os::ram_total() >> 30);
    return lt_tune::device(lt) + " | " + cpu + " | " + std::to_string(n_workers) + " workers | " + std::to_string(ram) + " GiB | " +
           std::to_string(cold_bytes);
}

bool Engine::cal_load() {
    FILE *f = cal_path.empty() ? nullptr : fopen(cal_path.c_str(), "r");
    if (!f) return false;
    const std::string key = cal_key();
    double v[7], got[7];
    bool found = false;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        const char *tab = strchr(line, '\t');
        if (tab && std::string(line, (size_t)(tab - line)) == key &&
            sscanf(tab + 1, "%lf %lf %lf %lf %lf %lf %lf", &got[0], &got[1], &got[2], &got[3], &got[4], &got[5], &got[6]) == 7 &&
            std::all_of(got, got + 7, [](double g) { return g > 0; })) {
            std::copy(got, got + 7, v);
            found = true;
        }
    }
    fclose(f);
    if (!found) return false;
    cal_cpu_bw = v[0]; cal_cpu_mac = v[1]; cal_dma_bw = v[2]; std::copy(v + 3, v + 6, cpu_bias); restore_s = v[6];
    gfrac = (float)(cal_dma_bw / (cal_dma_bw + cal_cpu_bw));
    calibrated = true;
    return true;
}

void Engine::cal_split(int id) {
    std::vector<int> toks(std::min(smax, SPLIT_AT[2]));
    uint64_t r = 1;
    for (int &t : toks) {
        r = r * 6364136223846793005ull + 1442695040888963407ull;
        t = (int)((r >> 33) % 200000);
    }
    double picked[3], speed[3];
    for (int s = 0; s < 3; s++) {
        const int n = std::min((int)toks.size(), SPLIT_AT[s]);
        double best = 1e30, pick = 1.0;
        for (int i = -1; i < 7; i++) {
            std::fill(cpu_bias, cpu_bias + 3, i < 0 ? 1.0 : std::exp2((i - 2) / 4.0));
            double t = 1e30;
            for (int rep = 0; rep < (n < 1024 ? 3 : 1); rep++) {
                seq_truncate(id, 0);
                const double t0 = now();
                prefill(id, toks.data(), nullptr, n);
                CK(hipStreamSynchronize(st));
                t = std::min(t, now() - t0);
            }
            if (i >= 0 && t < best) best = t, pick = cpu_bias[0];
        }
        picked[s] = pick;
        speed[s] = n / best;
    }
    std::copy(picked, picked + 3, cpu_bias);
    seq_truncate(id, 0);
    cal_save();
    fprintf(stderr, "chrysalis-fn: prefill split measured for this machine: CPU cost x%.2f / x%.2f / x%.2f at %d / %d / %d tokens (%.0f / %.0f / %.0f tok/s), "
            "an expert read back from disk in %.2f ms\n", cpu_bias[0], cpu_bias[1], cpu_bias[2], SPLIT_AT[0], SPLIT_AT[1], SPLIT_AT[2], speed[0], speed[1], speed[2],
            restore_s * 1e3);
}

void Engine::cal_save() {
    calibrated = true;
    if (FILE *f = cal_path.empty() ? nullptr : fopen(cal_path.c_str(), "a")) {
        fprintf(f, "%s\t%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", cal_key().c_str(), cal_cpu_bw, cal_cpu_mac, cal_dma_bw, cpu_bias[0], cpu_bias[1],
                cpu_bias[2], restore_s);
        fclose(f);
    }
}

void Engine::load(const std::string &path, int max_seqs, size_t reserve_bytes) {
    const double t0 = now();
    load_read = 0; load_total = 0; load_phase = 0;
    static GGUF g;
    if (!g.base && !g.open(path.c_str())) FAIL("cannot open %s", path.c_str());
    for (Layer *w = L; w <= &M; w++) *w = Layer{};
    hc_head = HcMix{}; mtp_head = HcMix{};
    if (g.strs["general.architecture"] != "chrysalisfn") FAIL("%s is not a Chrysalis Flash-Next file (engine/flash-next/quantize.py)", path.c_str());
    auto get = [&](const std::string &name) {
        const GGUFTensor *t = g.get(name);
        if (!t) FAIL("missing tensor %s", name.c_str());
        return t;
    };

    CK(hipInit(0));
    CK(hipSetDevice(0));
    hipDeviceProp_t gpu{};
    CK(hipGetDeviceProperties(&gpu, 0));
    if (!gpu_runs()) FAIL("%s cannot run this engine", gpu.name);
    CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    CK(hipStreamCreateWithFlags(&cst, hipStreamNonBlocking));
    CK(hipStreamCreateWithFlags(&lst, hipStreamNonBlocking));
    CK(hipEventCreateWithFlags(&staged, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&copied, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&topk_ready, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&front_ready, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&front_copied, hipEventDisableTiming));
    CK(hipEventRecord(copied, cst));

    struct Up { const GGUFTensor *t; QTensor *q = nullptr; float **f = nullptr; half **h = nullptr; int layer = -1;
                int64_t b0 = -1; };
    std::vector<Up> ups;
    int cur = -1;
    auto quant = [&](const std::string &name, QTensor &q) { ups.push_back({get(name), &q, nullptr, nullptr, cur}); };
    auto vec = [&](const std::string &name, float *&f) { ups.push_back({get(name), nullptr, &f, nullptr, cur}); };
    auto f16 = [&](const std::string &name, half *&h) { ups.push_back({get(name), nullptr, nullptr, &h, cur}); };
    n_slots = 0;
    std::vector<int> first_slot(NL, 0);
    auto hcmix = [&](const std::string &p, HcMix &m, bool inject) {
        vec(p + "norm.weight", m.norm); quant(p + "down.weight", m.down); quant(p + "up.weight", m.up);
        if (inject) quant(p + "inject.weight", m.inject);
    };
    quant("output.weight", out);
    hcmix("output_hc_", hc_head, false);
    std::vector<float *> ssm_a(NGDN), ssm_dt(NGDN);
    for (int l = 0, gl = 0; l <= NL; l++) {
        Layer &w = l < NL ? L[l] : M;
        const std::string p = "blk." + std::to_string(l) + ".";
        cur = l;
        hcmix(p + "hc_attn_", w.hca, true);
        hcmix(p + "hc_ffn_", w.hcf, true);
        if (l == NL || is_attn_layer(l)) {
            quant(p + "attn_q.weight", w.q); quant(p + "attn_k.weight", w.k); quant(p + "attn_v.weight", w.v);
            quant(p + "attn_output.weight", w.o);
            vec(p + "attn_q_norm.weight", w.q_norm); vec(p + "attn_k_norm.weight", w.k_norm);
            quant(p + "indexer.q_proj.weight", w.idx_q); quant(p + "indexer.k_proj.weight", w.idx_k);
            vec(p + "indexer.q_norm.weight", w.idx_qn); vec(p + "indexer.k_norm.weight", w.idx_kn);
        } else {
            quant(p + "attn_qkv.weight", w.qkv); quant(p + "attn_gate.weight", w.z); quant(p + "ssm_out.weight", w.ssm_out);
            f16(p + "ssm_ab.weight", w.Wab);
            vec(p + "ssm_conv1d.weight", w.conv); vec(p + "ssm_norm.weight", w.ssm_norm);
            vec(p + "ssm_a", ssm_a[gl]); vec(p + "ssm_dt.bias", ssm_dt[gl]);
            gl++;
            if (l == PLE_LAYER) {
                quant(p + "ple_key.weight", w.ple_key); quant(p + "ple_value.weight", w.ple_value);
                vec(p + "ple_norm_key.weight", w.ple_nk); vec(p + "ple_norm_query.weight", w.ple_nq);
                vec(p + "ple_norm_conv.weight", w.ple_nc); vec(p + "ple_conv1d.weight", w.ple_conv);
            }
        }
        f16(p + "ffn_router.weight", w.router);
        vec(p + "ffn_gate_inp_shexp.weight", w.sg_w);
        quant(p + "ffn_gate_shexp.weight", w.sh_gate); quant(p + "ffn_up_shexp.weight", w.sh_up); quant(p + "ffn_down_shexp.weight", w.sh_down);
        if (l == NL && g.get(p + "ffn_gate_mtp.weight")) {
            quant(p + "ffn_gate_mtp.weight", w.hot_gate); quant(p + "ffn_up_mtp.weight", w.hot_up); quant(p + "ffn_down_mtp.weight", w.hot_down);
            w.n_hot = (int)get(p + "ffn_gate_mtp.weight")->ne[2];
        }
        if (l < NL) {
            const GGUFTensor *hg = g.get(p + "ffn_gate_hot.weight");
            w.n_hot = hg ? (int)hg->ne[2] : 0;
            if (hg) {
                if (hg->type != IQ4_XS || get(p + "ffn_up_hot.weight")->type != IQ4_XS || get(p + "ffn_down_hot.weight")->type != IQ4_NL)
                    FAIL("layer %d: VRAM experts must be IQ4_XS gate / up, IQ4_NL down", l);
                const int64_t b0 = (int64_t)n_slots * FE * D / 256, d0 = (int64_t)n_slots * D * FE / 32;
                ups.push_back({hg, &pool_gate, nullptr, nullptr, cur, b0});
                ups.push_back({get(p + "ffn_up_hot.weight"), &pool_up, nullptr, nullptr, cur, b0});
                ups.push_back({get(p + "ffn_down_hot.weight"), &pool_down, nullptr, nullptr, cur, d0});
            }
            first_slot[l] = n_slots;
            n_slots += w.n_hot;
            const GGUFTensor *c = g.get(p + "ffn_down_cold.weight");
            w.n_cold = c ? (int)c->ne[2] : 0;
        }
        if (l == NL) {
            quant(p + "nextn.eh_proj.weight", eh_proj);
            vec(p + "nextn.enorm.weight", enorm); vec(p + "nextn.hnorm.weight", hnorm);
            hcmix(p + "nextn.hc_head_", mtp_head, false);
        }
    }
    size_t wbytes = 0;
    auto shape = [](const GGUFTensor *t, int &rows, int &cols) { cols = (int)t->ne[0]; rows = (int)(t->ne[1] * t->ne[2] * t->ne[3]); };
    for (Up &u : ups) {
        if (u.b0 >= 0) continue;
        if (u.q) {
            int rows, cols; shape(u.t, rows, cols);
            Arena dry; dry.cap = SIZE_MAX; make_qtensor(dry, u.t->type, rows, cols); wbytes += dry.used;
        } else wbytes += (u.t->nbytes + 255) & ~size_t(255);
    }
    {
        Arena dry; dry.cap = SIZE_MAX;
        make_qtensor(dry, IQ4_XS, n_slots * FE, D); make_qtensor(dry, IQ4_XS, n_slots * FE, D); make_qtensor(dry, IQ4_NL, n_slots * D, FE);
        dry.get<int>((size_t)NL * NE); dry.get<uint8_t>(n_slots + MAX_EXTRA);
        wbytes += dry.used;
    }
    wbytes += (size_t)NE * 4 + 2 * (size_t)NGDN * GV * 4 + (1 << 20);
    warena.init(wbytes);
    pool_gate = make_qtensor(warena, IQ4_XS, n_slots * FE, D);
    pool_up = make_qtensor(warena, IQ4_XS, n_slots * FE, D);
    pool_down = make_qtensor(warena, IQ4_NL, n_slots * D, FE);
    d_slot_of = warena.get<int>((size_t)NL * NE);
    d_fmt = warena.get<uint8_t>(n_slots + MAX_EXTRA);
    for (int l = 0; l < NL; l++) { L[l].hot_gate = pool_gate; L[l].hot_up = pool_up; L[l].hot_down = pool_down; L[l].slot_of = d_slot_of + (size_t)l * NE; }

    enum { G0, U0, G1, U1, DN };
    std::vector<std::array<const GGUFTensor *, 5>> ct(NL);
    std::vector<std::array<size_t, 5>> coff(NL);
    std::vector<std::tuple<const uint8_t *, size_t, size_t>> copies;
    cold.assign(NL, ColdLayer{});
    cold_bytes = 0;
    for (int l = 0; l < NL; l++) {
        ct[l] = {};
        if (!L[l].n_cold) continue;
        const std::string p = "blk." + std::to_string(l) + ".";
        for (int grp = 0; grp < 2; grp++) {
            const std::string sfx = "_cold" + std::to_string(grp) + ".weight";
            ct[l][2 * grp] = g.get(p + "ffn_gate" + sfx);
            ct[l][2 * grp + 1] = g.get(p + "ffn_up" + sfx);
        }
        ct[l][DN] = get(p + "ffn_down_cold.weight");
        if (!ct[l][G0] || !ct[l][U0]) FAIL("layer %d: missing the cold gate / up experts", l);
        for (int k = 0; k < 5; k++) {
            if (!ct[l][k]) continue;
            coff[l][k] = cold_bytes;
            copies.emplace_back(ct[l][k]->data, ct[l][k]->nbytes, cold_bytes);
            cold_bytes += (ct[l][k]->nbytes + 4095) & ~size_t(4095);
        }
    }
    {
        size_t vram_bytes = 0;
        for (const Up &u : ups) vram_bytes += u.t->nbytes;
        const GGUFTensor *te = g.get("token_embd.weight");
        load_total = cold_bytes + vram_bytes + (te ? te->nbytes : 0);
    }
    std::thread cold_th;
    if (cold_bytes) {
        cold_mem = (uint8_t *)os::anon(cold_bytes);
        if (!cold_mem) FAIL("cannot allocate %.1f GB of system RAM for the experts", cold_bytes / 1e9);
        os::huge(cold_mem, cold_bytes, true);
        cold_th = std::thread([&] {
            for (auto [src, n, off] : copies) {
                read_file(g, cold_mem + off, src, n);
                g.drop(src, n);
            }
        });
    }
    struct Joiner {
        std::thread &th;
        ~Joiner() { if (th.joinable()) th.join(); }
    } joiner{cold_th};

    constexpr size_t STAGE = 64 << 20;
    uint8_t *pin[2], *dev[2];
    hipEvent_t ev[2];
    for (int i = 0; i < 2; i++) {
        CK(hipHostMalloc(&pin[i], STAGE));
        CK(hipMalloc(&dev[i], STAGE));
        CK(hipEventCreateWithFlags(&ev[i], hipEventDisableTiming));
        CK(hipEventRecord(ev[i], st));
    }
    int slot = 0;
    size_t total = 0;
    auto fetch = [&](uint8_t *dst, const uint8_t *src, size_t len) { read_file(g, dst, src, len); };
    auto put = [&](const uint8_t *src, size_t n, auto &&consume) {
        for (size_t off = 0; off < n; off += STAGE) {
            const size_t len = std::min(STAGE, n - off);
            CK(hipEventSynchronize(ev[slot]));
            fetch(pin[slot], src + off, len);
            CK(hipMemcpyAsync(dev[slot], pin[slot], len, hipMemcpyHostToDevice, st));
            consume(dev[slot], off, len);
            CK(hipEventRecord(ev[slot], st));
            slot ^= 1;
        }
        total += n;
    };
    for (Up &u : ups) {
        if (u.q) {
            int rows, cols; shape(u.t, rows, cols);
            if (u.b0 < 0) *u.q = make_qtensor(warena, u.t->type, rows, cols);
            const int bb = qbytes_of(u.t->type);
            if (!bb) FAIL("unsupported type %d for %s", u.t->type, "tensor");
            const size_t piece = STAGE / bb * bb;
            for (size_t off = 0; off < u.t->nbytes; off += piece) {
                const size_t len = std::min(piece, u.t->nbytes - off);
                CK(hipEventSynchronize(ev[slot]));
                fetch(pin[slot], u.t->data + off, len);
                CK(hipMemcpyAsync(dev[slot], pin[slot], len, hipMemcpyHostToDevice, st));
                relayout(*u.q, dev[slot], st, std::max<int64_t>(u.b0, 0) + off / bb, len / bb);
                CK(hipEventRecord(ev[slot], st));
                slot ^= 1;
            }
            g.drop(u.t->data, u.t->nbytes);
            total += u.t->nbytes;
        } else {
            uint8_t *d = warena.get(u.t->nbytes);
            put(u.t->data, u.t->nbytes, [&](uint8_t *src, size_t off, size_t len) {
                CK(hipMemcpyAsync(d + off, src, len, hipMemcpyDeviceToDevice, st));
            });
            g.drop(u.t->data, u.t->nbytes);
            if (u.f) *u.f = (float *)d;
            if (u.h) *u.h = (half *)d;
        }
    }
    gdn_A = warena.get<float>((size_t)NGDN * GV);
    gdn_dt = warena.get<float>((size_t)NGDN * GV);
    for (int gl = 0; gl < NGDN; gl++) {
        CK(hipMemcpyAsync(gdn_A + gl * GV, ssm_a[gl], GV * 4, hipMemcpyDeviceToDevice, st));
        CK(hipMemcpyAsync(gdn_dt + gl * GV, ssm_dt[gl], GV * 4, hipMemcpyDeviceToDevice, st));
    }
    {
        std::vector<int> id(NE);
        for (int e = 0; e < NE; e++) id[e] = e;
        M.slot_of = warena.get<int>(NE);
        CK(hipMemcpyAsync(M.slot_of, id.data(), NE * 4, hipMemcpyHostToDevice, st));
        M.slot_h = id;
        std::vector<int> all((size_t)NL * NE);
        for (int l = 0; l < NL; l++) {
            const int *s = (const int *)get("blk." + std::to_string(l) + ".ffn_exp_slot")->data;
            L[l].slot_h.resize(NE);
            for (int e = 0; e < NE; e++) {
                if (s[e] >= L[l].n_hot || -1 - s[e] >= L[l].n_cold) FAIL("layer %d: expert %d has slot %d", l, e, s[e]);
                L[l].slot_h[e] = all[(size_t)l * NE + e] = s[e] >= 0 ? first_slot[l] + s[e] : s[e];
            }
        }
        CK(hipMemcpyAsync(d_slot_of, all.data(), all.size() * 4, hipMemcpyHostToDevice, st));
        std::vector<uint8_t> f(n_slots, IQ4_XS);
        CK(hipMemcpyAsync(d_fmt, f.data(), f.size(), hipMemcpyHostToDevice, st));
        CK(hipStreamSynchronize(st));
        file_fd = g.fd; file_base = g.base;
    }
    for (int i = 0; i < 2; i++) { CK(hipHostFree(pin[i])); CK(hipFree(dev[i])); CK(hipEventDestroy(ev[i])); }
    const double t_vram = now();

    if (cold_th.joinable()) cold_th.join();
    cold_dev.assign(NL, ColdDec{});
    if (cold_bytes) {
        void *dp = nullptr;
        cold_gpu = hipHostRegister(cold_mem, cold_bytes, hipHostRegisterMapped) == hipSuccess && hipHostGetDevicePointer(&dp, cold_mem, 0) == hipSuccess;
        (void)hipGetLastError();
        d_cold = cold_gpu ? (uint8_t *)dp : nullptr;
    }
    const char *hot_names[3] = {"ffn_gate_hot.weight", "ffn_up_hot.weight", "ffn_down_hot.weight"};
    auto page_up = [](size_t n) { return (n + 4095) & ~size_t(4095); };
    spill_bytes = 0;
    for (int l = 0; l < NL; l++)
        if (L[l].n_hot) for (const char *t : hot_names) spill_bytes += page_up(get("blk." + std::to_string(l) + "." + t)->nbytes);
    if (spill_bytes) {
        spill_mem = (uint8_t *)os::reserve(spill_bytes);
        if (!spill_mem) FAIL("cannot reserve %.1f GB of address space for the spill mirror", spill_bytes / 1e9);
        os::huge(spill_mem, spill_bytes, false);
    }
    size_t spill_at = 0;
    for (int l = 0; l < NL; l++) {
        ColdLayer &c = cold[l];
        if (L[l].n_hot) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const GGUFTensor *t[3] = {get(p + hot_names[0]), get(p + hot_names[1]), get(p + hot_names[2])};
            const uint8_t **mirror[3] = {&c.gate[2], &c.up[2], &c.down2}, **file[3] = {&c.file_gate, &c.file_up, &c.file_down2};
            for (int m = 0; m < 3; m++) { *file[m] = t[m]->data; *mirror[m] = spill_mem + spill_at; spill_at += page_up(t[m]->nbytes); }
            c.type_gu[2] = IQ4_XS; c.row_gu[2] = qrow_bytes(IQ4_XS, D); c.n2 = L[l].n_hot;
            c.type_dn = IQ4_NL; c.row_dn = qrow_bytes(IQ4_NL, FE);
        }
        if (!L[l].n_cold) continue;
        ColdDec &cd = cold_dev[l];
        for (int grp = 0; grp < 2; grp++) {
            const GGUFTensor *tg = ct[l][2 * grp];
            if (!tg) continue;
            c.gate[grp] = cold_mem + coff[l][2 * grp]; c.up[grp] = cold_mem + coff[l][2 * grp + 1];
            c.type_gu[grp] = tg->type; c.row_gu[grp] = qrow_bytes(tg->type, D);
            cd.gate[grp] = d_cold + coff[l][2 * grp]; cd.up[grp] = d_cold + coff[l][2 * grp + 1]; cd.type[grp] = tg->type;
        }
        c.down = cold_mem + coff[l][DN]; c.type_dn = ct[l][DN]->type; c.row_dn = qrow_bytes(c.type_dn, FE);
        cd.down = d_cold + coff[l][DN];
        c.n = L[l].n_cold;
        c.n0 = cd.n0 = (int)ct[l][G0]->ne[2];
        const int n1 = ct[l][G1] ? (int)ct[l][G1]->ne[2] : 0;
        if (c.n0 + n1 != c.n || c.type_dn != IQ4_NL) FAIL("layer %d: cold experts %d + %d of %d, down type %d", l, c.n0, n1, c.n, c.type_dn);
    }
    const double t_cold = now();

    {
        const GGUFTensor *t = get("token_embd.weight");
        if (t->type != Q8_0) FAIL("token embeddings must be Q8_0");
        void *host;
        CK(hipHostMalloc(&host, t->nbytes, hipHostMallocMapped));
        read_file(g, (uint8_t *)host, t->data, t->nbytes);
        g.drop(t->data, t->nbytes);
        embd_host = host;
        void *devp;
        CK(hipHostGetDevicePointer(&devp, host, 0));
        embd.type = Q8_0; embd.rows = (int)t->ne[1]; embd.cols = (int)t->ne[0]; embd.q = (uint8_t *)devp;
    }
    if (!ngram_map) {
        std::string f = g.strs["chrysalisfn.ngram_file"];
        if (f.empty()) FAIL("no n-gram table path in %s", path.c_str());
        if (!os::absolute(f)) f = os::folder(path) + f;
        const os::File fd = os::open_read(f.c_str());
        if (fd == os::NO_FILE) FAIL("cannot open the n-gram table %s", f.c_str());
        ngram_map_bytes = os::size(fd);
        ngram_map = (uint8_t *)os::map(fd, ngram_map_bytes);
        os::close(fd);
        if (!ngram_map) FAIL("cannot map %s", f.c_str());
        os::random(ngram_map, ngram_map_bytes);
        ngram = ngram_map + g.ints["chrysalisfn.ngram_offset"];
        const auto &m = g.arrs["chrysalisfn.ple.layer_multipliers"], &o = g.arrs["chrysalisfn.ple.head_offsets"], &v = g.arrs["chrysalisfn.ple.head_vocab_sizes"];
        if (m.size() < 3 || o.size() < PLE_HEADS || v.size() < PLE_HEADS) FAIL("missing n-gram hash constants");
        for (int i = 0; i < 3; i++) ple_mult[i] = (uint64_t)m[i];
        for (int h = 0; h < PLE_HEADS; h++) { ple_off[h] = (uint32_t)o[h]; ple_voc[h] = (uint32_t)v[h]; }
    }
    {
        auto shared = [](MoeShared *&h, MoeShared *&d) {
            void *p, *dp;
            CK(hipHostMalloc(&p, sizeof(MoeShared), hipHostMallocCoherent | hipHostMallocMapped));
            memset(p, 0, sizeof(MoeShared));
            CK(hipHostGetDevicePointer(&dp, p, 0));
            h = (MoeShared *)p; d = (MoeShared *)dp;
        };
        shared(sh, d_sh);
        shared(sh_pred, d_sh_pred);
        void *p;
        CK(hipHostMalloc(&p, sizeof(MoeEmu), hipHostMallocCoherent | hipHostMallocMapped));
        memset(p, 0, sizeof(MoeEmu));
        me = (MoeEmu *)p;
        CK(hipHostGetDevicePointer((void **)&d_me, p, 0));
        n_workers = std::max(1, physical_cores() - 1);
        cpu.start(sh, sh_pred, cold, n_workers, me);
    }
    hipblasLtCreate(&lt);
    CK(hipMalloc(&lt_ws, lt_ws_size));
    {
        void *p, *dp;
        CK(hipHostMalloc(&p, 64, hipHostMallocMapped));
        memset(p, 0, 64);
        CK(hipHostGetDevicePointer(&dp, p, 0));
        la_lost = (unsigned *)p; d_la_lost = (unsigned *)dp;
    }
    alloc_buffers(max_seqs);
    experts_init();
    size_t free_b, total_b;
    CK(hipMemGetInfo(&free_b, &total_b));
    const size_t want = free_b > reserve_bytes ? free_b - reserve_bytes : 0;
    const size_t max_pages = ((size_t)max_ctx * 3 / 2 + PAGE - 1) / PAGE;
    n_pages = (int)std::min(max_pages, want / KV_PAGE);
    if (n_pages < 64) FAIL("not enough VRAM left for the KV cache");
    CK(hipMalloc(&kv_pool, (size_t)n_pages * KV_PAGE));
    load_phase = 1;
    if (cold_bytes && !cal_load()) calibrate();
    if (cold_bytes && !cold_gpu) {
        gfrac = 0;
        gfrac_pinned = true;
        fprintf(stderr, "chrysalis-fn: this system lets the GPU read less than the %.1f GB of experts kept in RAM: the CPU runs them all\n", cold_bytes / 1e9);
    }
    if (cold_bytes && cpu.can_emulate() && !xfrac_pinned) {
        const ColdLayer *c0 = nullptr;
        for (const ColdLayer &c : cold) if (c.n) { c0 = &c; break; }
        const double dma = measure_dma(), gu = 2.0 * FE * c0->row_gu[0], a = gu / (gu + (double)D * c0->row_dn);
        xfrac = gfrac * (cal_cpu_bw + a * dma) > dma ? (float)std::min(1.0, a * dma / (gfrac * (cal_cpu_bw + a * dma))) : 1.0f;
    }
    free_pages.clear();
    for (int p = 0; p < n_pages; p++) free_pages.push_back(p);
    page_ref.assign(n_pages, 0);
    extra_init();
    fprintf(stderr, "chrysalis-fn: CPU %d workers %.0f GB/s %.0f GMAC/s, DMA %.0f GB/s: decode GPU share of cold picks %.2f, %.2f of it run on the GPU\n",
            n_workers, cal_cpu_bw / 1e9, cal_cpu_mac / 1e9, cal_dma_bw / 1e9, gfrac, xfrac);
    fprintf(stderr, "chrysalis-fn: weights %.2f GB VRAM (%.1f s), cold experts %.2f GB RAM (%.1f s), buffers %.2f GB (passes of %d tokens), "
            "KV pool %d pages = %d tokens (%.2f GB), load %.1f s\n",
            warena.used / 1e9, t_vram - t0, cold_bytes / 1e9, t_cold - t_vram, barena.used / 1e9, smax, n_pages, n_pages * PAGE,
            (double)n_pages * KV_PAGE / 1e9, now() - t0);
    (void)total;
}

void Engine::alloc_buffers(int max_seqs) {
    const int U = umax, S = smax, T8 = 8;
    const size_t big = (size_t)U * HCD, N = (size_t)S * TOPK_E, P = S, MC = std::min(U, MTP_CHUNK);
    stage_gu = stage_soa = 0;
    for (const ColdLayer &c : cold)
        for (int g = 0; g < 3; g++) {
            if (!c.gate[g]) continue;
            Arena dry; dry.cap = SIZE_MAX;
            make_qtensor(dry, c.type_gu[g], STAGE_G * FE, D);
            stage_soa = std::max(stage_soa, dry.used);
            stage_gu = std::max(stage_gu, (size_t)STAGE_G * FE * c.row_gu[g]);
        }
    const size_t raw_bytes = 2 * stage_gu + (size_t)STAGE_G * D * (FE / 32 * 18);
    const int nb = (VOCAB + 4095) / 4096;
    auto carve = [&](Arena &a) {
        auto qact = [&](size_t n) { QAct q; q.q = a.get<int8_t>(n); q.d = a.get<float>(n / 32 + 1); return q; };
        b.tokens = a.get<int>(S);
        b.x = a.get<float>((size_t)U * D);
        b.res = a.get<float>((size_t)S * HCD);
        b.xn = a.get<float>(big);
        b.lo = a.get<float>((size_t)U * HCR);
        b.inv = a.get<float>((size_t)8 * HC);
        b.rr_counter = a.get<unsigned>(64);
        la.lo = a.get<float>((size_t)8 * HCR); la.inv = a.get<float>((size_t)8 * HC);
        la.mx = a.get<float>((size_t)8 * D); la.hh = a.get<half>((size_t)8 * D); la.hq = qact((size_t)8 * D);
        la.logits = a.get<float>((size_t)8 * NE); la.sg = a.get<float>(8); la.wts = a.get<float>(8 * TOPK_E);
        la.ids = a.get<int>(8 * TOPK_E); la.hot = a.get<int>(8 * TOPK_E); la.counter = a.get<unsigned>(64);
        b.idx_raw = a.get<half>((size_t)NKVL * U * IDX_D);
        b.gate = a.get<float>(big);
        b.inj_a = a.get<float>((size_t)U * HC);
        b.inj_f = a.get<float>((size_t)S * HC);
        b.mx = a.get<float>((size_t)U * D);
        b.proj = a.get<float>((size_t)U * std::max(2 * NH * HD, QKV));
        b.zb = a.get<float>((size_t)U * GV * GD);
        b.kb = a.get<float>((size_t)U * std::max(NKV * HD, 2 * GV));
        b.vb = a.get<float>((size_t)U * NKV * HD);
        b.qidx = a.get<float>((size_t)U * IDX_H * IDX_D);
        b.kidx = a.get<float>((size_t)U * IDX_D);
        b.blk = a.get<float>((size_t)U * D);
        b.xn_h = a.get<half>(big);
        b.lo_h = a.get<half>((size_t)U * HCR);
        b.hh = a.get<half>((size_t)U * D);
        b.Q16 = a.get<half>((size_t)U * NH * HD);
        b.qidx16 = a.get<half>((size_t)U * IDX_H * IDX_D);
        b.att_h = a.get<half>((size_t)U * NH * HD);
        b.deq = a.get<half>(DEQ_ELEMS);
        b.gemm16 = a.get<half>((size_t)U * 12288);
        b.xn_q = qact(big);
        b.lo_q = qact((size_t)U * HCR);
        b.hq = qact((size_t)U * D);
        b.mq = qact((size_t)U * NH * HD);
        b.part_o = a.get<float>((size_t)PART_ROWS * NH * HD);
        b.part_ml = a.get<float>((size_t)PART_ROWS * NH * 2);
        b.score = a.get<float>((size_t)SEL_G * (max_ctx / KPOOL + 1));
        b.sel_hist = a.get<uint32_t>((size_t)SEL_G * 1024);
        b.sel = a.get<int>((size_t)U * (SEL_BLOCKS + 1));
        b.qn = a.get<float>((size_t)U * GK * GD);
        b.kn = a.get<float>((size_t)U * GK * GD);
        b.vc = a.get<float>((size_t)U * GV * GD);
        b.v_raw = a.get<float>((size_t)NGDN * 8 * QKV);
        b.v_kn = a.get<float>((size_t)NGDN * 8 * GK * GD);
        b.v_vc = a.get<float>((size_t)NGDN * 8 * GV * GD);
        b.v_ab = a.get<float>((size_t)NGDN * 8 * 2 * GV);
        b.logits_e = a.get<float>((size_t)T8 * NE);
        b.sg = a.get<float>(T8);
        b.wts = a.get<float>((size_t)T8 * TOPK_E);
        b.ids = a.get<int>((size_t)T8 * TOPK_E);
        b.hot = a.get<int>((size_t)T8 * TOPK_E);
        b.y = a.get<float>((size_t)T8 * (TOPK_E + 1) * D);
        b.moe = a.get<float>((size_t)S * D);
        b.hq_e = qact((size_t)T8 * (TOPK_E + 1) * FE);
        b.d_ple_rows = a.get<uint8_t>((size_t)U * PLE_HEADS * 170);
        b.emb = a.get<float>((size_t)U * D);
        b.value = a.get<float>((size_t)U * D);
        b.emb_h = a.get<half>((size_t)U * D);
        b.emb_q = qact((size_t)U * D);
        b.logits = a.get<float>((size_t)T8 * VOCAB);
        b.tsv = a.get<float>((size_t)T8 * nb * TOPK);
        b.tsi = a.get<int>((size_t)T8 * nb * TOPK);
        b.topv = a.get<float>((size_t)T8 * TOPK);
        b.topi = a.get<int>((size_t)T8 * TOPK);
        b.tops = a.get<float>((size_t)T8 * LOGIT_STATS);
        b.mtp_res = a.get<float>(MC * HCD);
        b.mtp_e = a.get<float>(MC * D);
        b.mtp_in = a.get<float>((size_t)8 * HC * 2 * D);
        b.mtp_in_h = a.get<half>(MC * HC * 2 * D);
        b.mtp_in_q = qact((size_t)8 * 2 * D);
        b.draft = a.get<int>(16);
        b.rlog = a.get<float>((size_t)U * NE);
        b.sgT = a.get<float>(S);
        b.wtsT = a.get<float>(N); b.ent_w = a.get<float>(N);
        b.idsT = a.get<int>(N); b.ent_tok = a.get<int>(N); b.pos = a.get<int>(N); b.btok = a.get<int>(N);
        b.work = a.get<MoeWork>(N / 64 + 2 * NE);
        b.hmoe = a.get<half>((size_t)S * D);
        const size_t xy = std::max(P * D * 2, (size_t)U * HCD * 4);
        b.Xs = (half *)a.get<uint8_t>(xy); b.Hs = a.get<half>(P * FE); b.Ys = (half *)a.get<uint8_t>(xy);
        b.key = (float *)b.Xs; b.xconv = (float *)b.Ys;
        for (int k = 0; k < NSLOT; k++) {
            b.raw[k] = a.get<uint8_t>(raw_bytes);
            b.sgu[k] = a.get<uint8_t>(2 * stage_soa);
            b.sd_[k] = make_qtensor(a, IQ4_NL, STAGE_G * D, FE);
        }
        b.img = a.get<const half *>(S);
        b.rpos = a.get<int>((size_t)S * 3);
    };
    Arena dry; dry.cap = SIZE_MAX;
    carve(dry);
    barena.init(dry.used + (1 << 20));
    carve(barena);
    CK(hipMemsetAsync(b.rr_counter, 0, 4, st));
    CK(hipMemsetAsync(la.counter, 0, 64 * 4, st));
    CK(hipHostMalloc(&b.ple_rows, (size_t)S * PLE_HEADS * 170));
    CK(hipHostMalloc(&b.h_ids, N * 4)); CK(hipHostMalloc(&b.h_wts, N * 4)); CK(hipHostMalloc(&b.h_tok, N * 4));
    CK(hipHostMalloc(&b.h_entw, N * 4)); CK(hipHostMalloc(&b.h_pos, N * 4)); CK(hipHostMalloc(&b.h_btok, N * 4));
    CK(hipHostMalloc(&b.h_work, (N / 64 + 2 * NE) * sizeof(MoeWork)));
    CK(hipHostMalloc(&b.h_x16, (size_t)S * D * 2)); CK(hipHostMalloc(&b.h_cout, P * D * 2));
    CK(hipHostMalloc(&b.h_topv, (size_t)8 * TOPK * 4)); CK(hipHostMalloc(&b.h_topi, (size_t)8 * TOPK * 4)); CK(hipHostMalloc(&b.h_draft, 64));
    CK(hipHostMalloc(&b.h_tops, (size_t)8 * LOGIT_STATS * 4));
    for (int k = 0; k < NSLOT; k++) {
        CK(hipHostMalloc(&b.bounce[k], raw_bytes));
        CK(hipEventCreateWithFlags(&raw_free[k], hipEventDisableTiming));
        CK(hipEventCreateWithFlags(&raw_ready[k], hipEventDisableTiming));
        CK(hipEventRecord(raw_free[k], st));
    }
    seqs.assign(max_seqs, Seq{});
    for (Seq &s : seqs) {
        CK(hipMalloc(&s.d_pages, ((size_t)max_ctx * 2 / PAGE + 8) * 4));
        CK(hipMalloc(&s.ssm, ssm_bytes()));
        CK(hipMalloc(&s.conv, conv_bytes()));
        CK(hipMalloc(&s.ple, ple_bytes()));
        CK(hipMalloc(&s.h_last, (size_t)HCD * 4));
        CK(hipMalloc(&s.idx_tail, tail_bytes()));
    }
    CK(hipMalloc(&stage_state, state_bytes()));
}

void Engine::warmup() {
    load_phase = 2;
    struct Done { ~Done() { load_phase = 3; } } done_;
    snap_reserve_start();
    int id = seq_new();
    const bool borrowed = id < 0;
    if (borrowed) id = 0;
    std::vector<int> toks(24, 198);
    const bool on = xc.on;
    xc.on = false;
    prefill(id, toks.data(), nullptr, (int)toks.size());
    Sampling sp;
    sp.temperature = 0.0f; sp.n_draft = 1;
    int out[4];
    generate(id, sp, out, 4);
    CK(hipStreamSynchronize(st));
    if (!calibrated && cold_bytes) cal_split(id);
    xc.on = on;
    std::fill(xc.fresh.begin(), xc.fresh.end(), 0u);
    std::fill(xc.total.begin(), xc.total.end(), 0ull);
    xc.tokens = 0; xc.counts_pending = false;
    CK(hipMemsetAsync(xc.d_counts, 0, (size_t)NL * NE * 4, st));
    if (borrowed) seq_reset(seqs[0]); else seq_free(id);
    snap_warm = true;
    snap_reserve_finish();
}

void Engine::shutdown() {
    cpu.stop();
    experts_stop();
    CK(hipDeviceSynchronize());
    if (kv_pool) (void)hipFree(kv_pool);
    for (Seq &s : seqs) { (void)hipFree(s.d_pages); (void)hipFree(s.ssm); (void)hipFree(s.conv); (void)hipFree(s.ple); (void)hipFree(s.h_last); (void)hipFree(s.idx_tail); }
    seqs.clear();
    snap_reserve_finish();
    auto in_region = [&](const void *p) {
        for (auto &r : snap_regions) if (p >= r.first && (const char *)p < (const char *)r.first + r.second) return true;
        return false;
    };
    std::vector<void *> own;
    for (Snap &sn : snaps) if (sn.host && !in_region(sn.host)) own.push_back(sn.host);
    for (void *p : snap_spare) if (!in_region(p)) own.push_back(p);
    for (void *p : own) { (void)hipHostUnregister(p); os::release(p, (state_bytes() + (2 << 20) - 1) & ~size_t((2 << 20) - 1)); }
    for (auto &r : snap_regions) { (void)hipHostUnregister(r.first); os::release(r.first, r.second); }
    snaps.clear(); snap_spare.clear(); snap_regions.clear();
    if (warena.base) (void)hipFree(warena.base);
    if (barena.base) (void)hipFree(barena.base);
    warena = Arena{}; barena = Arena{};
    for (void *p : {(void *)b.ple_rows, (void *)b.h_ids, (void *)b.h_wts, (void *)b.h_tok, (void *)b.h_entw, (void *)b.h_pos, (void *)b.h_btok,
                    (void *)b.h_work, (void *)b.h_x16, (void *)b.h_cout, (void *)b.h_topv, (void *)b.h_tops, (void *)b.h_topi, (void *)b.h_draft, (void *)sh,
                    (void *)sh_pred, (void *)me})
        if (p) (void)hipHostFree(p);
    if (embd.q) { void *h = nullptr; (void)hipHostGetDevicePointer(&h, embd.q, 0); }
    if (embd_host) (void)hipHostFree(embd_host);
    if (la_lost) (void)hipHostFree(la_lost);
    la_lost = d_la_lost = nullptr;
    embd_host = nullptr; embd.q = nullptr; sh = d_sh = nullptr; sh_pred = d_sh_pred = nullptr; me = d_me = nullptr;
    if (stage_state) (void)hipFree(stage_state);
    if (d_mask) (void)hipFree(d_mask);
    d_mask = nullptr;
    for (uint8_t *&p : b.bounce) { if (p) (void)hipHostFree(p); p = nullptr; }
    if (lt_ws) (void)hipFree(lt_ws);
    if (lt) hipblasLtDestroy(lt);
    lt = nullptr; lt_ws = nullptr; stage_state = nullptr; kv_pool = nullptr;
    lt_plans.clear();
    free_pages.clear(); page_ref.clear();
    if (cold_mem && cold_gpu) (void)hipHostUnregister(cold_mem);
    (void)hipGetLastError();
    (void)hipStreamDestroy(st); (void)hipStreamDestroy(cst); (void)hipStreamDestroy(lst);
    os::release(cold_mem, cold_bytes);
    os::release(spill_mem, spill_bytes);
    os::unmap(ngram_map, ngram_map_bytes);
    cold_mem = nullptr; spill_mem = nullptr; ngram_map = nullptr; cold_bytes = spill_bytes = 0; d_cold = nullptr; cold_gpu = false;
    cold.clear(); cold_dev.clear();
}
