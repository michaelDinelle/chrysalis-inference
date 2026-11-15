#include "model.h"
#include "../gpu.h"
#include "../../gguf.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <emmintrin.h>
#include <thread>

namespace {

constexpr size_t PIECE = 32 << 20;
constexpr size_t RING = 256 << 20;
constexpr size_t SLAB = 1 << 30;
constexpr int WRITERS = 3;

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <typename T> void wait_for(const std::atomic<T> &a, T v) {
    for (T cur = a.load(std::memory_order_acquire); cur < v; cur = a.load(std::memory_order_acquire))
        a.wait(cur, std::memory_order_acquire);
}
template <typename T> void publish(std::atomic<T> &a, T v) { a.store(v, std::memory_order_release); a.notify_all(); }

void stream_copy(uint8_t *dst, const uint8_t *src, size_t n) {
    size_t i = 0;
    for (; i + 64 <= n; i += 64)
        for (int k = 0; k < 64; k += 16)
            _mm_stream_si128((__m128i *)(dst + i + k), _mm_loadu_si128((const __m128i *)(src + i + k)));
    if (i < n) memcpy(dst + i, src + i, n - i);
    _mm_sfence();
}

struct Job {
    enum Kind { QUANT, VEC, ALPHA, BETA, GDN_A, GDN_DT } kind;
    const GGUFTensor *t;
    QTensor *q = nullptr;
    float **vec = nullptr;
    QTensor made{};
    int gl = 0;
    bool carve = false;
    int slab = -1;
    size_t off = 0, bytes = 0;
    int unit = 4;
};
struct Piece { int job; size_t src, len, ring; int dep; };

int block_bytes(int type) { return type == Q8_0 ? 34 : type == Q6_K ? 210 : type == IQ4_XS ? 136 : 4; }

}

void Engine::add_kv_memory(uint8_t *p, size_t bytes) {
    uint8_t *base = kv_pool ? kv_pool : p;
    size_t shift = 0;
    if (p < base) {
        shift = (size_t)(base - p) / KV_PAGE;
        base -= shift * KV_PAGE;
    }
    const size_t first = p > base ? ((size_t)(p - base) + KV_PAGE - 1) / KV_PAGE : 0;
    const size_t last = (size_t)(p + bytes - base) / KV_PAGE;
    const size_t top = std::max(last, free_pages.empty() ? 0 : *std::max_element(free_pages.begin(), free_pages.end()) + 1 + shift);
    if (top > (1u << 24)) { fprintf(stderr, "chrysalis: KV segment too far from the pool, not used\n"); return; }
    kv_pool = base;
    for (int &id : free_pages) id += (int)shift;
    for (size_t i = first; i < last; i++) free_pages.push_back((int)i);
    std::sort(free_pages.begin(), free_pages.end(), std::greater<int>());
    n_pages = (int)free_pages.size();
    page_ref.assign(free_pages.empty() ? 0 : free_pages.front() + 1, 0);
}

static std::atomic<size_t> load_done{0}, load_total{0};
static std::atomic<int> load_phase{0};
float load_progress() {
    if (load_phase >= 3) return 1.0f;
    if (load_phase == 2) return 0.97f;
    return load_total ? 0.95f * (float)load_done / (float)load_total : 0.0f;
}

void Engine::load(const std::string &path, int max_seqs, size_t reserve_bytes) {
    load_done = 0; load_total = 0; load_phase = 0;
    const double t0 = now();
    GGUF g;
    if (!g.open(path.c_str())) FAIL("cannot open %s", path.c_str());
    if (g.ints["qwen35.block_count"] != NL + 1 || g.ints["qwen35.embedding_length"] != D)
        FAIL("%s is not a Qwen3.8-27B GGUF with an MTP layer", path.c_str());
    {
        const GGUFTensor *t = g.get("token_embd.weight");
        if (!t || t->type != IQ4_XS) FAIL("token embeddings must be IQ4_XS");
        model_path = path;
        embd_offset = t->data - g.base;
        embd.type = IQ4_XS; embd.rows = (int)t->ne[1]; embd.cols = (int)t->ne[0];
    }

    std::vector<Job> jobs;
    auto get = [&](const std::string &name) {
        const GGUFTensor *t = g.get(name);
        if (!t) FAIL("missing tensor %s", name.c_str());
        return t;
    };
    auto quant = [&](const std::string &name, QTensor &q) { Job j{Job::QUANT, get(name)}; j.q = &q; jobs.push_back(j); };
    auto vec = [&](const std::string &name, float *&v) { Job j{Job::VEC, get(name)}; j.vec = &v; jobs.push_back(j); };
    quant("output.weight", out);
    vec("output_norm.weight", out_norm);
    for (int l = 0, gl = 0; l <= NL; l++) {
        Layer &w = l < NL ? L[l] : M;
        const std::string p = "blk." + std::to_string(l) + ".";
        vec(p + "attn_norm.weight", w.attn_norm);
        vec(p + "post_attention_norm.weight", w.post_norm);
        quant(p + "ffn_gate.weight", w.ffn_gate); quant(p + "ffn_up.weight", w.ffn_up); quant(p + "ffn_down.weight", w.ffn_down);
        if (l == NL || is_attn_layer(l)) {
            quant(p + "attn_q.weight", w.q); quant(p + "attn_k.weight", w.k);
            quant(p + "attn_v.weight", w.v); quant(p + "attn_output.weight", w.o);
            vec(p + "attn_q_norm.weight", w.q_norm); vec(p + "attn_k_norm.weight", w.k_norm);
        } else {
            quant(p + "attn_qkv.weight", w.qkv); quant(p + "attn_gate.weight", w.z); quant(p + "ssm_out.weight", w.ssm_out);
            vec(p + "ssm_conv1d.weight", w.conv); vec(p + "ssm_norm.weight", w.ssm_norm);
            for (auto [name, kind] : {std::pair{"ssm_alpha.weight", Job::ALPHA}, {"ssm_beta.weight", Job::BETA}}) {
                Job j{kind, get(p + name)};
                if (j.t->type != F32 || j.t->ne[0] * j.t->ne[1] != (int64_t)GV * D) FAIL("unexpected %s", (p + name).c_str());
                j.q = &w.ab;
                jobs.push_back(j);
            }
            for (auto [name, kind] : {std::pair{"ssm_a", Job::GDN_A}, {"ssm_dt.bias", Job::GDN_DT}}) {
                Job j{kind, get(p + name)};
                j.gl = gl;
                jobs.push_back(j);
            }
            gl++;
        }
    }
    const std::string p = "blk." + std::to_string(NL) + ".";
    quant(p + "nextn.eh_proj.weight", eh_proj);
    vec(p + "nextn.enorm.weight", enorm); vec(p + "nextn.hnorm.weight", hnorm);
    vec(p + "nextn.shared_head_norm.weight", head_norm);
    std::sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) { return a.t->data < b.t->data; });

    std::vector<size_t> slab_bytes;
    std::vector<const QTensor *> abs_made;
    for (Job &j : jobs) {
        int type = j.t->type, rows = (int)j.t->ne[1], cols = (int)j.t->ne[0];
        if (j.kind == Job::ALPHA || j.kind == Job::BETA) {
            j.carve = std::find(abs_made.begin(), abs_made.end(), j.q) == abs_made.end();
            abs_made.push_back(j.q);
            type = F16; rows = 2 * GV; cols = D;
        } else {
            j.carve = j.kind == Job::QUANT || j.kind == Job::VEC;
        }
        if (j.kind == Job::QUANT) j.unit = block_bytes(j.t->type);
        if (j.unit == 4 && j.t->type != F32) FAIL("unsupported tensor type %d", j.t->type);
        if (!j.carve) continue;
        Arena dry;
        dry.cap = SIZE_MAX;
        make_qtensor(dry, type, rows, cols);
        j.bytes = dry.used;
        if (slab_bytes.empty() || (slab_bytes.back() + j.bytes > SLAB && slab_bytes.back() > 0)) slab_bytes.push_back(0);
        j.slab = (int)slab_bytes.size() - 1;
        j.off = slab_bytes.back();
        slab_bytes.back() += j.bytes;
    }
    std::vector<Piece> pieces;
    size_t head = 0, total_bytes = 0;
    for (int ji = 0; ji < (int)jobs.size(); ji++) {
        const Job &j = jobs[ji];
        const size_t step = PIECE / j.unit * j.unit;
        for (size_t s = 0; s < j.t->nbytes; s += step) {
            const size_t len = std::min(step, j.t->nbytes - s), alloc = (len + 255) & ~size_t(255);
            if (head + alloc > RING) head = 0;
            Piece pc{ji, s, len, head, -1};
            head += alloc;
            for (int k = (int)pieces.size() - 1; k >= 0; k--) {
                const Piece &o = pieces[k];
                if (o.ring < pc.ring + alloc && pc.ring < o.ring + ((o.len + 255) & ~size_t(255))) { pc.dep = k; break; }
            }
            pieces.push_back(pc);
            total_bytes += len;
        }
    }
    const int n = (int)pieces.size();
    load_total = total_bytes;

    std::atomic<size_t> prefetched{0};
    const size_t lo = jobs.front().t->data - g.base, hi = jobs.back().t->data - g.base + jobs.back().t->nbytes;
    size_t disk_bytes = 0;
    std::thread prefetch([&] {
        const os::File fd = os::open_read(path.c_str());
        uint8_t *scratch = (uint8_t *)os::aligned(4096, PIECE);
        for (size_t a = lo; a < hi; a += PIECE) {
            const size_t b = std::min(a + PIECE, hi), pa = a & ~size_t(4095), pb = (b + 4095) & ~size_t(4095);
            const size_t missing = os::missing(g.base + pa, pb - pa);
            for (size_t o = a; missing && o < b;) {
                const int64_t r = os::read_at(fd, scratch, b - o, o);
                if (r <= 0) break;
                o += r;
            }
            disk_bytes += missing * 4096;
            publish(prefetched, b);
        }
        os::aligned_free(scratch);
        os::close(fd);
    });

    CK(hipInit(0));
    CK(hipSetDevice(0));
    CK(hipFree(nullptr));
    const double t_gpu = now();
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    if (!gpu_runs()) FAIL("%s cannot run this engine", prop.name);
    uint8_t *ring;
    if (prop.isLargeBar) CK(hipExtMallocWithFlags((void **)&ring, RING, hipDeviceMallocFinegrained));
    else CK(hipHostMalloc(&ring, RING));
    std::vector<hipEvent_t> done(n);
    for (hipEvent_t &e : done) CK(hipEventCreateWithFlags(&e, hipEventDisableTiming));

    std::atomic<int> next{0}, submitted{0};
    std::vector<std::atomic<uint8_t>> filled(n);
    std::vector<std::thread> writers;
    for (int w = 0; w < WRITERS; w++) writers.emplace_back([&] {
        for (int i; (i = next.fetch_add(1)) < n;) {
            const Piece &pc = pieces[i];
            const uint8_t *src = jobs[pc.job].t->data + pc.src;
            wait_for(prefetched, (size_t)(src - g.base) + pc.len);
            if (pc.dep >= 0) {
                wait_for(submitted, pc.dep + 1);
                CK(hipEventSynchronize(done[pc.dep]));
            }
            stream_copy(ring + pc.ring, src, pc.len);
            publish(filled[i], (uint8_t)1);
        }
    });
    CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    CK(hipStreamCreateWithFlags(&cst, hipStreamNonBlocking));
    CK(hipEventCreateWithFlags(&staged, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&copied, hipEventDisableTiming));
    CK(hipEventRecord(copied, cst));
    f32_to_f16(nullptr, nullptr, 0, st);
    CK(hipStreamSynchronize(st));
    std::atomic<int> slabs_ready{0}, gdn_ready{0};
    std::vector<uint8_t *> slab_base(slab_bytes.size());
    std::thread vram([&] {
        CK(hipMalloc(&slab_base[0], slab_bytes[0]));
        publish(slabs_ready, 1);
        CK(hipMalloc(&gdn_A, 48 * GV * 4));
        CK(hipMalloc(&gdn_dt, 48 * GV * 4));
        publish(gdn_ready, 1);
        for (size_t k = 1; k < slab_bytes.size(); k++) {
            CK(hipMalloc(&slab_base[k], slab_bytes[k]));
            publish(slabs_ready, (int)k + 1);
        }
        weight_bytes = 0;
        for (size_t b : slab_bytes) weight_bytes += b;
        alloc_buffers(max_seqs);
        CK(hipMalloc(&lt_ws, lt_ws_size));
        size_t free_b, total_b;
        CK(hipMemGetInfo(&free_b, &total_b));
        for (size_t left = (free_b - std::min(free_b, reserve_bytes)) / KV_PAGE; left;) {
            const size_t pages = std::min(left, (size_t)(SLAB / KV_PAGE));
            uint8_t *kv;
            CK(hipMalloc(&kv, pages * KV_PAGE));
            add_kv_memory(kv, pages * KV_PAGE);
            left -= pages;
        }
    });

    for (int i = 0; i < n; i++) {
        const Piece &pc = pieces[i];
        Job &j = jobs[pc.job];
        if (pc.src == 0) {
            if (j.carve) {
                wait_for(slabs_ready, j.slab + 1);
                Arena a;
                a.base = slab_base[j.slab] + j.off;
                a.cap = j.bytes;
                const bool ab = j.kind == Job::ALPHA || j.kind == Job::BETA;
                j.made = make_qtensor(a, ab ? F16 : j.t->type, ab ? 2 * GV : (int)j.t->ne[1], ab ? D : (int)j.t->ne[0]);
                if (j.kind == Job::VEC) *j.vec = (float *)j.made.f;
                else *j.q = j.made;
            }
            if (j.kind == Job::GDN_A || j.kind == Job::GDN_DT) wait_for(gdn_ready, 1);
        }
        wait_for(filled[i], (uint8_t)1);
        const uint8_t *src = ring + pc.ring;
        switch (j.kind) {
            case Job::QUANT:
                if (j.t->type == F32) CK(hipMemcpyAsync((uint8_t *)j.q->f + pc.src, src, pc.len, hipMemcpyDefault, st));
                else relayout(*j.q, src, st, pc.src / j.unit, pc.len / j.unit);
                break;
            case Job::VEC:
                CK(hipMemcpyAsync((uint8_t *)j.made.f + pc.src, src, pc.len, hipMemcpyDefault, st));
                break;
            case Job::ALPHA: case Job::BETA:
                f32_to_f16((const float *)src, (half *)j.q->f + (j.kind == Job::BETA ? (size_t)GV * D : 0) + pc.src / 4, pc.len / 4, st);
                break;
            case Job::GDN_A: case Job::GDN_DT:
                CK(hipMemcpyAsync((j.kind == Job::GDN_A ? gdn_A : gdn_dt) + (size_t)j.gl * GV + pc.src / 4, src, pc.len, hipMemcpyDefault, st));
                break;
        }
        CK(hipEventRecord(done[i], st));
        publish(submitted, i + 1);
        load_done += pc.len;
        const uintptr_t a = (uintptr_t)(j.t->data + pc.src) & ~uintptr_t(4095);
        const uintptr_t e = ((uintptr_t)(j.t->data + pc.src + pc.len) + 4095) & ~uintptr_t(4095);
        os::drop_mapped((void *)a, e - a);
    }
    CK(hipStreamSynchronize(st));
    const double t_weights = now();
    for (std::thread &w : writers) w.join();
    prefetch.join();
    for (hipEvent_t e : done) CK(hipEventDestroy(e));
    if (prop.isLargeBar) CK(hipFree(ring));
    else CK(hipHostFree(ring));

    vram.join();
    if (prop.isLargeBar) {
        size_t free_b, total_b;
        CK(hipMemGetInfo(&free_b, &total_b));
        const size_t bytes = (free_b - std::min(free_b, reserve_bytes)) / KV_PAGE * KV_PAGE;
        uint8_t *kv;
        if (bytes && hipMalloc(&kv, bytes) == hipSuccess) add_kv_memory(kv, bytes);
    }
    const double t1 = now();
    fprintf(stderr, "chrysalis: weights %.2f GB, buffers %.2f GB, KV pool %d pages = %d tokens (%.2f GB)\n",
            weight_bytes / 1e9, barena.used / 1e9, n_pages, n_pages * PAGE, (double)n_pages * KV_PAGE / 1e9);
    fprintf(stderr, "chrysalis: loaded in %.2f s: GPU ready after %.2f s, %.1f GB of weights at %.1f GB/s, %.1f GB read from disk\n",
            t1 - t0, t_gpu - t0, total_bytes / 1e9, total_bytes / 1e9 / (t_weights - t_gpu), disk_bytes / 1e9);
}

void Engine::warmup() {
    if (embd.q) return;
    load_phase = 2;
    struct Done { ~Done() { load_phase = 3; } } done_;
    const double t0 = now();
    const size_t ne = (size_t)embd.rows * embd.cols, nb = ne / 256;
    CK(hipHostMalloc(&embd.q, ne / 2)); CK(hipHostMalloc(&embd.sc, ne / 64));
    CK(hipHostMalloc(&embd.sh, nb * 2)); CK(hipHostMalloc(&embd.d, nb * 2));
    std::vector<std::thread> th;
    const int nt = 4;
    for (int w = 0; w < nt; w++) th.emplace_back([&, w] {
        const os::File fd = os::open_read(model_path.c_str());
        constexpr size_t CHUNK = 7710;
        std::vector<uint8_t> buf(CHUNK * 136);
        const size_t b0 = nb * w / nt, b1 = nb * (w + 1) / nt;
        for (size_t b = b0; b < b1; b += CHUNK) {
            const size_t m = std::min(CHUNK, b1 - b);
            if (os::read_at(fd, buf.data(), m * 136, embd_offset + b * 136) != (int64_t)(m * 136)) FAIL("cannot read the token embeddings");
            for (size_t k = 0; k < m; k++) {
                const uint8_t *blk = buf.data() + k * 136;
                memcpy((uint8_t *)embd.d + (b + k) * 2, blk, 2);
                memcpy((uint8_t *)embd.sh + (b + k) * 2, blk + 2, 2);
                memcpy(embd.sc + (b + k) * 4, blk + 4, 4);
                memcpy(embd.q + (b + k) * 128, blk + 8, 128);
            }
        }
        os::close(fd);
    });
    snaps.resize(std::max<size_t>(snaps.size(), 8));
    for (Snap &sn : snaps)
        if (!sn.host) CK(hipHostMalloc(&sn.host, state_bytes()));
    for (std::thread &t : th) t.join();
    if (!lt) hipblasLtCreate(&lt);
    int id = seq_new();
    const bool borrowed = id < 0;
    if (borrowed) id = 0;
    const std::vector<int> toks(9, 0);
    prefill(id, toks.data(), nullptr, 1);
    prefill(id, toks.data(), nullptr, 9);
    if (borrowed) seq_truncate(id, 0);
    else seq_free(id);
    CK(hipStreamSynchronize(st));
    fprintf(stderr, "chrysalis: warmed up in %.2f s\n", now() - t0);
}

void Engine::alloc_buffers(int max_seqs) {
    umax = (umax + 63) / 64 * 64;
    const size_t U = umax;
    {
        size_t need = 0;
        auto add = [&](size_t bytes) { need += (bytes + 255) & ~size_t(255); };
        add(U * 4); add(U * D * 4 * 3);
        add(U * 16384 * 4); add(U * NKV * HD * 4 * 2);
        add(U * GK * GD * 4 * 2); add(U * GV * GD * 4 * 2);
        add(U * 2 * D * 2); add((size_t)FF * D * 2);
        add(U * NH * HD * 2 * 2); add(U * FF * 2 * 2);
        add(U * 2 * NH * HD * 2);
        add(8 * FF * 4); add(8 * VOCAB * 4);
        add((size_t)MAX_SPLITS * 8 * NH * HD * 4); add((size_t)MAX_SPLITS * 8 * NH * 2 * 4);
        add(8 * 2 * D * 5 / 4 * 3);
        add(8 * 64 * TOPK * 8 * 2);
        add(8 * LOGIT_STATS * 4);
        add(48 * 8 * (QKV + GK * GD + GV * GD + 2 * GV) * 4);
        add(8 * D * 4 * 2); add(64);
        add(U * sizeof(half *)); add(U * 3 * 4);
        barena.init(need + (1 << 20));
    }
    b.tokens = barena.get<int>(U);
    b.x = barena.get<float>(U * D); b.hfin = barena.get<float>(U * D); b.mtp_h = barena.get<float>(U * D);
    b.proj = barena.get<float>(U * 16384); b.k = barena.get<float>(U * NKV * HD); b.v = barena.get<float>(U * NKV * HD);
    b.qn = barena.get<float>(U * GK * GD); b.kn = barena.get<float>(U * GK * GD);
    b.vc = barena.get<float>(U * GV * GD); b.o = barena.get<float>(U * GV * GD);
    b.hh = barena.get<half>(U * 2 * D); b.deq = barena.get<half>((size_t)FF * D);
    b.Q16 = barena.get<half>(U * NH * HD); b.mix = barena.get<half>(U * NH * HD);
    b.ffn_g = barena.get<half>(U * FF); b.ffn_u = barena.get<half>(U * FF);
    b.gemm16 = barena.get<half>(U * 2 * NH * HD);
    b.act = barena.get<float>(8 * FF); b.logits = barena.get<float>(8 * (size_t)VOCAB);
    b.part_o = barena.get<float>((size_t)MAX_SPLITS * 8 * NH * HD); b.part_ml = barena.get<float>((size_t)MAX_SPLITS * 8 * NH * 2);
    b.hq = {barena.get<int8_t>(8 * 2 * D), barena.get<float>(8 * 2 * D / 32)};
    b.mq = {barena.get<int8_t>(8 * 2 * D), barena.get<float>(8 * 2 * D / 32)};
    b.aq = {barena.get<int8_t>(8 * FF), barena.get<float>(8 * FF / 32)};
    b.topv = barena.get<float>(8 * TOPK); b.topi = barena.get<int>(8 * TOPK);
    b.tops = barena.get<float>(8 * LOGIT_STATS);
    b.tsv = barena.get<float>(8 * 64 * TOPK); b.tsi = barena.get<int>(8 * 64 * TOPK);
    b.v_raw = barena.get<float>(48 * 8 * QKV); b.v_kn = barena.get<float>(48 * 8 * GK * GD);
    b.v_vc = barena.get<float>(48 * 8 * GV * GD); b.v_ab = barena.get<float>(48 * 8 * 2 * GV);
    b.mtp_x = barena.get<float>(8 * D); b.draft = barena.get<int>(16);
    b.img = barena.get<const half *>(U); b.rpos = barena.get<int>(U * 3);

    CK(hipMalloc(&stage_state, state_bytes()));
    seqs.resize(max_seqs);
    const size_t ssm_bytes = (size_t)48 * GV * GD * GD * 4, conv_bytes = (size_t)48 * QKV * 3 * 4;
    CK(hipMalloc(&ssm_pool, ssm_bytes * max_seqs));
    CK(hipMalloc(&conv_pool, conv_bytes * max_seqs));
    for (int i = 0; i < max_seqs; i++) {
        seqs[i].ssm = ssm_pool + ssm_bytes / 4 * i;
        seqs[i].conv = conv_pool + conv_bytes / 4 * i;
        CK(hipMalloc(&seqs[i].h_last, D * 4));
        CK(hipMalloc(&seqs[i].d_pages, 262144 / PAGE * 4 * 2));
    }
}
