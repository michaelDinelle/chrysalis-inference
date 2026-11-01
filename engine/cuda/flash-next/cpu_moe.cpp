#include "cpu_moe.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <immintrin.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <string>

namespace {
constexpr int D_ = 2560, FE_ = 640;
inline uint32_t load_acq(const uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void store_rel(uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
}

void CpuMoe::start(MoeShared *shared, MoeShared *pred, const std::vector<ColdLayer> &layers, int threads, MoeEmu *emu) {
    ggml_cpu_init();
    sh = shared;
    me = emu;
    pr = pred;
    L = layers;
    n = threads;
    const ColdLayer *c0 = nullptr;
    for (const ColdLayer &c : L) if (c.n) { c0 = &c; break; }
    if (!c0) return;
    int t[3] = {c0->type_gu[0], c0->type_gu[0], c0->type_gu[0]};
    for (int g = 1; g < 3; g++)
        for (const ColdLayer &c : L) if (c.gate[g]) { t[g] = c.type_gu[g]; break; }
    for (const ColdLayer &c : L) {
        bool same = !c.n || c.type_dn == c0->type_dn;
        for (int g = 0; g < 3; g++) same &= !c.gate[g] || c.type_gu[g] == t[g];
        if (!same) fprintf(stderr, "chrysalis: cold expert formats differ between layers\n"), abort();
    }
    for (int g = 0; g < 3; g++) dot_gu[g] = ggml_get_type_traits_cpu((ggml_type)t[g])->vec_dot_type;
    dot_same = dot_gu[0] == dot_gu[1] && dot_gu[0] == dot_gu[2];
    const ggml_type dt_dn = ggml_get_type_traits_cpu((ggml_type)c0->type_dn)->vec_dot_type;
    for (int g = 0; g < 3; g++) { xq_row[g] = ggml_row_size((ggml_type)dot_gu[g], D_); xq[g].resize(xq_row[g] * MOE_MAX_T); }
    hq_row = ggml_row_size(dt_dn, FE_);
    gu.resize((size_t)MOE_MAX_T * MOE_MAX_E * 2 * FE_);
    hq.resize(hq_row * MOE_MAX_T * MOE_MAX_E);
    seen = load_acq(&sh->go);
    jstart = jgen.load();
    scr = std::vector<Scratch>(n);
    vnni = vnni_supported();
    for (const ColdLayer &c : L) if (c.n && !vnni_formats(c)) vnni = false;
    vnni_ok = vnni;
    dxq.resize((size_t)MOE_MAX_T * D_); dxs.resize(MOE_MAX_T * (D_ / 32)); dxsum.resize(MOE_MAX_T * (D_ / 16));
    ds16.resize(MOE_MAX_T * (D_ / 16)); df16.resize(MOE_MAX_T * (D_ / 16));
    dhq.resize((size_t)MOE_MAX_T * MOE_MAX_E * FE_); dhs.resize(MOE_MAX_T * MOE_MAX_E * (FE_ / 32)); dhc.resize(MOE_MAX_T * MOE_MAX_E * (FE_ / 32));
    std::vector<int> core;
    for (int c = 1; c < CPU_SETSIZE && (int)core.size() < n; c++) {
        FILE *f = fopen(("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list").c_str(), "r");
        if (!f) continue;
        int first = -1;
        if (fscanf(f, "%d", &first) == 1 && first == c) core.push_back(c);
        fclose(f);
    }
    for (int i = 0; i < n; i++) {
        th.emplace_back(&CpuMoe::worker, this, i);
        if (i < (int)core.size()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(core[i], &set);
            pthread_setaffinity_np(th.back().native_handle(), sizeof set, &set);
        }
    }
}

void CpuMoe::stop() {
    if (th.empty()) return;
    quit = true;
    wake.fetch_add(1, std::memory_order_release);
    wake.notify_all();
    for (auto &t : th) t.join();
    th.clear();
    quit = false;
}

void CpuMoe::arm() { armed.store(1, std::memory_order_release); wake.fetch_add(1, std::memory_order_release); wake.notify_all(); }
void CpuMoe::disarm() { armed.store(0, std::memory_order_release); }

void CpuMoe::barrier() {
    const uint32_t g = bar_gen.load(std::memory_order_acquire);
    if (bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
        bar_count.store(0, std::memory_order_relaxed);
        bar_gen.fetch_add(1, std::memory_order_release);
    } else {
        while (bar_gen.load(std::memory_order_acquire) == g) _mm_pause();
    }
}

void CpuMoe::submit(const MoeJob &j) {
    job = j;
    for (int g = 0; g < (dot_same ? 1 : 3); g++) if (pxq[g].size() < xq_row[g] * j.T) pxq[g].resize(xq_row[g] * j.T);
    if (vnni && vxq.size() < (size_t)j.T * D_) { vxq.resize((size_t)j.T * D_); vxs.resize((size_t)j.T * (D_ / 32)); vxsum.resize((size_t)j.T * (D_ / 16)); }
    jtok.store(0, std::memory_order_relaxed); jexp.store(0, std::memory_order_relaxed); jfin.store(0, std::memory_order_relaxed);
    jgen.fetch_add(1, std::memory_order_release);
    wake.fetch_add(1, std::memory_order_release);
    wake.notify_all();
}

void CpuMoe::wait() {
    while (jfin.load(std::memory_order_acquire) < n) _mm_pause();
}

void CpuMoe::prefill_work(int id) {
    const MoeJob &J = job;
    const ColdLayer &c = L[J.layer];
    const auto *tr_dn = ggml_get_type_traits_cpu((ggml_type)c.type_dn);
    const auto from_dn = ggml_get_type_traits_cpu(tr_dn->vec_dot_type)->from_float;
    for (int t; (t = jtok.fetch_add(1, std::memory_order_relaxed)) < J.T;) {
        float x[D_];
        ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)J.x + (size_t)t * D_, x, D_);
        for (int g = 0; g < (dot_same ? 1 : 3); g++)
            ggml_get_type_traits_cpu((ggml_type)dot_gu[g])->from_float(x, pxq[g].data() + xq_row[g] * t, D_);
        if (vnni) vnni_quant_x(x, vxq.data() + (size_t)t * D_, vxs.data() + (size_t)t * (D_ / 32), vxsum.data() + (size_t)t * (D_ / 16));
    }
    barrier();
    Scratch &s = scr[id];
    for (int e; (e = jexp.fetch_add(1, std::memory_order_relaxed)) < J.n_exp;) {
        const int p0 = J.first[e], np = J.first[e + 1] - p0, slot = J.slot[e];
        if (vnni && np >= VNNI_MIN_TOKENS) {
            std::vector<const int8_t *> xq(np); std::vector<const float *> xs(np); std::vector<const int32_t *> xsum(np);
            std::vector<uint16_t *> out(np);
            for (int i = 0; i < np; i++) {
                const int t = J.tok[p0 + i];
                xq[i] = vxq.data() + (size_t)t * D_; xs[i] = vxs.data() + (size_t)t * (D_ / 32);
                xsum[i] = vxsum.data() + (size_t)t * (D_ / 16);
                out[i] = J.out + (size_t)(p0 + i) * D_;
            }
            expert_vnni(c, slot, np, xq.data(), xs.data(), xsum.data(), J.w + p0, out.data(), s.v);
            continue;
        }
        if (s.gu.size() < (size_t)np * 2 * FE_) { s.gu.resize((size_t)np * 2 * FE_); s.y.resize((size_t)np * D_); s.hq.resize(hq_row * np); }
        const int g = c.grp(slot), gx = dot_same ? 0 : g;
        const auto *tr_gu = ggml_get_type_traits_cpu((ggml_type)c.type_gu[g]);
        for (int r = 0; r < 2 * FE_; r++) {
            const uint8_t *row = c.gu_row(r >= FE_, slot, r % FE_);
            for (int i = 0; i < np; i++)
                tr_gu->vec_dot(D_, &s.gu[(size_t)i * 2 * FE_ + r], 0, row, 0, pxq[gx].data() + xq_row[gx] * J.tok[p0 + i], 0, 1);
        }
        for (int i = 0; i < np; i++) {
            float h[FE_];
            const float *g2 = &s.gu[(size_t)i * 2 * FE_];
            for (int k = 0; k < FE_; k++) { const float a = g2[k]; h[k] = a / (1.0f + expf(-a)) * g2[FE_ + k]; }
            from_dn(h, s.hq.data() + hq_row * i, FE_);
        }
        for (int r = 0; r < D_; r++) {
            const uint8_t *row = c.dn_row(slot, r);
            for (int i = 0; i < np; i++) tr_dn->vec_dot(FE_, &s.y[(size_t)i * D_ + r], 0, row, 0, s.hq.data() + hq_row * i, 0, 1);
        }
        for (int i = 0; i < np; i++) {
            float *y = &s.y[(size_t)i * D_];
            const float w = J.w[p0 + i];
            for (int k = 0; k < D_; k++) y[k] *= w;
            ggml_cpu_fp32_to_fp16(y, (ggml_fp16_t *)J.out + (size_t)(p0 + i) * D_, D_);
        }
    }
    jfin.fetch_add(1, std::memory_order_release);
}

void CpuMoe::preload(int id, uint32_t real, uint32_t job) {
    const int layer = pr->layer, T = std::min(pr->T, MOE_MAX_T);
    if (layer < 0 || layer >= (int)L.size()) return;
    const ColdLayer &c = L[layer];
    int slot[MOE_MAX_T * MOE_MAX_E], np = 0;
    for (int j = 0; j < T * MOE_MAX_E; j++) { const int s = pr->slot[j] <= -2 ? -2 - pr->slot[j] : pr->slot[j]; if (s >= 0 && s < c.n + c.n2) slot[np++] = s; }
    uint64_t sink = 0;
    for (int p = 0; p < np; p++) {
        const uint8_t *seg[3] = {c.gu_row(false, slot[p], 0), c.gu_row(true, slot[p], 0), c.dn_row(slot[p], 0)};
        const size_t g = c.grp(slot[p]), len[3] = {640 * c.row_gu[g], 640 * c.row_gu[g], 2560 * c.row_dn};
        for (int k = 0; k < 3; k++) {
            const size_t a = len[k] * id / n / 64 * 64, b = len[k] * (id + 1) / n;
            for (size_t o = a; o < b; o += 64 * 32) {
                if (load_acq(&sh->go) != real || jgen.load(std::memory_order_relaxed) != job) { sink_ += sink; return; }
                const size_t e = std::min(b, o + 64 * 32);
                for (size_t q = o; q < e; q += 64) sink += *(const volatile uint64_t *)(seg[k] + q);
            }
        }
    }
    sink_ += sink;
}

void CpuMoe::worker(int id) {
    uint32_t last = seen.load(), jlast = jstart, plast = pr ? load_acq(&pr->go) : 0;
    for (;;) {
        uint32_t g;
        bool host_job = false;
        for (;;) {
            const uint32_t wk = wake.load(std::memory_order_acquire);
            if (quit.load(std::memory_order_acquire)) return;
            const uint32_t j = jgen.load(std::memory_order_acquire);
            if (j != jlast) { jlast = j; host_job = true; break; }
            g = load_acq(&sh->go);
            if (g != last) break;
            if (!armed.load(std::memory_order_acquire)) { wake.wait(wk, std::memory_order_acquire); continue; }
            if (pr) {
                const uint32_t p = load_acq(&pr->go);
                if (p != plast) {
                    plast = p;
                    if (p == last + 1) { preload(id, last, jlast); continue; }
                }
            }
            _mm_pause();
        }
        if (host_job) { prefill_work(id); continue; }
        last = g;
        const int layer = sh->layer, T = sh->T;
        const ColdLayer &c = L[layer];
        int pick[MOE_MAX_T * MOE_MAX_E], np = 0, emu[MOE_MAX_T * MOE_MAX_E], ne = 0;
        int uslot[MOE_MAX_T * MOE_MAX_E], ufirst[MOE_MAX_T * MOE_MAX_E + 1], umem[MOE_MAX_T * MOE_MAX_E], nu = 0;
        for (int j = 0; j < T * MOE_MAX_E; j++) {
            if (sh->slot[j] >= 0) pick[np++] = j;
            else if (sh->slot[j] <= -2) emu[ne++] = j;
        }
        for (int p = 0; p < np; p++) {
            const int s = sh->slot[pick[p]];
            int u = 0;
            while (u < nu && uslot[u] != s) u++;
            if (u == nu) uslot[nu++] = s;
        }
        ufirst[0] = 0;
        for (int u = 0, m = 0; u < nu; u++) {
            for (int p = 0; p < np; p++)
                if (sh->slot[pick[p]] == uslot[u]) umem[m++] = p;
            ufirst[u + 1] = m;
        }
        const ggml_type_traits_cpu *tr_gu[3] = {ggml_get_type_traits_cpu((ggml_type)c.type_gu[0]),
                                                ggml_get_type_traits_cpu((ggml_type)(c.gate[1] ? c.type_gu[1] : c.type_gu[0])),
                                                ggml_get_type_traits_cpu((ggml_type)(c.gate[2] ? c.type_gu[2] : c.type_gu[0]))};
        const auto *tr_dn = ggml_get_type_traits_cpu((ggml_type)c.type_dn);
        const bool vx = vnni;
        if (np + ne > 0) {
            for (int t = id; t < T && ne; t += n) emu_quant_x(sh->x + (size_t)t * D_, ea[t]);
            for (int t = id; t < T && np; t += n) {
                if (vx) {
                    int32_t *xsum = dxsum.data() + t * (D_ / 16);
                    vnni_quant_x(sh->x + (size_t)t * D_, dxq.data() + (size_t)t * D_, dxs.data() + t * (D_ / 32), xsum);
                    vnni_act1(dxs.data() + t * (D_ / 32), xsum, ds16.data() + t * (D_ / 16), df16.data() + t * (D_ / 16));
                } else {
                    for (int g = 0; g < (dot_same ? 1 : 3); g++)
                        ggml_get_type_traits_cpu((ggml_type)dot_gu[g])->from_float(sh->x + (size_t)t * D_, xq[g].data() + xq_row[g] * t, D_);
                }
            }
            barrier();
            const int R1 = (nu + ne) * 2 * FE_;
            for (int r = id * R1 / n; r < (id + 1) * R1 / n; r++) {
                if (r / (2 * FE_) >= nu) {
                    const int j = emu[r / (2 * FE_) - nu], k = r % (2 * FE_), slot = -2 - sh->slot[j];
                    me->gu[j][k] = emu_gu_row(c.gu_row(k >= FE_, slot, k % FE_), c.type_gu[c.grp(slot)], ea[j / MOE_MAX_E], scr[id].v);
                    continue;
                }
                const int u = r / (2 * FE_), k = r % (2 * FE_), slot = uslot[u], g = c.grp(slot), gx = dot_same ? 0 : g;
                const uint8_t *row = c.gu_row(k >= FE_, slot, k % FE_);
                for (int m = ufirst[u]; m < ufirst[u + 1]; m++) {
                    const int p = umem[m], t = pick[p] / MOE_MAX_E;
                    if (vx)
                        gu[(size_t)p * 2 * FE_ + k] = vnni_dot_gu1(row, c.type_gu[g], dxq.data() + (size_t)t * D_, dxs.data() + t * (D_ / 32),
                                                                   dxsum.data() + t * (D_ / 16), ds16.data() + t * (D_ / 16),
                                                                   df16.data() + t * (D_ / 16), scr[id].v);
                    else tr_gu[g]->vec_dot(D_, &gu[(size_t)p * 2 * FE_ + k], 0, row, 0, xq[gx].data() + xq_row[gx] * t, 0, 1);
                }
            }
            barrier();
            if (ne && id == 0) store_rel(&me->ready, g);
            for (int p = id; p < np; p += n) {
                float h[FE_];
                const float *g2 = &gu[(size_t)p * 2 * FE_];
                for (int i = 0; i < FE_; i++) { const float a = g2[i]; h[i] = a / (1.0f + expf(-a)) * g2[FE_ + i]; }
                if (vx) vnni_quant_h(h, dhq.data() + (size_t)p * FE_, dhs.data() + p * (FE_ / 32), dhc.data() + p * (FE_ / 32));
                else ggml_get_type_traits_cpu(tr_dn->vec_dot_type)->from_float(h, hq.data() + hq_row * p, FE_);
            }
            barrier();
        }
        for (int r = id * D_ / n; r < (id + 1) * D_ / n; r++) {
            float acc[MOE_MAX_T] = {};
            for (int p = 0; p < np; p++) {
                const int j = pick[p], slot = sh->slot[j];
                float v;
                if (vx) v = vnni_dot_dn1(c.dn_row(slot, r), dhq.data() + (size_t)p * FE_, dhs.data() + p * (FE_ / 32), dhc.data() + p * (FE_ / 32));
                else tr_dn->vec_dot(FE_, &v, 0, c.dn_row(slot, r), 0, hq.data() + hq_row * p, 0, 1);
                acc[j / MOE_MAX_E] += sh->w[j] * v;
            }
            for (int t = 0; t < T; t++) sh->y[(size_t)t * D_ + r] = acc[t];
        }
        for (int e = 0; e < ne; e++) {
            const int j = emu[e], slot = -2 - sh->slot[j];
            for (int b = 0; b < FE_ / 32; b++) while (load_acq(&me->flag[j][b]) != g) _mm_pause();
            alignas(64) int32_t hc[FE_ / 32];
            for (int b = 0; b < FE_ / 32; b++) { int s = 0; for (int i = 0; i < 32; i++) s += me->hq[j][b * 32 + i]; hc[b] = 128 * s; }
            for (int r = id * D_ / n; r < (id + 1) * D_ / n; r++)
                me->y[j][r] = sh->w[j] * emu_dn_row(c.dn_row(slot, r), me->hq[j], me->hd[j], hc, scr[id].v);
        }
        barrier();
        if (id == 0) { std::atomic_thread_fence(std::memory_order_release); store_rel(&sh->done, g); }
    }
}
