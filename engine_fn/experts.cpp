// Which experts the VRAM pool holds. The file splits each layer's experts into a VRAM tier, chosen from coding work, and
// a RAM tier that the CPU computes; a reply in other prose picks the VRAM tier for a fifth of its experts (measured:
// 20%) where the 5,600 experts it uses most would serve nine tenths (89-92%), and every pick the CPU computes costs a
// read of its weights from RAM, which bounds decode. So the pool follows the passes: picks are counted per (layer,
// expert) (decode by the router on the GPU, prefill on the host), decayed by tokens, and an expert the CPU would read
// more megabytes of than the least of the pool's (by a margin) takes that one's slot.
//
// An expert keeps its own format wherever it is (the VRAM tier's IQ4_XS, the RAM tier's Q3_K / Q2_K): moving it changes
// where it is computed, not its weights. One out of the pool goes back to its RAM copy, or for the file's VRAM tier, which
// has none, to the model file's mapping (ColdLayer group 2), so RAM use stays as it was. Coming in, its blocks are copied
// to a VRAM stage (a load: by DMA from RAM, or read from the file into pinned memory first, on the loader thread), and
// at a pass boundary one launch writes the stages into their slots and patches the routing tables, in stream order with
// the passes, which therefore see either expert whole. Results depend on where an expert was computed only by rounding.
#include "model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace {
constexpr float PRIOR = 8;          // picks a VRAM-tier expert starts with: the file's choice holds until the passes outvote it
constexpr float MARGIN = 0.25f;     // an expert takes a slot from one it beats by this share of the latter's megabytes ...
constexpr float MIN_GAIN = 2.0f;    // ... plus this many (about one pick)
constexpr size_t STAGE_BYTES = 2 * (size_t)FE * D / 256 * 136 + (size_t)D * FE / 32 * 18;   // the widest expert: IQ4_XS
}

void Engine::file_read(uint8_t *dst, const uint8_t *src, size_t n) const {
    const off_t off = (off_t)(src - file_base);
    for (size_t got = 0; got < n;) {
        const ssize_t r = pread(file_fd, dst + got, n - got, off + (off_t)got);
        if (r <= 0) FAIL("reading experts from the model file failed");
        got += (size_t)r;
    }
}

void Engine::experts_init() {
    Experts &x = xc;
    const size_t keys = (size_t)NL * NE;
    x.owner.assign(n_slots, -1);
    x.fmt.assign(n_slots, IQ4_XS);
    x.busy.assign(n_slots, 0);
    x.slot.assign(keys, -1); x.home.assign(keys, 0); x.type.assign(keys, IQ4_XS);
    x.score.assign(keys, 0.0f); x.mb.assign(keys, 0.0f);
    x.fresh.assign(keys, 0); x.total.assign(keys, 0); x.coming.assign(keys, 0);
    for (int l = 0; l < NL; l++) {
        const ColdLayer &c = cold[l];
        int first = n_slots;                     // the layer's VRAM tier fills pool slots first.. in file order
        for (int e = 0; e < NE; e++) if (L[l].slot_h[e] >= 0) first = std::min(first, L[l].slot_h[e]);
        for (int e = 0; e < NE; e++) {
            const size_t k = (size_t)l * NE + e;
            const int s = L[l].slot_h[e];
            x.home[k] = s >= 0 ? c.n + (s - first) : -1 - s;   // RAM slot: the VRAM tier's in group 2, the file's own
            if (s >= 0) { x.slot[k] = s; x.owner[s] = (int)k; x.score[k] = PRIOR; }
            x.type[k] = c.type_gu[c.grp(x.home[k])];
            x.mb[k] = (float)(c.bytes(x.home[k]) / 1e6);
        }
    }
    CK(hipMalloc(&x.d_counts, keys * 4));
    CK(hipMemsetAsync(x.d_counts, 0, keys * 4, st));
    CK(hipHostMalloc(&x.h_counts, keys * 4));
    CK(hipEventCreateWithFlags(&x.counted, hipEventDisableTiming));
    CK(hipStreamCreateWithFlags(&x.xst, hipStreamNonBlocking));
    for (Experts::Load &ld : x.loads) {
        CK(hipMalloc(&ld.stage, STAGE_BYTES));
        CK(hipHostMalloc(&ld.bounce, STAGE_BYTES));
        CK(hipEventCreate(&ld.start));
        CK(hipEventCreate(&ld.copied));
        CK(hipEventCreateWithFlags(&ld.done, hipEventDisableTiming));
    }
    x.quit = false;
    x.loader = std::thread(&Engine::experts_loader, this);
}

void Engine::experts_stop() {
    Experts &x = xc;
    if (x.loader.joinable()) {
        { std::lock_guard<std::mutex> lk(x.mu); x.quit = true; }
        x.cv.notify_all();
        x.loader.join();
    }
    if (x.xst) CK(hipStreamSynchronize(x.xst));
    for (Experts::Load &ld : x.loads) {
        if (ld.stage) (void)hipFree(ld.stage);
        if (ld.bounce) (void)hipHostFree(ld.bounce);
        if (ld.start) (void)hipEventDestroy(ld.start);
        if (ld.copied) (void)hipEventDestroy(ld.copied);
        if (ld.done) (void)hipEventDestroy(ld.done);
        ld.stage = ld.bounce = nullptr; ld.start = ld.copied = ld.done = nullptr;
        ld.state = Experts::FREE; ld.key = ld.slot = -1;
    }
    if (x.d_counts) (void)hipFree(x.d_counts);
    if (x.h_counts) (void)hipHostFree(x.h_counts);
    if (x.counted) (void)hipEventDestroy(x.counted);
    if (x.xst) (void)hipStreamDestroy(x.xst);
    x.d_counts = nullptr; x.h_counts = nullptr; x.counted = nullptr; x.xst = nullptr;
    x.counts_pending = false;
}

// The loads' copies. A load's stage is free once the placement that last read it ran (its `done` event, waited for on
// the copy stream); its bounce buffer once the copy that last read it finished (`copied`, waited for here). Between
// loads, readahead of the file pages of experts that left the pool (madvise may wait to submit the reads: not on the
// thread that queues the passes).
void Engine::experts_loader() {
    Experts &x = xc;
    for (;;) {
        int k = -1, ra = -1;
        {
            std::unique_lock<std::mutex> lk(x.mu);
            x.cv.wait(lk, [&] {
                if (x.quit || !x.readahead.empty()) return true;
                for (int i = 0; i < PLACE_MAX; i++) if (x.loads[i].state.load(std::memory_order_acquire) == Experts::QUEUED) return true;
                return false;
            });
            if (x.quit) return;
            for (int i = 0; i < PLACE_MAX && k < 0; i++) if (x.loads[i].state.load(std::memory_order_acquire) == Experts::QUEUED) k = i;
            if (k < 0) { ra = x.readahead.front(); x.readahead.pop_front(); }
        }
        if (ra >= 0) {
            const ColdLayer &c = cold[ra / NE];
            const int h = x.home[ra];
            const uint8_t *r[3] = {c.gu_row(false, h, 0), c.gu_row(true, h, 0), c.dn_row(h, 0)};
            const size_t n[3] = {FE * c.row_gu[2], FE * c.row_gu[2], D * c.row_dn};
            for (int m = 0; m < 3; m++) {
                const uintptr_t a = (uintptr_t)r[m] & ~uintptr_t(4095);
                madvise((void *)a, (uintptr_t)r[m] + n[m] - a, MADV_WILLNEED);
            }
            continue;
        }
        Experts::Load &ld = x.loads[k];
        const int l = ld.key / NE, h = x.home[ld.key];
        const ColdLayer &c = cold[l];
        const size_t gu = (size_t)FE * c.row_gu[c.grp(h)], dn = (size_t)D * c.row_dn;
        const uint8_t *src[3] = {c.gu_row(false, h, 0), c.gu_row(true, h, 0), c.dn_row(h, 0)};
        const size_t len[3] = {gu, gu, dn};
        CK(hipStreamWaitEvent(x.xst, ld.done, 0));
        if (c.grp(h) == 2) {                     // not in RAM: from the file, into pinned memory
            CK(hipEventSynchronize(ld.copied));
            size_t at = 0;
            for (int m = 0; m < 3; m++) { file_read(ld.bounce + at, src[m], len[m]); at += len[m]; }
            CK(hipEventRecord(ld.start, x.xst));
            CK(hipMemcpyAsync(ld.stage, ld.bounce, at, hipMemcpyHostToDevice, x.xst));
        } else {
            size_t at = 0;
            CK(hipEventRecord(ld.start, x.xst));
            for (int m = 0; m < 3; m++) { CK(hipMemcpyAsync(ld.stage + at, src[m], len[m], hipMemcpyHostToDevice, x.xst)); at += len[m]; }
        }
        CK(hipEventRecord(ld.copied, x.xst));
        ld.state.store(Experts::COPYING, std::memory_order_release);
    }
}

// At a pass's start (cheap: the GPU waits for this): loads that have arrived go into their slots with the routing tables,
// in one launch, and the router's pick counts are copied out for experts_plan.
void Engine::experts_place(int T) {
    Experts &x = xc;
    if (x.owner.empty()) return;
    x.tokens += T;
    const size_t keys = (size_t)NL * NE;
    if (!x.counts_pending) {
        CK(hipMemcpyAsync(x.h_counts, x.d_counts, keys * 4, hipMemcpyDeviceToHost, st));
        CK(hipMemsetAsync(x.d_counts, 0, keys * 4, st));
        CK(hipEventRecord(x.counted, st));
        x.counts_pending = true;
    }
    if (!x.on) return;
    PlaceArgs pa;
    pa.gate = pool_gate; pa.up = pool_up; pa.down = pool_down; pa.slot_of = d_slot_of; pa.fmt = d_fmt;
    int placed[PLACE_MAX], np = 0;
    for (int k = 0; k < PLACE_MAX; k++) {
        Experts::Load &ld = x.loads[k];
        if (ld.state.load(std::memory_order_acquire) != Experts::COPYING || hipEventQuery(ld.copied) != hipSuccess) continue;
        const int key = ld.key, s = ld.slot, old = x.owner[s];
        float ms = 0;
        if (hipEventElapsedTime(&ms, ld.start, ld.copied) == hipSuccess && ms > 0) {
            x.copied_bytes += cold[key / NE].bytes(x.home[key]); x.copied_ms += ms;
        }
        pa.p[pa.n++] = {ld.stage, s, x.type[key]};
        if (old >= 0) {
            pa.patch[pa.npatch++] = {old, -1, -1 - x.home[old]};
            x.slot[old] = -1;
            const int lo = old / NE;
            L[lo].slot_h[old % NE] = -1 - x.home[old];
            if (cold[lo].grp(x.home[old]) == 2) {   // back to the file: the loader starts reading its pages for the CPU
                std::lock_guard<std::mutex> lk(x.mu);
                x.readahead.push_back(old);
            }
        }
        pa.patch[pa.npatch++] = {key, -1, s};
        pa.patch[pa.npatch++] = {-1, s, x.type[key]};
        x.slot[key] = s; x.owner[s] = key; x.fmt[s] = (uint8_t)x.type[key];
        L[key / NE].slot_h[key % NE] = s;
        x.busy[s] = 0; x.coming[key] = 0;
        x.placed++;
        x.loaded_file += cold[key / NE].grp(x.home[key]) == 2;
        placed[np++] = k;
    }
    // The link's speed changes under the engine (this card's PCIe link has come up for whole sessions at 2.5 GT/s, 3.5 GB/s,
    // and at 8 GT/s, 12.5 GB/s, in the same hour, measured), so the loads' copies keep the load-time DMA rate current:
    // the prefill split's cost model and the decode GPU share follow what the link does now.
    if (x.copied_ms > 0 && x.copied_bytes >= 64 * 2.0e6) {
        cal_dma_bw = 0.5 * cal_dma_bw + 0.5 * x.copied_bytes / (x.copied_ms * 1e-3);
        if (!gfrac_pinned) gfrac = (float)(cal_dma_bw / (cal_dma_bw + cal_cpu_bw));
        x.copied_bytes = x.copied_ms = 0;
    }
    if (pa.n) {
        place_experts(pa, st);
        for (int i = 0; i < np; i++) {
            CK(hipEventRecord(x.loads[placed[i]].done, st));
            x.loads[placed[i]].state.store(Experts::FREE, std::memory_order_release);
        }
        x.cv.notify_one();
    }
}

// Once a pass is queued (the host would wait for the GPU anyway): the picks counted so far into the scores, and loads for
// the free stages, the experts the CPU would read the most megabytes of against the pool's least.
void Engine::experts_plan() {
    Experts &x = xc;
    if (x.owner.empty()) return;
    const size_t keys = (size_t)NL * NE;
    if (x.counts_pending && hipEventQuery(x.counted) == hipSuccess) {
        uint64_t picks = 0, hits = 0;
        for (size_t k = 0; k < keys; k++) {
            const uint32_t c = x.h_counts[k];
            x.fresh[k] += c; x.total[k] += c;
            picks += c; hits += x.slot[k] >= 0 ? c : 0;
        }
        x.picks += picks; x.hits += hits;
        if (picks) {                             // over about the last 32 decode passes
            const double a = std::min(1.0, picks / (32.0 * 30 * NL));
            x.recent_hits = (1 - a) * x.recent_hits + a * (double)hits / picks;
        }
        x.counts_pending = false;
    }
    if (x.tokens > 0) {
        const float decay = (float)std::exp2(-x.tokens / x.half_life);
        for (size_t k = 0; k < keys; k++) { x.score[k] = x.score[k] * decay + (float)x.fresh[k]; x.fresh[k] = 0; }
        x.tokens = 0;
    }
    if (!x.on) return;
    int free_k[PLACE_MAX], nf = 0;
    for (int k = 0; k < PLACE_MAX; k++) if (x.loads[k].state.load(std::memory_order_acquire) == Experts::FREE) free_k[nf++] = k;
    if (!nf) return;
    std::vector<std::pair<float, int>> in, out;            // (value, key) of candidates; (value, slot) of the pool's
    in.reserve(nf + 1); out.reserve(nf + 1);
    auto keep = [nf](std::vector<std::pair<float, int>> &v, std::pair<float, int> e, bool largest) {
        // the nf largest (or smallest) seen so far, sorted best first
        auto better = [largest](const std::pair<float, int> &a, const std::pair<float, int> &b) { return largest ? a.first > b.first : a.first < b.first; };
        if ((int)v.size() == nf && !better(e, v.back())) return;
        v.insert(std::upper_bound(v.begin(), v.end(), e, better), e);
        if ((int)v.size() > nf) v.pop_back();
    };
    for (size_t k = 0; k < keys; k++)
        if (x.slot[k] < 0 && !x.coming[k] && x.score[k] > 0) keep(in, {x.score[k] * x.mb[k], (int)k}, true);
    if (in.empty()) return;
    for (int s = 0; s < n_slots; s++)
        if (!x.busy[s]) keep(out, {x.owner[s] >= 0 ? x.score[x.owner[s]] * x.mb[x.owner[s]] : -1.0f, s}, false);
    bool queued = false;
    for (int i = 0; i < (int)std::min(in.size(), out.size()); i++) {
        if (in[i].first <= out[i].first * (1 + MARGIN) + MIN_GAIN) break;
        Experts::Load &ld = x.loads[free_k[i]];
        ld.key = in[i].second; ld.slot = out[i].second;
        x.coming[ld.key] = 1; x.busy[ld.slot] = 1;
        {
            std::lock_guard<std::mutex> lk(x.mu);
            ld.state.store(Experts::QUEUED, std::memory_order_release);
        }
        queued = true;
    }
    if (queued) x.cv.notify_one();
}
