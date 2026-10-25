#include "model.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr float PRIOR = 8;
constexpr float MARGIN = 0.25f;
constexpr float MIN_GAIN = 2.0f;
constexpr size_t STAGE_BYTES = 2 * (size_t)FE * D / 256 * 136 + (size_t)D * FE / 32 * 18;

struct HotCopy { uint8_t *mirror[3]; const uint8_t *file[3]; size_t n[3]; };
HotCopy hot_copy(const ColdLayer &c, int h) {
    const uint8_t *m[3] = {c.gu_row(false, h, 0), c.gu_row(true, h, 0), c.dn_row(h, 0)};
    const uint8_t *mb[3] = {c.gate[2], c.up[2], c.down2}, *fb[3] = {c.file_gate, c.file_up, c.file_down2};
    HotCopy hc;
    for (int i = 0; i < 3; i++) {
        hc.mirror[i] = const_cast<uint8_t *>(m[i]);
        hc.file[i] = fb[i] + (m[i] - mb[i]);
        hc.n[i] = i < 2 ? (size_t)FE * c.row_gu[2] : (size_t)D * c.row_dn;
    }
    return hc;
}
}

static double mem_available(Engine::Experts &x) {
    const auto now = std::chrono::steady_clock::now();
    if (now - x.avail_at < std::chrono::seconds(1)) return x.avail;
    x.avail_at = now;
    if (FILE *f = fopen("/proc/meminfo", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "MemAvailable:", 13)) { x.avail = atof(line + 13) * 1024; break; }
        fclose(f);
    }
    return x.avail;
}

void Engine::file_read(uint8_t *dst, const uint8_t *src, size_t n) const {
    const off_t off = (off_t)(src - file_base);
    for (size_t got = 0; got < n;) {
        const ssize_t r = pread(file_fd, dst + got, n - got, off + (off_t)got);
        if (r <= 0) FAIL("reading experts from the model file failed");
        got += (size_t)r;
    }
    posix_fadvise(file_fd, off, (off_t)n, POSIX_FADV_DONTNEED);
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
        int first = n_slots;
        for (int e = 0; e < NE; e++) if (L[l].slot_h[e] >= 0) first = std::min(first, L[l].slot_h[e]);
        for (int e = 0; e < NE; e++) {
            const size_t k = (size_t)l * NE + e;
            const int s = L[l].slot_h[e];
            x.home[k] = s >= 0 ? c.n + (s - first) : -1 - s;
            if (s >= 0) { x.slot[k] = s; x.owner[s] = (int)k; x.score[k] = PRIOR; }
            x.type[k] = c.type_gu[c.grp(x.home[k])];
            x.mb[k] = (float)(c.bytes(x.home[k]) / 1e6);
        }
    }
    size_t hot_bytes = 0;
    int hot = 0;
    for (const ColdLayer &c : cold) if (c.n2) { hot_bytes = c.bytes(c.n); hot += c.n2; }
    const char *gb = getenv("CHRYSALIS_FN_SPILL_GB");
    x.spilled = 0;
    x.spill_cap = gb && hot_bytes ? std::min(hot, (int)(atof(gb) * 1e9 / (double)hot_bytes)) : hot;
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
        CK(hipEventCreateWithFlags(&ld.done, hipEventDisableTiming | hipEventBlockingSync));
    }
    x.quit = false;
    x.loader = std::thread(&Engine::experts_loader, this);
}

constexpr uint64_t POOL_MAGIC = 0x324c4f4f50465243ull;

uint64_t Engine::model_bytes() const {
    struct stat st_{};
    return file_fd >= 0 && fstat(file_fd, &st_) == 0 ? (uint64_t)st_.st_size : 0;
}

bool Engine::pool_save(const char *path) const {
    const Experts &x = xc;
    if (x.score.empty()) return false;
    const std::string tmp = std::string(path) + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const uint64_t head[4] = {POOL_MAGIC, x.score.size(), x.owner.size(), model_bytes()};
    bool ok = fwrite(head, sizeof head, 1, f) == 1 && fwrite(x.score.data(), 4, x.score.size(), f) == x.score.size() &&
              fwrite(x.owner.data(), 4, x.owner.size(), f) == x.owner.size();
    ok = fclose(f) == 0 && ok;
    return ok && rename(tmp.c_str(), path) == 0;
}

int Engine::pool_warm(const char *path) {
    Experts &x = xc;
    if (x.score.empty() || !x.on) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint64_t head[4];
    std::vector<float> s(x.score.size());
    std::vector<int> owner(x.owner.size());
    const bool ok = fread(head, sizeof head, 1, f) == 1 && head[0] == POOL_MAGIC && head[1] == s.size() && head[2] == owner.size() &&
                    head[3] == model_bytes() && fread(s.data(), 4, s.size(), f) == s.size() &&
                    fread(owner.data(), 4, owner.size(), f) == owner.size();
    fclose(f);
    if (!ok) return -1;
    std::vector<float> want(s.size(), 0.0f);
    for (int k : owner) if (k >= 0 && k < (int)want.size()) want[k] = 1e9f;
    x.score = want;
    const int start = x.placed;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < end) {
        const int before = x.placed;
        experts_place(0);
        experts_plan();
        bool busy = false;
        for (Experts::Load &ld : x.loads) busy |= ld.state.load(std::memory_order_acquire) != Experts::FREE;
        if (!busy && x.placed == before) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CK(hipStreamSynchronize(st));
    x.score = s;
    return x.placed - start;
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
        ld.state = Experts::FREE; ld.key = ld.slot = ld.out = -1;
    }
    x.jobs.clear();
    if (x.d_counts) (void)hipFree(x.d_counts);
    if (x.h_counts) (void)hipHostFree(x.h_counts);
    if (x.counted) (void)hipEventDestroy(x.counted);
    if (x.xst) (void)hipStreamDestroy(x.xst);
    x.d_counts = nullptr; x.h_counts = nullptr; x.counted = nullptr; x.xst = nullptr;
    x.counts_pending = false;
}

void Engine::experts_loader() {
    Experts &x = xc;
    for (;;) {
        Experts::Job job;
        {
            std::unique_lock<std::mutex> lk(x.mu);
            x.cv.wait(lk, [&] { return x.quit || !x.jobs.empty(); });
            if (x.quit) return;
            job = x.jobs.front();
            x.jobs.pop_front();
        }
        if (job.what < 0) {
            const int key = -1 - job.what;
            CK(hipEventSynchronize(x.loads[job.after].done));
            const HotCopy hc = hot_copy(cold[key / NE], x.home[key]);
            for (int m = 0; m < 3; m++) {
                const uintptr_t a = ((uintptr_t)hc.mirror[m] + 4095) & ~uintptr_t(4095), b = ((uintptr_t)hc.mirror[m] + hc.n[m]) & ~uintptr_t(4095);
                if (b > a) madvise((void *)a, b - a, MADV_DONTNEED);
            }
            continue;
        }
        Experts::Load &ld = x.loads[job.what];
        if (ld.out >= 0) {
            const HotCopy hc = hot_copy(cold[ld.out / NE], x.home[ld.out]);
            for (int m = 0; m < 3; m++) file_read(hc.mirror[m], hc.file[m], hc.n[m]);
        }
        const int l = ld.key / NE, h = x.home[ld.key];
        const ColdLayer &c = cold[l];
        const size_t gu = (size_t)FE * c.row_gu[c.grp(h)], dn = (size_t)D * c.row_dn;
        const uint8_t *src[3] = {c.gu_row(false, h, 0), c.gu_row(true, h, 0), c.dn_row(h, 0)};
        const size_t len[3] = {gu, gu, dn};
        CK(hipStreamWaitEvent(x.xst, ld.done, 0));
        if (c.grp(h) == 2) {
            CK(hipEventSynchronize(ld.copied));
            size_t at = 0;
            for (int m = 0; m < 3; m++) { memcpy(ld.bounce + at, src[m], len[m]); at += len[m]; }
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

void Engine::experts_place(int T) {
    Experts &x = xc;
    if (x.owner.empty()) return;
    x.tokens += T;
    x.decoding = T > 0 && T <= 8;
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
    Experts::Job freed[PLACE_MAX];
    int nfreed = 0;
    for (int k = 0; k < PLACE_MAX; k++) {
        Experts::Load &ld = x.loads[k];
        if (ld.state.load(std::memory_order_acquire) != Experts::COPYING || hipEventQuery(ld.copied) != hipSuccess) continue;
        const int key = ld.key, s = ld.slot, old = x.owner[s];
        float ms = 0;
        if (!x.decoding && hipEventElapsedTime(&ms, ld.start, ld.copied) == hipSuccess && ms > 0) {
            x.copied_bytes += cold[key / NE].bytes(x.home[key]); x.copied_ms += ms;
        }
        pa.p[pa.n++] = {ld.stage, s, x.type[key]};
        if (old >= 0) {
            pa.patch[pa.npatch++] = {old, -1, -1 - x.home[old]};
            x.slot[old] = -1;
            const int lo = old / NE;
            L[lo].slot_h[old % NE] = -1 - x.home[old];
        }
        pa.patch[pa.npatch++] = {key, -1, s};
        pa.patch[pa.npatch++] = {-1, s, x.type[key]};
        x.slot[key] = s; x.owner[s] = key; x.fmt[s] = (uint8_t)x.type[key];
        L[key / NE].slot_h[key % NE] = s;
        x.busy[s] = 0; x.coming[key] = 0;
        x.placed++;
        if (cold[key / NE].grp(x.home[key]) == 2) {
            x.loaded_file++;
            x.spilled--;
            freed[nfreed++] = {-1 - key, k};
        }
        placed[np++] = k;
    }
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
        if (nfreed) {
            std::lock_guard<std::mutex> lk(x.mu);
            x.jobs.insert(x.jobs.end(), freed, freed + nfreed);
        }
        x.cv.notify_one();
    }
}

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
        if (picks) {
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
    int free_k[PLACE_MAX], nf = 0, busy = 0;
    for (int k = 0; k < PLACE_MAX; k++) if (x.loads[k].state.load(std::memory_order_acquire) == Experts::FREE) free_k[nf++] = k; else busy++;
    if (x.decoding) nf = std::max(0, std::min(nf, x.decode_loads - busy));
    if (!nf) return;
    std::vector<std::pair<float, int>> in, out;
    in.reserve(nf + 1); out.reserve(nf + 1);
    auto keep = [nf](std::vector<std::pair<float, int>> &v, std::pair<float, int> e, bool largest) {
        auto better = [largest](const std::pair<float, int> &a, const std::pair<float, int> &b) { return largest ? a.first > b.first : a.first < b.first; };
        if ((int)v.size() == nf && !better(e, v.back())) return;
        v.insert(std::upper_bound(v.begin(), v.end(), e, better), e);
        if ((int)v.size() > nf) v.pop_back();
    };
    auto file_tier = [&](int k) { return k >= 0 && cold[k / NE].grp(x.home[k]) == 2; };
    const double avail = mem_available(x);
    const bool yield = x.spilled > 0 && avail < 0.75 * x.spare;
    for (size_t k = 0; k < keys; k++)
        if (x.slot[k] < 0 && !x.coming[k] && (yield ? file_tier((int)k) : x.score[k] > 0)) keep(in, {x.score[k] * x.mb[k], (int)k}, true);
    if (in.empty()) return;
    int room = x.spill_cap - x.spilled;
    if (room > 0 && avail < x.spare) room = 0;
    for (int s = 0; s < n_slots; s++)
        if (!x.busy[s] && (room > 0 || !file_tier(x.owner[s])))
            keep(out, {x.owner[s] >= 0 ? x.score[x.owner[s]] * x.mb[x.owner[s]] : -1.0f, s}, false);
    bool queued = false;
    for (int i = 0, j = 0, q = 0; i < (int)in.size() && j < (int)out.size(); i++, j++) {
        while (j < (int)out.size() && room <= 0 && file_tier(x.owner[out[j].second])) j++;
        if (j == (int)out.size() || (!yield && in[i].first <= out[j].first * (1 + MARGIN) + MIN_GAIN)) break;
        Experts::Load &ld = x.loads[free_k[q]];
        ld.key = in[i].second; ld.slot = out[j].second;
        ld.out = file_tier(x.owner[ld.slot]) ? x.owner[ld.slot] : -1;
        if (ld.out >= 0) { x.spilled++; room--; }
        x.coming[ld.key] = 1; x.busy[ld.slot] = 1;
        {
            std::lock_guard<std::mutex> lk(x.mu);
            ld.state.store(Experts::QUEUED, std::memory_order_release);
            x.jobs.push_back({free_k[q++], -1});
        }
        queued = true;
    }
    if (queued) x.cv.notify_one();
}
