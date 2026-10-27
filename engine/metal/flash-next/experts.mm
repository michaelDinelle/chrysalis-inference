#include "engine.h"
#include <algorithm>
#include <cmath>
#include <pthread/qos.h>
#include <dlfcn.h>
#include <numeric>
#include <sys/stat.h>

static std::string stats_path() {
    if (const char *p = getenv("CHRYSALIS_FN_STATS")) return p;
    const char *h = getenv("HOME");
    return std::string(h ? h : "/tmp") + "/.cache/chrysalis-fn-mac-experts.bin";
}

static std::string profile_path() {
    if (const char *p = getenv("CHRYSALIS_FN_PROFILE")) return p;
    Dl_info di;
    char real[PATH_MAX];
    if (!dladdr((void *)&profile_path, &di) || !realpath(di.dli_fname, real)) return "";
    std::string d = real;
    d = d.substr(0, d.find_last_of('/'));
    return d.substr(0, d.find_last_of('/')) + "/engine/metal/flash-next/expert-profile.bin";
}

void Engine::keep_warm() {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [m.q2 commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:m.pipe("keep_warm")];
        [en setBuffer:warm_buf.buf offset:warm_buf.off atIndex:0];
        [en dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [en endEncoding];
        [cb commit];
    }
}

void Engine::io_submit(const std::vector<Experts::Job> &jobs, bool warm) {
    if (jobs.empty()) return;
    Experts &x = xc;
    {
        std::lock_guard<std::mutex> lk(x.mu);
        for (const auto &j : jobs) x.jobs.push_back(j);
        x.inflight += (int)jobs.size();
    }
    x.cv.notify_all();
    std::unique_lock<std::mutex> lk(x.mu);
    if (!warm) { x.done_cv.wait(lk, [&] { return x.inflight == 0; }); return; }
    while (!x.done_cv.wait_for(lk, std::chrono::microseconds(50), [&] { return x.inflight == 0; })) {
        lk.unlock();
        keep_warm();
        lk.lock();
    }
}

void Engine::experts_init(size_t cache_bytes, int n_stage) {
    Experts &x = xc;
    const int all = (int)(cache_bytes / XSLOT);
    int ring = NE;
    for (int it = 0; it < 4; it++) {
        const int cache = std::min(all - n_stage - 2 * ring, NL * NE);
        const int ml = std::max(0, std::min(cache / NL - 8, NE));
        ring = NE - ml;
    }
    x.n_slots = std::min(all - n_stage - 2 * ring, NL * NE);
    if (x.n_slots < NL * 64) FAIL("expert cache of %d slots is too small", x.n_slots);
    x.n_stage = n_stage;
    x.ring0 = x.n_slots + n_stage;
    x.ring = ring;
    const int total = x.n_slots + n_stage + 2 * ring;
    const size_t per_buf = ((size_t)4 << 30) / XSLOT;
    for (int s0 = 0; s0 < total; s0 += (int)per_buf) {
        const int n = std::min<int>((int)per_buf, total - s0);
        View v = m.alloc((size_t)n * XSLOT);
        if (mlock(v.host(), (size_t)n * XSLOT)) fprintf(stderr, "chrysalis-fn: cannot wire the expert cache (%s); it stays pageable\n", strerror(errno));
        x.bufs.push_back(v);
        for (int i = 0; i < n; i++) { x.slot_host.push_back(v.host<uint8_t>() + (size_t)i * XSLOT); x.slot_gpu.push_back(v.gpu() + (size_t)i * XSLOT); }
    }
    x.owner.assign(total, -1);
    x.last_use.assign(total, 0);
    x.slot_of.assign((size_t)NL * NE, -1);
    x.score.assign((size_t)NL * NE, 0.f);
    x.per_layer.assign(NL, 0);
    x.stage_free.clear();
    for (int s = x.n_slots + n_stage - 1; s >= x.n_slots; s--) x.stage_free.push_back(s);
    x.ahead_of.assign((size_t)NL * NE, -1);
    x.reserved.assign(total, 0);
    x.table = m.alloc((size_t)(NL + 1) * NE * 8);
    warm_buf = m.alloc(256);
    memset(x.table.host(), 0, (size_t)(NL + 1) * NE * 8);
    x.fmt = m.alloc((size_t)(NL + 1) * NE);
    for (int i = 0; i < NL * NE; i++) x.fmt.host<uint8_t>()[i] = (uint8_t)x.src[i].type;
    x.fd = dup(g.fds[0]);
    fcntl(x.fd, F_NOCACHE, 1);
    x.quit = false;
    for (int k = 0, nk = getenv("CHRYSALIS_FN_IO_THREADS") ? atoi(getenv("CHRYSALIS_FN_IO_THREADS")) : 16; k < nk; k++)
        x.io.emplace_back([this] {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            Experts &x = xc;
            for (;;) {
                Experts::Job j;
                {
                    std::unique_lock<std::mutex> lk(x.mu);
                    x.cv.wait(lk, [&] { return x.quit || !x.jobs.empty(); });
                    if (x.quit) return;
                    j = x.jobs.front();
                    x.jobs.pop_front();
                }
                for (size_t done = 0; done < j.n;) {
                    const ssize_t r = pread(j.fd, j.dst + done, j.n - done, j.off + done);
                    if (r <= 0) FAIL("expert read failed at %zu: %s", j.off + done, strerror(errno));
                    done += r;
                }
                if (j.left) { j.left->fetch_sub(1, std::memory_order_release); continue; }
                std::lock_guard<std::mutex> lk(x.mu);
                if (--x.inflight == 0) x.done_cv.notify_all();
            }
        });
    std::vector<float> prior((size_t)NL * NE, 0.f);
    for (const std::string &p : {stats_path(), profile_path()}) {
        FILE *f = fopen(p.c_str(), "rb");
        if (!f) continue;
        const bool ok = fread(prior.data(), 4, prior.size(), f) == prior.size();
        fclose(f);
        if (ok) { fprintf(stderr, "chrysalis-fn: expert prior %s\n", p.c_str()); break; }
        std::fill(prior.begin(), prior.end(), 0.f);
    }
    x.counts = prior;
    static const double prior_w = getenv("CHRYSALIS_FN_PRIOR_W") ? atof(getenv("CHRYSALIS_FN_PRIOR_W")) : 0.25;
    const double sum = std::accumulate(prior.begin(), prior.end(), 0.0);
    if (sum > 0) for (float &v : prior) v = (float)(v * prior_w * NL * TOPK_E / (1 - x.decay) / sum);
    x.score = prior;
    x.min_layer = NE - ring;
}

void Engine::experts_ahead(int layer, const int *ids, int n) {
    Experts &x = xc;
    uint64_t *table = (uint64_t *)x.table.host();
    std::vector<Experts::Job> jobs;
    for (int i = 0; i < n; i++) {
        const int key = layer * NE + ids[i];
        if (x.slot_of[key] >= 0 || x.ahead_of[key] >= 0) continue;
        int victim = -1;
        float vs = INFINITY;
        for (int k = 0; k < 64; k++) {
            const int c = std::uniform_int_distribution<int>(0, x.n_slots - 1)(x.rng);
            if (x.last_use[c] == x.pass || x.reserved[c]) continue;
            const int o = x.owner[c];
            if (o < 0) { victim = c; vs = -1; break; }
            if (o / NE == layer || (o / NE != layer && x.per_layer[o / NE] <= x.min_layer)) continue;
            if (x.score[o] < vs) { vs = x.score[o]; victim = c; }
        }
        if (victim < 0 || !(x.score[key] + x.boost > vs)) continue;
        const int o = x.owner[victim];
        if (o >= 0) { table[o] = 0; x.slot_of[o] = -1; x.per_layer[o / NE]--; }
        x.owner[victim] = key;
        x.reserved[victim] = 1;
        auto h = std::make_unique<Experts::Ahead>();
        h->key = key; h->slot = victim; h->left = 3;
        expert_jobs(layer, ids[i], x.slot_host[victim], &h->left, jobs);
        x.ahead_of[key] = (int)x.ahead.size();
        x.ahead.push_back(std::move(h));
        x.ahead_reads++;
    }
    if (jobs.empty()) return;
    {
        std::lock_guard<std::mutex> lk(x.mu);
        for (const auto &j : jobs) x.jobs.push_back(j);
    }
    x.cv.notify_all();
}

void Engine::experts_stage_layer(int layer, int half) {
    Experts &x = xc;
    uint64_t *table = (uint64_t *)x.table.host();
    std::vector<Experts::Job> jobs;
    std::vector<int> &keys = x.ring_keys[half];
    keys.clear();
    for (int e = 0; e < NE; e++) {
        const int key = layer * NE + e;
        if (x.slot_of[key] >= 0) continue;
        if ((int)keys.size() >= x.ring) FAIL("layer %d: more than %d experts out of the cache", layer, x.ring);
        uint8_t *dst = x.slot_host[x.ring0 + half * x.ring + keys.size()];
        expert_jobs(layer, e, dst, nullptr, jobs);
        keys.push_back(key);
    }
    const double t0 = now_s();
    io_submit(jobs);
    x.read_s += now_s() - t0;
    for (const auto &j : jobs) x.read_bytes += j.n;
    x.staged += keys.size();
    for (size_t i = 0; i < keys.size(); i++) table[keys[i]] = x.slot_gpu[x.ring0 + half * x.ring + i];
}

void Engine::experts_unstage(int half) {
    uint64_t *table = (uint64_t *)xc.table.host();
    for (int key : xc.ring_keys[half]) table[key] = 0;
    xc.ring_keys[half].clear();
}

void Engine::experts_read(int layer, int e, uint8_t *dst) {
    std::vector<Experts::Job> jobs;
    expert_jobs(layer, e, dst, nullptr, jobs);
    io_submit(jobs);
}

void Engine::expert_jobs(int layer, int e, uint8_t *dst, std::atomic<int> *left, std::vector<Experts::Job> &jobs) const {
    const XSrc &s = xc.src[(size_t)layer * NE + e];
    const size_t gu = xgu_bytes(s.type);
    jobs.push_back({xc.fd, s.off[0], XDN, dst, left});
    jobs.push_back({xc.fd, s.off[1], gu, dst + XDN, left});
    jobs.push_back({xc.fd, s.off[2], gu, dst + XDN + gu, left});
}

void Engine::experts_fill(double) {
    Experts &x = xc;
    std::vector<int> order((size_t)NL * NE);
    for (int i = 0; i < NL * NE; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x.score[a] > x.score[b]; });
    std::vector<int> chosen, per(NL, 0);
    std::vector<uint8_t> taken((size_t)NL * NE, 0);
    for (int key : order) if (per[key / NE] < x.min_layer) { taken[key] = 1; per[key / NE]++; chosen.push_back(key); }
    for (int key : order) { if ((int)chosen.size() >= x.n_slots) break; if (!taken[key]) { taken[key] = 1; chosen.push_back(key); } }
    std::vector<Experts::Job> jobs;
    for (size_t s = 0; s < chosen.size(); s++) {
        const int key = chosen[s], l = key / NE, e = key % NE;
        expert_jobs(l, e, x.slot_host[s], nullptr, jobs);
        x.owner[s] = key; x.slot_of[key] = (int)s; x.per_layer[l]++;
        ((uint64_t *)x.table.host())[key] = x.slot_gpu[s];
    }
    double bytes = 0;
    for (const auto &j : jobs) bytes += j.n;
    const double t0 = now_s();
    io_submit(jobs);
    const double dt = now_s() - t0;
    fprintf(stderr, "chrysalis-fn: expert cache: %d slots filled in %.1f s (%.2f GB/s), every layer >= %d\n", (int)chosen.size(), dt,
            bytes / dt / 1e9, x.min_layer);
}

int Engine::experts_resolve(int layer, const int *ids, int n, const std::function<void()> &publish_ok) {
    Experts &x = xc;
    uint64_t *table = (uint64_t *)x.table.host();
    for (int s = x.n_slots; s < x.n_slots + x.n_stage; s++)
        if (x.owner[s] >= 0) { table[x.owner[s]] = 0; x.owner[s] = -1; x.stage_free.push_back(s); }
    const uint32_t stamp = ++x.pass;
    int uniq[NE], nu = 0;
    for (int i = 0; i < n; i++) {
        bool dup = false;
        for (int j = 0; j < nu; j++) dup |= uniq[j] == ids[i];
        if (!dup) uniq[nu++] = ids[i];
        x.score[layer * NE + ids[i]] += x.boost;
        if ((x.counts[layer * NE + ids[i]] += 1) > 1e7f) for (float &v : x.counts) v *= 0.5f;
    }
    x.picks += n;
    for (int i = 0; i < nu; i++) { const int s = x.slot_of[layer * NE + uniq[i]]; if (s >= 0) x.last_use[s] = stamp; }
    std::vector<Experts::Job> jobs;
    std::vector<std::pair<int, int>> placed;
    std::vector<std::unique_ptr<Experts::Ahead>> mine;
    for (int e = 0; e < NE && !x.ahead.empty(); e++) {
        const int key = layer * NE + e, a = x.ahead_of[key];
        if (a < 0) continue;
        Experts::Ahead &h = *x.ahead[a];
        x.reserved[h.slot] = 0;
        x.slot_of[key] = h.slot; x.per_layer[layer]++;
        x.last_use[h.slot] = stamp;
        placed.push_back({key, h.slot});
        x.ahead_of[key] = -1;
        for (int k = 0; k < n; k++) if (ids[k] == e) { x.ahead_used++; break; }
    }
    for (auto &h : x.ahead) if (h->key / NE == layer) mine.push_back(std::move(h));
    x.ahead.erase(std::remove_if(x.ahead.begin(), x.ahead.end(), [](const std::unique_ptr<Experts::Ahead> &h) { return !h; }), x.ahead.end());
    for (size_t i = 0; i < x.ahead.size(); i++) x.ahead_of[x.ahead[i]->key] = (int)i;
    static const bool admit_all = !getenv("CHRYSALIS_FN_ADMIT_SCORE");
    for (int i = 0; i < nu; i++) {
        const int key = layer * NE + uniq[i];
        int s = x.slot_of[key];
        if (s >= 0 && std::find(placed.begin(), placed.end(), std::make_pair(key, s)) != placed.end()) continue;
        if (s >= 0) {
            x.last_use[s] = stamp;
            for (int k = 0; k < n; k++) x.hits += ids[k] == uniq[i];
            continue;
        }
        int victim = -1;
        float vs = INFINITY;
        for (int k = 0; k < 64 && x.n_slots; k++) {
            const int c = std::uniform_int_distribution<int>(0, x.n_slots - 1)(x.rng);
            if (x.last_use[c] == stamp || x.reserved[c]) continue;
            const int o = x.owner[c];
            if (o < 0) { victim = c; vs = -1; break; }
            if (o / NE != layer && x.per_layer[o / NE] <= x.min_layer) continue;
            if (x.score[o] < vs) { vs = x.score[o]; victim = c; }
        }
        if (victim >= 0 && (admit_all || x.score[key] > vs)) {
            const int o = x.owner[victim];
            if (o >= 0) { table[o] = 0; x.slot_of[o] = -1; x.per_layer[o / NE]--; }
            x.owner[victim] = key; x.slot_of[key] = victim; x.per_layer[layer]++;
            s = victim;
        } else {
            if (x.stage_free.empty()) FAIL("expert staging slots exhausted (%d)", x.n_stage);
            s = x.stage_free.back();
            x.stage_free.pop_back();
            x.owner[s] = key;
            x.staged++;
        }
        x.last_use[s] = stamp;
        expert_jobs(layer, uniq[i], x.slot_host[s], nullptr, jobs);
        placed.push_back({key, s});
    }
    if (!jobs.empty()) {
        const double t0 = now_s();
        static const bool warm = !getenv("CHRYSALIS_FN_NOWARM");
        if (warm) keep_warm();
        io_submit(jobs, warm);
        x.read_s += now_s() - t0;
        for (const auto &j : jobs) x.read_bytes += j.n;
        x.reads += placed.size();
    }
    for (auto &h : mine) while (h->left.load(std::memory_order_acquire) > 0) std::this_thread::yield();
    if (!placed.empty()) publish_ok();
    for (auto [key, s] : placed) table[key] = x.slot_gpu[s];
    return (int)placed.size();
}

static void save_scores(const Experts &x) {
    if (x.counts.empty()) return;
    const std::vector<float> &s = x.counts;
    const std::string p = stats_path(), tmp = p + ".tmp";
    mkdir(p.substr(0, p.find_last_of('/')).c_str(), 0755);
    if (FILE *f = fopen(tmp.c_str(), "wb")) {
        const bool ok = fwrite(s.data(), 4, s.size(), f) == s.size();
        if (fclose(f) == 0 && ok) rename(tmp.c_str(), p.c_str());
    }
}

void Engine::experts_tick(int T) {
    Experts &x = xc;
    x.boost /= std::pow(x.decay, T);
    if (x.boost > 1e20) {
        for (float &v : x.score) v = (float)(v / x.boost);
        x.boost = 1.0;
    }
    static double saved = now_s();
    if (now_s() - saved > 30) { saved = now_s(); save_scores(x); }
}

void Engine::experts_stop() {
    Experts &x = xc;
    {
        std::lock_guard<std::mutex> lk(x.mu);
        x.quit = true;
    }
    x.cv.notify_all();
    for (auto &t : x.io) t.join();
    x.io.clear();
    save_scores(x);
    if (x.fd >= 0) ::close(x.fd);
    x.fd = -1;
}
