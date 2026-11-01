#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

constexpr int MOE_MAX_T = 8;
constexpr int MOE_MAX_E = 10;
constexpr int VNNI_MIN_TOKENS = 3;

struct MoeShared {
    alignas(64) uint32_t go;
    alignas(64) uint32_t done;
    alignas(64) int layer, T;
    int slot[MOE_MAX_T * MOE_MAX_E];
    float w[MOE_MAX_T * MOE_MAX_E];
    alignas(64) float x[MOE_MAX_T * 2560];
    alignas(64) float y[MOE_MAX_T * 2560];
};

struct MoeEmu {
    alignas(64) uint32_t ready;
    alignas(64) uint32_t flag[MOE_MAX_T * MOE_MAX_E][20];
    alignas(64) float gu[MOE_MAX_T * MOE_MAX_E][2 * 640];
    alignas(64) float y[MOE_MAX_T * MOE_MAX_E][2560];
    alignas(64) int8_t hq[MOE_MAX_T * MOE_MAX_E][640];
    float hd[MOE_MAX_T * MOE_MAX_E][20];
};

struct EmuAct {
    alignas(64) int8_t q[2560];
    alignas(64) float d[80];
    alignas(64) int32_t s16[160];
    alignas(64) int32_t c32[80];
};

struct ColdLayer {
    const uint8_t *gate[3] = {}, *up[3] = {}, *down = nullptr, *down2 = nullptr;
    const uint8_t *file_gate = nullptr, *file_up = nullptr, *file_down2 = nullptr;
    int type_gu[3] = {-1, -1, -1}, type_dn = -1;
    size_t row_gu[3] = {}, row_dn = 0;
    int n0 = 0, n = 0, n2 = 0;
    int grp(int s) const { return s < n0 ? 0 : s < n ? 1 : 2; }
    int idx(int s) const { return s < n0 ? s : s < n ? s - n0 : s - n; }
    const uint8_t *gu_row(bool is_up, int s, int r) const {
        const int g = grp(s);
        return (is_up ? up[g] : gate[g]) + ((size_t)idx(s) * 640 + r) * row_gu[g];
    }
    const uint8_t *dn_row(int s, int r) const {
        return s < n ? down + ((size_t)s * 2560 + r) * row_dn : down2 + ((size_t)(s - n) * 2560 + r) * row_dn;
    }
    size_t bytes(int s) const { return 2 * 640 * row_gu[grp(s)] + 2560 * row_dn; }
};

struct MoeJob {
    int layer = 0, T = 0, n_exp = 0;
    const uint16_t *x = nullptr;
    const int *slot = nullptr;
    const int *first = nullptr;
    const int *tok = nullptr;
    const float *w = nullptr;
    uint16_t *out = nullptr;
};

struct VnniScratch {
    uint8_t *w = nullptr; float *ws = nullptr, *wa = nullptr, *wb = nullptr;
    uint32_t *xi = nullptr; float *xs = nullptr; int32_t *xc = nullptr;
    float *xf = nullptr;
    float *h = nullptr; uint32_t *hi = nullptr; float *hs = nullptr; int32_t *hc = nullptr;
    float *y = nullptr;
    void reserve();
    ~VnniScratch();
};
bool vnni_supported();
bool vnni_formats(const ColdLayer &c);
void vnni_quant_x(const float *x, int8_t *q, float *s, int32_t *sum);
void expert_vnni(const ColdLayer &c, int slot, int n, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
                 const float *w, uint16_t *const *out, VnniScratch &S);
void vnni_act1(const float *xs, const int32_t *xsum, float *s16, float *f16);
void vnni_act_x(const int32_t *xsum, float *xf);
void vnni_dot_gu2(const uint8_t *r0, const uint8_t *r1, const int8_t *xq, const float *xs, const float *xf, float *out);
float vnni_dot_gu1(const uint8_t *row, int type, const int8_t *xq, const float *xs, const int32_t *xsum, const float *s16,
                   const float *f16, VnniScratch &S);
void vnni_quant_h(const float *h, int8_t *q, float *s, float *c);
float vnni_dot_dn1(const uint8_t *row, const int8_t *hq, const float *hs, const float *hc);
void emu_quant_x(const float *x, EmuAct &a);
float emu_gu_row(const uint8_t *row, int type, const EmuAct &a, VnniScratch &S);
float emu_dn_row(const uint8_t *row, const int8_t *hq, const float *hd, const int32_t *hc, VnniScratch &S);

class CpuMoe {
public:
    void start(MoeShared *shared, MoeShared *pred, const std::vector<ColdLayer> &layers, int threads, MoeEmu *emu = nullptr);
    bool can_emulate() const { return me && vnni_ok; }
    void stop();
    void arm();
    void disarm();
    void submit(const MoeJob &job);
    void wait();
    bool use_vnni(int on) { const bool was = vnni; vnni = on && vnni_ok; return was; }
    ~CpuMoe() { stop(); }

private:
    void worker(int id);
    void prefill_work(int id);
    void preload(int id, uint32_t real, uint32_t job);
    MoeShared *sh = nullptr, *pr = nullptr;
    MoeEmu *me = nullptr;
    EmuAct ea[MOE_MAX_T];
    std::vector<ColdLayer> L;
    std::vector<std::thread> th;
    int n = 0;
    std::atomic<int> armed{0};
    std::atomic<uint32_t> wake{0};
    std::atomic<bool> quit{false};
    std::atomic<uint32_t> seen{0};
    std::atomic<int> bar_count{0};
    std::atomic<uint32_t> bar_gen{0};
    void barrier();
    std::vector<float> gu;
    std::vector<uint8_t> xq[3];
    std::vector<uint8_t> hq;
    std::vector<int8_t> dxq, dhq; std::vector<float> dxs, ds16, df16, dxf, dhs, dhc; std::vector<int32_t> dxsum;
    size_t xq_row[3] = {}, hq_row = 0;
    int dot_gu[3] = {-1, -1, -1};
    bool dot_same = true;
    MoeJob job;
    std::atomic<uint32_t> jgen{0};
    uint32_t jstart = 0;
    std::atomic<uint64_t> sink_{0};

    std::atomic<int> jtok{0}, jexp{0}, jfin{0};
    std::vector<uint8_t> pxq[3];
    std::vector<int8_t> vxq; std::vector<float> vxs; std::vector<int32_t> vxsum;
    bool vnni = false, vnni_ok = false;
    struct Scratch { std::vector<float> gu, y; std::vector<uint8_t> hq; VnniScratch v; };
    std::vector<Scratch> scr;
};
