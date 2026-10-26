#include "cpu_moe.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int D_ = 2560, FE_ = 640;
constexpr int NB_GU = D_ / 32, NB_DN = FE_ / 32;
alignas(16) const uint8_t KV128[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};

#define VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c,avx2,fma,bmi2")))

VNNI void decode_iq4xs(const uint8_t *row, int K, uint8_t *w, float *ws) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)KV128)), m4 = _mm256_set1_epi8(0x0f);
    for (int sb = 0; sb < K / 256; sb++) {
        const uint8_t *blk = row + sb * 136;
        const uint64_t lo4 = _pdep_u64(*(const uint32_t *)(blk + 4), 0x0F0F0F0F0F0F0F0Full);
        const uint64_t hi2 = _pdep_u64(*(const uint16_t *)(blk + 2), 0x0303030303030303ull);
        const __m256i ls = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)(lo4 | hi2 << 4)));
        _mm256_storeu_ps(ws + sb * 8, _mm256_mul_ps(_mm256_set1_ps(_cvtsh_ss(*(const uint16_t *)blk)),
                                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(ls, _mm256_set1_epi32(32)))));
        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 8 + 16 * ib));
            const __m256i lo = _mm256_shuffle_epi8(tbl, _mm256_and_si256(q, m4));
            const __m256i hi = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
            uint8_t *o = w + (sb * 8 + ib) * 32;
            _mm256_storeu_si256((__m256i *)o, _mm256_permute2x128_si256(lo, hi, 0x20));
            _mm256_storeu_si256((__m256i *)(o + 32), _mm256_permute2x128_si256(lo, hi, 0x31));
        }
    }
}

VNNI void decode_iq4nl(const uint8_t *row, int K, uint8_t *w, float *ws) {
    const __m128i tbl = _mm_load_si128((const __m128i *)KV128), m4 = _mm_set1_epi8(0x0f);
    for (int b = 0; b < K / 32; b++) {
        const uint8_t *blk = row + b * 18;
        const __m128i q = _mm_loadu_si128((const __m128i *)(blk + 2));
        _mm_storeu_si128((__m128i *)(w + b * 32), _mm_shuffle_epi8(tbl, _mm_and_si128(q, m4)));
        _mm_storeu_si128((__m128i *)(w + b * 32 + 16), _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q, 4), m4)));
        ws[b] = _cvtsh_ss(*(const uint16_t *)blk);
    }
}

VNNI void decode_q3k(const uint8_t *row, int K, uint8_t *w, float *wa, float *wb) {
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    for (int sb = 0; sb < K / 256; sb++) {
        const uint8_t *blk = row + sb * 110;
        uint32_t a[3];
        memcpy(a, blk + 96, 12);
        const uint32_t s0 = (a[0] & 0x0f0f0f0f) | ((a[2] & 0x03030303) << 4), s1 = (a[1] & 0x0f0f0f0f) | (((a[2] >> 2) & 0x03030303) << 4);
        const uint32_t s2 = ((a[0] >> 4) & 0x0f0f0f0f) | (((a[2] >> 4) & 0x03030303) << 4), s3 = ((a[1] >> 4) & 0x0f0f0f0f) | (((a[2] >> 6) & 0x03030303) << 4);
        const __m512 sc = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_set_epi32((int)s3, (int)s2, (int)s1, (int)s0)), _mm512_set1_epi32(32)));
        const __m512 av = _mm512_mul_ps(_mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)(blk + 108))), sc);
        _mm512_storeu_ps(wa + sb * 16, av);
        _mm512_storeu_ps(wb + sb * 16, _mm512_mul_ps(av, _mm512_set1_ps(4.0f)));
        const __m256i hm = _mm256_loadu_si256((const __m256i *)blk);
        for (int n = 0; n < 2; n++) {
            const __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 + 32 * n));
            for (int j = 0; j < 4; j++) {
                const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3);
                const __m256i hb = _mm256_and_si256(_mm256_srli_epi16(hm, 4 * n + j), one);
                _mm256_storeu_si256((__m256i *)(w + sb * 256 + 128 * n + 32 * j), _mm256_or_si256(lo, _mm256_slli_epi16(hb, 2)));
            }
        }
    }
}

VNNI void decode_q2k(const uint8_t *row, int K, uint8_t *w, float *wa, float *wb) {
    const __m256i m3 = _mm256_set1_epi8(3);
    for (int sb = 0; sb < K / 256; sb++) {
        const uint8_t *blk = row + sb * 84;
        const __m512i s = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)blk));
        _mm512_storeu_ps(wa + sb * 16, _mm512_mul_ps(_mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)(blk + 80))),
                                                     _mm512_cvtepi32_ps(_mm512_and_si512(s, _mm512_set1_epi32(15)))));
        _mm512_storeu_ps(wb + sb * 16, _mm512_mul_ps(_mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)(blk + 82))),
                                                     _mm512_cvtepi32_ps(_mm512_srli_epi32(s, 4))));
        for (int n = 0; n < 2; n++) {
            const __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 16 + 32 * n));
            for (int j = 0; j < 4; j++)
                _mm256_storeu_si256((__m256i *)(w + sb * 256 + 128 * n + 32 * j), _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3));
        }
    }
}

template <int L> struct Lanes;
template <> struct Lanes<16> {
    using I = __m512i; using F = __m512;
    VNNI static I zero() { return _mm512_setzero_si512(); }
    VNNI static I dp(I s, int32_t w, const uint32_t *x) { return _mm512_dpbusd_epi32(s, _mm512_set1_epi32(w), _mm512_load_si512(x)); }
    VNNI static F block(I s0, I s1, const int32_t *xc, float ws, const float *xs, F a) {
        const F v = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_add_epi32(s0, s1), _mm512_load_si512(xc)));
        return _mm512_fmadd_ps(v, _mm512_mul_ps(_mm512_set1_ps(ws), _mm512_load_ps(xs)), a);
    }
    VNNI static F fzero() { return _mm512_setzero_ps(); }
    VNNI static void store(float *o, F a, F b) { _mm512_store_ps(o, _mm512_add_ps(a, b)); }
    VNNI static F block16(I s0, I s1, float a0, float a1, const float *xs, F a) {
        const F v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s1), _mm512_set1_ps(a1), _mm512_mul_ps(_mm512_cvtepi32_ps(s0), _mm512_set1_ps(a0)));
        return _mm512_fmadd_ps(v, _mm512_load_ps(xs), a);
    }
    VNNI static F offs16(float b0, const float *xf0, float b1, const float *xf1, F a) {
        return _mm512_fnmadd_ps(_mm512_set1_ps(b1), _mm512_load_ps(xf1), _mm512_fnmadd_ps(_mm512_set1_ps(b0), _mm512_load_ps(xf0), a));
    }
};
template <> struct Lanes<8> {
    using I = __m256i; using F = __m256;
    VNNI static I zero() { return _mm256_setzero_si256(); }
    VNNI static I dp(I s, int32_t w, const uint32_t *x) { return _mm256_dpbusd_epi32(s, _mm256_set1_epi32(w), _mm256_load_si256((const __m256i *)x)); }
    VNNI static F block(I s0, I s1, const int32_t *xc, float ws, const float *xs, F a) {
        const F v = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_add_epi32(s0, s1), _mm256_load_si256((const __m256i *)xc)));
        return _mm256_fmadd_ps(v, _mm256_mul_ps(_mm256_set1_ps(ws), _mm256_load_ps(xs)), a);
    }
    VNNI static F fzero() { return _mm256_setzero_ps(); }
    VNNI static void store(float *o, F a, F b) { _mm256_store_ps(o, _mm256_add_ps(a, b)); }
    VNNI static F block16(I s0, I s1, float a0, float a1, const float *xs, F a) {
        const F v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_set1_ps(a1), _mm256_mul_ps(_mm256_cvtepi32_ps(s0), _mm256_set1_ps(a0)));
        return _mm256_fmadd_ps(v, _mm256_load_ps(xs), a);
    }
    VNNI static F offs16(float b0, const float *xf0, float b1, const float *xf1, F a) {
        return _mm256_fnmadd_ps(_mm256_set1_ps(b1), _mm256_load_ps(xf1), _mm256_fnmadd_ps(_mm256_set1_ps(b0), _mm256_load_ps(xf0), a));
    }
};

template <int L>
VNNI inline void row_dot(const uint8_t *w, const float *ws, int nb, const uint32_t *xi, const float *xs, const int32_t *xc, float *out) {
    using V = Lanes<L>;
    typename V::F a0 = V::fzero(), a1 = V::fzero();
    const int32_t *wd = (const int32_t *)w;
    for (int b = 0; b < nb; b += 2) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int bb = b + u;
            const uint32_t *x = xi + bb * 8 * L;
            const int32_t *wb = wd + bb * 8;
            typename V::I s0 = V::dp(V::zero(), wb[0], x), s1 = V::dp(V::zero(), wb[1], x + L);
            s0 = V::dp(s0, wb[2], x + 2 * L); s1 = V::dp(s1, wb[3], x + 3 * L);
            s0 = V::dp(s0, wb[4], x + 4 * L); s1 = V::dp(s1, wb[5], x + 5 * L);
            s0 = V::dp(s0, wb[6], x + 6 * L); s1 = V::dp(s1, wb[7], x + 7 * L);
            if (u == 0) a0 = V::block(s0, s1, xc + bb * L, ws[bb], xs + bb * L, a0);
            else a1 = V::block(s0, s1, xc + bb * L, ws[bb], xs + bb * L, a1);
        }
    }
    V::store(out, a0, a1);
}

template <int L>
VNNI inline void row_dot16(const uint8_t *w, const float *wa, const float *wb, int nb, const uint32_t *xi, const float *xs, const float *xf,
                           float *out) {
    using V = Lanes<L>;
    typename V::F a0 = V::fzero(), a1 = V::fzero();
    const int32_t *wd = (const int32_t *)w;
    for (int b = 0; b < nb; b++) {
        const uint32_t *x = xi + b * 8 * L;
        const int32_t *wq = wd + b * 8;
        typename V::I s0 = V::dp(V::zero(), wq[0], x), s1 = V::dp(V::zero(), wq[4], x + 4 * L);
        s0 = V::dp(s0, wq[1], x + L); s1 = V::dp(s1, wq[5], x + 5 * L);
        s0 = V::dp(s0, wq[2], x + 2 * L); s1 = V::dp(s1, wq[6], x + 6 * L);
        s0 = V::dp(s0, wq[3], x + 3 * L); s1 = V::dp(s1, wq[7], x + 7 * L);
        a0 = V::block16(s0, s1, wa[2 * b], wa[2 * b + 1], xs + b * L, a0);
        a1 = V::offs16(wb[2 * b], xf + 2 * b * L, wb[2 * b + 1], xf + (2 * b + 1) * L, a1);
    }
    V::store(out, a0, a1);
}

template <int L>
VNNI void group(const ColdLayer &c, int slot, int ng, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
                const float *w, uint16_t *const *out, VnniScratch &S) {
    const int fmt = c.type_gu[c.grp(slot)];
    for (int b = 0; b < NB_GU; b++) {
        for (int j = 0; j < L; j++) {
            S.xs[b * L + j] = j < ng ? xs[j][b] : 0.0f;
            S.xc[b * L + j] = j < ng ? 128 * (xsum[j][2 * b] + xsum[j][2 * b + 1]) : 0;
            S.xf[2 * b * L + j] = j < ng ? xs[j][b] * xsum[j][2 * b] : 0.0f;
            S.xf[(2 * b + 1) * L + j] = j < ng ? xs[j][b] * xsum[j][2 * b + 1] : 0.0f;
        }
        for (int k = 0; k < 8; k++)
            for (int j = 0; j < L; j++) S.xi[(b * 8 + k) * L + j] = j < ng ? *(const uint32_t *)(xq[j] + b * 32 + 4 * k) : 0u;
    }
    alignas(64) float g[16], u[16];
    auto dot_row = [&](const uint8_t *row, float *o) {
        if (fmt == 23) { decode_iq4xs(row, D_, S.w, S.ws); row_dot<L>(S.w, S.ws, NB_GU, S.xi, S.xs, S.xc, o); }
        else {
            if (fmt == 11) decode_q3k(row, D_, S.w, S.wa, S.wb);
            else decode_q2k(row, D_, S.w, S.wa, S.wb);
            row_dot16<L>(S.w, S.wa, S.wb, NB_GU, S.xi, S.xs, S.xf, o);
        }
    };
    for (int r = 0; r < FE_; r++) {
        dot_row(c.gu_row(false, slot, r), g);
        dot_row(c.gu_row(true, slot, r), u);
        for (int j = 0; j < L; j++) S.h[r * L + j] = g[j] / (1.0f + expf(-g[j])) * u[j];
    }
    for (int b = 0; b < NB_DN; b++) {
        float d[16], id[16];
        for (int j = 0; j < L; j++) {
            float m = 0;
            for (int r = 0; r < 32; r++) m = std::max(m, fabsf(S.h[(b * 32 + r) * L + j]));
            d[j] = m / 127.0f; id[j] = d[j] > 0 ? 1.0f / d[j] : 0.0f;
            S.hs[b * L + j] = d[j];
        }
        int32_t sum[16] = {};
        for (int k = 0; k < 8; k++)
            for (int j = 0; j < L; j++) {
                uint32_t dw = 0;
                for (int e = 0; e < 4; e++) {
                    const int q = (int)lrintf(S.h[(b * 32 + 4 * k + e) * L + j] * id[j]);
                    sum[j] += q;
                    dw |= (uint32_t)(uint8_t)(int8_t)q << (8 * e);
                }
                S.hi[(b * 8 + k) * L + j] = dw;
            }
        for (int j = 0; j < L; j++) S.hc[b * L + j] = 128 * sum[j];
    }
    for (int r = 0; r < D_; r++) {
        decode_iq4nl(c.dn_row(slot, r), FE_, S.w, S.ws);
        row_dot<L>(S.w, S.ws, NB_DN, S.hi, S.hs, S.hc, S.y + (size_t)r * L);
    }
    alignas(64) float col[16];
    for (int j = 0; j < ng; j++) {
        uint16_t *o = out[j];
        for (int r = 0; r < D_; r += 16) {
            for (int e = 0; e < 16; e++) col[e] = S.y[(size_t)(r + e) * L + j] * w[j];
            _mm256_storeu_si256((__m256i *)(o + r), _mm512_cvtps_ph(_mm512_load_ps(col), _MM_FROUND_TO_NEAREST_INT));
        }
    }
}
}

void VnniScratch::reserve() {
    if (w) return;
    auto get = [](size_t bytes) { return aligned_alloc(64, (bytes + 63) / 64 * 64); };
    w = (uint8_t *)get(D_); ws = (float *)get(NB_GU * 4); wa = (float *)get(NB_GU * 2 * 4); wb = (float *)get(NB_GU * 2 * 4);
    xi = (uint32_t *)get(NB_GU * 8 * 16 * 4); xs = (float *)get(NB_GU * 16 * 4); xc = (int32_t *)get(NB_GU * 16 * 4);
    xf = (float *)get(NB_GU * 2 * 16 * 4);
    h = (float *)get(FE_ * 16 * 4); hi = (uint32_t *)get(NB_DN * 8 * 16 * 4); hs = (float *)get(NB_DN * 16 * 4); hc = (int32_t *)get(NB_DN * 16 * 4);
    y = (float *)get((size_t)D_ * 16 * 4);
}
VnniScratch::~VnniScratch() {
    for (void *p : {(void *)w, (void *)ws, (void *)wa, (void *)wb, (void *)xi, (void *)xs, (void *)xc, (void *)xf, (void *)h, (void *)hi,
                    (void *)hs, (void *)hc, (void *)y}) free(p);
}

bool vnni_supported() { return __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("bmi2"); }
bool vnni_formats(const ColdLayer &c) {
    for (int g = 0; g < 3; g++)
        if (c.gate[g] && c.type_gu[g] != 23 && c.type_gu[g] != 11 && c.type_gu[g] != 10) return false;
    return c.type_dn == 20;
}

VNNI void vnni_quant_x(const float *x, int8_t *q, float *s, int32_t *sum) {
    for (int b = 0; b < NB_GU; b++) {
        const __m512 v0 = _mm512_loadu_ps(x + b * 32), v1 = _mm512_loadu_ps(x + b * 32 + 16);
        const float m = std::max(_mm512_reduce_max_ps(_mm512_abs_ps(v0)), _mm512_reduce_max_ps(_mm512_abs_ps(v1)));
        const float d = m / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        const __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, _mm512_set1_ps(id)));
        const __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, _mm512_set1_ps(id)));
        _mm_storeu_si128((__m128i *)(q + b * 32), _mm512_cvtsepi32_epi8(q0));
        _mm_storeu_si128((__m128i *)(q + b * 32 + 16), _mm512_cvtsepi32_epi8(q1));
        s[b] = d;
        sum[2 * b] = _mm512_reduce_add_epi32(q0);
        sum[2 * b + 1] = _mm512_reduce_add_epi32(q1);
    }
}

namespace {
VNNI inline __m512i two16(int lo, int hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short)lo)), _mm256_set1_epi16((short)hi), 1);
}
VNNI inline __m512i lane_groups(int k) {
    return _mm512_set_epi32(4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2,
                            4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k, 4 * k, 4 * k, 4 * k);
}

VNNI float dot1_q3k(const uint8_t *row, const int8_t *xq, const float *s16, const float *f16) {
    const __m512i m3 = _mm512_set1_epi8(3), one = _mm512_set1_epi8(1);
    const __m512i cq[2] = {two16(0, 2), two16(4, 6)}, ch[4] = {two16(0, 1), two16(2, 3), two16(4, 5), two16(6, 7)};
    const __m512i gi[4] = {lane_groups(0), lane_groups(1), lane_groups(2), lane_groups(3)};
    __m512 acc = _mm512_setzero_ps(), cor = _mm512_setzero_ps();
    for (int sb = 0; sb < D_ / 256; sb++) {
        const uint8_t *blk = row + sb * 110;
        uint32_t a[3];
        memcpy(a, blk + 96, 12);
        const uint32_t s0 = (a[0] & 0x0f0f0f0f) | ((a[2] & 0x03030303) << 4), s1 = (a[1] & 0x0f0f0f0f) | (((a[2] >> 2) & 0x03030303) << 4);
        const uint32_t s2 = ((a[0] >> 4) & 0x0f0f0f0f) | (((a[2] >> 4) & 0x03030303) << 4), s3 = ((a[1] >> 4) & 0x0f0f0f0f) | (((a[2] >> 6) & 0x03030303) << 4);
        const __m512 av = _mm512_mul_ps(_mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)(blk + 108))),
            _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_set_epi32((int)s3, (int)s2, (int)s1, (int)s0)), _mm512_set1_epi32(32))));
        const __m512 cs = _mm512_mul_ps(av, _mm512_loadu_ps(s16 + sb * 16));
        cor = _mm512_fmadd_ps(av, _mm512_loadu_ps(f16 + sb * 16), cor);
        const __m512i hm = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)blk));
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const __m512i q = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 32 + 32 * n)));
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int k = 2 * n + kk;
                const __m512i lo = _mm512_and_si512(_mm512_srlv_epi16(q, cq[kk]), m3);
                const __m512i hb = _mm512_and_si512(_mm512_srlv_epi16(hm, ch[k]), one);
                const __m512i w = _mm512_or_si512(lo, _mm512_slli_epi16(hb, 2));
                const __m512i sdot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, _mm512_loadu_si512(xq + sb * 256 + 64 * k));
                acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sdot), _mm512_permutexvar_ps(gi[k], cs), acc);
            }
        }
    }
    return _mm512_reduce_add_ps(acc) - 4.0f * _mm512_reduce_add_ps(cor);
}

VNNI float dot1_q2k(const uint8_t *row, const int8_t *xq, const float *s16, const float *f16) {
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i cq[2] = {two16(0, 2), two16(4, 6)};
    const __m512i gi[4] = {lane_groups(0), lane_groups(1), lane_groups(2), lane_groups(3)};
    __m512 acc = _mm512_setzero_ps(), cor = _mm512_setzero_ps();
    for (int sb = 0; sb < D_ / 256; sb++) {
        const uint8_t *blk = row + sb * 84;
        const __m512i sv = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)blk));
        const __m512 av = _mm512_mul_ps(_mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)(blk + 80))),
                                        _mm512_cvtepi32_ps(_mm512_and_si512(sv, _mm512_set1_epi32(15))));
        const __m512 bv = _mm512_mul_ps(_mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)(blk + 82))), _mm512_cvtepi32_ps(_mm512_srli_epi32(sv, 4)));
        const __m512 cs = _mm512_mul_ps(av, _mm512_loadu_ps(s16 + sb * 16));
        cor = _mm512_fmadd_ps(bv, _mm512_loadu_ps(f16 + sb * 16), cor);
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const __m512i q = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 16 + 32 * n)));
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int k = 2 * n + kk;
                const __m512i w = _mm512_and_si512(_mm512_srlv_epi16(q, cq[kk]), m3);
                const __m512i sdot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, _mm512_loadu_si512(xq + sb * 256 + 64 * k));
                acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sdot), _mm512_permutexvar_ps(gi[k], cs), acc);
            }
        }
    }
    return _mm512_reduce_add_ps(acc) - _mm512_reduce_add_ps(cor);
}

VNNI float dot1_iq4xs(const uint8_t *row, const int8_t *xq, const float *xs, const int32_t *xsum, VnniScratch &S) {
    decode_iq4xs(row, D_, S.w, S.ws);
    __m512 acc = _mm512_setzero_ps();
    float cor = 0;
    for (int b = 0; b < NB_GU; b += 2) {
        const __m512i sdot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_loadu_si512(S.w + 32 * b), _mm512_loadu_si512(xq + 32 * b));
        const float c0 = S.ws[b] * xs[b], c1 = S.ws[b + 1] * xs[b + 1];
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sdot), _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(c0), _mm512_set1_ps(c1)), acc);
        cor += 128.0f * (c0 * (float)(xsum[2 * b] + xsum[2 * b + 1]) + c1 * (float)(xsum[2 * b + 2] + xsum[2 * b + 3]));
    }
    return _mm512_reduce_add_ps(acc) - cor;
}
}

void vnni_act1(const float *xs, const int32_t *xsum, float *s16, float *f16) {
    for (int g = 0; g < D_ / 16; g++) { s16[g] = xs[g / 2]; f16[g] = xs[g / 2] * (float)xsum[g]; }
}
float vnni_dot_gu1(const uint8_t *row, int type, const int8_t *xq, const float *xs, const int32_t *xsum, const float *s16,
                   const float *f16, VnniScratch &S) {
    if (type == 11) return dot1_q3k(row, xq, s16, f16);
    if (type == 10) return dot1_q2k(row, xq, s16, f16);
    S.reserve();
    return dot1_iq4xs(row, xq, xs, xsum, S);
}
VNNI void vnni_quant_h(const float *h, int8_t *q, float *s, float *c) {
    for (int b = 0; b < NB_DN; b++) {
        const __m512 v0 = _mm512_loadu_ps(h + b * 32), v1 = _mm512_loadu_ps(h + b * 32 + 16);
        const float m = std::max(_mm512_reduce_max_ps(_mm512_abs_ps(v0)), _mm512_reduce_max_ps(_mm512_abs_ps(v1)));
        const float d = m / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        const __m512i q0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, _mm512_set1_ps(id)));
        const __m512i q1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, _mm512_set1_ps(id)));
        _mm_storeu_si128((__m128i *)(q + b * 32), _mm512_cvtsepi32_epi8(q0));
        _mm_storeu_si128((__m128i *)(q + b * 32 + 16), _mm512_cvtsepi32_epi8(q1));
        s[b] = d;
        c[b] = 128.0f * d * (float)(_mm512_reduce_add_epi32(q0) + _mm512_reduce_add_epi32(q1));
    }
}
VNNI float vnni_dot_dn1(const uint8_t *row, const int8_t *hq, const float *hs, const float *hc) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)KV128)), m4 = _mm256_set1_epi8(0x0f);
    __m512 acc = _mm512_setzero_ps();
    float cor = 0;
#pragma unroll 2
    for (int b = 0; b < NB_DN; b += 2) {
        const uint8_t *b0 = row + 18 * b, *b1 = b0 + 18;
        const __m256i q = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(b0 + 2))),
                                                  _mm_loadu_si128((const __m128i *)(b1 + 2)), 1);
        const __m256i lo = _mm256_shuffle_epi8(tbl, _mm256_and_si256(q, m4));
        const __m256i hi = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
        const __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_permute2x128_si256(lo, hi, 0x20)),
                                             _mm256_permute2x128_si256(lo, hi, 0x31), 1);
        const __m512i sdot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, _mm512_loadu_si512(hq + 32 * b));
        const float w0 = _cvtsh_ss(*(const uint16_t *)b0), w1 = _cvtsh_ss(*(const uint16_t *)b1);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sdot),
                              _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(w0 * hs[b]), _mm512_set1_ps(w1 * hs[b + 1])), acc);
        cor += w0 * hc[b] + w1 * hc[b + 1];
    }
    return _mm512_reduce_add_ps(acc) - cor;
}

void expert_vnni(const ColdLayer &c, int slot, int n, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
                 const float *w, uint16_t *const *out, VnniScratch &S) {
    S.reserve();
    for (int g0 = 0; g0 < n;) {
        const int ng = std::min(16, n - g0);
        if (ng > 8) group<16>(c, slot, ng, xq + g0, xs + g0, xsum + g0, w + g0, out + g0, S);
        else group<8>(c, slot, ng, xq + g0, xs + g0, xsum + g0, w + g0, out + g0, S);
        g0 += ng;
    }
}
