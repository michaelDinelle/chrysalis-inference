struct H {
    TGT static __m256i dot(__m256i acc, __m256i w, __m256i x) {
#if EXACT_DOT == 1
        return _mm256_dpbusd_epi32(acc, w, x);
#elif EXACT_DOT == 2
        return _mm256_dpbusd_avx_epi32(acc, w, x);
#else
        const __m256i lo = _mm256_maddubs_epi16(_mm256_and_si256(w, _mm256_set1_epi8(0x7f)), x);
        const __m256i hi = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 7), _mm256_set1_epi8(1)), x);
        return _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_madd_epi16(lo, _mm256_set1_epi16(1)), _mm256_madd_epi16(hi, _mm256_set1_epi16(128))));
#endif
    }
    TGT static __m256i dot_small(__m256i acc, __m256i w, __m256i x) {
#if EXACT_DOT
        return dot(acc, w, x);
#else
        return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1)));
#endif
    }
};

#if EXACT_WIDE
struct V {
    using i = __m512i;
    using f = __m512;
    TGT static i zero() { return _mm512_setzero_si512(); }
    TGT static i load(const void *p) { return _mm512_loadu_si512(p); }
    TGT static i load_a(const void *p) { return _mm512_load_si512(p); }
    TGT static void store(void *p, i a) { _mm512_storeu_si512(p, a); }
    TGT static i set1(int v) { return _mm512_set1_epi32(v); }
    TGT static i set8(char v) { return _mm512_set1_epi8(v); }
    TGT static i bcast256(const void *p) { return _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)p)); }
    TGT static i join(__m256i lo, __m256i hi) { return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1); }
    TGT static i u8x16(__m128i v) { return _mm512_cvtepu8_epi32(v); }
    TGT static i add(i a, i b) { return _mm512_add_epi32(a, b); }
    TGT static i sub(i a, i b) { return _mm512_sub_epi32(a, b); }
    TGT static i mullo(i a, i b) { return _mm512_mullo_epi32(a, b); }
    TGT static i and_(i a, i b) { return _mm512_and_si512(a, b); }
    TGT static i or_(i a, i b) { return _mm512_or_si512(a, b); }
    TGT static i sll32(i a, int n) { return _mm512_slli_epi32(a, n); }
    TGT static i srl32(i a, int n) { return _mm512_srli_epi32(a, n); }
    TGT static i sll16(i a, int n) { return _mm512_slli_epi16(a, n); }
    TGT static i srl16(i a, int lo, int hi) { return _mm512_srlv_epi16(a, join(_mm256_set1_epi16((short)lo), _mm256_set1_epi16((short)hi))); }
    TGT static i dot(i acc, i w, i x) {
#if EXACT_DOT
        return _mm512_dpbusd_epi32(acc, w, x);
#else
        const i lo = _mm512_maddubs_epi16(_mm512_and_si512(w, _mm512_set1_epi8(0x7f)), x);
        const i hi = _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(w, 7), _mm512_set1_epi8(1)), x);
        return _mm512_add_epi32(acc, _mm512_add_epi32(_mm512_madd_epi16(lo, _mm512_set1_epi16(1)), _mm512_madd_epi16(hi, _mm512_set1_epi16(128))));
#endif
    }
    TGT static i dot_small(i acc, i w, i x) {
#if EXACT_DOT
        return dot(acc, w, x);
#else
        return _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(w, x), _mm512_set1_epi16(1)));
#endif
    }
    TGT static i pairs(i a, i b) {
        const i ev = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0), od = _mm512_add_epi32(ev, _mm512_set1_epi32(1));
        return _mm512_add_epi32(_mm512_permutex2var_epi32(a, ev, b), _mm512_permutex2var_epi32(a, od, b));
    }
    TGT static int isum(i a) { return _mm512_reduce_add_epi32(a); }
    TGT static i load4z(const int32_t *p) { return _mm512_maskz_loadu_epi32(0xf, p); }
    TGT static void store_i8(int8_t *p, i a) { _mm_storeu_si128((__m128i *)p, _mm512_cvtsepi32_epi8(a)); }
    TGT static f fzero() { return _mm512_setzero_ps(); }
    TGT static f fload(const float *p) { return _mm512_loadu_ps(p); }
    TGT static f fload_a(const float *p) { return _mm512_load_ps(p); }
    TGT static f fload4z(const float *p) { return _mm512_maskz_loadu_ps(0xf, p); }
    TGT static void fstore(float *p, f a) { _mm512_storeu_ps(p, a); }
    TGT static void fstore_a(float *p, f a) { _mm512_store_ps(p, a); }
    TGT static f fset1(float v) { return _mm512_set1_ps(v); }
    TGT static f halves(float lo, float hi) { return _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(lo), _mm512_set1_ps(hi)); }
    TGT static f pick2(__m256 c, int k) { return _mm512_permutexvar_ps(join(_mm256_set1_epi32(2 * k), _mm256_set1_epi32(2 * k + 1)), _mm512_castps256_ps512(c)); }
    TGT static f quads(f a, int k) {
        return _mm512_permutexvar_ps(_mm512_set_epi32(4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 3, 4 * k + 2, 4 * k + 2, 4 * k + 2, 4 * k + 2,
                                                      4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k + 1, 4 * k, 4 * k, 4 * k, 4 * k), a);
    }
    TGT static f cvt(i a) { return _mm512_cvtepi32_ps(a); }
    TGT static i cvt_round(f a) { return _mm512_cvtps_epi32(a); }
    TGT static f fadd(f a, f b) { return _mm512_add_ps(a, b); }
    TGT static f fmul(f a, f b) { return _mm512_mul_ps(a, b); }
    TGT static f fdiv(f a, f b) { return _mm512_div_ps(a, b); }
    TGT static f fma(f a, f b, f c) { return _mm512_fmadd_ps(a, b, c); }
    TGT static f fnma(f a, f b, f c) { return _mm512_fnmadd_ps(a, b, c); }
    TGT static f fms(f a, f b, f c) { return _mm512_fmsub_ps(a, b, c); }
    TGT static f fmul_rn(f a, f b) { return _mm512_mul_round_ps(a, b, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
    TGT static f fmul4z(f a, f b) { return _mm512_maskz_mul_ps(0xf, a, b); }
    TGT static f absf(f a) { return _mm512_abs_ps(a); }
    TGT static float hmax(f a) { return _mm512_reduce_max_ps(a); }
    TGT static float fsum(f a) { return _mm512_reduce_add_ps(a); }
    TGT static float lanes_sum(f a, f b) {
        f v = _mm512_add_ps(a, b);
        v = _mm512_add_ps(v, _mm512_shuffle_f32x4(v, v, 0x4E));
        v = _mm512_add_ps(v, _mm512_shuffle_f32x4(v, v, 0xB1));
        v = _mm512_add_ps(v, _mm512_permute_ps(v, 0x4E));
        v = _mm512_add_ps(v, _mm512_permute_ps(v, 0xB1));
        return _mm512_cvtss_f32(v);
    }
    TGT static void store_ph(uint16_t *p, f a) { _mm256_storeu_si256((__m256i *)p, _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT)); }
};
#else
struct V {
    struct i { __m256i a, b; };
    struct f { __m256 a, b; };
    TGT static i zero() { return {_mm256_setzero_si256(), _mm256_setzero_si256()}; }
    TGT static i load(const void *p) { return {_mm256_loadu_si256((const __m256i *)p), _mm256_loadu_si256((const __m256i *)p + 1)}; }
    TGT static i load_a(const void *p) { return {_mm256_load_si256((const __m256i *)p), _mm256_load_si256((const __m256i *)p + 1)}; }
    TGT static void store(void *p, i v) { _mm256_storeu_si256((__m256i *)p, v.a); _mm256_storeu_si256((__m256i *)p + 1, v.b); }
    TGT static i set1(int v) { const __m256i s = _mm256_set1_epi32(v); return {s, s}; }
    TGT static i set8(char v) { const __m256i s = _mm256_set1_epi8(v); return {s, s}; }
    TGT static i bcast256(const void *p) { const __m256i v = _mm256_loadu_si256((const __m256i *)p); return {v, v}; }
    TGT static i join(__m256i lo, __m256i hi) { return {lo, hi}; }
    TGT static i u8x16(__m128i v) { return {_mm256_cvtepu8_epi32(v), _mm256_cvtepu8_epi32(_mm_srli_si128(v, 8))}; }
    TGT static i add(i x, i y) { return {_mm256_add_epi32(x.a, y.a), _mm256_add_epi32(x.b, y.b)}; }
    TGT static i sub(i x, i y) { return {_mm256_sub_epi32(x.a, y.a), _mm256_sub_epi32(x.b, y.b)}; }
    TGT static i mullo(i x, i y) { return {_mm256_mullo_epi32(x.a, y.a), _mm256_mullo_epi32(x.b, y.b)}; }
    TGT static i and_(i x, i y) { return {_mm256_and_si256(x.a, y.a), _mm256_and_si256(x.b, y.b)}; }
    TGT static i or_(i x, i y) { return {_mm256_or_si256(x.a, y.a), _mm256_or_si256(x.b, y.b)}; }
    TGT static i sll32(i x, int n) { return {_mm256_slli_epi32(x.a, n), _mm256_slli_epi32(x.b, n)}; }
    TGT static i srl32(i x, int n) { return {_mm256_srli_epi32(x.a, n), _mm256_srli_epi32(x.b, n)}; }
    TGT static i sll16(i x, int n) { return {_mm256_slli_epi16(x.a, n), _mm256_slli_epi16(x.b, n)}; }
    TGT static i srl16(i x, int lo, int hi) { return {_mm256_srli_epi16(x.a, lo), _mm256_srli_epi16(x.b, hi)}; }
    TGT static i dot(i acc, i w, i x) { return {H::dot(acc.a, w.a, x.a), H::dot(acc.b, w.b, x.b)}; }
    TGT static i dot_small(i acc, i w, i x) { return {H::dot_small(acc.a, w.a, x.a), H::dot_small(acc.b, w.b, x.b)}; }
    TGT static __m256i pair8(__m256i x, __m256i y) { return _mm256_permute4x64_epi64(_mm256_hadd_epi32(x, y), 0xD8); }
    TGT static i pairs(i x, i y) { return {pair8(x.a, x.b), pair8(y.a, y.b)}; }
    TGT static int isum(i x) {
        const __m256i s = _mm256_add_epi32(x.a, x.b);
        __m128i t = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
        return _mm_cvtsi128_si32(t);
    }
    TGT static i load4z(const int32_t *p) { return {_mm256_set_m128i(_mm_setzero_si128(), _mm_loadu_si128((const __m128i *)p)), _mm256_setzero_si256()}; }
    TGT static void store_i8(int8_t *p, i x) {
        const __m256i w = _mm256_permute4x64_epi64(_mm256_packs_epi32(x.a, x.b), 0xD8);
        _mm_storeu_si128((__m128i *)p, _mm_packs_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1)));
    }
    TGT static f fzero() { return {_mm256_setzero_ps(), _mm256_setzero_ps()}; }
    TGT static f fload(const float *p) { return {_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8)}; }
    TGT static f fload_a(const float *p) { return {_mm256_load_ps(p), _mm256_load_ps(p + 8)}; }
    TGT static f fload4z(const float *p) { return {_mm256_set_m128(_mm_setzero_ps(), _mm_loadu_ps(p)), _mm256_setzero_ps()}; }
    TGT static void fstore(float *p, f x) { _mm256_storeu_ps(p, x.a); _mm256_storeu_ps(p + 8, x.b); }
    TGT static void fstore_a(float *p, f x) { _mm256_store_ps(p, x.a); _mm256_store_ps(p + 8, x.b); }
    TGT static f fset1(float v) { const __m256 s = _mm256_set1_ps(v); return {s, s}; }
    TGT static f halves(float lo, float hi) { return {_mm256_set1_ps(lo), _mm256_set1_ps(hi)}; }
    TGT static f pick2(__m256 c, int k) { return {_mm256_permutevar8x32_ps(c, _mm256_set1_epi32(2 * k)), _mm256_permutevar8x32_ps(c, _mm256_set1_epi32(2 * k + 1))}; }
    TGT static f quads(f x, int k) {
        const __m256 s = k < 2 ? x.a : x.b;
        const int o = 4 * (k & 1);
        return {_mm256_permutevar8x32_ps(s, _mm256_setr_epi32(o, o, o, o, o + 1, o + 1, o + 1, o + 1)),
                _mm256_permutevar8x32_ps(s, _mm256_setr_epi32(o + 2, o + 2, o + 2, o + 2, o + 3, o + 3, o + 3, o + 3))};
    }
    TGT static f cvt(i x) { return {_mm256_cvtepi32_ps(x.a), _mm256_cvtepi32_ps(x.b)}; }
    TGT static i cvt_round(f x) { return {_mm256_cvtps_epi32(x.a), _mm256_cvtps_epi32(x.b)}; }
    TGT static f fadd(f x, f y) { return {_mm256_add_ps(x.a, y.a), _mm256_add_ps(x.b, y.b)}; }
    TGT static f fmul(f x, f y) { return {_mm256_mul_ps(x.a, y.a), _mm256_mul_ps(x.b, y.b)}; }
    TGT static f fdiv(f x, f y) { return {_mm256_div_ps(x.a, y.a), _mm256_div_ps(x.b, y.b)}; }
    TGT static f fma(f x, f y, f z) { return {_mm256_fmadd_ps(x.a, y.a, z.a), _mm256_fmadd_ps(x.b, y.b, z.b)}; }
    TGT static f fnma(f x, f y, f z) { return {_mm256_fnmadd_ps(x.a, y.a, z.a), _mm256_fnmadd_ps(x.b, y.b, z.b)}; }
    TGT static f fms(f x, f y, f z) { return {_mm256_fmsub_ps(x.a, y.a, z.a), _mm256_fmsub_ps(x.b, y.b, z.b)}; }
    TGT static __m256 mul_rn(__m256 x, __m256 y) {
        __m256 r;
        __asm__("vmulps %2, %1, %0" : "=x"(r) : "x"(x), "x"(y));
        return r;
    }
    TGT static f fmul_rn(f x, f y) { return {mul_rn(x.a, y.a), mul_rn(x.b, y.b)}; }
    TGT static f fmul4z(f x, f y) { return {_mm256_blend_ps(_mm256_setzero_ps(), _mm256_mul_ps(x.a, y.a), 0x0f), _mm256_setzero_ps()}; }
    TGT static f absf(f x) {
        const __m256 m = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        return {_mm256_and_ps(x.a, m), _mm256_and_ps(x.b, m)};
    }
    TGT static float hmax(f x) {
        const __m256 s = _mm256_max_ps(x.a, x.b);
        __m128 t = _mm_max_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        t = _mm_max_ps(t, _mm_movehl_ps(t, t));
        return _mm_cvtss_f32(_mm_max_ss(t, _mm_movehdup_ps(t)));
    }
    TGT static float fsum(f x) {
        const __m256 s = _mm256_add_ps(x.a, x.b);
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
    }
    TGT static float lanes_sum(f x, f y) {
        __m256 s = _mm256_add_ps(_mm256_add_ps(x.a, y.a), _mm256_add_ps(x.b, y.b));
        s = _mm256_add_ps(s, _mm256_permute2f128_ps(s, s, 1));
        s = _mm256_add_ps(s, _mm256_permute_ps(s, 0x4E));
        s = _mm256_add_ps(s, _mm256_permute_ps(s, 0xB1));
        return _mm256_cvtss_f32(s);
    }
    TGT static void store_ph(uint16_t *p, f x) {
        _mm_storeu_si128((__m128i *)p, _mm256_cvtps_ph(x.a, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128((__m128i *)(p + 8), _mm256_cvtps_ph(x.b, _MM_FROUND_TO_NEAREST_INT));
    }
};
#endif

TGT inline uint64_t iq4xs_scales(const uint8_t *blk) {
#if EXACT_WIDE || EXACT_DOT
    const uint64_t lo4 = _pdep_u64(*(const uint32_t *)(blk + 4), 0x0F0F0F0F0F0F0F0Full);
    const uint64_t hi2 = _pdep_u64(*(const uint16_t *)(blk + 2), 0x0303030303030303ull);
#else
    uint64_t lo4 = *(const uint32_t *)(blk + 4), hi2 = *(const uint16_t *)(blk + 2);
    lo4 = (lo4 | lo4 << 16) & 0x0000FFFF0000FFFFull;
    lo4 = (lo4 | lo4 << 8) & 0x00FF00FF00FF00FFull;
    lo4 = (lo4 | lo4 << 4) & 0x0F0F0F0F0F0F0F0Full;
    hi2 = (hi2 | hi2 << 24) & 0x000000FF000000FFull;
    hi2 = (hi2 | hi2 << 12) & 0x000F000F000F000Full;
    hi2 = (hi2 | hi2 << 6) & 0x0303030303030303ull;
#endif
    return lo4 | hi2 << 4;
}

TGT inline __m256 iq4xs_ws8(const uint8_t *blk) {
    const __m256i ls = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)iq4xs_scales(blk)));
    return _mm256_mul_ps(_mm256_set1_ps(_cvtsh_ss(*(const uint16_t *)blk)), _mm256_cvtepi32_ps(_mm256_sub_epi32(ls, _mm256_set1_epi32(32))));
}

TGT void decode_iq4xs(const uint8_t *row, int K, uint8_t *w, float *ws) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)KV128)), m4 = _mm256_set1_epi8(0x0f);
    for (int sb = 0; sb < K / 256; sb++) {
        const uint8_t *blk = row + sb * 136;
        _mm256_storeu_ps(ws + sb * 8, iq4xs_ws8(blk));
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

TGT void decode_iq4nl(const uint8_t *row, int K, uint8_t *w, float *ws) {
    const __m128i tbl = _mm_load_si128((const __m128i *)KV128), m4 = _mm_set1_epi8(0x0f);
    for (int b = 0; b < K / 32; b++) {
        const uint8_t *blk = row + b * 18;
        const __m128i q = _mm_loadu_si128((const __m128i *)(blk + 2));
        _mm_storeu_si128((__m128i *)(w + b * 32), _mm_shuffle_epi8(tbl, _mm_and_si128(q, m4)));
        _mm_storeu_si128((__m128i *)(w + b * 32 + 16), _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(q, 4), m4)));
        ws[b] = _cvtsh_ss(*(const uint16_t *)blk);
    }
}

TGT inline V::f q3k_scales(const uint8_t *blk) {
    uint32_t a[3];
    memcpy(a, blk + 96, 12);
    const uint32_t s0 = (a[0] & 0x0f0f0f0f) | ((a[2] & 0x03030303) << 4), s1 = (a[1] & 0x0f0f0f0f) | (((a[2] >> 2) & 0x03030303) << 4);
    const uint32_t s2 = ((a[0] >> 4) & 0x0f0f0f0f) | (((a[2] >> 4) & 0x03030303) << 4), s3 = ((a[1] >> 4) & 0x0f0f0f0f) | (((a[2] >> 6) & 0x03030303) << 4);
    return V::fmul(V::fset1(_cvtsh_ss(*(const uint16_t *)(blk + 108))),
                   V::cvt(V::sub(V::u8x16(_mm_set_epi32((int)s3, (int)s2, (int)s1, (int)s0)), V::set1(32))));
}

TGT void decode_q3k(const uint8_t *row, int K, uint8_t *w, float *wa, float *wb) {
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    for (int sb = 0; sb < K / 256; sb++) {
        const uint8_t *blk = row + sb * 110;
        const V::f av = q3k_scales(blk);
        V::fstore(wa + sb * 16, av);
        V::fstore(wb + sb * 16, V::fmul(av, V::fset1(4.0f)));
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

TGT void decode_q2k(const uint8_t *row, int K, uint8_t *w, float *wa, float *wb) {
    const __m256i m3 = _mm256_set1_epi8(3);
    for (int sb = 0; sb < K / 256; sb++) {
        const uint8_t *blk = row + sb * 84;
        const V::i s = V::u8x16(_mm_loadu_si128((const __m128i *)blk));
        V::fstore(wa + sb * 16, V::fmul(V::fset1(_cvtsh_ss(*(const uint16_t *)(blk + 80))), V::cvt(V::and_(s, V::set1(15)))));
        V::fstore(wb + sb * 16, V::fmul(V::fset1(_cvtsh_ss(*(const uint16_t *)(blk + 82))), V::cvt(V::srl32(s, 4))));
        for (int n = 0; n < 2; n++) {
            const __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 16 + 32 * n));
            for (int j = 0; j < 4; j++)
                _mm256_storeu_si256((__m256i *)(w + sb * 256 + 128 * n + 32 * j), _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3));
        }
    }
}

template <int L> struct Lanes;
template <> struct Lanes<16> {
    using I = V::i; using F = V::f;
    TGT static I zero() { return V::zero(); }
    TGT static I dp(I s, int32_t w, const uint32_t *x) { return V::dot(s, V::set1(w), V::load_a(x)); }
    TGT static I dp_small(I s, int32_t w, const uint32_t *x) { return V::dot_small(s, V::set1(w), V::load_a(x)); }
    TGT static F block(I s0, I s1, const int32_t *xc, float ws, const float *xs, F a) {
        const F v = V::cvt(V::sub(V::add(s0, s1), V::load_a(xc)));
        return V::fma(v, V::fmul(V::fset1(ws), V::fload_a(xs)), a);
    }
    TGT static F fzero() { return V::fzero(); }
    TGT static void store(float *o, F a, F b) { V::fstore_a(o, V::fadd(a, b)); }
    TGT static F block16(I s0, I s1, float a0, float a1, const float *xs, F a) {
        const F v = V::fma(V::cvt(s1), V::fset1(a1), V::fmul(V::cvt(s0), V::fset1(a0)));
        return V::fma(v, V::fload_a(xs), a);
    }
    TGT static F offs16(float b0, const float *xf0, float b1, const float *xf1, F a) {
        return V::fnma(V::fset1(b1), V::fload_a(xf1), V::fnma(V::fset1(b0), V::fload_a(xf0), a));
    }
};
template <> struct Lanes<8> {
    using I = __m256i; using F = __m256;
    TGT static I zero() { return _mm256_setzero_si256(); }
    TGT static I dp(I s, int32_t w, const uint32_t *x) { return H::dot(s, _mm256_set1_epi32(w), _mm256_load_si256((const __m256i *)x)); }
    TGT static I dp_small(I s, int32_t w, const uint32_t *x) { return H::dot_small(s, _mm256_set1_epi32(w), _mm256_load_si256((const __m256i *)x)); }
    TGT static F block(I s0, I s1, const int32_t *xc, float ws, const float *xs, F a) {
        const F v = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_add_epi32(s0, s1), _mm256_load_si256((const __m256i *)xc)));
        return _mm256_fmadd_ps(v, _mm256_mul_ps(_mm256_set1_ps(ws), _mm256_load_ps(xs)), a);
    }
    TGT static F fzero() { return _mm256_setzero_ps(); }
    TGT static void store(float *o, F a, F b) { _mm256_store_ps(o, _mm256_add_ps(a, b)); }
    TGT static F block16(I s0, I s1, float a0, float a1, const float *xs, F a) {
        const F v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_set1_ps(a1), _mm256_mul_ps(_mm256_cvtepi32_ps(s0), _mm256_set1_ps(a0)));
        return _mm256_fmadd_ps(v, _mm256_load_ps(xs), a);
    }
    TGT static F offs16(float b0, const float *xf0, float b1, const float *xf1, F a) {
        return _mm256_fnmadd_ps(_mm256_set1_ps(b1), _mm256_load_ps(xf1), _mm256_fnmadd_ps(_mm256_set1_ps(b0), _mm256_load_ps(xf0), a));
    }
};

template <int L>
TGT inline void row_dot(const uint8_t *w, const float *ws, int nb, const uint32_t *xi, const float *xs, const int32_t *xc, float *out) {
    using P = Lanes<L>;
    typename P::F a0 = P::fzero(), a1 = P::fzero();
    const int32_t *wd = (const int32_t *)w;
    for (int b = 0; b < nb; b += 2) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int bb = b + u;
            const uint32_t *x = xi + bb * 8 * L;
            const int32_t *wb = wd + bb * 8;
            typename P::I s0 = P::dp(P::zero(), wb[0], x), s1 = P::dp(P::zero(), wb[1], x + L);
            s0 = P::dp(s0, wb[2], x + 2 * L); s1 = P::dp(s1, wb[3], x + 3 * L);
            s0 = P::dp(s0, wb[4], x + 4 * L); s1 = P::dp(s1, wb[5], x + 5 * L);
            s0 = P::dp(s0, wb[6], x + 6 * L); s1 = P::dp(s1, wb[7], x + 7 * L);
            if (u == 0) a0 = P::block(s0, s1, xc + bb * L, ws[bb], xs + bb * L, a0);
            else a1 = P::block(s0, s1, xc + bb * L, ws[bb], xs + bb * L, a1);
        }
    }
    P::store(out, a0, a1);
}

template <int L>
TGT inline void row_dot16(const uint8_t *w, const float *wa, const float *wb, int nb, const uint32_t *xi, const float *xs, const float *xf,
                          float *out) {
    using P = Lanes<L>;
    typename P::F a0 = P::fzero(), a1 = P::fzero();
    const int32_t *wd = (const int32_t *)w;
    for (int b = 0; b < nb; b++) {
        const uint32_t *x = xi + b * 8 * L;
        const int32_t *wq = wd + b * 8;
        typename P::I s0 = P::dp_small(P::zero(), wq[0], x), s1 = P::dp_small(P::zero(), wq[4], x + 4 * L);
        s0 = P::dp_small(s0, wq[1], x + L); s1 = P::dp_small(s1, wq[5], x + 5 * L);
        s0 = P::dp_small(s0, wq[2], x + 2 * L); s1 = P::dp_small(s1, wq[6], x + 6 * L);
        s0 = P::dp_small(s0, wq[3], x + 3 * L); s1 = P::dp_small(s1, wq[7], x + 7 * L);
        a0 = P::block16(s0, s1, wa[2 * b], wa[2 * b + 1], xs + b * L, a0);
        a1 = P::offs16(wb[2 * b], xf + 2 * b * L, wb[2 * b + 1], xf + (2 * b + 1) * L, a1);
    }
    P::store(out, a0, a1);
}

template <int L>
TGT void group(const ColdLayer &c, int slot, int ng, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
               const float *w, uint16_t *const *out, ExactScratch &S) {
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
            V::store_ph(o + r, V::fload_a(col));
        }
    }
}

void expert(const ColdLayer &c, int slot, int n, const int8_t *const *xq, const float *const *xs, const int32_t *const *xsum,
                const float *w, uint16_t *const *out, ExactScratch &S) {
    S.reserve();
    for (int g0 = 0; g0 < n;) {
        const int ng = std::min(16, n - g0);
        if (ng > 8) group<16>(c, slot, ng, xq + g0, xs + g0, xsum + g0, w + g0, out + g0, S);
        else group<8>(c, slot, ng, xq + g0, xs + g0, xsum + g0, w + g0, out + g0, S);
        g0 += ng;
    }
}

TGT void quant_x(const float *x, int8_t *q, float *s, int32_t *sum) {
    for (int b = 0; b < NB_GU; b++) {
        const V::f v0 = V::fload(x + b * 32), v1 = V::fload(x + b * 32 + 16);
        const float m = std::max(V::hmax(V::absf(v0)), V::hmax(V::absf(v1)));
        const float d = m / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        const V::i q0 = V::cvt_round(V::fmul(v0, V::fset1(id)));
        const V::i q1 = V::cvt_round(V::fmul(v1, V::fset1(id)));
        V::store_i8(q + b * 32, q0);
        V::store_i8(q + b * 32 + 16, q1);
        s[b] = d;
        sum[2 * b] = V::isum(q0);
        sum[2 * b + 1] = V::isum(q1);
    }
}

TGT float dot1_q3k(const uint8_t *row, const int8_t *xq, const float *s16, const float *f16) {
    const V::i m3 = V::set8(3), one = V::set8(1);
    V::f acc = V::fzero(), cor = V::fzero();
    for (int sb = 0; sb < D_ / 256; sb++) {
        const uint8_t *blk = row + sb * 110;
        const V::f av = q3k_scales(blk);
        const V::f cs = V::fmul(av, V::fload(s16 + sb * 16));
        cor = V::fma(av, V::fload(f16 + sb * 16), cor);
        const V::i hm = V::bcast256(blk);
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const V::i q = V::bcast256(blk + 32 + 32 * n);
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int k = 2 * n + kk;
                const V::i lo = V::and_(V::srl16(q, 4 * kk, 4 * kk + 2), m3);
                const V::i hb = V::and_(V::srl16(hm, 2 * k, 2 * k + 1), one);
                const V::i w = V::or_(lo, V::sll16(hb, 2));
                const V::i sdot = V::dot_small(V::zero(), w, V::load(xq + sb * 256 + 64 * k));
                acc = V::fma(V::cvt(sdot), V::quads(cs, k), acc);
            }
        }
    }
    return V::fsum(acc) - 4.0f * V::fsum(cor);
}

TGT float dot1_q2k(const uint8_t *row, const int8_t *xq, const float *s16, const float *f16) {
    const V::i m3 = V::set8(3);
    V::f acc = V::fzero(), cor = V::fzero();
    for (int sb = 0; sb < D_ / 256; sb++) {
        const uint8_t *blk = row + sb * 84;
        const V::i sv = V::u8x16(_mm_loadu_si128((const __m128i *)blk));
        const V::f av = V::fmul(V::fset1(_cvtsh_ss(*(const uint16_t *)(blk + 80))), V::cvt(V::and_(sv, V::set1(15))));
        const V::f bv = V::fmul(V::fset1(_cvtsh_ss(*(const uint16_t *)(blk + 82))), V::cvt(V::srl32(sv, 4)));
        const V::f cs = V::fmul(av, V::fload(s16 + sb * 16));
        cor = V::fma(bv, V::fload(f16 + sb * 16), cor);
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const V::i q = V::bcast256(blk + 16 + 32 * n);
#pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                const int k = 2 * n + kk;
                const V::i w = V::and_(V::srl16(q, 4 * kk, 4 * kk + 2), m3);
                const V::i sdot = V::dot_small(V::zero(), w, V::load(xq + sb * 256 + 64 * k));
                acc = V::fma(V::cvt(sdot), V::quads(cs, k), acc);
            }
        }
    }
    return V::fsum(acc) - V::fsum(cor);
}

TGT float dot1_iq4xs(const uint8_t *row, const int8_t *xq, const float *xs, const int32_t *xsum, ExactScratch &S) {
    decode_iq4xs(row, D_, S.w, S.ws);
    V::f acc = V::fzero();
    float cor = 0;
    for (int b = 0; b < NB_GU; b += 2) {
        const V::i sdot = V::dot(V::zero(), V::load(S.w + 32 * b), V::load(xq + 32 * b));
        const float c0 = S.ws[b] * xs[b], c1 = S.ws[b + 1] * xs[b + 1];
        acc = V::fma(V::cvt(sdot), V::halves(c0, c1), acc);
        cor += 128.0f * (c0 * (float)(xsum[2 * b] + xsum[2 * b + 1]) + c1 * (float)(xsum[2 * b + 2] + xsum[2 * b + 3]));
    }
    return V::fsum(acc) - cor;
}

TGT inline V::i iq4xs_w64(const uint8_t *qp, __m256i tbl, __m256i m4) {
    const __m256i q = _mm256_loadu_si256((const __m256i *)qp);
    const __m256i lo = _mm256_shuffle_epi8(tbl, _mm256_and_si256(q, m4));
    const __m256i hi = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
    return V::join(_mm256_permute2x128_si256(lo, hi, 0x20), _mm256_permute2x128_si256(lo, hi, 0x31));
}

template <int NR>
TGT void dotn_iq4xs(const uint8_t *const *row, const int8_t *xq, const float *xs, const float *xf, float *out) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)KV128)), m4 = _mm256_set1_epi8(0x0f);
    V::f acc[NR];
    float cor[NR];
    for (int i = 0; i < NR; i++) { acc[i] = V::fzero(); cor[i] = 0; }
    for (int sb = 0; sb < D_ / 256; sb++) {
        const __m256 x8 = _mm256_loadu_ps(xs + sb * 8), f8 = _mm256_loadu_ps(xf + sb * 8);
        __m256 cz[NR];
        alignas(16) float t[NR][4];
        const __m256i ev = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6), od = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
        const __m128 fe = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(f8, ev)), fo = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(f8, od));
        for (int i = 0; i < NR; i++) {
            const __m256 c8 = _mm256_mul_ps(iq4xs_ws8(row[i] + sb * 136), x8);
            cz[i] = c8;
            const __m128 ce = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(c8, ev)), co = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(c8, od));
            _mm_store_ps(t[i], _mm_fmadd_ps(ce, fe, _mm_mul_ps(co, fo)));
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const V::i xv = V::load(xq + sb * 256 + 64 * k);
            for (int i = 0; i < NR; i++) {
                const V::i sdot = V::dot(V::zero(), iq4xs_w64(row[i] + sb * 136 + 8 + 32 * k, tbl, m4), xv);
                acc[i] = V::fma(V::cvt(sdot), V::pick2(cz[i], k), acc[i]);
            }
        }
        for (int i = 0; i < NR; i++) {
            for (int k = 0; k < 4; k++) cor[i] = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(128.0f), _mm_set_ss(t[i][k]), _mm_set_ss(cor[i])));
        }
    }
    for (int i = 0; i < NR; i++) out[i] = V::fsum(acc[i]) - cor[i];
}

float dot_gu1(const uint8_t *row, int type, const int8_t *xq, const float *xs, const int32_t *xsum, const float *s16, const float *f16,
                  ExactScratch &S) {
    if (type == 11) return dot1_q3k(row, xq, s16, f16);
    if (type == 10) return dot1_q2k(row, xq, s16, f16);
    S.reserve();
    return dot1_iq4xs(row, xq, xs, xsum, S);
}

void dot_gu2(const uint8_t *r0, const uint8_t *r1, const int8_t *xq, const float *xs, const float *xf, float *out) {
    const uint8_t *rows[2] = {r0, r1};
    dotn_iq4xs<2>(rows, xq, xs, xf, out);
}

TGT void quant_h(const float *h, int8_t *q, float *s, float *c) {
    for (int b = 0; b < NB_DN; b++) {
        const V::f v0 = V::fload(h + b * 32), v1 = V::fload(h + b * 32 + 16);
        const float m = std::max(V::hmax(V::absf(v0)), V::hmax(V::absf(v1)));
        const float d = m / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        const V::i q0 = V::cvt_round(V::fmul(v0, V::fset1(id)));
        const V::i q1 = V::cvt_round(V::fmul(v1, V::fset1(id)));
        V::store_i8(q + b * 32, q0);
        V::store_i8(q + b * 32 + 16, q1);
        s[b] = d;
        c[b] = 128.0f * d * (float)(V::isum(q0) + V::isum(q1));
    }
}

TGT float dot_dn1(const uint8_t *row, const int8_t *hq, const float *hs, const float *hc) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)KV128)), m4 = _mm256_set1_epi8(0x0f);
    V::f acc = V::fzero();
    float cor = 0;
#pragma unroll 2
    for (int b = 0; b < NB_DN; b += 2) {
        const uint8_t *b0 = row + 18 * b, *b1 = b0 + 18;
        const __m256i q = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(b0 + 2))),
                                                  _mm_loadu_si128((const __m128i *)(b1 + 2)), 1);
        const __m256i lo = _mm256_shuffle_epi8(tbl, _mm256_and_si256(q, m4));
        const __m256i hi = _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
        const V::i w = V::join(_mm256_permute2x128_si256(lo, hi, 0x20), _mm256_permute2x128_si256(lo, hi, 0x31));
        const V::i sdot = V::dot(V::zero(), w, V::load(hq + 32 * b));
        const float w0 = _cvtsh_ss(*(const uint16_t *)b0), w1 = _cvtsh_ss(*(const uint16_t *)b1);
        acc = V::fma(V::cvt(sdot), V::halves(w0 * hs[b], w1 * hs[b + 1]), acc);
        cor += w0 * hc[b] + w1 * hc[b + 1];
    }
    return V::fsum(acc) - cor;
}

template <bool BIG>
TGT inline V::i dot64(const uint8_t *w, const int8_t *x) {
    if (BIG) return V::dot(V::zero(), V::load(w), V::load(x));
    return V::dot_small(V::zero(), V::load(w), V::load(x));
}
template <bool BIG>
TGT inline V::i groups16(const uint8_t *w, const int8_t *x) {
    return V::pairs(V::pairs(dot64<BIG>(w, x), dot64<BIG>(w + 64, x + 64)), V::pairs(dot64<BIG>(w + 128, x + 128), dot64<BIG>(w + 192, x + 192)));
}
TGT inline float chunk_sum(const V::f *s, const V::f *i) {
    const V::f z = V::fzero();
    const V::f a = V::fma(s[4], i[4], V::fma(s[2], i[2], V::fma(s[0], i[0], z)));
    const V::f b = V::fma(s[3], i[3], V::fma(s[1], i[1], z));
    return V::lanes_sum(a, b);
}
TGT void decode_q3k_raw(const uint8_t *blk, uint8_t *w, int32_t *sc) {
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    const __m256i hm = _mm256_loadu_si256((const __m256i *)blk);
    for (int n = 0; n < 2; n++) {
        const __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 + 32 * n));
        for (int j = 0; j < 4; j++) {
            const __m256i lo = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * j)), m3);
            const __m256i hb = _mm256_and_si256(_mm256_srl_epi16(hm, _mm_cvtsi32_si128(4 * n + j)), one);
            _mm256_storeu_si256((__m256i *)(w + 128 * n + 32 * j), _mm256_or_si256(lo, _mm256_slli_epi16(hb, 2)));
        }
    }
    uint32_t a[3];
    memcpy(a, blk + 96, 12);
    const uint32_t s4[4] = {(a[0] & 0x0f0f0f0f) | ((a[2] & 0x03030303) << 4), (a[1] & 0x0f0f0f0f) | (((a[2] >> 2) & 0x03030303) << 4),
                            ((a[0] >> 4) & 0x0f0f0f0f) | (((a[2] >> 4) & 0x03030303) << 4), ((a[1] >> 4) & 0x0f0f0f0f) | (((a[2] >> 6) & 0x03030303) << 4)};
    V::store(sc, V::sub(V::u8x16(_mm_loadu_si128((const __m128i *)s4)), V::set1(32)));
}
TGT void decode_q2k_raw(const uint8_t *blk, uint8_t *w) {
    const __m256i m3 = _mm256_set1_epi8(3);
    for (int n = 0; n < 2; n++) {
        const __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 16 + 32 * n));
        for (int j = 0; j < 4; j++)
            _mm256_storeu_si256((__m256i *)(w + 128 * n + 32 * j), _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * j)), m3));
    }
}

TGT void emu_quant_x(const float *x, EmuAct &a) {
    for (int b = 0; b < NB_GU; b++) {
        const V::f v0 = V::fload(x + b * 32), v1 = V::fload(x + b * 32 + 16);
        const float d = std::max(V::hmax(V::absf(v0)), V::hmax(V::absf(v1))) / 127.0f;
        V::i q0 = V::zero(), q1 = q0;
        if (d > 0) {
            q0 = V::cvt_round(V::fdiv(v0, V::fset1(d)));
            q1 = V::cvt_round(V::fdiv(v1, V::fset1(d)));
        }
        V::store_i8(a.q + b * 32, q0);
        V::store_i8(a.q + b * 32 + 16, q1);
        a.d[b] = d;
        a.s16[2 * b] = V::isum(q0);
        a.s16[2 * b + 1] = V::isum(q1);
        a.c32[b] = 128 * (a.s16[2 * b] + a.s16[2 * b + 1]);
    }
}

TGT float emu_gu_row(const uint8_t *row, int type, const EmuAct &a, ExactScratch &S) {
    S.reserve();
    V::i g[10];
    V::f sv[5], iv[5];
    if (type == 23) {
        decode_iq4xs(row, D_, S.w, S.ws);
        for (int k = 0; k < 10; k++) g[k] = groups16<true>(S.w + 256 * k, a.q + 256 * k);
        for (int m = 0; m < 5; m++) {
            iv[m] = V::cvt(V::sub(V::pairs(g[2 * m], g[2 * m + 1]), V::load(a.c32 + 16 * m)));
            sv[m] = V::fmul(V::fload(S.ws + 16 * m), V::fload(a.d + 16 * m));
        }
        return chunk_sum(sv, iv);
    }
    alignas(64) int32_t sc[160];
    alignas(64) float d[20], dm[20];
    for (int sb = 0; sb < 10; sb++) {
        const uint8_t *blk = row + sb * (type == 11 ? 110 : 84);
        if (type == 11) {
            decode_q3k_raw(blk, S.w + 256 * sb, sc + 16 * sb);
            d[sb] = _cvtsh_ss(*(const uint16_t *)(blk + 108));
        } else {
            decode_q2k_raw(blk, S.w + 256 * sb);
            V::store(sc + 16 * sb, V::u8x16(_mm_loadu_si128((const __m128i *)blk)));
            d[sb] = _cvtsh_ss(*(const uint16_t *)(blk + 80));
            dm[sb] = _cvtsh_ss(*(const uint16_t *)(blk + 82));
        }
    }
    for (int k = 0; k < 10; k++) g[k] = groups16<false>(S.w + 256 * k, a.q + 256 * k);
    const V::i m4 = V::set1(15);
    for (int m = 0; m < 5; m++) {
        const V::f dv = V::halves(d[2 * m], d[2 * m + 1]);
        const V::f xd = V::fload(a.d + 16 * m);
        V::i t[2], u[2];
        for (int h = 0; h < 2; h++) {
            const int k = 2 * m + h;
            const V::i s16 = V::load(a.s16 + 16 * k), scv = V::load(sc + 16 * k);
            if (type == 11) t[h] = V::mullo(V::sub(g[k], V::sll32(s16, 2)), scv);
            else {
                t[h] = V::mullo(g[k], V::and_(scv, m4));
                u[h] = V::mullo(s16, V::srl32(scv, 4));
            }
        }
        const V::f fa = V::cvt(V::pairs(t[0], t[1]));
        if (type == 11) { sv[m] = V::fmul(dv, xd); iv[m] = fa; continue; }
        const V::f fb = V::cvt(V::pairs(u[0], u[1]));
        const V::f dmv = V::halves(dm[2 * m], dm[2 * m + 1]);
        iv[m] = V::fms(dv, fa, V::fmul(dmv, fb));
        sv[m] = xd;
    }
    return chunk_sum(sv, iv);
}

TGT float emu_dn_row(const uint8_t *row, const int8_t *hq, const float *hd, const int32_t *hc, ExactScratch &S) {
    S.reserve();
    decode_iq4nl(row, FE_, S.w, S.ws);
    const V::i z = V::zero();
    const V::i g0 = groups16<true>(S.w, hq), g1 = groups16<true>(S.w + 256, hq + 256);
    const V::i g2 = V::pairs(V::pairs(dot64<true>(S.w + 512, hq + 512), dot64<true>(S.w + 576, hq + 576)), z);
    const V::f i0 = V::cvt(V::sub(V::pairs(g0, g1), V::load(hc)));
    const V::f i1 = V::cvt(V::sub(V::pairs(g2, z), V::load4z(hc + 16)));
    const V::f a = V::fmul_rn(V::fmul(V::fload(S.ws), V::fload(hd)), i0);
    const V::f b = V::fmul_rn(V::fmul4z(V::fload4z(S.ws + 16), V::fload4z(hd + 16)), i1);
    return V::lanes_sum(a, b);
}

const ExactKernels kernels = {quant_x, expert, dot_gu1, dot_gu2, quant_h, dot_dn1, emu_quant_x, emu_gu_row, emu_dn_row};
