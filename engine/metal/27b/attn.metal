#include "kcommon.h"

inline device uchar *kv_block(device uchar *pool, device const long *pages, int layer, int head, int pos) {
    return pool + pages[pos / PAGE] + (ulong)layer * KV_LAYER + (ulong)head * KV_HEAD;
}

kernel void attn_prep(constant PrepA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort i [[thread_index_in_threadgroup]]) {
    threadgroup float red[8], buf[HD];
    const int job = tg.x + a.job0, t = tg.y, pos = a.pos0 + t;
    float x;
    if (job < NH) x = a.qg[((ulong)t * NH + job) * 2 * HD + i];
    else if (job < NH + NKV) x = a.k[((ulong)t * NKV + job - NH) * HD + i];
    else x = a.v[((ulong)t * NKV + job - NH - NKV) * HD + i];
    if (job < NH + NKV) {
        float ss = simd_sum(x * x);
        if ((i & 31) == 0) red[i >> 5] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = 0;
        for (int k = 0; k < 8; k++) ss += red[k];
        x *= rsqrt(ss / HD + EPS) * (job < NH ? a.qnw : a.knw)[i];
        if (i < ROT) buf[i] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i < ROT) {
            const int f = i % (ROT / 2);
            const int sec = f % 3 == 1 && f < 33 ? 1 : f % 3 == 2 && f < 30 ? 2 : 0;
            const int rp = a.rpos ? a.rpos[t * 3 + sec] : a.rope0 + t;
            const float ang = float(rp) * pow(ROPE_BASE, -2.0f * f / ROT);
            const float s = precise::sin(ang), c = precise::cos(ang);
            x = i < ROT / 2 ? x * c - buf[i + ROT / 2] * s : x * c + buf[i - ROT / 2] * s;
        }
    }
    if (job < NH) {
        const int g = job / 6, r = t * 6 + job % 6;
        a.Qg[((ulong)g * a.qrows + r) * HD + i] = half(x * 0.0625f);
        return;
    }
    const bool isv = job >= NH + NKV;
    const int head = (job - NH) % NKV, slot = pos % PAGE;
    device uchar *blk = kv_block(a.pool, a.pages, a.layer, head, pos);
    const half d = half(simd_max(fabs(x)) / 127.0f);
    const float df = float(d);
    ((device char *)blk)[isv ? KV_V + v_off(slot, i) : k_off(slot, i)] = char(df > 0 ? rint(x / df) : 0.0f);
    if ((i & 31) == 0) ((device half *)(blk + (isv ? KV_VS : KV_KS)))[slot * (HD / 32) + i / 32] = d;
}

template <int R>
kernel void attn_vec(constant AttnA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int KT = 4;
    const int split = tg.x, g = tg.y, nrows = a.T * 6, rb = sg * R;
    const int kbeg = split * a.split_len, kend = min(a.pos0 + a.T, kbeg + a.split_len);
    half4 q0[R], q1[R];
    for (int r = 0; r < R; r++) {
        half v[8];
        for (int i = 0; i < 8; i++) {
            const int p = 8 * lane + i, dim = 8 * ((p % 64) / 2) + 2 * (p / 64) + p % 2;
            v[i] = rb + r < nrows ? a.Qg[((ulong)g * a.qrows + rb + r) * HD + dim] : 0.0h;
        }
        q0[r] = half4(v[0], v[1], v[2], v[3]); q1[r] = half4(v[4], v[5], v[6], v[7]);
    }
    const int kblk = lane % 8;
    const int vf = lane % 8, vd = (lane / 8) * 8;
    float m[R], l[R];
    float4 o0[R], o1[R];
    for (int r = 0; r < R; r++) { m[r] = -INFINITY; l[r] = 0; o0[r] = 0; o1[r] = 0; }
    for (int k0 = kbeg; k0 < kend; k0 += KT) {
        device const uchar *blk = a.pool + a.pages[k0 / PAGE] + (ulong)a.layer * KV_LAYER + (ulong)g * KV_HEAD;
        const int s0 = k0 % PAGE, n = min(KT, kend - k0);
        float p[KT][R];
        for (int j = 0; j < KT; j++) {
            const uint2 w = *(device const uint2 *)(blk + (s0 + j) * HD + 8 * lane);
            const half ks = ((device const half *)(blk + KV_KS))[(s0 + j) * 8 + kblk];
            const half4 k0v = half4(as_type<char4>(w.x)) * ks, k1v = half4(as_type<char4>(w.y)) * ks;
            for (int r = 0; r < R; r++) p[j][r] = simd_sum(float(dot(k0v, q0[r])) + float(dot(k1v, q1[r])));
        }
        for (int r = 0; r < R; r++) {
            const int qpos = a.pos0 + (rb + r) / 6;
            float mx = m[r];
            for (int j = 0; j < KT; j++) {
                if (j >= n || k0 + j > qpos) p[j][r] = -INFINITY;
                mx = max(mx, p[j][r]);
            }
            const float al = mx == -INFINITY ? 1.0f : exp(m[r] - mx);
            float ps = 0;
            for (int j = 0; j < KT; j++) { p[j][r] = p[j][r] == -INFINITY ? 0.0f : exp(p[j][r] - mx); ps += p[j][r]; }
            l[r] = l[r] * al + ps;
            m[r] = mx;
            o0[r] *= al; o1[r] *= al;
        }
        const int g0 = (s0 % 8) / 2;
        for (int jp = 0; jp < KT / 2; jp++) {
            const int gg = g0 + jp, sl = s0 + 2 * jp;
            const uint4 w = *(device const uint4 *)(blk + KV_V + (((s0 / 8) * 8 + vf) * 4 + gg) * 64 + 2 * vd);
            device const half *vs = (device const half *)(blk + KV_VS);
            const float a0 = float(vs[sl * 8 + vd / 4]), a1 = float(vs[sl * 8 + vd / 4 + 1]);
            const float b0 = float(vs[(sl + 1) * 8 + vd / 4]), b1 = float(vs[(sl + 1) * 8 + vd / 4 + 1]);
            const char4 c0 = as_type<char4>(w.x), c1 = as_type<char4>(w.y), c2 = as_type<char4>(w.z), c3 = as_type<char4>(w.w);
            const float4 k0lo = float4(c0.x, c0.z, c1.x, c1.z) * a0, k0hi = float4(c2.x, c2.z, c3.x, c3.z) * a1;
            const float4 k1lo = float4(c0.y, c0.w, c1.y, c1.w) * b0, k1hi = float4(c2.y, c2.w, c3.y, c3.w) * b1;
            for (int r = 0; r < R; r++) {
                o0[r] += p[2 * jp][r] * k0lo + p[2 * jp + 1][r] * k1lo;
                o1[r] += p[2 * jp][r] * k0hi + p[2 * jp + 1][r] * k1hi;
            }
        }
    }
    const ulong base = ((ulong)split * NKV + g) * a.qrows;
    for (int r = 0; r < R && rb + r < nrows; r++) {
        device float *op = a.part_o + (base + rb + r) * HD + vf;
        for (int i = 0; i < 4; i++) { op[8 * (vd + i)] = o0[r][i]; op[8 * (vd + 4 + i)] = o1[r][i]; }
        if (lane == 0) { a.part_ml[(base + rb + r) * 2] = m[r]; a.part_ml[(base + rb + r) * 2 + 1] = l[r]; }
    }
}
template [[host_name("attn_vec")]] kernel decltype(attn_vec<3>) attn_vec<3>;

constant constexpr int KB = 16, KLD = HD + 8;

kernel void attn_pf(constant AttnA &a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NSG = 16, NR = NSG / 2;
    threadgroup half Ks[KB * KLD], Vs[KB * KLD];
    threadgroup float Ss[NSG][8 * KB];
    threadgroup half Ps[NSG][8 * KB];
    threadgroup float Al[NSG][8];
    const int split = tg.x, g = tg.y, nrows = a.T * 6, rt = sg / 2, hf = sg % 2;
    const int tile0 = tg.z * NR * 8, r0 = tile0 + rt * 8;
    const int kbeg = split * a.split_len;
    const int last_row = min(tile0 + NR * 8, nrows) - 1;
    const int kend = min(a.pos0 + last_row / 6 + 1, kbeg + a.split_len);
    const bool active = r0 < nrows;
    simdgroup_half8x8 Q[16];
    simdgroup_float8x8 O[16];
    device const half *qp = a.Qg + ((ulong)g * a.qrows + r0) * HD + hf * 128;
    for (int d = 0; d < 16; d++) {
        simdgroup_load(Q[d], qp + d * 8, HD);
        O[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const int row = lane / 4, part = lane % 4;
    const int qpos = a.pos0 + (r0 + row) / 6;
    const int fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    float m = -INFINITY, l = 0.0f;
    for (int k0 = kbeg; k0 < kend; k0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < KB * 16; i += NSG * 32) {
            {
                const int kk = i / 16, gg = (i / 4) % 4, q4 = i % 4, pos = k0 + kk;
                uint4 w = 0; half s0 = 0, s1 = 0;
                if (pos < kend) {
                    device const uchar *blk = kv_block((device uchar *)a.pool, a.pages, a.layer, g, pos);
                    const int slot = pos % PAGE;
                    w = *(device const uint4 *)(blk + slot * HD + gg * 64 + q4 * 16);
                    s0 = ((device const half *)(blk + KV_KS))[slot * 8 + 2 * q4];
                    s1 = ((device const half *)(blk + KV_KS))[slot * 8 + 2 * q4 + 1];
                }
                const uint ws[4] = {w.x, w.y, w.z, w.w};
                for (int h = 0; h < 8; h++) {
                    const uint pr = ws[h / 2] >> (16 * (h & 1));
                    const half sc = h < 4 ? s0 : s1;
                    *(threadgroup half2 *)&Ks[kk * KLD + 8 * (8 * q4 + h) + 2 * gg] = half2(half(char(pr & 0xff)), half(char((pr >> 8) & 0xff))) * sc;
                }
            }
            {
                const int kt = i / 128, f = (i / 16) % 8, gg = (i / 4) % 4, q4 = i % 4;
                const int key0 = k0 + kt * 8 + 2 * gg;
                uint4 w = 0; half4 sc = 0;
                if (key0 < kend) {
                    device const uchar *blk = kv_block((device uchar *)a.pool, a.pages, a.layer, g, key0);
                    const int slot = key0 % PAGE;
                    w = *(device const uint4 *)(blk + KV_V + (((slot / 8) * 8 + f) * 4 + gg) * 64 + q4 * 16);
                    device const half *vs = (device const half *)(blk + KV_VS);
                    sc = half4(vs[slot * 8 + 2 * q4], key0 + 1 < kend ? vs[(slot + 1) * 8 + 2 * q4] : 0.0h,
                               vs[slot * 8 + 2 * q4 + 1], key0 + 1 < kend ? vs[(slot + 1) * 8 + 2 * q4 + 1] : 0.0h);
                }
                const uint ws[4] = {w.x, w.y, w.z, w.w};
                const int r0 = (kt * 8 + 2 * gg) * KLD;
                for (int h = 0; h < 8; h++) {
                    const uint pr = ws[h / 2] >> (16 * (h & 1));
                    const int dim = 8 * (8 * q4 + h) + f;
                    Vs[r0 + dim] = half(char(pr & 0xff)) * (h < 4 ? sc.x : sc.z);
                    Vs[r0 + KLD + dim] = half(char((pr >> 8) & 0xff)) * (h < 4 ? sc.y : sc.w);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            simdgroup_float8x8 S0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), S1 = S0;
            for (int d = 0; d < 16; d++) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, &Ks[hf * 128 + d * 8], KLD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S0, Q[d], Kt, S0);
                simdgroup_load(Kt, &Ks[8 * KLD + hf * 128 + d * 8], KLD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S1, Q[d], Kt, S1);
            }
            simdgroup_store(S0, &Ss[sg][0], KB);
            simdgroup_store(S1, &Ss[sg][8], KB);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) continue;
        float s[4], mx = -INFINITY;
        for (int j = 0; j < 4; j++) {
            const int key = k0 + part * 4 + j, at = row * KB + part * 4 + j;
            s[j] = key > qpos || key >= kend ? -INFINITY : Ss[rt * 2][at] + Ss[rt * 2 + 1][at];
            mx = max(mx, s[j]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx);
        const float alpha = mn == -INFINITY ? 1.0f : exp(m - mn);
        float ps = 0;
        for (int j = 0; j < 4; j++) {
            const float p = s[j] == -INFINITY ? 0.0f : exp(s[j] - mn);
            Ps[sg][row * KB + part * 4 + j] = half(p);
            ps += p;
        }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 2);
        l = l * alpha + ps;
        m = mn;
        if (part == 0) Al[sg][row] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float al = Al[sg][fm];
        simdgroup_half8x8 P0, P1;
        simdgroup_load(P0, &Ps[sg][0], KB);
        simdgroup_load(P1, &Ps[sg][8], KB);
        for (int d = 0; d < 16; d++) {
            O[d].thread_elements()[0] *= al;
            O[d].thread_elements()[1] *= al;
            simdgroup_half8x8 V;
            simdgroup_load(V, &Vs[hf * 128 + d * 8], KLD);
            simdgroup_multiply_accumulate(O[d], P0, V, O[d]);
            simdgroup_load(V, &Vs[8 * KLD + hf * 128 + d * 8], KLD);
            simdgroup_multiply_accumulate(O[d], P1, V, O[d]);
        }
    }
    if (!active) return;
    const ulong base = ((ulong)split * NKV + g) * a.qrows + r0;
    for (int d = 0; d < 16; d++) simdgroup_store(O[d], a.part_o + base * HD + hf * 128 + d * 8, HD);
    if (part == 0 && hf == 0) { a.part_ml[(base + row) * 2] = m; a.part_ml[(base + row) * 2 + 1] = l; }
}

kernel void kv_unpack(constant UnpackA &a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]], ushort i [[thread_index_in_threadgroup]]) {
    const int pos = tg.x * 8 + i / 32, dim0 = (i % 32) * 8;
    if (pos >= a.L) return;
    device const uchar *blk = a.pool + a.pages[pos / PAGE] + (ulong)a.layer * KV_LAYER + (ulong)a.head * KV_HEAD;
    const int slot = pos % PAGE;
    const half ks = ((device const half *)(blk + KV_KS))[slot * 8 + dim0 / 32];
    const half vs = ((device const half *)(blk + KV_VS))[slot * 8 + dim0 / 32];
    for (int j = 0; j < 8; j++) {
        const int dim = dim0 + j;
        a.kc[(ulong)pos * HD + dim] = half(((device const char *)blk)[k_off(slot, dim)]) * ks;
        a.vt[(ulong)dim * a.L + pos] = half(((device const char *)blk + KV_V)[v_off(slot, dim)]) * vs;
    }
}

kernel void attn_smax(constant SmaxA &a [[buffer(0)]], uint r [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8];
    const int qpos = a.pos0 + (a.r0 + int(r)) / 6;
    device const float *S = a.S + (ulong)r * a.n;
    float mx = -INFINITY;
    for (int j = tid; j < a.n; j += 256) if (a.k0 + j <= qpos) mx = max(mx, S[j]);
    mx = simd_max(mx);
    if ((tid & 31) == 0) red[tid >> 5] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (int k = 1; k < 8; k++) mx = max(mx, red[k]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool first = a.k0 == 0;
    const float m0 = first ? -INFINITY : a.ml[r * 2], l0 = first ? 0.0f : a.ml[r * 2 + 1];
    const float mn = max(m0, mx);
    float ps = 0;
    device half *P = a.P + (ulong)r * a.n;
    for (int j = tid; j < a.n; j += 256) {
        const float p = a.k0 + j <= qpos && mn != -INFINITY ? exp(S[j] - mn) : 0.0f;
        P[j] = half(p);
        ps += p;
    }
    ps = simd_sum(ps);
    if ((tid & 31) == 0) red[tid >> 5] = ps;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0;
        for (int k = 0; k < 8; k++) t += red[k];
        const float al = m0 == -INFINITY ? 0.0f : exp(m0 - mn);
        a.ml[r * 2] = mn;
        a.ml[r * 2 + 1] = l0 * al + t;
    }
    const float al = m0 == -INFINITY ? 0.0f : exp(m0 - mn);
    device float *o = a.O + (ulong)r * HD;
    o[tid] = first ? 0.0f : o[tid] * al;
}

kernel void attn_combine(constant CombineA &a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]]) {
    const int t = tg / NH, h = tg % NH, g = h / 6, r = t * 6 + h % 6;
    float M = -INFINITY;
    for (int s = 0; s < a.nsplit; s++) M = max(M, a.part_ml[(((ulong)s * NKV + g) * a.qrows + r) * 2]);
    float L = 0, o = 0;
    for (int s = 0; s < a.nsplit; s++) {
        const ulong b = ((ulong)s * NKV + g) * a.qrows + r;
        const float ms = a.part_ml[b * 2];
        if (ms == -INFINITY) continue;
        const float w = exp(ms - M);
        L += w * a.part_ml[b * 2 + 1];
        o += w * a.part_o[b * HD + d];
    }
    const float y = (L > 0 ? o / L : 0.0f) * sigmoid(a.qg[((ulong)t * NH + h) * 2 * HD + HD + d]);
    const ulong oi = (ulong)t * NH * HD + h * HD + d;
    if (a.of) a.of[oi] = y;
    if (a.oh) a.oh[oi] = half(y);
}
