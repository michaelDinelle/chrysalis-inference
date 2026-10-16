// Gated DeltaNet layers (36 of 48): causal conv + gates, the recurrent delta-rule scan, gated RMSNorm.
// Flash-Next's one difference from Qwen3.5's GDN: the output gate is sigmoid(z), not silu(z).
//
// Per value head h (key head h % 16), with the state S stored column-major (S[j][i], i = key dim):
//   S = exp(g) S;  delta_j = beta (v_j - sum_i S_ij k_i);  S_ij += k_i delta_j;  o_j = sum_i S_ij q_i / sqrt(128)
// where k, q are L2-normalized, g = A * softplus(alpha + dt_bias) and beta = sigmoid(b).
#include "kernels.h"

__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

// Depthwise causal conv (kernel 4) over T tokens + SiLU, then L2 norm for q/k heads.
// raw [T][QKV] -> qn, kn [T][GK][GD], vc [T][GV][GD]. state [QKV][3] holds the previous 3 inputs.
__global__ __launch_bounds__(128) void k_gdn_conv(const float *raw, int T, const float *w, float *state, int update,
                                                  float *qn, float *kn, float *vc) {
    __shared__ float red[4];
    const int head = blockIdx.x, i = threadIdx.x;            // heads: 16 q, 16 k, 48 v
    const int c = head * GD + i;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    for (int t = 0; t < T; t++) {
        const float x = raw[(size_t)t * QKV + c];
        float y = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
        s0 = s1; s1 = s2; s2 = x;
        if (head < 2 * GK) {
            const float ss = warp_sum(y * y);
            if ((i & 31) == 0) red[i >> 5] = ss;
            __syncthreads();
            y *= rsqrtf(red[0] + red[1] + red[2] + red[3] + EPS);
            __syncthreads();
            (head < GK ? qn : kn)[((size_t)t * GK + (head % GK)) * GD + i] = y;
        } else {
            vc[((size_t)t * GV + head - 2 * GK) * GD + i] = y;
        }
    }
    if (update) { state[c * 3] = s0; state[c * 3 + 1] = s1; state[c * 3 + 2] = s2; }
}

// Prefill version: blocks of 16 tokens run in parallel (each reads its 3 predecessors from raw or the
// old state), then k_gdn_conv_state saves the last 3 inputs as the new state.
__global__ __launch_bounds__(128) void k_gdn_conv_par(const float *raw, int T, const float *w, const float *state,
                                                      float *qn, float *kn, float *vc) {
    __shared__ float red[4];
    const int head = blockIdx.x, t0 = blockIdx.y * 16, i = threadIdx.x, c = head * GD + i;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    auto in = [&](int t) { return t >= 0 ? raw[(size_t)t * QKV + c] : state[c * 3 + 3 + t]; };
    float s0 = in(t0 - 3), s1 = in(t0 - 2), s2 = in(t0 - 1);
    for (int t = t0; t < min(T, t0 + 16); t++) {
        const float x = raw[(size_t)t * QKV + c];
        float y = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
        s0 = s1; s1 = s2; s2 = x;
        if (head < 2 * GK) {
            const float ss = warp_sum(y * y);
            if ((i & 31) == 0) red[i >> 5] = ss;
            __syncthreads();
            y *= rsqrtf(red[0] + red[1] + red[2] + red[3] + EPS);
            __syncthreads();
            (head < GK ? qn : kn)[((size_t)t * GK + (head % GK)) * GD + i] = y;
        } else {
            vc[((size_t)t * GV + head - 2 * GK) * GD + i] = y;
        }
    }
}
__global__ void k_gdn_conv_state(const float *raw, int T, float *state) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < QKV) for (int k = 0; k < 3; k++) state[c * 3 + k] = raw[(size_t)(T - 3 + k) * QKV + c];
}

void gdn_conv(const float *raw, int T, const float *w, float *state, bool update, float *qn, float *kn, float *vc, hipStream_t st) {
    if (T < 16 || !update) { k_gdn_conv<<<2 * GK + GV, 128, 0, st>>>(raw, T, w, state, update, qn, kn, vc); return; }
    k_gdn_conv_par<<<dim3(2 * GK + GV, (T + 15) / 16), 128, 0, st>>>(raw, T, w, state, qn, kn, vc);
    k_gdn_conv_state<<<(QKV + 255) / 256, 256, 0, st>>>(raw, T, state);
}

// Recompute the conv state after a verify pass: the last 3 inputs of [old state, accepted tokens].
// One launch covers all GDN layers (states and raw inputs are stored per layer, contiguously).
__global__ void k_gdn_conv_commit(float *state, const float *raw, int n, int raw_layer_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, layer = blockIdx.y;
    if (c >= QKV) return;
    float *s = state + ((size_t)layer * QKV + c) * 3;
    const float *r = raw + (size_t)layer * raw_layer_stride + c;
    float w[3] = {s[0], s[1], s[2]};
    for (int t = 0; t < n; t++) { w[0] = w[1]; w[1] = w[2]; w[2] = r[(size_t)t * QKV]; }
    s[0] = w[0]; s[1] = w[1]; s[2] = w[2];
}

void gdn_conv_commit(float *state, const float *raw, int n, int raw_layer_stride, int nlayers, hipStream_t st) {
    k_gdn_conv_commit<<<dim3((QKV + 255) / 256, nlayers), 256, 0, st>>>(state, raw, n, raw_layer_stride);
}



// The delta-rule scan. Grid (GV heads, 4 column blocks, layers); 128 threads = 32 columns x 4 row-quarters.
// State stays in registers across tokens; k, q, v and the gates of CT tokens at a time are staged in LDS
// (rows padded so the 4 row-quarters hit different banks). out == nullptr skips outputs (commit);
// write_state=0 leaves S untouched (verify).
constexpr int CT = 32, PADQ = 36;
__global__ __launch_bounds__(128) void k_gdn_scan(GdnScanArgs a) {
    __shared__ __attribute__((aligned(16))) float ks[CT][4 * PADQ], qs[CT][4 * PADQ];
    __shared__ float vs[CT][32], dec[CT], bet[CT];
    const int h = blockIdx.x, kh = h % GK, layer = blockIdx.z, tid = threadIdx.x;
    const int j = blockIdx.y * 32 + tid / 4, part = tid % 4;
    float *Sj = a.S + (size_t)layer * a.S_layer + ((size_t)h * GD + j) * GD + part * 32;
    const float *qn = a.qn + (size_t)layer * a.in_layer, *kn = a.kn + (size_t)layer * a.in_layer;
    const float *vc = a.vc + (size_t)layer * a.v_layer, *ab = a.ab + (size_t)layer * a.ab_layer;
    float s[32];
#pragma unroll
    for (int i = 0; i < 32; i += 4) *(float4 *)&s[i] = *(const float4 *)&Sj[i];
    const float An = a.A[layer * GV + h], db = a.dtb[layer * GV + h];
    for (int t0 = 0; t0 < a.T; t0 += CT) {
        const int n = min(CT, a.T - t0);
        __syncthreads();
        for (int idx = tid; idx < n * GD / 4; idx += 128) {
            const int t = idx / (GD / 4), c4 = idx % (GD / 4), at = (c4 / 8) * PADQ + (c4 % 8) * 4;
            *(float4 *)&ks[t][at] = *(const float4 *)&kn[((size_t)(t0 + t) * GK + kh) * GD + c4 * 4];
            if (a.out) *(float4 *)&qs[t][at] = *(const float4 *)&qn[((size_t)(t0 + t) * GK + kh) * GD + c4 * 4];
        }
        for (int idx = tid; idx < n * 32; idx += 128)
            vs[idx / 32][idx % 32] = vc[((size_t)(t0 + idx / 32) * GV + h) * GD + blockIdx.y * 32 + idx % 32];
        if (tid < n) {
            dec[tid] = __expf(An * softplus(ab[(t0 + tid) * 2 * GV + h] + db));
            bet[tid] = sigmoid(ab[(t0 + tid) * 2 * GV + GV + h]);
        }
        __syncthreads();
        for (int t = 0; t < n; t++) {
            const float decay = dec[t], beta = bet[t];
            const float *kp = &ks[t][part * PADQ];
            float kv = 0;
#pragma unroll
            for (int i = 0; i < 32; i++) kv += s[i] * kp[i];
            kv += __shfl_xor(kv, 1, 32); kv += __shfl_xor(kv, 2, 32);
            const float delta = beta * (vs[t][tid / 4] - decay * kv);
#pragma unroll
            for (int i = 0; i < 32; i++) s[i] = decay * s[i] + kp[i] * delta;
            if (a.out) {
                const float *qp = &qs[t][part * PADQ];
                float o = 0;
#pragma unroll
                for (int i = 0; i < 32; i++) o += s[i] * qp[i];
                o += __shfl_xor(o, 1, 32); o += __shfl_xor(o, 2, 32);
                if (part == 0) a.out[((size_t)(t0 + t) * GV + h) * GD + j] = o * 0.08838834764831845f;   // 1/sqrt(128)
            }
        }
    }
    if (a.write_state) {
#pragma unroll
        for (int i = 0; i < 32; i += 4) *(float4 *)&Sj[i] = *(const float4 *)&s[i];
    }
}

void gdn_scan(const GdnScanArgs &a, int nlayers, hipStream_t st) {
    k_gdn_scan<<<dim3(GV, GD / 32, nlayers), 128, 0, st>>>(a);
}

// Gated RMSNorm per value head: y = rmsnorm(o) * w * sigmoid(z). Writes fp16 and/or int8 blocks.
__global__ __launch_bounds__(128) void k_gdn_out(const float *o, const float *z, const float *w, half *oh, QAct oq) {
    __shared__ float red[4];
    const size_t idx = ((size_t)blockIdx.y * GV + blockIdx.x) * GD + threadIdx.x;
    const float v = o[idx];
    const float ss = warp_sum(v * v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    const float y = v * rsqrtf((red[0] + red[1] + red[2] + red[3]) / GD + EPS) * w[threadIdx.x] * sigmoid(z[idx]);
    if (oh) oh[idx] = __float2half(y);
    if (oq.q) {
        const float d = warp_max(fabsf(y)) / 127.0f;
        oq.q[idx] = (int8_t)__float2int_rn(d > 0 ? y / d : 0.0f);
        if ((threadIdx.x & 31) == 0) oq.d[idx / 32] = d;
    }
}

void gdn_out(const float *o, const float *z, const float *w, int T, half *oh, QAct oq, hipStream_t st) {
    k_gdn_out<<<dim3(GV, T), 128, 0, st>>>(o, z, w, oh, oq);
}

// Decode / verify path (T <= 8): one 512-thread workgroup per value head runs alpha/beta, the conv,
// L2 norms, the scan (state in registers) and the gated norm, writing int8 input for ssm_out.
// The conv state is never written here (gdn_conv_commit does it after all layers); verify mode also
// leaves the recurrent state untouched and records k, v and the gates for a later commit.
__global__ __launch_bounds__(512) void k_gdn_step(GdnStepArgs a) {
    __shared__ float qs[8][GD], ks[8][GD], vs[8][GD], os[8][GD];
    __shared__ float red[16][16], gates[8][2], rms[8];
    const int h = blockIdx.x, kh = h % GK, tid = threadIdx.x, warp = tid / 32, lane = tid % 32, T = a.T;

    float pa[8] = {}, pb[8] = {};                                  // alpha / beta projections (fp16 dot products)
    const half2 *wa = (const half2 *)(a.Wab + (size_t)h * D), *wb = (const half2 *)(a.Wab + (size_t)(GV + h) * D);
    for (int i = tid; i < D / 2; i += 512) {
        const float2 fa = __half22float2(wa[i]), fb = __half22float2(wb[i]);
        for (int t = 0; t < T; t++) {
            const float2 x = __half22float2(((const half2 *)(a.hh + (size_t)t * D))[i]);
            pa[t] += fa.x * x.x + fa.y * x.y;
            pb[t] += fb.x * x.x + fb.y * x.y;
        }
    }
    for (int t = 0; t < T; t++) {
        const float sa = warp_sum(pa[t]), sb = warp_sum(pb[t]);
        if (lane == 0) { red[warp][2 * t] = sa; red[warp][2 * t + 1] = sb; }
    }
    __syncthreads();
    if (tid < T) {
        float al = 0, be = 0;
        for (int w = 0; w < 16; w++) { al += red[w][2 * tid]; be += red[w][2 * tid + 1]; }
        gates[tid][0] = __expf(a.A[h] * softplus(al + a.dt[h]));
        gates[tid][1] = sigmoid(be);
        if (a.verify) { a.ab_out[tid * 2 * GV + h] = al; a.ab_out[tid * 2 * GV + GV + h] = be; }
    }

    if (tid < 3 * GD) {                                            // conv + silu: this head's q, k, v channels
        const int which = tid / GD, i = tid % GD;
        const int c = which == 0 ? kh * GD + i : which == 1 ? GK * GD + kh * GD + i : 2 * GK * GD + h * GD + i;
        const float w0 = a.conv_w[c * 4], w1 = a.conv_w[c * 4 + 1], w2 = a.conv_w[c * 4 + 2], w3 = a.conv_w[c * 4 + 3];
        const float *cs = a.conv_state + c * 3;
        float s0 = cs[0], s1 = cs[1], s2 = cs[2];
        float (*dst)[GD] = which == 0 ? qs : which == 1 ? ks : vs;
        for (int t = 0; t < T; t++) {
            const float x = a.raw[(size_t)t * QKV + c];
            dst[t][i] = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
            s0 = s1; s1 = s2; s2 = x;
        }
        // the new conv state is written later by gdn_conv_commit: three heads read each q/k channel here
    }
    __syncthreads();
    for (int r = warp; r < 2 * T; r += 16) {                       // L2-normalize q and k rows
        float *row = (r & 1) ? ks[r / 2] : qs[r / 2];
        float4 v = *(float4 *)&row[lane * 4];
        const float inv = rsqrtf(warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) + EPS);
        v.x *= inv; v.y *= inv; v.z *= inv; v.w *= inv;
        *(float4 *)&row[lane * 4] = v;
    }
    __syncthreads();
    if (a.verify) {
        for (int idx = tid; idx < T * GD; idx += 512) {
            const int t = idx / GD, i = idx % GD;
            if (h < GK) a.kn_out[((size_t)t * GK + kh) * GD + i] = ks[t][i];
            a.vc_out[((size_t)t * GV + h) * GD + i] = vs[t][i];
        }
    }

    const int j = tid / 4, part = tid % 4;                         // scan: 128 columns x 4 row quarters
    float *Sj = a.S + ((size_t)h * GD + j) * GD + part * 32;
    float st[32];
#pragma unroll
    for (int i = 0; i < 32; i += 4) *(float4 *)&st[i] = *(const float4 *)&Sj[i];
    for (int t = 0; t < T; t++) {
        const float decay = gates[t][0], beta = gates[t][1];
        const float *kp = &ks[t][part * 32], *qp = &qs[t][part * 32];
        float kv = 0;
#pragma unroll
        for (int i = 0; i < 32; i++) kv += st[i] * kp[i];
        kv += __shfl_xor(kv, 1, 32); kv += __shfl_xor(kv, 2, 32);
        const float delta = beta * (vs[t][j] - decay * kv);
        float o = 0;
#pragma unroll
        for (int i = 0; i < 32; i++) { st[i] = decay * st[i] + kp[i] * delta; o += st[i] * qp[i]; }
        o += __shfl_xor(o, 1, 32); o += __shfl_xor(o, 2, 32);
        if (part == 0) os[t][j] = o * 0.08838834764831845f;
    }
    if (!a.verify) {
#pragma unroll
        for (int i = 0; i < 32; i += 4) *(float4 *)&Sj[i] = *(const float4 *)&st[i];
    }
    __syncthreads();

    if (warp < T) {                                                // gated RMSNorm, int8 out for ssm_out
        float4 v = *(float4 *)&os[warp][lane * 4];
        const float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        if (lane == 0) rms[warp] = rsqrtf(ss / GD + EPS);
    }
    __syncthreads();
    for (int r = warp; r < 4 * T; r += 16) {
        const int t = r / 4, col = (r % 4) * 32 + lane;
        const size_t idx = (size_t)t * GV * GD + h * GD + col;
        const float y = os[t][col] * rms[t] * a.norm_w[col] * sigmoid(a.z[idx]);
        const float d = warp_max(fabsf(y)) / 127.0f;
        a.out.q[idx] = (int8_t)__float2int_rn(d > 0 ? y / d : 0.0f);
        if (lane == 0) a.out.d[idx / 32] = d;
    }
}

void gdn_step(const GdnStepArgs &a, hipStream_t st) {
    k_gdn_step<<<GV, 512, 0, st>>>(a);
}
