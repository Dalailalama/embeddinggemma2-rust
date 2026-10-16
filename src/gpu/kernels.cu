// CUDA kernels for the EmbeddingGemma 2 GPU backend. Compiled at runtime with NVRTC.
// All tensors are dense row-major. Every kernel mirrors a CPU routine in src/ops.rs,
// src/text.rs, src/vision.rs or src/audio.rs; see those for the reference semantics.
//
// Storage types: tensors are float32, except that in bf16 mode (EG_BF16 defined) the activations that
// only feed matrix products are stored as bfloat16 bits (`bf16_t`). Kernels that touch those are
// templates over their input / output storage type, reading through ld() and writing through st(): the
// arithmetic is float32 either way, and a bf16 store rounds to nearest even exactly like the conversion
// pass that used to sit in front of each product. The float32 instantiations keep their names and
// results; the bf16 ones (suffix _fb = float32 in / bf16 out, _bb = bf16 in / bf16 out) are only
// compiled in bf16 mode.

// NVRTC has no <math.h>.
#define INFINITY __int_as_float(0x7f800000)

typedef unsigned short bf16_t;

// float32 -> bfloat16 bits, round to nearest even (NaN stays a quiet NaN). Written out by hand
// because NVRTC is run without the CUDA include directory (<cuda_bf16.h>).
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

__device__ __forceinline__ float ld(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float ld(const bf16_t* p, long long i) { return __uint_as_float(((unsigned int)p[i]) << 16); }
__device__ __forceinline__ void st(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void st(bf16_t* p, long long i, float v) { p[i] = f32_to_bf16_rne(v); }

// GEMM input for the bf16 path: y = bf16(clamp(x, lo, hi)) (pass +-inf for no clamping). Four
// values per thread when the length allows it.
extern "C" __global__ void to_bf16(const float* x, unsigned short* y, long long n, float lo, float hi) {
    long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(x + i);
        ushort4 o;
        o.x = f32_to_bf16_rne(fminf(fmaxf(v.x, lo), hi));
        o.y = f32_to_bf16_rne(fminf(fmaxf(v.y, lo), hi));
        o.z = f32_to_bf16_rne(fminf(fmaxf(v.z, lo), hi));
        o.w = f32_to_bf16_rne(fminf(fmaxf(v.w, lo), hi));
        *reinterpret_cast<ushort4*>(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = f32_to_bf16_rne(fminf(fmaxf(x[i], lo), hi));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Block-wide reductions (blockDim.x must be a multiple of 32, at most 1024).
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = (lane < nw) ? sh[lane] : 0.f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ float block_max(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = (lane < nw) ? sh[lane] : -INFINITY;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float beta = 0.7978845608f, kappa = 0.044715f;
    float inner = beta * (x + kappa * x * x * x);
    return 0.5f * x * (1.f + tanhf(inner));
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }

#define ELEMWISE_IDX long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;

// The row kernels below reduce with the same per-thread order (thread t sums elements t, t+B, ...)
// and the same explicit roundings as the original one-op-per-kernel pipeline, so fusing them changes
// no bit of the result by itself (__fmul_rn/__fadd_rn stop the compiler from contracting a product
// that used to be stored to memory into an FMA with the next operation).

// out[row] = act(clamp_out((cin(x[row]) * pre) * (mean((cin(x) * pre)^2) + eps)^-0.5 (* w))), where cin is a clamp
// to [in_lo, in_hi] (+-inf: none), act is SiLU when `act` is set, and clamp_out clamps to [out_lo, out_hi].
// x == nullptr: in place on out (float32 only).
template <typename TO>
__device__ __forceinline__ void rms_norm_row(const float* r, TO* o, const float* w, int dim, float eps, int has_w, float pre,
                                             float in_lo, float in_hi, float out_lo, float out_hi, int act) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = __fmul_rn(fminf(fmaxf(r[i], in_lo), in_hi), pre); ss = fmaf(v, v, ss); }
    ss = block_sum(ss);
    float inv = powf(ss / (float)dim + eps, -0.5f);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float n = __fmul_rn(__fmul_rn(fminf(fmaxf(r[i], in_lo), in_hi), pre), inv);
        if (has_w) n = __fmul_rn(n, w[i]);
        if (act) n = silu_f(n);
        st(o, i, fminf(fmaxf(n, out_lo), out_hi));
    }
}
extern "C" __global__ void rms_norm(const float* x, float* out, const float* w, int rows, int dim, float eps, int has_w, float pre,
                                    float in_lo, float in_hi, float out_lo, float out_hi, int act) {
    int row = blockIdx.x;
    if (row >= rows) return;
    rms_norm_row((x ? x : out) + (long long)row * dim, out + (long long)row * dim, w, dim, eps, has_w, pre, in_lo, in_hi, out_lo, out_hi, act);
}

// Residual block tail, one block per row:  c = clamp(o[row], lo, hi);  n = rmsnorm(c) * w1 (n = c when w1 == nullptr);
//   mode 0: h[row] = (h[row] + n) * scalar      mode 1: h[row] = fma(n, scalar, h[row])
// and, when t != nullptr, t[row] = clamp(rmsnorm(clamp(h[row], t_in)) * w2, t_out) (the next sub-block's pre-norm),
// or with w2 == nullptr a plain copy t[row] = clamp(clamp(h[row], t_in), t_out) (a matrix product's input).
template <typename TI, typename TT>
__device__ __forceinline__ void add_norm_row(float* hr, const TI* orow, const float* w1, TT* tr, const float* w2,
                                             int dim, float eps, float scalar, float lo, float hi, int mode,
                                             float t_in_lo, float t_in_hi, float t_out_lo, float t_out_hi) {
    float inv1 = 1.f;
    if (w1) {
        float ss = 0.f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = fminf(fmaxf(ld(orow, i), lo), hi); ss = fmaf(v, v, ss); }
        ss = block_sum(ss);
        inv1 = powf(ss / (float)dim + eps, -0.5f);
    }
    float ss2 = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = fminf(fmaxf(ld(orow, i), lo), hi);
        float n = w1 ? __fmul_rn(__fmul_rn(v, inv1), w1[i]) : v;
        float hv = mode ? fmaf(n, scalar, hr[i]) : __fmul_rn(__fadd_rn(hr[i], n), scalar);
        hr[i] = hv;
        float tv = fminf(fmaxf(hv, t_in_lo), t_in_hi);
        ss2 = fmaf(tv, tv, ss2);
    }
    if (!tr) return;
    if (!w2) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) st(tr, i, fminf(fmaxf(fminf(fmaxf(hr[i], t_in_lo), t_in_hi), t_out_lo), t_out_hi));
        return;
    }
    ss2 = block_sum(ss2);
    float inv2 = powf(ss2 / (float)dim + eps, -0.5f);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float tv = fminf(fmaxf(hr[i], t_in_lo), t_in_hi);
        st(tr, i, fminf(fmaxf(__fmul_rn(__fmul_rn(tv, inv2), w2[i]), t_out_lo), t_out_hi));
    }
}
#define ADD_NORM(NAME, TI, TT) \
    extern "C" __global__ void NAME(float* h, const TI* o, const float* w1, TT* t, const float* w2, \
                                    int rows, int dim, float eps, float scalar, float lo, float hi, int mode, \
                                    float t_in_lo, float t_in_hi, float t_out_lo, float t_out_hi) { \
        int row = blockIdx.x; \
        if (row >= rows) return; \
        long long off = (long long)row * dim; \
        add_norm_row(h + off, o + off, w1, t ? t + off : (TT*)nullptr, w2, dim, eps, scalar, lo, hi, mode, t_in_lo, t_in_hi, t_out_lo, t_out_hi); \
    }
ADD_NORM(add_norm, float, float)

// Attention input prep, one block per (row, head slot) over [q heads | k heads | v heads]:
// per-head RMS norm (q, k with weights, v without) and RoPE on q and k, written to q / k / v
// (`[rows, nh * hd]`, `[rows, nkv * hd]`). Head j of the q / k / v part of row r is read from
// sq / sk / sv + r * stride + j * hd: one fused projection output `qkv[row, (nh + 2 nkv) * hd]`
// (pointers at its q | k | v columns, stride (nh + 2 nkv) * hd), three separate projection outputs, or
// q / k / v themselves (in place, float32 only).
// RoPE rotates each head in parts of `part` channels with the split-halves rule using
// cos/sin[pos[row], part_index*part + i] (tables are [positions, hd]); `part == hd` is 1-D RoPE,
// `hd / 2` the vision tower's 2-D RoPE.
template <typename TI, typename TO>
__device__ __forceinline__ void qkv_post_body(const TI* sq, const TI* sk, const TI* sv, int sqs, int sks, int svs,
                                              TO* q, TO* k, TO* v, const float* wq, const float* wk,
                                              const float* cs, const float* sn, const int* pos,
                                              int rows, int nh, int nkv, int hd, int part, float eps) {
    int S = nh + 2 * nkv;
    int row = blockIdx.x / S, s = blockIdx.x % S;
    if (row >= rows) return;
    const TI* src; TO* dst; const float* w; bool rot;
    if (s < nh) {
        dst = q + ((long long)row * nh + s) * hd; w = wq; rot = true;
        src = sq + (long long)row * sqs + (long long)s * hd;
    } else if (s < nh + nkv) {
        dst = k + ((long long)row * nkv + (s - nh)) * hd; w = wk; rot = true;
        src = sk + (long long)row * sks + (long long)(s - nh) * hd;
    } else {
        dst = v + ((long long)row * nkv + (s - nh - nkv)) * hd; w = nullptr; rot = false;
        src = sv + (long long)row * svs + (long long)(s - nh - nkv) * hd;
    }
    __shared__ float buf[512];
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) { float x = ld(src, i); ss = fmaf(x, x, ss); }
    ss = block_sum(ss);
    float inv = powf(ss / (float)hd + eps, -0.5f);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float n = __fmul_rn(ld(src, i), inv);
        buf[i] = w ? __fmul_rn(n, w[i]) : n;
    }
    __syncthreads();
    if (!rot) {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) st(dst, i, buf[i]);
        return;
    }
    int half = part / 2;
    long long trow = pos ? pos[row] : row;
    for (int idx = threadIdx.x; idx < hd / 2; idx += blockDim.x) {
        int pi = idx / half, j = idx % half;
        int b = pi * part;
        const float* c = cs + trow * hd + b;
        const float* sv_ = sn + trow * hd + b;
        float x1 = buf[b + j], x2 = buf[b + half + j];
        // The two FMA placements below are exactly what the compiler produced for the standalone
        // `x1 * c - x2 * s` / `x2 * c + x1 * s` of the original rope kernel.
        st(dst, b + j, fmaf(x1, c[j], -__fmul_rn(x2, sv_[j])));
        st(dst, b + half + j, fmaf(x2, c[half + j], __fmul_rn(x1, sv_[half + j])));
    }
}
#define QKV_POST(NAME, TI, TO) \
    extern "C" __global__ void NAME(const TI* sq, const TI* sk, const TI* sv, int sqs, int sks, int svs, \
                                    TO* q, TO* k, TO* v, const float* wq, const float* wk, \
                                    const float* cs, const float* sn, const int* pos, \
                                    int rows, int nh, int nkv, int hd, int part, float eps) { \
        qkv_post_body(sq, sk, sv, sqs, sks, svs, q, k, v, wq, wk, cs, sn, pos, rows, nh, nkv, hd, part, eps); \
    }
QKV_POST(qkv_post, float, float)

extern "C" __global__ void gelu_mul(float* g, const float* u, long long n) { ELEMWISE_IDX g[i] = gelu_tanh(g[i]) * u[i]; }
// Fused gate/up projection output gu[row, 2 ff] (gate | up): out[row, c] = gelu(gate) * up
extern "C" __global__ void gelu_mul_gu(float* out, const float* gu, long long n, int ff) {
    ELEMWISE_IDX
    long long r = i / ff; int c = (int)(i % ff);
    const float* src = gu + r * 2 * (long long)ff;
    out[i] = gelu_tanh(src[c]) * src[ff + c];
}
extern "C" __global__ void add(float* a, const float* b, long long n) { ELEMWISE_IDX a[i] += b[i]; }
extern "C" __global__ void scale(float* a, float s, long long n) { ELEMWISE_IDX a[i] *= s; }
extern "C" __global__ void clamp(float* a, float lo, float hi, long long n) { ELEMWISE_IDX a[i] = fminf(fmaxf(a[i], lo), hi); }
extern "C" __global__ void clamp_copy(float* out, const float* in, float lo, float hi, long long n) {
    ELEMWISE_IDX out[i] = fminf(fmaxf(in[i], lo), hi);
}
extern "C" __global__ void bias_add(float* y, const float* b, int rows, int cols) {
    long long n = (long long)rows * cols;
    ELEMWISE_IDX y[i] += b[i % cols];
}

// Per-layer-input gate: g[t, c] = gelu(g[t, c]) * ple[t, layer, c]
extern "C" __global__ void ple_gate(float* g, const float* ple, int rows, int p, int n_layers, int layer) {
    long long n = (long long)rows * p;
    ELEMWISE_IDX
    long long t = i / p; int c = (int)(i % p);
    g[i] = gelu_tanh(g[i]) * ple[(t * n_layers + layer) * p + c];
}

#ifdef EG_BF16
// bf16-mode variants of the kernels above (same arithmetic; see the storage-type note at the top).
extern "C" __global__ void rms_norm_fb(const float* x, bf16_t* out, const float* w, int rows, int dim, float eps, int has_w, float pre,
                                       float in_lo, float in_hi, float out_lo, float out_hi, int act) {
    int row = blockIdx.x;
    if (row >= rows) return;
    rms_norm_row(x + (long long)row * dim, out + (long long)row * dim, w, dim, eps, has_w, pre, in_lo, in_hi, out_lo, out_hi, act);
}
ADD_NORM(add_norm_fb, float, bf16_t)
ADD_NORM(add_norm_bb, bf16_t, bf16_t)
QKV_POST(qkv_post_fb, float, bf16_t)
QKV_POST(qkv_post_bb, bf16_t, bf16_t)
QKV_POST(qkv_post_bf, bf16_t, float)

// The same attention input prep with one warp per (row, head slot) instead of one block: shuffle reductions
// and no block barriers, so short heads (the vision tower's 64 channels) do not leave most of a block idle
// waiting on __syncthreads. Lane l owns channels l + 32 e (e < PER = hd / 32). RoPE partners (channel i and
// i +- part / 2) are in the same lane for 1-D RoPE (ROPE1D: part == hd, partner e +- PER / 2) and in lane
// l ^ (part / 2) for parts of at most 32 channels (the vision tower's 2-D RoPE). Same arithmetic per channel
// as qkv_post_body; only the order of the sum of squares differs.
template <typename TI, typename TO, int PER, bool ROPE1D>
__device__ __forceinline__ void qkv_post_warp_body(const TI* sq, const TI* sk, const TI* sv, int sqs, int sks, int svs,
                                                   TO* q, TO* k, TO* v, const float* wq, const float* wk,
                                                   const float* cs, const float* sn, const int* pos,
                                                   int rows, int nh, int nkv, int part, float eps) {
    constexpr int HD = PER * 32;
    const int S = nh + 2 * nkv;
    long long wi = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wi >= (long long)rows * S) return;
    const int row = (int)(wi / S), s = (int)(wi % S);
    const TI* src; TO* dst; const float* w; bool rot;
    if (s < nh) {
        dst = q + ((long long)row * nh + s) * HD; w = wq; rot = true;
        src = sq + (long long)row * sqs + (long long)s * HD;
    } else if (s < nh + nkv) {
        dst = k + ((long long)row * nkv + (s - nh)) * HD; w = wk; rot = true;
        src = sk + (long long)row * sks + (long long)(s - nh) * HD;
    } else {
        dst = v + ((long long)row * nkv + (s - nh - nkv)) * HD; w = nullptr; rot = false;
        src = sv + (long long)row * svs + (long long)(s - nh - nkv) * HD;
    }
    float x[PER];
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < PER; ++e) { x[e] = ld(src, lane + 32 * e); ss = fmaf(x[e], x[e], ss); }
    ss = warp_sum(ss);
    const float inv = powf(ss / (float)HD + eps, -0.5f);
#pragma unroll
    for (int e = 0; e < PER; ++e) {
        float n = __fmul_rn(x[e], inv);
        x[e] = w ? __fmul_rn(n, w[lane + 32 * e]) : n;
    }
    if (!rot) {
#pragma unroll
        for (int e = 0; e < PER; ++e) st(dst, lane + 32 * e, x[e]);
        return;
    }
    const long long trow = pos ? pos[row] : row;
    const float* c = cs + trow * HD;
    const float* sn_ = sn + trow * HD;
    const int half = part / 2;
#pragma unroll
    for (int e = 0; e < PER; ++e) {
        const int i = lane + 32 * e;
        bool first;
        float xp;
        if (ROPE1D) {
            first = e < PER / 2;
            xp = first ? x[(e + PER / 2) % PER] : x[(e + PER - PER / 2) % PER];
        } else {
            first = (lane & half) == 0;
            xp = __shfl_xor_sync(0xffffffff, x[e], half);
        }
        // x1 * c - x2 * s for the first half of a part, x2 * c + x1 * s for the second (see qkv_post_body).
        st(dst, i, first ? fmaf(x[e], c[i], -__fmul_rn(xp, sn_[i])) : fmaf(x[e], c[i], __fmul_rn(xp, sn_[i])));
    }
}
#define QKV_POST_WARP(NAME, TI, TO, PER, R1) \
    extern "C" __global__ void NAME(const TI* sq, const TI* sk, const TI* sv, int sqs, int sks, int svs, \
                                    TO* q, TO* k, TO* v, const float* wq, const float* wk, \
                                    const float* cs, const float* sn, const int* pos, \
                                    int rows, int nh, int nkv, int hd, int part, float eps) { \
        qkv_post_warp_body<TI, TO, PER, R1>(sq, sk, sv, sqs, sks, svs, q, k, v, wq, wk, cs, sn, pos, rows, nh, nkv, part, eps); \
    }
#define QKV_POST_WARP_IO(SUF, TI, TO) \
    QKV_POST_WARP(qkv_post_w2p_##SUF, TI, TO, 2, false) \
    QKV_POST_WARP(qkv_post_w8r_##SUF, TI, TO, 8, true) \
    QKV_POST_WARP(qkv_post_w16r_##SUF, TI, TO, 16, true)
QKV_POST_WARP_IO(fb, float, bf16_t)
QKV_POST_WARP_IO(bb, bf16_t, bf16_t)
QKV_POST_WARP_IO(bf, bf16_t, float)

// out[r, c] = gelu(g[r * gs + c]) * u[r * us + c] for c < ff: a fused gate|up output (g = gu, u = gu + ff,
// strides 2 ff) or separate gate / up outputs (strides ff).
template <typename TI>
__device__ __forceinline__ void gelu_mul_body(bf16_t* out, const TI* g, const TI* u, long long n, int ff, int gs, int us) {
    ELEMWISE_IDX
    long long r = i / ff; int c = (int)(i % ff);
    st(out, i, gelu_tanh(ld(g, r * gs + c)) * ld(u, r * us + c));
}
extern "C" __global__ void gelu_mul_fb(bf16_t* out, const float* g, const float* u, long long n, int ff, int gs, int us) { gelu_mul_body(out, g, u, n, ff, gs, us); }
extern "C" __global__ void gelu_mul_bb(bf16_t* out, const bf16_t* g, const bf16_t* u, long long n, int ff, int gs, int us) { gelu_mul_body(out, g, u, n, ff, gs, us); }

// out[t, c] = gelu(g[t, c]) * ple[t, layer, c] (out of place: the result is the next product's bf16 input).
template <typename TI>
__device__ __forceinline__ void ple_gate_body(bf16_t* out, const TI* g, const float* ple, int rows, int p, int n_layers, int layer) {
    long long n = (long long)rows * p;
    ELEMWISE_IDX
    long long t = i / p; int c = (int)(i % p);
    st(out, i, gelu_tanh(ld(g, i)) * ple[(t * n_layers + layer) * p + c]);
}
extern "C" __global__ void ple_gate_fb(bf16_t* out, const float* g, const float* ple, int rows, int p, int n_layers, int layer) { ple_gate_body(out, g, ple, rows, p, n_layers, layer); }
extern "C" __global__ void ple_gate_bb(bf16_t* out, const bf16_t* g, const float* ple, int rows, int p, int n_layers, int layer) { ple_gate_body(out, g, ple, rows, p, n_layers, layer); }

// y = bf16(clamp(x, lo, hi)) for bf16 x: the input clamp of a clipped linear whose input is already bf16.
extern "C" __global__ void clamp_bf16(const bf16_t* x, bf16_t* y, float lo, float hi, long long n) {
    ELEMWISE_IDX y[i] = f32_to_bf16_rne(fminf(fmaxf(ld(x, i), lo), hi));
}
#endif

// Row softmax over attention scores [nrows, kb] with optional bidirectional window.
// Row r belongs to query q0 + (r % rows_per_head); key column j is position k0 + j.
extern "C" __global__ void softmax_window(float* s, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    int r = blockIdx.x;
    if (r >= nrows) return;
    float* row = s + (long long)r * kb;
    int qi = q0 + (r % rows_per_head);
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < kb; j += blockDim.x) {
        int d = qi - (k0 + j);
        bool ok = window < 0 || (d <= window && -d <= window);
        if (ok) mx = fmaxf(mx, row[j] * scale);
    }
    mx = block_max(mx);
    float sum = 0.f;
    for (int j = threadIdx.x; j < kb; j += blockDim.x) {
        int d = qi - (k0 + j);
        bool ok = window < 0 || (d <= window && -d <= window);
        float e = ok ? expf(row[j] * scale - mx) : 0.f;
        row[j] = e;
        sum += e;
    }
    sum = block_sum(sum);
    float inv = 1.f / sum;
    for (int j = threadIdx.x; j < kb; j += blockDim.x) row[j] *= inv;
}

// Same softmax with the row held in registers (R values per thread): one read and one write of the
// row instead of three passes. The arithmetic and the reduction order are identical to the kernel above,
// so the results are bit-identical; it needs kb <= R * blockDim.x.
template <int R>
__device__ __forceinline__ void softmax_cached(float* s, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    int r = blockIdx.x;
    if (r >= nrows) return;
    float* row = s + (long long)r * kb;
    int qi = q0 + (r % rows_per_head);
    float x[R];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        int j = threadIdx.x + i * blockDim.x;
        x[i] = 0.f;
        if (j < kb) {
            x[i] = row[j];
            int d = qi - (k0 + j);
            bool ok = window < 0 || (d <= window && -d <= window);
            if (ok) mx = fmaxf(mx, x[i] * scale);
        }
    }
    mx = block_max(mx);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        int j = threadIdx.x + i * blockDim.x;
        if (j < kb) {
            int d = qi - (k0 + j);
            bool ok = window < 0 || (d <= window && -d <= window);
            float e = ok ? expf(x[i] * scale - mx) : 0.f;
            x[i] = e;
            sum += e;
        }
    }
    sum = block_sum(sum);
    float inv = 1.f / sum;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        int j = threadIdx.x + i * blockDim.x;
        if (j < kb) row[j] = x[i] * inv;
    }
}
extern "C" __global__ void softmax_window_r8(float* s, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    softmax_cached<8>(s, nrows, rows_per_head, kb, q0, k0, window, scale);
}
extern "C" __global__ void softmax_window_r16(float* s, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    softmax_cached<16>(s, nrows, rows_per_head, kb, q0, k0, window, scale);
}

// bf16-output softmax for the bf16 attention path: the same arithmetic as above, but the probabilities are
// written to `p` as bf16 (the next GEMM's input) instead of back into `s` as float32, saving a full float32
// write + re-read of the score matrix.
template <int R>
__device__ __forceinline__ void softmax_cached_bf16(const float* s, unsigned short* p, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    int r = blockIdx.x;
    if (r >= nrows) return;
    const float* row = s + (long long)r * kb;
    unsigned short* out = p + (long long)r * kb;
    int qi = q0 + (r % rows_per_head);
    float x[R];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        int j = threadIdx.x + i * blockDim.x;
        x[i] = 0.f;
        if (j < kb) {
            x[i] = row[j];
            int d = qi - (k0 + j);
            bool ok = window < 0 || (d <= window && -d <= window);
            if (ok) mx = fmaxf(mx, x[i] * scale);
        }
    }
    mx = block_max(mx);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        int j = threadIdx.x + i * blockDim.x;
        if (j < kb) {
            int d = qi - (k0 + j);
            bool ok = window < 0 || (d <= window && -d <= window);
            float e = ok ? expf(x[i] * scale - mx) : 0.f;
            x[i] = e;
            sum += e;
        }
    }
    sum = block_sum(sum);
    float inv = 1.f / sum;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        int j = threadIdx.x + i * blockDim.x;
        if (j < kb) out[j] = f32_to_bf16_rne(x[i] * inv);
    }
}
extern "C" __global__ void softmax_window_r8_bf16(const float* s, unsigned short* p, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    softmax_cached_bf16<8>(s, p, nrows, rows_per_head, kb, q0, k0, window, scale);
}
extern "C" __global__ void softmax_window_r16_bf16(const float* s, unsigned short* p, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    softmax_cached_bf16<16>(s, p, nrows, rows_per_head, kb, q0, k0, window, scale);
}
// Rows too long for registers: the three-pass kernel, with the exponentials kept in `s` and the final
// normalised values written to `p` as bf16.
extern "C" __global__ void softmax_window_bf16(float* s, unsigned short* p, int nrows, int rows_per_head, int kb, int q0, int k0, int window, float scale) {
    int r = blockIdx.x;
    if (r >= nrows) return;
    float* row = s + (long long)r * kb;
    unsigned short* out = p + (long long)r * kb;
    int qi = q0 + (r % rows_per_head);
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < kb; j += blockDim.x) {
        int d = qi - (k0 + j);
        bool ok = window < 0 || (d <= window && -d <= window);
        if (ok) mx = fmaxf(mx, row[j] * scale);
    }
    mx = block_max(mx);
    float sum = 0.f;
    for (int j = threadIdx.x; j < kb; j += blockDim.x) {
        int d = qi - (k0 + j);
        bool ok = window < 0 || (d <= window && -d <= window);
        float e = ok ? expf(row[j] * scale - mx) : 0.f;
        row[j] = e;
        sum += e;
    }
    sum = block_sum(sum);
    float inv = 1.f / sum;
    for (int j = threadIdx.x; j < kb; j += blockDim.x) out[j] = f32_to_bf16_rne(row[j] * inv);
}

// Flash-style attention for head_dim 64 (the vision tower): the score matrix never touches memory.
// One block = 64 query rows of one head, 256 threads in a 16x16 grid; each thread owns a 4x4 tile of
// the 64x64 score tile and of the 64x64 output tile (FP32 FMAs, online softmax across 64-key tiles).
// Q and K are staged transposed ([d][row]) so the inner loops read float4s; V stays [key][d].
// q/k/v/out are [*, heads*64] row-major with row strides qs / kvs; the sequence occupies rows
// row0 .. row0+seq-1 (bidirectional, optionally limited to |i - j| <= window).
extern "C" __global__ void __launch_bounds__(256, 2)
flash_attn_d64(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out,
               int row0, int seq, int nh, int nkv, int window, float scale, int qs, int kvs) {
    const int D = 64, BQ = 64, BK = 64;
    __shared__ __align__(16) float Qs[D][BQ];
    __shared__ __align__(16) float Ks[D][BK];   // later reused as P[query][key]
    __shared__ __align__(16) float Vs[BK][D];
    const int h = blockIdx.y, g = h / (nh / nkv);
    const int q0 = blockIdx.x * BQ;
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;

    // Q^T tile: thread -> (query = tid % 64, d4 = tid / 64 + 4 i); a warp covers 32 queries, so the
    // four transposing stores per float4 hit 32 different banks.
    {
        int r = tid & 63;
        for (int i = 0; i < 4; ++i) {
            int d4 = (tid >> 6) + 4 * i;
            float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q0 + r < seq) x = *reinterpret_cast<const float4*>(q + (long long)(row0 + q0 + r) * qs + h * D + d4 * 4);
            Qs[d4 * 4 + 0][r] = x.x; Qs[d4 * 4 + 1][r] = x.y; Qs[d4 * 4 + 2][r] = x.z; Qs[d4 * 4 + 3][r] = x.w;
        }
    }
    float o[4][4], m[4], l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY; l[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
    }
    int kt0 = 0, kt1 = (seq + BK - 1) / BK;
    if (window >= 0) {
        kt0 = max(0, (q0 - window) / BK);
        kt1 = min(kt1, (q0 + BQ - 1 + window) / BK + 1);
    }
    for (int kt = kt0; kt < kt1; ++kt) {
        const int k0 = kt * BK;
        __syncthreads();   // previous tile's P / V reads are done
        {
            int r = tid & 63;
            for (int i = 0; i < 4; ++i) {
                int d4 = (tid >> 6) + 4 * i;
                float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
                if (k0 + r < seq) x = *reinterpret_cast<const float4*>(k + (long long)(row0 + k0 + r) * kvs + g * D + d4 * 4);
                Ks[d4 * 4 + 0][r] = x.x; Ks[d4 * 4 + 1][r] = x.y; Ks[d4 * 4 + 2][r] = x.z; Ks[d4 * 4 + 3][r] = x.w;
            }
            int d4 = tid & 15;
            for (int i = 0; i < 4; ++i) {
                int key = (tid >> 4) + 16 * i;
                float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
                if (k0 + key < seq) x = *reinterpret_cast<const float4*>(v + (long long)(row0 + k0 + key) * kvs + g * D + d4 * 4);
                *reinterpret_cast<float4*>(&Vs[key][d4 * 4]) = x;
            }
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
#pragma unroll 8
        for (int d = 0; d < D; ++d) {
            float4 a = *reinterpret_cast<const float4*>(&Qs[d][ty * 4]);
            float4 b = *reinterpret_cast<const float4*>(&Ks[d][tx * 4]);
            float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) s[i][j] = fmaf(av[i], bv[j], s[i][j]);
        }
        // Masking + online softmax; a row's 64 scores live in the 16 lanes sharing `ty`.
        float p[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qi = q0 + ty * 4 + i;
            float mt = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kj = k0 + tx * 4 + j;
                bool ok = qi < seq && kj < seq && (window < 0 || (qi - kj <= window && kj - qi <= window));
                s[i][j] = ok ? s[i][j] * scale : -INFINITY;
                mt = fmaxf(mt, s[i][j]);
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
            const float m_new = fmaxf(m[i], mt);
            const float ms = (m_new == -INFINITY) ? 0.f : m_new;
            const float corr = expf(m[i] - ms);
            float ps = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[i][j] = (s[i][j] == -INFINITY) ? 0.f : expf(s[i][j] - ms);
                ps += p[i][j];
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffff, ps, off);
            l[i] = l[i] * corr + ps;
            m[i] = m_new;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[i][j] *= corr;
        }
        __syncthreads();   // everyone is done reading K^T: reuse it for P
        float* Ps = &Ks[0][0];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            *reinterpret_cast<float4*>(&Ps[(ty * 4 + i) * BK + tx * 4]) = make_float4(p[i][0], p[i][1], p[i][2], p[i][3]);
        __syncthreads();
#pragma unroll 2
        for (int k4 = 0; k4 < BK / 4; ++k4) {
            float4 pr[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) pr[i] = *reinterpret_cast<const float4*>(&Ps[(ty * 4 + i) * BK + k4 * 4]);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                float4 vv = *reinterpret_cast<const float4*>(&Vs[k4 * 4 + kk][tx * 4]);
                float vb[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float pk = kk == 0 ? pr[i].x : kk == 1 ? pr[i].y : kk == 2 ? pr[i].z : pr[i].w;
#pragma unroll
                    for (int j = 0; j < 4; ++j) o[i][j] = fmaf(pk, vb[j], o[i][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int qi = q0 + ty * 4 + i;
        if (qi < seq) {
            const float inv = 1.f / l[i];
            *reinterpret_cast<float4*>(out + (long long)(row0 + qi) * qs + h * D + tx * 4) =
                make_float4(o[i][0] * inv, o[i][1] * inv, o[i][2] * inv, o[i][3] * inv);
        }
    }
}

// Fused attention for many short packed sequences: one warp per (row, group of HW query heads), where the
// heads of a group share one K/V head. With HW = G (all heads that share a K/V head) each K/V row is read
// once per group instead of once per head; HW = 1 trades that sharing for G times more warps, which is
// what a handful of rows needs to fill the GPU.
// Row t belongs to the sequence [seg_start[t], seg_start[t] + seg_len[t]); keys are restricted to
// that sequence (block-diagonal) and to |i - j| <= window when window >= 0. Uses an online
// softmax, so any sequence length works; the host only routes short sequences here.
// Each lane owns PER = head_dim / 32 channels of every head; PER and HW are compile-time so the
// per-head state lives in registers. The arithmetic per head (dot product order, FMA placement,
// online-softmax update) is unchanged from the one-warp-per-head version.
template <int PER, int G, int HW, typename TI, typename TO>
__device__ __forceinline__ void attn_packed_body(const TI* q, const TI* k, const TI* v, TO* out,
                                                 const int* seg_start, const int* seg_len,
                                                 int rows, int nkv, int window) {
    constexpr int HD = PER * 32;
    constexpr int UNITS = G / HW;                 // warps per (row, K/V head)
    long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= (long long)rows * nkv * UNITS) return;
    int unit = (int)(w % (nkv * UNITS)); int t = (int)(w / (nkv * UNITS));
    int gi = unit / UNITS;                        // K/V head
    int head0 = unit * HW;                        // first query head of this warp
    int nh = nkv * G;
    int qs = nh * HD, kvs = nkv * HD;
    int s0 = seg_start[t], len = seg_len[t];
    int i = t - s0;
    int lo = 0, hi = len;
    if (window >= 0) { lo = max(0, i - window); hi = min(len, i + window + 1); }
    const TI* qv = q + (long long)t * qs + (long long)head0 * HD;
    float qr[HW][PER], acc[HW][PER], m[HW], l[HW];
#pragma unroll
    for (int g = 0; g < HW; ++g) {
#pragma unroll
        for (int e = 0; e < PER; ++e) { acc[g][e] = 0.f; qr[g][e] = ld(qv, g * HD + lane + 32 * e); }
        m[g] = -INFINITY; l[g] = 0.f;
    }
    for (int j = lo; j < hi; ++j) {
        const TI* kk = k + (long long)(s0 + j) * kvs + (long long)gi * HD + lane;
        const TI* vv = v + (long long)(s0 + j) * kvs + (long long)gi * HD + lane;
        float kr[PER], vr[PER];
#pragma unroll
        for (int e = 0; e < PER; ++e) { kr[e] = ld(kk, 32 * e); vr[e] = ld(vv, 32 * e); }
#pragma unroll
        for (int g = 0; g < HW; ++g) {
            float s = 0.f;
#pragma unroll
            for (int e = 0; e < PER; ++e) s = fmaf(qr[g][e], kr[e], s);
            s = warp_sum(s);
            float m_new = fmaxf(m[g], s);
            float corr = expf(m[g] - m_new);
            float p = expf(s - m_new);
            l[g] = fmaf(l[g], corr, p);
#pragma unroll
            for (int e = 0; e < PER; ++e) acc[g][e] = fmaf(acc[g][e], corr, __fmul_rn(p, vr[e]));
            m[g] = m_new;
        }
    }
    TO* o = out + (long long)t * qs + (long long)head0 * HD;
#pragma unroll
    for (int g = 0; g < HW; ++g) {
        float inv = 1.f / l[g];
#pragma unroll
        for (int e = 0; e < PER; ++e) st(o, g * HD + lane + 32 * e, acc[g][e] * inv);
    }
}

#define ATTN_PACKED(SUF, TI, TO, PER, G, HW) \
    extern "C" __global__ void attn_packed##SUF##_##PER##_##G##_##HW(const TI* q, const TI* k, const TI* v, TO* out, \
                                                              const int* seg_start, const int* seg_len, int rows, int nkv, int window) { \
        attn_packed_body<PER, G, HW, TI, TO>(q, k, v, out, seg_start, seg_len, rows, nkv, window); \
    }
#define ATTN_PACKED_PER(SUF, TI, TO, PER) \
    ATTN_PACKED(SUF, TI, TO, PER, 1, 1) ATTN_PACKED(SUF, TI, TO, PER, 2, 2) ATTN_PACKED(SUF, TI, TO, PER, 2, 1) \
    ATTN_PACKED(SUF, TI, TO, PER, 4, 4) ATTN_PACKED(SUF, TI, TO, PER, 4, 1)
#define ATTN_PACKED_ALL(SUF, TI, TO) \
    ATTN_PACKED_PER(SUF, TI, TO, 2) ATTN_PACKED_PER(SUF, TI, TO, 4) ATTN_PACKED_PER(SUF, TI, TO, 8) ATTN_PACKED_PER(SUF, TI, TO, 16)
// The float32 module writes float32; the bf16 module writes the bf16 attention output (the next product's input).
#ifndef EG_BF16
ATTN_PACKED_ALL(, float, float)
#else
ATTN_PACKED_ALL(_fb, float, bf16_t)
#endif

// ---------------------------------------------------------------------------------------------
// Audio conformer
// ---------------------------------------------------------------------------------------------

// x = clamp(silu(clamp(x, lo1, hi1)), lo2, hi2)   (the output clamp of one clipped linear, SiLU, and the
// input clamp of the next)
extern "C" __global__ void act_chain(float* x, long long n, float lo1, float hi1, float lo2, float hi2) {
    ELEMWISE_IDX
    x[i] = fminf(fmaxf(silu_f(fminf(fmaxf(x[i], lo1), hi1)), lo2), hi2);
}

// Fused q|k|v projection output qkv[T, 3h] -> separate q, k, v [T, h] with each part's output clamp and the
// attention scalings: q *= qscale * dim_scale[c % hd], k *= kscale, v unscaled. All three share one clamp range
// here because the host only fuses the projections when their clip bounds agree.
extern "C" __global__ void audio_qkv_post(const float* qkv, float* q, float* k, float* v, const float* dim_scale,
                                          long long n, int h, int hd, float qscale, float kscale,
                                          float qlo, float qhi, float klo, float khi, float vlo, float vhi) {
    ELEMWISE_IDX
    long long t = i / h; int c = (int)(i % h);
    const float* src = qkv + t * 3 * h;
    q[i] = __fmul_rn(fminf(fmaxf(src[c], qlo), qhi), __fmul_rn(qscale, dim_scale[c % hd]));
    k[i] = __fmul_rn(fminf(fmaxf(src[h + c], klo), khi), kscale);
    v[i] = fminf(fmaxf(src[2 * h + c], vlo), vhi);
}

// LayerNorm over `dim` channels (no bias, biased variance, statistics in double like the host reference)
// of one row per warp, in place, followed by ReLU. Each lane owns channels lane, lane + 32, ... (dim <= 128).
__device__ __forceinline__ void ln_relu_row(float (&v)[4], int dim, int lane, const float* w, float eps, float* dst) {
    double s = 0.0;
#pragma unroll
    for (int j = 0; j < 4; ++j) if (lane + 32 * j < dim) s += (double)v[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    double mean = s / (double)dim;
    double q = 0.0;
#pragma unroll
    for (int j = 0; j < 4; ++j) if (lane + 32 * j < dim) { double d = (double)v[j] - mean; q += d * d; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
    float var = (float)(q / (double)dim);
    float inv = 1.f / sqrtf(var + eps);
    float mf = (float)mean;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int c = lane + 32 * j;
        if (c < dim) dst[c] = fmaxf(__fmul_rn(__fmul_rn(__fsub_rn(v[j], mf), inv), w[c]), 0.f);
    }
}

// Audio subsampler layer 0: 3x3 stride-2 pad-1 convolution of the [frames, n_mels] log-mel image into
// c0 channels (weights [c0, 9]), then LayerNorm over the channels and ReLU. One warp per output position;
// out is [t1 * w1, c0] with the channels of a position contiguous.
extern "C" __global__ void audio_conv0(const float* mel, const float* wt, const float* nw, float* out,
                                       int frames, int n_mels, int t1, int w1, int c0, float eps) {
    long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= (long long)t1 * w1) return;
    int ty = (int)(warp / w1), fx = (int)(warp % w1);
    float col[9];
#pragma unroll
    for (int kh = 0; kh < 3; ++kh)
#pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            int ti = 2 * ty + kh - 1, fi = 2 * fx + kw - 1;
            col[kh * 3 + kw] = (ti >= 0 && ti < frames && fi >= 0 && fi < n_mels) ? mel[(long long)ti * n_mels + fi] : 0.f;
        }
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int c = lane + 32 * j;
        float a = 0.f;
        if (c < c0) {
#pragma unroll
            for (int tap = 0; tap < 9; ++tap) a = fmaf(col[tap], wt[c * 9 + tap], a);
        }
        v[j] = a;
    }
    ln_relu_row(v, c0, lane, nw, eps, out + warp * c0);
}

// Same LayerNorm + ReLU over rows of `dim` (<= 128) channels, in place.
extern "C" __global__ void ln_relu(float* x, const float* nw, long long rows, int dim, float eps) {
    long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    float* r = x + warp * dim;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = (lane + 32 * j < dim) ? r[lane + 32 * j] : 0.f;
    ln_relu_row(v, dim, lane, nw, eps, r);
}

// im2col for the subsampler's second convolution: col[pos, (kh * 3 + kw) * c0 + ci] =
// y0[(2 ty + kh - 1) * w1 + (2 fx + kw - 1), ci] (zero outside), pos = ty * w2 + fx.
extern "C" __global__ void audio_im2col1(const float* y0, float* col, int t1, int w1, int c0, int t2, int w2) {
    long long n = (long long)t2 * w2 * 9 * c0;
    ELEMWISE_IDX
    long long k1 = 9LL * c0;
    long long pos = i / k1; int rem = (int)(i % k1);
    int tap = rem / c0, ci = rem % c0;
    int ty = (int)(pos / w2), fx = (int)(pos % w2);
    int ti = 2 * ty + tap / 3 - 1, fi = 2 * fx + tap % 3 - 1;
    col[i] = (ti >= 0 && ti < t1 && fi >= 0 && fi < w1) ? y0[((long long)ti * w1 + fi) * c0 + ci] : 0.f;
}

// q[t, h, d] *= qscale * dim_scale[d]
extern "C" __global__ void audio_qscale(float* q, const float* dim_scale, float qscale, int rows, int nh, int hd) {
    long long n = (long long)rows * nh * hd;
    ELEMWISE_IDX q[i] *= qscale * dim_scale[i % hd];
}

// qrel[t, h, p] = q[t, h, :] . rel[p, h, :]   (one warp per output)
extern "C" __global__ void audio_qrel(const float* q, const float* rel, float* qrel, int T, int nh, int hd, int npos) {
    long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= (long long)T * nh * npos) return;
    int p = (int)(w % npos); long long th = w / npos;
    int h = (int)(th % nh); long long t = th / nh;
    int H = nh * hd;
    const float* qv = q + t * H + (long long)h * hd;
    const float* rv = rel + (long long)p * H + (long long)h * hd;
    float acc = 0.f;
    for (int d = lane; d < hd; d += 32) acc += qv[d] * rv[d];
    acc = warp_sum(acc);
    if (lane == 0) qrel[w] = acc;
}

// Chunked local attention with relative-position bias and logit soft-cap (one warp per (t, h)).
extern "C" __global__ void audio_attn(const float* q, const float* k, const float* v, const float* qrel, float* out,
                                      int T, int nh, int hd, int chunk, int past, int ctx, int npos, float cap, float invalid,
                                      float out_lo, float out_hi) {
    long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= (long long)T * nh) return;
    int h = (int)(w % nh); int t = (int)(w / nh);
    int H = nh * hd;
    int b = t / chunk, i = t % chunk;
    const float* qv = q + (long long)t * H + (long long)h * hd;
    float logits[64];
    for (int j = 0; j < ctx; ++j) {
        int kv = b * chunk + j - past;
        int dist = t - kv;
        float lg = invalid;
        if (kv >= 0 && kv < T && dist >= 0 && dist < past) {
            const float* kk = k + (long long)kv * H + (long long)h * hd;
            float acc = 0.f;
            for (int d = lane; d < hd; d += 32) acc += qv[d] * kk[d];
            acc = warp_sum(acc);
            int f = i * ctx + j;
            int r = f / (ctx + 1), c = f % (ctx + 1);
            float bd = 0.f;
            int qr = b * chunk + r;
            if (c < npos && qr < T) bd = qrel[((long long)qr * nh + h) * npos + c];
            lg = tanhf((acc + bd) / cap) * cap;
        }
        logits[j] = lg;
    }
    float mx = -INFINITY;
    for (int j = 0; j < ctx; ++j) mx = fmaxf(mx, logits[j]);
    float sum = 0.f;
    for (int j = 0; j < ctx; ++j) { logits[j] = expf(logits[j] - mx); sum += logits[j]; }
    float inv = 1.f / sum;
    for (int d = lane; d < hd; d += 32) {
        float acc = 0.f;
        for (int j = 0; j < ctx; ++j) {
            int kv = b * chunk + j - past;
            float p = logits[j] * inv;
            if (p == 0.f || kv < 0 || kv >= T) continue;
            acc += p * v[(long long)kv * H + (long long)h * hd + d];
        }
        out[(long long)t * H + (long long)h * hd + d] = fminf(fmaxf(acc, out_lo), out_hi);
    }
}

// GLU: out[t, c] = in[t, c] * sigmoid(in[t, h + c])
extern "C" __global__ void glu(float* out, const float* in, int rows, int h, float lo, float hi) {
    long long n = (long long)rows * h;
    ELEMWISE_IDX
    long long t = i / h; int c = (int)(i % h);
    const float* src = in + t * 2 * h;
    out[i] = fminf(fmaxf(src[c], lo), hi) / (1.f + expf(-fminf(fmaxf(src[h + c], lo), hi)));
}

// Causal depthwise conv1d (left pad = K-1) followed by clamp to +-gc.
extern "C" __global__ void dwconv_causal(float* out, const float* in, const float* w, int T, int C, int K, float gc) {
    long long n = (long long)T * C;
    ELEMWISE_IDX
    int t = (int)(i / C), c = (int)(i % C);
    float acc = 0.f;
    for (int kk = 0; kk < K; ++kk) {
        int src = t - (K - 1) + kk;
        if (src >= 0) acc += w[c * K + kk] * in[(long long)src * C + c];
    }
    out[i] = fminf(fmaxf(acc, -gc), gc);
}
