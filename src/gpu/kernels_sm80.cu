// Kernels that need compute capability 8.0+ (bf16 tensor-core `mma`). Appended to kernels.cu (whose
// helpers they use) and compiled by NVRTC for compute_80 only when the model runs in bf16 mode, so the
// float32 module keeps working on older GPUs.

// Two floats -> one register of two bf16 (the lower column in the low half, as mma fragments expect).
__device__ __forceinline__ unsigned int pack_bf16(float lo, float hi) {
    return (unsigned int)f32_to_bf16_rne(lo) | ((unsigned int)f32_to_bf16_rne(hi) << 16);
}

// d = a . b + d for one m16n8k16 tile: bf16 inputs, f32 accumulation.
__device__ __forceinline__ void mma_bf16(float* d, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two / four consecutive elements of a float32 or bf16 tensor as packed bf16 pairs, and a pair store.
__device__ __forceinline__ unsigned int load2_bf16(const float* p) { float2 x = *reinterpret_cast<const float2*>(p); return pack_bf16(x.x, x.y); }
__device__ __forceinline__ unsigned int load2_bf16(const bf16_t* p) { return *reinterpret_cast<const unsigned int*>(p); }
__device__ __forceinline__ uint2 load4_bf16(const float* p) {
    float4 x = *reinterpret_cast<const float4*>(p);
    return make_uint2(pack_bf16(x.x, x.y), pack_bf16(x.z, x.w));
}
__device__ __forceinline__ uint2 load4_bf16(const bf16_t* p) { return *reinterpret_cast<const uint2*>(p); }
__device__ __forceinline__ void store2(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }
__device__ __forceinline__ void store2(bf16_t* p, float a, float b) { *reinterpret_cast<unsigned int*>(p) = pack_bf16(a, b); }

// Flash attention for head_dim 64 on bf16 tensor cores (the vision tower in bf16 mode).
// Same interface and semantics as `flash_attn_d64`: q/k/v/out are [*, heads*64] with row strides qs / kvs
// (float32 or bf16 storage, see TI / TO), the sequence occupies rows row0 .. row0+seq-1, bidirectional,
// optionally |i - j| <= window. One block = 64 query rows of one head = 4 warps x 16 rows. Q/K/V and the
// probabilities are rounded to bf16 for the two products; scores, softmax state and the output accumulate in
// float32.
// Fragment layouts follow the PTX ISA for mma.m16n8k16 (g = lane / 4, t = lane % 4):
//   A (16x16):  reg0 = row g,   cols 2t..2t+1   reg1 = row g+8, cols 2t..2t+1
//               reg2 = row g,   cols 2t+8..+9   reg3 = row g+8, cols 2t+8..+9
//   B (16x8):   reg0 = rows 2t..2t+1, col g     reg1 = rows 2t+8..2t+9, col g
//   C (16x8):   c0,c1 = row g, cols 2t..2t+1    c2,c3 = row g+8, cols 2t..2t+1
template <typename TI, typename TO>
__device__ __forceinline__ void flash_d64_body(const TI* __restrict__ q, const TI* __restrict__ k, const TI* __restrict__ v, TO* __restrict__ out,
                                               int row0, int seq, int nh, int nkv, int window, float scale, int qs, int kvs) {
    const int D = 64, BQ = 64, BK = 64, LD = 72;   // LD: padded smem row (bf16), conflict-free fragment reads
    __shared__ __align__(16) unsigned short Ks[BK * LD];   // [key][dim]
    __shared__ __align__(16) unsigned short Vt[D * LD];    // [dim][key] (transposed: B fragments of P.V)
    const int h = blockIdx.y, kvh = h / (nh / nkv);
    const int q0 = blockIdx.x * BQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int qr0 = q0 + warp * 16 + g, qr1 = qr0 + 8;   // the two query rows this thread holds

    // Q fragments for the 4 k-steps of head_dim 64, straight from global memory (rows past `seq` are zero).
    unsigned int qa[4][4];
    {
        const TI* r0p = q + (long long)(row0 + qr0) * qs + h * D;
        const TI* r1p = q + (long long)(row0 + qr1) * qs + h * D;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int c = 16 * s + 2 * t;
            qa[s][0] = qr0 < seq ? load2_bf16(r0p + c) : 0u;
            qa[s][1] = qr1 < seq ? load2_bf16(r1p + c) : 0u;
            qa[s][2] = qr0 < seq ? load2_bf16(r0p + c + 8) : 0u;
            qa[s][3] = qr1 < seq ? load2_bf16(r1p + c + 8) : 0u;
        }
    }
    float o[8][4];
#pragma unroll
    for (int u = 0; u < 8; ++u) o[u][0] = o[u][1] = o[u][2] = o[u][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    int kt0 = 0, kt1 = (seq + BK - 1) / BK;
    if (window >= 0) {
        kt0 = max(0, (q0 - window) / BK);
        kt1 = min(kt1, (q0 + BQ - 1 + window) / BK + 1);
    }
    for (int kt = kt0; kt < kt1; ++kt) {
        const int k0 = kt * BK;
        __syncthreads();   // the previous tile's K / V reads are done
        // Stage K as [key][dim] and V transposed as [dim][key], rounding to bf16 (keys past `seq` are zero).
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int c = tid + 128 * it;          // 1024 chunks of 4 elements per 64x64 tile
            const int key = c >> 4, d = (c & 15) * 4;
            uint2 kx = make_uint2(0u, 0u), vx = kx;
            if (k0 + key < seq) {
                const long long off = (long long)(row0 + k0 + key) * kvs + kvh * D + d;
                kx = load4_bf16(k + off);
                vx = load4_bf16(v + off);
            }
            *reinterpret_cast<uint2*>(&Ks[key * LD + d]) = kx;
            Vt[(d + 0) * LD + key] = (unsigned short)(vx.x & 0xffffu);
            Vt[(d + 1) * LD + key] = (unsigned short)(vx.x >> 16);
            Vt[(d + 2) * LD + key] = (unsigned short)(vx.y & 0xffffu);
            Vt[(d + 3) * LD + key] = (unsigned short)(vx.y >> 16);
        }
        __syncthreads();

        // S = Q K^T for this warp's 16 rows x 64 keys (8 n-tiles of 8 keys).
        float sacc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
            const unsigned short* kr = &Ks[(8 * j + g) * LD];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const unsigned int b0 = *reinterpret_cast<const unsigned int*>(kr + 16 * s + 2 * t);
                const unsigned int b1 = *reinterpret_cast<const unsigned int*>(kr + 16 * s + 2 * t + 8);
                mma_bf16(sacc[j], qa[s], b0, b1);
            }
        }

        // Mask, scale and the online softmax update; rows g and g+8 are shared by the 4 lanes of a group.
        float mt0 = -INFINITY, mt1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int qi = e < 2 ? qr0 : qr1;
                const int kj = k0 + 8 * j + 2 * t + (e & 1);
                const bool ok = qi < seq && kj < seq && (window < 0 || (qi - kj <= window && kj - qi <= window));
                sacc[j][e] = ok ? sacc[j][e] * scale : -INFINITY;
            }
            mt0 = fmaxf(mt0, fmaxf(sacc[j][0], sacc[j][1]));
            mt1 = fmaxf(mt1, fmaxf(sacc[j][2], sacc[j][3]));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffff, mt0, off));
            mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffff, mt1, off));
        }
        const float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
        const float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        const float c0 = expf(m0 - ms0), c1 = expf(m1 - ms1);
        float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sacc[j][0] = sacc[j][0] == -INFINITY ? 0.f : expf(sacc[j][0] - ms0);
            sacc[j][1] = sacc[j][1] == -INFINITY ? 0.f : expf(sacc[j][1] - ms0);
            sacc[j][2] = sacc[j][2] == -INFINITY ? 0.f : expf(sacc[j][2] - ms1);
            sacc[j][3] = sacc[j][3] == -INFINITY ? 0.f : expf(sacc[j][3] - ms1);
            ps0 += sacc[j][0] + sacc[j][1];
            ps1 += sacc[j][2] + sacc[j][3];
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            ps0 += __shfl_xor_sync(0xffffffff, ps0, off);
            ps1 += __shfl_xor_sync(0xffffffff, ps1, off);
        }
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            o[u][0] *= c0; o[u][1] *= c0;
            o[u][2] *= c1; o[u][3] *= c1;
        }

        // O += P V: the probability accumulators of n-tiles 2s and 2s+1 are exactly the A fragment of k-step s.
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            unsigned int pa[4];
            pa[0] = pack_bf16(sacc[2 * s][0], sacc[2 * s][1]);
            pa[1] = pack_bf16(sacc[2 * s][2], sacc[2 * s][3]);
            pa[2] = pack_bf16(sacc[2 * s + 1][0], sacc[2 * s + 1][1]);
            pa[3] = pack_bf16(sacc[2 * s + 1][2], sacc[2 * s + 1][3]);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const unsigned short* vr = &Vt[(8 * u + g) * LD + 16 * s + 2 * t];
                const unsigned int b0 = *reinterpret_cast<const unsigned int*>(vr);
                const unsigned int b1 = *reinterpret_cast<const unsigned int*>(vr + 8);
                mma_bf16(o[u], pa, b0, b1);
            }
        }
    }

    // Normalise and store (rows past `seq` are dropped).
    const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int c = h * D + 8 * u + 2 * t;
        if (qr0 < seq) store2(out + (long long)(row0 + qr0) * qs + c, o[u][0] * i0, o[u][1] * i0);
        if (qr1 < seq) store2(out + (long long)(row0 + qr1) * qs + c, o[u][2] * i1, o[u][3] * i1);
    }
}

// float32 q/k/v/out (callers with float32 activations; bf16 ones use flash_attn_d64_bb below).
extern "C" __global__ void __launch_bounds__(128)
flash_attn_d64_bf16(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out,
                    int row0, int seq, int nh, int nkv, int window, float scale, int qs, int kvs) {
    flash_d64_body(q, k, v, out, row0, seq, nh, nkv, window, scale, qs, kvs);
}


// ---------------------------------------------------------------------------------------------
// Flash attention for wide heads (the text backbone's 256-channel heads) on bf16 tensor cores.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

// 16-byte global -> shared copy that bypasses registers; `valid == false` writes zeros instead.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// Four 8x8 bf16 matrices from shared memory (lanes 8i..8i+7 give the row addresses of matrix i); `_t` transposes.
__device__ __forceinline__ void ldsm_x4(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

// Element offset of 16-byte chunk `c` of row `r` in a [rows][D] bf16 tile whose chunks are XOR-swizzled with the
// row (chunk c ^ (r & 7)): the 8 rows an ldmatrix reads at one logical chunk land in 8 different bank groups.
template <int D>
__device__ __forceinline__ int swz(int r, int c) { return r * D + ((c ^ (r & 7)) << 3); }

// Same contract as flash_attn_d64_bf16 for head_dim D (a multiple of 64: 64 for the vision tower, 256 for the text
// backbone's sliding-window layers), with bf16 q/k/v/out [*, heads*D] with row strides qs / kvs, the sequence at rows row0 .. row0+seq-1, bidirectional, optionally
// |i - j| <= window (whole key tiles outside the window are skipped). One block = 64 query rows of one head =
// 4 warps x 16 rows; keys go in tiles of BK (a multiple of 16). Q stays in shared memory for the whole block; each
// key tile is staged as K, used for S = Q K^T, then overwritten by V for O += P V (48 KB in all for D = 256 and
// BK = 32, so two blocks share an SM and hide each other's copies). Scores, softmax state and O accumulate in
// float32; the probabilities are rounded to bf16 for the second product (as in flash_attn_d64_bf16).
//
// Tile-list mode (`tiles != nullptr`, many short sequences packed back to back in one launch): grid.x indexes
// 64-query tiles instead of covering one sequence, and tile t is the int triple {row0, len, q0} at tiles[3t ..]: the
// sequence occupies rows row0 .. row0+len-1 of q / k / v / out and this block's queries start at q0 within it
// (row0 / seq of the parameter list are then ignored). Keys are the tile's own sequence only (block-diagonal), within
// the window when given. A tile of len 0 does nothing, so a launch with a fixed grid (a recorded CUDA graph) can
// carry padding tiles. Everything else is the single-sequence code, so with `tiles == nullptr` the arithmetic is unchanged.
template <int D, int BK>
__device__ __forceinline__ void flash_tc_body(const bf16_t* __restrict__ q, const bf16_t* __restrict__ k, const bf16_t* __restrict__ v,
                                                bf16_t* __restrict__ out, const int* __restrict__ tiles, int row0, int seq, int nh,
                                                int nkv, int window, float scale, int qs, int kvs) {
    // CH: 16-byte chunks per row; NT: 8-channel output tiles; NJ: 8-key score tiles; NS: 16-key steps of P V.
    constexpr int BQ = 64, CH = D / 8, NT = D / 8, NJ = BK / 8, NS = BK / 16;
    __shared__ __align__(16) bf16_t sQ[BQ * D];
    __shared__ __align__(16) bf16_t sKV[BK * D];
    const int h = blockIdx.y, kvh = h / (nh / nkv);
    int q0 = blockIdx.x * BQ;
    if (tiles != nullptr) {
        const int* tile = tiles + 3 * blockIdx.x;
        row0 = tile[0];
        seq = tile[1];
        q0 = tile[2];
        if (seq <= 0) return;   // a padding tile (uniform across the block, and before any barrier)
    }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int qr0 = q0 + warp * 16 + g, qr1 = qr0 + 8;   // the two query rows this thread accumulates

    // Q tile (rows past `seq` are zero).
#pragma unroll 4
    for (int i = tid; i < BQ * CH; i += 128) {
        const int r = i / CH, c = i % CH, qi = q0 + r;
        const bool ok = qi < seq;
        cp_async16(sQ + swz<D>(r, c), q + (long long)(row0 + (ok ? qi : 0)) * qs + h * D + c * 8, ok);
    }
    cp_async_commit();

    float o[NT][4];
#pragma unroll
    for (int u = 0; u < NT; ++u) o[u][0] = o[u][1] = o[u][2] = o[u][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    int k_lo = 0, k_hi = seq;
    if (window >= 0) {
        k_lo = max(0, q0 - window);
        k_hi = min(seq, q0 + BQ - 1 + window + 1);
    }
    const int kt0 = k_lo / BK, kt1 = (k_hi + BK - 1) / BK;
    const bf16_t* kbase = k + (long long)row0 * kvs + kvh * D;
    const bf16_t* vbase = v + (long long)row0 * kvs + kvh * D;

    for (int kt = kt0; kt < kt1; ++kt) {
        const int k0 = kt * BK;
        __syncthreads();   // the previous tile's V reads are done
#pragma unroll 2
        for (int i = tid; i < BK * CH; i += 128) {
            const int r = i / CH, c = i % CH, key = k0 + r;
            const bool ok = key < seq;
            cp_async16(sKV + swz<D>(r, c), kbase + (long long)(ok ? key : 0) * kvs + c * 8, ok);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // S = Q K^T for this warp's 16 rows x BK keys (n-tiles of 8 keys), two k-steps of 16 channels at a time.
        float sacc[NJ][4];
#pragma unroll
        for (int j = 0; j < NJ; ++j) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < D / 32; ++s2) {
            unsigned a0[4], a1[4];
            ldsm_x4(a0, sQ + swz<D>(warp * 16 + (lane & 15), 4 * s2 + (lane >> 4)));
            ldsm_x4(a1, sQ + swz<D>(warp * 16 + (lane & 15), 4 * s2 + 2 + (lane >> 4)));
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                unsigned b[4];   // b0, b1 of k-step 2 s2, then of k-step 2 s2 + 1
                ldsm_x4(b, sKV + swz<D>(8 * j + (lane & 7), 4 * s2 + (lane >> 3)));
                mma_bf16(sacc[j], a0, b[0], b[1]);
                mma_bf16(sacc[j], a1, b[2], b[3]);
            }
        }

        // Mask, scale and the online softmax update; rows g and g+8 are shared by the 4 lanes of a group.
        float mt0 = -INFINITY, mt1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int qi = e < 2 ? qr0 : qr1;
                const int kj = k0 + 8 * j + 2 * t + (e & 1);
                const bool ok = qi < seq && kj < seq && (window < 0 || (qi - kj <= window && kj - qi <= window));
                sacc[j][e] = ok ? sacc[j][e] * scale : -INFINITY;
            }
            mt0 = fmaxf(mt0, fmaxf(sacc[j][0], sacc[j][1]));
            mt1 = fmaxf(mt1, fmaxf(sacc[j][2], sacc[j][3]));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffff, mt0, off));
            mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffff, mt1, off));
        }
        const float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
        const float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        const float c0 = expf(m0 - ms0), c1 = expf(m1 - ms1);
        float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            sacc[j][0] = sacc[j][0] == -INFINITY ? 0.f : expf(sacc[j][0] - ms0);
            sacc[j][1] = sacc[j][1] == -INFINITY ? 0.f : expf(sacc[j][1] - ms0);
            sacc[j][2] = sacc[j][2] == -INFINITY ? 0.f : expf(sacc[j][2] - ms1);
            sacc[j][3] = sacc[j][3] == -INFINITY ? 0.f : expf(sacc[j][3] - ms1);
            ps0 += sacc[j][0] + sacc[j][1];
            ps1 += sacc[j][2] + sacc[j][3];
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            ps0 += __shfl_xor_sync(0xffffffff, ps0, off);
            ps1 += __shfl_xor_sync(0xffffffff, ps1, off);
        }
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int u = 0; u < NT; ++u) {
            o[u][0] *= c0; o[u][1] *= c0;
            o[u][2] *= c1; o[u][3] *= c1;
        }
        // The probability accumulators of n-tiles 2s and 2s+1 are the A fragment of k-step s of P V.
        unsigned pa[NS][4];
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            pa[s][0] = pack_bf16(sacc[2 * s][0], sacc[2 * s][1]);
            pa[s][1] = pack_bf16(sacc[2 * s][2], sacc[2 * s][3]);
            pa[s][2] = pack_bf16(sacc[2 * s + 1][0], sacc[2 * s + 1][1]);
            pa[s][3] = pack_bf16(sacc[2 * s + 1][2], sacc[2 * s + 1][3]);
        }

        __syncthreads();   // every warp is done with K: stage V in its place
#pragma unroll 2
        for (int i = tid; i < BK * CH; i += 128) {
            const int r = i / CH, c = i % CH, key = k0 + r;
            const bool ok = key < seq;
            cp_async16(sKV + swz<D>(r, c), vbase + (long long)(ok ? key : 0) * kvs + c * 8, ok);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // O += P V: V^T fragments by transposing loads, two 8-channel n-tiles per ldmatrix.
#pragma unroll
        for (int s = 0; s < NS; ++s) {
#pragma unroll
            for (int u2 = 0; u2 < NT / 2; ++u2) {
                unsigned b[4];   // b0, b1 of n-tile 2 u2, then of n-tile 2 u2 + 1
                ldsm_x4_t(b, sKV + swz<D>(16 * s + (lane & 15), 2 * u2 + (lane >> 4)));
                mma_bf16(o[2 * u2], pa[s], b[0], b[1]);
                mma_bf16(o[2 * u2 + 1], pa[s], b[2], b[3]);
            }
        }
    }

    // Normalise and store (rows past `seq` are dropped).
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    bf16_t* orow0 = out + (long long)(row0 + qr0) * qs + h * D;
    bf16_t* orow1 = out + (long long)(row0 + qr1) * qs + h * D;
#pragma unroll
    for (int u = 0; u < NT; ++u) {
        const int c = 8 * u + 2 * t;
        if (qr0 < seq) store2(orow0 + c, o[u][0] * i0, o[u][1] * i0);
        if (qr1 < seq) store2(orow1 + c, o[u][2] * i1, o[u][3] * i1);
    }
}

extern "C" __global__ void __launch_bounds__(128, 2)
flash_attn_d256_bb(const bf16_t* __restrict__ q, const bf16_t* __restrict__ k, const bf16_t* __restrict__ v, bf16_t* __restrict__ out,
                   int row0, int seq, int nh, int nkv, int window, float scale, int qs, int kvs) {
    flash_tc_body<256, 32>(q, k, v, out, nullptr, row0, seq, nh, nkv, window, scale, qs, kvs);
}

// The same kernel over many short sequences packed back to back (the text backbone's fused passes): grid =
// (tiles, query heads), one tile per 64 queries of a sequence, see the tile-list mode of flash_tc_body. `tiles` comes
// right after `out`; row0 / seq are unused (kept so that the tile-list kernels, this one and flash_attn_d512_bb,
// share one parameter list; the single-sequence d64 / d256 kernels have no `tiles`).
extern "C" __global__ void __launch_bounds__(128, 2)
flash_attn_d256_packed_bb(const bf16_t* __restrict__ q, const bf16_t* __restrict__ k, const bf16_t* __restrict__ v, bf16_t* __restrict__ out,
                          const int* __restrict__ tiles, int row0, int seq, int nh, int nkv, int window, float scale, int qs, int kvs) {
    flash_tc_body<256, 32>(q, k, v, out, tiles, row0, seq, nh, nkv, window, scale, qs, kvs);
}

// The same kernel for head_dim 64 with bf16 I/O (the vision tower's bf16 activations): 64-key tiles, 16 KB of shared
// memory. Measured on the 2394-patch test image (16 layers): 13.2 ms, against 15.0 with 32-key and 14.4 with
// 128-key tiles, and 16.0 for flash_attn_d64_bf16's register-staged loads.
extern "C" __global__ void __launch_bounds__(128, 4)
flash_attn_d64_bb(const bf16_t* __restrict__ q, const bf16_t* __restrict__ k, const bf16_t* __restrict__ v, bf16_t* __restrict__ out,
                  int row0, int seq, int nh, int nkv, int window, float scale, int qs, int kvs) {
    flash_tc_body<64, 64>(q, k, v, out, nullptr, row0, seq, nh, nkv, window, scale, qs, kvs);
}


// ---------------------------------------------------------------------------------------------
// Flash attention for 512-channel heads (the text backbone's global layers) on bf16 tensor cores.
// ---------------------------------------------------------------------------------------------

// A warp's 16 x 512 output tile would need 256 float32 registers per thread, so the head dimension is split across a
// PAIR of warps: block = 8 warps = 4 row groups (16 query rows each, 64 per block) x 2 halves of the channels. Warp
// (row group r, half h) owns the Q fragments of its 16 rows over channels 256h .. 256h+255 (16 k-steps x 4 registers,
// loaded once and kept for the whole kernel) and the 128 accumulators of the output channels 256h .. 256h+255.
// For each tile of BK keys, a warp computes the PARTIAL scores S_h = Q_h K_h^T over its half of the channels; the
// pair exchanges them through shared memory (score tile j, lane l as float4 [j][l]: conflict-free) and both warps
// form S = S_0 + S_1. A float add is commutative, so the two warps hold bit-identical scores and run bit-identical
// online-softmax updates (mask, running max and sum, rescale) without any further communication; each then
// accumulates O_h += P V_h for its own channels. K and V live in separate swizzled buffers so that the copy of V (and
// of K for the next tile) overlaps the arithmetic of the current tile:
//   B1  K(t) visible, the previous tile's P V reads done  -> start copying V(t)
//       S_h = Q_h K_h^T; publish it
//   B2  V(t) and every partial S visible, K reads done    -> start copying K(t+1)
//       S = S_0 + S_1, softmax, O_h += P V_h
// The exchange needs both warps of a pair, and B2 is a block barrier anyway (K may only be overwritten once all four
// row groups have read it), so it doubles as the pair's barrier. Shared memory: K and V tiles of BK x 512 bf16 plus
// 8 warps x 16 x BK floats of partial scores (80 KB for BK = 32: dynamic, and one block per SM anyway since Q and O
// take ~190 registers per thread). Q is staged through the (still unused) K / V buffers, one private area per warp,
// and read into registers with ldmatrix before the key loop.
//
// Parameters as flash_tc_body, plus `tiles` for packed passes. tiles == nullptr: the sequence occupies rows
// row0 .. row0+seq-1 and the block covers the queries q0 = blockIdx.x * 64 ..; otherwise blockIdx.x indexes a list of
// int triples {row0, len, q0} (a tile of 64 queries starting at query q0 of the sequence that occupies rows
// row0 .. row0+len-1; the keys are that sequence's alone), and len == 0 marks a padding tile whose block exits at once
// (row0 / seq are then ignored). blockIdx.y is the query head; a window of -1 means none.

// One BK x 512 tile of K or V (keys k0 ..) into a swizzled buffer; keys past `seq` are zero. 256 threads: thread
// (rb, c) = (tid / 64, tid % 64) copies chunk c of rows rb, rb + 4, rb + 8 ... Since rb < 4, row rb + 4 it has
// r & 7 = rb ^ 4 (it & 1), so the destinations of all iterations are two registers plus immediate offsets (written
// out so that the compiler does not keep BK / 4 separate invariant addresses alive across the whole kernel).
template <int BK>
__device__ __forceinline__ void load_kv_tile_d512(bf16_t* dst, const bf16_t* base, int k0, int seq, int kvs, int tid) {
    const int rb = tid >> 6, c = tid & 63;
    const int off[2] = {rb * 512 + ((c ^ rb) << 3), rb * 512 + ((c ^ rb ^ 4) << 3)};
#pragma unroll
    for (int it = 0; it < BK / 4; ++it) {
        const int key = k0 + rb + 4 * it;
        const bool ok = key < seq;
        cp_async16(dst + off[it & 1] + it * 4 * 512, base + (long long)(ok ? key : 0) * kvs + c * 8, ok);
    }
}

template <int BK>
__device__ __forceinline__ void flash_d512_body(unsigned char* smem, const bf16_t* __restrict__ q, const bf16_t* __restrict__ k, const bf16_t* __restrict__ v,
                                                bf16_t* __restrict__ out, const int* __restrict__ tiles, int row0, int seq, int nh, int nkv,
                                                int window, float scale, int qs, int kvs) {
    // D: head width; HD: channels per warp; BQ: query rows per block; NKS: 16-channel k-steps of Q K^T over HD; NT: 8-channel
    // output tiles over HD; NJ: 8-key score tiles; NS: 16-key steps of P V; QC: channels of Q staged per round.
    constexpr int D = 512, HD = 256, BQ = 64, NKS = HD / 16, NT = HD / 8, NJ = BK / 8, NS = BK / 16;
    constexpr int QC = BK * 8 < HD ? BK * 8 : HD, QR = HD / QC;
    bf16_t* sK = reinterpret_cast<bf16_t*>(smem);
    bf16_t* sV = sK + BK * D;
    float4* sX = reinterpret_cast<float4*>(sV + BK * D);   // [warp][score tile][lane]
    const int h = blockIdx.y, kvh = h / (nh / nkv);
    int q0 = blockIdx.x * BQ;
    if (tiles) {
        const int* tl = tiles + 3 * blockIdx.x;
        row0 = tl[0];
        seq = tl[1];
        q0 = tl[2];
        if (seq <= 0) return;
    }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int rg = warp >> 1, hf = warp & 1;               // row group, channel half
    const int qr0 = q0 + 16 * rg + g, qr1 = qr0 + 8;       // the two query rows this thread holds

    // Q fragments of 16 rows x this warp's 256 channels, via a private staging area in the K / V buffers (rows past
    // `seq` are zero): QR rounds of QC channels.
    unsigned qa[NKS][4];
    {
        bf16_t* sq = sK + warp * (16 * QC);
#pragma unroll
        for (int rd = 0; rd < QR; ++rd) {
#pragma unroll
            for (int it = 0; it < QC / 16; ++it) {
                const int i = lane + 32 * it, r = i / (QC / 8), c = i % (QC / 8), qi = q0 + 16 * rg + r;
                const bool ok = qi < seq;
                cp_async16(sq + swz<QC>(r, c), q + (long long)(row0 + (ok ? qi : 0)) * qs + h * D + hf * HD + rd * QC + c * 8, ok);
            }
            cp_async_commit();
            cp_async_wait_all();
            __syncwarp();
#pragma unroll
            for (int s = 0; s < QC / 16; ++s) ldsm_x4(qa[rd * (QC / 16) + s], sq + swz<QC>(lane & 15, 2 * s + (lane >> 4)));
            __syncwarp();   // the next round overwrites the staging area
        }
    }
    __syncthreads();   // every warp has read its Q: the K / V buffers are free

    float o[NT][4];
#pragma unroll
    for (int u = 0; u < NT; ++u) o[u][0] = o[u][1] = o[u][2] = o[u][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    int k_lo = 0, k_hi = seq;
    if (window >= 0) {
        k_lo = max(0, q0 - window);
        k_hi = min(seq, q0 + BQ - 1 + window + 1);
    }
    const int kt0 = k_lo / BK, kt1 = (k_hi + BK - 1) / BK;
    const bf16_t* kbase = k + (long long)row0 * kvs + kvh * D;
    const bf16_t* vbase = v + (long long)row0 * kvs + kvh * D;
    float4* xw = sX + warp * (NJ * 32) + lane;               // this warp's partial scores
    const float4* xp = sX + (warp ^ 1) * (NJ * 32) + lane;   // its partner's

    if (kt0 < kt1) {
        load_kv_tile_d512<BK>(sK, kbase, kt0 * BK, seq, kvs, tid);
        cp_async_commit();
    }
    for (int kt = kt0; kt < kt1; ++kt) {
        const int k0 = kt * BK;
        cp_async_wait_all();
        __syncthreads();   // B1: K(t) has landed; the previous tile's V and partial-score reads are done
        load_kv_tile_d512<BK>(sV, vbase, k0, seq, kvs, tid);
        cp_async_commit();

        // Partial scores S_h = Q_h K_h^T for this warp's 16 rows x BK keys over its 256 channels. One ldmatrix gives the
        // K fragments of two score tiles for one k-step, so consecutive mma's use different accumulators.
        float sacc[NJ][4];
#pragma unroll
        for (int j = 0; j < NJ; ++j) sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NKS; ++ks) {
#pragma unroll
            for (int jp = 0; jp < NJ / 2; ++jp) {
                unsigned b[4];   // b0, b1 of score tile 2 jp, then of tile 2 jp + 1
                ldsm_x4(b, sK + swz<D>(16 * jp + 8 * (lane >> 4) + (lane & 7), hf * (HD / 8) + 2 * ks + ((lane >> 3) & 1)));
                mma_bf16(sacc[2 * jp], qa[ks], b[0], b[1]);
                mma_bf16(sacc[2 * jp + 1], qa[ks], b[2], b[3]);
            }
        }
#pragma unroll
        for (int j = 0; j < NJ; ++j) xw[j * 32] = make_float4(sacc[j][0], sacc[j][1], sacc[j][2], sacc[j][3]);
        cp_async_wait_all();
        __syncthreads();   // B2: V(t) has landed and every partial score is published; K is free again
        if (kt + 1 < kt1) {
            load_kv_tile_d512<BK>(sK, kbase, k0 + BK, seq, kvs, tid);
            cp_async_commit();
        }
        // S = S_0 + S_1: the same two floats in either order, so both warps of the pair agree exactly.
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const float4 x = xp[j * 32];
            sacc[j][0] += x.x; sacc[j][1] += x.y; sacc[j][2] += x.z; sacc[j][3] += x.w;
        }

        // Scale, mask and the online softmax update; rows g and g+8 are shared by the 4 lanes of a group. A tile that is
        // entirely inside the sequence and the window needs no mask (query rows past `seq` are zero rows whose finite
        // scores are never stored).
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) sacc[j][e] *= scale;
        const bool plain = k0 + BK <= seq && (window < 0 || max(q0 + BQ - 1 - k0, k0 + BK - 1 - q0) <= window);
        if (!plain) {
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int qi = e < 2 ? qr0 : qr1;
                    const int kj = k0 + 8 * j + 2 * t + (e & 1);
                    const bool ok = qi < seq && kj < seq && (window < 0 || (qi - kj <= window && kj - qi <= window));
                    if (!ok) sacc[j][e] = -INFINITY;
                }
            }
        }
        float mt0 = -INFINITY, mt1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            mt0 = fmaxf(mt0, fmaxf(sacc[j][0], sacc[j][1]));
            mt1 = fmaxf(mt1, fmaxf(sacc[j][2], sacc[j][3]));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffff, mt0, off));
            mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffff, mt1, off));
        }
        const float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
        const float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        const float c0 = expf(m0 - ms0), c1 = expf(m1 - ms1);
        float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            sacc[j][0] = sacc[j][0] == -INFINITY ? 0.f : expf(sacc[j][0] - ms0);
            sacc[j][1] = sacc[j][1] == -INFINITY ? 0.f : expf(sacc[j][1] - ms0);
            sacc[j][2] = sacc[j][2] == -INFINITY ? 0.f : expf(sacc[j][2] - ms1);
            sacc[j][3] = sacc[j][3] == -INFINITY ? 0.f : expf(sacc[j][3] - ms1);
            ps0 += sacc[j][0] + sacc[j][1];
            ps1 += sacc[j][2] + sacc[j][3];
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            ps0 += __shfl_xor_sync(0xffffffff, ps0, off);
            ps1 += __shfl_xor_sync(0xffffffff, ps1, off);
        }
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = mn0;
        m1 = mn1;
        // Rescaling O by exactly 1 changes nothing: skip its 128 multiplies once the running maxima settle.
        if (!__all_sync(0xffffffff, c0 == 1.f && c1 == 1.f)) {
#pragma unroll
            for (int u = 0; u < NT; ++u) {
                o[u][0] *= c0; o[u][1] *= c0;
                o[u][2] *= c1; o[u][3] *= c1;
            }
        }
        // The probability accumulators of score tiles 2s and 2s+1 are the A fragment of k-step s of P V.
        unsigned pa[NS][4];
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            pa[s][0] = pack_bf16(sacc[2 * s][0], sacc[2 * s][1]);
            pa[s][1] = pack_bf16(sacc[2 * s][2], sacc[2 * s][3]);
            pa[s][2] = pack_bf16(sacc[2 * s + 1][0], sacc[2 * s + 1][1]);
            pa[s][3] = pack_bf16(sacc[2 * s + 1][2], sacc[2 * s + 1][3]);
        }

        // O_h += P V_h: V^T fragments by transposing loads, two 8-channel output tiles per ldmatrix.
#pragma unroll
        for (int s = 0; s < NS; ++s) {
#pragma unroll
            for (int u2 = 0; u2 < NT / 2; ++u2) {
                unsigned b[4];   // b0, b1 of output tile 2 u2, then of tile 2 u2 + 1
                ldsm_x4_t(b, sV + swz<D>(16 * s + (lane & 15), hf * (HD / 8) + 2 * u2 + (lane >> 4)));
                mma_bf16(o[2 * u2], pa[s], b[0], b[1]);
                mma_bf16(o[2 * u2 + 1], pa[s], b[2], b[3]);
            }
        }
    }

    // Normalise and store (rows past `seq` are dropped).
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    bf16_t* orow0 = out + (long long)(row0 + qr0) * qs + h * D + hf * HD;
    bf16_t* orow1 = out + (long long)(row0 + qr1) * qs + h * D + hf * HD;
#pragma unroll
    for (int u = 0; u < NT; ++u) {
        const int c = 8 * u + 2 * t;
        if (qr0 < seq) store2(orow0 + c, o[u][0] * i0, o[u][1] * i0);
        if (qr1 < seq) store2(orow1 + c, o[u][2] * i1, o[u][3] * i1);
    }
}

// Keys per tile (16 or 32; the host defines it from its FLASH_D512_BK, which also sizes the dynamic shared memory:
// 2 x BK x 512 bf16 for K and V, 8 x 16 x BK floats of partial scores).
#ifndef FLASH_D512_BK
#define FLASH_D512_BK 32
#endif
extern "C" __global__ void __launch_bounds__(256, 1)
flash_attn_d512_bb(const bf16_t* __restrict__ q, const bf16_t* __restrict__ k, const bf16_t* __restrict__ v, bf16_t* __restrict__ out,
                   const int* __restrict__ tiles, int row0, int seq, int nh, int nkv, int window, float scale, int qs, int kvs) {
    extern __shared__ __align__(16) unsigned char smem_d512[];
    flash_d512_body<FLASH_D512_BK>(smem_d512, q, k, v, out, tiles, row0, seq, nh, nkv, window, scale, qs, kvs);
}
