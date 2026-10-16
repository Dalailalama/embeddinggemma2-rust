//! Numerical kernels: GEMM wrappers, norms, activations, RoPE and blocked attention.
//!
//! All tensors are dense row-major `f32` slices; shapes are passed explicitly.

use anyhow::Result;
use gemm::Parallelism;
use rayon::prelude::*;

use crate::weights::Weights;

// ---------------------------------------------------------------------------------------------
// Size-aware thread pools
// ---------------------------------------------------------------------------------------------

/// Passes over fewer rows than this run on the small pool.
pub const SMALL_ROWS: usize = 256;

static SMALL_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SMALL_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

/// Threads used for small passes (short texts). Waking every core for a 17-row matmul costs more
/// than it saves, and on hybrid CPUs the slower cores become stragglers at every barrier; a
/// smaller pool stays on the fast cores. Default: a third of the logical CPUs, clamped to 2..=8.
/// Must be set before the first small pass; `0` means "use the default".
pub fn set_small_pool_threads(n: usize) {
    SMALL_THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
}

fn small_pool() -> &'static rayon::ThreadPool {
    SMALL_POOL.get_or_init(|| {
        let n = match SMALL_THREADS.load(std::sync::atomic::Ordering::Relaxed) {
            0 => std::thread::available_parallelism().map(|n| n.get() / 3).unwrap_or(4).clamp(2, 8),
            n => n,
        };
        rayon::ThreadPoolBuilder::new().num_threads(n).build().expect("building small thread pool")
    })
}

/// Run `f` on a pool sized for a pass over `rows` rows.
pub fn run_sized<R: Send>(rows: usize, f: impl FnOnce() -> R + Send) -> R {
    if rows < SMALL_ROWS && small_pool().current_num_threads() < rayon::current_num_threads() {
        small_pool().install(f)
    } else {
        f()
    }
}

// ---------------------------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------------------------

fn par_for(m: usize, n: usize, k: usize) -> Parallelism {
    if m.saturating_mul(n).saturating_mul(k) >= (1 << 21) { Parallelism::Rayon(0) } else { Parallelism::None }
}

/// `c[m x n] (+)= a[m x k] . b[n x k]^T`, with arbitrary row strides.
///
/// # Safety
/// Pointers must be valid for the described extents; `c` must not alias `a` or `b`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
    par: Parallelism,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                unsafe { std::ptr::write_bytes(c.add(i * ldc), 0, n) };
            }
        }
        return;
    }
    unsafe {
        gemm::gemm::<f32>(
            m, n, k, c, 1, ldc as isize, accumulate, a, 1, lda as isize, b, ldb as isize, 1, 1.0, 1.0, false,
            false, false, par,
        );
    }
}

/// `c[m x n] (+)= a[m x k] . b[k x n]`, with arbitrary row strides.
///
/// # Safety
/// Same contract as [`sgemm_nt`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_nn(
    m: usize,
    n: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
    par: Parallelism,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                unsafe { std::ptr::write_bytes(c.add(i * ldc), 0, n) };
            }
        }
        return;
    }
    unsafe {
        gemm::gemm::<f32>(
            m, n, k, c, 1, ldc as isize, accumulate, a, 1, lda as isize, b, 1, ldb as isize, 1.0, 1.0, false,
            false, false, par,
        );
    }
}

/// `out[m x n] = x[m x k] . w[n x k]^T`.
pub fn matmul_nt(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);
    let mut out = vec![0f32; m * n];
    unsafe { sgemm_nt(m, n, k, x.as_ptr(), k, w.as_ptr(), k, out.as_mut_ptr(), n, false, par_for(m, n, k)) };
    out
}

/// A dense layer `y = x W^T (+ b)` with `W` stored `[out, in]` like `nn.Linear`.
pub struct Linear {
    pub w: Vec<f32>,
    pub out: usize,
    pub inp: usize,
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    pub fn load(w: &Weights, prefix: &str, out: usize, inp: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            w: w.f32_shaped(&format!("{prefix}.weight"), &[out, inp])?,
            out,
            inp,
            bias: if bias { Some(w.f32_shaped(&format!("{prefix}.bias"), &[out])?) } else { None },
        })
    }

    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut y = matmul_nt(x, m, self.inp, &self.w, self.out);
        if let Some(b) = &self.bias {
            y.par_chunks_mut(self.out).for_each(|row| {
                for (v, bb) in row.iter_mut().zip(b) {
                    *v += *bb;
                }
            });
        }
        y
    }
}

/// `nn.Linear` wrapped with optional input/output clamping (`Gemma4ClippableLinear`).
pub struct ClippableLinear {
    pub linear: Linear,
    pub clip: Option<[f32; 4]>, // input_min, input_max, output_min, output_max
}

impl ClippableLinear {
    pub fn load(w: &Weights, prefix: &str, out: usize, inp: usize, clipped: bool) -> Result<Self> {
        let linear = Linear::load(w, &format!("{prefix}.linear"), out, inp, false)?;
        let clip = if clipped {
            Some([
                w.scalar(&format!("{prefix}.input_min"))?,
                w.scalar(&format!("{prefix}.input_max"))?,
                w.scalar(&format!("{prefix}.output_min"))?,
                w.scalar(&format!("{prefix}.output_max"))?,
            ])
        } else {
            None
        };
        Ok(Self { linear, clip })
    }

    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        match self.clip {
            None => self.linear.forward(x, m),
            Some([imin, imax, omin, omax]) => {
                let xc: Vec<f32> = x.iter().map(|v| v.clamp(imin, imax)).collect();
                let mut y = self.linear.forward(&xc, m);
                y.par_iter_mut().for_each(|v| *v = v.clamp(omin, omax));
                y
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Norms and activations
// ---------------------------------------------------------------------------------------------

/// RMS-normalise every `dim`-sized row in place, optionally scaling by `weight`.
/// Matches `Gemma4RMSNorm`: `x * (mean(x^2) + eps)^-0.5 * weight`.
pub fn rms_norm(x: &mut [f32], dim: usize, weight: Option<&[f32]>, eps: f32) {
    debug_assert_eq!(x.len() % dim, 0);
    let one = |row: &mut [f32]| {
        let mut ss = 0f64;
        for v in row.iter() {
            ss += (*v as f64) * (*v as f64);
        }
        let ms = (ss / dim as f64) as f32 + eps;
        let inv = ms.powf(-0.5);
        match weight {
            Some(w) => {
                for (v, wv) in row.iter_mut().zip(w) {
                    *v = *v * inv * *wv;
                }
            }
            None => {
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        }
    };
    if x.len() >= 1 << 15 {
        x.par_chunks_mut(dim).for_each(one);
    } else {
        x.chunks_mut(dim).for_each(one);
    }
}

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const BETA: f32 = 0.797_884_56; // sqrt(2/pi)
    const KAPPA: f32 = 0.044715;
    let inner = BETA * (x + KAPPA * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    if a.len() >= 1 << 15 {
        a.par_iter_mut().zip(b.par_iter()).for_each(|(x, y)| *x += *y);
    } else {
        for (x, y) in a.iter_mut().zip(b) {
            *x += *y;
        }
    }
}

/// In-place softmax of one row (stable). Masked entries must already be `-inf`.
pub fn softmax_row(row: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for v in row.iter() {
        if *v > mx {
            mx = *v;
        }
    }
    let mut sum = 0f32;
    for v in row.iter_mut() {
        *v = (*v - mx).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

// ---------------------------------------------------------------------------------------------
// Rotary embeddings
// ---------------------------------------------------------------------------------------------

/// `inv_freq[i] = 1 / base^(2i / dim)` for `i in 0..dim/2` (computed in f32, like the reference).
pub fn rope_inv_freq(base: f64, dim: usize) -> Vec<f32> {
    (0..dim / 2).map(|i| 1.0 / (base as f32).powf((2 * i) as f32 / dim as f32)).collect()
}

/// `cos/sin` tables of shape `[len, dim]` using the duplicated-halves layout `cat(freqs, freqs)`.
pub fn rope_tables(positions: &[f32], inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let dim = half * 2;
    let mut cos = vec![0f32; positions.len() * dim];
    let mut sin = vec![0f32; positions.len() * dim];
    for (t, p) in positions.iter().enumerate() {
        for i in 0..half {
            let f = *p * inv_freq[i];
            let (s, c) = f.sin_cos();
            cos[t * dim + i] = c;
            cos[t * dim + half + i] = c;
            sin[t * dim + i] = s;
            sin[t * dim + half + i] = s;
        }
    }
    (cos, sin)
}

/// Apply `x*cos + rotate_half(x)*sin` to every `dim`-sized chunk of one head slice.
/// `x` is one `dim`-vector; `cos`/`sin` are the matching `dim`-vectors.
#[inline]
pub fn rope_apply(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        let (x1, x2) = (x[i], x[half + i]);
        x[i] = x1 * cos[i] - x2 * sin[i];
        x[half + i] = x2 * cos[half + i] + x1 * sin[half + i];
    }
}

/// RoPE over `[tokens, heads * head_dim]` with per-token tables `[tokens, head_dim]`.
pub fn rope_rows(x: &mut [f32], heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let stride = heads * head_dim;
    x.par_chunks_mut(stride).enumerate().for_each(|(t, row)| {
        let c = &cos[t * head_dim..(t + 1) * head_dim];
        let s = &sin[t * head_dim..(t + 1) * head_dim];
        for h in 0..heads {
            rope_apply(&mut row[h * head_dim..(h + 1) * head_dim], c, s);
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub struct AttnShape {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Inclusive radius of a bidirectional sliding window (`|q-k| <= window`), or `None` for global.
    pub window: Option<usize>,
    pub scale: f32,
}

const Q_BLOCK: usize = 128;

/// Bidirectional multi-head attention over a single sequence.
///
/// `q` is `[seq, n_heads*hd]`, `k`/`v` are `[seq, n_kv_heads*hd]`; the result `[seq, n_heads*hd]`.
/// Every sequence position is valid (no padding), which is how this port always runs.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, s: &AttnShape) -> Vec<f32> {
    let hd = s.head_dim;
    let qs = s.n_heads * hd;
    let kvs = s.n_kv_heads * hd;
    assert_eq!(q.len(), seq * qs);
    assert_eq!(k.len(), seq * kvs);
    assert_eq!(v.len(), seq * kvs);
    let group = s.n_heads / s.n_kv_heads;
    let mut out = vec![0f32; seq * qs];
    let out_ptr = SendPtr(out.as_mut_ptr());
    let n_blocks = seq.div_ceil(Q_BLOCK);

    (0..s.n_heads * n_blocks).into_par_iter().for_each(|task| {
        // Rebinding makes the closure capture the whole `Send` wrapper instead of its raw-pointer
        // field (edition-2021 disjoint capture would otherwise grab the non-`Send` `*mut f32`).
        #[allow(clippy::redundant_locals)]
        let out_ptr = out_ptr;
        let h = task / n_blocks;
        let blk = task % n_blocks;
        let g = h / group;
        let q0 = blk * Q_BLOCK;
        let q1 = (q0 + Q_BLOCK).min(seq);
        let (k0, k1) = match s.window {
            Some(w) => (q0.saturating_sub(w), (q1 - 1 + w + 1).min(seq)),
            None => (0, seq),
        };
        let (qb, kb) = (q1 - q0, k1 - k0);
        let mut scores = vec![0f32; qb * kb];
        unsafe {
            sgemm_nt(
                qb,
                kb,
                hd,
                q.as_ptr().add(q0 * qs + h * hd),
                qs,
                k.as_ptr().add(k0 * kvs + g * hd),
                kvs,
                scores.as_mut_ptr(),
                kb,
                false,
                Parallelism::None,
            );
        }
        for i in 0..qb {
            let qi = q0 + i;
            let row = &mut scores[i * kb..(i + 1) * kb];
            if s.scale != 1.0 {
                for v in row.iter_mut() {
                    *v *= s.scale;
                }
            }
            if let Some(w) = s.window {
                let lo = qi.saturating_sub(w);
                let hi = (qi + w + 1).min(seq);
                for (jj, v) in row.iter_mut().enumerate() {
                    let j = k0 + jj;
                    if j < lo || j >= hi {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
            softmax_row(row);
        }
        unsafe {
            sgemm_nn(
                qb,
                hd,
                kb,
                scores.as_ptr(),
                kb,
                v.as_ptr().add(k0 * kvs + g * hd),
                kvs,
                out_ptr.0.add(q0 * qs + h * hd),
                qs,
                false,
                Parallelism::None,
            );
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------
// Pooling helpers
// ---------------------------------------------------------------------------------------------

/// Mean over rows of a `[rows, dim]` matrix.
pub fn mean_rows(x: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f64; dim];
    for r in 0..rows {
        for (o, v) in out.iter_mut().zip(&x[r * dim..(r + 1) * dim]) {
            *o += *v as f64;
        }
    }
    out.iter().map(|v| (*v / rows as f64) as f32).collect()
}

/// L2-normalise a vector in place (`F.normalize`, eps = 1e-12).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x = (*x as f64 / norm) as f32;
    }
}
