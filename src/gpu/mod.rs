//! CUDA backend: cuBLAS for every matrix product, small NVRTC-compiled kernels for the rest.
//!
//! Activations stay resident on the GPU for a whole tower forward pass; only the inputs
//! (embeddings / patches / subsampled audio) go up and the final per-token states come down.
//! CUDA is loaded dynamically at runtime, so the crate builds without the CUDA SDK and the
//! model falls back to the CPU when no driver is present.
//!
//! Most of the work is arranged around two costs: every kernel launch or device allocation costs host
//! time (so layers use few, fused kernels and per-pass scratch buffers, and short text passes replay a
//! recorded CUDA graph), and memory traffic (so the vision tower uses a fused flash-style attention kernel
//! and, in bf16 mode, the tensors that go into and come out of matrix products are stored as bf16, see
//! [`Act`]; residual streams, normalisation statistics, softmax and all accumulation stay float32).
//! Debug switches: `EGEMMA2_NO_FLASH` (cuBLAS / float32 attention instead of the fused flash kernels),
//! `EGEMMA2_NO_GRAPH` (no CUDA graphs for short text passes), `EGEMMA2_PACKED_FLASH` (which packed text
//! passes use the tensor-core flash kernel, see `text::PackedFlash`) and `EGEMMA2_PROFILE` (device time per op,
//! see `examples/profile_passes.rs`).

pub mod audio;
pub mod text;
pub mod vision;

use anyhow::{Context, Result, anyhow, bail, ensure};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, sys as cublas_sys};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::backend::Precision;
use crate::ops::{ClippableLinear, Linear};

pub type Buf = CudaSlice<f32>;
/// bfloat16 values as raw bits.
pub type Buf16 = CudaSlice<u16>;

/// An activation tensor: float32, or bf16 bits. In bf16 mode the inputs of matrix products (normalised hidden
/// states, attention outputs, MLP activations, ...) are stored as bf16: the products round them to bf16
/// anyway, so the kernel that produces one rounds it on the way out instead, which halves its memory traffic,
/// drops a conversion pass per product and changes no result. The products' outputs are stored as bf16 too
/// (cuBLAS rounds its float32 accumulators); that one does cost a little accuracy (embeddings stay within
/// cosine 0.99995 of float32). [`Gpu::alloc_act`] picks the format for the device; float32 mode keeps
/// everything float32.
pub enum Act {
    F32(Buf),
    Bf16(Buf16),
}

impl Act {
    pub fn is_bf16(&self) -> bool {
        matches!(self, Act::Bf16(_))
    }

    /// Device address of element `off`.
    fn ptr(&self, s: &Arc<CudaStream>, off: usize) -> u64 {
        match self {
            Act::F32(b) => b.device_ptr(s).0 + (off * 4) as u64,
            Act::Bf16(b) => b.device_ptr(s).0 + (off * 2) as u64,
        }
    }

    fn f32(&self) -> Result<&Buf> {
        match self {
            Act::F32(b) => Ok(b),
            Act::Bf16(_) => Err(anyhow!("internal: expected a float32 activation buffer")),
        }
    }

    fn f32_mut(&mut self) -> Result<&mut Buf> {
        match self {
            Act::F32(b) => Ok(b),
            Act::Bf16(_) => Err(anyhow!("internal: expected a float32 activation buffer")),
        }
    }
}

const KERNELS: &str = include_str!("kernels.cu");
/// Kernels that need compute capability 8.0+ (bf16 tensor-core `mma`). In bf16 mode they are appended to
/// `KERNELS` and the whole module is compiled for compute_80 with the bf16 variants enabled.
const KERNELS_SM80: &str = include_str!("kernels_sm80.cu");

/// Fused short-sequence attention kernels, keyed by (head_dim / 32, query heads per KV head, query heads per warp).
type PackedKernels = Vec<((usize, usize, usize), CudaFunction)>;

/// Kernels of the bf16 module (suffix `_fb`: float32 in, bf16 out; `_bb`: bf16 in and out; `_bf`: bf16 in,
/// float32 out), see the storage-type note in `kernels.cu`.
struct BfKernels {
    rms_norm_fb: CudaFunction,
    add_norm_fb: CudaFunction,
    add_norm_bb: CudaFunction,
    qkv_post_fb: CudaFunction,
    qkv_post_bb: CudaFunction,
    qkv_post_bf: CudaFunction,
    gelu_mul_fb: CudaFunction,
    gelu_mul_bb: CudaFunction,
    ple_gate_fb: CudaFunction,
    ple_gate_bb: CudaFunction,
    clamp_bf16: CudaFunction,
    /// Tensor-core flash attention for head_dim 64: float32 I/O, and bf16 I/O.
    flash_d64: CudaFunction,
    flash_d64_bb: CudaFunction,
    /// Tensor-core flash attention for head_dim 256 (the text backbone's sliding-window layers), bf16 I/O, and its
    /// tile-list variant over many packed sequences (see [`Gpu::attention_flash_packed`]).
    flash_d256_bb: CudaFunction,
    flash_d256_packed_bb: CudaFunction,
    /// Tensor-core flash attention for head_dim 512 (the text backbone's global layers), bf16 I/O, with an optional
    /// tile list for packed passes; needs [`FLASH_D512_SMEM`] bytes of dynamic shared memory.
    flash_d512_bb: CudaFunction,
    /// Packed attention with a bf16 output, keyed like [`Kernels::attn_packed`] (which is empty in bf16 mode).
    attn_packed_fb: PackedKernels,
    /// Warp-per-head `qkv_post`, keyed by (head_dim / 32, bf16 input, bf16 output); head_dim 64 takes RoPE
    /// parts of at most 32 channels, head_dim 256 / 512 1-D RoPE.
    qkv_post_warp: Vec<((usize, bool, bool), CudaFunction)>,
}

impl BfKernels {
    fn qkv_post_warp(&self, hd: usize, part: usize, in_bf16: bool, out_bf16: bool) -> Option<&CudaFunction> {
        let per = hd / 32;
        let ok = match per {
            2 => part <= 32 && part.is_power_of_two() && part >= 2,
            8 | 16 => part == hd,
            _ => false,
        };
        if !ok || !hd.is_multiple_of(32) {
            return None;
        }
        self.qkv_post_warp.iter().find(|(key, _)| *key == (per, in_bf16, out_bf16)).map(|(_, f)| f)
    }
}

struct Kernels {
    rms_norm: CudaFunction,
    add_norm: CudaFunction,
    qkv_post: CudaFunction,
    gelu_mul: CudaFunction,
    gelu_mul_gu: CudaFunction,
    add: CudaFunction,
    scale: CudaFunction,
    clamp: CudaFunction,
    clamp_copy: CudaFunction,
    bias_add: CudaFunction,
    ple_gate: CudaFunction,
    flash_attn_d64: CudaFunction,
    softmax_window: CudaFunction,
    softmax_window_r8: CudaFunction,
    softmax_window_r16: CudaFunction,
    /// Fused short-sequence attention (float32 module only), see [`PackedKernels`].
    attn_packed: PackedKernels,
    audio_qscale: CudaFunction,
    audio_qrel: CudaFunction,
    audio_attn: CudaFunction,
    glu: CudaFunction,
    dwconv_causal: CudaFunction,
    act_chain: CudaFunction,
    audio_qkv_post: CudaFunction,
    audio_conv0: CudaFunction,
    audio_im2col1: CudaFunction,
    ln_relu: CudaFunction,
    to_bf16: CudaFunction,
    softmax_window_bf16: CudaFunction,
    softmax_window_r8_bf16: CudaFunction,
    softmax_window_r16_bf16: CudaFunction,
}

/// A CUDA device with its stream, cuBLAS handle and compiled kernels.
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Kernels,
    lock: Mutex<()>,
    /// Placeholder argument for kernels whose optional pointer is unused.
    dummy: Buf,
    pub name: String,
    /// GEMM arithmetic, see [`Precision`]. Under `Bf16` the linear layers and the attention products use bf16
    /// tensor cores and the few remaining float32 products (the audio tower's relative-position table) TF32.
    pub precision: Precision,
    /// Use the fused flash-style attention kernels where available (`EGEMMA2_NO_FLASH` turns them off).
    flash: bool,
    /// The bf16 kernels (tensor-core attention, bf16-activation variants); present exactly in bf16 mode.
    bf: Option<BfKernels>,
    /// Per-op timing (`EGEMMA2_PROFILE`): events recorded between ops, see [`Gpu::mark`].
    prof: Option<Mutex<Marks>>,
}

/// Profiling marks: (label of the op that ended here, event recorded after it).
type Marks = Vec<(&'static str, cudarc::driver::CudaEvent)>;

/// A captured sequence of device work, see [`Gpu::capture`].
pub struct Graph(cudarc::driver::CudaGraph);

// Graphs are only created, replayed and dropped while holding the owning `Gpu`'s lock.
unsafe impl Send for Graph {}

// The cuBLAS handle is only used while holding `lock` (see `Gpu::lock`).
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// Packed attention uses one warp per query head (rather than per K/V head) up to this many rows x K/V
/// heads: a handful of rows needs the extra warps to fill the GPU, while bigger packs are better off
/// reading each K/V row once per group (measured crossover on the development GPU).
const PACKED_SPLIT_UP_TO: usize = 130;

/// Query rows per tile of the packed flash kernels' tile lists (the kernels' 64 queries per block).
pub const FLASH_TILE_ROWS: usize = 64;

/// Keys per tile of `flash_attn_d512_bb` (kernels_sm80.cu; 16 or 32), and its dynamic shared memory: K and V tiles of
/// that many keys x 512 bf16 plus the partial scores of 8 warps x 16 rows x that many keys.
const FLASH_D512_BK: usize = 32;
const FLASH_D512_SMEM: usize = 2 * FLASH_D512_BK * 512 * 2 + 8 * 16 * FLASH_D512_BK * 4;

/// "No clamping" bounds.
pub const NO_CLAMP: (f32, f32) = (f32::NEG_INFINITY, f32::INFINITY);

/// Extras of [`Gpu::rms_norm_opts`]: the input is clamped to `clamp_in`, pre-scaled by `pre`, normalised
/// (times the weight), optionally passed through SiLU and finally clamped to `clamp_out`.
#[derive(Clone, Copy)]
pub struct NormOpts {
    pub pre: f32,
    pub clamp_in: (f32, f32),
    pub clamp_out: (f32, f32),
    pub silu: bool,
}

impl Default for NormOpts {
    fn default() -> Self {
        Self { pre: 1.0, clamp_in: NO_CLAMP, clamp_out: NO_CLAMP, silu: false }
    }
}

/// Options of [`Gpu::add_norm_ex`]: `n = clamp(o, clamp_o)`, normalised and weighted by `w1` when given;
/// then `h = (h + n) * scalar`, or with `axpy` `h = fma(n, scalar, h)`. The optional next pre-norm
/// `t = clamp(norm(clamp(h, next_in)) * w2, next_out)` is produced from the updated `h`.
#[derive(Clone, Copy)]
pub struct AddNorm<'a> {
    pub w1: Option<&'a Buf>,
    pub scalar: f32,
    pub axpy: bool,
    pub clamp_o: (f32, f32),
    pub next_in: (f32, f32),
    pub next_out: (f32, f32),
}

impl Default for AddNorm<'_> {
    fn default() -> Self {
        Self { w1: None, scalar: 1.0, axpy: false, clamp_o: NO_CLAMP, next_in: NO_CLAMP, next_out: NO_CLAMP }
    }
}

/// Threads per block of a row-wise reduction over `dim` elements (this fixes the summation order).
fn norm_threads(dim: usize) -> u32 {
    if dim >= 1024 { 256 } else if dim >= 256 { 128 } else { 64 }
}

fn launch_1d(n: usize) -> LaunchConfig {
    let threads = 256u32;
    LaunchConfig { grid_dim: ((n as u64).div_ceil(threads as u64) as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

impl Gpu {
    /// Open CUDA device `ordinal` with the given GEMM [`Precision`].
    pub fn new(ordinal: usize, precision: Precision) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(|e| anyhow!("CUDA device {ordinal} unavailable: {e:?}"))?;
        let name = ctx.name().unwrap_or_else(|_| "CUDA device".into());
        let (major, minor) = ctx.compute_capability().map_err(|e| anyhow!("querying compute capability: {e:?}"))?;
        ensure!(
            precision != Precision::Bf16 || major >= 8,
            "bf16 mode needs bf16 tensor cores (compute capability 8.0+, Ampere or newer); {name} is {major}.{minor}"
        );
        // One private stream for everything: it can be captured into CUDA graphs (the legacy default stream
        // cannot), and with a single stream cudarc's cross-stream event bookkeeping is pure overhead.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(|e| anyhow!("creating a CUDA stream: {e:?}"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| anyhow!("cuBLAS init failed: {e:?}"))?;
        let mode = if precision == Precision::F32 { cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH } else { cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH };
        let st = unsafe { cublas_sys::cublasSetMathMode(*blas.handle(), mode) };
        ensure!(st == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cublasSetMathMode failed: {st:?}");

        let bf16 = precision == Precision::Bf16;
        let ptx = if bf16 {
            let src = format!("#define EG_BF16\n#define FLASH_D512_BK {FLASH_D512_BK}\n{KERNELS}\n{KERNELS_SM80}");
            compile_ptx_with_opts(src, CompileOptions { use_fast_math: Some(false), arch: Some("compute_80"), ..Default::default() })
        } else {
            compile_ptx_with_opts(KERNELS, CompileOptions { use_fast_math: Some(false), ..Default::default() })
        }
        .map_err(|e| anyhow!("NVRTC failed to compile the kernels: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(|e| anyhow!("loading kernels: {e:?}"))?;
        let f = |name: &str| module.load_function(name).map_err(|e| anyhow!("kernel {name}: {e:?}"));
        let packed = |prefix: &str| -> Result<PackedKernels> {
            let mut v = Vec::new();
            for per in [2, 4, 8, 16] {
                for (g, hw) in [(1, 1), (2, 2), (2, 1), (4, 4), (4, 1)] {
                    v.push(((per, g, hw), f(&format!("{prefix}_{per}_{g}_{hw}"))?));
                }
            }
            Ok(v)
        };
        let k = Kernels {
            rms_norm: f("rms_norm")?,
            add_norm: f("add_norm")?,
            qkv_post: f("qkv_post")?,
            gelu_mul: f("gelu_mul")?,
            gelu_mul_gu: f("gelu_mul_gu")?,
            add: f("add")?,
            scale: f("scale")?,
            clamp: f("clamp")?,
            clamp_copy: f("clamp_copy")?,
            bias_add: f("bias_add")?,
            ple_gate: f("ple_gate")?,
            flash_attn_d64: f("flash_attn_d64")?,
            softmax_window: f("softmax_window")?,
            softmax_window_r8: f("softmax_window_r8")?,
            softmax_window_r16: f("softmax_window_r16")?,
            attn_packed: if bf16 { Vec::new() } else { packed("attn_packed")? },
            audio_qscale: f("audio_qscale")?,
            audio_qrel: f("audio_qrel")?,
            audio_attn: f("audio_attn")?,
            glu: f("glu")?,
            dwconv_causal: f("dwconv_causal")?,
            act_chain: f("act_chain")?,
            audio_qkv_post: f("audio_qkv_post")?,
            audio_conv0: f("audio_conv0")?,
            audio_im2col1: f("audio_im2col1")?,
            ln_relu: f("ln_relu")?,
            to_bf16: f("to_bf16")?,
            softmax_window_bf16: f("softmax_window_bf16")?,
            softmax_window_r8_bf16: f("softmax_window_r8_bf16")?,
            softmax_window_r16_bf16: f("softmax_window_r16_bf16")?,
        };
        // Keep freed device memory in the stream-ordered allocator's pool across synchronisations
        // (by default it is returned to the driver at every sync, so each forward pass would pay for
        // fresh multi-megabyte allocations again).
        unsafe {
            use cudarc::driver::{result, sys};
            if let Ok(pool) = result::device::get_default_mem_pool(ctx.cu_device()) {
                let mut keep: u64 = 1 << 30;
                let _ = result::mem_pool::set_attribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &mut keep as *mut u64 as *mut std::ffi::c_void);
            }
        }
        let bf = if bf16 {
            let flash_d512_bb = f("flash_attn_d512_bb")?;
            // More than the 48 KB a launch gets by default: opt in (sm_86 allows up to 99 KB per block).
            flash_d512_bb
                .set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, FLASH_D512_SMEM as i32)
                .map_err(|e| anyhow!("flash_attn_d512_bb needs {FLASH_D512_SMEM} bytes of shared memory per block: {e:?}"))?;
            Some(BfKernels {
                rms_norm_fb: f("rms_norm_fb")?,
                add_norm_fb: f("add_norm_fb")?,
                add_norm_bb: f("add_norm_bb")?,
                qkv_post_fb: f("qkv_post_fb")?,
                qkv_post_bb: f("qkv_post_bb")?,
                qkv_post_bf: f("qkv_post_bf")?,
                gelu_mul_fb: f("gelu_mul_fb")?,
                gelu_mul_bb: f("gelu_mul_bb")?,
                ple_gate_fb: f("ple_gate_fb")?,
                ple_gate_bb: f("ple_gate_bb")?,
                clamp_bf16: f("clamp_bf16")?,
                flash_d64: f("flash_attn_d64_bf16")?,
                flash_d64_bb: f("flash_attn_d64_bb")?,
                flash_d256_bb: f("flash_attn_d256_bb")?,
                flash_d256_packed_bb: f("flash_attn_d256_packed_bb")?,
                flash_d512_bb,
                attn_packed_fb: packed("attn_packed_fb")?,
                qkv_post_warp: {
                    let mut v = Vec::new();
                    for (per, kind) in [(2, "p"), (8, "r"), (16, "r")] {
                        for (io, ib, ob) in [("fb", false, true), ("bb", true, true), ("bf", true, false)] {
                            v.push(((per, ib, ob), f(&format!("qkv_post_w{per}{kind}_{io}"))?));
                        }
                    }
                    v
                },
            })
        } else {
            None
        };
        let dummy = stream.alloc_zeros::<f32>(1).map_err(|e| anyhow!("{e:?}"))?;
        let prof = std::env::var_os("EGEMMA2_PROFILE").map(|_| Mutex::new(Vec::new()));
        Ok(Self { ctx, stream, blas, k, lock: Mutex::new(()), dummy, name, precision, flash: std::env::var_os("EGEMMA2_NO_FLASH").is_none(), bf, prof })
    }

    /// Whether activations that feed matrix products are stored as bf16 (bf16 mode), see [`Act`].
    pub fn bf16_act(&self) -> bool {
        self.bf.is_some()
    }

    fn bfk(&self) -> Result<&BfKernels> {
        self.bf.as_ref().ok_or_else(|| anyhow!("internal: bf16 kernels used outside bf16 mode"))
    }

    /// Turn the fused flash-style attention kernels on or off (like `EGEMMA2_NO_FLASH`), e.g. to compare them with
    /// the cuBLAS path.
    #[doc(hidden)]
    pub fn set_flash(&mut self, on: bool) {
        self.flash = on;
    }

    /// Whether the fused flash-style attention kernels are in use (`EGEMMA2_NO_FLASH` turns them off).
    pub fn flash_on(&self) -> bool {
        self.flash
    }

    /// Copy an activation buffer to the host as float32.
    pub fn download_act(&self, a: &Act) -> Result<Vec<f32>> {
        match a {
            Act::F32(b) => self.download(b),
            Act::Bf16(b) => {
                let bits = self.stream.clone_dtoh(b).map_err(|e| anyhow!("device->host copy: {e:?}"))?;
                Ok(bits.into_iter().map(|u| f32::from_bits((u as u32) << 16)).collect())
            }
        }
    }

    /// Whether per-op profiling is on (it also turns CUDA graphs off: marks must not be captured).
    pub fn profiling(&self) -> bool {
        self.prof.is_some()
    }

    /// Profiling: attribute the device time since the previous mark to `label`. No-op unless profiling.
    pub fn mark(&self, label: &'static str) {
        if let Some(p) = &self.prof
            && let Ok(ev) = self.ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            && ev.record(&self.stream).is_ok()
        {
            p.lock().unwrap_or_else(|e| e.into_inner()).push((label, ev));
        }
    }

    /// Profiling: device milliseconds per label since the last call (ops in first-seen order), then reset.
    pub fn take_profile(&self) -> Vec<(&'static str, f64)> {
        let Some(p) = &self.prof else { return Vec::new() };
        let marks = std::mem::take(&mut *p.lock().unwrap_or_else(|e| e.into_inner()));
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for pair in marks.windows(2) {
            let ms = pair[0].1.elapsed_ms(&pair[1].1).unwrap_or(0.0) as f64;
            // "start" opens a pass: the gap before it is host time between passes, not an op.
            if pair[1].0 == "start" {
                continue;
            }
            match out.iter_mut().find(|(l, _)| *l == pair[1].0) {
                Some(e) => e.1 += ms,
                None => out.push((pair[1].0, ms)),
            }
        }
        out
    }


    /// Serialise use of the stream / cuBLAS handle across host threads.
    pub fn lock(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    // -----------------------------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------------------------

    pub fn upload(&self, x: &[f32]) -> Result<Buf> {
        self.stream.clone_htod(x).map_err(|e| anyhow!("host->device copy: {e:?}"))
    }

    pub fn download(&self, x: &Buf) -> Result<Vec<f32>> {
        self.stream.clone_dtoh(x).map_err(|e| anyhow!("device->host copy: {e:?}"))
    }

    /// Uninitialised device buffer (every op below fully overwrites its outputs).
    pub fn alloc(&self, n: usize) -> Result<Buf> {
        unsafe { self.stream.alloc::<f32>(n.max(1)) }.map_err(|e| anyhow!("device alloc of {n} floats: {e:?}"))
    }

    /// Uninitialised buffer of `n` bf16 values.
    pub fn alloc16(&self, n: usize) -> Result<Buf16> {
        unsafe { self.stream.alloc::<u16>(n.max(1)) }.map_err(|e| anyhow!("device alloc of {n} bf16: {e:?}"))
    }

    /// Uninitialised activation buffer of `n` values in the device's activation format (see [`Act`]).
    pub fn alloc_act(&self, n: usize) -> Result<Act> {
        Ok(if self.bf16_act() { Act::Bf16(self.alloc16(n)?) } else { Act::F32(self.alloc(n)?) })
    }

    /// Uninitialised float32 activation buffer, whatever the device's format (tensors that feed float32 kernels).
    pub fn alloc_act_f32(&self, n: usize) -> Result<Act> {
        Ok(Act::F32(self.alloc(n)?))
    }

    // -----------------------------------------------------------------------------------------
    // GEMM
    // -----------------------------------------------------------------------------------------

    /// Switch the cuBLAS float32 products that follow to plain float32 (`exact`) or back to the configured
    /// mode (TF32 unless the device runs in `F32`). Used for the audio front end, which ran in float32 on
    /// the host before it moved to the GPU and is cheap enough not to need tensor cores.
    pub fn exact_math(&self, exact: bool) -> Result<()> {
        let mode = if exact || self.precision == Precision::F32 { cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH } else { cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH };
        let st = unsafe { cublas_sys::cublasSetMathMode(*self.blas.handle(), mode) };
        ensure!(st == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "cublasSetMathMode failed: {st:?}");
        Ok(())
    }

    /// `y[m x n] = x[m x k] . w[n x k]^T` (row-major), the `nn.Linear` product.
    pub fn matmul_nt<X: DevicePtr<f32>, W: DevicePtr<f32>, Y: DevicePtrMut<f32>>(
        &self,
        x: &X,
        m: usize,
        k: usize,
        w: &W,
        n: usize,
        y: &mut Y,
    ) -> Result<()> {
        // Row-major Y = X W^T is column-major Y^T = W . X^T, i.e. gemm(T, N) on the raw buffers.
        let cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: k as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };
        unsafe { self.blas.gemm(cfg, w, x, y) }.map_err(|e| anyhow!("cublas gemm: {e:?}"))
    }

    /// `y[m x n] = bf16(clamp(x))[m x k] . w[n x k]^T` with bf16 weights, on bf16 tensor cores with float32
    /// accumulation and a float32 result. The activations are rounded to bf16 into a temporary buffer from
    /// the stream-ordered pool (inside a CUDA-graph capture that becomes a graph memory node, so recorded
    /// graphs never point at a buffer that is later reallocated).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_nt_bf16(&self, x: &Buf, m: usize, k: usize, w: &Buf16, n: usize, y: &mut Buf, clamp: (f32, f32)) -> Result<()> {
        let len = m * k;
        let mut xb = self.alloc16(len)?;
        let nn = len as i64;
        let cfg = LaunchConfig { grid_dim: ((len as u64).div_ceil(4 * 256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { self.stream.launch_builder(&self.k.to_bf16).arg(x).arg(&mut xb).arg(&nn).arg(&clamp.0).arg(&clamp.1).launch(cfg) }
            .map_err(|e| anyhow!("to_bf16: {e:?}"))?;
        self.mark("to_bf16 (GEMM input)");
        let xp = xb.device_ptr(&self.stream).0;
        let yp = y.device_ptr_mut(&self.stream).0;
        self.gemm_bf16(xp, m, k, w, n, yp, false)
    }

    /// The bf16 tensor-core product on raw device addresses: `y[m x n] = x[m x k] . w[n x k]^T` with bf16 `x`
    /// and `w`, float32 accumulation, and `y` stored as float32 or (`y_bf16`) rounded to bf16.
    #[allow(clippy::too_many_arguments)]
    fn gemm_bf16(&self, x: u64, m: usize, k: usize, w: &Buf16, n: usize, y: u64, y_bf16: bool) -> Result<()> {
        use cudarc::cublas::result as blas;
        use cublas_sys::cudaDataType::{CUDA_R_16BF, CUDA_R_32F};
        let (alpha, beta) = (1.0f32, 0.0f32);
        let wp = w.device_ptr(&self.stream).0;
        // Row-major Y = X W^T is column-major Y^T = W . X^T (see `matmul_nt`).
        unsafe {
            blas::gemm_ex(
                *self.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                wp as *const _,
                CUDA_R_16BF,
                k as i32,
                x as *const _,
                CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const _,
                y as *mut _,
                if y_bf16 { CUDA_R_16BF } else { CUDA_R_32F },
                n as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
            )
        }
        .map_err(|e| anyhow!("cublas bf16 gemm: {e:?}"))
    }

    /// Weights as bfloat16 bits, rounded to nearest even. EmbeddingGemma 2 ships bf16 weights, so the
    /// float32 copies made at load time convert back exactly.
    pub fn upload_bf16(&self, w: &[f32]) -> Result<CudaSlice<u16>> {
        let bits: Vec<u16> = w
            .iter()
            .map(|v| {
                let u = v.to_bits();
                if v.is_nan() { ((u >> 16) | 0x40) as u16 } else { ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16 }
            })
            .collect();
        self.stream.clone_htod(&bits).map_err(|e| anyhow!("host->device copy: {e:?}"))
    }

    // -----------------------------------------------------------------------------------------
    // Element-wise / row kernels
    // -----------------------------------------------------------------------------------------

    /// In-place RMS norm of `rows` rows of `dim` (`x = x * (mean(x^2) + eps)^-0.5 (* w)`).
    pub fn rms_norm(&self, x: &mut Buf, rows: usize, dim: usize, w: Option<&Buf>, eps: f32) -> Result<()> {
        self.rms_norm_impl(None, x, rows, dim, w, eps, NormOpts::default())
    }

    /// In-place `x = norm(pre * x) * w` (see [`rms_norm_to`](Self::rms_norm_to)).
    pub fn rms_norm_scaled(&self, x: &mut Buf, rows: usize, dim: usize, w: &Buf, eps: f32, pre: f32) -> Result<()> {
        self.rms_norm_impl(None, x, rows, dim, Some(w), eps, NormOpts { pre, ..Default::default() })
    }

    /// Out-of-place RMS norm of `pre * x`: `out = norm(pre * x) (* w)`. `pre` is a rounded pre-scale
    /// (same arithmetic as scaling `x` in a separate pass first).
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_to(&self, x: &Buf, out: &mut Buf, rows: usize, dim: usize, w: Option<&Buf>, eps: f32, pre: f32) -> Result<()> {
        self.rms_norm_impl(Some(x), out, rows, dim, w, eps, NormOpts { pre, ..Default::default() })
    }

    /// RMS norm with optional input / output clamping and a SiLU, see [`NormOpts`]. `x == None`: in place on `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_opts(&self, x: Option<&Buf>, out: &mut Buf, rows: usize, dim: usize, w: Option<&Buf>, eps: f32, opts: NormOpts) -> Result<()> {
        self.rms_norm_impl(x, out, rows, dim, w, eps, opts)
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_norm_impl(&self, x: Option<&Buf>, out: &mut Buf, rows: usize, dim: usize, w: Option<&Buf>, eps: f32, o: NormOpts) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (norm_threads(dim), 1, 1), shared_mem_bytes: 0 };
        let (r, d, has, act) = (rows as i32, dim as i32, w.is_some() as i32, o.silu as i32);
        let null = 0u64;
        let mut b = self.stream.launch_builder(&self.k.rms_norm);
        match x {
            Some(x) => b.arg(x),
            None => b.arg(&null),
        };
        b.arg(out);
        b.arg(w.unwrap_or(&self.dummy));
        b.arg(&r).arg(&d).arg(&eps).arg(&has).arg(&o.pre);
        b.arg(&o.clamp_in.0).arg(&o.clamp_in.1).arg(&o.clamp_out.0).arg(&o.clamp_out.1).arg(&act);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("rms_norm: {e:?}"))
    }

    /// `out = rmsnorm(x) * w` into an activation buffer (see [`Act`]).
    pub fn rms_norm_act(&self, x: &Buf, out: &mut Act, rows: usize, dim: usize, w: &Buf, eps: f32) -> Result<()> {
        let o = match out {
            Act::F32(o) => return self.rms_norm_to(x, o, rows, dim, Some(w), eps, 1.0),
            Act::Bf16(o) => o,
        };
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (norm_threads(dim), 1, 1), shared_mem_bytes: 0 };
        let (r, d, has, act, pre) = (rows as i32, dim as i32, 1i32, 0i32, 1.0f32);
        let (lo, hi) = NO_CLAMP;
        let mut b = self.stream.launch_builder(&self.bfk()?.rms_norm_fb);
        b.arg(x).arg(o).arg(w).arg(&r).arg(&d).arg(&eps).arg(&has).arg(&pre).arg(&lo).arg(&hi).arg(&lo).arg(&hi).arg(&act);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("rms_norm (bf16 out): {e:?}"))
    }

    /// Residual tail of a sub-block in one launch: `h = (h + rmsnorm(clamp(o)) * w1) * scalar`, and with
    /// `next = Some((t, w2))` also `t = rmsnorm(h) * w2` (the following sub-block's pre-norm).
    #[allow(clippy::too_many_arguments)]
    pub fn add_norm(
        &self,
        h: &mut Buf,
        o: &Buf,
        w1: &Buf,
        next: Option<(&mut Buf, &Buf)>,
        rows: usize,
        dim: usize,
        eps: f32,
        scalar: f32,
        clamp: Option<(f32, f32)>,
    ) -> Result<()> {
        let tail = AddNorm { w1: Some(w1), scalar, clamp_o: clamp.unwrap_or(NO_CLAMP), ..Default::default() };
        self.add_norm_ex(h, o, next, rows, dim, eps, tail)
    }

    /// General form of [`add_norm`](Self::add_norm), see [`AddNorm`].
    #[allow(clippy::too_many_arguments)]
    pub fn add_norm_ex(&self, h: &mut Buf, o: &Buf, next: Option<(&mut Buf, &Buf)>, rows: usize, dim: usize, eps: f32, a: AddNorm) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (norm_threads(dim), 1, 1), shared_mem_bytes: 0 };
        let (r, d, mode) = (rows as i32, dim as i32, a.axpy as i32);
        let null = 0u64;
        let mut b = self.stream.launch_builder(&self.k.add_norm);
        b.arg(h).arg(o);
        match a.w1 {
            Some(w) => b.arg(w),
            None => b.arg(&null),
        };
        match next {
            Some((t, w2)) => b.arg(t).arg(w2),
            None => b.arg(&null).arg(&null),
        };
        b.arg(&r).arg(&d).arg(&eps).arg(&a.scalar).arg(&a.clamp_o.0).arg(&a.clamp_o.1).arg(&mode);
        b.arg(&a.next_in.0).arg(&a.next_in.1).arg(&a.next_out.0).arg(&a.next_out.1);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("add_norm: {e:?}"))
    }

    /// [`add_norm`](Self::add_norm) on activation buffers. `next = Some((t, Some(w2)))` writes the next pre-norm
    /// into `t`; `Some((t, None))` writes a plain copy of the updated `h` (a matrix product's bf16 input).
    #[allow(clippy::too_many_arguments)]
    pub fn add_norm_act(&self, h: &mut Buf, o: &Act, w1: &Buf, next: Option<(&mut Act, Option<&Buf>)>, rows: usize, dim: usize, eps: f32, scalar: f32) -> Result<()> {
        let f = match (o, next.as_ref().map(|(t, _)| t.is_bf16())) {
            (Act::F32(_), None | Some(false)) => None,
            (Act::F32(_), Some(true)) => Some(&self.bfk()?.add_norm_fb),
            (Act::Bf16(_), None | Some(true)) => Some(&self.bfk()?.add_norm_bb),
            (Act::Bf16(_), Some(false)) => bail!("internal: add_norm from a bf16 input into a float32 output"),
        };
        let Some(f) = f else {
            let next = match next {
                None => None,
                Some((t, Some(w2))) => Some((t.f32_mut()?, w2)),
                Some((_, None)) => bail!("internal: add_norm copy-out is a bf16-mode step"),
            };
            return self.add_norm(h, o.f32()?, w1, next, rows, dim, eps, scalar, None);
        };
        if rows == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (norm_threads(dim), 1, 1), shared_mem_bytes: 0 };
        let (r, d, mode) = (rows as i32, dim as i32, 0i32);
        let (lo, hi) = NO_CLAMP;
        let s = &self.stream;
        let op = o.ptr(s, 0);
        let (tp, w2p) = match &next {
            Some((t, w2)) => (t.ptr(s, 0), w2.map_or(0, |w| w.device_ptr(s).0)),
            None => (0, 0),
        };
        let mut b = self.stream.launch_builder(f);
        b.arg(h).arg(&op).arg(w1).arg(&tp).arg(&w2p);
        b.arg(&r).arg(&d).arg(&eps).arg(&scalar).arg(&lo).arg(&hi).arg(&mode);
        b.arg(&lo).arg(&hi).arg(&lo).arg(&hi);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("add_norm (bf16): {e:?}"))
    }

    /// Per-head RMS norm (q and k with weights, v without) plus RoPE on q and k in one launch.
    /// `q: [rows, nh, hd]`, `k`/`v: [rows, nkv, hd]` receive the result. The input is `qkv`
    /// (`[rows, (nh + 2 nkv) * hd]`, columns q | k | v, i.e. a fused projection output), or, with
    /// `qkv == None`, q/k/v themselves (in place). The cos/sin tables are `[positions, hd]` and row `r`
    /// uses table row `pos[r]` (or `r` when `pos` is `None`). RoPE rotates each head in parts of `part` channels
    /// (`part == hd` for 1-D RoPE, `hd / 2` for the vision tower's 2-D RoPE).
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_post(
        &self,
        qkv: Option<&Buf>,
        q: &mut Buf,
        k: &mut Buf,
        v: &mut Buf,
        wq: &Buf,
        wk: &Buf,
        cos: &Buf,
        sin: &Buf,
        pos: Option<&CudaView<'_, i32>>,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        part: usize,
        eps: f32,
    ) -> Result<()> {
        let s = &self.stream;
        let (qp, kp, vp) = (q.device_ptr(s).0, k.device_ptr(s).0, v.device_ptr(s).0);
        let src = match qkv {
            Some(x) => {
                let (p, w) = (x.device_ptr(s).0, (nh + 2 * nkv) * hd);
                [(p, w), (p + (nh * hd * 4) as u64, w), (p + ((nh + nkv) * hd * 4) as u64, w)]
            }
            None => [(qp, nh * hd), (kp, nkv * hd), (vp, nkv * hd)],
        };
        self.launch_qkv_post(&self.k.qkv_post, src, [qp, kp, vp], [wq, wk, cos, sin], pos, rows, nh, nkv, hd, part, eps)
    }

    /// [`qkv_post`](Self::qkv_post) from a fused projection output (float32 or bf16) into activation buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_post_act(&self, qkv: &Act, q: &mut Act, k: &mut Act, v: &mut Act, tabs: [&Buf; 4], pos: Option<&CudaView<'_, i32>>, rows: usize, nh: usize, nkv: usize, hd: usize, part: usize, eps: f32) -> Result<()> {
        let s = &self.stream;
        let w = (nh + 2 * nkv) * hd;
        let src = [(qkv.ptr(s, 0), w), (qkv.ptr(s, nh * hd), w), (qkv.ptr(s, (nh + nkv) * hd), w)];
        self.qkv_post_from(qkv.is_bf16(), src, q, k, v, tabs, pos, rows, nh, nkv, hd, part, eps)
    }

    /// [`qkv_post`](Self::qkv_post) from three separate projection outputs (one storage type) into activation buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_post_split_act(&self, src: [&Act; 3], q: &mut Act, k: &mut Act, v: &mut Act, tabs: [&Buf; 4], pos: Option<&CudaView<'_, i32>>, rows: usize, nh: usize, nkv: usize, hd: usize, part: usize, eps: f32) -> Result<()> {
        let bf = src[0].is_bf16();
        ensure!(src.iter().all(|x| x.is_bf16() == bf), "internal: qkv_post inputs of mixed storage types");
        let s = &self.stream;
        let srcp = [(src[0].ptr(s, 0), nh * hd), (src[1].ptr(s, 0), nkv * hd), (src[2].ptr(s, 0), nkv * hd)];
        self.qkv_post_from(bf, srcp, q, k, v, tabs, pos, rows, nh, nkv, hd, part, eps)
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_post_from(&self, src_bf16: bool, src: [(u64, usize); 3], q: &mut Act, k: &mut Act, v: &mut Act, tabs: [&Buf; 4], pos: Option<&CudaView<'_, i32>>, rows: usize, nh: usize, nkv: usize, hd: usize, part: usize, eps: f32) -> Result<()> {
        let out_bf16 = q.is_bf16();
        ensure!(k.is_bf16() == out_bf16 && v.is_bf16() == out_bf16, "internal: qkv_post outputs of mixed storage types");
        let s = &self.stream;
        let dst = [q.ptr(s, 0), k.ptr(s, 0), v.ptr(s, 0)];
        // bf16 mode: one warp per head where the layout allows it (8 heads per 256-thread block).
        if let Some(f) = self.bf.as_ref().and_then(|bf| bf.qkv_post_warp(hd, part, src_bf16, out_bf16)) {
            let cfg = LaunchConfig { grid_dim: (((rows * (nh + 2 * nkv)) as u64).div_ceil(8).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            return self.launch_qkv_post_cfg(f, cfg, src, dst, tabs, pos, rows, nh, nkv, hd, part, eps);
        }
        let f = match (src_bf16, out_bf16) {
            (false, false) => &self.k.qkv_post,
            (false, true) => &self.bfk()?.qkv_post_fb,
            (true, true) => &self.bfk()?.qkv_post_bb,
            (true, false) => &self.bfk()?.qkv_post_bf,
        };
        self.launch_qkv_post(f, src, dst, tabs, pos, rows, nh, nkv, hd, part, eps)
    }

    /// `src`: (address, row stride in elements) of the q / k / v parts; `dst`: q / k / v; `tabs`: wq, wk, cos, sin.
    #[allow(clippy::too_many_arguments)]
    fn launch_qkv_post(&self, f: &CudaFunction, src: [(u64, usize); 3], dst: [u64; 3], tabs: [&Buf; 4], pos: Option<&CudaView<'_, i32>>, rows: usize, nh: usize, nkv: usize, hd: usize, part: usize, eps: f32) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: ((rows * (nh + 2 * nkv)) as u32, 1, 1), block_dim: (norm_threads(hd), 1, 1), shared_mem_bytes: 0 };
        self.launch_qkv_post_cfg(f, cfg, src, dst, tabs, pos, rows, nh, nkv, hd, part, eps)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_qkv_post_cfg(&self, f: &CudaFunction, cfg: LaunchConfig, src: [(u64, usize); 3], dst: [u64; 3], tabs: [&Buf; 4], pos: Option<&CudaView<'_, i32>>, rows: usize, nh: usize, nkv: usize, hd: usize, part: usize, eps: f32) -> Result<()> {
        ensure!(hd <= 512 && hd.is_multiple_of(part) && part.is_multiple_of(2), "qkv_post: unsupported head_dim {hd} / part {part}");
        if rows == 0 {
            return Ok(());
        }
        let (r, h, kv, d, p) = (rows as i32, nh as i32, nkv as i32, hd as i32, part as i32);
        let (ptrs, strides) = (src.map(|(p, _)| p), src.map(|(_, st)| st as i32));
        let null = 0u64;
        let mut b = self.stream.launch_builder(f);
        b.arg(&ptrs[0]).arg(&ptrs[1]).arg(&ptrs[2]).arg(&strides[0]).arg(&strides[1]).arg(&strides[2]);
        b.arg(&dst[0]).arg(&dst[1]).arg(&dst[2]).arg(tabs[0]).arg(tabs[1]).arg(tabs[2]).arg(tabs[3]);
        match pos {
            Some(ps) => b.arg(ps),
            None => b.arg(&null),
        };
        b.arg(&r).arg(&h).arg(&kv).arg(&d).arg(&p).arg(&eps);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("qkv_post: {e:?}"))
    }

    /// `g[..n] = gelu(g[..n]) * u[..n]`
    pub fn gelu_mul(&self, g: &mut Buf, u: &Buf, n: usize) -> Result<()> {
        let n = n as i64;
        unsafe { self.stream.launch_builder(&self.k.gelu_mul).arg(g).arg(u).arg(&n).launch(launch_1d(n as usize)) }
            .map(|_| ())
            .map_err(|e| anyhow!("gelu_mul: {e:?}"))
    }

    /// `out[r, c] = gelu(gu[r, c]) * gu[r, ff + c]` for a fused gate|up projection output `gu[rows, 2 ff]`.
    pub fn gelu_mul_gu(&self, out: &mut Buf, gu: &Buf, rows: usize, ff: usize) -> Result<()> {
        let (n, f) = ((rows * ff) as i64, ff as i32);
        unsafe { self.stream.launch_builder(&self.k.gelu_mul_gu).arg(out).arg(gu).arg(&n).arg(&f).launch(launch_1d(n as usize)) }
            .map(|_| ())
            .map_err(|e| anyhow!("gelu_mul_gu: {e:?}"))
    }

    pub fn add(&self, a: &mut Buf, b: &Buf) -> Result<()> {
        let n = a.len() as i64;
        unsafe { self.stream.launch_builder(&self.k.add).arg(a).arg(b).arg(&n).launch(launch_1d(n as usize)) }
            .map(|_| ())
            .map_err(|e| anyhow!("add: {e:?}"))
    }

    pub fn scale(&self, a: &mut Buf, s: f32) -> Result<()> {
        let n = a.len() as i64;
        unsafe { self.stream.launch_builder(&self.k.scale).arg(a).arg(&s).arg(&n).launch(launch_1d(n as usize)) }
            .map(|_| ())
            .map_err(|e| anyhow!("scale: {e:?}"))
    }

    /// `a[..n] = clamp(a[..n], lo, hi)`
    pub fn clamp(&self, a: &mut Buf, lo: f32, hi: f32, n: usize) -> Result<()> {
        let n = n as i64;
        unsafe { self.stream.launch_builder(&self.k.clamp).arg(a).arg(&lo).arg(&hi).arg(&n).launch(launch_1d(n as usize)) }
            .map(|_| ())
            .map_err(|e| anyhow!("clamp: {e:?}"))
    }

    /// New buffer holding `clamp(x[..n], lo, hi)`.
    pub fn clamp_copy(&self, x: &Buf, lo: f32, hi: f32, n: usize) -> Result<Buf> {
        let mut out = self.alloc(n)?;
        let nn = n as i64;
        unsafe { self.stream.launch_builder(&self.k.clamp_copy).arg(&mut out).arg(x).arg(&lo).arg(&hi).arg(&nn).launch(launch_1d(n)) }
            .map_err(|e| anyhow!("clamp_copy: {e:?}"))?;
        Ok(out)
    }

    pub fn bias_add(&self, y: &mut Buf, b: &Buf, rows: usize, cols: usize) -> Result<()> {
        let (r, c) = (rows as i32, cols as i32);
        unsafe { self.stream.launch_builder(&self.k.bias_add).arg(y).arg(b).arg(&r).arg(&c).launch(launch_1d(rows * cols)) }
            .map(|_| ())
            .map_err(|e| anyhow!("bias_add: {e:?}"))
    }

    pub fn ple_gate(&self, g: &mut Buf, ple: &Buf, rows: usize, p: usize, n_layers: usize, layer: usize) -> Result<()> {
        let (r, pp, nl, l) = (rows as i32, p as i32, n_layers as i32, layer as i32);
        unsafe { self.stream.launch_builder(&self.k.ple_gate).arg(g).arg(ple).arg(&r).arg(&pp).arg(&nl).arg(&l).launch(launch_1d(rows * p)) }
            .map(|_| ())
            .map_err(|e| anyhow!("ple_gate: {e:?}"))
    }

    /// [`ple_gate`](Self::ple_gate) on activation buffers: in place on a float32 `g` (`out == None`), or into the
    /// bf16 `out` (the next product's input).
    #[allow(clippy::too_many_arguments)]
    pub fn ple_gate_act(&self, g: &mut Act, out: Option<&mut Act>, ple: &Buf, rows: usize, p: usize, n_layers: usize, layer: usize) -> Result<()> {
        let o = match out {
            None => return self.ple_gate(g.f32_mut()?, ple, rows, p, n_layers, layer),
            Some(Act::Bf16(o)) => o,
            Some(Act::F32(_)) => bail!("internal: out-of-place ple_gate writes bf16"),
        };
        let f = if g.is_bf16() { &self.bfk()?.ple_gate_bb } else { &self.bfk()?.ple_gate_fb };
        let gp = g.ptr(&self.stream, 0);
        let (r, pp, nl, l) = (rows as i32, p as i32, n_layers as i32, layer as i32);
        unsafe { self.stream.launch_builder(f).arg(o).arg(&gp).arg(ple).arg(&r).arg(&pp).arg(&nl).arg(&l).launch(launch_1d(rows * p)) }
            .map(|_| ())
            .map_err(|e| anyhow!("ple_gate (bf16 out): {e:?}"))
    }

    /// `out[r, c] = gelu(gu[r, c]) * gu[r, ff + c]` for a fused gate|up output, on activation buffers.
    pub fn gelu_mul_gu_act(&self, out: &mut Act, gu: &Act, rows: usize, ff: usize) -> Result<()> {
        match (out, gu) {
            (Act::F32(o), Act::F32(gu)) => self.gelu_mul_gu(o, gu, rows, ff),
            (out, gu) => {
                let s = &self.stream;
                let (g, u) = (gu.ptr(s, 0), gu.ptr(s, ff));
                self.gelu_mul_strided(out, gu.is_bf16(), g, u, rows, ff, 2 * ff)
            }
        }
    }

    /// `out = gelu(gate) * up` for separate gate / up outputs (`rows x ff` each) into a bf16 `out`. (With
    /// float32 activations the caller uses the in-place [`gelu_mul`](Self::gelu_mul).)
    pub fn gelu_mul_split_act(&self, out: &mut Act, gate: &Act, up: &Act, rows: usize, ff: usize) -> Result<()> {
        ensure!(gate.is_bf16() == up.is_bf16(), "internal: gelu_mul inputs of mixed storage types");
        let s = &self.stream;
        let (g, u) = (gate.ptr(s, 0), up.ptr(s, 0));
        self.gelu_mul_strided(out, gate.is_bf16(), g, u, rows, ff, ff)
    }

    #[allow(clippy::too_many_arguments)]
    fn gelu_mul_strided(&self, out: &mut Act, in_bf16: bool, g: u64, u: u64, rows: usize, ff: usize, stride: usize) -> Result<()> {
        let Act::Bf16(o) = out else { bail!("internal: strided gelu_mul writes bf16") };
        let f = if in_bf16 { &self.bfk()?.gelu_mul_bb } else { &self.bfk()?.gelu_mul_fb };
        let (n, ffi, st) = ((rows * ff) as i64, ff as i32, stride as i32);
        unsafe { self.stream.launch_builder(f).arg(o).arg(&g).arg(&u).arg(&n).arg(&ffi).arg(&st).arg(&st).launch(launch_1d(rows * ff)) }
            .map(|_| ())
            .map_err(|e| anyhow!("gelu_mul (bf16 out): {e:?}"))
    }

    /// New bf16 buffer holding `bf16(clamp(x[..n], lo, hi))`.
    fn clamp_bf16_copy(&self, x: &Buf16, lo: f32, hi: f32, n: usize) -> Result<Buf16> {
        let mut out = self.alloc16(n)?;
        let nn = n as i64;
        unsafe { self.stream.launch_builder(&self.bfk()?.clamp_bf16).arg(x).arg(&mut out).arg(&lo).arg(&hi).arg(&nn).launch(launch_1d(n)) }
            .map_err(|e| anyhow!("clamp_bf16: {e:?}"))?;
        Ok(out)
    }

    // -----------------------------------------------------------------------------------------
    // Attention: batched cuBLAS GEMMs + windowed softmax
    // -----------------------------------------------------------------------------------------

    /// Bidirectional attention for one sequence of `seq` rows that starts at row `row0` of the
    /// `[*, n_heads*hd]` / `[*, n_kv*hd]` buffers; the result is written to the same rows of `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &Buf,
        k: &Buf,
        v: &Buf,
        out: &mut Buf,
        row0: usize,
        seq: usize,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        window: Option<usize>,
    ) -> Result<()> {
        if self.flash && hd == 64 && n_heads.is_multiple_of(n_kv) {
            return self.flash_attention_d64(q, k, v, out, row0, seq, n_heads, n_kv, window);
        }
        if self.precision == Precision::Bf16 {
            return self.attention_bf16(q, k, v, out, row0, seq, n_heads, n_kv, hd, window);
        }
        const Q_BLOCK: usize = 1024;
        let (qs, kvs) = (n_heads * hd, n_kv * hd);
        let group = n_heads / n_kv;
        // One batched call over every head when K/V are per head; otherwise one per KV group.
        let (n_calls, batch, kv_stride) = if group == 1 { (1, n_heads, hd) } else { (n_kv, group, 0) };
        let win = window.map(|w| w as i32).unwrap_or(-1);
        let mut q0 = 0;
        while q0 < seq {
            let qb = Q_BLOCK.min(seq - q0);
            let (k0, k1) = match window {
                Some(w) => (q0.saturating_sub(w), (q0 + qb - 1 + w + 1).min(seq)),
                None => (0, seq),
            };
            let kb = k1 - k0;
            let mut scores = self.alloc(batch * qb * kb)?;
            for call in 0..n_calls {
                let g = if group == 1 { 0 } else { call };
                let h0 = g * group;
                let k_off = (row0 + k0) * kvs + g * hd;
                let q_off = (row0 + q0) * qs + h0 * hd;
                // scores^T[kb, qb] = K_g . Q_h^T for each head h of the group.
                let cfg = StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        m: kb as i32,
                        n: qb as i32,
                        k: hd as i32,
                        alpha: 1.0f32,
                        lda: kvs as i32,
                        ldb: qs as i32,
                        beta: 0.0f32,
                        ldc: kb as i32,
                    },
                    batch_size: batch as i32,
                    stride_a: kv_stride as i64,
                    stride_b: hd as i64,
                    stride_c: (qb * kb) as i64,
                };
                unsafe { self.blas.gemm_strided_batched(cfg, &k.slice(k_off..), &q.slice(q_off..), &mut scores) }
                    .map_err(|e| anyhow!("attention QK^T: {e:?}"))?;

                self.softmax_rows(&mut scores, batch * qb, qb, kb, q0, k0, win)?;

                // out_h^T[hd, qb] = V_g^T . P_h^T
                let cfg = StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        m: hd as i32,
                        n: qb as i32,
                        k: kb as i32,
                        alpha: 1.0f32,
                        lda: kvs as i32,
                        ldb: kb as i32,
                        beta: 0.0f32,
                        ldc: qs as i32,
                    },
                    batch_size: batch as i32,
                    stride_a: kv_stride as i64,
                    stride_b: (qb * kb) as i64,
                    stride_c: hd as i64,
                };
                let v_off = (row0 + k0) * kvs + g * hd;
                unsafe { self.blas.gemm_strided_batched(cfg, &v.slice(v_off..), &scores, &mut out.slice_mut(q_off..)) }
                    .map_err(|e| anyhow!("attention PV: {e:?}"))?;
            }
            q0 += qb;
        }
        Ok(())
    }

    /// In-place windowed softmax over `nrows` score rows of length `kb` (row `r` is query `q0 + r % rows_per_head`,
    /// column `j` is key `k0 + j`; `win < 0` means no window).
    #[allow(clippy::too_many_arguments)]
    fn softmax_rows(&self, scores: &mut Buf, nrows: usize, rows_per_head: usize, kb: usize, q0: usize, k0: usize, win: i32) -> Result<()> {
        let threads = if kb >= 1024 { 256 } else { 128 };
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let (nr, rph, kbi, q0i, k0i, one) = (nrows as i32, rows_per_head as i32, kb as i32, q0 as i32, k0 as i32, 1.0f32);
        // Rows that fit in registers (<= 8 or 16 values per thread) take the single-pass kernel.
        let per_thread = kb.div_ceil(threads as usize);
        let sm = if per_thread <= 8 {
            &self.k.softmax_window_r8
        } else if per_thread <= 16 {
            &self.k.softmax_window_r16
        } else {
            &self.k.softmax_window
        };
        unsafe { self.stream.launch_builder(sm).arg(scores).arg(&nr).arg(&rph).arg(&kbi).arg(&q0i).arg(&k0i).arg(&win).arg(&one).launch(cfg) }
            .map(|_| ())
            .map_err(|e| anyhow!("softmax: {e:?}"))
    }

    /// [`softmax_rows`](Self::softmax_rows) writing the probabilities to `p` as bf16 (the PV product's
    /// input); `scores` is left holding intermediate values.
    #[allow(clippy::too_many_arguments)]
    fn softmax_rows_bf16(&self, scores: &mut Buf, p: &mut CudaSlice<u16>, nrows: usize, rows_per_head: usize, kb: usize, q0: usize, k0: usize, win: i32) -> Result<()> {
        let threads = if kb >= 1024 { 256 } else { 128 };
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let (nr, rph, kbi, q0i, k0i, one) = (nrows as i32, rows_per_head as i32, kb as i32, q0 as i32, k0 as i32, 1.0f32);
        let per_thread = kb.div_ceil(threads as usize);
        let sm = if per_thread <= 8 {
            &self.k.softmax_window_r8_bf16
        } else if per_thread <= 16 {
            &self.k.softmax_window_r16_bf16
        } else {
            &self.k.softmax_window_bf16
        };
        unsafe { self.stream.launch_builder(sm).arg(scores).arg(p).arg(&nr).arg(&rph).arg(&kbi).arg(&q0i).arg(&k0i).arg(&win).arg(&one).launch(cfg) }
            .map(|_| ())
            .map_err(|e| anyhow!("softmax (bf16 out): {e:?}"))
    }

    /// `len` floats of `x` starting at `off`, rounded to bf16 into a fresh pool buffer.
    fn to_bf16_slice(&self, x: &Buf, off: usize, len: usize) -> Result<CudaSlice<u16>> {
        let mut y = unsafe { self.stream.alloc::<u16>(len.max(1)) }.map_err(|e| anyhow!("device alloc of {len} bf16: {e:?}"))?;
        let view = x.slice(off..off + len);
        let (nn, lo, hi) = (len as i64, NO_CLAMP.0, NO_CLAMP.1);
        let cfg = LaunchConfig { grid_dim: ((len as u64).div_ceil(4 * 256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { self.stream.launch_builder(&self.k.to_bf16).arg(&view).arg(&mut y).arg(&nn).arg(&lo).arg(&hi).launch(cfg) }
            .map_err(|e| anyhow!("to_bf16: {e:?}"))?;
        Ok(y)
    }

    /// [`attention`](Self::attention) with both products on bf16 tensor cores (float32 accumulation): Q, K, V
    /// and the softmax probabilities are rounded to bf16, while the scores, the softmax and the outputs
    /// stay float32.
    #[allow(clippy::too_many_arguments)]
    fn attention_bf16(&self, q: &Buf, k: &Buf, v: &Buf, out: &mut Buf, row0: usize, seq: usize, n_heads: usize, n_kv: usize, hd: usize, window: Option<usize>) -> Result<()> {
        if seq == 0 {
            return Ok(());
        }
        let (qs, kvs) = (n_heads * hd, n_kv * hd);
        // bf16 copies of this sequence's rows.
        let q16 = self.to_bf16_slice(q, row0 * qs, seq * qs)?;
        let k16 = self.to_bf16_slice(k, row0 * kvs, seq * kvs)?;
        let v16 = self.to_bf16_slice(v, row0 * kvs, seq * kvs)?;
        self.mark("attn: q/k/v to bf16");
        let s = &self.stream;
        let op = out.device_ptr(s).0 + (row0 * qs * 4) as u64;
        self.attention_bf16_core([q16.device_ptr(s).0, k16.device_ptr(s).0, v16.device_ptr(s).0], op, false, seq, n_heads, n_kv, hd, window)
    }

    /// [`attention`](Self::attention) on activation buffers: float32 q/k/v/out take the float32 paths, bf16 ones
    /// (bf16 mode) the tensor-core kernels directly, without conversion passes.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_act(&self, q: &Act, k: &Act, v: &Act, out: &mut Act, row0: usize, seq: usize, n_heads: usize, n_kv: usize, hd: usize, window: Option<usize>) -> Result<()> {
        match (q, k, v, out) {
            (Act::F32(q), Act::F32(k), Act::F32(v), Act::F32(o)) => self.attention(q, k, v, o, row0, seq, n_heads, n_kv, hd, window),
            (q @ Act::Bf16(_), k @ Act::Bf16(_), v @ Act::Bf16(_), o @ Act::Bf16(_)) => {
                if seq == 0 {
                    return Ok(());
                }
                let s = &self.stream;
                if self.flash && (hd == 64 || hd == 256) && n_heads.is_multiple_of(n_kv) {
                    let bf = self.bfk()?;
                    let f = if hd == 64 { &bf.flash_d64_bb } else { &bf.flash_d256_bb };
                    let ptrs = [q.ptr(s, 0), k.ptr(s, 0), v.ptr(s, 0), o.ptr(s, 0)];
                    return self.flash_launch(f, ptrs, row0, seq, n_heads, n_kv, hd, window);
                }
                if self.flash && hd == 512 && n_heads.is_multiple_of(n_kv) {
                    let ptrs = [q.ptr(s, 0), k.ptr(s, 0), v.ptr(s, 0), o.ptr(s, 0)];
                    return self.flash_d512_launch(ptrs, 0, seq.div_ceil(64), row0, seq, n_heads, n_kv, window);
                }
                let (qs, kvs) = (n_heads * hd, n_kv * hd);
                let qkv = [q.ptr(s, row0 * qs), k.ptr(s, row0 * kvs), v.ptr(s, row0 * kvs)];
                self.attention_bf16_core(qkv, o.ptr(s, row0 * qs), true, seq, n_heads, n_kv, hd, window)
            }
            _ => bail!("internal: attention on activation buffers of mixed storage types"),
        }
    }

    /// The two attention products of one sequence on bf16 tensor cores. `qkv` holds the device addresses of the
    /// sequence's first q / k / v rows (bf16, row strides `n_heads * hd` / `n_kv * hd`), `out` that of its first
    /// output row (float32, or bf16 with `out_bf16`).
    #[allow(clippy::too_many_arguments)]
    fn attention_bf16_core(&self, qkv: [u64; 3], out: u64, out_bf16: bool, seq: usize, n_heads: usize, n_kv: usize, hd: usize, window: Option<usize>) -> Result<()> {
        use cudarc::cublas::result as blas;
        // Query block: a windowed layer reads qb + 2w keys per block for queries that need only 2w + 1 of
        // them, so smaller blocks waste less score work, until per-launch overhead wins. Measured on the
        // 2208-token benchmark text: 1024 -> 80.2 ms, 512 -> 78.5, 256 -> 80.1, 128 -> 83.7.
        let q_block: usize = if window.is_some() { 512 } else { 1024 };
        let (qs, kvs) = (n_heads * hd, n_kv * hd);
        let group = n_heads / n_kv;
        let (n_calls, batch, kv_stride) = if group == 1 { (1, n_heads, hd) } else { (n_kv, group, 0) };
        let win = window.map(|w| w as i32).unwrap_or(-1);
        let (qp, kp, vp) = (qkv[0], qkv[1], qkv[2]);
        let (alpha, beta) = (1.0f32, 0.0f32);
        let bf = |p: u64, elems: usize| (p + (elems * 2) as u64) as *const std::ffi::c_void;
        let handle = *self.blas.handle();
        use cublas_sys::{cublasComputeType_t::CUBLAS_COMPUTE_32F as C32F, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT as ALGO};
        use cublas_sys::{cublasOperation_t::CUBLAS_OP_N as N, cublasOperation_t::CUBLAS_OP_T as T};
        use cublas_sys::cudaDataType::{CUDA_R_16BF as BF16, CUDA_R_32F as F32};
        let (out_type, out_elem) = if out_bf16 { (BF16, 2u64) } else { (F32, 4u64) };
        let mut q0 = 0;
        while q0 < seq {
            let qb = q_block.min(seq - q0);
            let (k0, k1) = match window {
                Some(w) => (q0.saturating_sub(w), (q0 + qb - 1 + w + 1).min(seq)),
                None => (0, seq),
            };
            let kb = k1 - k0;
            let mut scores = self.alloc(batch * qb * kb)?;
            let mut p16 = self.alloc16(batch * qb * kb)?;
            for call in 0..n_calls {
                let g = if group == 1 { 0 } else { call };
                let h0 = g * group;
                {
                    // scores^T[kb, qb] = K_g . Q_h^T for each head h of the group.
                    let sp = scores.device_ptr_mut(&self.stream).0;
                    unsafe {
                        blas::gemm_strided_batched_ex(
                            handle, T, N, kb as i32, qb as i32, hd as i32,
                            &alpha as *const f32 as *const _,
                            bf(kp, k0 * kvs + g * hd), BF16, kvs as i32, kv_stride as i64,
                            bf(qp, q0 * qs + h0 * hd), BF16, qs as i32, hd as i64,
                            &beta as *const f32 as *const _,
                            sp as *mut _, F32, kb as i32, (qb * kb) as i64,
                            batch as i32, C32F, ALGO,
                        )
                    }
                    .map_err(|e| anyhow!("attention QK^T (bf16): {e:?}"))?;
                }
                self.mark("attn: QK^T");
                self.softmax_rows_bf16(&mut scores, &mut p16, batch * qb, qb, kb, q0, k0, win)?;
                self.mark("attn: softmax");
                {
                    // out_h^T[hd, qb] = V_g^T . P_h^T, written straight into this sequence's rows of `out`.
                    let pp = p16.device_ptr(&self.stream).0;
                    let o_off = (q0 * qs + h0 * hd) as u64;
                    unsafe {
                        blas::gemm_strided_batched_ex(
                            handle, N, N, hd as i32, qb as i32, kb as i32,
                            &alpha as *const f32 as *const _,
                            bf(vp, k0 * kvs + g * hd), BF16, kvs as i32, kv_stride as i64,
                            pp as *const _, BF16, kb as i32, (qb * kb) as i64,
                            &beta as *const f32 as *const _,
                            (out + o_off * out_elem) as *mut _, out_type, qs as i32, hd as i64,
                            batch as i32, C32F, ALGO,
                        )
                    }
                    .map_err(|e| anyhow!("attention PV (bf16): {e:?}"))?;
                }
                self.mark("attn: PV");
            }
            q0 += qb;
        }
        Ok(())
    }

    /// `attention` for head_dim 64 as one fused launch (scores stay on chip), see `flash_attn_d64`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_d64(&self, q: &Buf, k: &Buf, v: &Buf, out: &mut Buf, row0: usize, seq: usize, n_heads: usize, n_kv: usize, window: Option<usize>) -> Result<()> {
        if seq == 0 {
            return Ok(());
        }
        if let Some(bf) = &self.bf {
            // bf16 mode: the same contract on tensor cores.
            let s = &self.stream;
            let ptrs = [q.device_ptr(s).0, k.device_ptr(s).0, v.device_ptr(s).0, out.device_ptr(s).0];
            return self.flash_launch(&bf.flash_d64, ptrs, row0, seq, n_heads, n_kv, 64, window);
        }
        let cfg = LaunchConfig { grid_dim: (seq.div_ceil(64) as u32, n_heads as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (r0, sq, nh, nkv, win, scale) = (row0 as i32, seq as i32, n_heads as i32, n_kv as i32, window.map(|w| w as i32).unwrap_or(-1), 1.0f32);
        let (qs, kvs) = ((n_heads * 64) as i32, (n_kv * 64) as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.flash_attn_d64)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(out)
                .arg(&r0)
                .arg(&sq)
                .arg(&nh)
                .arg(&nkv)
                .arg(&win)
                .arg(&scale)
                .arg(&qs)
                .arg(&kvs)
                .launch(cfg)
        }
        .map(|_| ())
        .map_err(|e| anyhow!("flash_attn_d64: {e:?}"))
    }

    /// Launch a tensor-core flash attention kernel (`flash_attn_d64_bf16` / `_bb`, `flash_attn_d256_bb`: 64 queries
    /// per block of 4 warps) on raw q / k / v / out addresses (buffer starts; the sequence begins at row `row0`).
    #[allow(clippy::too_many_arguments)]
    fn flash_launch(&self, f: &CudaFunction, ptrs: [u64; 4], row0: usize, seq: usize, n_heads: usize, n_kv: usize, hd: usize, window: Option<usize>) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (seq.div_ceil(64) as u32, n_heads as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (r0, sq, nh, nkv, win, scale) = (row0 as i32, seq as i32, n_heads as i32, n_kv as i32, window.map(|w| w as i32).unwrap_or(-1), 1.0f32);
        let (qs, kvs) = ((n_heads * hd) as i32, (n_kv * hd) as i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&ptrs[0]).arg(&ptrs[1]).arg(&ptrs[2]).arg(&ptrs[3]);
        b.arg(&r0).arg(&sq).arg(&nh).arg(&nkv).arg(&win).arg(&scale).arg(&qs).arg(&kvs);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("flash attention (tensor cores): {e:?}"))
    }

    /// Launch `flash_attn_d512_bb` (head_dim 512: 8 warps = 4 row groups x 2 channel halves per block of 64 queries, see
    /// kernels_sm80.cu) on raw q / k / v / out addresses. With `tiles == 0` the sequence occupies rows `row0` ..
    /// `row0 + seq` and `grid_x` = ceil(seq / 64) blocks cover it; otherwise `tiles` is the device address of
    /// `grid_x` int triples {row0, len, q0} (one 64-query tile each; `row0` / `seq` are then ignored).
    #[allow(clippy::too_many_arguments)]
    fn flash_d512_launch(&self, ptrs: [u64; 4], tiles: u64, grid_x: usize, row0: usize, seq: usize, n_heads: usize, n_kv: usize, window: Option<usize>) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (grid_x as u32, n_heads as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: FLASH_D512_SMEM as u32 };
        let (r0, sq, nh, nkv, win, scale) = (row0 as i32, seq as i32, n_heads as i32, n_kv as i32, window.map(|w| w as i32).unwrap_or(-1), 1.0f32);
        let (qs, kvs) = ((n_heads * 512) as i32, (n_kv * 512) as i32);
        let mut b = self.stream.launch_builder(&self.bfk()?.flash_d512_bb);
        b.arg(&ptrs[0]).arg(&ptrs[1]).arg(&ptrs[2]).arg(&ptrs[3]).arg(&tiles);
        b.arg(&r0).arg(&sq).arg(&nh).arg(&nkv).arg(&win).arg(&scale).arg(&qs).arg(&kvs);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("flash attention, head_dim 512 (tensor cores): {e:?}"))
    }

    /// Registers per thread, local-memory (spill) bytes per thread and dynamic shared memory bytes of the head_dim 512
    /// flash kernel (bf16 mode), for `examples/check_attention.rs`.
    #[doc(hidden)]
    pub fn flash_d512_info(&self) -> Result<(i32, i32, usize)> {
        let f = &self.bfk()?.flash_d512_bb;
        let regs = f.num_regs().map_err(|e| anyhow!("{e:?}"))?;
        let local = f.local_size_bytes().map_err(|e| anyhow!("{e:?}"))?;
        Ok((regs, local, FLASH_D512_SMEM))
    }

    /// Overwrite the start of `dst` with `x` (host -> existing device buffer).
    pub fn upload_into(&self, x: &[f32], dst: &mut Buf) -> Result<()> {
        let mut view = dst.slice_mut(0..x.len());
        self.stream.memcpy_htod(x, &mut view).map_err(|e| anyhow!("host->device copy: {e:?}"))
    }

    /// Overwrite the start of `dst` with `x` (`i32` flavour of [`upload_into`](Self::upload_into)).
    pub fn upload_i32_into(&self, x: &[i32], dst: &mut CudaSlice<i32>) -> Result<()> {
        let mut view = dst.slice_mut(0..x.len());
        self.stream.memcpy_htod(x, &mut view).map_err(|e| anyhow!("host->device copy: {e:?}"))
    }

    /// Copy the first `n` floats of `x` to the host.
    pub fn download_prefix(&self, x: &Buf, n: usize) -> Result<Vec<f32>> {
        let view = x.slice(0..n);
        self.stream.clone_dtoh(&view).map_err(|e| anyhow!("device->host copy: {e:?}"))
    }

    /// `dst[..src.len()] = src` on the device.
    pub fn copy_dtod(&self, src: &Buf, dst: &mut Buf) -> Result<()> {
        let mut view = dst.slice_mut(0..src.len());
        self.stream.memcpy_dtod(src, &mut view).map_err(|e| anyhow!("device copy: {e:?}"))
    }

    /// Record every launch `f` makes on the stream into a graph instead of running it. The graph captures
    /// the buffers' device addresses and the launch parameters, so it can be replayed (with different
    /// *contents* in those buffers) for a fraction of the host cost of the individual launches.
    pub fn capture<F: FnOnce() -> Result<()>>(&self, f: F) -> Result<Graph> {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
        self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).map_err(|e| anyhow!("begin capture: {e:?}"))?;
        let run = f();
        let graph = self.stream.end_capture(CUgraphInstantiate_flags(0)).map_err(|e| anyhow!("end capture: {e:?}"));
        run?;
        let graph = graph?.ok_or_else(|| anyhow!("stream capture produced no graph"))?;
        graph.upload().map_err(|e| anyhow!("graph upload: {e:?}"))?;
        Ok(Graph(graph))
    }

    /// Launch a captured [`Graph`] on the stream.
    pub fn replay(&self, g: &Graph) -> Result<()> {
        g.0.launch().map_err(|e| anyhow!("graph launch: {e:?}"))
    }

    /// Upload `i32` metadata (e.g. sequence offsets) to the device.
    pub fn upload_i32(&self, x: &[i32]) -> Result<CudaSlice<i32>> {
        self.stream.clone_htod(x).map_err(|e| anyhow!("host->device copy: {e:?}"))
    }

    /// Whether [`attention_packed_act`](Self::attention_packed_act) has a kernel for this head layout.
    pub fn packed_attention_supported(&self, n_heads: usize, n_kv: usize, hd: usize) -> bool {
        n_kv > 0 && n_heads.is_multiple_of(n_kv) && hd.is_multiple_of(32) && self.packed_kernel(n_heads / n_kv, hd / 32, n_heads / n_kv).is_some()
    }

    /// The packed-attention kernel for `group` query heads per K/V head, `per = head_dim / 32`, and `hw` heads per
    /// warp: float32 output, or in bf16 mode bf16 output (the only kind that mode compiles).
    fn packed_kernel(&self, group: usize, per: usize, hw: usize) -> Option<&CudaFunction> {
        let table = match &self.bf {
            Some(bf) => &bf.attn_packed_fb,
            None => &self.k.attn_packed,
        };
        table.iter().find(|(key, _)| *key == (per, group, hw)).map(|(_, f)| f)
    }

    /// Fused attention over many short sequences packed back to back (one launch per layer).
    /// `seg_start[t]` / `seg_len[t]` give the sequence of row `t`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_packed(
        &self,
        q: &Buf,
        k: &Buf,
        v: &Buf,
        out: &mut Buf,
        seg_start: &CudaView<'_, i32>,
        seg_len: &CudaView<'_, i32>,
        rows: usize,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        window: Option<usize>,
    ) -> Result<()> {
        ensure!(self.bf.is_none(), "internal: float32 packed attention in bf16 mode");
        ensure!(self.packed_attention_supported(n_heads, n_kv, hd), "attn_packed: unsupported layout ({n_heads} heads, {n_kv} kv heads, head_dim {hd})");
        // With few rows, one warp per query head (instead of per K/V head) gives the GPU more to chew on.
        let group = n_heads / n_kv;
        let hw = if rows * n_kv <= PACKED_SPLIT_UP_TO { 1 } else { group };
        let f = self.packed_kernel(group, hd / 32, hw).unwrap();
        let (r, kv) = (rows as i32, n_kv as i32);
        let win = window.map(|w| w as i32).unwrap_or(-1);
        unsafe { self.stream.launch_builder(f).arg(q).arg(k).arg(v).arg(out).arg(seg_start).arg(seg_len).arg(&r).arg(&kv).arg(&win).launch(launch_1d(rows * n_kv * (group / hw) * 32)) }
            .map(|_| ())
            .map_err(|e| anyhow!("attn_packed: {e:?}"))
    }

    /// [`attention_packed`](Self::attention_packed) on activation buffers: float32 q/k/v, and a float32 or (bf16
    /// mode) bf16 output. The arithmetic is float32 either way.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_packed_act(
        &self,
        q: &Act,
        k: &Act,
        v: &Act,
        out: &mut Act,
        seg_start: &CudaView<'_, i32>,
        seg_len: &CudaView<'_, i32>,
        rows: usize,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        window: Option<usize>,
    ) -> Result<()> {
        ensure!(!(q.is_bf16() || k.is_bf16() || v.is_bf16()), "internal: packed attention takes float32 q/k/v");
        if !out.is_bf16() {
            return self.attention_packed(q.f32()?, k.f32()?, v.f32()?, out.f32_mut()?, seg_start, seg_len, rows, n_heads, n_kv, hd, window);
        }
        self.bfk()?;
        ensure!(self.packed_attention_supported(n_heads, n_kv, hd), "attn_packed: unsupported layout ({n_heads} heads, {n_kv} kv heads, head_dim {hd})");
        let group = n_heads / n_kv;
        let hw = if rows * n_kv <= PACKED_SPLIT_UP_TO { 1 } else { group };
        let f = self.packed_kernel(group, hd / 32, hw).ok_or_else(|| anyhow!("attn_packed: no kernel for this layout"))?;
        let s = &self.stream;
        let ptrs = [q.ptr(s, 0), k.ptr(s, 0), v.ptr(s, 0), out.ptr(s, 0)];
        let (r, kv) = (rows as i32, n_kv as i32);
        let win = window.map(|w| w as i32).unwrap_or(-1);
        let mut b = self.stream.launch_builder(f);
        b.arg(&ptrs[0]).arg(&ptrs[1]).arg(&ptrs[2]).arg(&ptrs[3]).arg(seg_start).arg(seg_len).arg(&r).arg(&kv).arg(&win);
        unsafe { b.launch(launch_1d(rows * n_kv * (group / hw) * 32)) }.map(|_| ()).map_err(|e| anyhow!("attn_packed (bf16): {e:?}"))
    }

    /// The tensor-core kernel that runs a packed pass's attention for head_dim `hd` (bf16 mode with flash attention
    /// on), if there is one: the single place that decides which head widths take the flash path. Other widths stay
    /// on the float32 [`attention_packed_act`](Self::attention_packed_act) kernel.
    fn packed_flash_kernel(&self, hd: usize) -> Option<&CudaFunction> {
        let bf = self.bf.as_ref().filter(|_| self.flash)?;
        match hd {
            256 => Some(&bf.flash_d256_packed_bb),
            512 => Some(&bf.flash_d512_bb),
            _ => None,
        }
    }

    /// Whether [`attention_flash_packed`](Self::attention_flash_packed) has a kernel for this head layout. Layers
    /// that do not run on it use [`attention_packed_act`](Self::attention_packed_act) (float32 q/k/v), those that do
    /// need bf16 q/k/v.
    pub fn packed_flash_supported(&self, n_heads: usize, n_kv: usize, hd: usize) -> bool {
        n_kv > 0 && n_heads.is_multiple_of(n_kv) && self.packed_flash_kernel(hd).is_some()
    }

    /// Tile list for [`attention_flash_packed`](Self::attention_flash_packed): sequences of `lens` rows packed back
    /// to back from row `row0` become int triples {row0, len, q0}, one per [`FLASH_TILE_ROWS`] queries (q0 = the
    /// tile's first query within its sequence), followed by len-0 padding tiles up to at least `pad_to` tiles (a
    /// recorded CUDA graph has a fixed grid). The list never needs more than one tile per row.
    pub fn flash_packed_tiles(row0: usize, lens: &[usize], pad_to: usize) -> Vec<i32> {
        let mut tiles: Vec<i32> = Vec::with_capacity(3 * pad_to.max(lens.len()));
        let mut start = row0;
        for &len in lens {
            for q0 in (0..len).step_by(FLASH_TILE_ROWS) {
                tiles.extend([start as i32, len as i32, q0 as i32]);
            }
            start += len;
        }
        tiles.resize(tiles.len().max(3 * pad_to), 0);
        tiles
    }

    /// Bidirectional attention over many short sequences packed back to back, as one tensor-core launch (the packed
    /// counterpart of [`attention_act`](Self::attention_act)'s flash path; bf16 mode, bf16 q/k/v/out). `tiles` holds
    /// at least `n_tiles` tiles from [`flash_packed_tiles`](Self::flash_packed_tiles); tiles of length 0 do nothing,
    /// rows outside the listed sequences are not touched.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_flash_packed(
        &self,
        q: &Act,
        k: &Act,
        v: &Act,
        out: &mut Act,
        tiles: &CudaView<'_, i32>,
        n_tiles: usize,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        window: Option<usize>,
    ) -> Result<()> {
        ensure!(q.is_bf16() && k.is_bf16() && v.is_bf16() && out.is_bf16(), "internal: packed flash attention takes bf16 q/k/v/out");
        ensure!(self.packed_flash_supported(n_heads, n_kv, hd), "packed flash attention: unsupported layout ({n_heads} heads, {n_kv} kv heads, head_dim {hd})");
        ensure!(tiles.len() >= 3 * n_tiles, "internal: tile list of {} ints holds fewer than {n_tiles} tiles", tiles.len());
        let f = self.packed_flash_kernel(hd).ok_or_else(|| anyhow!("packed flash attention: no kernel for head_dim {hd}"))?;
        if n_tiles == 0 {
            return Ok(());
        }
        let s = &self.stream;
        let ptrs = [q.ptr(s, 0), k.ptr(s, 0), v.ptr(s, 0), out.ptr(s, 0)];
        if hd == 512 {
            // Its own block shape and dynamic shared memory; the same tile-list contract.
            return self.flash_d512_launch(ptrs, tiles.device_ptr(s).0, n_tiles, 0, 0, n_heads, n_kv, window);
        }
        let cfg = LaunchConfig { grid_dim: (n_tiles as u32, n_heads as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        // row0 / seq are unused in tile mode.
        let (r0, sq, nh, nkv, win, scale) = (0i32, 0i32, n_heads as i32, n_kv as i32, window.map(|w| w as i32).unwrap_or(-1), 1.0f32);
        let (qs, kvs) = ((n_heads * hd) as i32, (n_kv * hd) as i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&ptrs[0]).arg(&ptrs[1]).arg(&ptrs[2]).arg(&ptrs[3]).arg(tiles);
        b.arg(&r0).arg(&sq).arg(&nh).arg(&nkv).arg(&win).arg(&scale).arg(&qs).arg(&kvs);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| anyhow!("packed flash attention: {e:?}"))
    }

    // -----------------------------------------------------------------------------------------
    // Audio kernels
    // -----------------------------------------------------------------------------------------

    /// `x[..n] = clamp(silu(clamp(x, c1)), c2)`
    pub fn act_chain(&self, x: &mut Buf, n: usize, c1: (f32, f32), c2: (f32, f32)) -> Result<()> {
        let nn = n as i64;
        unsafe { self.stream.launch_builder(&self.k.act_chain).arg(x).arg(&nn).arg(&c1.0).arg(&c1.1).arg(&c2.0).arg(&c2.1).launch(launch_1d(n)) }
            .map(|_| ())
            .map_err(|e| anyhow!("act_chain: {e:?}"))
    }

    /// Split a fused q|k|v projection output `qkv[t, 3h]` into `q`, `k`, `v` (`[t, h]`) applying the
    /// output clamps and the conformer attention scalings (`q *= qscale * dim_scale`, `k *= kscale`).
    #[allow(clippy::too_many_arguments)]
    pub fn audio_qkv_post(
        &self,
        qkv: &Buf,
        q: &mut Buf,
        k: &mut Buf,
        v: &mut Buf,
        dim_scale: &Buf,
        t: usize,
        h: usize,
        hd: usize,
        qscale: f32,
        kscale: f32,
        clip: [(f32, f32); 3],
    ) -> Result<()> {
        let n = (t * h) as i64;
        let (hh, d) = (h as i32, hd as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.audio_qkv_post)
                .arg(qkv)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(dim_scale)
                .arg(&n)
                .arg(&hh)
                .arg(&d)
                .arg(&qscale)
                .arg(&kscale)
                .arg(&clip[0].0)
                .arg(&clip[0].1)
                .arg(&clip[1].0)
                .arg(&clip[1].1)
                .arg(&clip[2].0)
                .arg(&clip[2].1)
                .launch(launch_1d(t * h))
        }
        .map(|_| ())
        .map_err(|e| anyhow!("audio_qkv_post: {e:?}"))
    }

    pub fn audio_qscale(&self, q: &mut Buf, dim_scale: &Buf, qscale: f32, rows: usize, nh: usize, hd: usize) -> Result<()> {
        let (r, h, d) = (rows as i32, nh as i32, hd as i32);
        unsafe {
            self.stream.launch_builder(&self.k.audio_qscale).arg(q).arg(dim_scale).arg(&qscale).arg(&r).arg(&h).arg(&d).launch(launch_1d(rows * nh * hd))
        }
        .map(|_| ())
        .map_err(|e| anyhow!("audio_qscale: {e:?}"))
    }

    /// `qrel[t, h, p] = q[t, h, :] . rel[p, h, :]` into `out` (`t * nh * npos` floats).
    #[allow(clippy::too_many_arguments)]
    pub fn audio_qrel(&self, q: &Buf, rel: &Buf, out: &mut Buf, t: usize, nh: usize, hd: usize, npos: usize) -> Result<()> {
        let (tt, h, d, p) = (t as i32, nh as i32, hd as i32, npos as i32);
        unsafe { self.stream.launch_builder(&self.k.audio_qrel).arg(q).arg(rel).arg(out).arg(&tt).arg(&h).arg(&d).arg(&p).launch(launch_1d(t * nh * npos * 32)) }
            .map(|_| ())
            .map_err(|e| anyhow!("audio_qrel: {e:?}"))
    }

    /// Chunked local attention with relative-position bias into `out` (`t * nh * hd` floats), clamped to `clamp`.
    #[allow(clippy::too_many_arguments)]
    pub fn audio_attn(
        &self,
        q: &Buf,
        k: &Buf,
        v: &Buf,
        qrel: &Buf,
        out: &mut Buf,
        t: usize,
        nh: usize,
        hd: usize,
        chunk: usize,
        past: usize,
        ctx: usize,
        npos: usize,
        cap: f32,
        invalid: f32,
        clamp: (f32, f32),
    ) -> Result<()> {
        ensure!(ctx <= 64, "audio attention context {ctx} exceeds the kernel limit of 64");
        let a = [t, nh, hd, chunk, past, ctx, npos].map(|v| v as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.audio_attn)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(qrel)
                .arg(out)
                .arg(&a[0])
                .arg(&a[1])
                .arg(&a[2])
                .arg(&a[3])
                .arg(&a[4])
                .arg(&a[5])
                .arg(&a[6])
                .arg(&cap)
                .arg(&invalid)
                .arg(&clamp.0)
                .arg(&clamp.1)
                .launch(launch_1d(t * nh * 32))
        }
        .map(|_| ())
        .map_err(|e| anyhow!("audio_attn: {e:?}"))
    }

    /// `out[t, c] = in[t, c] * sigmoid(in[t, h + c])` with both inputs clamped to `clamp` first.
    pub fn glu(&self, out: &mut Buf, x: &Buf, rows: usize, h: usize, clamp: (f32, f32)) -> Result<()> {
        let (r, hh) = (rows as i32, h as i32);
        unsafe { self.stream.launch_builder(&self.k.glu).arg(out).arg(x).arg(&r).arg(&hh).arg(&clamp.0).arg(&clamp.1).launch(launch_1d(rows * h)) }
            .map(|_| ())
            .map_err(|e| anyhow!("glu: {e:?}"))
    }

    /// Causal depthwise convolution (kernel `k`, then clamp to +-`gc`) into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn dwconv_causal(&self, out: &mut Buf, x: &Buf, w: &Buf, t: usize, c: usize, k: usize, gc: f32) -> Result<()> {
        let (tt, cc, kk) = (t as i32, c as i32, k as i32);
        unsafe { self.stream.launch_builder(&self.k.dwconv_causal).arg(out).arg(x).arg(w).arg(&tt).arg(&cc).arg(&kk).arg(&gc).launch(launch_1d(t * c)) }
            .map(|_| ())
            .map_err(|e| anyhow!("dwconv: {e:?}"))
    }

    /// First subsampling convolution of the audio tower (3x3, stride 2, pad 1, one input channel) + LayerNorm +
    /// ReLU: `mel [frames, n_mels]` -> `out [t1 * w1, c0]`.
    #[allow(clippy::too_many_arguments)]
    pub fn audio_conv0(&self, mel: &Buf, w: &Buf, norm_w: &Buf, out: &mut Buf, frames: usize, n_mels: usize, t1: usize, w1: usize, c0: usize, eps: f32) -> Result<()> {
        ensure!(c0 <= 128, "audio_conv0 supports at most 128 channels (got {c0})");
        let a = [frames, n_mels, t1, w1, c0].map(|v| v as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.audio_conv0)
                .arg(mel)
                .arg(w)
                .arg(norm_w)
                .arg(out)
                .arg(&a[0])
                .arg(&a[1])
                .arg(&a[2])
                .arg(&a[3])
                .arg(&a[4])
                .arg(&eps)
                .launch(launch_1d(t1 * w1 * 32))
        }
        .map(|_| ())
        .map_err(|e| anyhow!("audio_conv0: {e:?}"))
    }

    /// im2col of `y0 [t1 * w1, c0]` for the second subsampling convolution -> `col [t2 * w2, 9 * c0]`.
    #[allow(clippy::too_many_arguments)]
    pub fn audio_im2col1(&self, y0: &Buf, col: &mut Buf, t1: usize, w1: usize, c0: usize, t2: usize, w2: usize) -> Result<()> {
        let a = [t1, w1, c0, t2, w2].map(|v| v as i32);
        unsafe { self.stream.launch_builder(&self.k.audio_im2col1).arg(y0).arg(col).arg(&a[0]).arg(&a[1]).arg(&a[2]).arg(&a[3]).arg(&a[4]).launch(launch_1d(t2 * w2 * 9 * c0)) }
            .map(|_| ())
            .map_err(|e| anyhow!("audio_im2col1: {e:?}"))
    }

    /// In-place LayerNorm (no bias) over rows of `dim <= 128` channels followed by ReLU.
    pub fn ln_relu(&self, x: &mut Buf, norm_w: &Buf, rows: usize, dim: usize, eps: f32) -> Result<()> {
        ensure!(dim <= 128, "ln_relu supports at most 128 channels (got {dim})");
        let (r, d) = (rows as i64, dim as i32);
        unsafe { self.stream.launch_builder(&self.k.ln_relu).arg(x).arg(norm_w).arg(&r).arg(&d).arg(&eps).launch(launch_1d(rows * 32)) }
            .map(|_| ())
            .map_err(|e| anyhow!("ln_relu: {e:?}"))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(|e| anyhow!("{e:?}"))
    }
}

/// Weights of a [`GpuLinear`]: float32, or bfloat16 bits for the [`Precision::Bf16`] tensor-core path.
pub enum LinW {
    F32(Buf),
    Bf16(CudaSlice<u16>),
}

impl LinW {
    /// Store `w` in the device's linear-layer format.
    fn new(gpu: &Gpu, w: &[f32]) -> Result<Self> {
        Ok(match gpu.precision {
            Precision::Bf16 => LinW::Bf16(gpu.upload_bf16(w)?),
            _ => LinW::F32(gpu.upload(w)?),
        })
    }
}

/// `nn.Linear` resident on the GPU, with optional `Gemma4ClippableLinear` clamping.
pub struct GpuLinear {
    pub w: LinW,
    pub out: usize,
    pub inp: usize,
    pub bias: Option<Buf>,
    pub clip: Option<[f32; 4]>,
}

impl GpuLinear {
    pub fn from_cpu(gpu: &Gpu, l: &Linear) -> Result<Self> {
        Ok(Self {
            w: LinW::new(gpu, &l.w)?,
            out: l.out,
            inp: l.inp,
            bias: l.bias.as_ref().map(|b| gpu.upload(b)).transpose()?,
            clip: None,
        })
    }

    /// Like [`from_cpu`](Self::from_cpu) but always float32, whatever the device precision (for small
    /// layers that are kept exact, e.g. the audio front end and the relative-position table).
    pub fn from_cpu_f32(gpu: &Gpu, l: &Linear) -> Result<Self> {
        let mut g = Self::from_cpu(gpu, l)?;
        if let LinW::Bf16(_) = g.w {
            g.w = LinW::F32(gpu.upload(&l.w)?);
        }
        Ok(g)
    }

    /// Several bias-free linears with the same input width stacked into one (`[sum(out), inp]`), so one
    /// GEMM produces all of their outputs side by side (columns in the order given).
    pub fn concat(gpu: &Gpu, parts: &[&Linear]) -> Result<Self> {
        let inp = parts[0].inp;
        ensure!(parts.iter().all(|l| l.inp == inp && l.bias.is_none()), "GpuLinear::concat needs bias-free linears with equal input width");
        let mut w = Vec::with_capacity(parts.iter().map(|l| l.w.len()).sum());
        for l in parts {
            w.extend_from_slice(&l.w);
        }
        Ok(Self { w: LinW::new(gpu, &w)?, out: parts.iter().map(|l| l.out).sum(), inp, bias: None, clip: None })
    }

    /// `y = clamp(x) W^T` for either weight format (`clamp` is applied to the input on the bf16 path, where
    /// it is folded into the conversion; the float32 path never clamps here).
    fn product(&self, gpu: &Gpu, x: &Buf, m: usize, y: &mut Buf, clamp: (f32, f32)) -> Result<()> {
        match &self.w {
            LinW::F32(w) => gpu.matmul_nt(x, m, self.inp, w, self.out, y),
            LinW::Bf16(w) => gpu.matmul_nt_bf16(x, m, self.inp, w, self.out, y, clamp),
        }
    }

    pub fn from_clippable(gpu: &Gpu, l: &ClippableLinear) -> Result<Self> {
        let mut g = Self::from_cpu(gpu, &l.linear)?;
        g.clip = l.clip;
        Ok(g)
    }

    /// `y[m, out] = x[m, inp] W^T (+ b)`, with input/output clamping when configured.
    pub fn forward(&self, gpu: &Gpu, x: &Buf, m: usize) -> Result<Buf> {
        let mut y = gpu.alloc(m * self.out)?;
        self.forward_into(gpu, x, m, &mut y)?;
        Ok(y)
    }

    /// The plain product `y = x W^T (+ b)` into a caller-owned buffer: no clamping, even when the
    /// layer has clip bounds (the caller has folded those into neighbouring kernels).
    pub fn forward_raw(&self, gpu: &Gpu, x: &Buf, m: usize, y: &mut Buf) -> Result<()> {
        self.product(gpu, x, m, y, NO_CLAMP)?;
        if let Some(b) = &self.bias {
            gpu.bias_add(y, b, m, self.out)?;
        }
        Ok(())
    }

    /// Like [`forward`](Self::forward) into a caller-owned buffer of at least `m * out` floats.
    pub fn forward_into(&self, gpu: &Gpu, x: &Buf, m: usize, y: &mut Buf) -> Result<()> {
        match (self.clip, &self.w) {
            (Some([imin, imax, _, _]), LinW::Bf16(_)) => self.product(gpu, x, m, y, (imin, imax))?,
            (Some([imin, imax, _, _]), LinW::F32(_)) => {
                let xc = gpu.clamp_copy(x, imin, imax, m * self.inp)?;
                self.product(gpu, &xc, m, y, NO_CLAMP)?;
            }
            (None, _) => self.product(gpu, x, m, y, NO_CLAMP)?,
        }
        if let Some(b) = &self.bias {
            gpu.bias_add(y, b, m, self.out)?;
        }
        if let Some([_, _, omin, omax]) = self.clip {
            gpu.clamp(y, omin, omax, m * self.out)?;
        }
        Ok(())
    }

    /// [`forward_into`](Self::forward_into) on activation buffers. A bf16 input (bf16 mode) goes straight into
    /// the tensor-core product; the result is stored like `y` (float32, or bf16 for a layer without bias and
    /// output clamp).
    pub fn forward_act(&self, gpu: &Gpu, x: &Act, m: usize, y: &mut Act) -> Result<()> {
        let x16 = match x {
            Act::F32(x) => return self.forward_into(gpu, x, m, y.f32_mut()?),
            Act::Bf16(x) => x,
        };
        let LinW::Bf16(w) = &self.w else { bail!("internal: bf16 activations with float32 weights") };
        let s = &gpu.stream;
        // An input clamp has to be applied to the bf16 values. (In practice only the audio tower clips, and it
        // keeps float32 activations.)
        let clamped;
        let xp = match self.clip {
            Some([lo, hi, _, _]) => {
                clamped = gpu.clamp_bf16_copy(x16, lo, hi, m * self.inp)?;
                clamped.device_ptr(s).0
            }
            None => x16.device_ptr(s).0,
        };
        match y {
            Act::F32(yb) => {
                gpu.gemm_bf16(xp, m, self.inp, w, self.out, yb.device_ptr(s).0, false)?;
                if let Some(b) = &self.bias {
                    gpu.bias_add(yb, b, m, self.out)?;
                }
                if let Some([_, _, omin, omax]) = self.clip {
                    gpu.clamp(yb, omin, omax, m * self.out)?;
                }
                Ok(())
            }
            Act::Bf16(yb) => {
                ensure!(self.bias.is_none() && self.clip.is_none(), "internal: bf16 output of a linear layer with bias or clamp");
                gpu.gemm_bf16(xp, m, self.inp, w, self.out, yb.device_ptr(s).0, true)
            }
        }
    }
}

/// Probe for a usable CUDA device without failing hard.
pub fn try_open(ordinal: usize, precision: Precision) -> Result<Gpu> {
    // With dynamic loading, cudarc panics (instead of returning an error) when a CUDA library
    // cannot be loaded, which would defeat `--device auto`'s CPU fallback. Probe first; the
    // catch_unwind is a backstop for anything the probe cannot see.
    let present = |name: &str, found: bool| if found { Ok(()) } else { Err(anyhow!("{name} not found")) };
    unsafe {
        present("NVIDIA driver (nvcuda)", cudarc::driver::sys::is_culib_present())
            .and(present("cuBLAS 13.x (cublas64_13.dll)", cudarc::cublas::sys::is_culib_present()))
            .and(present("NVRTC 13.x (nvrtc64_130_0.dll)", cudarc::nvrtc::sys::is_culib_present()))
    }
    .context("CUDA libraries are not available (install the NVIDIA driver and CUDA 13.x runtime, and put its bin directory on PATH)")?;
    match std::panic::catch_unwind(|| Gpu::new(ordinal, precision)) {
        Ok(gpu) => gpu.context("initialising CUDA (driver, cuBLAS and NVRTC must be installed)"),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            Err(anyhow!("initialising CUDA failed: {msg}"))
        }
    }
}

