//! Text backbone on the GPU (mirrors `crate::text::TextModel::forward_packed`).
//!
//! A forward pass is launch-bound for short inputs (24 layers, and every launch or allocation costs
//! host time), so the layer is built from few, fused launches working in scratch buffers that are
//! allocated once per pass: the residual tails and the per-head norm + RoPE steps are single kernels
//! (see `kernels.cu`), and the RoPE tables are cached on the device across calls.
//!
//! Short inputs (a query, a few short documents) are dominated by per-launch host cost (about 9 us per
//! kernel on the Windows/WDDM test machine, ~370 launches per pass), so those passes are recorded once
//! into a CUDA graph per padded row count and replayed: the graph's buffers keep their addresses and only
//! their contents (embeddings, sequence metadata) change between calls (`EGEMMA2_NO_GRAPH` turns this off).
//!
//! Attention over packed short sequences (a "fused" pass) is one launch per layer. In bf16 mode both the 256-wide
//! sliding-window and the 512-wide global layers run it on the tensor-core flash kernels with a per-pass tile list
//! (see `PackedFlash`); float32 and TF32 use the float32 packed kernel.

use anyhow::Result;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;

use super::{Act, Buf, Gpu, GpuLinear, Graph};
use crate::config::TextConfig;
use crate::ops;
use crate::text::{self, TextModel};
use crate::weights::Weights;

struct Layer {
    head_dim: usize,
    n_kv: usize,
    window: Option<usize>,
    theta: f64,
    input_ln: Buf,
    post_attn_ln: Buf,
    pre_ff_ln: Buf,
    post_ff_ln: Buf,
    /// q, k and v projections as one GEMM (output columns q | k | v).
    qkv_proj: GpuLinear,
    o_proj: GpuLinear,
    q_norm: Buf,
    k_norm: Buf,
    /// gate and up projections as one GEMM (output columns gate | up).
    gate_up: GpuLinear,
    down_proj: GpuLinear,
    ple_gate: GpuLinear,
    ple_proj: GpuLinear,
    ple_post_norm: Buf,
    layer_scalar: f32,
}

/// cos/sin tables `[len, head_dim]` for positions `0..len` (one entry per (theta, head_dim)).
struct RopeTable {
    len: usize,
    cos: Buf,
    sin: Buf,
}

pub struct GpuText {
    pub cfg: TextConfig,
    weights: Arc<Weights>,
    gpu: Arc<Gpu>,
    layers: Vec<Layer>,
    final_norm: Buf,
    ple_model_proj: GpuLinear,
    ple_norm: Buf,
    embed_proj: GpuLinear,
    /// Grow-only RoPE tables; row `i` depends only on position `i`, so a longer table serves shorter inputs.
    rope: Mutex<HashMap<(u64, usize), RopeTable>>,
    /// Recorded short passes by padded row count (see [`GRAPH_ROW_BUCKETS`]) and whether they use the packed flash
    /// kernels (see [`PackedFlash`]); empty when graphs are off.
    graphs: Mutex<HashMap<(usize, bool), TextGraph>>,
    use_graphs: AtomicBool,
    packed_flash: PackedFlash,
}

/// Row counts the graphs are recorded for: an input is padded up to the next bucket with a dummy
/// sequence at the end (rows are independent apart from attention, which stays within a sequence), so
/// a handful of graphs serves every short input. Longer inputs are GPU-bound and run without a graph.
const GRAPH_ROW_BUCKETS: [usize; 9] = [8, 16, 24, 32, 48, 64, 80, 96, 128];

/// Which fused bf16 passes run their attention on the tensor-core packed flash kernels instead of the
/// float32 packed kernel (`EGEMMA2_PACKED_FLASH`; float32 and TF32 modes always use the packed kernel).
///
/// A flash tile costs about the same whether its sequence has 5 tokens or 64 (a 64-row Q tile, a K/V tile, a few
/// barriers), while the packed kernel's cost grows with the keys each row walks. So the choice follows the average
/// sequence length of the pass ([`FLASH_MIN_MEAN_LEN`]), for eager passes and recorded graphs alike (graphs are cached
/// per row bucket and kernel). `EGEMMA2_PACKED_FLASH=0` turns the flash path off everywhere, `2` forces it on (`1`, or
/// unset, is the default length rule).
#[derive(Clone, Copy, PartialEq, Eq)]
enum PackedFlash {
    Off,
    Auto,
    All,
}

/// Average sequence length (tokens) from which a packed pass takes the flash path. Measured end to end on the
/// development GPU (bf16, passes replayed from CUDA graphs, packed kernel -> flash): 1 x 8 tokens 2.29 -> 2.35 ms,
/// 16 x 7: 3.44 -> 3.59, 8 x 15: 3.43 -> 3.48, 1 x 16 and 4 x 16: even; 1 x 32: 2.71 -> 2.55, 2 x 30: 2.94 -> 2.87,
/// 1 x 64: 3.32 -> 3.06, 2 x 60: 4.04 -> 3.63, 1 x 126: 4.87 -> 3.88; the batch of 64 documents (45 tokens on
/// average, eager) gains the most.
const FLASH_MIN_MEAN_LEN: usize = 24;

impl PackedFlash {
    fn from_env() -> Self {
        match std::env::var("EGEMMA2_PACKED_FLASH").as_deref() {
            Ok("0") => Self::Off,
            Ok("2") => Self::All,
            _ => Self::Auto,
        }
    }

    /// Whether a packed pass over sequences of `lens` tokens takes the flash path.
    fn enabled(self, lens: &[usize]) -> bool {
        match self {
            Self::Off => false,
            Self::Auto => lens.iter().sum::<usize>() >= FLASH_MIN_MEAN_LEN * lens.len(),
            Self::All => true,
        }
    }
}

/// Attention inputs and tile list of a fused pass whose layers use the packed flash kernels (`Work::flash`).
struct FlashWork {
    /// bf16 q/k/v of those layers (`Work::q` / `k` / `v`, float32, serve the layers that stay on the packed kernel).
    q: Act,
    k: Act,
    v: Act,
    /// Tile list ({row0, len, q0} triples, see `Gpu::flash_packed_tiles`), uploaded per call; `n_tiles` is the grid.
    tiles: CudaSlice<i32>,
    n_tiles: usize,
}

/// Per-pass device buffers (sized for `rows`), reused by all layers. The residual stream (`hidden`) and the
/// per-layer inputs are float32; the [`Act`] buffers, the inputs and outputs of the matrix products, are bf16
/// in bf16 mode.
struct Work {
    rows: usize,
    /// Pristine embeddings, copied into `hidden` at the start of a recorded pass (which updates it in place).
    x_in: Option<Buf>,
    hidden: Buf,
    ple: Buf,
    /// Per-row metadata: [position within its sequence | sequence start | sequence length].
    meta: CudaSlice<i32>,
    /// Normalised hidden states: the input of the q/k/v and gate/up products and of the final projection.
    t: Act,
    qkv: Act,
    /// Attention inputs: float32 for the packed kernel (float32 arithmetic), the activation format otherwise.
    q: Act,
    k: Act,
    v: Act,
    flash: Option<FlashWork>,
    a: Act,
    o: Act,
    gu: Act,
    gate: Act,
    pg: Act,
    /// bf16 mode: the hidden states as the per-layer-input gate's bf16 input, and the gated per-layer input
    /// (the next product's input). Float32 mode reads `hidden` and gates `pg` in place.
    hb: Option<Act>,
    pgb: Option<Act>,
    out: Act,
}

struct TextGraph {
    work: Work,
    graph: Graph,
}

// A graph and its buffers are only touched while holding the owning `Gpu`'s lock.
unsafe impl Send for TextGraph {}

impl GpuText {
    /// Upload a loaded CPU text model; the CPU copy of the weights is dropped afterwards.
    pub fn from_cpu(cpu: TextModel, gpu: Arc<Gpu>) -> Result<Self> {
        let g = &*gpu;
        // Uploads go through the shared stream: keep them out of another thread's pass (or graph recording).
        let lock_owner = gpu.clone();
        let _guard = lock_owner.lock();
        let mut layers = Vec::with_capacity(cpu.layers.len());
        for l in &cpu.layers {
            layers.push(Layer {
                head_dim: l.head_dim,
                n_kv: l.n_kv,
                window: l.window,
                theta: l.theta,
                input_ln: g.upload(&l.input_ln)?,
                post_attn_ln: g.upload(&l.post_attn_ln)?,
                pre_ff_ln: g.upload(&l.pre_ff_ln)?,
                post_ff_ln: g.upload(&l.post_ff_ln)?,
                qkv_proj: GpuLinear::concat(g, &[&l.q_proj, &l.k_proj, &l.v_proj])?,
                o_proj: GpuLinear::from_cpu(g, &l.o_proj)?,
                q_norm: g.upload(&l.q_norm)?,
                k_norm: g.upload(&l.k_norm)?,
                gate_up: GpuLinear::concat(g, &[&l.gate_proj, &l.up_proj])?,
                down_proj: GpuLinear::from_cpu(g, &l.down_proj)?,
                ple_gate: GpuLinear::from_cpu(g, &l.ple_gate)?,
                ple_proj: GpuLinear::from_cpu(g, &l.ple_proj)?,
                ple_post_norm: g.upload(&l.ple_post_norm)?,
                layer_scalar: l.layer_scalar,
            });
        }
        Ok(Self {
            cfg: cpu.cfg.clone(),
            weights: cpu.weights.clone(),
            layers,
            final_norm: g.upload(&cpu.final_norm)?,
            ple_model_proj: GpuLinear::from_cpu(g, &cpu.ple_model_proj)?,
            ple_norm: g.upload(&cpu.ple_norm)?,
            embed_proj: GpuLinear::from_cpu(g, &cpu.embed_proj)?,
            rope: Mutex::new(HashMap::new()),
            graphs: Mutex::new(HashMap::new()),
            use_graphs: AtomicBool::new(std::env::var_os("EGEMMA2_NO_GRAPH").is_none() && !gpu.profiling()),
            packed_flash: PackedFlash::from_env(),
            gpu,
        })
    }

    pub fn embed_tokens(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        text::embed_tokens(&self.weights, self.cfg.hidden_size, ids, out)
    }

    /// Make sure every (theta, head_dim) table used by the layers covers positions `0..max_len`.
    /// Returns whether an existing table had to be replaced (which invalidates recorded graphs).
    fn ensure_rope(&self, table: &mut HashMap<(u64, usize), RopeTable>, max_len: usize) -> Result<bool> {
        let g = &*self.gpu;
        let mut replaced = false;
        for l in &self.layers {
            let key = (l.theta.to_bits(), l.head_dim);
            if table.get(&key).is_some_and(|t| t.len >= max_len) {
                continue;
            }
            // Round up so slowly growing inputs do not rebuild the table on every call.
            let len = max_len.next_multiple_of(256);
            let positions: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let (c, s) = ops::rope_tables(&positions, &ops::rope_inv_freq(l.theta, l.head_dim));
            replaced |= table.insert(key, RopeTable { len, cos: g.upload(&c)?, sin: g.upload(&s)? }).is_some();
        }
        Ok(replaced)
    }

    /// Whether some layer's packed attention can run on the tensor-core flash kernel (bf16 mode).
    fn has_flash_layers(&self) -> bool {
        let nh = self.cfg.num_attention_heads;
        self.layers.iter().any(|l| self.gpu.packed_flash_supported(nh, l.n_kv, l.head_dim))
    }

    /// Device buffers for a pass over `rows` rows. `graph` buffers also keep the pristine input and the
    /// full (fused-attention) sequence metadata; `fused` passes use the packed attention kernels, the flash
    /// ones when `flash_tiles` (the grid of the tile list) is given.
    fn new_work(&self, rows: usize, graph: bool, fused: bool, flash_tiles: Option<usize>) -> Result<Work> {
        let g = &*self.gpu;
        let cfg = &self.cfg;
        let (h, p, nl, nh) = (cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers, cfg.num_attention_heads);
        // The flash layers' q/k/v live in `flash`; `q` / `k` / `v` only need to fit the other layers.
        let on_flash = |l: &Layer| flash_tiles.is_some() && g.packed_flash_supported(nh, l.n_kv, l.head_dim);
        let q_w = self.layers.iter().filter(|l| !on_flash(l)).map(|l| nh * l.head_dim).max().unwrap_or(0);
        let kv_w = self.layers.iter().filter(|l| !on_flash(l)).map(|l| l.n_kv * l.head_dim).max().unwrap_or(0);
        let fq_w = self.layers.iter().filter(|l| on_flash(l)).map(|l| nh * l.head_dim).max().unwrap_or(0);
        let fkv_w = self.layers.iter().filter(|l| on_flash(l)).map(|l| l.n_kv * l.head_dim).max().unwrap_or(0);
        let a_w = self.layers.iter().map(|l| nh * l.head_dim).max().unwrap_or(0);
        let qkv_w = self.layers.iter().map(|l| l.qkv_proj.out).max().unwrap_or(0);
        let ff_w = self.layers.iter().map(|l| l.gate_up.out / 2).max().unwrap_or(0);
        let bf = g.bf16_act();
        // The packed attention kernel works in float32 and is bound by its walk over the keys, not by their
        // bytes, so its q/k/v stay float32; the tensor-core path takes them in the activation format.
        let attn_in = |n: usize| if fused { g.alloc_act_f32(n) } else { g.alloc_act(n) };
        Ok(Work {
            rows,
            x_in: if graph { Some(g.alloc(rows * h)?) } else { None },
            hidden: g.alloc(rows * h)?,
            ple: g.alloc(rows * nl * p)?,
            meta: g.upload_i32(&vec![0; rows * 3])?,
            t: g.alloc_act(rows * h)?,
            qkv: g.alloc_act(rows * qkv_w)?,
            q: attn_in(rows * q_w)?,
            k: attn_in(rows * kv_w)?,
            v: attn_in(rows * kv_w)?,
            flash: match flash_tiles {
                Some(n_tiles) => Some(FlashWork {
                    q: g.alloc_act(rows * fq_w)?,
                    k: g.alloc_act(rows * fkv_w)?,
                    v: g.alloc_act(rows * fkv_w)?,
                    // All zeros: len-0 tiles, which the kernel skips.
                    tiles: g.upload_i32(&vec![0; 3 * n_tiles])?,
                    n_tiles,
                }),
                None => None,
            },
            a: g.alloc_act(rows * a_w)?,
            o: g.alloc_act(rows * h)?,
            gu: g.alloc_act(rows * 2 * ff_w)?,
            gate: g.alloc_act(rows * ff_w)?,
            pg: g.alloc_act(rows * p)?,
            hb: if bf { Some(g.alloc_act(rows * h)?) } else { None },
            pgb: if bf { Some(g.alloc_act(rows * p)?) } else { None },
            out: g.alloc_act_f32(rows * cfg.embedding_dim)?,
        })
    }

    /// Per-row metadata for sequences packed back to back: [position | sequence start | sequence length].
    /// Without `full` only the positions (all the unfused attention path needs) are produced.
    fn row_meta(lens: &[usize], full: bool) -> Vec<i32> {
        let rows: usize = lens.iter().sum();
        let mut meta: Vec<i32> = Vec::with_capacity(rows * 3);
        for &l in lens {
            meta.extend(0..l as i32);
        }
        if full {
            let mut off = 0i32;
            for &l in lens {
                meta.extend(std::iter::repeat_n(off, l));
                off += l as i32;
            }
            for &l in lens {
                meta.extend(std::iter::repeat_n(l as i32, l));
            }
        }
        meta
    }

    /// Enqueue the whole backbone over `w.rows` rows, reading `w.hidden` (or `w.x_in`) and leaving the
    /// per-token states in `w.out`. `fused` selects the packed-attention kernel (needs the full metadata);
    /// otherwise attention runs per sequence of `lens`.
    fn enqueue(&self, w: &mut Work, lens: &[usize], fused: bool, rope: &HashMap<(u64, usize), RopeTable>) -> Result<()> {
        let g = &*self.gpu;
        let cfg = &self.cfg;
        let (h, p, nl, nh) = (cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers, cfg.num_attention_heads);
        let rows = w.rows;
        let eps = cfg.rms_norm_eps;
        g.mark("start");
        if let Some(x_in) = &w.x_in {
            g.copy_dtod(x_in, &mut w.hidden)?;
        }
        self.ple_model_proj.forward_into(g, &w.hidden, rows, &mut w.ple)?;
        g.mark("GEMM ple_model 512->12288");
        // ple = rmsnorm(ple * h^-0.5) * ple_norm, per (token, layer) row.
        g.rms_norm_scaled(&mut w.ple, rows * nl, p, &self.ple_norm, eps, (h as f32).powf(-0.5))?;
        let pos = w.meta.slice(0..rows);
        let (start, len) = (w.meta.slice(rows..2 * rows), w.meta.slice(2 * rows..3 * rows));

        g.rms_norm_act(&w.hidden, &mut w.t, rows, h, &self.layers[0].input_ln, eps)?;
        g.mark("norms (ple, first)");
        for (i, l) in self.layers.iter().enumerate() {
            let hd = l.head_dim;
            let tab = &rope[&(l.theta.to_bits(), hd)];

            l.qkv_proj.forward_act(g, &w.t, rows, &mut w.qkv)?;
            g.mark("GEMM qkv");
            // Layers on the packed flash kernel get bf16 q/k/v and the pass's tile list, the others float32 ones.
            let mut flash = w.flash.as_mut().filter(|_| fused && g.packed_flash_supported(nh, l.n_kv, hd));
            let (qb, kb, vb) = match flash.as_mut() {
                Some(f) => (&mut f.q, &mut f.k, &mut f.v),
                None => (&mut w.q, &mut w.k, &mut w.v),
            };
            g.qkv_post_act(&w.qkv, qb, kb, vb, [&l.q_norm, &l.k_norm, &tab.cos, &tab.sin], Some(&pos), rows, nh, l.n_kv, hd, hd, eps)?;
            g.mark("qkv_post (norms + RoPE)");
            if let Some(f) = flash {
                g.attention_flash_packed(&f.q, &f.k, &f.v, &mut w.a, &f.tiles.slice(..), f.n_tiles, nh, l.n_kv, hd, l.window)?;
                g.mark(if hd == 256 { "attn: packed flash (d256)" } else { "attn: packed flash (d512)" });
            } else if fused {
                g.attention_packed_act(&w.q, &w.k, &w.v, &mut w.a, &start, &len, rows, nh, l.n_kv, hd, l.window)?;
                g.mark(if hd == 256 { "attn: packed (d256)" } else { "attn: packed (d512)" });
            } else {
                let mut row0 = 0;
                for &n in lens {
                    g.attention_act(&w.q, &w.k, &w.v, &mut w.a, row0, n, nh, l.n_kv, hd, l.window)?;
                    row0 += n;
                }
                g.mark(match hd {
                    256 if g.flash_on() => "attn: flash (d256)",
                    512 if g.bf16_act() && g.flash_on() => "attn: flash (d512)",
                    _ => "attn: cuBLAS rest",
                });
            }
            l.o_proj.forward_act(g, &w.a, rows, &mut w.o)?;
            g.mark("GEMM o");
            g.add_norm_act(&mut w.hidden, &w.o, &l.post_attn_ln, Some((&mut w.t, Some(&l.pre_ff_ln))), rows, h, eps, 1.0)?;
            g.mark("add_norm");

            l.gate_up.forward_act(g, &w.t, rows, &mut w.gu)?;
            g.mark("GEMM gate|up");
            g.gelu_mul_gu_act(&mut w.gate, &w.gu, rows, l.gate_up.out / 2)?;
            g.mark("gelu_mul");
            l.down_proj.forward_act(g, &w.gate, rows, &mut w.o)?;
            g.mark("GEMM down");
            // bf16 mode also writes the updated hidden states as the gate product's bf16 input.
            g.add_norm_act(&mut w.hidden, &w.o, &l.post_ff_ln, w.hb.as_mut().map(|hb| (hb, None)), rows, h, eps, 1.0)?;
            g.mark("add_norm");

            match &w.hb {
                Some(hb) => l.ple_gate.forward_act(g, hb, rows, &mut w.pg)?,
                None => l.ple_gate.forward_into(g, &w.hidden, rows, w.pg.f32_mut()?)?,
            }
            g.mark("GEMM ple gate/proj");
            g.ple_gate_act(&mut w.pg, w.pgb.as_mut(), &w.ple, rows, p, nl, i)?;
            g.mark("ple_gate");
            l.ple_proj.forward_act(g, w.pgb.as_ref().unwrap_or(&w.pg), rows, &mut w.o)?;
            g.mark("GEMM ple gate/proj");
            // hidden = (hidden + rmsnorm(o)) * layer_scalar, then t = the next layer's (or the final) norm of it.
            let next_w = self.layers.get(i + 1).map_or(&self.final_norm, |n| &n.input_ln);
            g.add_norm_act(&mut w.hidden, &w.o, &l.ple_post_norm, Some((&mut w.t, Some(next_w))), rows, h, eps, l.layer_scalar)?;
            g.mark("add_norm");
        }
        self.embed_proj.forward_act(g, &w.t, rows, &mut w.out)?;
        g.mark("GEMM embed_proj");
        Ok(())
    }

    /// Encode sequences packed back to back; returns `[rows, embedding_dim]` on the host.
    pub fn forward_packed(&self, inputs_embeds: &[f32], lens: &[usize]) -> Result<Vec<f32>> {
        let g = &*self.gpu;
        let _guard = g.lock();
        let cfg = &self.cfg;
        let (h, nh) = (cfg.hidden_size, cfg.num_attention_heads);
        let rows: usize = lens.iter().sum();
        if rows == 0 {
            return Ok(Vec::new());
        }

        // Short sequences use one fused attention launch per layer instead of a few cuBLAS calls per
        // sequence. The fused kernel walks the keys serially, so it beats the batched cuBLAS GEMMs up to
        // ~128 tokens for a lone sequence; packed batches use it up to 256 tokens each (the alternative is
        // several launches per sequence).
        const FUSED_MAX_LEN: usize = 256;
        const FUSED_MAX_LEN_SINGLE: usize = 128;
        let max_fused = if lens.len() == 1 { FUSED_MAX_LEN_SINGLE } else { FUSED_MAX_LEN };
        let fused = lens.iter().all(|&l| l <= max_fused) && self.layers.iter().all(|l| g.packed_attention_supported(nh, l.n_kv, l.head_dim));

        let mut rope = self.rope.lock().unwrap_or_else(|e| e.into_inner());
        let replaced = self.ensure_rope(&mut rope, lens.iter().copied().max().unwrap_or(0))?;
        let mut graphs = self.graphs.lock().unwrap_or_else(|e| e.into_inner());
        if replaced {
            graphs.clear(); // they hold the replaced tables' addresses
        }

        if let Some(bucket) = GRAPH_ROW_BUCKETS.iter().copied().find(|&b| b >= rows).filter(|_| fused && self.use_graphs.load(Ordering::Relaxed)) {
            // Pad with a dummy sequence so every input of this size class replays the same graph.
            let mut padded = lens.to_vec();
            if bucket > rows {
                padded.push(bucket - rows);
            }
            let meta = Self::row_meta(&padded, true);
            // The flash launch has a fixed grid in the recording: one tile per row is the most any split of `bucket`
            // rows into sequences needs (a sequence of n rows takes ceil(n / 64) <= n tiles), padded with empty ones.
            // The choice follows the real sequences, not the padding.
            let flash = self.packed_flash.enabled(lens) && self.has_flash_layers();
            let tiles = if flash { Gpu::flash_packed_tiles(0, &padded, bucket) } else { Vec::new() };
            debug_assert!(!flash || tiles.len() == 3 * bucket);
            if let Entry::Vacant(slot) = graphs.entry((bucket, flash)) {
                // Recording is cheap (a few ms); if it is refused for any reason, run without graphs from now on.
                let built = (|| -> Result<TextGraph> {
                    let mut work = self.new_work(bucket, true, true, flash.then_some(bucket))?;
                    g.upload_i32_into(&meta, &mut work.meta)?;
                    if let Some(f) = &mut work.flash {
                        g.upload_i32_into(&tiles, &mut f.tiles)?;
                    }
                    let graph = g.capture(|| self.enqueue(&mut work, &padded, true, &rope))?;
                    Ok(TextGraph { work, graph })
                })();
                match built {
                    Ok(tg) => {
                        slot.insert(tg);
                    }
                    Err(_) => self.use_graphs.store(false, Ordering::Relaxed),
                }
            }
            if let Some(tg) = graphs.get_mut(&(bucket, flash)) {
                let mut x = vec![0f32; bucket * h];
                x[..inputs_embeds.len()].copy_from_slice(inputs_embeds);
                g.upload_into(&x, tg.work.x_in.as_mut().unwrap())?;
                g.upload_i32_into(&meta, &mut tg.work.meta)?;
                if let Some(f) = &mut tg.work.flash {
                    g.upload_i32_into(&tiles, &mut f.tiles)?;
                }
                g.replay(&tg.graph)?;
                return g.download_prefix(tg.work.out.f32()?, rows * cfg.embedding_dim);
            }
        }

        let flash = fused && self.packed_flash.enabled(lens) && self.has_flash_layers();
        let tiles = if flash { Gpu::flash_packed_tiles(0, lens, 0) } else { Vec::new() };
        let mut w = self.new_work(rows, false, fused, flash.then_some(tiles.len() / 3))?;
        g.upload_into(inputs_embeds, &mut w.hidden)?;
        g.upload_i32_into(&Self::row_meta(lens, fused), &mut w.meta)?;
        if let Some(f) = &mut w.flash {
            g.upload_i32_into(&tiles, &mut f.tiles)?;
        }
        self.enqueue(&mut w, lens, fused, &rope)?;
        g.download(w.out.f32()?)
    }
}
