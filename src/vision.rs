//! The Gemma 4 vision tower plus the `embed_vision` multimodal embedder.
//!
//! 16 bidirectional layers over 16x16 patches with 2-D axial RoPE and learned 2-D position
//! tables, followed by 3x3 average pooling, `sqrt(hidden)` scaling and a projection into the
//! text model's embedding space (one soft token per 3x3 patch block).

use anyhow::{Result, bail};
use rayon::prelude::*;
use std::sync::Arc;

use crate::config::VisionConfig;
use crate::imageproc::PatchedImage;
use crate::ops::{self, AttnShape, ClippableLinear, Linear};
use crate::weights::Weights;

pub(crate) struct Layer {
    pub(crate) q_proj: ClippableLinear,
    pub(crate) k_proj: ClippableLinear,
    pub(crate) v_proj: ClippableLinear,
    pub(crate) o_proj: ClippableLinear,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) gate_proj: ClippableLinear,
    pub(crate) up_proj: ClippableLinear,
    pub(crate) down_proj: ClippableLinear,
    pub(crate) input_ln: Vec<f32>,
    pub(crate) post_attn_ln: Vec<f32>,
    pub(crate) pre_ff_ln: Vec<f32>,
    pub(crate) post_ff_ln: Vec<f32>,
}

/// The parts of the tower that stay on the host for every backend: position tables, RoPE
/// tables, pooling and the projection into text space.
pub(crate) struct VisionHead {
    pub(crate) cfg: VisionConfig,
    /// `[2, position_embedding_size, hidden]`: table 0 is indexed by x, table 1 by y.
    pos_table: Vec<f32>,
    embed_proj: Linear,
}

pub struct VisionTower {
    pub(crate) head: VisionHead,
    pub(crate) input_proj: Linear,
    pub(crate) layers: Vec<Layer>,
}

impl VisionHead {
    /// Validate a patched image (used by both the CPU and GPU towers).
    pub(crate) fn check(&self, img: &PatchedImage) -> Result<()> {
        let k = self.cfg.pooling_kernel_size;
        let n = img.n_patches();
        let patch_dim = 3 * self.cfg.patch_size * self.cfg.patch_size;
        if n == 0 || img.grid_h % k != 0 || img.grid_w % k != 0 {
            bail!("patch grid {}x{} is empty or not divisible by the pooling kernel {k}", img.grid_h, img.grid_w);
        }
        if img.patch_dim != patch_dim || img.patches.len() != n * patch_dim || img.positions.len() != n {
            bail!(
                "malformed patches: {} values / {} positions for a {}x{} grid of {patch_dim}-value patches",
                img.patches.len(),
                img.positions.len(),
                img.grid_h,
                img.grid_w
            );
        }
        let pe = self.cfg.position_embedding_size as u32;
        if img.positions.iter().any(|[x, y]| *x >= pe || *y >= pe) {
            bail!("patch position outside the {pe}-entry position table");
        }
        Ok(())
    }

    /// Patch pixels mapped to `[-1, 1]` (`2 * (x - 0.5)`), the input of `input_proj`.
    pub(crate) fn scaled_patches(img: &PatchedImage) -> Vec<f32> {
        img.patches.par_iter().map(|v| 2.0 * (*v - 0.5)).collect()
    }

    /// Add the learned 2-D position embeddings `pos_x[x] + pos_y[y]` to `[n, hidden]` rows.
    pub(crate) fn add_positions(&self, hidden: &mut [f32], img: &PatchedImage) {
        let h = self.cfg.hidden_size;
        let pe = self.cfg.position_embedding_size;
        hidden.par_chunks_mut(h).enumerate().for_each(|(t, row)| {
            let [x, y] = img.positions[t];
            let px = &self.pos_table[(x as usize) * h..(x as usize + 1) * h];
            let py = &self.pos_table[(pe + y as usize) * h..(pe + y as usize + 1) * h];
            for i in 0..h {
                row[i] += px[i] + py[i];
            }
        });
    }

    /// Axial RoPE tables `[n, head_dim]`: the first half of each head rotates with x, the second
    /// half with y, each half in split-halves layout.
    pub(crate) fn rope_tables(&self, img: &PatchedImage) -> (Vec<f32>, Vec<f32>) {
        let hd = self.cfg.head_dim;
        let n = img.n_patches();
        let half_spatial = hd / 4; // frequencies per axis
        let inv = ops::rope_inv_freq(self.cfg.rope_parameters.rope_theta, hd / 2);
        debug_assert_eq!(inv.len(), half_spatial);
        let mut cos = vec![0f32; n * hd];
        let mut sin = vec![0f32; n * hd];
        for t in 0..n {
            let [x, y] = img.positions[t];
            for (axis, pos) in [x, y].into_iter().enumerate() {
                for i in 0..half_spatial {
                    let (s, c) = (pos as f32 * inv[i]).sin_cos();
                    let base = t * hd + axis * (hd / 2);
                    cos[base + i] = c;
                    cos[base + half_spatial + i] = c;
                    sin[base + i] = s;
                    sin[base + half_spatial + i] = s;
                }
            }
        }
        (cos, sin)
    }

    /// k x k average pooling in row-major block order, `* sqrt(hidden)`, then the
    /// `Gemma4MultimodalEmbedder` (scale-free RMSNorm + projection into text space).
    pub(crate) fn pool_and_embed(&self, hidden: &[f32], img: &PatchedImage) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let k = self.cfg.pooling_kernel_size;
        let (oh, ow) = (img.grid_h / k, img.grid_w / k);
        let root = (h as f32).sqrt();
        let mut pooled = vec![0f32; oh * ow * h];
        pooled.par_chunks_mut(h).enumerate().for_each(|(o, out)| {
            let (oy, ox) = (o / ow, o % ow);
            let mut acc = vec![0f32; h];
            for dy in 0..k {
                for dx in 0..k {
                    let t = (oy * k + dy) * img.grid_w + ox * k + dx;
                    for (a, v) in acc.iter_mut().zip(&hidden[t * h..(t + 1) * h]) {
                        *a += *v;
                    }
                }
            }
            let denom = (k * k) as f32;
            for (o, a) in out.iter_mut().zip(&acc) {
                *o = (*a / denom) * root;
            }
        });
        ops::rms_norm(&mut pooled, h, None, self.cfg.rms_norm_eps);
        self.embed_proj.forward(&pooled, oh * ow)
    }
}

impl VisionTower {
    pub fn load(cfg: &VisionConfig, text_hidden: usize, w: &Arc<Weights>) -> Result<Self> {
        if cfg.standardize {
            bail!("vision_config.standardize=true is not supported by this port");
        }
        if cfg.rope_parameters.rope_type != "axial" {
            bail!("unsupported vision rope type `{}`", cfg.rope_parameters.rope_type);
        }
        let (h, hd, nh, ff) = (cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.intermediate_size);
        let clip = cfg.use_clipped_linears;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("vision_tower.encoder.layers.{i}");
            layers.push(Layer {
                q_proj: ClippableLinear::load(w, &format!("{p}.self_attn.q_proj"), nh * hd, h, clip)?,
                k_proj: ClippableLinear::load(w, &format!("{p}.self_attn.k_proj"), cfg.num_key_value_heads * hd, h, clip)?,
                v_proj: ClippableLinear::load(w, &format!("{p}.self_attn.v_proj"), cfg.num_key_value_heads * hd, h, clip)?,
                o_proj: ClippableLinear::load(w, &format!("{p}.self_attn.o_proj"), h, nh * hd, clip)?,
                q_norm: w.f32_shaped(&format!("{p}.self_attn.q_norm.weight"), &[hd])?,
                k_norm: w.f32_shaped(&format!("{p}.self_attn.k_norm.weight"), &[hd])?,
                gate_proj: ClippableLinear::load(w, &format!("{p}.mlp.gate_proj"), ff, h, clip)?,
                up_proj: ClippableLinear::load(w, &format!("{p}.mlp.up_proj"), ff, h, clip)?,
                down_proj: ClippableLinear::load(w, &format!("{p}.mlp.down_proj"), h, ff, clip)?,
                input_ln: w.f32_shaped(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_attn_ln: w.f32_shaped(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                pre_ff_ln: w.f32_shaped(&format!("{p}.pre_feedforward_layernorm.weight"), &[h])?,
                post_ff_ln: w.f32_shaped(&format!("{p}.post_feedforward_layernorm.weight"), &[h])?,
            });
        }
        Ok(Self {
            head: VisionHead {
                cfg: cfg.clone(),
                pos_table: w.f32_shaped(
                    "vision_tower.patch_embedder.position_embedding_table",
                    &[2, cfg.position_embedding_size, h],
                )?,
                embed_proj: Linear::load(w, "embed_vision.embedding_projection", text_hidden, h, false)?,
            },
            input_proj: Linear::load(
                w,
                "vision_tower.patch_embedder.input_proj",
                h,
                3 * cfg.patch_size * cfg.patch_size,
                false,
            )?,
            layers,
        })
    }

    /// Encode one patched image into soft tokens `[n_patches / k^2, text_hidden]`.
    pub fn encode(&self, img: &PatchedImage) -> Result<Vec<f32>> {
        let head = &self.head;
        head.check(img)?;
        let cfg = &head.cfg;
        let (h, hd, nh) = (cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads);
        let n = img.n_patches();

        let mut hidden = self.input_proj.forward(&VisionHead::scaled_patches(img), n);
        head.add_positions(&mut hidden, img);
        let (cos, sin) = head.rope_tables(img);

        let eps = cfg.rms_norm_eps;
        for l in &self.layers {
            let mut x = hidden.clone();
            ops::rms_norm(&mut x, h, Some(&l.input_ln), eps);
            let mut q = l.q_proj.forward(&x, n);
            let mut kk = l.k_proj.forward(&x, n);
            let mut v = l.v_proj.forward(&x, n);
            ops::rms_norm(&mut q, hd, Some(&l.q_norm), eps);
            ops::rms_norm(&mut kk, hd, Some(&l.k_norm), eps);
            ops::rms_norm(&mut v, hd, None, eps);
            rope_2d(&mut q, nh, hd, &cos, &sin);
            rope_2d(&mut kk, cfg.num_key_value_heads, hd, &cos, &sin);
            let a = ops::attention(
                &q,
                &kk,
                &v,
                n,
                &AttnShape {
                    n_heads: nh,
                    n_kv_heads: cfg.num_key_value_heads,
                    head_dim: hd,
                    window: None,
                    scale: 1.0,
                },
            );
            let mut a = l.o_proj.forward(&a, n);
            ops::rms_norm(&mut a, h, Some(&l.post_attn_ln), eps);
            ops::add_inplace(&mut hidden, &a);

            let mut x = hidden.clone();
            ops::rms_norm(&mut x, h, Some(&l.pre_ff_ln), eps);
            let mut gate = l.gate_proj.forward(&x, n);
            let up = l.up_proj.forward(&x, n);
            gate.par_iter_mut().zip(up.par_iter()).for_each(|(g, u)| *g = ops::gelu_tanh(*g) * *u);
            let mut d = l.down_proj.forward(&gate, n);
            ops::rms_norm(&mut d, h, Some(&l.post_ff_ln), eps);
            ops::add_inplace(&mut hidden, &d);
        }
        Ok(head.pool_and_embed(&hidden, img))
    }
}

/// Apply 2-axis RoPE: the first half of every head rotates with the x tables, the second half
/// with the y tables; each half uses the standard split-halves `rotate_half`.
fn rope_2d(x: &mut [f32], heads: usize, hd: usize, cos: &[f32], sin: &[f32]) {
    let stride = heads * hd;
    let part = hd / 2;
    x.par_chunks_mut(stride).enumerate().for_each(|(t, row)| {
        for h in 0..heads {
            for axis in 0..2 {
                let o = t * hd + axis * part;
                ops::rope_apply(
                    &mut row[h * hd + axis * part..h * hd + (axis + 1) * part],
                    &cos[o..o + part],
                    &sin[o..o + part],
                );
            }
        }
    });
}
