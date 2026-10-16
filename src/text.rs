//! The EmbeddingGemma 2 text backbone (`EmbeddingGemma2TextModel`).
//!
//! A 24-layer bidirectional encoder with per-layer-input (PLE) gating, 5:1 local/global attention
//! (sliding-window radius 512, wider heads on global layers), sandwich RMS norms and a final
//! 512 -> 768 projection that is applied per token (equivalent to projecting after mean pooling).

use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;

use crate::config::TextConfig;
use crate::ops::{self, AttnShape, Linear};
use crate::weights::Weights;

pub(crate) struct Layer {
    pub(crate) head_dim: usize,
    pub(crate) n_kv: usize,
    pub(crate) window: Option<usize>,
    pub(crate) theta: f64,
    pub(crate) input_ln: Vec<f32>,
    pub(crate) post_attn_ln: Vec<f32>,
    pub(crate) pre_ff_ln: Vec<f32>,
    pub(crate) post_ff_ln: Vec<f32>,
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
    pub(crate) ple_gate: Linear,
    pub(crate) ple_proj: Linear,
    pub(crate) ple_post_norm: Vec<f32>,
    pub(crate) layer_scalar: f32,
}

pub struct TextModel {
    pub cfg: TextConfig,
    pub(crate) weights: Arc<Weights>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) ple_model_proj: Linear,
    pub(crate) ple_norm: Vec<f32>,
    pub(crate) embed_proj: Linear,
}

const EMBED: &str = "language_model.embed_tokens.weight";

/// Scaled token embeddings (`embed_tokens(ids) * sqrt(hidden)`), read straight from the
/// memory-mapped BF16 table.
pub(crate) fn embed_tokens(weights: &Weights, h: usize, ids: &[u32], out: &mut [f32]) -> Result<()> {
    let scale = (h as f32).sqrt();
    assert_eq!(out.len(), ids.len() * h);
    for (t, &id) in ids.iter().enumerate() {
        let row = &mut out[t * h..(t + 1) * h];
        weights.row_into(EMBED, id as usize, row)?;
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
    Ok(())
}

impl TextModel {
    pub fn load(cfg: &TextConfig, weights: Arc<Weights>) -> Result<Self> {
        let w = &*weights;
        let h = cfg.hidden_size;
        let p = cfg.hidden_size_per_layer_input;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let (hd, n_kv) = cfg.layer_dims(i);
            let pre = format!("language_model.layers.{i}");
            layers.push(Layer {
                head_dim: hd,
                n_kv,
                window: cfg.is_sliding(i).then_some(cfg.sliding_window),
                theta: cfg.rope_theta(i),
                input_ln: w.f32_shaped(&format!("{pre}.input_layernorm.weight"), &[h])?,
                post_attn_ln: w.f32_shaped(&format!("{pre}.post_attention_layernorm.weight"), &[h])?,
                pre_ff_ln: w.f32_shaped(&format!("{pre}.pre_feedforward_layernorm.weight"), &[h])?,
                post_ff_ln: w.f32_shaped(&format!("{pre}.post_feedforward_layernorm.weight"), &[h])?,
                q_proj: Linear::load(w, &format!("{pre}.self_attn.q_proj"), cfg.num_attention_heads * hd, h, false)?,
                k_proj: Linear::load(w, &format!("{pre}.self_attn.k_proj"), n_kv * hd, h, false)?,
                v_proj: Linear::load(w, &format!("{pre}.self_attn.v_proj"), n_kv * hd, h, false)?,
                o_proj: Linear::load(w, &format!("{pre}.self_attn.o_proj"), h, cfg.num_attention_heads * hd, false)?,
                q_norm: w.f32_shaped(&format!("{pre}.self_attn.q_norm.weight"), &[hd])?,
                k_norm: w.f32_shaped(&format!("{pre}.self_attn.k_norm.weight"), &[hd])?,
                gate_proj: Linear::load(w, &format!("{pre}.mlp.gate_proj"), cfg.intermediate_size, h, false)?,
                up_proj: Linear::load(w, &format!("{pre}.mlp.up_proj"), cfg.intermediate_size, h, false)?,
                down_proj: Linear::load(w, &format!("{pre}.mlp.down_proj"), h, cfg.intermediate_size, false)?,
                ple_gate: Linear::load(w, &format!("{pre}.ple_block.per_layer_input_gate"), p, h, false)?,
                ple_proj: Linear::load(w, &format!("{pre}.ple_block.per_layer_projection"), h, p, false)?,
                ple_post_norm: w.f32_shaped(&format!("{pre}.ple_block.post_per_layer_input_norm.weight"), &[h])?,
                layer_scalar: w.scalar(&format!("{pre}.layer_scalar"))?,
            });
        }
        Ok(Self {
            cfg: cfg.clone(),
            layers,
            final_norm: w.f32_shaped("language_model.norm.weight", &[h])?,
            ple_model_proj: Linear::load(
                w,
                "language_model.ple.per_layer_model_projection",
                cfg.num_hidden_layers * p,
                h,
                false,
            )?,
            ple_norm: w.f32_shaped("language_model.ple.per_layer_projection_norm.weight", &[p])?,
            embed_proj: Linear::load(w, "language_model.embedding_projection", cfg.embedding_dim, h, false)?,
            weights,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Look up (and scale by `sqrt(hidden_size)`) the embedding of every token id into `out`.
    pub fn embed_tokens(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        embed_tokens(&self.weights, self.cfg.hidden_size, ids, out)
    }

    /// Run the encoder on `inputs_embeds` (`[seq, hidden]`, already merged with any multimodal soft
    /// tokens) and return the projected per-token states `[seq, embedding_dim]`.
    pub fn forward(&self, inputs_embeds: &[f32], seq: usize) -> Vec<f32> {
        self.forward_packed(inputs_embeds, &[seq])
    }

    /// Encode several sequences packed back to back (`lens[i]` rows each) in one pass.
    ///
    /// Every projection runs once over all rows, so the weights are streamed once per batch
    /// instead of once per sequence; positions restart at 0 for each sequence and attention is
    /// block-diagonal, so the result equals encoding each sequence on its own.
    pub fn forward_packed(&self, inputs_embeds: &[f32], lens: &[usize]) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let p = cfg.hidden_size_per_layer_input;
        let n_layers = cfg.num_hidden_layers;
        let seq: usize = lens.iter().sum();
        assert_eq!(inputs_embeds.len(), seq * h);
        let offsets: Vec<usize> = lens.iter().scan(0, |acc, &l| { let o = *acc; *acc += l; Some(o) }).collect();

        // Per-layer inputs: projection-only PLE, derived from the (merged) input embeddings.
        let mut ple = self.ple_model_proj.forward(inputs_embeds, seq);
        let ple_scale = (h as f32).powf(-0.5);
        ple.par_iter_mut().for_each(|v| *v *= ple_scale);
        ops::rms_norm(&mut ple, p, Some(&self.ple_norm), cfg.rms_norm_eps);

        // RoPE tables are shared by layers with the same (theta, head_dim); cache by key.
        let positions: Vec<f32> = lens.iter().flat_map(|&l| (0..l).map(|i| i as f32)).collect();
        let mut rope_cache: Vec<((u64, usize), (Vec<f32>, Vec<f32>))> = Vec::new();

        let mut hidden = inputs_embeds.to_vec();
        for (i, l) in self.layers.iter().enumerate() {
            let hd = l.head_dim;
            let key = (l.theta.to_bits(), hd);
            if !rope_cache.iter().any(|(k, _)| *k == key) {
                let inv = ops::rope_inv_freq(l.theta, hd);
                rope_cache.push((key, ops::rope_tables(&positions, &inv)));
            }
            let (cos, sin) = &rope_cache.iter().find(|(k, _)| *k == key).unwrap().1;

            // --- attention sub-block ---
            let mut x = hidden.clone();
            ops::rms_norm(&mut x, h, Some(&l.input_ln), cfg.rms_norm_eps);
            let mut q = l.q_proj.forward(&x, seq);
            let mut k = l.k_proj.forward(&x, seq);
            let mut v = l.v_proj.forward(&x, seq);
            ops::rms_norm(&mut q, hd, Some(&l.q_norm), cfg.rms_norm_eps);
            ops::rms_norm(&mut k, hd, Some(&l.k_norm), cfg.rms_norm_eps);
            ops::rms_norm(&mut v, hd, None, cfg.rms_norm_eps);
            ops::rope_rows(&mut q, cfg.num_attention_heads, hd, cos, sin);
            ops::rope_rows(&mut k, l.n_kv, hd, cos, sin);
            let shape = AttnShape {
                n_heads: cfg.num_attention_heads,
                n_kv_heads: l.n_kv,
                head_dim: hd,
                window: l.window,
                scale: 1.0,
            };
            let a = if lens.len() == 1 {
                ops::attention(&q, &k, &v, seq, &shape)
            } else {
                // Block-diagonal: each sequence attends only to itself.
                let (qs, kvs) = (cfg.num_attention_heads * hd, l.n_kv * hd);
                let parts: Vec<Vec<f32>> = lens
                    .par_iter()
                    .zip(offsets.par_iter())
                    .map(|(&len, &off)| {
                        ops::attention(
                            &q[off * qs..(off + len) * qs],
                            &k[off * kvs..(off + len) * kvs],
                            &v[off * kvs..(off + len) * kvs],
                            len,
                            &shape,
                        )
                    })
                    .collect();
                parts.concat()
            };
            let mut a = l.o_proj.forward(&a, seq);
            ops::rms_norm(&mut a, h, Some(&l.post_attn_ln), cfg.rms_norm_eps);
            ops::add_inplace(&mut hidden, &a);

            // --- feed-forward sub-block ---
            let mut x = hidden.clone();
            ops::rms_norm(&mut x, h, Some(&l.pre_ff_ln), cfg.rms_norm_eps);
            let mut gate = l.gate_proj.forward(&x, seq);
            let up = l.up_proj.forward(&x, seq);
            gate.par_iter_mut().zip(up.par_iter()).for_each(|(g, u)| *g = ops::gelu_tanh(*g) * *u);
            let mut d = l.down_proj.forward(&gate, seq);
            ops::rms_norm(&mut d, h, Some(&l.post_ff_ln), cfg.rms_norm_eps);
            ops::add_inplace(&mut hidden, &d);

            // --- per-layer-input (PLE) sub-block ---
            let mut g = l.ple_gate.forward(&hidden, seq);
            g.par_chunks_mut(p).enumerate().for_each(|(t, row)| {
                let pli = &ple[(t * n_layers + i) * p..(t * n_layers + i + 1) * p];
                for (gv, pv) in row.iter_mut().zip(pli) {
                    *gv = ops::gelu_tanh(*gv) * *pv;
                }
            });
            let mut pr = l.ple_proj.forward(&g, seq);
            ops::rms_norm(&mut pr, h, Some(&l.ple_post_norm), cfg.rms_norm_eps);
            ops::add_inplace(&mut hidden, &pr);

            let ls = l.layer_scalar;
            hidden.par_iter_mut().for_each(|v| *v *= ls);
        }

        ops::rms_norm(&mut hidden, h, Some(&self.final_norm), cfg.rms_norm_eps);
        self.embed_proj.forward(&hidden, seq)
    }
}
