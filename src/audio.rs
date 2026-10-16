//! The Gemma 4 audio tower (a Universal-Speech-Model style conformer) plus `embed_audio`.
//!
//! Two stride-2 Conv2d layers (4x time reduction, so one soft token per 40 ms), 12 conformer
//! layers (macaron feed-forwards, chunked local attention with relative position bias and logit
//! soft-capping, gated depthwise causal convolution) and a projection into the text embedding space.

use anyhow::{Result, bail, ensure};
use rayon::prelude::*;
use std::sync::Arc;

use crate::config::AudioConfig;
use crate::ops::{self, ClippableLinear, Linear};
use crate::weights::Weights;

const NORM_EPS: f32 = 1e-6;
/// Rows of im2col processed at a time in the second subsampling conv (bounds peak memory).
const IM2COL_CHUNK: usize = 4096;

pub(crate) struct FeedForward {
    pub(crate) ffw1: ClippableLinear,
    pub(crate) ffw2: ClippableLinear,
    pub(crate) pre_ln: Vec<f32>,
    pub(crate) post_ln: Vec<f32>,
}

pub(crate) struct LightConv {
    pub(crate) start: ClippableLinear,
    pub(crate) end: ClippableLinear,
    /// Depthwise kernel `[channels, kernel]`.
    pub(crate) conv_w: Vec<f32>,
    pub(crate) pre_ln: Vec<f32>,
    pub(crate) conv_norm: Vec<f32>,
}

pub(crate) struct Attention {
    pub(crate) q: ClippableLinear,
    pub(crate) k: ClippableLinear,
    pub(crate) v: ClippableLinear,
    pub(crate) post: ClippableLinear,
    pub(crate) rel_k: Linear,
    /// `softplus(per_dim_scale)`, precomputed.
    pub(crate) dim_scale: Vec<f32>,
}

pub(crate) struct Layer {
    pub(crate) ff1: FeedForward,
    pub(crate) ff2: FeedForward,
    pub(crate) attn: Attention,
    pub(crate) lconv: LightConv,
    pub(crate) norm_pre_attn: Vec<f32>,
    pub(crate) norm_post_attn: Vec<f32>,
    pub(crate) norm_out: Vec<f32>,
}

pub struct AudioTower {
    pub(crate) cfg: AudioConfig,
    pub(crate) conv0_w: Vec<f32>, // [c0, 9]
    pub(crate) norm0: Vec<f32>,
    pub(crate) conv1_w: Vec<f32>, // [c1, 9 * c0] with (kh, kw, cin) ordering
    pub(crate) norm1: Vec<f32>,
    pub(crate) in_proj: Linear,
    pub(crate) layers: Vec<Layer>,
    pub(crate) out_proj: Linear,
    pub(crate) embed_proj: Linear,
}

fn load_ff(w: &Weights, p: &str, cfg: &AudioConfig) -> Result<FeedForward> {
    let h = cfg.hidden_size;
    let clip = cfg.use_clipped_linears;
    Ok(FeedForward {
        ffw1: ClippableLinear::load(w, &format!("{p}.ffw_layer_1"), h * 4, h, clip)?,
        ffw2: ClippableLinear::load(w, &format!("{p}.ffw_layer_2"), h, h * 4, clip)?,
        pre_ln: w.f32_shaped(&format!("{p}.pre_layer_norm.weight"), &[h])?,
        post_ln: w.f32_shaped(&format!("{p}.post_layer_norm.weight"), &[h])?,
    })
}

/// Stride-2, kernel-3, pad-1 output length.
fn conv_out(n: usize) -> usize {
    (n + 2 - 3) / 2 + 1
}

/// Number of soft tokens the tower emits for `frames` mel frames.
pub fn num_audio_tokens(frames: usize) -> usize {
    if frames == 0 { 0 } else { conv_out(conv_out(frames)) }
}

/// `nn.LayerNorm(dim, bias=False)` over every `dim`-sized row (biased variance).
fn layer_norm_rows(x: &mut [f32], dim: usize, weight: &[f32], eps: f32) {
    x.par_chunks_mut(dim).for_each(|row| {
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / dim as f64;
        let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv = 1.0 / ((var as f32) + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v = ((*v - mean as f32) * inv) * *w;
        }
    });
}

impl AudioTower {
    pub fn load(cfg: &AudioConfig, text_hidden: usize, w: &Arc<Weights>) -> Result<Self> {
        if cfg.hidden_act != "silu" {
            bail!("unsupported audio activation `{}`", cfg.hidden_act);
        }
        ensure!(cfg.attention_context_right == 0, "right attention context is not supported by this port");
        ensure!(cfg.subsampling_conv_channels.len() == 2, "expected two subsampling conv layers");
        let h = cfg.hidden_size;
        let clip = cfg.use_clipped_linears;
        let (c0, c1) = (cfg.subsampling_conv_channels[0], cfg.subsampling_conv_channels[1]);
        let hd = h / cfg.num_attention_heads;

        let sub = "audio_tower.subsample_conv_projection";
        let conv1_raw = w.f32_shaped(&format!("{sub}.layer1.conv.weight"), &[c1, c0, 3, 3])?;
        // [c1, c0, 3, 3] -> [c1, 3, 3, c0] so the im2col gather copies contiguous channel blocks.
        let mut conv1_w = vec![0f32; conv1_raw.len()];
        for o in 0..c1 {
            for i in 0..c0 {
                for k in 0..9 {
                    conv1_w[o * 9 * c0 + k * c0 + i] = conv1_raw[(o * c0 + i) * 9 + k];
                }
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("audio_tower.layers.{i}");
            let dim_scale = w
                .f32_shaped(&format!("{p}.self_attn.per_dim_scale"), &[hd])?
                .into_iter()
                .map(ops::softplus)
                .collect();
            layers.push(Layer {
                ff1: load_ff(w, &format!("{p}.feed_forward1"), cfg)?,
                ff2: load_ff(w, &format!("{p}.feed_forward2"), cfg)?,
                attn: Attention {
                    q: ClippableLinear::load(w, &format!("{p}.self_attn.q_proj"), h, h, clip)?,
                    k: ClippableLinear::load(w, &format!("{p}.self_attn.k_proj"), h, h, clip)?,
                    v: ClippableLinear::load(w, &format!("{p}.self_attn.v_proj"), h, h, clip)?,
                    post: ClippableLinear::load(w, &format!("{p}.self_attn.post"), h, h, clip)?,
                    rel_k: Linear::load(w, &format!("{p}.self_attn.relative_k_proj"), h, h, false)?,
                    dim_scale,
                },
                lconv: LightConv {
                    start: ClippableLinear::load(w, &format!("{p}.lconv1d.linear_start"), h * 2, h, clip)?,
                    end: ClippableLinear::load(w, &format!("{p}.lconv1d.linear_end"), h, h, clip)?,
                    conv_w: w.f32_shaped(
                        &format!("{p}.lconv1d.depthwise_conv1d.weight"),
                        &[h, 1, cfg.conv_kernel_size],
                    )?,
                    pre_ln: w.f32_shaped(&format!("{p}.lconv1d.pre_layer_norm.weight"), &[h])?,
                    conv_norm: w.f32_shaped(&format!("{p}.lconv1d.conv_norm.weight"), &[h])?,
                },
                norm_pre_attn: w.f32_shaped(&format!("{p}.norm_pre_attn.weight"), &[h])?,
                norm_post_attn: w.f32_shaped(&format!("{p}.norm_post_attn.weight"), &[h])?,
                norm_out: w.f32_shaped(&format!("{p}.norm_out.weight"), &[h])?,
            });
        }

        Ok(Self {
            cfg: cfg.clone(),
            conv0_w: w.f32_shaped(&format!("{sub}.layer0.conv.weight"), &[c0, 1, 3, 3])?,
            norm0: w.f32_shaped(&format!("{sub}.layer0.norm.weight"), &[c0])?,
            conv1_w,
            norm1: w.f32_shaped(&format!("{sub}.layer1.norm.weight"), &[c1])?,
            in_proj: Linear::load(w, &format!("{sub}.input_proj_linear"), h, (c0 / 4) * c1, false)?,
            layers,
            out_proj: Linear::load(w, "audio_tower.output_proj", cfg.output_proj_dims, h, true)?,
            embed_proj: Linear::load(w, "embed_audio.embedding_projection", text_hidden, cfg.output_proj_dims, false)?,
        })
    }

    /// Stride-2 conv subsampling: `[frames, n_mels]` -> `[tokens, hidden]`.
    pub(crate) fn subsample(&self, mel: &[f32], frames: usize, n_mels: usize) -> Result<(Vec<f32>, usize)> {
        // Shared by the CPU and GPU towers, so both reject malformed input the same way.
        ensure!(frames > 0 && n_mels > 0 && mel.len() == frames * n_mels, "bad mel shape: {} values for {frames} x {n_mels}", mel.len());
        let (c0, c1) = (self.cfg.subsampling_conv_channels[0], self.cfg.subsampling_conv_channels[1]);

        // Layer 0: 1 -> c0 channels, input is a `[frames, n_mels]` image.
        let (t1, w1) = (conv_out(frames), conv_out(n_mels));
        let mut col0 = vec![0f32; t1 * w1 * 9];
        col0.par_chunks_mut(9).enumerate().for_each(|(pos, c)| {
            let (ty, fx) = (pos / w1, pos % w1);
            for kh in 0..3 {
                for kw in 0..3 {
                    let (ti, fi) = ((2 * ty + kh) as isize - 1, (2 * fx + kw) as isize - 1);
                    if ti >= 0 && (ti as usize) < frames && fi >= 0 && (fi as usize) < n_mels {
                        c[kh * 3 + kw] = mel[ti as usize * n_mels + fi as usize];
                    }
                }
            }
        });
        let mut y0 = ops::matmul_nt(&col0, t1 * w1, 9, &self.conv0_w, c0);
        drop(col0);
        layer_norm_rows(&mut y0, c0, &self.norm0, NORM_EPS);
        y0.par_iter_mut().for_each(|v| *v = v.max(0.0));

        // Layer 1: c0 -> c1 channels over the `[t1, w1]` grid, channels contiguous per position.
        let (t2, w2) = (conv_out(t1), conv_out(w1));
        let rows = t2 * w2;
        let k1 = 9 * c0;
        let mut y1 = vec![0f32; rows * c1];
        for (ci, chunk) in y1.chunks_mut(IM2COL_CHUNK * c1).enumerate() {
            let r0 = ci * IM2COL_CHUNK;
            let n = chunk.len() / c1;
            let mut col = vec![0f32; n * k1];
            col.par_chunks_mut(k1).enumerate().for_each(|(i, c)| {
                let pos = r0 + i;
                let (ty, fx) = (pos / w2, pos % w2);
                for kh in 0..3 {
                    for kw in 0..3 {
                        let (ti, fi) = ((2 * ty + kh) as isize - 1, (2 * fx + kw) as isize - 1);
                        if ti >= 0 && (ti as usize) < t1 && fi >= 0 && (fi as usize) < w1 {
                            let src = (ti as usize * w1 + fi as usize) * c0;
                            c[(kh * 3 + kw) * c0..(kh * 3 + kw + 1) * c0].copy_from_slice(&y0[src..src + c0]);
                        }
                    }
                }
            });
            chunk.copy_from_slice(&ops::matmul_nt(&col, n, k1, &self.conv1_w, c1));
        }
        drop(y0);
        layer_norm_rows(&mut y1, c1, &self.norm1, NORM_EPS);
        y1.par_iter_mut().for_each(|v| *v = v.max(0.0));

        // [t2, w2, c1] -> [t2, w2 * c1], then the input projection.
        ensure!(w2 * c1 == self.in_proj.inp, "unexpected mel width {n_mels}");
        Ok((self.in_proj.forward(&y1, t2), t2))
    }

    /// Sinusoidal relative positions `[context/2 + 1, hidden]` laid out as `[sin..., cos...]`,
    /// for distances `context/2, ..., 0`.
    pub(crate) fn rel_pos_emb(&self) -> (Vec<f32>, usize) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let context = cfg.attention_chunk_size + cfg.attention_context_left - 1 + cfg.attention_context_right;
        let n_pos = context / 2 + 1;
        let nts = h / 2;
        let inc = ((10000f64 / 1.0).ln() / (nts - 1).max(1) as f64) as f32;
        let inv: Vec<f32> = (0..nts).map(|i| (i as f32 * -inc).exp()).collect();
        let mut pe = vec![0f32; n_pos * h];
        for p in 0..n_pos {
            let pos = (context / 2 - p) as f32;
            for (i, iv) in inv.iter().enumerate() {
                let st = pos * iv;
                pe[p * h + i] = st.sin();
                pe[p * h + nts + i] = st.cos();
            }
        }
        (pe, n_pos)
    }

    /// Encode a log-mel spectrogram `[frames, n_mels]` into soft tokens `[tokens, text_hidden]`.
    pub fn encode(&self, mel: &[f32], frames: usize, n_mels: usize) -> Result<Vec<f32>> {
        ensure!(frames > 0 && mel.len() == frames * n_mels, "bad mel shape");
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let (mut x, t) = self.subsample(mel, frames, n_mels)?;

        let (pos_emb, n_pos) = self.rel_pos_emb();

        let gc = cfg.gradient_clipping.min(f32::MAX);
        let clamp = |v: &mut [f32]| v.par_iter_mut().for_each(|e| *e = e.clamp(-gc, gc));

        for l in &self.layers {
            x = self.feed_forward(&l.ff1, x, t, gc);

            let res = x.clone();
            clamp(&mut x);
            ops::rms_norm(&mut x, h, Some(&l.norm_pre_attn), NORM_EPS);
            let mut a = self.attention(&l.attn, &x, t, &pos_emb, n_pos);
            clamp(&mut a);
            ops::rms_norm(&mut a, h, Some(&l.norm_post_attn), NORM_EPS);
            x = res;
            ops::add_inplace(&mut x, &a);

            x = self.light_conv(&l.lconv, x, t, gc);
            x = self.feed_forward(&l.ff2, x, t, gc);
            clamp(&mut x);
            ops::rms_norm(&mut x, h, Some(&l.norm_out), NORM_EPS);
        }

        let mut out = self.out_proj.forward(&x, t);
        ops::rms_norm(&mut out, cfg.output_proj_dims, None, NORM_EPS);
        Ok(self.embed_proj.forward(&out, t))
    }

    fn feed_forward(&self, ff: &FeedForward, x: Vec<f32>, t: usize, gc: f32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let mut y: Vec<f32> = x.par_iter().map(|v| v.clamp(-gc, gc)).collect();
        ops::rms_norm(&mut y, h, Some(&ff.pre_ln), NORM_EPS);
        let mut y = ff.ffw1.forward(&y, t);
        y.par_iter_mut().for_each(|v| *v = ops::silu(*v));
        let mut y = ff.ffw2.forward(&y, t);
        y.par_iter_mut().for_each(|v| *v = v.clamp(-gc, gc));
        ops::rms_norm(&mut y, h, Some(&ff.post_ln), NORM_EPS);
        let rw = self.cfg.residual_weight;
        y.par_iter_mut().zip(x.par_iter()).for_each(|(v, r)| *v = *v * rw + *r);
        y
    }

    fn light_conv(&self, lc: &LightConv, x: Vec<f32>, t: usize, gc: f32) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let ks = self.cfg.conv_kernel_size;
        let mut y = x.clone();
        ops::rms_norm(&mut y, h, Some(&lc.pre_ln), NORM_EPS);
        let s = lc.start.forward(&y, t);
        // GLU: first half * sigmoid(second half).
        let mut g = vec![0f32; t * h];
        g.par_chunks_mut(h).enumerate().for_each(|(i, row)| {
            let src = &s[i * 2 * h..(i + 1) * 2 * h];
            for c in 0..h {
                row[c] = src[c] / (1.0 + (-src[h + c]).exp());
            }
        });
        // Causal depthwise conv: left pad = kernel - 1.
        let mut conv = vec![0f32; t * h];
        conv.par_chunks_mut(h).enumerate().for_each(|(ti, row)| {
            for c in 0..h {
                let mut acc = 0f32;
                for k in 0..ks {
                    let src = ti as isize - (ks as isize - 1) + k as isize;
                    if src >= 0 {
                        acc += lc.conv_w[c * ks + k] * g[src as usize * h + c];
                    }
                }
                row[c] = acc.clamp(-gc, gc);
            }
        });
        ops::rms_norm(&mut conv, h, Some(&lc.conv_norm), NORM_EPS);
        conv.par_iter_mut().for_each(|v| *v = ops::silu(*v));
        let mut out = lc.end.forward(&conv, t);
        ops::add_inplace(&mut out, &x);
        out
    }

    /// Chunked local attention with relative-position bias (`Gemma4AudioAttention`).
    fn attention(&self, a: &Attention, x: &[f32], t: usize, pos_emb: &[f32], n_pos: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let hd = h / nh;
        let chunk = cfg.attention_chunk_size;
        let past = cfg.attention_context_left - 1;
        let ctx = chunk + past + cfg.attention_context_right;
        let cap = cfg.attention_logit_cap;
        let invalid = cfg.attention_invalid_logits_value;

        let q_scale = (hd as f32).powf(-0.5) / std::f32::consts::LN_2;
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;

        let mut q = a.q.forward(x, t);
        let mut k = a.k.forward(x, t);
        let v = a.v.forward(x, t);
        q.par_chunks_mut(h).for_each(|row| {
            for hh in 0..nh {
                for d in 0..hd {
                    row[hh * hd + d] *= q_scale * a.dim_scale[d];
                }
            }
        });
        k.par_iter_mut().for_each(|e| *e *= k_scale);
        let rel = a.rel_k.forward(pos_emb, n_pos); // [n_pos, h]

        let n_blocks = t.div_ceil(chunk);
        let mut out = vec![0f32; t * h];
        out.par_chunks_mut(chunk * h).enumerate().for_each(|(b, block)| {
            let rows = block.len() / h;
            let mut logits = vec![0f32; ctx];
            let mut qrel = vec![0f32; chunk * n_pos];
            for head in 0..nh {
                // matrix_bd before the relative shift: q(row r) . rel[p] for every block row.
                for r in 0..chunk {
                    let qi = b * chunk + r;
                    for p in 0..n_pos {
                        qrel[r * n_pos + p] = if qi < t {
                            let qv = &q[qi * h + head * hd..qi * h + (head + 1) * hd];
                            let rv = &rel[p * h + head * hd..p * h + (head + 1) * hd];
                            qv.iter().zip(rv).map(|(a, b)| a * b).sum()
                        } else {
                            0.0
                        };
                    }
                }
                for i in 0..rows {
                    let qi = b * chunk + i;
                    let qv = &q[qi * h + head * hd..qi * h + (head + 1) * hd];
                    for (j, lg) in logits.iter_mut().enumerate() {
                        let kv = (b * chunk + j) as isize - past as isize;
                        let dist = qi as isize - kv;
                        let allowed = kv >= 0 && (kv as usize) < t && dist >= 0 && (dist as usize) < past;
                        *lg = if allowed {
                            let kvec = &k[kv as usize * h + head * hd..kv as usize * h + (head + 1) * hd];
                            let ac: f32 = qv.iter().zip(kvec).map(|(a, b)| a * b).sum();
                            // Transformer-XL relative shift: flatten the padded [chunk, ctx + 1] matrix.
                            let f = i * ctx + j;
                            let (r, c) = (f / (ctx + 1), f % (ctx + 1));
                            let bd = if c < n_pos { qrel[r * n_pos + c] } else { 0.0 };
                            ((ac + bd) / cap).tanh() * cap
                        } else {
                            invalid
                        };
                    }
                    ops::softmax_row(&mut logits);
                    let dst = &mut block[i * h + head * hd..i * h + (head + 1) * hd];
                    dst.fill(0.0);
                    for (j, p) in logits.iter().enumerate() {
                        let kv = (b * chunk + j) as isize - past as isize;
                        if *p == 0.0 || kv < 0 || kv as usize >= t {
                            continue;
                        }
                        let vv = &v[kv as usize * h + head * hd..kv as usize * h + (head + 1) * hd];
                        for (d, vd) in dst.iter_mut().zip(vv) {
                            *d += *p * *vd;
                        }
                    }
                }
            }
        });
        debug_assert!(n_blocks * chunk >= t);
        a.post.forward(&out, t)
    }
}
