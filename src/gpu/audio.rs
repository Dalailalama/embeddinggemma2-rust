//! Audio conformer on the GPU (mirrors `crate::audio::AudioTower::encode`): the two stride-2
//! subsampling convolutions, the 12 conformer layers and the output projections all run on the device.
//!
//! The clipped linears (`Gemma4ClippableLinear`) clamp their input and output; here those clamps are
//! folded into the neighbouring kernels (norms, activations, residual adds) instead of being separate
//! passes, and all scratch buffers are allocated once per clip.

use anyhow::{Result, ensure};
use std::sync::Arc;

use super::{AddNorm, Buf, Gpu, GpuLinear, NO_CLAMP, NormOpts};
use crate::audio::{AudioTower, FeedForward};

const NORM_EPS: f32 = 1e-6;

/// Clamp to `a`, then to `b`: the same function as the single clamp returned.
fn compose(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let c = |x: f32| x.max(b.0).min(b.1);
    (c(a.0), c(a.1))
}

fn clip_in(l: &GpuLinear) -> (f32, f32) {
    l.clip.map_or(NO_CLAMP, |c| (c[0], c[1]))
}

fn clip_out(l: &GpuLinear) -> (f32, f32) {
    l.clip.map_or(NO_CLAMP, |c| (c[2], c[3]))
}

struct Ff {
    ffw1: GpuLinear,
    ffw2: GpuLinear,
    pre_ln: Buf,
    post_ln: Buf,
}

/// q, k, v projections: one GEMM when they clamp their (shared, normed) input identically, else three.
#[allow(clippy::large_enum_variant)]
enum Qkv {
    Fused { w: GpuLinear, in_clip: (f32, f32), out_clip: [(f32, f32); 3] },
    Split(GpuLinear, GpuLinear, GpuLinear),
}

impl Qkv {
    fn in_clip(&self) -> (f32, f32) {
        match self {
            Qkv::Fused { in_clip, .. } => *in_clip,
            Qkv::Split(q, ..) => clip_in(q),
        }
    }
}

struct Layer {
    ff1: Ff,
    ff2: Ff,
    qkv: Qkv,
    post: GpuLinear,
    /// `relative_k_proj(pos_emb)`: depends only on the weights, so it is computed once at load time.
    rel: Buf,
    dim_scale: Buf,
    start: GpuLinear,
    end: GpuLinear,
    conv_w: Buf,
    lc_pre_ln: Buf,
    conv_norm: Buf,
    norm_pre_attn: Buf,
    norm_post_attn: Buf,
    norm_out: Buf,
}

/// Per-clip scratch buffers shared by all layers.
struct Scratch {
    /// Normed input of the current sub-block.
    y: Buf,
    /// Feed-forward hidden.
    z: Buf,
    /// Output of a projection that feeds a residual add.
    u: Buf,
    qkv: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    qrel: Buf,
    att: Buf,
    s: Buf,
    gl: Buf,
    c: Buf,
}

pub struct GpuAudio {
    /// Host-side config; its layer list is emptied.
    front: AudioTower,
    gpu: Arc<Gpu>,
    layers: Vec<Layer>,
    // Subsampling convolutions and output projections.
    conv0_w: Buf,
    norm0: Buf,
    conv1: GpuLinear,
    norm1: Buf,
    in_proj: GpuLinear,
    out_proj: GpuLinear,
    embed_proj: GpuLinear,
    n_pos: usize,
}

fn upload_ff(g: &Gpu, f: &FeedForward) -> Result<Ff> {
    Ok(Ff {
        ffw1: GpuLinear::from_clippable(g, &f.ffw1)?,
        ffw2: GpuLinear::from_clippable(g, &f.ffw2)?,
        pre_ln: g.upload(&f.pre_ln)?,
        post_ln: g.upload(&f.post_ln)?,
    })
}

impl GpuAudio {
    pub fn from_cpu(mut cpu: AudioTower, gpu: Arc<Gpu>) -> Result<Self> {
        let g = &*gpu;
        // Uploads go through the shared stream: keep them out of another thread's pass (or graph recording).
        let lock_owner = gpu.clone();
        let _guard = lock_owner.lock();
        let cpu_layers = std::mem::take(&mut cpu.layers);
        let (pe, n_pos) = cpu.rel_pos_emb();
        let pos_emb = g.upload(&pe)?;
        let layers = cpu_layers
            .iter()
            .map(|l| {
                let (q, k, v) = (&l.attn.q, &l.attn.k, &l.attn.v);
                let in_of = |p: &crate::ops::ClippableLinear| p.clip.map_or(NO_CLAMP, |c| (c[0], c[1]));
                let out_of = |p: &crate::ops::ClippableLinear| p.clip.map_or(NO_CLAMP, |c| (c[2], c[3]));
                let same_in = in_of(q) == in_of(k) && in_of(k) == in_of(v);
                let qkv = if same_in && [q, k, v].iter().all(|p| p.linear.bias.is_none()) {
                    let w = GpuLinear::concat(g, &[&q.linear, &k.linear, &v.linear])?;
                    Qkv::Fused { w, in_clip: in_of(q), out_clip: [out_of(q), out_of(k), out_of(v)] }
                } else {
                    Qkv::Split(GpuLinear::from_clippable(g, q)?, GpuLinear::from_clippable(g, k)?, GpuLinear::from_clippable(g, v)?)
                };
                let rel_k = GpuLinear::from_cpu_f32(g, &l.attn.rel_k)?;
                Ok(Layer {
                    ff1: upload_ff(g, &l.ff1)?,
                    ff2: upload_ff(g, &l.ff2)?,
                    qkv,
                    post: GpuLinear::from_clippable(g, &l.attn.post)?,
                    rel: rel_k.forward(g, &pos_emb, n_pos)?,
                    dim_scale: g.upload(&l.attn.dim_scale)?,
                    start: GpuLinear::from_clippable(g, &l.lconv.start)?,
                    end: GpuLinear::from_clippable(g, &l.lconv.end)?,
                    conv_w: g.upload(&l.lconv.conv_w)?,
                    lc_pre_ln: g.upload(&l.lconv.pre_ln)?,
                    conv_norm: g.upload(&l.lconv.conv_norm)?,
                    norm_pre_attn: g.upload(&l.norm_pre_attn)?,
                    norm_post_attn: g.upload(&l.norm_post_attn)?,
                    norm_out: g.upload(&l.norm_out)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (c0, c1) = (cpu.cfg.subsampling_conv_channels[0], cpu.cfg.subsampling_conv_channels[1]);
        Ok(Self {
            conv0_w: g.upload(&cpu.conv0_w)?,
            norm0: g.upload(&cpu.norm0)?,
            conv1: GpuLinear { w: super::LinW::F32(g.upload(&cpu.conv1_w)?), out: c1, inp: 9 * c0, bias: None, clip: None },
            norm1: g.upload(&cpu.norm1)?,
            in_proj: GpuLinear::from_cpu_f32(g, &cpu.in_proj)?,
            out_proj: GpuLinear::from_cpu(g, &cpu.out_proj)?,
            embed_proj: GpuLinear::from_cpu(g, &cpu.embed_proj)?,
            n_pos,
            layers,
            front: cpu,
            gpu,
        })
    }

    /// Stride-2 convolution subsampling on the device: `mel [frames, n_mels]` -> `[tokens, hidden]`.
    fn subsample(&self, mel: &[f32], frames: usize, n_mels: usize) -> Result<(Buf, usize)> {
        // Same contract as the CPU front end: frames = 0 would underflow the conv arithmetic below.
        ensure!(frames > 0 && n_mels > 0 && mel.len() == frames * n_mels, "bad mel shape: {} values for {frames} x {n_mels}", mel.len());
        let g = &*self.gpu;
        let (c0, c1) = (self.front.cfg.subsampling_conv_channels[0], self.front.cfg.subsampling_conv_channels[1]);
        let conv_out = |n: usize| (n + 2 - 3) / 2 + 1;
        let (t1, w1) = (conv_out(frames), conv_out(n_mels));
        let (t2, w2) = (conv_out(t1), conv_out(w1));
        ensure!(w2 * c1 == self.in_proj.inp, "unexpected mel width {n_mels}");
        g.exact_math(true)?;
        let mel = g.upload(mel)?;
        let mut y0 = g.alloc(t1 * w1 * c0)?;
        g.audio_conv0(&mel, &self.conv0_w, &self.norm0, &mut y0, frames, n_mels, t1, w1, c0, NORM_EPS)?;
        let rows = t2 * w2;
        let mut col = g.alloc(rows * 9 * c0)?;
        g.audio_im2col1(&y0, &mut col, t1, w1, c0, t2, w2)?;
        let mut y1 = g.alloc(rows * c1)?;
        self.conv1.forward_into(g, &col, rows, &mut y1)?;
        g.ln_relu(&mut y1, &self.norm1, rows, c1, NORM_EPS)?;
        // [t2, w2, c1] is already [t2, w2 * c1]: project each time step into the conformer width.
        let x = self.in_proj.forward(g, &y1, t2)?;
        g.exact_math(false)?;
        Ok((x, t2))
    }

    /// Feed-forward block on `x`, reading its pre-normed input from `sc.y`. `next` = (weights, output clamp) of
    /// the following sub-block's pre-norm, which is then left in `sc.y`.
    fn feed_forward(&self, f: &Ff, x: &mut Buf, sc: &mut Scratch, t: usize, next: Option<(&Buf, (f32, f32))>) -> Result<()> {
        let g = &*self.gpu;
        let cfg = &self.front.cfg;
        let h = cfg.hidden_size;
        let gc = cfg.gradient_clipping.min(f32::MAX);
        f.ffw1.forward_raw(g, &sc.y, t, &mut sc.z)?;
        g.act_chain(&mut sc.z, t * 4 * h, clip_out(&f.ffw1), clip_in(&f.ffw2))?;
        f.ffw2.forward_raw(g, &sc.z, t, &mut sc.u)?;
        // x = rmsnorm(clamp(u)) * residual_weight + x, then the next pre-norm of (clamp(x)).
        let tail = AddNorm {
            w1: Some(&f.post_ln),
            scalar: cfg.residual_weight,
            axpy: true,
            clamp_o: compose(clip_out(&f.ffw2), (-gc, gc)),
            next_in: (-gc, gc),
            ..Default::default()
        };
        match next {
            Some((w2, out_clip)) => g.add_norm_ex(x, &sc.u, Some((&mut sc.y, w2)), t, h, NORM_EPS, AddNorm { next_out: out_clip, ..tail }),
            None => g.add_norm_ex(x, &sc.u, None, t, h, NORM_EPS, tail),
        }
    }

    pub fn encode(&self, mel: &[f32], frames: usize, n_mels: usize) -> Result<Vec<f32>> {
        let cfg = &self.front.cfg;
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let hd = h / nh;
        let gc = cfg.gradient_clipping.min(f32::MAX);
        let nogc = (-gc, gc);
        let chunk = cfg.attention_chunk_size;
        let past = cfg.attention_context_left - 1;
        let ctx = chunk + past + cfg.attention_context_right;
        let q_scale = (hd as f32).powf(-0.5) / std::f32::consts::LN_2;
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        let n_pos = self.n_pos;

        let g = &*self.gpu;
        let _guard = g.lock();
        let (mut x, t) = self.subsample(mel, frames, n_mels)?;

        let mut sc = Scratch {
            y: g.alloc(t * h)?,
            z: g.alloc(t * 4 * h)?,
            u: g.alloc(t * h)?,
            qkv: g.alloc(t * 3 * h)?,
            q: g.alloc(t * h)?,
            k: g.alloc(t * h)?,
            v: g.alloc(t * h)?,
            qrel: g.alloc(t * nh * n_pos)?,
            att: g.alloc(t * h)?,
            s: g.alloc(t * 2 * h)?,
            gl: g.alloc(t * h)?,
            c: g.alloc(t * h)?,
        };

        // Pre-norm of the first feed-forward block: clamp(rmsnorm(clamp(x)) * w, input clamp of ffw1).
        let ff_pre = |g: &Gpu, x: &Buf, y: &mut Buf, l: &Layer| {
            g.rms_norm_opts(Some(x), y, t, h, Some(&l.ff1.pre_ln), NORM_EPS, NormOpts { clamp_in: nogc, clamp_out: clip_in(&l.ff1.ffw1), ..Default::default() })
        };
        ff_pre(g, &x, &mut sc.y, &self.layers[0])?;
        for (i, l) in self.layers.iter().enumerate() {
            self.feed_forward(&l.ff1, &mut x, &mut sc, t, Some((&l.norm_pre_attn, l.qkv.in_clip())))?;

            // Attention sub-block (its normed input is in sc.y).
            match &l.qkv {
                Qkv::Fused { w, out_clip, .. } => {
                    w.forward_raw(g, &sc.y, t, &mut sc.qkv)?;
                    g.audio_qkv_post(&sc.qkv, &mut sc.q, &mut sc.k, &mut sc.v, &l.dim_scale, t, h, hd, q_scale, k_scale, *out_clip)?;
                }
                Qkv::Split(wq, wk, wv) => {
                    wq.forward_into(g, &sc.y, t, &mut sc.q)?;
                    wk.forward_into(g, &sc.y, t, &mut sc.k)?;
                    wv.forward_into(g, &sc.y, t, &mut sc.v)?;
                    g.audio_qscale(&mut sc.q, &l.dim_scale, q_scale, t, nh, hd)?;
                    g.scale(&mut sc.k, k_scale)?;
                }
            }
            g.audio_qrel(&sc.q, &l.rel, &mut sc.qrel, t, nh, hd, n_pos)?;
            // The post projection clamps its input: fold that into the attention kernel's store.
            g.audio_attn(&sc.q, &sc.k, &sc.v, &sc.qrel, &mut sc.att, t, nh, hd, chunk, past, ctx, n_pos, cfg.attention_logit_cap, cfg.attention_invalid_logits_value, clip_in(&l.post))?;
            l.post.forward_raw(g, &sc.att, t, &mut sc.u)?;
            // x += rmsnorm(clamp(clamp(u, post out clip), +-gc)) * norm_post_attn; sc.y = the light conv's pre-norm.
            let tail = AddNorm { w1: Some(&l.norm_post_attn), clamp_o: compose(clip_out(&l.post), nogc), next_out: clip_in(&l.start), ..Default::default() };
            g.add_norm_ex(&mut x, &sc.u, Some((&mut sc.y, &l.lc_pre_ln)), t, h, NORM_EPS, tail)?;

            // Light convolution sub-block.
            l.start.forward_raw(g, &sc.y, t, &mut sc.s)?;
            g.glu(&mut sc.gl, &sc.s, t, h, clip_out(&l.start))?;
            g.dwconv_causal(&mut sc.c, &sc.gl, &l.conv_w, t, h, cfg.conv_kernel_size, gc)?;
            g.rms_norm_opts(None, &mut sc.c, t, h, Some(&l.conv_norm), NORM_EPS, NormOpts { silu: true, clamp_out: clip_in(&l.end), ..Default::default() })?;
            l.end.forward_raw(g, &sc.c, t, &mut sc.u)?;
            // x += clamp(u, end out clip); sc.y = the second feed-forward block's pre-norm.
            let tail = AddNorm { clamp_o: clip_out(&l.end), next_in: nogc, next_out: clip_in(&l.ff2.ffw1), ..Default::default() };
            g.add_norm_ex(&mut x, &sc.u, Some((&mut sc.y, &l.ff2.pre_ln)), t, h, NORM_EPS, tail)?;

            self.feed_forward(&l.ff2, &mut x, &mut sc, t, None)?;
            g.rms_norm_opts(None, &mut x, t, h, Some(&l.norm_out), NORM_EPS, NormOpts { clamp_in: nogc, ..Default::default() })?;
            if let Some(nl) = self.layers.get(i + 1) {
                ff_pre(g, &x, &mut sc.y, nl)?;
            }
        }
        let mut out = self.out_proj.forward(g, &x, t)?;
        g.rms_norm(&mut out, t, cfg.output_proj_dims, None, NORM_EPS)?;
        let emb = self.embed_proj.forward(g, &out, t)?;
        g.download(&emb)
    }
}
