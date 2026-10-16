//! Vision tower on the GPU (mirrors `crate::vision::VisionTower::encode`). Position tables,
//! RoPE tables and the final pooling/projection stay on the host (they are cheap).
//!
//! Like the text tower, a layer is a handful of fused launches working in scratch buffers that are
//! allocated once per image: q/k/v and gate/up are single GEMMs when no clamping is configured,
//! and the norm / RoPE / residual steps are the shared fused kernels.

use anyhow::Result;
use std::sync::Arc;

use super::{Buf, Gpu, GpuLinear};
use crate::imageproc::PatchedImage;
use crate::vision::{VisionHead, VisionTower};

/// q, k, v projections: one fused GEMM when none of them clamps, else three.
#[allow(clippy::large_enum_variant)]
enum QkvProj {
    Fused(GpuLinear),
    Split(GpuLinear, GpuLinear, GpuLinear),
}

/// gate and up projections: one fused GEMM when neither clamps, else two.
#[allow(clippy::large_enum_variant)]
enum GateUp {
    Fused(GpuLinear),
    Split(GpuLinear, GpuLinear),
}

struct Layer {
    qkv: QkvProj,
    o_proj: GpuLinear,
    q_norm: Buf,
    k_norm: Buf,
    gate_up: GateUp,
    down_proj: GpuLinear,
    input_ln: Buf,
    post_attn_ln: Buf,
    pre_ff_ln: Buf,
    post_ff_ln: Buf,
}

pub struct GpuVision {
    head: VisionHead,
    gpu: Arc<Gpu>,
    input_proj: GpuLinear,
    layers: Vec<Layer>,
}

impl GpuVision {
    pub fn from_cpu(cpu: VisionTower, gpu: Arc<Gpu>) -> Result<Self> {
        let VisionTower { head, input_proj, layers } = cpu;
        let g = &*gpu;
        // Uploads go through the shared stream: keep them out of another thread's pass (or graph recording).
        let lock_owner = gpu.clone();
        let _guard = lock_owner.lock();
        let plain = |ls: &[&crate::ops::ClippableLinear]| ls.iter().all(|l| l.clip.is_none() && l.linear.bias.is_none());
        let layers = layers
            .iter()
            .map(|l| {
                let qkv = if plain(&[&l.q_proj, &l.k_proj, &l.v_proj]) {
                    QkvProj::Fused(GpuLinear::concat(g, &[&l.q_proj.linear, &l.k_proj.linear, &l.v_proj.linear])?)
                } else {
                    QkvProj::Split(GpuLinear::from_clippable(g, &l.q_proj)?, GpuLinear::from_clippable(g, &l.k_proj)?, GpuLinear::from_clippable(g, &l.v_proj)?)
                };
                let gate_up = if plain(&[&l.gate_proj, &l.up_proj]) {
                    GateUp::Fused(GpuLinear::concat(g, &[&l.gate_proj.linear, &l.up_proj.linear])?)
                } else {
                    GateUp::Split(GpuLinear::from_clippable(g, &l.gate_proj)?, GpuLinear::from_clippable(g, &l.up_proj)?)
                };
                Ok(Layer {
                    qkv,
                    o_proj: GpuLinear::from_clippable(g, &l.o_proj)?,
                    q_norm: g.upload(&l.q_norm)?,
                    k_norm: g.upload(&l.k_norm)?,
                    gate_up,
                    down_proj: GpuLinear::from_clippable(g, &l.down_proj)?,
                    input_ln: g.upload(&l.input_ln)?,
                    post_attn_ln: g.upload(&l.post_attn_ln)?,
                    pre_ff_ln: g.upload(&l.pre_ff_ln)?,
                    post_ff_ln: g.upload(&l.post_ff_ln)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { input_proj: GpuLinear::from_cpu(g, &input_proj)?, head, layers, gpu })
    }

    pub fn encode(&self, img: &PatchedImage) -> Result<Vec<f32>> {
        let head = &self.head;
        head.check(img)?;
        let cfg = &head.cfg;
        let (h, hd, nh, nkv, ff) = (cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size);
        let n = img.n_patches();
        let eps = cfg.rms_norm_eps;

        // Position embeddings are added on the host to the projected patches: same arithmetic
        // as the CPU path (projection first, then `+ pos_x + pos_y`).
        let mut pos = vec![0f32; n * h];
        head.add_positions(&mut pos, img);
        let (cos, sin) = head.rope_tables(img);

        // Host-side (rayon) work happens before taking the device lock: the lock is not re-entrant,
        // and a rayon worker that blocks on it could otherwise be asked to run another embed call.
        let scaled = VisionHead::scaled_patches(img);
        let g = &*self.gpu;
        let hidden_host = {
            let _guard = g.lock();
            g.mark("start");
            let x = g.upload(&scaled)?;
            let mut hidden = self.input_proj.forward(g, &x, n)?;
            g.add(&mut hidden, &g.upload(&pos)?)?;
            let (cos, sin) = (g.upload(&cos)?, g.upload(&sin)?);

            // Scratch buffers shared by all layers (the fused-projection ones only when some layer uses them).
            // In bf16 mode the products' inputs and outputs are bf16, except the outputs of layers with a bias
            // or clamp (none in EmbeddingGemma 2's vision tower), whose epilogues run on float32.
            let fused_qkv = self.layers.iter().any(|l| matches!(l.qkv, QkvProj::Fused(_)));
            let fused_gu = self.layers.iter().any(|l| matches!(l.gate_up, GateUp::Fused(_)));
            let split_qkv = self.layers.iter().any(|l| matches!(l.qkv, QkvProj::Split(..)));
            let split_gu = self.layers.iter().any(|l| matches!(l.gate_up, GateUp::Split(..)));
            let plain = |w: &GpuLinear| w.bias.is_none() && w.clip.is_none();
            let o_plain = self.layers.iter().all(|l| plain(&l.o_proj) && plain(&l.down_proj));
            let bf = g.bf16_act();
            let mut t = g.alloc_act(n * h)?;
            let mut qkv = g.alloc_act(if fused_qkv { n * (nh + 2 * nkv) * hd } else { 0 })?;
            let mut q = g.alloc_act(n * nh * hd)?;
            let mut k = g.alloc_act(n * nkv * hd)?;
            let mut v = g.alloc_act(n * nkv * hd)?;
            // Split (clipped) projections in bf16 mode: their float32 outputs, before norm + RoPE.
            let mut qkv_split = if bf && split_qkv {
                Some([g.alloc_act_f32(n * nh * hd)?, g.alloc_act_f32(n * nkv * hd)?, g.alloc_act_f32(n * nkv * hd)?])
            } else {
                None
            };
            let mut a = g.alloc_act(n * nh * hd)?;
            let mut o = if o_plain { g.alloc_act(n * h)? } else { g.alloc_act_f32(n * h)? };
            let mut gu = g.alloc_act(if fused_gu { n * 2 * ff } else { 0 })?;
            let mut gate = g.alloc_act(n * ff)?;
            // Split gate / up: the float32 up output (and in bf16 mode the float32 gate output) before the GELU.
            let mut up = g.alloc_act_f32(if split_gu { n * ff } else { 0 })?;
            let mut gate_f = g.alloc_act_f32(if bf && split_gu { n * ff } else { 0 })?;

            if let Some(l) = self.layers.first() {
                g.rms_norm_act(&hidden, &mut t, n, h, &l.input_ln, eps)?;
            }
            g.mark("vision: input proj + first norm");
            for (i, l) in self.layers.iter().enumerate() {
                let tabs = [&l.q_norm, &l.k_norm, &cos, &sin];
                match (&l.qkv, &mut qkv_split) {
                    (QkvProj::Fused(w), _) => {
                        w.forward_act(g, &t, n, &mut qkv)?;
                        g.mark("vision: GEMM qkv");
                        g.qkv_post_act(&qkv, &mut q, &mut k, &mut v, tabs, None, n, nh, nkv, hd, hd / 2, eps)?;
                    }
                    (QkvProj::Split(wq, wk, wv), Some([qf, kf, vf])) => {
                        wq.forward_act(g, &t, n, qf)?;
                        wk.forward_act(g, &t, n, kf)?;
                        wv.forward_act(g, &t, n, vf)?;
                        g.mark("vision: GEMM qkv");
                        g.qkv_post_split_act([&*qf, &*kf, &*vf], &mut q, &mut k, &mut v, tabs, None, n, nh, nkv, hd, hd / 2, eps)?;
                    }
                    (QkvProj::Split(wq, wk, wv), None) => {
                        wq.forward_act(g, &t, n, &mut q)?;
                        wk.forward_act(g, &t, n, &mut k)?;
                        wv.forward_act(g, &t, n, &mut v)?;
                        g.mark("vision: GEMM qkv");
                        g.qkv_post(None, q.f32_mut()?, k.f32_mut()?, v.f32_mut()?, &l.q_norm, &l.k_norm, &cos, &sin, None, n, nh, nkv, hd, hd / 2, eps)?;
                    }
                }
                g.mark("vision: qkv_post (norms + RoPE)");
                g.attention_act(&q, &k, &v, &mut a, 0, n, nh, nkv, hd, None)?;
                g.mark("vision: attn: flash");
                l.o_proj.forward_act(g, &a, n, &mut o)?;
                g.mark("vision: GEMM o");
                g.add_norm_act(&mut hidden, &o, &l.post_attn_ln, Some((&mut t, Some(&l.pre_ff_ln))), n, h, eps, 1.0)?;
                g.mark("vision: add_norm");

                match &l.gate_up {
                    GateUp::Fused(w) => {
                        w.forward_act(g, &t, n, &mut gu)?;
                        g.mark("vision: GEMM gate|up");
                        g.gelu_mul_gu_act(&mut gate, &gu, n, ff)?;
                    }
                    GateUp::Split(wg, wu) => {
                        wu.forward_act(g, &t, n, &mut up)?;
                        if bf {
                            wg.forward_act(g, &t, n, &mut gate_f)?;
                            g.mark("vision: GEMM gate|up");
                            g.gelu_mul_split_act(&mut gate, &gate_f, &up, n, ff)?;
                        } else {
                            wg.forward_act(g, &t, n, &mut gate)?;
                            g.mark("vision: GEMM gate|up");
                            g.gelu_mul(gate.f32_mut()?, up.f32()?, n * ff)?;
                        }
                    }
                }
                g.mark("vision: gelu_mul");
                l.down_proj.forward_act(g, &gate, n, &mut o)?;
                g.mark("vision: GEMM down");
                // The last layer's output is the tower output; every other one feeds the next layer's pre-norm.
                let next = self.layers.get(i + 1).map(|nl| (&mut t, Some(&nl.input_ln)));
                g.add_norm_act(&mut hidden, &o, &l.post_ff_ln, next, n, h, eps, 1.0)?;
                g.mark("vision: add_norm");
            }
            g.download(&hidden)?
        };
        Ok(head.pool_and_embed(&hidden_host, img))
    }
}
