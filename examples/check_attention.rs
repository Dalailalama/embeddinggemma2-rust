//! Correctness check of the bf16 tensor-core attention kernels against the cuBLAS path and an exact float64
//! reference, on random inputs: tile-edge sequence lengths, with and without a window, logits from smooth to very
//! peaked, and the sequence placed at an offset inside a larger buffer (the rows around it must stay untouched).
//! Run with: cargo run --release --example check_attention

use embeddinggemma2::Precision;
use embeddinggemma2::gpu::{Act, Gpu};

/// Deterministic normal samples (xorshift + Box-Muller).
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    f32::from_bits(((u + 0x7fff + ((u >> 16) & 1)) >> 16) << 16)
}

#[allow(clippy::too_many_arguments)]
fn reference(q: &[f32], k: &[f32], v: &[f32], row0: usize, seq: usize, nh: usize, nkv: usize, hd: usize, window: Option<usize>) -> Vec<f64> {
    let (qs, kvs) = (nh * hd, nkv * hd);
    let mut out = vec![0f64; seq * qs];
    for h in 0..nh {
        let kh = h / (nh / nkv);
        for i in 0..seq {
            let keys: Vec<usize> = (0..seq).filter(|&j| window.is_none_or(|w| i.abs_diff(j) <= w)).collect();
            let logits: Vec<f64> = keys
                .iter()
                .map(|&j| (0..hd).map(|d| q[(row0 + i) * qs + h * hd + d] as f64 * k[(row0 + j) * kvs + kh * hd + d] as f64).sum())
                .collect();
            let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
            let sum: f64 = p.iter().sum();
            for d in 0..hd {
                out[i * qs + h * hd + d] = keys.iter().zip(&p).map(|(&j, p)| p * v[(row0 + j) * kvs + kh * hd + d] as f64).sum::<f64>() / sum;
            }
        }
    }
    out
}

fn main() -> anyhow::Result<()> {
    let mut g = Gpu::new(0, Precision::Bf16)?;
    let mut failures = 0;
    // The text backbone's sliding-window layers, and the vision tower.
    failures += check(&mut g, 4, 2, 256, &[1, 5, 31, 32, 33, 63, 64, 65, 127, 300, 700, 1100, 2208], &[None, Some(512), Some(7)])?;
    failures += check(&mut g, 12, 12, 64, &[1, 17, 63, 64, 65, 129, 700, 2394], &[None, Some(7)])?;
    anyhow::ensure!(failures == 0, "{failures} attention checks failed");
    Ok(())
}

fn check(g: &mut Gpu, nh: usize, nkv: usize, hd: usize, seqs: &[usize], windows: &[Option<usize>]) -> anyhow::Result<usize> {
    let (qs, kvs) = (nh * hd, nkv * hd);
    let (pad_before, pad_after) = (3usize, 2usize);
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut worst_flash = 0f64;
    let mut worst_cublas = 0f64;
    let mut failures = 0;
    // Logit spread: q.k over hd channels of N(0, sigma^2) values has standard deviation sigma^2 sqrt(hd).
    let base = (hd as f64).powf(-0.25);
    println!("\nheads {nh} / K/V heads {nkv} x {hd}");
    println!("{:>5} {:>7} {:>6}  {:>12} {:>12} {:>12}", "seq", "window", "sigma", "flash-cuBLAS", "flash-exact", "cuBLAS-exact");
    for &sigma in &[base, 2.0 * base, 4.0 * base] {
        for &seq in seqs {
            for &window in windows {
                let rows = pad_before + seq + pad_after;
                let mut sample = |n: usize, s: f64| -> Vec<f32> { (0..n).map(|_| bf16_round((rng.normal() * s) as f32)).collect() };
                let (q, k, v) = (sample(rows * qs, sigma), sample(rows * kvs, sigma), sample(rows * kvs, 1.0));
                let (qa, ka, va) = (Act::Bf16(g.upload_bf16(&q)?), Act::Bf16(g.upload_bf16(&k)?), Act::Bf16(g.upload_bf16(&v)?));
                let sentinel = vec![7.0f32; rows * qs];
                let mut outs = Vec::new();
                for flash in [true, false] {
                    g.set_flash(flash);
                    let mut o = Act::Bf16(g.upload_bf16(&sentinel)?);
                    g.attention_act(&qa, &ka, &va, &mut o, pad_before, seq, nh, nkv, hd, window)?;
                    outs.push(g.download_act(&o)?);
                }
                let (fl, cb) = (&outs[0], &outs[1]);
                // Rows outside the sequence must be untouched.
                let outside_ok = [fl, cb].iter().all(|o| {
                    o[..pad_before * qs].iter().chain(&o[(pad_before + seq) * qs..]).all(|&x| x == 7.0)
                });
                let inside = pad_before * qs..(pad_before + seq) * qs;
                // Errors relative to max(1, |value|): bf16 outputs are only good to 2^-8 of their magnitude.
                let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
                let d_fc = fl[inside.clone()].iter().zip(&cb[inside.clone()]).map(|(a, b)| rel(*a as f64, *b as f64)).fold(0.0, f64::max);
                let (d_fe, d_ce) = if seq <= 700 {
                    let r = reference(&q, &k, &v, pad_before, seq, nh, nkv, hd, window);
                    let e = |o: &[f32]| o[inside.clone()].iter().zip(&r).map(|(a, b)| rel(*a as f64, *b)).fold(0.0, f64::max);
                    (e(fl), e(cb))
                } else {
                    (f64::NAN, f64::NAN)
                };
                if d_fe.is_finite() {
                    worst_flash = worst_flash.max(d_fe);
                    worst_cublas = worst_cublas.max(d_ce);
                }
                // A few bf16 steps (2^-8 relative each) of disagreement are expected; the flash kernel must also stay
                // within that of the exact result.
                let ok = outside_ok && d_fc < 2.5e-2 && (d_fe.is_nan() || d_fe <= 2.5e-2) && fl.iter().all(|x| x.is_finite());
                if !ok {
                    failures += 1;
                }
                let w = window.map_or("-".to_string(), |w| w.to_string());
                println!(
                    "{seq:>5} {w:>7} {sigma:>6.2}  {d_fc:>12.3e} {d_fe:>12.3e} {d_ce:>12.3e}{}{}",
                    if outside_ok { "" } else { "  OUTSIDE ROWS WRITTEN" },
                    if ok { "" } else { "  FAIL" }
                );
            }
        }
    }
    println!("worst relative error vs exact: flash {worst_flash:.3e}, cuBLAS path {worst_cublas:.3e}; {failures} failures");
    Ok(failures)
}
