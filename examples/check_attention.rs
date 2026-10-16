//! Correctness check of the bf16 tensor-core attention kernels against the cuBLAS path and an exact float64
//! reference, on random inputs: tile-edge sequence lengths, with and without a window, logits from smooth to very
//! peaked, and the sequence placed at an offset inside a larger buffer (the rows around it must stay untouched).
//! The packed flash kernels (many short sequences in one launch, driven by a tile list) are checked per sequence
//! against the exact reference and the float32 packed kernel, with padding tiles and untouched gaps between sequences.
//! Run with: cargo run --release --example check_attention [-- 64 | 256 | 512 | packed]

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
    // Optional argument: run one part only (`256`, `64`, `512` or `packed`).
    let only = std::env::args().nth(1);
    anyhow::ensure!(matches!(only.as_deref(), None | Some("64" | "256" | "512" | "packed")), "unknown part {only:?}: expected 64, 256, 512 or packed");
    let run = |part: &str| only.as_deref().is_none_or(|o| o == part);
    let mut g = Gpu::new(0, Precision::Bf16)?;
    let (regs, local, smem) = g.flash_d512_info()?;
    println!("flash_attn_d512_bb: {regs} registers per thread, {local} bytes of local memory (spills), {smem} bytes of dynamic shared memory per block");
    let mut failures = 0;
    // The text backbone's sliding-window layers, and the vision tower.
    if run("256") {
        failures += check(&mut g, 4, 2, 256, &[1, 5, 31, 32, 33, 63, 64, 65, 127, 300, 700, 1100, 2208], &[None, Some(512), Some(7)])?;
    }
    if run("64") {
        failures += check(&mut g, 12, 12, 64, &[1, 17, 63, 64, 65, 129, 700, 2394], &[None, Some(7)])?;
    }
    // The text backbone's global layers: 4 query heads share one K/V head of 512 channels.
    if run("512") {
        failures += check(&mut g, 4, 1, 512, &[1, 5, 31, 32, 33, 63, 64, 65, 127, 300, 700, 1100, 2208], &[None, Some(7)])?;
    }
    // Packed sequences of mixed lengths (the text backbone's fused passes): tile-edge lengths, then a document-like
    // batch, through both packed flash kernels.
    if run("packed") {
        let tile_edges = [1, 5, 31, 32, 33, 63, 64, 65, 130, 300];
        let mut rng = Rng(0x2545f4914f6cdd1d);
        let docs: Vec<usize> = (0..64).map(|_| 10 + (rng.uniform() * 111.0) as usize).collect();
        for lens in [&tile_edges[..], &docs[..]] {
            failures += check_packed(&mut g, 4, 2, 256, lens, &[None, Some(512), Some(7)])?;
            failures += check_packed(&mut g, 4, 1, 512, lens, &[None, Some(7)])?;
        }
    }
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

/// How the sequences of a packed case sit in the buffer and how the tile list is arranged.
struct Layout {
    name: &'static str,
    /// Untouched rows after every sequence (not covered by any tile).
    gap: usize,
    /// Reversed tile order with empty tiles (len 0, nonsense row0 / q0) in front and in the middle, then padding.
    scrambled: bool,
}

/// A flat tile list as `{row0, len, q0}` triples.
fn tile_triples(flat: &[i32]) -> Vec<[i32; 3]> {
    flat.chunks(3).map(|t| [t[0], t[1], t[2]]).collect()
}

/// The packed flash kernel on sequences of `lens` rows packed (almost) back to back at an offset in a larger buffer,
/// per sequence against the exact float64 reference and the float32 packed kernel, and bit for bit against the
/// single-sequence flash kernel (the packed one is the same code with a different source of row0 / seq / q0).
fn check_packed(g: &mut Gpu, nh: usize, nkv: usize, hd: usize, lens: &[usize], windows: &[Option<usize>]) -> anyhow::Result<usize> {
    let (qs, kvs) = (nh * hd, nkv * hd);
    let (pad_before, pad_after) = (5usize, 3usize);
    g.set_flash(true); // `check` leaves it off
    let layouts = [Layout { name: "dense", gap: 0, scrambled: false }, Layout { name: "gaps", gap: 3, scrambled: true }];
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let (mut worst_flash, mut worst_packed) = (0f64, 0f64);
    let mut failures = 0;
    let base = (hd as f64).powf(-0.25);
    println!("\npacked flash: {} sequences ({} rows), heads {nh} / K/V heads {nkv} x {hd}", lens.len(), lens.iter().sum::<usize>());
    println!("{:>7} {:>6} {:>6} {:>7}  {:>12} {:>12} {:>12}", "window", "sigma", "layout", "tiles", "flash-packed", "flash-exact", "packed-exact");
    for &sigma in &[base, 2.0 * base, 4.0 * base] {
        for &window in windows {
            for lay in &layouts {
                // Sequence starts; every sequence is followed by `gap` rows that no tile covers.
                let mut starts = Vec::new();
                let mut r = pad_before;
                for &l in lens {
                    starts.push(r);
                    r += l + lay.gap;
                }
                let rows = r + pad_after;
                let mut covered = vec![false; rows];
                for (&s, &l) in starts.iter().zip(lens) {
                    covered[s..s + l].fill(true);
                }

                // Tile list: real tiles (all at once, or built per sequence), optionally reversed around empty ones,
                // then padding tiles.
                let mut tiles: Vec<[i32; 3]> = if lay.gap == 0 {
                    tile_triples(&Gpu::flash_packed_tiles(pad_before, lens, 0))
                } else {
                    starts.iter().zip(lens).flat_map(|(&s, &l)| tile_triples(&Gpu::flash_packed_tiles(s, &[l], 0))).collect()
                };
                if lay.scrambled {
                    tiles.reverse();
                    tiles.insert(0, [rows as i32 + 1000, 0, 64]);
                    tiles.insert(tiles.len() / 2, [pad_before as i32, 0, 0]);
                }
                let mut flat: Vec<i32> = tiles.iter().flatten().copied().collect();
                flat.resize(flat.len() + 3 * 7, 0); // trailing padding tiles
                let n_tiles = flat.len() / 3;

                let mut sample = |n: usize, s: f64| -> Vec<f32> { (0..n).map(|_| bf16_round((rng.normal() * s) as f32)).collect() };
                let (q, k, v) = (sample(rows * qs, sigma), sample(rows * kvs, sigma), sample(rows * kvs, 1.0));
                let (qa, ka, va) = (Act::Bf16(g.upload_bf16(&q)?), Act::Bf16(g.upload_bf16(&k)?), Act::Bf16(g.upload_bf16(&v)?));
                let sentinel = vec![7.0f32; rows * qs];
                let dev_tiles = g.upload_i32(&flat)?;
                let mut o_flash = Act::Bf16(g.upload_bf16(&sentinel)?);
                g.attention_flash_packed(&qa, &ka, &va, &mut o_flash, &dev_tiles.slice(..), n_tiles, nh, nkv, hd, window)?;

                // The float32 packed kernel on the same (bf16-representable) values; rows outside the sequences are
                // sequences of their own.
                let (mut seg_start, mut seg_len): (Vec<i32>, Vec<i32>) = (0..rows as i32).map(|t| (t, 1)).unzip();
                for (&s, &l) in starts.iter().zip(lens) {
                    seg_start[s..s + l].fill(s as i32);
                    seg_len[s..s + l].fill(l as i32);
                }
                let (qf, kf, vf) = (Act::F32(g.upload(&q)?), Act::F32(g.upload(&k)?), Act::F32(g.upload(&v)?));
                let (ds, dl) = (g.upload_i32(&seg_start)?, g.upload_i32(&seg_len)?);
                let mut o_packed = Act::Bf16(g.upload_bf16(&sentinel)?);
                g.attention_packed_act(&qf, &kf, &vf, &mut o_packed, &ds.slice(..), &dl.slice(..), rows, nh, nkv, hd, window)?;

                // The single-sequence flash kernel, one launch per sequence: the same code, so the same bits.
                let mut o_single = Act::Bf16(g.upload_bf16(&sentinel)?);
                for (&s, &l) in starts.iter().zip(lens) {
                    g.attention_act(&qa, &ka, &va, &mut o_single, s, l, nh, nkv, hd, window)?;
                }

                let (fl, pk, sg) = (g.download_act(&o_flash)?, g.download_act(&o_packed)?, g.download_act(&o_single)?);
                let same_as_single = fl == sg;
                // Rows no tile covers (before, between and after the sequences) must be untouched.
                let outside_ok = (0..rows).filter(|&t| !covered[t]).all(|t| fl[t * qs..(t + 1) * qs].iter().all(|&x| x == 7.0));
                let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
                let (mut d_fp, mut d_fe, mut d_pe) = (0f64, 0f64, 0f64);
                let mut worst_len = 0;
                for (&s, &l) in starts.iter().zip(lens) {
                    let seq_rows = s * qs..(s + l) * qs;
                    let exact = reference(&q, &k, &v, s, l, nh, nkv, hd, window);
                    let err = |o: &[f32]| o[seq_rows.clone()].iter().zip(&exact).map(|(a, b)| rel(*a as f64, *b)).fold(0.0, f64::max);
                    let fe = err(&fl);
                    if fe > d_fe {
                        worst_len = l;
                    }
                    d_fe = d_fe.max(fe);
                    d_pe = d_pe.max(err(&pk));
                    d_fp = d_fp.max(fl[seq_rows.clone()].iter().zip(&pk[seq_rows.clone()]).map(|(a, b)| rel(*a as f64, *b as f64)).fold(0.0, f64::max));
                }
                worst_flash = worst_flash.max(d_fe);
                worst_packed = worst_packed.max(d_pe);
                let ok = outside_ok && same_as_single && d_fp < 2.5e-2 && d_fe <= 2.5e-2 && fl.iter().all(|x| x.is_finite());
                if !ok {
                    failures += 1;
                }
                let w = window.map_or("-".to_string(), |w| w.to_string());
                println!(
                    "{w:>7} {sigma:>6.2} {:>6} {n_tiles:>7}  {d_fp:>12.3e} {d_fe:>12.3e} {d_pe:>12.3e}{}{}{}",
                    lay.name,
                    if outside_ok { "" } else { "  OUTSIDE ROWS WRITTEN" },
                    if same_as_single { "" } else { "  DIFFERS FROM THE SINGLE-SEQUENCE KERNEL" },
                    if ok { String::new() } else { format!("  FAIL (worst flash error at length {worst_len})") }
                );
            }
        }
    }
    println!("worst relative error vs exact: packed flash {worst_flash:.3e}, float32 packed kernel {worst_packed:.3e}; {failures} failures");
    Ok(failures)
}
