//! GPU micro-benchmark of the vision tower's building blocks (2394 patches, the 641x479 test image):
//! its GEMM shapes in fp32 / TF32 / bf16 and its attention (12 heads x 64).
//! Run with: cargo run --release --example profile_gpu

use embeddinggemma2::Precision;
use embeddinggemma2::gpu::{Gpu, GpuLinear};
use embeddinggemma2::ops::Linear;
use std::time::Instant;

fn time_ms<F: FnMut() -> anyhow::Result<()>>(g: &Gpu, iters: usize, mut f: F) -> anyhow::Result<f64> {
    f()?;
    g.synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    g.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

fn main() -> anyhow::Result<()> {
    let (n, h, ff, heads, hd) = (2394usize, 768usize, 3072usize, 12usize, 64usize);
    for precision in [Precision::F32, Precision::Tf32, Precision::Bf16] {
        let g = Gpu::new(0, precision)?;
        let _lock = g.lock();
        let lin = |out: usize, inp: usize| -> anyhow::Result<GpuLinear> {
            let w: Vec<f32> = (0..out * inp).map(|i| ((i % 97) as f32 - 48.0) * 1e-3).collect();
            GpuLinear::from_cpu(&g, &Linear { w, out, inp, bias: None })
        };
        let x = g.upload(&vec![0.01f32; n * ff])?;
        let mut y = g.alloc(n * ff)?;
        let mut total = 0.0;
        println!("--- {precision} ({} patches)", n);
        for (name, out, inp, count) in [("qkv 768->2304", 3 * h, h, 1), ("o 768->768", h, h, 1), ("gate|up 768->6144", 2 * ff, h, 1), ("down 3072->768", h, ff, 1)] {
            let l = lin(out, inp)?;
            let ms = time_ms(&g, 20, || l.forward_raw(&g, &x, n, &mut y))?;
            let tflops = 2.0 * (n * out * inp) as f64 / (ms / 1e3) / 1e12;
            println!("  {name:<20} {ms:7.3} ms  {tflops:6.1} TFLOPS");
            total += ms * count as f64;
        }
        let q = g.upload(&vec![0.01f32; n * heads * hd])?;
        let mut a = g.alloc(n * heads * hd)?;
        let ms = time_ms(&g, 20, || g.attention(&q, &q, &q, &mut a, 0, n, heads, heads, hd, None))?;
        let tflops = 4.0 * (heads * n * n * hd) as f64 / (ms / 1e3) / 1e12;
        println!("  {:<20} {ms:7.3} ms  {tflops:6.1} TFLOPS", "attention 12x64");
        total += ms;
        println!("  per layer ~{total:.2} ms -> 16 layers ~{:.0} ms", 16.0 * total);

        // Text backbone shapes for a 64-document packed batch (~2240 rows), per layer x 24.
        let m = 2240usize;
        let xt = g.upload(&vec![0.01f32; m * 2048])?;
        let mut yt = g.alloc(m * 12288)?;
        let mut ttotal = 0.0;
        println!("  text ({m} rows):");
        for (name, out, inp) in [("qkv 512->2048", 2048, 512), ("o 1024->512", 512, 1024), ("gate|up 512->4096", 4096, 512), ("down 2048->512", 512, 2048), ("ple gate 512->512", 512, 512), ("ple proj 512->512", 512, 512)] {
            let l = lin(out, inp)?;
            let ms = time_ms(&g, 20, || l.forward_raw(&g, &xt, m, &mut yt))?;
            let tflops = 2.0 * (m * out * inp) as f64 / (ms / 1e3) / 1e12;
            println!("    {name:<20} {ms:7.3} ms  {tflops:6.1} TFLOPS");
            ttotal += ms;
        }
        let l = lin(12288, 512)?;
        let ms = time_ms(&g, 20, || l.forward_raw(&g, &xt, m, &mut yt))?;
        println!("    {:<20} {ms:7.3} ms  (once per pass)", "ple model 512->12288");
        println!("    GEMMs per layer ~{ttotal:.2} ms -> 24 layers ~{:.0} ms", 24.0 * ttotal);
    }
    Ok(())
}
