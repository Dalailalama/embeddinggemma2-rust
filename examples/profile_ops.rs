//! Micro-benchmark of the matmul shapes used by the text backbone at small sequence lengths.
//! Run with: cargo run --release --example profile_ops

use embeddinggemma2::ops::matmul_nt;
use std::time::Instant;

fn time<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    // (name, n_out, k_in, count per forward)
    let shapes = [
        ("q_proj 1024x512", 1024, 512, 20),
        ("q_proj(global) 2048x512", 2048, 512, 4),
        ("k/v 512x512", 512, 512, 48),
        ("o_proj 512x1024", 512, 1024, 20),
        ("o_proj(global) 512x2048", 512, 2048, 4),
        ("gate/up 2048x512", 2048, 512, 48),
        ("down 512x2048", 512, 2048, 24),
        ("ple gate/proj 512x512", 512, 512, 48),
        ("ple_model 12288x512", 12288, 512, 1),
    ];
    for m in [1usize, 17, 64, 256, 1024] {
        let mut total = 0.0;
        println!("--- m = {m} tokens");
        for (name, n, k, count) in shapes {
            let x = vec![0.01f32; m * k];
            let w = vec![0.02f32; n * k];
            let ms = time(|| { std::hint::black_box(matmul_nt(&x, m, k, &w, n)); }, 20);
            let gbps = (n * k * 4) as f64 / (ms / 1e3) / 1e9;
            println!("{name:<28} {ms:8.3} ms  ({gbps:6.1} GB/s weight stream)  x{count}");
            total += ms * count as f64;
        }
        println!("{:<28} {total:8.1} ms  (sum of all matmuls in one forward)", "TOTAL");
    }
}
