//! End-to-end time of short packed text passes (replayed from CUDA graphs, bf16) by sequence count and length: the
//! measurement behind `FLASH_MIN_MEAN_LEN` in src/gpu/text.rs. Compare the kernels with
//! `EGEMMA2_PACKED_FLASH=0` (float32 packed kernel), `2` (packed flash) and unset (the length rule):
//! cargo run --release --example profile_packed
use embeddinggemma2::{Device, EmbeddingGemma2, Precision};
use std::path::Path;
use std::time::Instant;

fn median<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut v: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = EmbeddingGemma2::load_on(root.join("models/embeddinggemma-2"), Device::Cuda { ordinal: 0, precision: Precision::Bf16 })?;
    let words = "the quick brown fox jumps over the lazy dog while seven bright stars shine above a quiet harbour town ".repeat(20);
    let words: Vec<&str> = words.split_whitespace().collect();
    // A text of about n tokens (one word is usually one token here).
    let text = |n: usize| words[..n.saturating_sub(2).max(1)].join(" ");
    for (count, len) in [(1, 8), (1, 16), (1, 32), (1, 64), (1, 100), (1, 126), (2, 30), (2, 60), (4, 16), (4, 30), (8, 8), (8, 15), (16, 7)] {
        let texts: Vec<String> = (0..count).map(|_| text(len)).collect();
        let rows: usize = texts.iter().map(|t| model.tokenize(t).unwrap().len()).sum();
        let ms = median(|| drop(model.embed_texts(&texts).unwrap()), 60);
        println!("{count:>3} x {len:>3} tokens ({rows:>3} rows): {ms:6.3} ms");
    }
    Ok(())
}
