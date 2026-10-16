//! Where the time of a packed text batch goes: tokenisation vs the whole embed_texts call.
//! Run with: cargo run --release --example profile_text -- [cpu|cuda|tf32|bf16]

use embeddinggemma2::{Device, EmbeddingGemma2, Precision};
use std::path::Path;
use std::time::Instant;

fn median<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    f();
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
    let mode = std::env::args().nth(1).unwrap_or_else(|| "bf16".into());
    let device = match mode.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda { ordinal: 0, precision: Precision::F32 },
        "tf32" => Device::Cuda { ordinal: 0, precision: Precision::Tf32 },
        _ => Device::Cuda { ordinal: 0, precision: Precision::Bf16 },
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = EmbeddingGemma2::load_on(root.join("models/embeddinggemma-2"), device)?;
    let inputs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("testdata/bench_inputs.json"))?)?;
    let batch: Vec<String> = inputs["batch64"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let tokens: usize = batch.iter().map(|t| model.tokenize(t).unwrap().len()).sum();
    println!("device {device}: {} docs, {tokens} tokens", batch.len());
    println!("tokenise 64 docs           {:7.2} ms", median(|| batch.iter().for_each(|t| drop(model.tokenize(t).unwrap())), 10));
    println!("embed_texts (packed)       {:7.2} ms", median(|| drop(model.embed_texts(&batch).unwrap()), 10));
    let half: Vec<String> = batch[..32].to_vec();
    println!("embed_texts, first 32 docs {:7.2} ms", median(|| drop(model.embed_texts(&half).unwrap()), 10));
    let one = vec![batch[0].clone()];
    println!("embed_texts, 1 doc         {:7.2} ms", median(|| drop(model.embed_texts(&one).unwrap()), 10));
    Ok(())
}
