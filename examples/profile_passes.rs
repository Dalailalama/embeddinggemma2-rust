//! Device time per operation of whole text / image passes on the GPU (the `bench` inputs).
//! Run with: EGEMMA2_PROFILE=1 cargo run --release --example profile_passes -- [fp32|tf32|bf16]

use embeddinggemma2::{Device, EmbeddingGemma2, Input, Precision, media};
use std::path::Path;

type Run<'a> = Box<dyn Fn() -> anyhow::Result<()> + 'a>;

fn main() -> anyhow::Result<()> {
    anyhow::ensure!(std::env::var_os("EGEMMA2_PROFILE").is_some(), "set EGEMMA2_PROFILE=1 to enable the per-op timers");
    let precision = match std::env::args().nth(1).as_deref().unwrap_or("bf16") {
        "fp32" => Precision::F32,
        "tf32" => Precision::Tf32,
        _ => Precision::Bf16,
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = EmbeddingGemma2::load_on(root.join("models/embeddinggemma-2"), Device::Cuda { ordinal: 0, precision })?;
    let g = model.gpu().expect("CUDA model");
    let data = root.join("testdata");
    let inputs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(data.join("bench_inputs.json"))?)?;
    let long = inputs["long"].as_str().unwrap().to_string();
    let batch: Vec<String> = inputs["batch64"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let img = media::load_image(&data.join("image/img_a_641x479.png"))?;

    let runs: Vec<(&str, Run)> = vec![
        ("long document (2208 tokens)", Box::new(|| model.embed_texts(std::slice::from_ref(&long)).map(|_| ()))),
        ("batch of 64 short documents", Box::new(|| model.embed_texts(&batch).map(|_| ()))),
        ("image 641x479 (vision tower + text)", Box::new(|| model.embed(&Input::image(img.clone())).map(|_| ()))),
    ];
    println!("device {}", model.device);
    for (name, run) in &runs {
        // Two warm-up passes, then five profiled ones (averaged).
        for _ in 0..2 {
            run()?;
        }
        g.take_profile();
        const N: usize = 5;
        for _ in 0..N {
            run()?;
        }
        let prof = g.take_profile();
        let total: f64 = prof.iter().map(|(_, ms)| ms).sum::<f64>() / N as f64;
        println!("\n{name}: {total:.1} ms of device time per pass");
        for (label, ms) in &prof {
            let ms = ms / N as f64;
            println!("  {label:<28} {ms:7.2} ms  {:5.1}%", 100.0 * ms / total);
        }
    }
    Ok(())
}
