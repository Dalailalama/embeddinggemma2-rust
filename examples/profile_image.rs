//! Stage timings of the image path: preprocessing, vision tower, full embedding.
//! Run with: cargo run --release --example profile_image -- [cpu|cuda|tf32|bf16]

use embeddinggemma2::{Device, EmbeddingGemma2, Input, Precision, imageproc, media};
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
    let img = media::load_image(&root.join("testdata/image/img_a_641x479.png"))?;
    let ic = &model.proc_cfg.image_processor;
    let prep = || {
        imageproc::preprocess_rgb(img.as_raw(), img.height() as usize, img.width() as usize, ic.patch_size, ic.pooling_kernel_size, ic.max_soft_tokens, model.resize_mode).unwrap()
    };
    let patched = prep();
    println!("device {device}, {} patches", patched.n_patches());
    println!("preprocess (resize + patchify)  {:7.1} ms", median(|| drop(prep()), 10));
    println!("vision tower + pooling          {:7.1} ms", median(|| drop(model.soft_tokens_from_patches(&patched).unwrap()), 10));
    println!("full embed (image only)         {:7.1} ms", median(|| drop(model.embed(&Input::image(img.clone())).unwrap()), 10));
    Ok(())
}
