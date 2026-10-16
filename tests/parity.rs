//! Parity against reference data dumped from the original PyTorch implementation.
//!
//! Needs the model weights in `models/embeddinggemma-2` and the reference dumps in `testdata/`
//! (produced by `tools/ref_*.py`). Tests are skipped, loudly, when those are absent.
//!
//! Environment: `EGEMMA2_MODEL_DIR`, `EGEMMA2_TESTDATA` relocate the inputs. `EGEMMA2_DEVICE=cuda` runs on the
//! GPU in plain fp32 (same strict tolerances as the CPU); adding `EGEMMA2_TF32=1` (TF32 tensor cores) or
//! `EGEMMA2_BF16=1` (bf16 linear layers) switches `verify` to that precision's measured tolerance profile
//! (looser embeddings / soft tokens, plus a cosine floor).

use embeddinggemma2::{EmbeddingGemma2, verify};
use std::path::Path;
use std::sync::OnceLock;

fn model() -> Option<&'static EmbeddingGemma2> {
    static MODEL: OnceLock<Option<EmbeddingGemma2>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            // EGEMMA2_MODEL_DIR overrides the default (useful from a git worktree).
            let dir = std::env::var_os("EGEMMA2_MODEL_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("models/embeddinggemma-2"));
            // EGEMMA2_TF32 / EGEMMA2_BF16 only matter on CUDA; "0"/"false"/empty count as off.
            let on = |k: &str| std::env::var(k).is_ok_and(|v| !matches!(v.trim(), "" | "0" | "false"));
            let precision = if on("EGEMMA2_BF16") {
                embeddinggemma2::Precision::Bf16
            } else if on("EGEMMA2_TF32") {
                embeddinggemma2::Precision::Tf32
            } else {
                embeddinggemma2::Precision::F32
            };
            let device = match std::env::var("EGEMMA2_DEVICE").as_deref() {
                Ok("cuda") => embeddinggemma2::Device::Cuda { ordinal: 0, precision },
                _ => embeddinggemma2::Device::Cpu,
            };
            if !dir.join("model.safetensors").exists() {
                eprintln!("SKIPPED: {} has no model.safetensors", dir.display());
                return None;
            }
            Some(EmbeddingGemma2::load_on(&dir, device).expect("load model"))
        })
        .as_ref()
}

fn data(rel: &str) -> Option<std::path::PathBuf> {
    let root = std::env::var_os("EGEMMA2_TESTDATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata"));
    let p = root.join(rel);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIPPED: missing {}", p.display());
        None
    }
}

fn assert_all(m: &EmbeddingGemma2, checks: Vec<verify::Check>) {
    println!("device {}, tolerance profile {}", m.device, verify::Tolerances::for_model(m));
    for c in &checks {
        println!("{c}");
    }
    let failed: Vec<String> = checks.iter().filter(|c| !c.passed()).map(|c| c.to_string()).collect();
    assert!(failed.is_empty(), "{} parity checks failed:\n{}", failed.len(), failed.join("\n"));
}

#[test]
fn text_matches_pytorch() {
    let (Some(m), Some(d)) = (model(), data("ref_text.json")) else { return };
    assert_all(m, verify::check_text(m, &d).unwrap());
}

#[test]
fn image_matches_pytorch() {
    let (Some(m), Some(d)) = (model(), data("image")) else { return };
    assert_all(m, verify::check_image(m, &d).unwrap());
}

#[test]
fn audio_matches_pytorch() {
    let (Some(m), Some(d)) = (model(), data("audio")) else { return };
    assert_all(m, verify::check_audio(m, &d).unwrap());
}

#[test]
fn video_matches_pytorch() {
    let (Some(m), Some(d)) = (model(), data("video")) else { return };
    let mut checks = verify::check_video(m, &d).unwrap();
    checks.extend(verify::check_video_files(m, &d).unwrap());
    assert_all(m, checks);
}

#[test]
fn extra_cases_match_pytorch() {
    let (Some(m), Some(d)) = (model(), data("extra")) else { return };
    assert_all(m, verify::check_extra(m, &d).unwrap());
}
