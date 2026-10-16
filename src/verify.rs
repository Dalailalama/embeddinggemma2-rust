//! Parity checks against reference data dumped from the original PyTorch implementation by the
//! scripts in `tools/` (`ref_text.py`, `ref_image.py`, `ref_audio.py`, `ref_video.py`).

use anyhow::Result;
use std::path::Path;

use crate::backend::Device;
use crate::backend::Precision;
use crate::imageproc::PatchedImage;
use crate::media;
use crate::model::{EmbeddingGemma2, Input, Part, VideoClip, truncate_and_normalize};
use crate::videoproc::{self, SampleParams};

/// Largest acceptable absolute difference per kind of quantity (f32 vs f32 reference).
/// These are the strict float32 limits, used on the CPU and on CUDA without TF32 (`Tolerances::FLOAT32`).
pub const TOL_EMBEDDING: f64 = 5e-6;
pub const TOL_SOFT_TOKENS: f64 = 1e-3; // soft tokens have magnitude ~6, accumulated over 12-16 layers
pub const TOL_PIXELS: f64 = 1e-6;
pub const TOL_MEL: f64 = 1e-5;

/// What a `Check` measures; used to group the end-of-run summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Embedding,
    SoftTokens,
    Pixels,
    Mel,
    Other,
}

impl Kind {
    const ALL: [Kind; 5] = [Kind::Embedding, Kind::SoftTokens, Kind::Pixels, Kind::Mel, Kind::Other];

    fn name(self) -> &'static str {
        match self {
            Kind::Embedding => "embeddings",
            Kind::SoftTokens => "soft tokens",
            Kind::Pixels => "patches",
            Kind::Mel => "log-mel",
            Kind::Other => "other",
        }
    }
}

/// Acceptance limit for one kind of quantity: a max-abs bound and, optionally, a cosine floor.
#[derive(Clone, Copy, Debug)]
pub struct Limit {
    pub kind: Kind,
    pub max_abs: f64,
    /// Minimum cosine similarity to the reference. `None` = only `max_abs` is enforced.
    pub min_cos: Option<f64>,
}

/// The set of limits in force for a run; picked from the device the model actually runs on.
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    pub name: &'static str,
    pub embedding: Limit,
    pub soft_tokens: Limit,
    pub pixels: Limit,
    pub mel: Limit,
}

impl Tolerances {
    /// Strict float32 limits (CPU, and CUDA with plain fp32 cuBLAS). Never loosened for other profiles.
    pub const FLOAT32: Tolerances = Tolerances {
        name: "float32",
        embedding: Limit { kind: Kind::Embedding, max_abs: TOL_EMBEDDING, min_cos: None },
        soft_tokens: Limit { kind: Kind::SoftTokens, max_abs: TOL_SOFT_TOKENS, min_cos: None },
        pixels: Limit { kind: Kind::Pixels, max_abs: TOL_PIXELS, min_cos: None },
        mel: Limit { kind: Kind::Mel, max_abs: TOL_MEL, min_cos: None },
    };

    /// CUDA with `--tf32` (cuBLAS GEMMs on TF32 tensor cores, 10-bit mantissa inputs).
    ///
    /// Bounds come from measurement on an RTX A3000 (CUDA 13.2, 84 checks of the reference set):
    ///   embeddings (incl. Matryoshka cuts)  worst max|d| 3.6e-4, worst cos 0.99999553
    ///   soft tokens (incl. tower-only)      worst max|d| 1.6e-1, worst cos 0.99999684
    /// Each max-abs limit sits ~3x above the worst value seen (embeddings 2.8x, soft tokens 3.2x), so another
    /// GPU generation or cuBLAS algorithm choice still passes. The cosine floor of 0.99999 leaves ~2.2x
    /// headroom on 1-cos (4.5e-6 observed, 1e-5 allowed) and is what keeps a real bug from hiding inside
    /// the loose max-abs bound: a wrong layer, row or scale moves 1-cos by orders of magnitude more, and
    /// it caps the relative L2 error at ~0.45%. Patches and log-mel are computed on the CPU in every
    /// configuration, so they keep the strict float32 limits.
    pub const TF32: Tolerances = Tolerances {
        name: "tf32",
        embedding: Limit { kind: Kind::Embedding, max_abs: 1e-3, min_cos: Some(0.99999) },
        soft_tokens: Limit { kind: Kind::SoftTokens, max_abs: 5e-1, min_cos: Some(0.99999) },
        pixels: Self::FLOAT32.pixels,
        mel: Self::FLOAT32.mel,
    };

    /// CUDA with `--bf16`: every linear layer and the attention products on bf16 tensor cores (bf16 weights,
    /// the products' inputs and outputs stored as bf16, float32 accumulation; norms, softmax and the residual
    /// stream stay float32).
    ///
    /// Bounds come from measurement on an RTX A3000 (CUDA 13.2, the 90 checks of the reference set), first with
    /// cuBLAS text attention and then with the tensor-core flash kernels (second figures):
    ///   embeddings (incl. Matryoshka cuts)  worst max|d| 1.50e-3 / 1.29e-3, worst cos 0.99996550 / 0.99996712
    ///   soft tokens (incl. tower-only)      worst max|d| 1.33 / 1.33,       worst cos 0.99980725 / 0.99980725
    /// Same headroom rules as TF32: max-abs ~2.7x above the worst value seen, cosine floors with >= 2x
    /// headroom on 1-cos (3.5e-5 -> 9e-5 for embeddings, 1.9e-4 -> 4e-4 for soft tokens). For scale,
    /// PyTorch's own bf16 mode reaches cos 0.99975 on the 9.3 s audio clip.
    pub const BF16: Tolerances = Tolerances {
        name: "bf16",
        embedding: Limit { kind: Kind::Embedding, max_abs: 4e-3, min_cos: Some(0.99991) },
        soft_tokens: Limit { kind: Kind::SoftTokens, max_abs: 3.5, min_cos: Some(0.9996) },
        pixels: Self::FLOAT32.pixels,
        mel: Self::FLOAT32.mel,
    };

    pub fn for_device(device: Device) -> Tolerances {
        match device {
            Device::Cuda { precision: Precision::Tf32, .. } => Self::TF32,
            Device::Cuda { precision: Precision::Bf16, .. } => Self::BF16,
            _ => Self::FLOAT32,
        }
    }

    pub fn for_model(model: &EmbeddingGemma2) -> Tolerances {
        Self::for_device(model.device)
    }
}

impl std::fmt::Display for Tolerances {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:", self.name)?;
        for (i, l) in [self.embedding, self.soft_tokens, self.pixels, self.mel].iter().enumerate() {
            // Two significant digits where needed: 2.5e-3 must not print as 3e-3.
            let bound = format!("{:.1e}", l.max_abs).replace(".0e", "e");
            write!(f, "{} {} <= {bound}", if i == 0 { "" } else { "," }, l.kind.name())?;
            if let Some(c) = l.min_cos {
                write!(f, " & cos >= {c}")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Check {
    pub label: String,
    pub max_abs: f64,
    pub cos: Option<f64>,
    pub ids_ok: Option<bool>,
    pub limit: Limit,
    pub note: String,
}

impl Check {
    pub fn passed(&self) -> bool {
        // `>=` on the cosine (not `!(< ..)`) so that a NaN cosine fails.
        self.max_abs.is_finite()
            && self.max_abs <= self.limit.max_abs
            && self.limit.min_cos.is_none_or(|m| self.cos.is_some_and(|c| c >= m))
            && self.ids_ok.unwrap_or(true)
    }
}

impl std::fmt::Display for Check {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ok = self.passed();
        write!(f, "{} {:<34} max|d|={:.2e}", if ok { "PASS" } else { "FAIL" }, self.label, self.max_abs)?;
        if let Some(c) = self.cos {
            write!(f, "  cos={c:.8}")?;
        }
        if let Some(ok) = self.ids_ok {
            write!(f, "  ids {}", if ok { "OK" } else { "MISMATCH" })?;
        }
        if !self.note.is_empty() {
            write!(f, "  {}", self.note)?;
        }
        if !ok {
            // Say what was violated, so a FAIL line is self-explanatory.
            write!(f, "  [limit max|d|<={:.1e}", self.limit.max_abs)?;
            if let Some(m) = self.limit.min_cos {
                write!(f, ", cos>={m}")?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

/// Worst observed value per kind of quantity and how much of its limit that uses, one line each.
/// Gives the margin at a glance (and the data to re-derive the limits if the kernels change).
pub fn summarize(checks: &[Check]) -> String {
    let mut out = String::new();
    for kind in Kind::ALL {
        let of_kind: Vec<&Check> = checks.iter().filter(|c| c.limit.kind == kind).collect();
        if of_kind.is_empty() {
            continue;
        }
        let worst = of_kind.iter().map(|c| c.max_abs).fold(0.0, f64::max);
        // A few checks (YUV video decode) carry a looser limit than their kind's default, so compare per check.
        let used = of_kind.iter().map(|c| c.max_abs / c.limit.max_abs).fold(0.0, f64::max);
        out += &format!("  {:<12} n={:<3} worst max|d|={:.2e} ({:.0}% of its limit)", kind.name(), of_kind.len(), worst, used * 100.0);
        if let Some(m) = of_kind.iter().find_map(|c| c.limit.min_cos) {
            let min_cos = of_kind.iter().filter_map(|c| c.cos).fold(f64::INFINITY, f64::min);
            out += &format!("  min cos={min_cos:.8} (floor {m})");
        }
        out.push('\n');
    }
    out
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Like `read_f32`, but `None` when the dump is absent (very large dumps are not committed to git).
fn read_f32_opt(path: &Path) -> Result<Option<Vec<f32>>> {
    if path.exists() { read_f32(path).map(Some) } else { Ok(None) }
}

fn json_f32(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn json_ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn vec_check(label: impl Into<String>, got: &[f32], want: &[f32], limit: Limit, ids_ok: Option<bool>, note: String) -> Check {
    Check {
        label: label.into(),
        max_abs: max_abs_diff(got, want),
        cos: (got.len() == want.len()).then(|| cosine(got, want)),
        ids_ok,
        limit,
        note,
    }
}

/// Text: token ids, 768-d embeddings and Matryoshka-truncated variants, one by one and packed.
pub fn check_text(model: &EmbeddingGemma2, reference: &Path) -> Result<Vec<Check>> {
    let tol = Tolerances::for_model(model);
    let mut out = Vec::new();
    let refs = read_json(reference)?;
    let refs = refs.as_array().unwrap();

    // Packed batch path: all reference texts in shared forward passes.
    let mut texts = Vec::new();
    for r in refs {
        let prefix = match r["prompt"].as_str() {
            Some(p) => model.prompt(p)?,
            None => "",
        };
        texts.push(format!("{prefix}{}", r["text"].as_str().unwrap()));
    }
    let batch = model.embed_texts(&texts)?;
    let got: Vec<f32> = batch.concat();
    let want: Vec<f32> = refs.iter().flat_map(|r| json_f32(&r["emb"])).collect();
    out.push(vec_check(format!("text packed batch ({} texts)", refs.len()), &got, &want, tol.embedding, None, String::new()));

    // Only short texts: on the GPU this takes the fused packed-attention kernel.
    let short: Vec<usize> = (0..refs.len()).filter(|&i| model.tokenize(&texts[i]).map(|t| t.len() <= 256).unwrap_or(false)).collect();
    let batch = model.embed_texts(&short.iter().map(|&i| texts[i].clone()).collect::<Vec<_>>())?;
    let want: Vec<f32> = short.iter().flat_map(|&i| json_f32(&refs[i]["emb"])).collect();
    out.push(vec_check(format!("text packed batch, short only ({})", short.len()), &batch.concat(), &want, tol.embedding, None, String::new()));

    for r in refs {
        let prompt = r["prompt"].as_str();
        let text = r["text"].as_str().unwrap();
        let prefix = match prompt {
            Some(p) => model.prompt(p)?,
            None => "",
        };
        let ids = model.tokenize(&format!("{prefix}{text}"))?;
        let ids_ok = ids == json_ids(&r["ids"]);
        let emb = model.embed_ids(&ids)?;
        let label = format!("text[{}] {} tok", prompt.unwrap_or("-"), ids.len());
        out.push(vec_check(label.clone(), &emb, &json_f32(&r["emb"]), tol.embedding, Some(ids_ok), String::new()));
        for dim in [512usize, 256, 128] {
            let t = truncate_and_normalize(emb.clone(), Some(dim))?;
            out.push(vec_check(format!("{label} @{dim}d"), &t, &json_f32(&r[format!("emb{dim}")]), tol.embedding, None, String::new()));
        }
    }
    Ok(out)
}

/// Images: resize + patchify, vision tower (also isolated on the reference's own patches), embeddings.
pub fn check_image(model: &EmbeddingGemma2, dir: &Path) -> Result<Vec<Check>> {
    let tol = Tolerances::for_model(model);
    let meta = read_json(&dir.join("ref_image.json"))?;
    let mut out = Vec::new();
    for c in meta["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let img = media::load_image(Path::new(c["png"].as_str().unwrap()))?;
        let (patched, soft) = model.image_soft_tokens(&img)?;
        let ref_patches = read_f32(&dir.join(format!("{name}.patches.bin")))?;
        let ref_soft = read_f32(&dir.join(format!("{name}.soft.bin")))?;
        let grid_ok = patched.grid_w == c["grid_w"].as_u64().unwrap() as usize
            && patched.grid_h == c["grid_h"].as_u64().unwrap() as usize;
        out.push(vec_check(format!("image[{name}] patches"), &patched.patches, &ref_patches, tol.pixels, Some(grid_ok), String::new()));
        out.push(vec_check(format!("image[{name}] soft tokens"), &soft, &ref_soft, tol.soft_tokens, None, format!("{} tok", soft.len() / 512)));
        let isolated = PatchedImage {
            patches: ref_patches.clone(),
            positions: patched.positions.clone(),
            grid_h: patched.grid_h,
            grid_w: patched.grid_w,
            patch_dim: patched.patch_dim,
        };
        let iso = model.soft_tokens_from_patches(&isolated)?;
        out.push(vec_check(format!("image[{name}] tower only"), &iso, &ref_soft, tol.soft_tokens, None, String::new()));

        let layouts: [(&str, Option<&str>, Vec<Part>); 3] = [
            ("emb_image", Some("ids_image"), vec![Part::Image(img.clone())]),
            ("emb_text_image", Some("ids_text_image"), vec![Part::Text("a photo of".into()), Part::Image(img.clone())]),
            ("emb_image_text", None, vec![Part::Image(img.clone()), Part::Text("a photo of".into())]),
        ];
        for (ek, idk, parts) in layouts {
            let (ids, emb) = model.embed_with_ids(&Input { parts })?;
            let ids_ok = idk.map(|k| json_ids(&c[k]) == ids);
            out.push(vec_check(format!("image[{name}] {ek}"), &emb, &json_f32(&c[ek]), tol.embedding, ids_ok, String::new()));
        }
    }
    if meta["multi"].get("ids_text_two_images").is_some() {
        let a = media::load_image(&dir.join("img_a_641x479.png"))?;
        let b = media::load_image(&dir.join("img_b_96x64.png"))?;
        let (ids, emb) = model.embed_with_ids(&Input { parts: vec![Part::Text("A:".into()), Part::Image(a), Part::Image(b)] })?;
        out.push(vec_check(
            "image[two images + text]",
            &emb,
            &json_f32(&meta["multi"]["emb_text_two_images"]),
            tol.embedding,
            Some(json_ids(&meta["multi"]["ids_text_two_images"]) == ids),
            String::new(),
        ));
    }
    Ok(out)
}

/// Audio: log-mel features, audio tower (also isolated on reference mel), embeddings.
pub fn check_audio(model: &EmbeddingGemma2, dir: &Path) -> Result<Vec<Check>> {
    let tol = Tolerances::for_model(model);
    let meta = read_json(&dir.join("ref_audio.json"))?;
    let mut out = Vec::new();
    for c in meta["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let clip = media::load_audio(Path::new(c["wav"].as_str().unwrap()))?;
        let (mel, frames, soft) = model.audio_soft_tokens(&clip)?;
        let ref_mel = read_f32(&dir.join(format!("{name}.mel.bin")))?;
        let ref_soft = read_f32(&dir.join(format!("{name}.soft.bin")))?;
        out.push(vec_check(
            format!("audio[{name}] log-mel"),
            &mel,
            &ref_mel,
            tol.mel,
            Some(frames == c["frames"].as_u64().unwrap() as usize),
            format!("{frames} frames"),
        ));
        out.push(vec_check(format!("audio[{name}] soft tokens"), &soft, &ref_soft, tol.soft_tokens, None, format!("{} tok", soft.len() / 512)));
        let iso = model.soft_tokens_from_mel(&ref_mel, frames)?;
        out.push(vec_check(format!("audio[{name}] tower only"), &iso, &ref_soft, tol.soft_tokens, None, String::new()));
        let layouts: [(&str, &str, Vec<Part>); 2] = [
            ("emb_audio", "ids_audio", vec![Part::Audio(clip.clone())]),
            ("emb_text_audio", "ids_text_audio", vec![Part::Text("a recording of".into()), Part::Audio(clip.clone())]),
        ];
        for (ek, idk, parts) in layouts {
            let (ids, emb) = model.embed_with_ids(&Input { parts })?;
            out.push(vec_check(format!("audio[{name}] {ek}"), &emb, &json_f32(&c[ek]), tol.embedding, Some(json_ids(&c[idk]) == ids), String::new()));
        }
    }
    Ok(out)
}

/// Video files decoded through ffmpeg (reference decodes with PyAV). Skipped without ffmpeg.
pub fn check_video_files(model: &EmbeddingGemma2, dir: &Path) -> Result<Vec<Check>> {
    let tol = Tolerances::for_model(model);
    let path = dir.join("ref_video_file.json");
    if !path.exists() {
        return Ok(vec![]);
    }
    let meta = read_json(&path)?;
    let mut out = Vec::new();
    let sp = model.video_sample_params();
    for c in meta["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let clip = match media::load_video(&dir.join(name), None, &sp) {
            Ok(clip) => clip,
            Err(e) if format!("{e:#}").contains("needs ffmpeg") => {
                eprintln!("skipping {name}: {e:#}");
                continue;
            }
            Err(e) => return Err(e),
        };
        let (frames, _soft) = model.video_soft_tokens(&clip)?;
        let patches: Vec<f32> = frames.iter().flat_map(|f| f.patches.iter().copied()).collect();
        let ref_patches = read_f32(&dir.join(format!("{name}.patches.bin")))?;
        let n_ok = frames.len() == c["n_frames"].as_u64().unwrap() as usize;
        // Lossless RGB video must decode exactly; YUV 4:2:0 may differ by colour-conversion rounding.
        let lim = if name.contains("yuv") { Limit { max_abs: 2.0 / 255.0 + 1e-6, ..tol.pixels } } else { tol.pixels };
        out.push(vec_check(format!("video file[{name}] decoded patches"), &patches, &ref_patches, lim, Some(n_ok), format!("{} frames", frames.len())));
        let (ids, emb) = model.embed_with_ids(&Input { parts: vec![Part::Video(clip)] })?;
        // The decode slack is on top of the profile's own noise floor, so take the looser of the two.
        let lim = if name.contains("yuv") { Limit { max_abs: tol.embedding.max_abs.max(1e-3), ..tol.embedding } } else { tol.embedding };
        out.push(vec_check(format!("video file[{name}] emb_video"), &emb, &json_f32(&c["emb_video"]), lim, Some(json_ids(&c["ids_video"]) == ids), String::new()));
    }
    Ok(out)
}

/// Extra cases from `tools/ref_extra.py`: a prompt in front of media (sentence-transformers puts it
/// before everything), audio longer than the 11.2 s `audio_seq_length` and the 30 s truncation,
/// and a sub-1 fps video whose 1 fps sampling repeats frames (needs ffmpeg; skipped without it).
pub fn check_extra(model: &EmbeddingGemma2, dir: &Path) -> Result<Vec<Check>> {
    let tol = Tolerances::for_model(model);
    let path = dir.join("ref_extra.json");
    if !path.exists() {
        return Ok(vec![]);
    }
    let meta = read_json(&path)?;
    let mut out = Vec::new();
    for c in meta["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let file = |k: &str| Path::new(c[k].as_str().unwrap()).to_path_buf();
        let mut parts = Vec::new();
        if let Some(img) = c.get("image").and_then(|v| v.as_str()) {
            let img = media::load_image(Path::new(img))?;
            match c.get("order").and_then(|v| v.as_str()) {
                Some("text,image") => parts.extend([Part::Text(c["text"].as_str().unwrap().into()), Part::Image(img)]),
                Some("image,text") => parts.extend([Part::Image(img), Part::Text(c["text"].as_str().unwrap().into())]),
                _ => parts.push(Part::Image(img)),
            }
        } else if c.get("audio").is_some() {
            parts.push(Part::Audio(media::load_audio(&file("audio"))?));
        } else if c.get("video").is_some() {
            match media::load_video(&file("video"), None, &model.video_sample_params()) {
                Ok(clip) => parts.push(Part::Video(clip)),
                Err(e) if format!("{e:#}").contains("needs ffmpeg") => {
                    eprintln!("skipping {name}: {e:#}");
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
        let prompt = match c["prompt_name"].as_str() {
            Some(p) => model.prompt(p)?.to_string(),
            None => String::new(),
        };
        let (ids, emb) = model.embed_with_ids(&Input { parts }.with_prompt(&prompt))?;
        let ref_ids = json_ids(&c["ids"]);
        let note = format!("{} tokens", ids.len());
        out.push(vec_check(format!("extra[{name}]"), &emb, &json_f32(&c["emb"]), tol.embedding, Some(ref_ids == ids), note));
    }
    Ok(out)
}

/// Video: frame sampling decisions, patches, vision tower, embeddings.
pub fn check_video(model: &EmbeddingGemma2, dir: &Path) -> Result<Vec<Check>> {
    let tol = Tolerances::for_model(model);
    let meta = read_json(&dir.join("ref_video.json"))?;
    let mut out = Vec::new();
    let sp = model.video_sample_params();
    let rows = meta["sampling"].as_array().unwrap();
    let mut bad = Vec::new();
    for row in rows {
        let total = row["total"].as_u64().unwrap() as usize;
        let fps = row["fps"].as_f64();
        let want: Vec<usize> = row["indices"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let params = if fps.is_some() { sp.clone() } else { SampleParams { fps: None, ..sp.clone() } };
        if videoproc::sample_frames(total, fps, &params)? != want {
            bad.push(format!("total={total} fps={fps:?}"));
        }
    }
    out.push(Check {
        label: format!("video frame sampling ({} cases)", rows.len()),
        max_abs: bad.len() as f64,
        cos: None,
        ids_ok: None,
        limit: Limit { kind: Kind::Other, max_abs: 0.0, min_cos: None },
        note: if bad.is_empty() { String::new() } else { format!("mismatches: {bad:?}") },
    });
    for c in meta["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let clip = media::load_video(Path::new(c["dir"].as_str().unwrap()), None, &sp)?;
        let (frames, soft) = model.video_soft_tokens(&clip)?;
        let patches: Vec<f32> = frames.iter().flat_map(|f| f.patches.iter().copied()).collect();
        let ref_soft = read_f32(&dir.join(format!("{name}.soft.bin")))?;
        if let Some(ref_patches) = read_f32_opt(&dir.join(format!("{name}.patches.bin")))? {
            out.push(vec_check(
                format!("video[{name}] patches"),
                &patches,
                &ref_patches,
                tol.pixels,
                Some(frames.len() == c["n_frames"].as_u64().unwrap() as usize),
                format!("{} frames", frames.len()),
            ));
        } else {
            eprintln!("note: {name}.patches.bin not present (regenerate with tools/ref_video.py); skipping that check");
        }
        out.push(vec_check(format!("video[{name}] soft tokens"), &soft, &ref_soft, tol.soft_tokens, None, format!("{} tok", soft.len() / 512)));
        let layouts: [(&str, &str, Vec<Part>); 2] = [
            ("emb_video", "ids_video", vec![Part::Video(VideoClip { frames: clip.frames.clone(), fps: None })]),
            (
                "emb_text_video",
                "ids_text_video",
                vec![Part::Text("a clip of".into()), Part::Video(VideoClip { frames: clip.frames.clone(), fps: None })],
            ),
        ];
        for (ek, idk, parts) in layouts {
            let (ids, emb) = model.embed_with_ids(&Input { parts })?;
            out.push(vec_check(format!("video[{name}] {ek}"), &emb, &json_f32(&c[ek]), tol.embedding, Some(json_ids(&c[idk]) == ids), String::new()));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit-norm 768-d vector with a deterministic, roughly zero-mean pattern.
    fn unit_vec() -> Vec<f32> {
        let v: Vec<f32> = (0..768).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn check(got: &[f32], want: &[f32], limit: Limit) -> Check {
        vec_check("t", got, want, limit, None, String::new())
    }

    #[test]
    fn profile_follows_device() {
        let cuda = |precision| Device::Cuda { ordinal: 0, precision };
        assert_eq!(Tolerances::for_device(Device::Cpu).name, "float32");
        assert_eq!(Tolerances::for_device(cuda(Precision::F32)).name, "float32");
        assert_eq!(Tolerances::for_device(cuda(Precision::Tf32)).name, "tf32");
        assert_eq!(Tolerances::for_device(cuda(Precision::Bf16)).name, "bf16");
    }

    #[test]
    fn tf32_profile_keeps_strict_cpu_side_limits() {
        // Patches and log-mel are CPU-computed in every configuration: never loosened.
        let f = Tolerances::FLOAT32;
        for t in [Tolerances::TF32, Tolerances::BF16] {
            assert_eq!((t.pixels.max_abs, t.pixels.min_cos), (f.pixels.max_abs, f.pixels.min_cos));
            assert_eq!((t.mel.max_abs, t.mel.min_cos), (f.mel.max_abs, f.mel.min_cos));
        }
        // And the float32 profile still carries the original constants.
        assert_eq!(f.embedding.max_abs, TOL_EMBEDDING);
        assert_eq!(f.soft_tokens.max_abs, TOL_SOFT_TOKENS);
        assert_eq!(f.pixels.max_abs, TOL_PIXELS);
        assert_eq!(f.mel.max_abs, TOL_MEL);
    }

    #[test]
    fn float32_profile_rejects_tf32_sized_noise() {
        let want = unit_vec();
        let mut got = want.clone();
        got[0] += 3.6e-4; // the worst TF32 embedding deviation measured
        assert!(!check(&got, &want, Tolerances::FLOAT32.embedding).passed());
        assert!(check(&got, &want, Tolerances::TF32.embedding).passed());
    }

    #[test]
    fn tf32_profile_accepts_tf32_noise_and_rejects_real_errors() {
        let want = unit_vec();
        let signed = |d: f32| -> Vec<f32> { want.iter().enumerate().map(|(i, x)| x + if i % 2 == 0 { d } else { -d }).collect() };
        // Small noise on every component (1e-4 abs): TF32-like, must pass both bounds.
        let c = check(&signed(1e-4), &want, Tolerances::TF32.embedding);
        assert!(c.passed(), "{c}");
        // Every component off by 9e-4: inside the 1e-3 max-abs bound, but 1-cos ~ 3e-4 is far below the floor.
        let c = check(&signed(9e-4), &want, Tolerances::TF32.embedding);
        assert!(c.max_abs <= Tolerances::TF32.embedding.max_abs && !c.passed(), "cosine floor must catch it: {c}");
        // A max-abs violation alone also fails, as does a different vector altogether.
        let mut big = want.clone();
        big[5] += 2e-3;
        assert!(!check(&big, &want, Tolerances::TF32.embedding).passed());
        let reversed: Vec<f32> = want.iter().rev().copied().collect();
        assert!(!check(&reversed, &want, Tolerances::TF32.embedding).passed());
        // Soft tokens (~150k values, rms ~1): one entry off by 0.1 is fine, sign flips on a sixteenth of them are not.
        let soft: Vec<f32> = (0..200).flat_map(|_| want.iter().map(|x| x * 768f32.sqrt())).collect();
        let mut soft_noisy = soft.clone();
        soft_noisy[3] += 0.1;
        assert!(check(&soft_noisy, &soft, Tolerances::TF32.soft_tokens).passed());
        let flipped: Vec<f32> = soft.iter().enumerate().map(|(i, x)| if i % 16 == 0 { -x } else { *x }).collect();
        assert!(!check(&flipped, &soft, Tolerances::TF32.soft_tokens).passed());
    }

    #[test]
    fn bf16_profile_accepts_bf16_noise_and_rejects_real_errors() {
        let want = unit_vec();
        // bf16-sized noise, shaped like the measured worst case: +-2e-4 everywhere (1 - cos ~ 2e-5) plus one
        // component off by 1.5e-3 (above the TF32 max-abs limit, inside bf16's).
        let mut noisy: Vec<f32> = want.iter().enumerate().map(|(i, x)| x + if i % 2 == 0 { 2e-4 } else { -2e-4 }).collect();
        noisy[7] += 1.5e-3;
        assert!(check(&noisy, &want, Tolerances::BF16.embedding).passed());
        assert!(!check(&noisy, &want, Tolerances::TF32.embedding).passed());
        assert!(!check(&noisy, &want, Tolerances::FLOAT32.embedding).passed());
        // A real bug -- a sign flip on a sixteenth of the components -- stays far below the cosine floor.
        let flipped: Vec<f32> = want.iter().enumerate().map(|(i, x)| if i % 16 == 0 { -x } else { *x }).collect();
        assert!(!check(&flipped, &want, Tolerances::BF16.embedding).passed());
        // So does a uniform 3e-3 error, which passes the max-abs bound alone.
        let shifted: Vec<f32> = want.iter().map(|x| x + 3e-3).collect();
        assert!(!check(&shifted, &want, Tolerances::BF16.embedding).passed());
    }

    #[test]
    fn nan_and_length_mismatch_fail() {
        let want = unit_vec();
        let mut got = want.clone();
        got[0] = f32::NAN;
        assert!(!check(&got, &want, Tolerances::TF32.embedding).passed());
        assert!(!check(&want[..767], &want, Tolerances::TF32.embedding).passed());
        // A cosine floor with no cosine available can never pass.
        let c = Check { label: "t".into(), max_abs: 0.0, cos: None, ids_ok: None, limit: Tolerances::TF32.embedding, note: String::new() };
        assert!(!c.passed());
    }

    #[test]
    fn id_mismatch_fails_in_every_profile() {
        let v = unit_vec();
        for t in [Tolerances::FLOAT32, Tolerances::TF32, Tolerances::BF16] {
            assert!(!vec_check("t", &v, &v, t.embedding, Some(false), String::new()).passed());
        }
    }
}
