use anyhow::{Context, Result, bail};
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, Subcommand};
use embeddinggemma2::{Device, EmbeddingGemma2, Precision, Input, Part, imageproc::ResizeMode, media, model::truncate_and_normalize, verify};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "egemma2",
    version,
    about = "EmbeddingGemma 2 in Rust: text, image, audio and video embeddings in one space",
    long_about = None
)]
struct Cli {
    /// Directory with config.json, model.safetensors, tokenizer.json, ...
    #[arg(long, global = true, default_value = "models/embeddinggemma-2")]
    model_dir: PathBuf,
    /// Number of worker threads for large passes (default: all logical CPUs).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Threads for small passes, e.g. short texts (default: a third of the logical CPUs, 2..=8).
    #[arg(long, global = true)]
    small_threads: Option<usize>,
    /// auto (CUDA if available, else CPU), cpu, cuda or cuda:N.
    #[arg(long, global = true, default_value = "auto")]
    device: String,
    /// Let cuBLAS use TF32 tensor cores on the GPU (faster; ~4e-4 deviation from float32).
    #[arg(long, global = true, conflicts_with = "bf16")]
    tf32: bool,
    /// Run the GPU's linear layers on bf16 tensor cores (bf16 weights, float32 accumulation; fastest).
    #[arg(long, global = true)]
    bf16: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Embed texts, images, audio clips and videos.
    ///
    /// Every --text/--image/--audio/--video value becomes its own embedding. With --combine, all of
    /// them (in command-line order) form ONE interleaved input instead.
    Embed {
        /// Text to embed (repeatable). Values may start with '-'.
        #[arg(long, allow_hyphen_values = true)]
        text: Vec<String>,
        /// Image file (png/jpg/webp/gif/bmp/tiff; repeatable).
        #[arg(long)]
        image: Vec<PathBuf>,
        /// Audio file (.wav; repeatable).
        #[arg(long)]
        audio: Vec<PathBuf>,
        /// Video: a directory of frames, an animated .gif, or any file if ffmpeg is installed.
        #[arg(long)]
        video: Vec<PathBuf>,
        /// Embed everything given as a single interleaved input.
        #[arg(long)]
        combine: bool,
        /// Prompt: SearchQuery, Document, QuestionAnswering, FactChecking, CodeRetrieval,
        /// Classification, Clustering, SentenceSimilarity, ... Like sentence-transformers, it is
        /// placed before the whole input (media included); the model card uses prompts for text only.
        #[arg(long)]
        prompt_name: Option<String>,
        /// Matryoshka dimension: 768 (default), 512, 256 or 128.
        #[arg(long)]
        dim: Option<usize>,
        /// Native frame rate of --video inputs that are frame directories (enables 1 fps sampling).
        #[arg(long)]
        fps: Option<f64>,
        /// Truncate audio to this many seconds (reference default: 30).
        #[arg(long)]
        max_audio_seconds: Option<f64>,
        /// Use Pillow's resampler for images instead of PyTorch's (the reference's).
        #[arg(long)]
        pil_resize: bool,
        /// Write JSON here instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Rank documents against a query (SearchQuery vs Document prompts, cosine similarity).
    Similarity {
        #[arg(long, allow_hyphen_values = true)]
        query: String,
        /// Document text (repeatable).
        #[arg(long, allow_hyphen_values = true)]
        doc: Vec<String>,
        /// File with one document per line.
        #[arg(long)]
        docs_file: Option<PathBuf>,
        #[arg(long)]
        dim: Option<usize>,
    },
    /// Time the standard benchmark inputs (see tools/bench_ref.py for the PyTorch counterpart).
    Bench {
        #[arg(long, default_value = "testdata")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 5)]
        iters: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Check parity against reference data dumped from PyTorch by the scripts in tools/.
    Verify {
        #[arg(long, default_value = "testdata")]
        reference_dir: PathBuf,
        /// Only run: text, image, audio, video or extra.
        #[arg(long)]
        only: Option<String>,
    },
}

enum Spec {
    Text(String),
    Image(PathBuf),
    Audio(PathBuf),
    Video(PathBuf),
}

impl Spec {
    fn label(&self) -> String {
        match self {
            Spec::Text(t) => format!("text: {}", if t.chars().count() > 60 { format!("{}...", t.chars().take(60).collect::<String>()) } else { t.clone() }),
            Spec::Image(p) => format!("image: {}", p.display()),
            Spec::Audio(p) => format!("audio: {}", p.display()),
            Spec::Video(p) => format!("video: {}", p.display()),
        }
    }
}

/// Collect `--text/--image/--audio/--video` values in the order they appeared on the command line.
fn ordered_specs(m: &ArgMatches) -> Vec<Spec> {
    fn collect<T: Clone + Send + Sync + 'static>(m: &ArgMatches, id: &str, f: fn(T) -> Spec, out: &mut Vec<(usize, Spec)>) {
        if let (Some(vals), Some(idx)) = (m.get_many::<T>(id), m.indices_of(id)) {
            out.extend(idx.zip(vals.cloned()).map(|(i, s)| (i, f(s))));
        }
    }
    let mut v: Vec<(usize, Spec)> = Vec::new();
    collect::<String>(m, "text", Spec::Text, &mut v);
    collect::<PathBuf>(m, "image", Spec::Image, &mut v);
    collect::<PathBuf>(m, "audio", Spec::Audio, &mut v);
    collect::<PathBuf>(m, "video", Spec::Video, &mut v);
    v.sort_by_key(|(i, _)| *i);
    v.into_iter().map(|(_, s)| s).collect()
}

fn load_part(model: &EmbeddingGemma2, spec: &Spec, fps: Option<f64>) -> Result<Part> {
    Ok(match spec {
        Spec::Text(t) => Part::Text(t.clone()),
        Spec::Image(p) => Part::Image(media::load_image(p)?),
        Spec::Audio(p) => Part::Audio(media::load_audio(p)?),
        Spec::Video(p) => Part::Video(media::load_video(p, fps, &model.video_sample_params())?),
    })
}

fn cmd_embed(model: &mut EmbeddingGemma2, m: &ArgMatches) -> Result<()> {
    let get = |k: &str| m.get_one::<String>(k).cloned();
    let prompt_name = get("prompt_name");
    let dim = m.get_one::<usize>("dim").copied();
    let fps = m.get_one::<f64>("fps").copied();
    if let Some(f) = fps {
        if !(f.is_finite() && f > 0.0) {
            bail!("--fps must be a positive number (got {f})");
        }
    }
    if m.get_flag("pil_resize") {
        model.resize_mode = ResizeMode::Pillow;
    }
    if let Some(s) = m.get_one::<f64>("max_audio_seconds") {
        if !(s.is_finite() && *s > 0.0) {
            bail!("--max-audio-seconds must be a positive number (got {s})");
        }
        model.max_audio_samples = (s * model.proc_cfg.feature_extractor.sampling_rate as f64) as usize;
    }
    let prefix = match &prompt_name {
        Some(n) => model.prompt(n)?.to_string(),
        None => String::new(),
    };
    let specs = ordered_specs(m);
    if specs.is_empty() {
        bail!("nothing to embed: pass at least one of --text, --image, --audio, --video");
    }

    let groups: Vec<Vec<&Spec>> = if m.get_flag("combine") { vec![specs.iter().collect()] } else { specs.iter().map(|s| vec![s]).collect() };
    let (mut labels, mut embs) = (Vec::new(), Vec::new());
    for g in groups {
        let t0 = Instant::now();
        let parts = g.iter().map(|s| load_part(model, s, fps)).collect::<Result<Vec<_>>>()?;
        let emb = truncate_and_normalize(model.embed(&Input { parts }.with_prompt(&prefix))?, dim)?;
        let label = g.iter().map(|s| s.label()).collect::<Vec<_>>().join(" + ");
        eprintln!("embedded [{label}] in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        labels.push(label);
        embs.push(emb);
    }
    let json = serde_json::json!({ "dim": embs[0].len(), "inputs": labels, "embeddings": embs });
    match m.get_one::<PathBuf>("out") {
        Some(p) => std::fs::write(p, serde_json::to_string(&json)?).with_context(|| format!("writing {}", p.display()))?,
        None => println!("{}", serde_json::to_string(&json)?),
    }
    Ok(())
}

fn cmd_similarity(model: &EmbeddingGemma2, m: &ArgMatches) -> Result<()> {
    let query = m.get_one::<String>("query").unwrap();
    let dim = m.get_one::<usize>("dim").copied();
    let mut docs: Vec<String> = m.get_many::<String>("doc").map(|v| v.cloned().collect()).unwrap_or_default();
    if let Some(f) = m.get_one::<PathBuf>("docs_file") {
        let text = std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
        // Windows editors often write a UTF-8 BOM; it would otherwise become part of the first document.
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        docs.extend(text.lines().filter(|l| !l.trim().is_empty()).map(String::from));
    }
    if docs.is_empty() {
        bail!("pass at least one --doc or --docs-file");
    }
    let q = truncate_and_normalize(model.embed_text(Some("SearchQuery"), query)?, dim)?;
    let mut scored: Vec<(f32, &String)> = Vec::new();
    for d in &docs {
        let e = truncate_and_normalize(model.embed_text(Some("Document"), d)?, dim)?;
        scored.push((q.iter().zip(&e).map(|(a, b)| a * b).sum(), d));
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (s, d) in scored {
        println!("{s:.4}  {d}");
    }
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn cmd_bench(model: &EmbeddingGemma2, m: &ArgMatches) -> Result<()> {
    let dir = m.get_one::<PathBuf>("data_dir").unwrap();
    let iters = *m.get_one::<usize>("iters").unwrap();
    if iters == 0 {
        bail!("--iters must be at least 1");
    }
    let path = dir.join("bench_inputs.json");
    let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {} (made by tools/bench_ref.py)", path.display()))?;
    let inputs: serde_json::Value = serde_json::from_str(&raw)?;
    let field = |k: &str| inputs[k].as_str().map(String::from).with_context(|| format!("{} has no string `{k}`", path.display()));
    let (short, long) = (field("short")?, field("long")?);
    let batch: Vec<String> = inputs["batch64"]
        .as_array()
        .and_then(|a| a.iter().map(|v| v.as_str().map(String::from)).collect())
        .with_context(|| format!("{} has no string array `batch64`", path.display()))?;
    let img = media::load_image(&dir.join("image/img_a_641x479.png"))?;
    let wav = media::load_audio(&dir.join("audio/c_9p3s.wav"))?;

    type Case<'a> = (&'a str, Box<dyn Fn() -> Result<()> + 'a>);
    let cases: Vec<Case> = vec![
        ("short (17 tok)", Box::new(|| model.embed(&Input::text(short.clone())).map(|_| ()))),
        ("long (2208 tok)", Box::new(|| model.embed(&Input::text(long.clone())).map(|_| ()))),
        ("batch of 64 short docs", Box::new(|| model.embed_texts(&batch).map(|_| ()))),
        ("image 641x479", Box::new(|| model.embed(&Input::image(img.clone())).map(|_| ()))),
        ("audio 9.3 s", Box::new(|| model.embed(&Input::audio(wav.clone())).map(|_| ()))),
    ];
    let mut res = serde_json::Map::new();
    for (name, f) in &cases {
        f()?; // warm-up
        let mut ts = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            f()?;
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let (med, min) = (median(ts.clone()), ts.iter().cloned().fold(f64::INFINITY, f64::min));
        eprintln!("{name:<26} median {med:9.1} ms   min {min:9.1} ms");
        res.insert(name.to_string(), serde_json::json!({ "median_ms": med, "min_ms": min }));
    }
    res.insert("_threads".into(), serde_json::json!(rayon::current_num_threads()));
    if let Some(p) = m.get_one::<PathBuf>("out") {
        std::fs::write(p, serde_json::to_string(&res)?)?;
    }
    Ok(())
}

fn cmd_verify(model: &EmbeddingGemma2, m: &ArgMatches) -> Result<()> {
    let dir = m.get_one::<PathBuf>("reference_dir").unwrap();
    let only = m.get_one::<String>("only").map(String::as_str);
    let want = |k: &str| only.is_none() || only == Some(k);
    // The limits follow the device the model really runs on: strict float32 for the CPU and plain
    // CUDA, a measured TF32 profile (max|d| bound + cosine floor) for `--device cuda --tf32`.
    println!("device {}, tolerance profile {}", model.device, verify::Tolerances::for_model(model));
    let mut checks: Vec<verify::Check> = Vec::new();
    let mut run = |name: &str, f: &dyn Fn() -> Result<Vec<verify::Check>>| -> Result<()> {
        if want(name) {
            let t0 = Instant::now();
            let r = f()?;
            eprintln!("[{name}] {} checks in {:.1}s", r.len(), t0.elapsed().as_secs_f64());
            checks.extend(r);
        }
        Ok(())
    };
    run("text", &|| verify::check_text(model, &dir.join("ref_text.json")))?;
    run("image", &|| verify::check_image(model, &dir.join("image")))?;
    run("audio", &|| verify::check_audio(model, &dir.join("audio")))?;
    run("video", &|| verify::check_video(model, &dir.join("video")))?;
    run("video", &|| verify::check_video_files(model, &dir.join("video")))?;
    run("extra", &|| verify::check_extra(model, &dir.join("extra")))?;
    for c in &checks {
        println!("{c}");
    }
    println!("\nworst case per quantity:\n{}", verify::summarize(&checks));
    let failed = checks.iter().filter(|c| !c.passed()).count();
    println!("{} checks, {} failed", checks.len(), failed);
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn load_model(dir: &Path, device: &str, precision: Precision) -> Result<EmbeddingGemma2> {
    let cuda = |ordinal| Device::Cuda { ordinal, precision };
    match device {
        "cpu" => EmbeddingGemma2::load_on(dir, Device::Cpu),
        "cuda" => EmbeddingGemma2::load_on(dir, cuda(0)),
        "auto" => EmbeddingGemma2::load_on(dir, cuda(0)).or_else(|e| {
            eprintln!("CUDA unavailable ({e:#}); running on the CPU");
            EmbeddingGemma2::load_on(dir, Device::Cpu)
        }),
        d => match d.strip_prefix("cuda:").and_then(|n| n.parse().ok()) {
            Some(n) => EmbeddingGemma2::load_on(dir, cuda(n)),
            None => bail!("--device must be auto, cpu, cuda or cuda:N (got `{d}`)"),
        },
    }
}

fn main() -> Result<()> {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches)?;
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }
    if let Some(n) = cli.small_threads {
        embeddinggemma2::ops::set_small_pool_threads(n);
    }
    let (name, sub) = matches.subcommand().expect("subcommand required");
    let t0 = Instant::now();
    let precision = if cli.bf16 {
        Precision::Bf16
    } else if cli.tf32 {
        Precision::Tf32
    } else {
        Precision::F32
    };
    let mut model = load_model(&cli.model_dir, &cli.device, precision)
        .with_context(|| format!("loading model from {}", Path::new(&cli.model_dir).display()))?;
    eprintln!("model loaded on {} in {:.0} ms", model.device, t0.elapsed().as_secs_f64() * 1e3);
    match name {
        "embed" => cmd_embed(&mut model, sub),
        "similarity" => cmd_similarity(&model, sub),
        "verify" => cmd_verify(&model, sub),
        "bench" => cmd_bench(&model, sub),
        other => bail!("unknown subcommand {other}"),
    }
}
