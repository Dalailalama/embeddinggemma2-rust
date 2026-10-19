//! Top-level model: prompt assembly, tokenisation, multimodal soft-token merging, pooling.

use anyhow::{Context, Result, anyhow, bail, ensure};
use image::RgbImage;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::Tokenizer;

use crate::audio::{self, AudioTower};
use crate::backend::{AudioBackend, Device, TextBackend, VisionBackend};
use crate::audioproc::{self, MelExtractor};
use crate::config::{self, ModelConfig, ProcessorConfig};
use crate::imageproc::{self, PatchedImage, ResizeMode};
use crate::media::AudioClip;
use crate::ops;
use crate::text::TextModel;
use crate::videoproc::{self, SampleParams};
use crate::vision::VisionTower;
use crate::weights::Weights;

pub const IMAGE_PLACEHOLDER: &str = "<|image|>";
pub const VIDEO_PLACEHOLDER: &str = "<|video|>";
pub const AUDIO_PLACEHOLDER: &str = "<|audio|>";
const BOI: &str = "<|image>";
const EOI: &str = "<image|>";
const BOA: &str = "<|audio>";
const EOA: &str = "<audio|>";

/// A decoded video: RGB frames plus the native frame rate when known.
#[derive(Clone)]
pub struct VideoClip {
    pub frames: Vec<RgbImage>,
    pub fps: Option<f64>,
}

/// One piece of an (optionally interleaved) multimodal input.
#[derive(Clone)]
pub enum Part {
    Text(String),
    Image(RgbImage),
    Audio(AudioClip),
    Video(VideoClip),
}

/// A single item to embed. Parts are concatenated in order into one prompt.
#[derive(Clone)]
pub struct Input {
    pub parts: Vec<Part>,
}

impl Input {
    pub fn text(s: impl Into<String>) -> Self {
        Self { parts: vec![Part::Text(s.into())] }
    }
    pub fn image(img: RgbImage) -> Self {
        Self { parts: vec![Part::Image(img)] }
    }
    pub fn audio(a: AudioClip) -> Self {
        Self { parts: vec![Part::Audio(a)] }
    }
    pub fn video(v: VideoClip) -> Self {
        Self { parts: vec![Part::Video(v)] }
    }

    /// Put a prompt (e.g. `model.prompt("SearchQuery")?`) in front of the whole input.
    ///
    /// sentence-transformers passes the prompt to this model as a system message, which its chat
    /// template renders before all other content -- media included -- with no separator. So the
    /// prompt goes first even when the input starts with an image; an empty prompt is a no-op.
    pub fn with_prompt(mut self, prompt: &str) -> Self {
        if !prompt.is_empty() {
            self.parts.insert(0, Part::Text(prompt.to_string()));
        }
        self
    }
}

pub struct EmbeddingGemma2 {
    pub dir: PathBuf,
    pub cfg: ModelConfig,
    pub proc_cfg: ProcessorConfig,
    pub prompts: HashMap<String, String>,
    weights: Arc<Weights>,
    tokenizer: Tokenizer,
    text: TextBackend,
    mel: MelExtractor,
    vision: OnceLock<VisionBackend>,
    audio: OnceLock<AudioBackend>,
    /// Serialises lazy tower construction so concurrent first calls load (and upload) a tower once.
    tower_init: Mutex<()>,
    #[cfg(feature = "cuda")]
    gpu: Option<Arc<crate::gpu::Gpu>>,
    /// Device the towers run on.
    pub device: Device,
    /// Optional cap on text-only sequence length (tokens, including `<bos>`/`<eos>`); longer texts keep
    /// their first `n - 1` tokens plus `<eos>`. `None` (the default) embeds every token, as
    /// sentence-transformers does for this model (its `max_seq_length` is unbounded). The model was
    /// trained on positions up to `text_config.max_position_embeddings` (262 144), and memory grows with
    /// the length: about 120 KB per token on the GPU, plus the attention scores of the global layers.
    pub max_text_tokens: Option<usize>,
    /// Resampler used for images and video frames (the reference uses PyTorch's).
    pub resize_mode: ResizeMode,
    /// Audio longer than this many 16 kHz samples is truncated (the reference extractor's default
    /// is 480 000 = 30 s).
    pub max_audio_samples: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Image,
    Video,
    Audio,
}

const KINDS: [(Kind, &str); 3] =
    [(Kind::Image, IMAGE_PLACEHOLDER), (Kind::Video, VIDEO_PLACEHOLDER), (Kind::Audio, AUDIO_PLACEHOLDER)];

/// An [`Input`] after tokenisation and media preprocessing (resized patches, log-mel features), ready for
/// the towers and the text backbone.
///
/// Making one ([`EmbeddingGemma2::prepare_input`]) runs on the CPU only, so callers can prepare the next
/// inputs on other threads while the device runs [`EmbeddingGemma2::embed_prepared`].
pub struct PreparedInput {
    ids: Vec<u32>,
    images: Vec<PatchedImage>,
    videos: Vec<Vec<PatchedImage>>,
    /// Log-mel features `[frames, n_mels]` and the frame count, per audio clip.
    mels: Vec<(Vec<f32>, usize)>,
}

impl PreparedInput {
    /// Token ids fed to the text backbone (prompt, text, special tokens and soft-token slots).
    pub fn token_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Sequence length the text backbone will process.
    pub fn num_tokens(&self) -> usize {
        self.ids.len()
    }

    /// Whether the vision or audio tower has to run for this input.
    pub fn has_media(&self) -> bool {
        !(self.images.is_empty() && self.videos.is_empty() && self.mels.is_empty())
    }
}

/// Soft tokens of one input's media, `[n, hidden]` each, in placeholder order.
struct SoftTokens {
    image: Vec<f32>,
    video: Vec<f32>,
    audio: Vec<f32>,
}

/// Rows of one shared text-backbone pass when inputs are batched (the weights are then streamed once per
/// pack, not once per sequence); a longer sequence forms a pack of its own.
const PACK_TOKENS: usize = 2048;
/// Most inputs [`EmbeddingGemma2::embed_batch`] prepares at once, and most media among them (an image or audio
/// clip counts 1, a video half its frames): this bounds the memory held by preprocessed images (about 8 MB
/// each at the largest size).
const PREPARE_WINDOW: usize = 256;
const PREPARE_WINDOW_MEDIA: usize = 16;

impl EmbeddingGemma2 {
    /// Load on the CPU.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    /// Load on a given device. CUDA requires the `cuda` feature (on by default) and a working
    /// NVIDIA driver with cuBLAS/NVRTC; see [`Device`].
    pub fn load_on(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let cfg = ModelConfig::load(&dir)?;
        let proc_cfg = ProcessorConfig::load(&dir)?;
        let prompts = config::load_prompts(&dir)?;
        let weights = Arc::new(Weights::open(&dir.join("model.safetensors"))?);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let cpu_text = TextModel::load(&cfg.text_config, weights.clone()).context("loading text backbone")?;
        #[cfg(feature = "cuda")]
        let gpu = match device {
            Device::Cpu => None,
            Device::Cuda { ordinal, precision } => Some(Arc::new(crate::gpu::try_open(ordinal, precision)?)),
        };
        #[cfg(not(feature = "cuda"))]
        if let Device::Cuda { .. } = device {
            bail!("this build has no CUDA support (rebuild with `--features cuda`)");
        }
        #[cfg(feature = "cuda")]
        let text = match &gpu {
            Some(g) => TextBackend::Gpu(crate::gpu::text::GpuText::from_cpu(cpu_text, g.clone())?),
            None => TextBackend::Cpu(cpu_text),
        };
        #[cfg(not(feature = "cuda"))]
        let text = TextBackend::Cpu(cpu_text);
        let mel = MelExtractor::new(&proc_cfg.feature_extractor);
        Ok(Self {
            dir,
            cfg,
            proc_cfg,
            prompts,
            weights,
            tokenizer,
            text,
            mel,
            vision: OnceLock::new(),
            audio: OnceLock::new(),
            tower_init: Mutex::new(()),
            #[cfg(feature = "cuda")]
            gpu,
            device,
            max_text_tokens: None,
            resize_mode: ResizeMode::default(),
            max_audio_samples: audioproc::DEFAULT_MAX_SAMPLES,
        })
    }

    fn vision(&self) -> Result<&VisionBackend> {
        if let Some(v) = self.vision.get() {
            return Ok(v);
        }
        let _init = self.tower_init.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = self.vision.get() {
            return Ok(v);
        }
        let vc = self.cfg.vision_config.as_ref().ok_or_else(|| anyhow!("model has no vision_config"))?;
        let tower = VisionTower::load(vc, self.cfg.text_config.hidden_size, &self.weights)
            .context("loading vision tower")?;
        #[cfg(feature = "cuda")]
        let backend = match &self.gpu {
            Some(g) => VisionBackend::Gpu(crate::gpu::vision::GpuVision::from_cpu(tower, g.clone())?),
            None => VisionBackend::Cpu(tower),
        };
        #[cfg(not(feature = "cuda"))]
        let backend = VisionBackend::Cpu(tower);
        Ok(self.vision.get_or_init(|| backend))
    }

    fn audio(&self) -> Result<&AudioBackend> {
        if let Some(a) = self.audio.get() {
            return Ok(a);
        }
        let _init = self.tower_init.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(a) = self.audio.get() {
            return Ok(a);
        }
        let ac = self.cfg.audio_config.as_ref().ok_or_else(|| anyhow!("model has no audio_config"))?;
        let tower = AudioTower::load(ac, self.cfg.text_config.hidden_size, &self.weights)
            .context("loading audio tower")?;
        #[cfg(feature = "cuda")]
        let backend = match &self.gpu {
            Some(g) => AudioBackend::Gpu(crate::gpu::audio::GpuAudio::from_cpu(tower, g.clone())?),
            None => AudioBackend::Cpu(tower),
        };
        #[cfg(not(feature = "cuda"))]
        let backend = AudioBackend::Cpu(tower);
        Ok(self.audio.get_or_init(|| backend))
    }

    /// Prompt prefix for a sentence-transformers prompt name such as `SearchQuery` or `Document`.
    pub fn prompt(&self, name: &str) -> Result<&str> {
        self.prompts.get(name).map(String::as_str).ok_or_else(|| {
            anyhow!("unknown prompt name `{name}` (known: {:?})", self.prompts.keys().collect::<Vec<_>>())
        })
    }

    /// The CUDA device the model runs on, if any (for profiling tools).
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    pub fn gpu(&self) -> Option<&crate::gpu::Gpu> {
        self.gpu.as_deref()
    }

    /// Tokenise a plain string the way the reference does (`<bos> ... <eos>`).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn truncate_text_ids(&self, mut ids: Vec<u32>) -> Vec<u32> {
        if let Some(max) = self.max_text_tokens.filter(|&m| ids.len() > m) {
            // Keep `<bos>` first and `<eos>` last, like HF truncation with special tokens.
            let eos = *ids.last().unwrap();
            ids.truncate(max.max(2) - 1);
            ids.push(eos);
        }
        ids
    }

    // -----------------------------------------------------------------------------------------
    // Input preparation
    // -----------------------------------------------------------------------------------------

    /// Tokenise an input and preprocess its media, without running any tower (CPU only; see
    /// [`PreparedInput`]).
    pub fn prepare_input(&self, input: &Input) -> Result<PreparedInput> {
        // 1. Assemble the template. Media become placeholders in part order, unless the text
        //    already carries manual placeholders (the chat template's `has_manual_placeholders`).
        let manual = input.parts.iter().any(|p| match p {
            Part::Text(t) => KINDS.iter().any(|(_, ph)| t.contains(ph)),
            _ => false,
        });
        let mut template = String::new();
        let mut images: Vec<&RgbImage> = Vec::new();
        let mut videos: Vec<&VideoClip> = Vec::new();
        let mut audios: Vec<&AudioClip> = Vec::new();
        for p in &input.parts {
            match p {
                Part::Text(t) => template.push_str(t),
                Part::Image(i) => {
                    images.push(i);
                    if !manual {
                        template.push_str(IMAGE_PLACEHOLDER);
                    }
                }
                Part::Video(v) => {
                    videos.push(v);
                    if !manual {
                        template.push_str(VIDEO_PLACEHOLDER);
                    }
                }
                Part::Audio(a) => {
                    audios.push(a);
                    if !manual {
                        template.push_str(AUDIO_PLACEHOLDER);
                    }
                }
            }
        }
        for (ph, n) in [
            (IMAGE_PLACEHOLDER, images.len()),
            (VIDEO_PLACEHOLDER, videos.len()),
            (AUDIO_PLACEHOLDER, audios.len()),
        ] {
            ensure!(
                template.matches(ph).count() == n,
                "number of `{ph}` placeholders does not match the number of media items passed ({n})"
            );
        }

        // 2. Preprocess media (cheap) to learn the soft-token counts.
        let ic = &self.proc_cfg.image_processor;
        let patched_images: Vec<PatchedImage> = images
            .iter()
            .map(|im| {
                imageproc::preprocess_rgb(
                    im.as_raw(),
                    im.height() as usize,
                    im.width() as usize,
                    ic.patch_size,
                    ic.pooling_kernel_size,
                    ic.max_soft_tokens,
                    self.resize_mode,
                )
            })
            .collect::<Result<_>>()?;

        let vc = &self.proc_cfg.video_processor;
        let mut patched_videos: Vec<Vec<PatchedImage>> = Vec::new();
        for v in &videos {
            patched_videos.push(self.preprocess_video(v)?);
        }

        let n_mels = self.proc_cfg.feature_extractor.feature_size;
        let mut mels: Vec<(Vec<f32>, usize)> = Vec::new();
        for a in &audios {
            let wav = a.at_rate_truncated(self.mel.sampling_rate(), self.max_audio_samples)?;
            let mel = self.mel.extract(&wav)?;
            let frames = mel.len() / n_mels;
            mels.push((mel, frames));
        }

        // 3. Expand each placeholder into `<begin> soft-token slots <end>`, in text order.
        let mut expanded = String::with_capacity(template.len() + 4096);
        let (mut ni, mut nv, mut na) = (0usize, 0usize, 0usize);
        let mut rest = template.as_str();
        while !rest.is_empty() {
            let next = KINDS
                .iter()
                .filter_map(|(k, ph)| rest.find(ph).map(|pos| (pos, *k, *ph)))
                .min_by_key(|(pos, _, _)| *pos);
            let Some((pos, kind, ph)) = next else {
                expanded.push_str(rest);
                break;
            };
            expanded.push_str(&rest[..pos]);
            match kind {
                Kind::Image => {
                    let n = patched_images[ni].num_soft_tokens(ic.pooling_kernel_size);
                    ni += 1;
                    expanded.push_str(BOI);
                    expanded.push_str(&IMAGE_PLACEHOLDER.repeat(n));
                    expanded.push_str(EOI);
                }
                Kind::Video => {
                    for f in &patched_videos[nv] {
                        let n = f.num_soft_tokens(vc.pooling_kernel_size);
                        expanded.push_str(BOI);
                        expanded.push_str(&VIDEO_PLACEHOLDER.repeat(n));
                        expanded.push_str(EOI);
                    }
                    nv += 1;
                }
                Kind::Audio => {
                    let n = audio::num_audio_tokens(mels[na].1);
                    na += 1;
                    expanded.push_str(BOA);
                    expanded.push_str(&AUDIO_PLACEHOLDER.repeat(n));
                    expanded.push_str(EOA);
                }
            }
            rest = &rest[pos + ph.len()..];
        }

        // 4. Tokenise (text-only inputs are truncated to `max_text_tokens` if set; media inputs are not).
        let mut ids = self.tokenize(&expanded)?;
        if images.is_empty() && videos.is_empty() && audios.is_empty() {
            ids = self.truncate_text_ids(ids);
        }

        Ok(PreparedInput { ids, images: patched_images, videos: patched_videos, mels })
    }

    /// Run the vision / audio towers over a prepared input's media.
    fn soft_tokens(&self, p: &PreparedInput) -> Result<SoftTokens> {
        let mut st = SoftTokens { image: Vec::new(), video: Vec::new(), audio: Vec::new() };
        if !p.images.is_empty() || !p.videos.is_empty() {
            let tower = self.vision()?;
            for im in &p.images {
                st.image.extend(tower.encode(im)?);
            }
            for frames in &p.videos {
                for f in frames {
                    st.video.extend(tower.encode(f)?);
                }
            }
        }
        if !p.mels.is_empty() {
            let tower = self.audio()?;
            let n_mels = self.proc_cfg.feature_extractor.feature_size;
            for (mel, frames) in &p.mels {
                st.audio.extend(tower.encode(mel, *frames, n_mels)?);
            }
        }
        Ok(st)
    }

    /// Write the input embeddings `[ids.len(), hidden]` of a prepared input into `embeds`: hard-token
    /// embeddings, with the towers' soft tokens at their placeholder positions.
    fn fill_embeds(&self, p: &PreparedInput, embeds: &mut [f32]) -> Result<()> {
        let h = self.cfg.text_config.hidden_size;
        self.text.embed_tokens(&p.ids, embeds)?;
        if !p.has_media() {
            return Ok(());
        }
        let soft = self.soft_tokens(p)?;
        let mut used = [0usize; 3];
        let streams: [(&[f32], u32); 3] = [
            (&soft.image, self.cfg.image_token_id),
            (&soft.video, self.cfg.video_token_id),
            (&soft.audio, self.cfg.audio_token_id),
        ];
        for (t, &id) in p.ids.iter().enumerate() {
            for (s, (buf, tok)) in streams.iter().enumerate() {
                if id == *tok {
                    let n = used[s];
                    ensure!((n + 1) * h <= buf.len(), "more media placeholders than soft tokens");
                    embeds[t * h..(t + 1) * h].copy_from_slice(&buf[n * h..(n + 1) * h]);
                    used[s] += 1;
                }
            }
        }
        for (s, (buf, _)) in streams.iter().enumerate() {
            ensure!(used[s] * h == buf.len(), "soft tokens ({}) != placeholders ({})", buf.len() / h, used[s]);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------------------------

    /// Embed one input; returns the L2-normalised 768-d vector.
    pub fn embed(&self, input: &Input) -> Result<Vec<f32>> {
        Ok(self.embed_with_ids(input)?.1)
    }

    /// Like [`embed`](Self::embed) but also returns the token ids that were fed to the encoder.
    pub fn embed_with_ids(&self, input: &Input) -> Result<(Vec<u32>, Vec<f32>)> {
        let p = self.prepare_input(input)?;
        let emb = self.embed_prepared(std::slice::from_ref(&p))?.pop().unwrap();
        Ok((p.ids, emb))
    }

    /// Embed pre-tokenised text.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let p = PreparedInput { ids: ids.to_vec(), images: vec![], videos: vec![], mels: vec![] };
        Ok(self.embed_prepared(std::slice::from_ref(&p))?.pop().unwrap())
    }

    /// Embed prepared inputs (see [`prepare_input`](Self::prepare_input)); returns one L2-normalised 768-d
    /// vector per input.
    ///
    /// The towers run input by input; the text backbone then processes consecutive inputs packed into
    /// shared passes of up to `PACK_TOKENS` rows, media inputs included.
    pub fn embed_prepared(&self, items: &[PreparedInput]) -> Result<Vec<Vec<f32>>> {
        let h = self.cfg.text_config.hidden_size;
        let dim = self.cfg.text_config.embedding_dim;
        let mut out = Vec::with_capacity(items.len());
        let mut start = 0;
        while start < items.len() {
            // Greedy pack in input order.
            let mut end = start + 1;
            let mut rows = items[start].ids.len();
            while end < items.len() && rows + items[end].ids.len() <= PACK_TOKENS {
                rows += items[end].ids.len();
                end += 1;
            }
            let lens: Vec<usize> = items[start..end].iter().map(|p| p.ids.len()).collect();
            ensure!(lens.iter().all(|&l| l > 0), "empty token sequence");
            let mut embeds = vec![0f32; rows * h];
            let mut off = 0;
            for p in &items[start..end] {
                self.fill_embeds(p, &mut embeds[off * h..(off + p.ids.len()) * h])?;
                off += p.ids.len();
            }
            let hidden = self.text.forward_packed(&embeds, &lens)?;
            let mut off = 0;
            for &l in &lens {
                let mut pooled = ops::mean_rows(&hidden[off * dim..(off + l) * dim], l, dim);
                ops::l2_normalize(&mut pooled);
                out.push(pooled);
                off += l;
            }
            start = end;
        }
        Ok(out)
    }

    /// Embed a batch and optionally apply Matryoshka truncation (+ re-normalisation).
    ///
    /// Inputs are prepared in parallel on the CPU, a window at a time, then embedded with
    /// [`embed_prepared`](Self::embed_prepared): every input, media included, shares packed text-backbone
    /// passes with its neighbours.
    pub fn embed_batch(&self, inputs: &[Input], truncate_dim: Option<usize>) -> Result<Vec<Vec<f32>>> {
        use rayon::prelude::*;
        let max_frames = self.proc_cfg.video_processor.max_frames;
        let media_weight = |i: &Input| -> usize {
            i.parts
                .iter()
                .map(|p| match p {
                    Part::Text(_) => 0,
                    Part::Image(_) | Part::Audio(_) => 1,
                    Part::Video(v) => v.frames.len().min(max_frames).div_ceil(2),
                })
                .sum()
        };
        let mut out = Vec::with_capacity(inputs.len());
        let mut start = 0;
        while start < inputs.len() {
            let (mut end, mut media) = (start, 0);
            while end < inputs.len() && end - start < PREPARE_WINDOW {
                let m = media_weight(&inputs[end]);
                if end > start && media + m > PREPARE_WINDOW_MEDIA {
                    break;
                }
                media += m;
                end += 1;
            }
            let window = &inputs[start..end];
            let prepared: Vec<PreparedInput> = if window.len() == 1 {
                vec![self.prepare_input(&window[0])?]
            } else {
                window.par_iter().map(|i| self.prepare_input(i)).collect::<Result<_>>()?
            };
            for e in self.embed_prepared(&prepared)? {
                out.push(truncate_and_normalize(e, truncate_dim)?);
            }
            start = end;
        }
        Ok(out)
    }

    /// Embed many text strings (prompt prefixes must already be applied), packed into shared passes.
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<Input> = texts.iter().map(|t| Input::text(t.clone())).collect();
        self.embed_batch(&inputs, None)
    }

    /// Convenience: embed text with a named prompt (`SearchQuery`, `Document`, ...).
    pub fn embed_text(&self, prompt_name: Option<&str>, text: &str) -> Result<Vec<f32>> {
        let prefix = match prompt_name {
            Some(n) => self.prompt(n)?,
            None => "",
        };
        self.embed(&Input::text(format!("{prefix}{text}")))
    }

    /// Frame-sampling parameters from `processor_config.json` (`fps`, `max_frames`, overflow strategy).
    pub fn video_sample_params(&self) -> SampleParams {
        let vc = &self.proc_cfg.video_processor;
        SampleParams { fps: Some(vc.fps), max_frames: vc.max_frames, overflow_strategy: vc.overflow_strategy.clone() }
    }

    /// Sample frames from a clip and patchify each one.
    pub fn preprocess_video(&self, clip: &VideoClip) -> Result<Vec<PatchedImage>> {
        let vc = &self.proc_cfg.video_processor;
        let idx = videoproc::sample_frames(clip.frames.len(), clip.fps, &self.video_sample_params())?;
        idx.iter()
            .map(|&i| {
                let f = &clip.frames[i];
                imageproc::preprocess_rgb(
                    f.as_raw(),
                    f.height() as usize,
                    f.width() as usize,
                    vc.patch_size,
                    vc.pooling_kernel_size,
                    vc.max_soft_tokens,
                    self.resize_mode,
                )
            })
            .collect()
    }

    /// Patchified frames and concatenated soft tokens of a clip (used by tests and tools).
    pub fn video_soft_tokens(&self, clip: &VideoClip) -> Result<(Vec<PatchedImage>, Vec<f32>)> {
        let frames = self.preprocess_video(clip)?;
        let tower = self.vision()?;
        let mut soft = Vec::new();
        for f in &frames {
            soft.extend(tower.encode(f)?);
        }
        Ok((frames, soft))
    }

    /// Soft tokens of one image (used by tests and debugging tools).
    pub fn image_soft_tokens(&self, img: &RgbImage) -> Result<(PatchedImage, Vec<f32>)> {
        let ic = &self.proc_cfg.image_processor;
        let patched = imageproc::preprocess_rgb(
            img.as_raw(),
            img.height() as usize,
            img.width() as usize,
            ic.patch_size,
            ic.pooling_kernel_size,
            ic.max_soft_tokens,
            self.resize_mode,
        )?;
        let tokens = self.vision()?.encode(&patched)?;
        Ok((patched, tokens))
    }

    /// Run the vision tower on already-patchified pixels (isolates the tower from preprocessing).
    pub fn soft_tokens_from_patches(&self, patched: &PatchedImage) -> Result<Vec<f32>> {
        self.vision()?.encode(patched)
    }

    /// Log-mel features and soft tokens of one audio clip: `(mel[frames, n_mels], frames, soft_tokens)`.
    pub fn audio_soft_tokens(&self, clip: &AudioClip) -> Result<(Vec<f32>, usize, Vec<f32>)> {
        let wav = clip.at_rate_truncated(self.mel.sampling_rate(), self.max_audio_samples)?;
        let mel = self.mel.extract(&wav)?;
        let n_mels = self.proc_cfg.feature_extractor.feature_size;
        let frames = mel.len() / n_mels;
        let soft = self.audio()?.encode(&mel, frames, n_mels)?;
        Ok((mel, frames, soft))
    }

    /// Run the audio tower on externally computed mel features.
    pub fn soft_tokens_from_mel(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        let n_mels = self.proc_cfg.feature_extractor.feature_size;
        self.audio()?.encode(mel, frames, n_mels)
    }
}

/// Matryoshka truncation: keep the first `dim` components and re-normalise.
pub fn truncate_and_normalize(mut v: Vec<f32>, dim: Option<usize>) -> Result<Vec<f32>> {
    if let Some(d) = dim {
        if d == 0 || d > v.len() {
            bail!("truncate_dim {d} out of range 1..={}", v.len());
        }
        if d < v.len() {
            v.truncate(d);
            ops::l2_normalize(&mut v);
        }
    }
    Ok(v)
}
