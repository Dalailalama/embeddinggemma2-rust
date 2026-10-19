//! A long-running embedding server: the model loads once and answers many requests.
//!
//! Two transports share one request format:
//! * [`run_stdio`]: JSON Lines. One request per line on stdin, one response per line on stdout, in request
//!   order. Meant for driving `egemma2 serve` as a subprocess from any language.
//! * [`run_http`]: an OpenAI-compatible `POST /v1/embeddings` endpoint, plus `GET /health` and
//!   `GET /v1/models`, so OpenAI-style clients (the `openai` SDK, LangChain, LlamaIndex, ...) work as they are.
//!
//! A request is the OpenAI embeddings body with a few extensions:
//!
//! ```json
//! {"input": ["a text", {"image": "page1.png"}, {"text": "caption", "image": "data:image/png;base64,..."}],
//!  "prompt_name": "Document", "dimensions": 256, "encoding_format": "float", "id": 7}
//! ```
//!
//! * `input`: a string, one input, or a list of inputs; each input becomes one embedding. An input is a
//!   string (text) or an object whose `text` / `image` / `audio` / `video` keys form one interleaved
//!   input in key order, like sentence-transformers' dict inputs. `{"parts": [...]}` lists any sequence of
//!   parts (several images, say). OpenAI-style content parts (`{"type": "image_url", "image_url": {"url": ...}}`)
//!   are accepted too.
//! * Media values: a file path, a `data:<type>;base64,...` URL, or an object: `{"path"}`, `{"base64"}`, raw
//!   pixels `{"rgb": <base64 RGB8, row-major>, "width", "height"}`, raw audio
//!   `{"pcm_f32": <base64 little-endian f32>, "sample_rate"}`, or a video as a list of frames (`[image, ...]`
//!   or `{"frames": [...]}`) with an optional `fps`. Audio files must be WAV. Remote URLs are not fetched.
//! * `prompt_name` (`SearchQuery`, `Document`, ...) or a literal `prompt` goes before the whole input, as in
//!   sentence-transformers. Without either, a `model` that names a prompt selects it, then the server's
//!   default prompt applies.
//! * `dimensions` (alias `dim`): Matryoshka truncation, re-normalised. `encoding_format`: `float` (default) or
//!   `base64` (little-endian f32, what the `openai` Python SDK asks for by default).
//! * `id` (stdio only): echoed in the response.
//!
//! The response is the OpenAI one (`{"object": "list", "data": [{"index", "embedding"}], "usage": ...}`); a
//! failed request gets `{"error": {"message", "type"}}` instead (stdio keeps going with the next line).
//!
//! Inputs are decoded and preprocessed on the thread that received the request (stdio: the reader thread),
//! so that CPU work overlaps the device work of earlier requests. One worker owns the device and merges
//! queued requests into shared batches.

use crate::media::{self, AudioClip};
use crate::model::{EmbeddingGemma2, Input, Part, PreparedInput, VideoClip, truncate_and_normalize};
use anyhow::{Context, Result, anyhow, bail, ensure};
use image::RgbImage;
use rayon::prelude::*;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

/// Settings shared by both transports.
#[derive(Clone, Debug)]
pub struct ServeOptions {
    /// Name reported as `model` in responses.
    pub model_name: String,
    /// Prompt name applied when a request names none.
    pub default_prompt: Option<String>,
    /// Whether inputs may name files on this machine.
    pub allow_file_paths: bool,
}

/// HTTP transport settings.
#[derive(Clone, Debug)]
pub struct HttpOptions {
    /// Address to listen on, e.g. `127.0.0.1:8080`.
    pub addr: String,
    /// Threads that receive, decode and preprocess requests.
    pub threads: usize,
    /// Largest accepted request body.
    pub max_body_bytes: usize,
    /// Origins whose browser (cross-origin) requests are accepted; `*` accepts any. Requests that carry an
    /// `Origin` header are refused otherwise, so web pages cannot use a local server behind the user's back.
    pub allow_origins: Vec<String>,
    /// When set, requests must carry `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
}

/// Most tokens and media inputs the worker merges from queued requests into one batch.
const MERGE_TOKENS: usize = 16384;
const MERGE_MEDIA: usize = 32;

const TOKEN_IDS: &str = "token-id inputs are not supported: send the text itself (LangChain's OpenAIEmbeddings \
                         tokenizes by default; pass check_embedding_ctx_length=False)";

// ---------------------------------------------------------------------------------------------
// Request format
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbedRequest {
    #[serde(default)]
    id: Option<Value>,
    input: Inputs,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt_name: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default, alias = "dim")]
    dimensions: Option<usize>,
    #[serde(default)]
    encoding_format: Option<String>,
    /// OpenAI's end-user tag; accepted and ignored.
    #[serde(default, rename = "user")]
    _user: Option<Value>,
}

/// Just the `id` of a request that failed to parse, so the error response can still carry it.
#[derive(Deserialize)]
struct IdOnly {
    #[serde(default)]
    id: Option<Value>,
}

/// The `input` field: one input or a list of them.
struct Inputs(Vec<InputSpec>);

/// One embedding input: its parts in order.
#[derive(Debug)]
struct InputSpec(Vec<PartSpec>);

/// A part as sent; media are decoded later, on the request's thread.
#[derive(Debug)]
enum PartSpec {
    Text(String),
    Image(Value),
    Audio(Value),
    Video { src: Value, fps: Option<f64> },
}

impl<'de> Deserialize<'de> for Inputs {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Inputs;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string, an input object or a list of inputs")
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<Inputs, E> {
                Ok(Inputs(vec![InputSpec(vec![PartSpec::Text(s.to_owned())])]))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Inputs, A::Error> {
                Ok(Inputs(vec![InputVisitor.visit_map(map)?]))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Inputs, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = seq.next_element::<InputSpec>()? {
                    v.push(x);
                }
                Ok(Inputs(v))
            }
        }
        d.deserialize_any(V)
    }
}

struct InputVisitor;

impl<'de> Visitor<'de> for InputVisitor {
    type Value = InputSpec;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a string or an input object")
    }
    fn visit_str<E: de::Error>(self, s: &str) -> Result<InputSpec, E> {
        Ok(InputSpec(vec![PartSpec::Text(s.to_owned())]))
    }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<InputSpec, E> {
        Err(E::custom(TOKEN_IDS))
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<InputSpec, E> {
        Err(E::custom(TOKEN_IDS))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, _: A) -> Result<InputSpec, A::Error> {
        Err(de::Error::custom(TOKEN_IDS))
    }
    /// Keys are read in document order: `{"text": .., "image": ..}` is text then image.
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<InputSpec, A::Error> {
        let mut parts = Vec::new();
        let mut fps = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "text" => parts.push(PartSpec::Text(map.next_value()?)),
                "image" | "image_url" => parts.push(PartSpec::Image(map.next_value()?)),
                "audio" => parts.push(PartSpec::Audio(map.next_value()?)),
                "video" => parts.push(PartSpec::Video { src: map.next_value()?, fps: None }),
                "parts" | "content" => {
                    let items: Vec<InputSpec> = map.next_value()?;
                    parts.extend(items.into_iter().flat_map(|i| i.0));
                }
                "fps" => fps = Some(map.next_value::<f64>()?),
                // OpenAI-style content parts name their one data key in `type`; the key itself is what counts.
                "type" => drop(map.next_value::<String>()?),
                k => return Err(de::Error::custom(format!("unknown input field `{k}` (expected text, image, audio, video, parts or fps)"))),
            }
        }
        if parts.is_empty() {
            return Err(de::Error::custom("an input object needs at least one of text, image, audio, video or parts"));
        }
        if let Some(f) = fps {
            for p in &mut parts {
                if let PartSpec::Video { fps: v @ None, .. } = p {
                    *v = Some(f);
                }
            }
        }
        Ok(InputSpec(parts))
    }
}

impl<'de> Deserialize<'de> for InputSpec {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(InputVisitor)
    }
}

// ---------------------------------------------------------------------------------------------
// Media values
// ---------------------------------------------------------------------------------------------

fn b64_decode(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    let r = if s.bytes().any(|b| b.is_ascii_whitespace()) {
        base64::decode(s.bytes().filter(|b| !b.is_ascii_whitespace()).collect::<Vec<u8>>())
    } else {
        base64::decode(s)
    };
    r.map_err(|e| anyhow!("invalid base64: {e}"))
}

/// Where a media value's bytes come from.
enum Source {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

/// A media string: a data URL or a file path.
fn source_of(s: &str, opts: &ServeOptions) -> Result<Source> {
    if let Some(rest) = s.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or_else(|| anyhow!("malformed data URL (no comma)"))?;
        ensure!(meta.ends_with(";base64"), "data URLs must be base64-encoded (`data:<type>;base64,...`)");
        return Ok(Source::Bytes(b64_decode(data)?));
    }
    ensure!(
        !(s.starts_with("http://") || s.starts_with("https://")),
        "remote URLs are not fetched: download the file and send its path or a data URL"
    );
    ensure!(opts.allow_file_paths, "file paths are disabled on this server: send the data inline (a data URL or base64)");
    Ok(Source::Path(PathBuf::from(s)))
}

/// The source named by one of an object's `path` / `url` / `base64` keys.
fn object_source(o: &Map<String, Value>, opts: &ServeOptions) -> Result<Source> {
    let get = |k: &str| o.get(k).map(|v| v.as_str().ok_or_else(|| anyhow!("`{k}` must be a string"))).transpose();
    match (get("path")?, get("url")?, get("base64")?) {
        (Some(p), None, None) => source_of(p, opts).and_then(|s| match s {
            Source::Path(p) => Ok(Source::Path(p)),
            Source::Bytes(_) => bail!("`path` holds a data URL; use `url` for those"),
        }),
        (None, Some(u), None) => source_of(u, opts),
        (None, None, Some(b)) => Ok(Source::Bytes(b64_decode(b)?)),
        _ => bail!("give exactly one of `path`, `url` or `base64`"),
    }
}

fn check_keys(o: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<()> {
    match o.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => bail!("unknown {what} field `{k}` (expected one of {allowed:?})"),
        None => Ok(()),
    }
}

fn dim_field(o: &Map<String, Value>, k: &str) -> Result<u32> {
    let v = o.get(k).and_then(Value::as_u64).ok_or_else(|| anyhow!("`{k}` must be a positive integer"))?;
    ensure!(v > 0 && v <= u32::MAX as u64, "`{k}` must be a positive integer");
    Ok(v as u32)
}

fn load_image_value(v: &Value, opts: &ServeOptions) -> Result<RgbImage> {
    let src = match v {
        Value::String(s) => source_of(s, opts)?,
        Value::Object(o) if o.contains_key("rgb") => {
            check_keys(o, &["rgb", "width", "height"], "image")?;
            let (w, h) = (dim_field(o, "width")?, dim_field(o, "height")?);
            let data = b64_decode(o["rgb"].as_str().ok_or_else(|| anyhow!("`rgb` must be a base64 string"))?)?;
            let want = w as usize * h as usize * 3;
            ensure!(data.len() == want, "`rgb` holds {} bytes; {w}x{h} RGB8 pixels need {want}", data.len());
            return Ok(RgbImage::from_raw(w, h, data).expect("size checked"));
        }
        Value::Object(o) => {
            check_keys(o, &["path", "url", "base64", "detail"], "image")?;
            object_source(o, opts)?
        }
        _ => bail!("an image must be a path, a data URL or an object"),
    };
    match src {
        Source::Path(p) => media::load_image(&p),
        Source::Bytes(b) => media::decode_image(&b),
    }
}

fn load_audio_value(v: &Value, opts: &ServeOptions) -> Result<AudioClip> {
    let src = match v {
        Value::String(s) => source_of(s, opts)?,
        Value::Object(o) if o.contains_key("pcm_f32") || o.contains_key("samples") => {
            check_keys(o, &["pcm_f32", "samples", "sample_rate"], "audio")?;
            let rate = dim_field(o, "sample_rate")?;
            let samples: Vec<f32> = match (o.get("pcm_f32"), o.get("samples")) {
                (Some(Value::String(b)), None) => {
                    let bytes = b64_decode(b)?;
                    ensure!(bytes.len() % 4 == 0, "`pcm_f32` must hold whole little-endian f32 samples");
                    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
                }
                (None, Some(Value::Array(a))) => {
                    a.iter().map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| anyhow!("`samples` must be numbers"))).collect::<Result<_>>()?
                }
                _ => bail!("give either `pcm_f32` (a base64 string) or `samples` (a list of numbers)"),
            };
            return Ok(AudioClip::new(samples, rate));
        }
        Value::Object(o) => {
            check_keys(o, &["path", "url", "base64"], "audio")?;
            object_source(o, opts)?
        }
        _ => bail!("audio must be a path, a data URL or an object"),
    };
    match src {
        Source::Path(p) => media::load_audio(&p),
        Source::Bytes(b) => media::decode_audio(&b),
    }
}

fn load_video_value(model: &EmbeddingGemma2, v: &Value, fps: Option<f64>, opts: &ServeOptions) -> Result<VideoClip> {
    if let Some(f) = fps {
        ensure!(f.is_finite() && f > 0.0, "`fps` must be a positive number (got {f})");
    }
    let frames = |list: &[Value]| -> Result<VideoClip> {
        ensure!(!list.is_empty(), "a video needs at least one frame");
        let frames = list.par_iter().map(|f| load_image_value(f, opts)).collect::<Result<Vec<_>>>()?;
        Ok(VideoClip { frames, fps })
    };
    match v {
        Value::String(s) => match source_of(s, opts)? {
            Source::Path(p) => media::load_video(&p, fps, &model.video_sample_params()),
            Source::Bytes(_) => bail!("videos cannot be sent as data URLs: send a file path or a list of frames"),
        },
        Value::Array(list) => frames(list),
        Value::Object(o) => {
            check_keys(o, &["path", "frames", "fps"], "video")?;
            let fps = match o.get("fps") {
                Some(f) => Some(f.as_f64().ok_or_else(|| anyhow!("`fps` must be a number"))?),
                None => fps,
            };
            match (o.get("path"), o.get("frames")) {
                (Some(Value::String(p)), None) => load_video_value(model, &Value::String(p.clone()), fps, opts),
                (None, Some(Value::Array(list))) => {
                    let mut clip = frames(list)?;
                    clip.fps = fps;
                    Ok(clip)
                }
                _ => bail!("a video object needs `path` (a string) or `frames` (a list)"),
            }
        }
        _ => bail!("a video must be a path, a list of frames or an object"),
    }
}

fn load_part(model: &EmbeddingGemma2, p: &PartSpec, opts: &ServeOptions) -> Result<Part> {
    Ok(match p {
        PartSpec::Text(t) => Part::Text(t.clone()),
        PartSpec::Image(v) => Part::Image(load_image_value(v, opts).context("image")?),
        PartSpec::Audio(v) => Part::Audio(load_audio_value(v, opts).context("audio")?),
        PartSpec::Video { src, fps } => Part::Video(load_video_value(model, src, *fps, opts).context("video")?),
    })
}

// ---------------------------------------------------------------------------------------------
// Preparing requests, the device worker, responses
// ---------------------------------------------------------------------------------------------

/// What the response needs besides the embeddings.
struct Meta {
    id: Option<Value>,
    dim: Option<usize>,
    base64: bool,
    tokens: usize,
}

/// Most inputs per job, and most media per job (an image or audio clip counts 1, a video half its frames):
/// a large request goes to the device in several jobs, which bounds the preprocessed data (about 8 MB per
/// image) it holds while it waits.
const JOB_INPUTS: usize = 256;
const JOB_MEDIA: usize = 16;

fn media_weight(spec: &InputSpec) -> usize {
    spec.0
        .iter()
        .map(|p| match p {
            PartSpec::Text(_) => 0,
            PartSpec::Image(_) | PartSpec::Audio(_) => 1,
            PartSpec::Video { src: Value::Array(frames), .. } => frames.len().min(32).div_ceil(2),
            PartSpec::Video { .. } => 16, // a file or directory: up to 32 sampled frames
        })
        .sum()
}

/// Consecutive runs of inputs that form one job each.
fn job_ranges(specs: &[InputSpec]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < specs.len() {
        let (mut end, mut media) = (start, 0);
        while end < specs.len() && end - start < JOB_INPUTS {
            let m = media_weight(&specs[end]);
            if end > start && media + m > JOB_MEDIA {
                break;
            }
            media += m;
            end += 1;
        }
        out.push(start..end);
        start = end;
    }
    out
}

/// A request that could not be queued: its id (if any), whether the server is at fault, and why.
struct Rejected {
    id: Option<Value>,
    server: bool,
    err: anyhow::Error,
}

impl Rejected {
    fn status(&self) -> u16 {
        if self.server { 500 } else { 400 }
    }
    fn json(&self) -> String {
        error_json(self.id.clone(), if self.server { "server_error" } else { "invalid_request_error" }, &self.err)
    }
}

/// A request whose inputs are queued for the device, one job per range of inputs.
struct Submitted {
    meta: Meta,
    replies: Vec<mpsc::Receiver<Reply>>,
}

impl Submitted {
    /// Wait for every job and build the response line (`Err`: an error line).
    fn finish(self, opts: &ServeOptions) -> Result<String, String> {
        let id = self.meta.id.clone();
        let fail = |e: anyhow::Error| error_json(id.clone(), "server_error", &e);
        let mut embs = Vec::new();
        for rx in self.replies {
            embs.extend(rx.recv().map_err(|_| fail(anyhow!("the model worker has stopped")))?.map_err(fail)?);
        }
        response_json(self.meta, embs, opts).map_err(fail)
    }
}

/// Parse a request body, then decode and prepare its inputs (in parallel, a job's worth at a time) and queue
/// them for the device worker. The bounded job queue makes a large request wait here for the device.
fn submit_request(model: &EmbeddingGemma2, opts: &ServeOptions, body: &str, jobs: &mpsc::SyncSender<Job>) -> Result<Submitted, Rejected> {
    let req: EmbedRequest = serde_json::from_str(body.strip_prefix('\u{feff}').unwrap_or(body)).map_err(|e| Rejected {
        id: serde_json::from_str::<IdOnly>(body).ok().and_then(|r| r.id),
        server: false,
        err: anyhow!("invalid request: {e}"),
    })?;
    let id = req.id.clone();
    let reject = |err| Rejected { id: id.clone(), server: false, err };
    let (prompt, base64) = check_request(model, opts, &req).map_err(reject)?;
    let prepare_one = |(i, spec): (usize, &InputSpec)| -> Result<PreparedInput> {
        let parts = spec.0.iter().map(|p| load_part(model, p, opts)).collect::<Result<Vec<_>>>();
        let input = Input { parts: parts.with_context(|| format!("input {i}"))? }.with_prompt(&prompt);
        model.prepare_input(&input).with_context(|| format!("input {i}"))
    };
    let (mut replies, mut tokens) = (Vec::new(), 0);
    for range in job_ranges(&req.input.0) {
        let specs = &req.input.0[range.clone()];
        let items: Vec<PreparedInput> = if specs.len() == 1 {
            vec![prepare_one((range.start, &specs[0])).map_err(reject)?]
        } else {
            specs.par_iter().enumerate().map(|(j, s)| prepare_one((range.start + j, s))).collect::<Result<_>>().map_err(reject)?
        };
        tokens += items.iter().map(PreparedInput::num_tokens).sum::<usize>();
        let (tx, rx) = mpsc::channel();
        jobs.send(Job { items, reply: tx }).map_err(|_| Rejected { id: id.clone(), server: true, err: anyhow!("the model worker has stopped") })?;
        replies.push(rx);
    }
    Ok(Submitted { meta: Meta { id: req.id, dim: req.dimensions, base64, tokens }, replies })
}

/// Validate the request's options; returns the prompt to prepend and whether to answer in base64.
fn check_request(model: &EmbeddingGemma2, opts: &ServeOptions, req: &EmbedRequest) -> Result<(String, bool)> {
    ensure!(!req.input.0.is_empty(), "`input` is empty");
    let name = match (&req.prompt, &req.prompt_name) {
        (Some(_), Some(_)) => bail!("give either `prompt` or `prompt_name`, not both"),
        (Some(_), None) => None,
        (None, Some(n)) => Some(n.as_str()),
        (None, None) => req.model.as_deref().filter(|m| model.prompts.contains_key(*m)).or(opts.default_prompt.as_deref()),
    };
    let prompt = match (name, &req.prompt) {
        (Some(n), _) => model.prompt(n)?.to_string(),
        (None, Some(p)) => p.clone(),
        (None, None) => String::new(),
    };
    let base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(f) => bail!("`encoding_format` must be float or base64 (got `{f}`)"),
    };
    let full = model.cfg.text_config.embedding_dim;
    if let Some(d) = req.dimensions {
        ensure!((1..=full).contains(&d), "`dimensions` must be between 1 and {full} (got {d}); the model card's Matryoshka sizes are 768, 512, 256 and 128");
    }
    Ok((prompt, base64))
}

type Reply = Result<Vec<Vec<f32>>>;

/// Prepared inputs waiting for the device, and where their embeddings go.
struct Job {
    items: Vec<PreparedInput>,
    reply: mpsc::Sender<Reply>,
}

impl Job {
    fn size(&self) -> (usize, usize) {
        (self.items.iter().map(PreparedInput::num_tokens).sum(), self.items.iter().filter(|p| p.has_media()).count())
    }
}

fn embed_guarded(model: &EmbeddingGemma2, items: &[PreparedInput]) -> Reply {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.embed_prepared(items)))
        .unwrap_or_else(|p| {
            let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
            Err(anyhow!("internal error: {}", msg.unwrap_or_else(|| "panic".into())))
        })
}

/// The one thread that runs the model: takes jobs in order and merges whatever else is queued (up to
/// `MERGE_TOKENS` / `MERGE_MEDIA`) into the same `embed_prepared` call.
fn worker(model: &EmbeddingGemma2, jobs: mpsc::Receiver<Job>) {
    let mut held: Option<Job> = None;
    loop {
        let first = match held.take().map_or_else(|| jobs.recv().ok(), Some) {
            Some(j) => j,
            None => return,
        };
        let (mut tokens, mut media) = first.size();
        let mut batch = vec![first];
        while let Ok(j) = jobs.try_recv() {
            let (t, m) = j.size();
            if tokens + t > MERGE_TOKENS || media + m > MERGE_MEDIA {
                held = Some(j);
                break;
            }
            (tokens, media) = (tokens + t, media + m);
            batch.push(j);
        }
        run_batch(model, batch);
    }
}

fn run_batch(model: &EmbeddingGemma2, batch: Vec<Job>) {
    let mut sizes = Vec::with_capacity(batch.len());
    let mut replies = Vec::with_capacity(batch.len());
    let mut items = Vec::new();
    for j in batch {
        sizes.push(j.items.len());
        replies.push(j.reply);
        items.extend(j.items);
    }
    let merged = replies.len() > 1;
    match embed_guarded(model, &items) {
        Ok(mut embs) => {
            for (n, reply) in sizes.into_iter().zip(replies) {
                let rest = embs.split_off(n);
                let _ = reply.send(Ok(embs));
                embs = rest;
            }
        }
        // One bad request must not fail the others it was merged with: run them one by one.
        Err(_) if merged => {
            for (n, reply) in sizes.into_iter().zip(replies) {
                let rest = items.split_off(n);
                let _ = reply.send(embed_guarded(model, &items));
                items = rest;
            }
        }
        Err(e) => {
            let _ = replies[0].send(Err(e));
        }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
struct EmbeddingItem {
    object: &'static str,
    index: usize,
    embedding: EmbeddingValue,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct EmbedResponse<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    object: &'static str,
    data: Vec<EmbeddingItem>,
    model: &'a str,
    usage: Usage,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct ErrorResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    error: ErrorDetail,
}

fn error_json(id: Option<Value>, kind: &'static str, e: &anyhow::Error) -> String {
    serde_json::to_string(&ErrorResponse { id, error: ErrorDetail { message: format!("{e:#}"), kind } }).expect("serialisable")
}

fn response_json(meta: Meta, embs: Vec<Vec<f32>>, opts: &ServeOptions) -> Result<String> {
    let data = embs
        .into_iter()
        .enumerate()
        .map(|(index, e)| {
            let e = truncate_and_normalize(e, meta.dim)?;
            let embedding = if meta.base64 {
                EmbeddingValue::Base64(base64::encode(e.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()))
            } else {
                EmbeddingValue::Float(e)
            };
            Ok(EmbeddingItem { object: "embedding", index, embedding })
        })
        .collect::<Result<Vec<_>>>()?;
    let usage = Usage { prompt_tokens: meta.tokens, total_tokens: meta.tokens };
    Ok(serde_json::to_string(&EmbedResponse { id: meta.id, object: "list", data, model: &opts.model_name, usage })?)
}

// ---------------------------------------------------------------------------------------------
// stdio transport
// ---------------------------------------------------------------------------------------------

/// A response line in the making, in request order.
enum Pending {
    Ready(String),
    Waiting(Submitted),
}

/// Serve JSON Lines on stdin / stdout until stdin closes. Blank lines are skipped.
///
/// A reader thread (this one) parses and prepares each request and queues it for the device worker; a writer
/// thread emits the responses in request order, flushing after each line. So the next request is decoded and
/// prepared while the device works on the current one.
pub fn run_stdio(model: &EmbeddingGemma2, opts: &ServeOptions) -> Result<()> {
    let (job_tx, job_rx) = mpsc::sync_channel::<Job>(4);
    let (out_tx, out_rx) = mpsc::sync_channel::<Pending>(16);
    std::thread::scope(|s| {
        s.spawn(|| worker(model, job_rx));
        let writer = s.spawn(move || -> Result<()> {
            let mut out = std::io::BufWriter::new(std::io::stdout().lock());
            for p in out_rx {
                let line = match p {
                    Pending::Ready(l) => l,
                    Pending::Waiting(sub) => sub.finish(opts).unwrap_or_else(|e| e),
                };
                out.write_all(line.as_bytes())?;
                out.write_all(b"\n")?;
                out.flush()?;
            }
            Ok(())
        });
        let reader = (|| -> Result<()> {
            for line in std::io::stdin().lock().lines() {
                let line = line.context("reading stdin")?;
                if line.trim().is_empty() {
                    continue;
                }
                let pending = match submit_request(model, opts, &line, &job_tx) {
                    Ok(sub) => Pending::Waiting(sub),
                    Err(r) if r.server => bail!("{:#}", r.err),
                    Err(r) => Pending::Ready(r.json()),
                };
                if out_tx.send(pending).is_err() {
                    break; // the writer failed (stdout closed); its error is reported below
                }
            }
            Ok(())
        })();
        drop(out_tx);
        drop(job_tx);
        let written = writer.join().unwrap_or_else(|_| Err(anyhow!("writer thread panicked")));
        reader.and(written)
    })
}

// ---------------------------------------------------------------------------------------------
// HTTP transport
// ---------------------------------------------------------------------------------------------

fn header<'r>(rq: &'r tiny_http::Request, name: &'static str) -> Option<&'r str> {
    rq.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str())
}

fn http_header(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).expect("valid header")
}

/// Constant-time comparison, so response timing does not reveal how much of a key matched.
fn same_key(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Answer HTTP requests on `http.addr` until the process ends.
pub fn run_http(model: &EmbeddingGemma2, opts: &ServeOptions, http: &HttpOptions) -> Result<()> {
    let server = tiny_http::Server::http(&http.addr).map_err(|e| anyhow!("cannot listen on {}: {e}", http.addr))?;
    let addr = server.server_addr().to_ip().map_or_else(|| http.addr.clone(), |a| a.to_string());
    eprintln!("listening on http://{addr} (POST /v1/embeddings, GET /health, GET /v1/models)");
    let (job_tx, job_rx) = mpsc::sync_channel::<Job>(64);
    std::thread::scope(|s| {
        s.spawn(|| worker(model, job_rx));
        for _ in 0..http.threads.max(1) {
            let (tx, server) = (job_tx.clone(), &server);
            s.spawn(move || {
                while let Ok(rq) = server.recv() {
                    handle_http(model, opts, http, &tx, rq);
                }
            });
        }
        drop(job_tx);
    });
    Ok(())
}

fn handle_http(model: &EmbeddingGemma2, opts: &ServeOptions, http: &HttpOptions, jobs: &mpsc::SyncSender<Job>, mut rq: tiny_http::Request) {
    let t0 = Instant::now();
    let method = rq.method().clone();
    let path = rq.url().split('?').next().unwrap_or("").to_string();
    let origin = header(&rq, "Origin").map(str::to_string);
    let cors = origin.as_deref().filter(|o| http.allow_origins.iter().any(|a| a == "*" || a == o)).map(str::to_string);
    let invalid = |m: &str| (400u16, error_json(None, "invalid_request_error", &anyhow!("{m}")));

    let (status, body) = (|| -> Result<(u16, String), (u16, String)> {
        use tiny_http::Method;
        if origin.is_some() && cors.is_none() {
            return Err((403, error_json(None, "invalid_request_error", &anyhow!("cross-origin requests are refused (start the server with --allow-origin)"))));
        }
        if method == Method::Options {
            return Ok((204, String::new()));
        }
        // Health checks need no key; everything else does when one is set.
        if let Some(key) = http.api_key.as_ref().filter(|_| path != "/health") {
            let given = header(&rq, "Authorization").and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
            if !same_key(given.trim(), key) {
                return Err((401, error_json(None, "authentication_error", &anyhow!("missing or wrong API key (Authorization: Bearer <key>)"))));
            }
        }
        match (&method, path.as_str()) {
            (Method::Get, "/health") => Ok((200, r#"{"status":"ok"}"#.to_string())),
            (Method::Get, "/v1/models" | "/models") => Ok((
                200,
                serde_json::json!({"object": "list", "data": [{"id": opts.model_name, "object": "model", "owned_by": "egemma2"}]}).to_string(),
            )),
            (Method::Post, "/v1/embeddings" | "/embeddings") => {
                if rq.body_length().is_some_and(|n| n > http.max_body_bytes) {
                    return Err((413, error_json(None, "invalid_request_error", &anyhow!("request body over {} MB (see --max-request-mb)", http.max_body_bytes >> 20))));
                }
                let mut buf = Vec::new();
                rq.as_reader().take(http.max_body_bytes as u64 + 1).read_to_end(&mut buf).map_err(|e| invalid(&format!("reading the body: {e}")))?;
                if buf.len() > http.max_body_bytes {
                    return Err((413, error_json(None, "invalid_request_error", &anyhow!("request body over {} MB (see --max-request-mb)", http.max_body_bytes >> 20))));
                }
                let text = std::str::from_utf8(&buf).map_err(|_| invalid("the body is not UTF-8"))?;
                let sub = submit_request(model, opts, text, jobs).map_err(|r| (r.status(), r.json()))?;
                sub.finish(opts).map(|b| (200, b)).map_err(|e| (500, e))
            }
            (_, "/v1/embeddings" | "/embeddings" | "/health" | "/v1/models" | "/models") => Err((405, error_json(None, "invalid_request_error", &anyhow!("method not allowed")))),
            _ => Err((404, error_json(None, "invalid_request_error", &anyhow!("no route for {method} {path}")))),
        }
    })()
    .unwrap_or_else(|e| e);

    if status >= 400 {
        eprintln!("{method} {path} -> {status} in {:.0} ms: {body}", t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut resp = tiny_http::Response::from_string(body).with_status_code(status).with_header(http_header("Content-Type", "application/json"));
    if let Some(o) = cors {
        resp.add_header(http_header("Access-Control-Allow-Origin", &o));
        resp.add_header(http_header("Vary", "Origin"));
        resp.add_header(http_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"));
        resp.add_header(http_header("Access-Control-Allow-Headers", "Content-Type, Authorization"));
    }
    let _ = rq.respond(resp);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<EmbedRequest, serde_json::Error> {
        serde_json::from_str(s)
    }

    fn kinds(spec: &InputSpec) -> Vec<&'static str> {
        spec.0
            .iter()
            .map(|p| match p {
                PartSpec::Text(_) => "text",
                PartSpec::Image(_) => "image",
                PartSpec::Audio(_) => "audio",
                PartSpec::Video { .. } => "video",
            })
            .collect()
    }

    fn opts(paths: bool) -> ServeOptions {
        ServeOptions { model_name: "m".into(), default_prompt: None, allow_file_paths: paths }
    }

    #[test]
    fn input_shapes() {
        assert_eq!(parse(r#"{"input": "hi"}"#).unwrap().input.0.len(), 1);
        let r = parse(r#"{"input": ["a", {"image": "x.png"}, {"text": "t", "image": "y.png"}, {"image": "y.png", "text": "t"}]}"#).unwrap();
        let k: Vec<_> = r.input.0.iter().map(kinds).collect();
        assert_eq!(k, vec![vec!["text"], vec!["image"], vec!["text", "image"], vec!["image", "text"]]);
        let r = parse(r#"{"input": {"parts": ["A:", {"image": "a.png"}, {"image": "b.png"}]}}"#).unwrap();
        assert_eq!(kinds(&r.input.0[0]), vec!["text", "image", "image"]);
        let r = parse(r#"{"input": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}"#).unwrap();
        assert_eq!(kinds(&r.input.0[0]), vec!["image"]);
        let r = parse(r#"{"input": {"video": "clip.mp4", "fps": 24}}"#).unwrap();
        assert!(matches!(r.input.0[0].0[0], PartSpec::Video { fps: Some(24.0), .. }));
        let r = parse(r#"{"input": "x", "dim": 128, "model": "m", "user": "u", "id": [1]}"#).unwrap();
        assert_eq!(r.dimensions, Some(128));
    }

    #[test]
    fn rejected_requests() {
        for (body, needle) in [
            (r#"{"input": [[1, 2, 3]]}"#, "token-id"),
            (r#"{"input": [1, 2, 3]}"#, "token-id"),
            (r#"{"input": {"imgae": "x.png"}}"#, "unknown input field `imgae`"),
            (r#"{"input": "x", "prompt_nme": "Document"}"#, "prompt_nme"),
            (r#"{"input": {}}"#, "at least one"),
            (r#"{"input": {"text": 3}}"#, "invalid type"),
        ] {
            let e = parse(body).err().unwrap_or_else(|| panic!("{body} parsed")).to_string();
            assert!(e.contains(needle), "{body}: {e}");
        }
    }

    #[test]
    fn media_values() {
        // A 1x1 red PNG.
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
        let img = load_image_value(&Value::String(format!("data:image/png;base64,{png}")), &opts(false)).unwrap();
        assert_eq!((img.width(), img.height(), img.get_pixel(0, 0).0), (1, 1, [255, 0, 0]));
        let raw = serde_json::json!({"rgb": base64::encode([1u8, 2, 3, 4, 5, 6]), "width": 2, "height": 1});
        assert_eq!(load_image_value(&raw, &opts(false)).unwrap().into_raw(), vec![1, 2, 3, 4, 5, 6]);
        let short = serde_json::json!({"rgb": base64::encode([1u8, 2]), "width": 2, "height": 1});
        assert!(load_image_value(&short, &opts(false)).is_err());
        let e = load_image_value(&Value::String("page.png".into()), &opts(false)).unwrap_err().to_string();
        assert!(e.contains("file paths are disabled"), "{e}");
        let e = load_image_value(&Value::String("https://x/y.png".into()), &opts(true)).unwrap_err().to_string();
        assert!(e.contains("not fetched"), "{e}");
        let pcm: Vec<u8> = [0.5f32, -0.25].iter().flat_map(|x| x.to_le_bytes()).collect();
        let clip = load_audio_value(&serde_json::json!({"pcm_f32": base64::encode(pcm), "sample_rate": 16000}), &opts(false)).unwrap();
        assert_eq!((clip.samples, clip.sample_rate), (vec![0.5, -0.25], 16000));
    }

    #[test]
    fn response_encodings() {
        let meta = |base64| Meta { id: Some(Value::from(7)), dim: Some(2), base64, tokens: 5 };
        let v: Value = serde_json::from_str(&response_json(meta(false), vec![vec![3.0, 4.0, 12.0]], &opts(true)).unwrap()).unwrap();
        assert_eq!(v["id"], 7);
        assert_eq!(v["data"][0]["embedding"], serde_json::json!([0.6, 0.8]));
        assert_eq!(v["usage"]["total_tokens"], 5);
        let v: Value = serde_json::from_str(&response_json(meta(true), vec![vec![3.0, 4.0, 12.0]], &opts(true)).unwrap()).unwrap();
        let bytes = base64::decode(v["data"][0]["embedding"].as_str().unwrap()).unwrap();
        let back: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(back, vec![0.6, 0.8]);
    }

    #[test]
    fn large_requests_split_into_jobs() {
        let specs = |body: &str| parse(body).unwrap().input.0;
        let ranges = |body: &str| job_ranges(&specs(body));
        let images = format!(r#"{{"input": [{}]}}"#, vec![r#"{"image": "p.png"}"#; 40].join(","));
        assert_eq!(ranges(&images), vec![0..16, 16..32, 32..40]);
        let texts = format!(r#"{{"input": [{}]}}"#, vec![r#""t""#; 300].join(","));
        assert_eq!(ranges(&texts), vec![0..256, 256..300]);
        // A video file counts as 16 images; one over-heavy input still forms a job of its own.
        assert_eq!(ranges(r#"{"input": [{"video": "a.mp4"}, {"image": "p.png"}, {"video": "b.mp4"}]}"#), vec![0..1, 1..2, 2..3]);
        assert_eq!(ranges(r#"{"input": [{"video": ["f.png", "f.png"]}, {"image": "p.png"}]}"#), vec![0..2]);
    }

    #[test]
    fn keys_compare() {
        assert!(same_key("abc", "abc"));
        assert!(!same_key("abc", "abd"));
        assert!(!same_key("abc", "ab"));
    }
}
