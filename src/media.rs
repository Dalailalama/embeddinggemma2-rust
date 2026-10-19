//! Loading images, audio and video from disk.
//!
//! * Images: anything the `image` crate decodes, converted to RGB8 (`do_convert_rgb`).
//! * Audio: WAV (PCM / float) via `hound`, mixed down to mono; resampled to 16 kHz on demand.
//! * Video: a directory of frame images, an animated GIF, or any container via an `ffmpeg` binary.
//!   Every loader returns only the frames the model will use (already sampled, `fps: None`).

use anyhow::{Context, Result, anyhow, bail, ensure};
use image::{AnimationDecoder, RgbImage};
use serde::Deserialize;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use crate::model::VideoClip;
use crate::videoproc::{self, SampleParams};

/// Mono audio samples in `[-1, 1]`.
#[derive(Clone)]
pub struct AudioClip {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioClip {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self { samples, sample_rate }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 { 0.0 } else { self.samples.len() as f64 / self.sample_rate as f64 }
    }

    /// Samples at `target_rate` (resampled with a windowed-sinc filter when the rates differ).
    pub fn at_rate(&self, target_rate: u32) -> Vec<f32> {
        if self.sample_rate == target_rate {
            self.samples.clone()
        } else {
            resample(&self.samples, self.sample_rate, target_rate)
        }
    }

    /// The first `max_samples` samples at `target_rate`.
    ///
    /// Only the source span that can influence those samples (plus the filter's half-width) is
    /// resampled, so a long file costs no more than the part that is kept; the result is identical
    /// to resampling everything and truncating afterwards.
    pub fn at_rate_truncated(&self, target_rate: u32, max_samples: usize) -> Result<Vec<f32>> {
        ensure!(self.sample_rate > 0 && target_rate > 0, "audio sample rate must be positive (got {} Hz)", self.sample_rate);
        if self.sample_rate == target_rate {
            return Ok(self.samples[..self.samples.len().min(max_samples)].to_vec());
        }
        let ratio = self.sample_rate as f64 / target_rate as f64;
        let need = (max_samples as f64 * ratio + kernel_half_width(self.sample_rate, target_rate)).ceil() as usize + 2;
        let mut out = resample(&self.samples[..self.samples.len().min(need)], self.sample_rate, target_rate);
        out.truncate(max_samples);
        Ok(out)
    }
}

pub fn load_image(path: &Path) -> Result<RgbImage> {
    Ok(image::open(path).with_context(|| format!("decoding image {}", path.display()))?.to_rgb8())
}

/// Decode an encoded image (png/jpg/webp/gif/bmp/tiff, detected from the bytes) to RGB.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    Ok(image::load_from_memory(bytes).context("decoding image bytes")?.to_rgb8())
}

pub fn load_audio(path: &Path) -> Result<AudioClip> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    ensure!(
        ext == "wav" || ext == "wave",
        "only .wav audio is supported natively (got `.{ext}`); convert with ffmpeg: ffmpeg -i in.{ext} -ar 16000 -ac 1 out.wav"
    );
    let reader = hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    read_wav(reader, &path.display().to_string())
}

/// Decode the bytes of a WAV file (PCM / float), mixed down to mono.
pub fn decode_audio(bytes: &[u8]) -> Result<AudioClip> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .context("decoding audio bytes (only WAV is supported; convert with: ffmpeg -i in -ar 16000 -ac 1 out.wav)")?;
    read_wav(reader, "audio bytes")
}

fn read_wav<R: std::io::Read>(mut reader: hound::WavReader<R>, what: &str) -> Result<AudioClip> {
    let spec = reader.spec();
    ensure!(spec.sample_rate > 0, "{what}: the WAV header declares a sample rate of 0 Hz");
    ensure!(spec.channels > 0, "{what}: the WAV header declares 0 channels");
    let channels = spec.channels as usize;
    let read = || format!("reading samples from {what}");
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<_, _>>().with_context(read)?,
        (hound::SampleFormat::Int, bits @ (8 | 16 | 24 | 32)) => {
            let scale = 1.0 / (1u64 << (bits - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>().with_context(read)?
        }
        (fmt, bits) => bail!("{what}: unsupported WAV format {fmt:?} / {bits} bits"),
    };
    let samples = if channels == 1 {
        interleaved
    } else {
        interleaved.chunks_exact(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
    };
    Ok(AudioClip { samples, sample_rate: spec.sample_rate })
}

/// Half-width, in input samples, of the resampling kernel (16 zero crossings of the low-pass).
fn kernel_half_width(from: u32, to: u32) -> f64 {
    16.0 / (to as f64 / from as f64).min(1.0)
}

/// Band-limited resampling with a Hann-windowed sinc kernel (16 zero crossings).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    use std::f64::consts::PI;
    if from == 0 || to == 0 {
        return Vec::new();
    }
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64; // input samples per output sample
    let cutoff = (to as f64 / from as f64).min(1.0);
    let half = kernel_half_width(from, to);
    let n_out = ((x.len() as f64) / ratio).floor() as usize;
    let sinc = |v: f64| if v.abs() < 1e-12 { 1.0 } else { (PI * v).sin() / (PI * v) };
    (0..n_out)
        .map(|m| {
            let t = m as f64 * ratio;
            let (lo, hi) = ((t - half).ceil().max(0.0) as usize, ((t + half).floor() as usize).min(x.len() - 1));
            let (mut acc, mut norm) = (0f64, 0f64);
            for (j, xv) in x.iter().enumerate().take(hi + 1).skip(lo) {
                let d = t - j as f64;
                let k = cutoff * sinc(cutoff * d) * (0.5 + 0.5 * (PI * d / half).cos());
                acc += k * *xv as f64;
                norm += k;
            }
            if norm.abs() > 1e-12 { (acc / norm) as f32 } else { 0.0 }
        })
        .collect()
}

fn is_image_file(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "bmp" | "webp" | "gif" | "tif" | "tiff")
    )
}

/// Order names so that digit runs compare by value: `frame2.png` < `frame10.png`.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    fn trim_zeros(s: &[u8]) -> &[u8] {
        &s[s.iter().take_while(|c| **c == b'0').count()..]
    }
    let (mut x, mut y) = (a.as_bytes(), b.as_bytes());
    loop {
        match (x.first(), y.first()) {
            (None, None) => return Ordering::Equal,
            (None, _) => return Ordering::Less,
            (_, None) => return Ordering::Greater,
            (Some(c), Some(d)) if c.is_ascii_digit() && d.is_ascii_digit() => {
                let lx = x.iter().take_while(|c| c.is_ascii_digit()).count();
                let ly = y.iter().take_while(|c| c.is_ascii_digit()).count();
                let (nx, ny) = (trim_zeros(&x[..lx]), trim_zeros(&y[..ly]));
                let ord = nx.len().cmp(&ny.len()).then_with(|| nx.cmp(ny)).then_with(|| lx.cmp(&ly));
                if ord != Ordering::Equal {
                    return ord;
                }
                (x, y) = (&x[lx..], &y[ly..]);
            }
            (Some(c), Some(d)) => {
                if c != d {
                    return c.cmp(d);
                }
                (x, y) = (&x[1..], &y[1..]);
            }
        }
    }
}

/// Load a video and return the frames the model will embed (already sampled, so `fps` is `None`).
///
/// Sources: a directory of frame images (natural file-name order, so `2.png` precedes `10.png`), an
/// animated GIF, or -- when `ffmpeg`/`ffprobe` are on `PATH` -- any container. `fps` gives / overrides
/// the native frame rate, which enables the processor's 1 fps sampling.
pub fn load_video(path: &Path, fps: Option<f64>, sample: &SampleParams) -> Result<VideoClip> {
    if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .with_context(|| format!("listing {}", path.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_image_file(p))
            .collect();
        let name = |p: &PathBuf| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        files.sort_by(|a, b| natural_cmp(&name(a), &name(b)));
        ensure!(!files.is_empty(), "no image frames found in {}", path.display());
        // Choose the frames first so that only those are decoded.
        let idx = videoproc::sample_frames(files.len(), fps, sample)?;
        let frames = idx.iter().map(|&i| load_image(&files[i])).collect::<Result<Vec<_>>>()?;
        return Ok(VideoClip { frames, fps: None });
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "gif" {
        let file = std::io::BufReader::new(std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?);
        let decoder = image::codecs::gif::GifDecoder::new(file)?;
        let mut frames = Vec::new();
        let mut total_ms = 0f64;
        for f in decoder.into_frames() {
            let f = f?;
            let (num, den) = f.delay().numer_denom_ms();
            total_ms += num as f64 / den.max(1) as f64;
            frames.push(image::DynamicImage::ImageRgba8(f.into_buffer()).to_rgb8());
        }
        ensure!(!frames.is_empty(), "GIF has no frames");
        let native_fps = fps.or(if total_ms > 0.0 { Some(frames.len() as f64 / (total_ms / 1000.0)) } else { None });
        let idx = videoproc::sample_frames(frames.len(), native_fps, sample)?;
        return Ok(VideoClip { frames: idx.iter().map(|&i| frames[i].clone()).collect(), fps: None });
    }
    load_video_ffmpeg(path, fps, sample)
}

#[derive(Deserialize)]
struct Probe {
    #[serde(default)]
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    index: usize,
    width: Option<usize>,
    height: Option<usize>,
    avg_frame_rate: Option<String>,
    nb_read_frames: Option<String>,
    disposition: Option<Disposition>,
}

#[derive(Deserialize)]
struct Disposition {
    #[serde(default)]
    attached_pic: i64,
}

/// The real video: the first video stream that is not an embedded cover picture.
fn pick_video_stream(probe: &Probe) -> Option<&ProbeStream> {
    probe.streams.iter().find(|s| s.disposition.as_ref().is_none_or(|d| d.attached_pic == 0))
}

/// Decode only the frames the sampler will keep, using `ffprobe`/`ffmpeg` subprocesses.
///
/// Mirrors the reference's PyAV path: every frame is counted, the native rate is the stream's
/// average frame rate, frames are chosen by the processor's sampling rule, pixels are the stored
/// (un-rotated) frames converted to RGB24 with swscale's bilinear setting (PyAV's `to_ndarray`
/// default), and embedded cover art is skipped in favour of the real video stream.
fn load_video_ffmpeg(path: &Path, fps_override: Option<f64>, sample: &SampleParams) -> Result<VideoClip> {
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v", "-count_frames", "-show_entries"])
        .arg("stream=index,width,height,avg_frame_rate,nb_read_frames:stream_disposition=attached_pic")
        .args(["-of", "json"])
        .arg(path)
        .output()
        .map_err(|e| {
            anyhow!(
                "decoding `{}` needs ffmpeg/ffprobe on PATH ({e}); alternatively pass a directory of frames or a GIF",
                path.display()
            )
        })?;
    ensure!(probe.status.success(), "ffprobe failed on {}: {}", path.display(), String::from_utf8_lossy(&probe.stderr));
    let probe: Probe = serde_json::from_slice(&probe.stdout).context("parsing ffprobe output")?;
    let st = pick_video_stream(&probe).with_context(|| format!("{} has no video stream", path.display()))?;
    let missing = |what: &str| anyhow!("ffprobe reported no {what} for {}", path.display());
    let (w, h) = (st.width.ok_or_else(|| missing("width"))?, st.height.ok_or_else(|| missing("height"))?);
    let total: usize = st.nb_read_frames.as_deref().and_then(|v| v.parse().ok()).ok_or_else(|| missing("frame count"))?;
    let native = match st.avg_frame_rate.as_deref().unwrap_or("0/1").split_once('/') {
        Some((n, d)) => n.parse::<f64>().unwrap_or(0.0) / d.parse::<f64>().unwrap_or(1.0).max(1e-9),
        None => 0.0,
    };
    let native = fps_override.or(if native > 0.0 { Some(native) } else { None });
    let indices = videoproc::sample_frames(total, native, sample)?;

    // `select` emits each frame once, so decode the distinct indices and repeat duplicates
    // (the sampler repeats frames when the native rate is below the target rate).
    let mut uniq = indices.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let select = uniq.iter().map(|i| format!("eq(n\\,{i})")).collect::<Vec<_>>().join("+");
    let decode = |timing: &[&str]| -> std::io::Result<Output> {
        Command::new("ffmpeg")
            .args(["-v", "error", "-noautorotate", "-i"])
            .arg(path)
            .args(["-map", &format!("0:{}", st.index), "-vf", &format!("select={select}")])
            .args(timing)
            .args(["-sws_flags", "bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .output()
    };
    let mut decoded = decode(&["-fps_mode", "passthrough"])?;
    if !decoded.status.success() && String::from_utf8_lossy(&decoded.stderr).contains("fps_mode") {
        decoded = decode(&["-vsync", "0"])?; // ffmpeg older than 5.1
    }
    ensure!(decoded.status.success(), "ffmpeg failed on {}: {}", path.display(), String::from_utf8_lossy(&decoded.stderr));
    let frame_bytes = w * h * 3;
    ensure!(
        decoded.stdout.len() == frame_bytes * uniq.len(),
        "ffmpeg returned {} bytes, expected {} frames of {w}x{h}",
        decoded.stdout.len(),
        uniq.len()
    );
    let distinct: Vec<RgbImage> = decoded
        .stdout
        .chunks_exact(frame_bytes)
        .map(|c| RgbImage::from_raw(w as u32, h as u32, c.to_vec()).expect("frame size"))
        .collect();
    let frames = indices.iter().map(|i| distinct[uniq.binary_search(i).expect("index decoded")].clone()).collect();
    Ok(VideoClip { frames, fps: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_order() {
        let mut v = vec!["10.png", "2.png", "1.png", "frame_010.png", "frame_9.png", "a.png"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, ["1.png", "2.png", "10.png", "a.png", "frame_9.png", "frame_010.png"]);
        // Zero padding does not change the order of equal-width names.
        let mut w = vec!["f_0002.png", "f_0010.png", "f_0001.png"];
        w.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(w, ["f_0001.png", "f_0002.png", "f_0010.png"]);
    }

    #[test]
    fn cover_art_stream_is_skipped() {
        // ffprobe -show_entries stream=...:stream_disposition=attached_pic -of json
        let json = r#"{"streams": [
            {"index": 0, "width": 32, "height": 32, "avg_frame_rate": "0/0", "nb_read_frames": "1", "disposition": {"attached_pic": 1}},
            {"index": 1, "width": 96, "height": 48, "avg_frame_rate": "10/1", "nb_read_frames": "30", "disposition": {"attached_pic": 0}}
        ]}"#;
        let probe: Probe = serde_json::from_str(json).unwrap();
        assert_eq!(pick_video_stream(&probe).map(|s| s.index), Some(1));
        let only_cover: Probe = serde_json::from_str(r#"{"streams": [{"index": 0, "disposition": {"attached_pic": 1}}]}"#).unwrap();
        assert!(pick_video_stream(&only_cover).is_none());
        let no_disposition: Probe = serde_json::from_str(r#"{"streams": [{"index": 0, "width": 8, "height": 8}]}"#).unwrap();
        assert_eq!(pick_video_stream(&no_disposition).map(|s| s.index), Some(0));
    }

    #[test]
    fn truncated_resample_matches_full() {
        let x: Vec<f32> = (0..48_000).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let clip = AudioClip::new(x, 48_000);
        let full = {
            let mut v = clip.at_rate(16_000);
            v.truncate(5_000);
            v
        };
        assert_eq!(clip.at_rate_truncated(16_000, 5_000).unwrap(), full);
        assert!(AudioClip::new(vec![0.0; 10], 0).at_rate_truncated(16_000, 5).is_err());
    }
}
