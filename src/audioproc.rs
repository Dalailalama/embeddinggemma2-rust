//! Audio front end: waveform -> log-mel spectrogram (`Gemma4AudioFeatureExtractor`).
//!
//! 16 kHz mono input, 20 ms periodic-Hann frames with a 10 ms hop, magnitude STFT (512-point FFT),
//! 128 HTK mel bins over 0-8 kHz, `log(mel + 1e-3)`. Only frames whose whole analysis window lies in
//! real audio are produced (the reference masks the rest, which is numerically equivalent).

use anyhow::{Result, bail};
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::config::AudioFeatureConfig;

/// Reference default: audio longer than this many samples (30 s @ 16 kHz) is truncated.
pub const DEFAULT_MAX_SAMPLES: usize = 480_000;

pub struct MelExtractor {
    cfg: AudioFeatureConfig,
    window: Vec<f32>,
    /// `[n_bins, n_mels]` triangular filters, row-major.
    filters: Vec<f64>,
    n_bins: usize,
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

impl MelExtractor {
    pub fn new(cfg: &AudioFeatureConfig) -> Self {
        let n_bins = cfg.fft_length / 2 + 1;
        // Periodic Hann window: 0.5 - 0.5 cos(2 pi n / N).
        let window: Vec<f32> = (0..cfg.frame_length)
            .map(|n| (0.5 - 0.5 * (2.0 * PI * n as f64 / cfg.frame_length as f64).cos()) as f32)
            .collect();

        // transformers.audio_utils.mel_filter_bank(norm=None, mel_scale="htk").
        let n_mels = cfg.feature_size;
        let (mel_min, mel_max) = (hz_to_mel(cfg.min_frequency), hz_to_mel(cfg.max_frequency));
        let filter_freqs: Vec<f64> = (0..n_mels + 2)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
            .collect();
        let nyquist = (cfg.sampling_rate / 2) as f64;
        let fft_freqs: Vec<f64> = (0..n_bins).map(|i| nyquist * i as f64 / (n_bins - 1) as f64).collect();
        let diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();
        let mut filters = vec![0f64; n_bins * n_mels];
        for (b, f) in fft_freqs.iter().enumerate() {
            for m in 0..n_mels {
                let down = -(filter_freqs[m] - f) / diff[m];
                let up = (filter_freqs[m + 2] - f) / diff[m + 1];
                filters[b * n_mels + m] = down.min(up).max(0.0);
            }
        }
        Self { cfg: cfg.clone(), window, filters, n_bins }
    }

    pub fn sampling_rate(&self) -> u32 {
        self.cfg.sampling_rate
    }

    /// Number of valid mel frames for `n_samples` of audio.
    pub fn num_frames(&self, n_samples: usize) -> usize {
        let pad_left = self.cfg.frame_length / 2;
        let need = self.cfg.frame_length + 1;
        if n_samples + pad_left < need { 0 } else { (n_samples + pad_left - need) / self.cfg.hop_length + 1 }
    }

    /// Log-mel features `[frames, feature_size]` for a 16 kHz mono waveform.
    pub fn extract(&self, wav: &[f32]) -> Result<Vec<f32>> {
        let frames = self.num_frames(wav.len());
        if frames == 0 {
            bail!(
                "audio too short: need at least {} samples ({} ms)",
                self.cfg.frame_length + 1 - self.cfg.frame_length / 2,
                1000 * (self.cfg.frame_length + 1 - self.cfg.frame_length / 2) / self.cfg.sampling_rate as usize
            );
        }
        let (fl, hop, nfft) = (self.cfg.frame_length, self.cfg.hop_length, self.cfg.fft_length);
        let pad_left = fl / 2;
        let n_mels = self.cfg.feature_size;
        let scale = self.cfg.input_scale_factor;
        let sample = |idx: usize| -> f32 {
            // `idx` indexes the left-padded signal.
            if idx < pad_left { 0.0 } else { wav.get(idx - pad_left).copied().unwrap_or(0.0) * scale }
        };
        let twiddles = FftPlan::new(nfft);
        let mut out = vec![0f32; frames * n_mels];
        out.par_chunks_mut(n_mels).enumerate().for_each(|(i, row)| {
            let mut re = vec![0f64; nfft];
            let mut im = vec![0f64; nfft];
            for j in 0..fl {
                // float32 multiply, as in the reference (`frames * window`).
                re[j] = (sample(i * hop + j) * self.window[j]) as f64;
            }
            twiddles.fft(&mut re, &mut im);
            let mag: Vec<f64> = (0..self.n_bins).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()).collect();
            for m in 0..n_mels {
                let mut acc = 0f64;
                for (k, mg) in mag.iter().enumerate() {
                    acc += mg * self.filters[k * n_mels + m];
                }
                row[m] = (acc + self.cfg.mel_floor).ln() as f32;
            }
        });
        Ok(out)
    }
}

/// Iterative radix-2 complex FFT with precomputed twiddles (length must be a power of two).
struct FftPlan {
    n: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
    rev: Vec<usize>,
}

impl FftPlan {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let rev = (0..n).map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) }).collect();
        let cos = (0..n / 2).map(|k| (2.0 * PI * k as f64 / n as f64).cos()).collect();
        let sin = (0..n / 2).map(|k| -(2.0 * PI * k as f64 / n as f64).sin()).collect();
        Self { n, cos, sin, rev }
    }

    fn fft(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let (half, step) = (len / 2, n / len);
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}
