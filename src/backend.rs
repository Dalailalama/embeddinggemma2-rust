//! CPU / CUDA dispatch for the three towers.

use anyhow::Result;

use crate::audio::{self, AudioTower};
use crate::imageproc::PatchedImage;
use crate::ops;
use crate::text::TextModel;
use crate::vision::VisionTower;

#[cfg(feature = "cuda")]
use crate::gpu::{audio::GpuAudio, text::GpuText, vision::GpuVision};

/// Arithmetic of the matrix products on the GPU. Everything else (norms, softmax, residual stream,
/// accumulation) is float32 in every mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// Plain float32 GEMMs: matches the float32 reference to rounding (~1e-6).
    #[default]
    F32,
    /// TF32 tensor cores (10-bit mantissa inputs, float32 accumulation): ~4e-4 deviation.
    Tf32,
    /// bfloat16 tensor cores for every linear layer: weights stay in bf16 on the GPU (the checkpoint's
    /// own format, so loading is lossless and needs half the memory), activations are rounded to bf16
    /// at each GEMM input, accumulation is float32; attention products use TF32.
    Bf16,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Precision::F32 => "fp32",
            Precision::Tf32 => "tf32",
            Precision::Bf16 => "bf16",
        })
    }
}

/// Where the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// CUDA device `ordinal` with the given GEMM [`Precision`].
    Cuda { ordinal: usize, precision: Precision },
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda { ordinal, precision: Precision::F32 } => write!(f, "cuda:{ordinal}"),
            Device::Cuda { ordinal, precision } => write!(f, "cuda:{ordinal} ({precision})"),
        }
    }
}

pub(crate) enum TextBackend {
    Cpu(TextModel),
    #[cfg(feature = "cuda")]
    Gpu(GpuText),
}

impl TextBackend {
    pub(crate) fn embed_tokens(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        match self {
            Self::Cpu(m) => m.embed_tokens(ids, out),
            #[cfg(feature = "cuda")]
            Self::Gpu(m) => m.embed_tokens(ids, out),
        }
    }

    /// Per-token states `[rows, embedding_dim]` for sequences packed back to back.
    pub(crate) fn forward_packed(&self, embeds: &[f32], lens: &[usize]) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(m) => Ok(ops::run_sized(lens.iter().sum(), || m.forward_packed(embeds, lens))),
            #[cfg(feature = "cuda")]
            Self::Gpu(m) => m.forward_packed(embeds, lens),
        }
    }
}

pub(crate) enum VisionBackend {
    Cpu(VisionTower),
    #[cfg(feature = "cuda")]
    Gpu(GpuVision),
}

impl VisionBackend {
    pub(crate) fn encode(&self, img: &PatchedImage) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(t) => ops::run_sized(img.n_patches(), || t.encode(img)),
            #[cfg(feature = "cuda")]
            Self::Gpu(t) => t.encode(img),
        }
    }
}

pub(crate) enum AudioBackend {
    Cpu(AudioTower),
    #[cfg(feature = "cuda")]
    Gpu(GpuAudio),
}

impl AudioBackend {
    pub(crate) fn encode(&self, mel: &[f32], frames: usize, n_mels: usize) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(t) => ops::run_sized(audio::num_audio_tokens(frames), || t.encode(mel, frames, n_mels)),
            #[cfg(feature = "cuda")]
            Self::Gpu(t) => t.encode(mel, frames, n_mels),
        }
    }
}
