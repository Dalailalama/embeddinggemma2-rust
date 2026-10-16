//! Typed views over the Hugging Face config files shipped with `google/embeddinggemma-2`.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerOverride {
    pub head_dim: Option<usize>,
    pub num_key_value_heads: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_bias: bool,
    pub pad_token_id: u32,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerOverride>,
    pub rope_parameters: HashMap<String, RopeParams>,
}

impl TextConfig {
    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types[layer] == "sliding_attention"
    }

    /// `(head_dim, num_kv_heads)` for a layer; global layers override both.
    pub fn layer_dims(&self, layer: usize) -> (usize, usize) {
        let ov = self
            .per_layer_config
            .get(&format!("{layer:02}"))
            .or_else(|| self.per_layer_config.get(&layer.to_string()));
        match ov {
            Some(o) => (
                o.head_dim.unwrap_or(self.head_dim),
                o.num_key_value_heads.unwrap_or(self.num_key_value_heads),
            ),
            None => (self.head_dim, self.num_key_value_heads),
        }
    }

    pub fn rope_theta(&self, layer: usize) -> f64 {
        let key = if self.is_sliding(layer) { "sliding_attention" } else { "full_attention" };
        self.rope_parameters[key].rope_theta
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
    pub rope_parameters: RopeParams,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub output_proj_dims: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    pub attention_logit_cap: f32,
    pub attention_invalid_logits_value: f32,
    pub conv_kernel_size: usize,
    pub gradient_clipping: f32,
    pub residual_weight: f32,
    pub rms_norm_eps: f32,
    pub subsampling_conv_channels: Vec<usize>,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub hidden_act: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default)]
    pub audio_config: Option<AudioConfig>,
    pub image_token_id: u32,
    pub audio_token_id: u32,
    pub video_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_index: u32,
}

impl ModelConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: ModelConfig = serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        if cfg.text_config.attention_bias {
            bail!("attention_bias=true is not supported by this port");
        }
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioFeatureConfig {
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub fft_length: usize,
    pub frame_length: usize,
    pub hop_length: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_floor: f64,
    #[serde(default)]
    pub preemphasis: f32,
    #[serde(default = "yes")]
    pub preemphasis_htk_flavor: bool,
    #[serde(default = "one")]
    pub input_scale_factor: f32,
}

fn yes() -> bool {
    true
}
fn one() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageProcConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub max_soft_tokens: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VideoProcConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub max_soft_tokens: usize,
    #[serde(default = "default_fps")]
    pub fps: f64,
    #[serde(default = "default_max_frames")]
    pub max_frames: usize,
    #[serde(default = "default_overflow")]
    pub overflow_strategy: String,
    #[serde(default)]
    pub add_timestamps: bool,
}

fn default_fps() -> f64 {
    1.0
}
fn default_max_frames() -> usize {
    32
}
fn default_overflow() -> String {
    "uniform".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorConfig {
    pub feature_extractor: AudioFeatureConfig,
    pub image_processor: ImageProcConfig,
    pub video_processor: VideoProcConfig,
}

impl ProcessorConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("processor_config.json");
        let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
    }
}

/// Sentence-transformers prompt table (`config_sentence_transformers.json`).
pub fn load_prompts(dir: &Path) -> Result<HashMap<String, String>> {
    #[derive(Deserialize)]
    struct St {
        #[serde(default)]
        prompts: HashMap<String, String>,
    }
    let path = dir.join("config_sentence_transformers.json");
    let raw = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_str::<St>(&raw)?.prompts)
}
