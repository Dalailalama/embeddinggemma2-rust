//! EmbeddingGemma 2 in Rust: text, image, audio and video embeddings in one 768-d space.
//!
//! The crate is a from-scratch port of the Hugging Face `embedding_gemma2` reference
//! implementation (text backbone + Gemma 4 vision/audio towers), running on CPU in `f32`.

pub mod audio;
pub mod audioproc;
pub mod backend;
pub mod config;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod imageproc;
pub mod media;
pub mod model;
pub mod ops;
pub mod serve;
pub mod text;
pub mod verify;
pub mod videoproc;
pub mod vision;
pub mod weights;

pub use backend::{Device, Precision};
pub use media::AudioClip;
pub use model::{EmbeddingGemma2, Input, Part, PreparedInput, VideoClip};
