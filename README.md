# EmbeddingGemma 2 in Rust

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-dawgog-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/dawgog)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DawgOG%2Fembeddinggemma2--rust-yellow)](https://huggingface.co/DawgOG/embeddinggemma2-rust)
[![CI](https://github.com/Dalailalama/embeddinggemma2-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/Dalailalama/embeddinggemma2-rust/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

A from-scratch Rust port of [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
([announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)):
text, images, audio and video embedded into one 768-d space, with Matryoshka truncation to
512 / 256 / 128. It runs on the CPU in `f32` or on an NVIDIA GPU through CUDA in fp32, TF32 or bf16
(cuBLAS plus runtime-compiled kernels, including tensor-core flash attention; the CUDA libraries are
loaded dynamically). It needs no Python or PyTorch at runtime.

**Unofficial:** an independent implementation, not affiliated with or endorsed by Google.
Code: <https://github.com/Dalailalama/embeddinggemma2-rust> · Hugging Face:
<https://huggingface.co/DawgOG/embeddinggemma2-rust> · Prebuilt binaries (Windows / Linux x64):
[Releases](https://github.com/Dalailalama/embeddinggemma2-rust/releases). If it is useful to you,
you can [buy me a coffee](https://buymeacoffee.com/dawgog).

Everything is ported from the Hugging Face reference implementation (`embedding_gemma2` +
Gemma 4 vision/audio towers) and checked number-for-number against it:

| Part | What it is |
|---|---|
| Text backbone | 24 bidirectional layers, per-layer-input (PLE) gating, 5:1 sliding/global attention (window radius 512, 512-wide heads on global layers), sandwich RMSNorm, 512 -> 768 projection, mean pooling, L2 norm |
| Vision tower | 16 layers over 16x16 patches, learned 2-D position tables + axial 2-D RoPE, 3x3 pooling -> soft tokens (280 per image by default) |
| Audio tower | log-mel front end (16 kHz, 128 mels), 2x stride-2 conv subsampling (25 tokens/s), 12 conformer layers with chunked local attention + relative position bias |
| Video | frame sampling (1 fps, up to 32 frames) through the vision tower, 140 soft tokens per frame |
| Prompts | all sentence-transformers prompt names (`SearchQuery`, `Document`, `CodeRetrieval`, ...), placed like sentence-transformers places them: before the whole input, media included |

## Parity with the original (PyTorch, float32)

`egemma2 verify` / `cargo test --release` compare against dumps produced by `tools/ref_*.py`
with transformers 5.19 + sentence-transformers 6.1. There are 90 checks:

| Check | Max abs. difference (CPU) | Notes |
|---|---|---|
| Text embeddings (17 – 2,208 tokens, 4 prompts, multilingual, code) | 4e-7 | token ids identical; 512/256/128-d identical; packed batches identical |
| Image preprocessing (4 sizes, up- and down-scaling) | 6e-8 | bit-exact port of PyTorch's uint8 antialiased bicubic resize |
| Image soft tokens / embeddings (incl. text+image, two images) | 2e-4 / 1e-6 | soft-token magnitude ~6 |
| Audio log-mel / soft tokens / embeddings (0.5 – 9.3 s) | 5e-7 / 2e-4 / 8e-7 | |
| Video frame sampling (15 cases) / embeddings | exact / 2e-7 | |
| Video files decoded by ffmpeg (mp4: lossless RGB and yuv420p H.264) | exact frames | bit-for-bit equal to the reference's PyAV decode |
| Extra cases (`tools/ref_extra.py`): prompt before an image / text+image / image+text, 20 s and 40 s audio, a 0.5 fps video | 5e-7 | token ids identical; confirms there is no 11.2 s audio cap, the 30 s truncation, and frame repetition below 1 fps |

These are float32 rounding differences; cosine similarity with the reference prints as
`1.00000000` everywhere.

| Backend | Tolerance profile | Largest embedding difference | Result |
|---|---|---|---|
| CPU, float32 | float32 (embeddings <= 5e-6) | 9.9e-7 | 90 / 90 |
| CUDA, float32 (default) | float32 (embeddings <= 5e-6) | 8.4e-7 | 90 / 90 |
| CUDA, `--tf32` | tf32 (embeddings <= 1e-3 **and** cosine >= 0.99999) | 4.3e-4, min cosine 0.9999946 | 90 / 90 |
| CUDA, `--bf16` | bf16 (embeddings <= 4e-3 **and** cosine >= 0.99991) | 1.4e-3, min cosine 0.9999625 | 90 / 90 |

TF32 rounds GEMM inputs to a 10-bit mantissa, so `--tf32` gets its own, measured tolerance profile.
The max-abs bound has about 2.3x headroom over the worst case seen. The cosine floor stops a real bug
from hiding inside it: a uniform 9e-4 error passes the max-abs bound but fails the floor. Quantities
computed on the CPU (patches, log-mel) keep the strict limits in every profile. `verify` prints the
active profile first and the worst case per quantity last. The bf16 profile was derived the same way
(max-abs ~2.7x and 1 - cos at least 2x above the worst case seen). For scale, PyTorch's own bf16 mode
lands at cosine 0.99975 on the 9.3 s audio clip, about 7x further from the reference (in 1 - cos) than
this port's bf16 worst case.

## Speed: Rust vs. the Python original

Same machine: i7-12850HX (8P+8E cores / 24 threads), 64 GB, RTX A3000 12 GB laptop GPU, Windows 11.
Warm medians of 10 runs (CPU: 5), end to end: preprocessing + model + pooling. The reference is
PyTorch 2.14.1 + transformers 5.19 + sentence-transformers 6.1 with its default SDPA attention.

### GPU vs. GPU (same RTX A3000)

| Workload | PyTorch fp32 | **Rust fp32** | PyTorch TF32 | **Rust TF32** | PyTorch bf16 | **Rust bf16** |
|---|---|---|---|---|---|---|
| short query (17 tokens) | 45.4 ms | **4.1 ms** (11x) | 45.6 ms | **3.6 ms** (12.7x) | 52.9 ms | **2.6 ms** (20x) |
| long document (2,208 tokens) | 297 ms | **190 ms** (1.56x) | 243 ms | **105 ms** (2.3x) | 136 ms | **47.1 ms** (2.9x) |
| batch of 64 short documents | 217 ms | **156 ms** (1.39x) | 142 ms | **80.4 ms** (1.77x) | 119 ms | **45.1 ms** (2.6x) |
| image 641x479 | 307 ms | **219 ms** (1.4x) | 237 ms | **136 ms** (1.75x) | 169 ms | **59.1 ms** (2.9x) |
| audio 9.3 s | 96.8 ms | **52.1 ms** (1.86x) | 85.7 ms | **25.7 ms** (3.3x) | 100 ms | **19.5 ms** (5.1x) |
| GPU memory (rise over idle) | 3.7 GB | **2.5 GB** | 3.7 GB | **2.5 GB** | 2.2 GB | **1.3 GB** |
| worst cosine vs. fp32 reference | 1.0000000 | 1.0000000 | 0.999997 | 0.9999946 | 0.99975 | 0.9999625 |

Speed-ups are Rust vs. PyTorch at the same precision; all six columns come from one run of
`tools/bench_gpu_compare.py`. bf16 is the model card's recommendation on GPUs. **Rust bf16 is the
fastest configuration on every workload: 2.6 - 20x faster than PyTorch bf16, with 40% less GPU
memory, and about 7x closer to the float32 reference** (1 - cos 3.7e-5 vs. 2.5e-4). PyTorch's image
time varies more than the rest between runs (151 - 169 ms over the last two), so read that ratio as
2.5 - 2.9x. Rust TF32 also beats PyTorch bf16 on all five workloads while staying ~50x closer to the
reference; PyTorch bf16 is still faster than Rust *fp32* on the long document, the batch and the
image. Worst cosines for Rust are over the 90 `verify` checks; for PyTorch, over the four benchmark
inputs (the audio clip in both reduced-precision modes).

The short query is where the gap is biggest. PyTorch takes 45 - 53 ms on the GPU, about as long as on
the CPU (48 ms), so its short-query time is fixed per-call overhead rather than compute. Rust replays
the 17-token pass from a recorded CUDA graph in 2.6 - 4 ms. The large inputs are bound by
matrix-multiply throughput and memory traffic on both sides, so the margin there is smaller:
1.4 - 1.9x at fp32, 1.75 - 3.3x with TF32 and 2.6 - 5x with bf16. In bf16 mode three steps took the
large inputs most of the way (see [GPU](#gpu-cuda)): storing the tensors around each matrix product
as bf16 (long document 79.6 -> 62.8 ms, batch 69.1 -> 53.6 ms, image 79.2 -> 62.2 ms), tensor-core
flash attention for the text's sliding-window layers and the vision tower (long document 50.0 ms,
image 59.2 ms), and flash kernels for the text's 512-wide global layers and for packed batches of
short texts (long document 47.1 ms, batch 45.1 ms).

### CPU vs. CPU

| Workload | PyTorch CPU fp32 | **Rust CPU fp32** | |
|---|---|---|---|
| short query (17 tokens) | 48 ms | **26 ms** | 1.9x |
| long document (2,208 tokens) | 3,169 ms | **1,811 ms** | 1.75x |
| batch of 64 short documents | 2,446 ms | **1,840 ms** | 1.33x |
| image 641x479 | 3,241 ms | **2,646 ms** | 1.22x |
| audio 9.3 s | 700 ms | **551 ms** | 1.27x |

Two things matter for small inputs on the CPU. **Size-aware thread pools:** passes under 256 rows
run on a small pool, because waking all 24 threads, including the slower E-cores, for a 17-row
matmul cost more than it saved. **Sequence packing:** batched texts share one forward pass, so the
weights are streamed once per batch instead of once per document. Without these two, Rust was
*slower* than PyTorch on short queries (63 ms) and on batches (6.8 s).

### Start-up and memory (launch -> 1 embedding -> exit)

| | PyTorch + sentence-transformers | **Rust** | |
|---|---|---|---|
| cold start, CPU | 11.1 s | **0.9 s** | 12x |
| peak host memory, CPU | 3,660 MB | **862 MB** | 4.2x less |
| cold start, GPU | 14.2 s | **1.4 s** | 10x |
| peak host memory, GPU | 4,426 MB | **1,159 MB** | 3.8x less |

Towers load lazily from a memory-mapped file only when an input needs them.

To reproduce (`bench_ref.py` also writes `testdata/bench_inputs.json`, which `egemma2 bench` reads):

```bash
python tools/bench_ref.py models/embeddinggemma-2 testdata testdata/bench_python.json        # PyTorch CPU
egemma2 --device cpu bench --out testdata/bench_rust.json                                     # Rust CPU
# GPU vs GPU: PyTorch and Rust in fp32/TF32/bf16, one process at a time, with VRAM sampling
python tools/bench_gpu_compare.py models/embeddinggemma-2 testdata <cuda-python> target/release/egemma2.exe 10
python tools/bench_proc.py models/embeddinggemma-2 <python> target/release/egemma2.exe cuda   # cold start
```

## Build

Requires a Rust toolchain (1.85+, edition 2024).

```bash
git clone https://github.com/Dalailalama/embeddinggemma2-rust.git
cd embeddinggemma2-rust
cargo build --release          # binary: target/release/egemma2 (egemma2.exe on Windows)
```

The CUDA backend is the cargo feature `cuda`, on by default. It needs nothing at build time (no
CUDA SDK, no `nvcc`): the NVIDIA driver, cuBLAS and NVRTC are loaded dynamically at run time. For a
build without CUDA support, use `cargo build --release --no-default-features`.

Or skip the build: the [Releases](https://github.com/Dalailalama/embeddinggemma2-rust/releases)
page has prebuilt `egemma2` binaries for Windows and Linux (x64; CUDA support included, used when a
driver is present).

The model files (about 1.5 GB, Apache-2.0, not gated) go in `models/embeddinggemma-2/`. With the
Hugging Face CLI (`hf`):

```bash
hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
```

or with plain `curl`:

```bash
mkdir -p models/embeddinggemma-2/1_Pooling models/embeddinggemma-2/2_Normalize
cd models/embeddinggemma-2
for f in config.json processor_config.json preprocessor_config.json tokenizer.json tokenizer_config.json config_sentence_transformers.json modules.json model.safetensors; do curl -L -O "https://huggingface.co/google/embeddinggemma-2/resolve/main/$f"; done
```

## Command line

```bash
# Text with a task prompt, full 768-d
egemma2 embed --prompt-name SearchQuery --text "which planet is known as the red planet"

# Several items at once (one embedding each), truncated to 256-d
egemma2 embed --prompt-name Document --text "first doc" --text "second doc" --dim 256

# Image, audio (.wav), video (frame directory, animated GIF, or any file if ffmpeg is installed)
egemma2 embed --image cat.png --audio speech.wav --video clip.mp4

# One interleaved multimodal input, in command-line order
egemma2 embed --combine --text "a photo of" --image cat.png

# Rank documents for a query (SearchQuery vs Document prompts)
egemma2 similarity --query "capital of Japan" --doc "Tokyo is the capital of Japan." --doc "Berlin is in Germany."

# Choose the device (default: auto = CUDA if available, else CPU) and the GPU precision
egemma2 --device cuda --bf16 embed --text "hello"     # bf16 tensor cores: fastest
egemma2 --device cuda --tf32 embed --text "hello"     # TF32 tensor cores
egemma2 --device cpu embed --text "hello"

# Parity checks and benchmark (both honour --device)
egemma2 verify
egemma2 bench
```

Output is JSON: `{"dim": 768, "inputs": [...], "embeddings": [[...], ...]}`. Use `--out file.json`
to write it to a file. The first line on stderr, `model loaded on cpu` / `cuda:0` /
`cuda:0 (tf32)`, tells you which device was actually used.

**Prompts.** Like sentence-transformers, which passes the prompt to this model as a system message,
`--prompt-name` puts the prompt before the whole input, before an image or audio clip too. The model
card applies prompts to text only, so leave `--prompt-name` off for image/audio/video-only inputs
unless you want that. Text values may start with `-` (`--text "- a bullet"`).

Global flags (accepted before or after the subcommand): `--model-dir` (default
`models/embeddinggemma-2`), `--device`, `--tf32`, `--bf16`, `--threads`, `--small-threads`. `embed` also takes
`--prompt-name`, `--dim`, `--fps`, `--combine`, `--max-audio-seconds`, `--pil-resize` (Pillow's
resampler instead of PyTorch's) and `--out`; `similarity` takes `--query`, `--doc`, `--docs-file` and
`--dim`; `verify` takes `--reference-dir` and `--only text|image|audio|video|extra`; `bench` takes
`--data-dir`, `--iters` and `--out`.

## GPU (CUDA)

All three towers run on an NVIDIA GPU through the CUDA backend. The device is chosen with
`--device`:

| Flag | Effect |
|---|---|
| `--device auto` (default) | CUDA device 0 if it can be opened, otherwise the CPU, with a "CUDA unavailable (...); running on the CPU" message on stderr. Missing CUDA libraries are detected up front, so this falls back cleanly. |
| `--device cpu` | CPU only |
| `--device cuda`, `--device cuda:N` | CUDA device 0, or device number N; a clean error instead of a fallback if CUDA can't be used |
| `--tf32` | let cuBLAS use TF32 tensor cores: faster, embeddings deviate from float32 by up to ~4e-4 (see [Parity](#parity-with-the-original-pytorch-float32)). Off by default; ignored on the CPU |
| `--bf16` | run every linear layer and the attention products on bf16 tensor cores, with bf16 activations around them: fastest, embeddings deviate by up to ~1.4e-3 (cosine >= 0.99996). Needs compute capability 8.0+ (Ampere or newer). Off by default; ignored on the CPU |

**Requirements.** An NVIDIA GPU with its driver, and the CUDA 13.x runtime libraries
`cublas64_13.dll` and `nvrtc64_130_0.dll` findable on `PATH` (the `bin` directory of a CUDA Toolkit
13.x install, for example). Nothing is needed at build time, because the libraries are loaded
dynamically. Developed and tested on Windows 11 with CUDA 13.2 (driver 610.88); other platforms are
untested.

**Design** (`src/gpu/`):

- Activations stay on the GPU for a whole tower forward pass. Only the inputs (token embeddings,
  image patches, log-mel features) go up and the final per-token states come down.
- Every matrix product is a cuBLAS GEMM, with q|k|v and gate|up merged into single GEMMs. In fp32 and
  TF32 mode, text attention uses batched cuBLAS GEMMs around a windowed-softmax kernel (sliding-window
  layers only multiply the keys within reach of each query block), the vision tower a fused
  flash-style kernel (head_dim 64), and packed batches of short texts one fused attention launch per
  layer. bf16 mode runs all attention on tensor-core flash kernels, see below.
- Launch and allocation overhead is the bottleneck for short inputs. Layers therefore use few, fused
  kernels (norm + RoPE, residual add + norm, gated activations), scratch buffers are allocated once
  per pass, and short text passes (up to 128 rows) are recorded once as CUDA graphs and replayed.
  The device-memory pool keeps up to 1 GiB of freed memory cached between calls.
- The rest (RMSNorm, GELU/SiLU gating, RoPE, PLE gating, clamping, the conformer's relative
  attention, GLU, depthwise conv and its two subsampling convolutions) are CUDA kernels in
  `src/gpu/kernels.cu`, compiled by NVRTC when the model is opened on a GPU. The audio front end
  stays exact float32 even under `--tf32`.
- Still on the host: tokenisation, image resize, log-mel, video decoding, the vision tower's position
  and RoPE tables, and its final pooling/projection.
- Weights are uploaded (as float32, or bf16 in bf16 mode): the text backbone when the model is opened,
  the vision and audio towers the first time an input needs them (once, even with concurrent first calls).
- **bf16 mode.** Weights stay in bf16 on the GPU. That is the checkpoint's own format, so it is
  lossless and needs half the memory of fp32. cuBLAS multiplies on bf16 tensor cores with float32
  accumulation, and the tensors that go into and come out of the products are stored as bf16 too. The
  kernel that produces a product's input (a norm, the GELU gate, an attention output, ...) rounds it on
  the way out, which is exactly the rounding the product would apply, so that part changes no result;
  cuBLAS rounds the outputs. Together that halves the memory traffic of every element-wise step. The
  residual stream, the norms' statistics, softmax and all sums stay float32.
- **Tensor-core flash attention (bf16 mode).** Every attention layer runs on flash kernels in
  `kernels_sm80.cu`: `cp.async` copies into XOR-swizzled shared memory, `ldmatrix` fragment loads and
  inline-PTX `mma.m16n8k16`, so the score matrix never leaves the chip and keys outside a query tile's
  window are skipped. The vision tower's 64-wide heads and the text's 256-wide sliding-window heads
  share one kernel (64 queries per block of 4 warps). The text's 512-wide global heads have their own:
  a warp cannot hold a 16 x 512 float32 output tile, so its 8 warps pair up, each half of a pair
  keeping Q in registers and accumulating the output for 256 of the 512 channels, with the partial
  scores exchanged through shared memory. They run at ~21-25 TFLOPS, against ~5 for the fp32 flash
  kernel, and land closer to an exact float64 reference than the cuBLAS path
  (`examples/check_attention.rs`). Packed batches of short texts use the same kernels in a tile-list
  mode (one launch covers all sequences, 64 queries per tile) when their sequences average 24 tokens
  or more; below that the float32 packed kernel, which walks each row's few keys serially, is faster
  (`examples/profile_packed.rs`). The per-head norm + RoPE step runs one warp per head. The audio tower
  keeps float32 activations; its front end (subsampling convs) is exact float32 in every mode and its
  relative-position table keeps float32 weights.
- Debug switches: `EGEMMA2_NO_FLASH` (cuBLAS attention instead of the fused kernels),
  `EGEMMA2_PACKED_FLASH` (packed text batches: by default the tensor-core flash kernels when the
  sequences average 24+ tokens, else the float32 packed-attention kernel; `0`: always the packed
  kernel, `2`: always flash),
  `EGEMMA2_NO_GRAPH` (no CUDA graphs) and `EGEMMA2_PROFILE` (device time per operation; see
  `examples/profile_passes.rs`).

## Library

```rust
use embeddinggemma2::{Device, EmbeddingGemma2, Input, Part, Precision, media};
use std::path::Path;

let model = EmbeddingGemma2::load("models/embeddinggemma-2")?;
let query = model.embed_text(Some("SearchQuery"), "red planet")?;   // Vec<f32>, L2-normalised
let mars = media::load_image(Path::new("mars.jpg"))?;
let image = model.embed(&Input::image(mars.clone()))?;
let mixed = model.embed(&Input { parts: vec![Part::Text("a photo of".into()), Part::Image(mars)] })?;
let score: f32 = query.iter().zip(&image).map(|(a, b)| a * b).sum();  // cosine similarity

// A prompt in front of a whole (multimodal) input, as sentence-transformers does it.
let prompted = model.embed(&Input::text("red planet").with_prompt(model.prompt("SearchQuery")?))?;

// Many texts in packed forward passes, truncated to 256-d (Matryoshka) and re-normalised.
let docs: Vec<Input> = ["title: none | text: Mars is red.", "title: none | text: Oceans are blue."]
    .iter().map(|t| Input::text(*t)).collect();
let doc_embs = model.embed_batch(&docs, Some(256))?;

// The same on the GPU (`Precision::F32`, `Tf32` or `Bf16`). Returns an error if CUDA can't be
// initialised; the automatic CPU fallback is a CLI feature (`--device auto`), not a library one.
let gpu_model = EmbeddingGemma2::load_on("models/embeddinggemma-2", Device::Cuda { ordinal: 0, precision: Precision::Bf16 })?;
```

## Code map

| File | Contents |
|---|---|
| `src/model.rs` | public API, prompt assembly, placeholder expansion, soft-token merge, pooling, batching |
| `src/backend.rs` | `Device` and the CPU / CUDA dispatch for the three towers |
| `src/text.rs` | text backbone |
| `src/vision.rs`, `src/imageproc.rs` | vision tower; resize (PyTorch and Pillow variants) and patchify |
| `src/audio.rs`, `src/audioproc.rs` | audio conformer; log-mel spectrogram |
| `src/videoproc.rs`, `src/media.rs` | frame sampling; image/WAV/video loading, resampling |
| `src/ops.rs` | GEMM wrappers, norms, RoPE, blocked attention, size-aware thread pools |
| `src/gpu/` | CUDA backend: `mod.rs` (device, fp32/TF32/bf16 cuBLAS GEMMs, attention, kernel launchers, graphs), `text.rs` / `vision.rs` / `audio.rs` (tower forward passes), `kernels.cu` (NVRTC kernels), `kernels_sm80.cu` (bf16 tensor-core flash attention, compute capability 8.0+) |
| `src/weights.rs`, `src/config.rs` | memory-mapped safetensors (BF16/F16/F32), config files |
| `src/verify.rs`, `tests/parity.rs` | parity checks and tolerance profiles; parity tests |
| `examples/` | `check_attention` (flash kernels vs. the cuBLAS path and an exact reference), profiling tools: `profile_passes` (device time per operation of whole GPU passes, run with `EGEMMA2_PROFILE=1`), `profile_gpu` (GEMM and attention shapes per precision), `profile_text`, `profile_image`, `profile_ops` |
| `tools/` | reference-dump and benchmark scripts (Python, used only for verification), `publish_public.py` (publishes a release snapshot) |

## Testing

```bash
cargo test --release                                       # CPU (unit tests + parity)
EGEMMA2_DEVICE=cuda cargo test --release                   # CUDA, float32
EGEMMA2_DEVICE=cuda EGEMMA2_TF32=1 cargo test --release    # CUDA, TF32 (tf32 tolerance profile)
EGEMMA2_DEVICE=cuda EGEMMA2_BF16=1 cargo test --release    # CUDA, bf16 (bf16 tolerance profile)
```

`tests/parity.rs` has one test each for text, image, audio, video and the extra cases, all built on
the checks in `src/verify.rs`; unit tests cover the tolerance profiles, frame sampling, natural
frame ordering and truncated resampling. Environment variables:

| Variable | Meaning |
|---|---|
| `EGEMMA2_MODEL_DIR` | model directory (default `models/embeddinggemma-2`; handy from a git worktree) |
| `EGEMMA2_TESTDATA` | reference-data directory (default `testdata`) |
| `EGEMMA2_DEVICE` | `cuda` runs on CUDA device 0; anything else, or unset, runs on the CPU |
| `EGEMMA2_TF32` | `1` (with `EGEMMA2_DEVICE=cuda`) uses TF32 and the tf32 tolerance profile |
| `EGEMMA2_BF16` | `1` (with `EGEMMA2_DEVICE=cuda`) uses bf16 and the bf16 tolerance profile |

When the model or the reference data are missing, the tests print `SKIPPED` and return early,
which counts as a pass (run with `-- --nocapture` to see the messages). Some individual checks can
also be skipped, with a message. The mp4 checks are skipped when ffmpeg isn't on `PATH`. And
`testdata/video/v40_96x160.patches.bin` (119 MB) is not in git, because it is over GitHub's 100 MB
file limit. The pixel check of that 40-frame clip is skipped unless you regenerate the file with
`tools/ref_video.py` (see below). Its soft-token and embedding checks still run.

## Limitations

- **GPU: CUDA only, one device.** There is no ROCm/Metal/Vulkan backend. A model uses a single GPU
  (pick it with `cuda:N`) and calls into it are serialised. bf16 needs compute capability 8.0+.
  The tensor-core flash kernels are bf16-only; fp32 and TF32 keep the cuBLAS attention path for long
  texts and the float32 packed kernel for batches of short ones.
- **Audio input is WAV** (8/16/24/32-bit PCM or 32-bit float), mixed down to mono. Other sample
  rates are resampled with a windowed-sinc filter. That works fine but is not bit-identical to
  librosa or torchaudio. Audio is truncated at 30 s like the reference feature extractor;
  `--max-audio-seconds` raises the limit (the model accepts about 5.5 minutes).
- **Video containers (mp4, ...)** need `ffmpeg`/`ffprobe` on `PATH`. This is tested with 9.0.2;
  there is a `-vsync 0` fallback for ffmpeg older than 5.1, which is untested. Frames are the stored
  ones: rotation metadata is ignored, as in the reference's PyAV decode (a 90-degree-rotated copy
  embeds identically to the original). Streams flagged as attached pictures (cover art) are skipped.
  Frame directories are read in natural file-name order (`2.png` before `10.png`). Frame directories
  and animated GIFs work without ffmpeg.
- The reference is float32. A bfloat16 PyTorch run differs from it by about 1e-3 (cosine ~0.9998).
- JPEG decoders can differ from Pillow by +-1 per pixel, so JPEG inputs may differ very slightly
  more than the PNG test inputs.

## Regenerating the reference data

```bash
python -m venv .venv && .venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install "transformers>=5.18" "sentence-transformers>=6.1" pillow numpy psutil av
.venv/Scripts/python tools/ref_text.py  models/embeddinggemma-2 testdata/ref_text.json
.venv/Scripts/python tools/ref_image.py models/embeddinggemma-2 testdata/image
.venv/Scripts/python tools/ref_audio.py models/embeddinggemma-2 testdata/audio
.venv/Scripts/python tools/ref_video.py models/embeddinggemma-2 testdata/video       # also recreates v40_96x160.patches.bin
.venv/Scripts/python tools/ref_video_file.py models/embeddinggemma-2 testdata/video  # mp4 references (PyAV decode)
.venv/Scripts/python tools/ref_extra.py models/embeddinggemma-2 testdata/extra <path-to-ffmpeg>
```

On Windows, install the CUDA build of PyTorch (`--index-url https://download.pytorch.org/whl/cu130`)
into a venv with a short path. Otherwise a deeply nested license file in the wheel exceeds the
260-character path limit and pip skips torchvision.

## License and support

The code is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE). The model
weights are Google's, distributed by Google under Apache-2.0, and are not part of this repository.
"Gemma" and "EmbeddingGemma" are trademarks of Google LLC.

If this project saves you time, you can support it on
[Buy Me a Coffee](https://buymeacoffee.com/dawgog). Updates are published here and on
[Hugging Face](https://huggingface.co/DawgOG/embeddinggemma2-rust).
