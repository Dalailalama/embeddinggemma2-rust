---
license: apache-2.0
base_model: google/embeddinggemma-2
pipeline_tag: feature-extraction
tags:
- rust
- cuda
- embeddings
- sentence-similarity
- multimodal
- gemma
- embeddinggemma
---

# EmbeddingGemma 2 in Rust (`egemma2`)

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-dawgog-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/dawgog)
[![GitHub](https://img.shields.io/badge/GitHub-embeddinggemma2--rust-181717?logo=github)](https://github.com/Dalailalama/embeddinggemma2-rust)

An unofficial, from-scratch Rust implementation of
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2): text, images, audio and
video embedded into one 768-d space (Matryoshka cuts to 512 / 256 / 128), on the CPU or on an NVIDIA
GPU (fp32, TF32 or bf16 with tensor-core flash attention). No Python or PyTorch at runtime; the CUDA
libraries are loaded dynamically, so one binary runs with or without a GPU.

This repository holds the documentation and the prebuilt binaries. The **weights are Google's**: this
runtime reads the original checkpoint unchanged, so download it from
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) (Apache-2.0, not gated).
Source code, tests and the full write-up: **<https://github.com/Dalailalama/embeddinggemma2-rust>**.
Not affiliated with or endorsed by Google.

## Quick start

```bash
# 1. the runtime: a binary from this repo's `releases/` folder or GitHub Releases, or build it
cargo install --git https://github.com/Dalailalama/embeddinggemma2-rust
# 2. the weights
hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
# 3. embed (the model directory defaults to models/embeddinggemma-2)
egemma2 embed --prompt-name SearchQuery --text "which planet is known as the red planet"
egemma2 --device cuda --bf16 embed --image photo.jpg --audio clip.wav --video talk.mp4
egemma2 similarity --query "red planet" --doc "Mars is the fourth planet" --doc "Venus is the hottest planet"
```

## Same answers as the original

Checked number for number against the PyTorch reference (transformers + sentence-transformers) on 90
checks covering text, images, audio, video and prompts: float32 embeddings differ by at most 1e-6.
bf16 mode stays within cosine 0.99996 of float32, about 8x closer than PyTorch's own bf16 mode.

## Speed (RTX A3000 laptop GPU, warm medians)

| Workload | PyTorch bf16 | **Rust bf16** | |
|---|---|---|---|
| short query (17 tokens) | 49.4 ms | **2.5 ms** | 19x |
| long document (2,208 tokens) | 136 ms | **50.0 ms** | 2.7x |
| batch of 64 short documents | 117 ms | **53.6 ms** | 2.2x |
| image 641x479 | 151 ms | **59.2 ms** | 2.55x |
| audio 9.3 s | 96.7 ms | **19.1 ms** | 5.1x |

On the CPU (i7-12850HX) Rust is 1.2 - 1.9x faster than PyTorch, starts in 0.9 s instead of 11 s and
uses about 4x less memory. fp32/TF32 numbers and methodology are in the
[GitHub README](https://github.com/Dalailalama/embeddinggemma2-rust#speed-rust-vs-the-python-original).

## Support

If this saves you time, you can [buy me a coffee](https://buymeacoffee.com/dawgog). New releases are
published on [GitHub](https://github.com/Dalailalama/embeddinggemma2-rust/releases) and mirrored here.

## License

The code is Apache-2.0. The model weights are Google's, distributed under Apache-2.0. "Gemma" and
"EmbeddingGemma" are trademarks of Google LLC.
