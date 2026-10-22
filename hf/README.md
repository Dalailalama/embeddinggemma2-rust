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
# 1. the runtime: `pip install egemma2` (binary + Python client), a binary from this repo's `releases/`
#    folder or GitHub Releases, or build it
pip install egemma2
# 2. the weights
hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
# 3. embed (the model directory defaults to models/embeddinggemma-2)
egemma2 embed --prompt-name SearchQuery --text "which planet is known as the red planet"
egemma2 --device cuda --bf16 embed --image photo.jpg --audio clip.wav --video talk.mp4
egemma2 similarity --query "red planet" --doc "Mars is the fourth planet" --doc "Venus is the hottest planet"
```

## Keep the model loaded: `egemma2 serve`

For applications that embed continuously, one process serves every request instead of reloading the
model per call:

```bash
egemma2 --device cuda --bf16 serve                          # JSON lines on stdin/stdout (subprocess use)
egemma2 --device cuda --bf16 serve --http 127.0.0.1:8080    # OpenAI-compatible POST /v1/embeddings
```

Requests use the OpenAI embeddings format plus multimodal inputs and prompts, for example
`{"input": ["a query", {"image": "page1.png"}, {"text": "Figure 3:", "image": "fig3.png"}], "prompt_name": "SearchQuery"}`.
Images can be paths, data URLs or raw RGB pixels; concurrent requests are merged into shared GPU
passes. OpenAI-style clients (the `openai` SDK, LangChain, LlamaIndex) work against the HTTP server,
and `pip install egemma2` brings a dependency-free Python client for the stdio server
(`from egemma2 import Egemma2`).
Through it a short query takes 2.6 ms round trip. Details:
[Server mode](https://github.com/Dalailalama/embeddinggemma2-rust#server-mode-keep-the-model-loaded).

Texts are not truncated, as in sentence-transformers: a 100,000-token document embeds in 14 s in bf16
on a 12 GB laptop GPU, where PyTorch runs out of memory above ~32,000 tokens. `--max-tokens` sets a cap.

## Same answers as the original

Checked number for number against the PyTorch reference (transformers + sentence-transformers) on 92
checks covering text (up to 11,774 tokens), images, audio, video, prompts and batches: float32
embeddings differ by at most 1e-6.
bf16 mode stays within cosine 0.99996 of float32, about 7x closer than PyTorch's own bf16 mode.

## Speed (RTX A3000 laptop GPU, warm medians)

| Workload | PyTorch bf16 | **Rust bf16** | |
|---|---|---|---|
| short query (17 tokens) | 52.3 ms | **2.5 ms** | 21x |
| long document (2,208 tokens) | 136 ms | **47.6 ms** | 2.85x |
| batch of 64 short documents | 121 ms | **46.0 ms** | 2.6x |
| image 641x479, one per call | 151 - 183 ms | **53.7 ms** | 2.8 - 3.4x |
| 16 images per call, per image | 99.9 ms | **51.4 ms** | 1.9x |
| audio 9.3 s | 91.0 ms | **18.9 ms** | 4.8x |

On the CPU (i7-12850HX) Rust is 1.2 - 1.9x faster than PyTorch, starts in 0.9 s instead of 11 s and
uses about 4x less memory. fp32/TF32 numbers and methodology are in the
[GitHub README](https://github.com/Dalailalama/embeddinggemma2-rust#speed-rust-vs-the-python-original).

## Support

If this saves you time, you can [buy me a coffee](https://buymeacoffee.com/dawgog). New releases are
published on [GitHub](https://github.com/Dalailalama/embeddinggemma2-rust/releases) and mirrored here.

## License

The code is Apache-2.0. The model weights are Google's, distributed under Apache-2.0. "Gemma" and
"EmbeddingGemma" are trademarks of Google LLC.
