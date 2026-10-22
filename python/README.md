# egemma2: EmbeddingGemma 2 in Rust, for Python

An unofficial, from-scratch Rust implementation of Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2): text, images, audio and video
embedded into one 768-d space (Matryoshka cuts to 512 / 256 / 128), on the CPU or an NVIDIA GPU.
This package installs the `egemma2` binary plus a small Python client that keeps it loaded, so
Python code gets the speed without PyTorch:

| Workload (RTX A3000 laptop GPU, bf16) | PyTorch + sentence-transformers | egemma2 |
|---|---|---|
| short query | 52.3 ms | **2.5 ms** |
| 2,208-token document | 136 ms | **47.6 ms** |
| 16 images per call, per image | 99.9 ms | **51.4 ms** |
| 9.3 s audio clip | 91.0 ms | **18.9 ms** |

Embeddings match the PyTorch reference: up to 1e-6 apart in float32, and within cosine 0.99996 in
bf16 mode. Texts are not truncated, as in sentence-transformers.

## Install

```bash
pip install "egemma2[hub]"     # with huggingface_hub, to download the model on first use (~1.5 GB, once)
pip install egemma2            # or just the binary and the client, for a model you already have
```

Wheels: Windows and Linux x64, any Python 3.8+. The GPU path needs an NVIDIA driver and the CUDA 13
runtime libraries (cuBLAS, NVRTC) on `PATH`, for example from the CUDA Toolkit; bf16 needs compute
capability 8.0+ (Ampere or newer). Without them egemma2 runs on the CPU.

## Use

```python
from pathlib import Path
from egemma2 import Egemma2

with Egemma2(precision="bf16") as model:            # device="auto": CUDA if available, else the CPU
    docs = model.embed(["Mars is red.", "Oceans are blue."], prompt_name="Document")
    query = model.embed("which planet is red?", prompt_name="SearchQuery")[0]
    scores = [sum(q * d for q, d in zip(query, doc)) for doc in docs]   # cosine similarity

    pages = model.embed([Path("page1.png"), Path("page2.png")])          # files, by extension
    mixed = model.embed([{"text": "Figure 3:", "image": pil_image}])     # one interleaved input
    matrix = model.embed_numpy(texts, dimensions=256)                    # (n, 256) float32
```

- **Inputs:** a `str` is text, a `Path` is a file (image, `.wav` audio, video file or frame
  directory). Images can also be Pillow images, HxWx3 uint8 numpy arrays or encoded bytes; audio can
  be `(samples, sample_rate)`. A dict such as `{"text": ..., "image": ...}` is **one** input whose
  parts follow the dict's order, like sentence-transformers' dict inputs.
- **Prompts:** `prompt_name` takes the model's sentence-transformers prompts (`SearchQuery`,
  `Document`, `QuestionAnswering`, `CodeRetrieval`, ...).
- **Model directory:** `Egemma2(model_dir=...)`, else `$EGEMMA2_MODEL_DIR`, else
  `./models/embeddinggemma-2`, else a download of `google/embeddinggemma-2` via huggingface_hub.
- **Other options:** `device="cpu" | "cuda" | "cuda:N"`, `precision=None | "tf32" | "bf16"`,
  `max_tokens=N` to cut long texts.

The client starts `egemma2 serve` once and sends it JSON lines, so the model loads once; a short
query takes 2.6 ms round trip. Requests from several threads are queued safely.

## Command line and HTTP server

The same install gives you the `egemma2` command (also `python -m egemma2`):

```bash
egemma2 --model-dir <dir> embed --prompt-name SearchQuery --text "which planet is red?"
egemma2 --model-dir <dir> --bf16 serve --http 127.0.0.1:8080     # OpenAI-compatible /v1/embeddings
```

Any OpenAI-style client (the `openai` SDK, LangChain, LlamaIndex) can use the HTTP server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
r = client.embeddings.create(model="embeddinggemma-2", input=["a document"], extra_body={"prompt_name": "Document"})
```

## More

Documentation, benchmarks, the parity tests and the source:
<https://github.com/Dalailalama/embeddinggemma2-rust>. Not affiliated with or endorsed by Google;
the model weights are Google's (Apache-2.0). Code: Apache-2.0. If it saves you time, you can
[buy me a coffee](https://buymeacoffee.com/dawgog).
