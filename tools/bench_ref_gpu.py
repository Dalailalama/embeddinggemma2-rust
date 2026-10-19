"""PyTorch + sentence-transformers timings on the GPU for the shared benchmark inputs.

usage: bench_ref_gpu.py MODEL_DIR TESTDATA_DIR OUT_JSON MODE [ITERS]
MODE: fp32 (TF32 off, PyTorch's default), tf32 (TF32 matmuls allowed), bf16 (weights + compute in bfloat16,
the model card's recommendation on GPUs that support it).

Besides timings, records each mode's accuracy against the float32 CPU reference dumps
(cosine and max |diff| of the final embeddings) so speed is never compared without the precision cost.
"""
import json, os, statistics, sys, time, wave

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL, DATA, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 10

torch.backends.cuda.matmul.allow_tf32 = MODE == "tf32"
torch.backends.cudnn.allow_tf32 = MODE == "tf32"
torch.set_float32_matmul_precision("high" if MODE == "tf32" else "highest")
dtype = torch.bfloat16 if MODE == "bf16" else torch.float32

bench = json.load(open(os.path.join(DATA, "bench_inputs.json")))
m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": dtype}, device="cuda")
img = Image.open(os.path.join(DATA, "image", "img_a_641x479.png")).convert("RGB")
with wave.open(os.path.join(DATA, "audio", "c_9p3s.wav")) as w:
    wav = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def run(fn):
    out = fn()
    torch.cuda.synchronize()
    return out


cases = {
    "short (17 tok)": lambda: m.encode(bench["short"]),
    "long (2208 tok)": lambda: m.encode(bench["long"]),
    "batch of 64 short docs": lambda: m.encode(bench["batch64"], batch_size=64),
    "image 641x479": lambda: m.encode({"image": img}),
    "16 images 641x479, batched": lambda: m.encode([{"image": img}] * 16, batch_size=16),
    "audio 9.3 s": lambda: m.encode({"audio": wav}),
}
res = {}
for name, fn in cases.items():
    run(fn), run(fn)  # warm-up (cuBLAS heuristics, allocator)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter()
        run(fn)
        ts.append((time.perf_counter() - t) * 1e3)
    res[name] = {"median_ms": statistics.median(ts), "min_ms": min(ts), "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
    print(f"{name:<26} median {res[name]['median_ms']:8.1f} ms   min {res[name]['min_ms']:8.1f} ms   "
          f"peak alloc {res[name]['peak_mem_mb']:6.0f} MB", file=sys.stderr)


def acc(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return {"cos": float(got @ want / np.linalg.norm(got) / np.linalg.norm(want)), "max_abs": float(np.abs(got - want).max())}


ref_text = json.load(open(os.path.join(DATA, "ref_text.json")))
ref_img = json.load(open(os.path.join(DATA, "image", "ref_image.json")))["cases"][0]
ref_aud = [c for c in json.load(open(os.path.join(DATA, "audio", "ref_audio.json")))["cases"] if c["name"] == "c_9p3s"][0]
res["_accuracy_vs_fp32_cpu"] = {
    "short text": acc(m.encode(ref_text[0]["text"], prompt_name=ref_text[0]["prompt"]), ref_text[0]["emb"]),
    "long text": acc(m.encode(ref_text[5]["text"], prompt_name=ref_text[5]["prompt"]), ref_text[5]["emb"]),
    "image": acc(m.encode({"image": img}), ref_img["emb_image"]),
    "audio": acc(m.encode({"audio": wav}), ref_aud["emb_audio"]),
}
for k, v in res["_accuracy_vs_fp32_cpu"].items():
    print(f"accuracy {k:<11} cos {v['cos']:.8f}  max|d| {v['max_abs']:.2e}", file=sys.stderr)
res["_mode"] = MODE
res["_attn_impl"] = getattr(m[0].model.config, "_attn_implementation", None)
res["_device"] = torch.cuda.get_device_name(0)
res["_torch"] = torch.__version__
json.dump(res, open(OUT, "w"), indent=1)
