"""PyTorch/sentence-transformers timings (fp32, CPU) for the shared benchmark inputs."""
import json, sys, time, statistics, os, torch
from PIL import Image
import numpy as np, wave
from sentence_transformers import SentenceTransformer

MODEL, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 5

ref = json.load(open(os.path.join(DATA, "ref_text.json")))
LONG = ref[5]["text"]
batch = [f"title: none | text: Document {i} discusses topic {i % 13}: the quick brown fox number {i} jumps over lazy dog {i * 7}, "
         f"and then explains how embeddings place similar meanings close together." for i in range(64)]
bench = {"short": "task: search result | query: which planet is known as the red planet", "long": LONG, "batch64": batch}
json.dump(bench, open(os.path.join(DATA, "bench_inputs.json"), "w"))

m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
img = Image.open(os.path.join(DATA, "image", "img_a_641x479.png")).convert("RGB")
with wave.open(os.path.join(DATA, "audio", "c_9p3s.wav")) as w:
    wav = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

cases = {
    "short (17 tok)": lambda: m.encode(bench["short"]),
    "long (2208 tok)": lambda: m.encode(bench["long"]),
    "batch of 64 short docs": lambda: m.encode(bench["batch64"], batch_size=64),
    "image 641x479": lambda: m.encode({"image": img}),
    "audio 9.3 s": lambda: m.encode({"audio": wav}),
}
res = {}
for name, fn in cases.items():
    fn()  # warm-up
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    res[name] = {"median_ms": statistics.median(ts), "min_ms": min(ts)}
    print(f"{name:<26} median {res[name]['median_ms']:9.1f} ms   min {res[name]['min_ms']:9.1f} ms", file=sys.stderr)
res["_threads"] = torch.get_num_threads()
json.dump(res, open(OUT, "w"))
