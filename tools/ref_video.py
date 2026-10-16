"""Reference video path: patches, vision-tower soft tokens, embeddings, and frame-sampling decisions."""
import json, sys, os, numpy as np, torch
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer

MODEL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
hf = m[0].model
vp = m[0].processor.video_processor


def make_frame(w, h, k, n):
    rng = np.random.default_rng(100 + k)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x + 5 * k) % w / w, y / h, 0.5 + 0.4 * np.sin(k / 3 + x / 20.0)], -1) * 200
    base += rng.normal(0, 8, base.shape)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    cx = int(w * (0.1 + 0.8 * k / max(n - 1, 1)))
    d.ellipse([cx - 14, h // 2 - 14, cx + 14, h // 2 + 14], fill=(255, 40, 40))
    return img


cases = {"v6_120x200": (6, 200, 120), "v40_96x160": (40, 160, 96)}
meta = []
for name, (n, w, h) in cases.items():
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    for k in range(n):
        make_frame(w, h, k, n).save(os.path.join(d, f"frame_{k:04d}.png"))
    frames = [Image.open(os.path.join(d, f"frame_{k:04d}.png")).convert("RGB") for k in range(n)]
    feats = m.preprocess([{"video": frames}])
    pv, pos = feats["pixel_values_videos"], feats["video_position_ids"]
    nf = int(feats["num_frames_per_video"][0])
    real = (pos != -1).all(-1)
    with torch.no_grad():
        out = hf.get_video_features(pv, pos, feats["num_frames_per_video"], return_dict=True)
    soft = out.pooler_output[0]
    pv[real].numpy().astype("<f4").tofile(os.path.join(OUT, f"{name}.patches.bin"))
    soft.numpy().astype("<f4").tofile(os.path.join(OUT, f"{name}.soft.bin"))
    entry = {
        "name": name, "dir": d, "n_in": n, "n_frames": nf, "n_real_per_frame": int(real[0].sum()), "n_soft": int(soft.shape[0]),
        "ids_video": feats["input_ids"][0].tolist(),
        "emb_video": m.encode({"video": frames}).tolist(),
        "ids_text_video": m.preprocess([{"text": "a clip of", "video": frames}])["input_ids"][0].tolist(),
        "emb_text_video": m.encode({"text": "a clip of", "video": frames}).tolist(),
    }
    meta.append(entry)
    print(name, "frames", nf, "soft", entry["n_soft"], file=sys.stderr)

# Frame-sampling decisions of the HF processor for fps-based inputs.
from transformers.video_processing_utils import VideoMetadata
table = []
for total, fps in [(300, 30.0), (30, 30.0), (45, 15.0), (100, 24.0), (1000, 25.0), (7, 30.0), (10, 1.0), (4000, 30.0), (61, 29.97), (5, 2.0)]:
    md = VideoMetadata(total_num_frames=total, fps=fps, duration=total / fps)
    idx = vp.sample_frames(md, fps=vp.fps, max_frames=vp.max_frames, overflow_strategy=vp.overflow_strategy)
    table.append({"total": total, "fps": fps, "indices": [int(i) for i in idx]})
for total in (6, 40, 32, 33, 100):   # no-metadata path: keep all, then budget
    md = VideoMetadata(total_num_frames=total, fps=None, duration=None)
    idx = vp.sample_frames(md, fps=None, max_frames=vp.max_frames, overflow_strategy=vp.overflow_strategy)
    table.append({"total": total, "fps": None, "indices": [int(i) for i in idx]})
json.dump({"cases": meta, "sampling": table, "fps": vp.fps, "max_frames": vp.max_frames}, open(os.path.join(OUT, "ref_video.json"), "w"))
print("sampling table:", len(table), "rows", file=sys.stderr)
