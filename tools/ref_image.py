"""Reference image path: preprocessed patches, vision-tower soft tokens, and final embeddings."""
import json, sys, os, numpy as np, torch
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer

MODEL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
hf = m[0].model            # EmbeddingGemma2Model
proc = m[0].processor      # EmbeddingGemma2Processor


def make_image(w, h, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x / w, y / h, (x + y) / (w + h)], -1) * 200
    base += rng.normal(0, 12, base.shape)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.2, h * 0.2, w * 0.6, h * 0.7], outline=(255, 30, 30), width=max(2, w // 60))
    d.rectangle([w * 0.55, h * 0.1, w * 0.9, h * 0.45], fill=(20, 200, 40))
    d.line([0, 0, w, h], fill=(255, 255, 255), width=3)
    return img


cases = {"a_641x479": (641, 479, 1), "b_96x64": (96, 64, 2), "c_1000x300": (1000, 300, 3), "d_48x48": (48, 48, 4)}
meta = []
for name, (w, h, seed) in cases.items():
    img = make_image(w, h, seed)
    png = os.path.join(OUT, f"img_{name}.png")
    img.save(png)
    img = Image.open(png).convert("RGB")      # exactly what Rust will decode
    feats = m.preprocess([{"image": img}])
    pv, pos = feats["pixel_values"][0], feats["image_position_ids"][0]
    real = (pos != -1).all(-1)
    n_real = int(real.sum())
    with torch.no_grad():
        soft = hf.get_image_features(feats["pixel_values"], feats["image_position_ids"], return_dict=True).pooler_output[0]
    pv[real].numpy().astype("<f4").tofile(os.path.join(OUT, f"{name}.patches.bin"))
    soft.numpy().astype("<f4").tofile(os.path.join(OUT, f"{name}.soft.bin"))
    entry = {
        "name": name, "png": png, "n_real": n_real,
        "grid_w": int(pos[real][:, 0].max()) + 1, "grid_h": int(pos[real][:, 1].max()) + 1,
        "n_soft": int(soft.shape[0]),
        "ids_image": feats["input_ids"][0].tolist(),
        "emb_image": m.encode({"image": img}).tolist(),
        "ids_text_image": m.preprocess([{"text": "a photo of", "image": img}])["input_ids"][0].tolist(),
        "emb_text_image": m.encode({"text": "a photo of", "image": img}).tolist(),
        "emb_image_text": m.encode({"image": img, "text": "a photo of"}).tolist(),
        "emb_image512": m.encode({"image": img}, truncate_dim=512, normalize_embeddings=True).tolist(),
    }
    meta.append(entry)
    print(name, "patches", n_real, "soft", entry["n_soft"], file=sys.stderr)

# Two images interleaved with text in one embedding.
a = Image.open(os.path.join(OUT, "img_a_641x479.png")).convert("RGB")
b = Image.open(os.path.join(OUT, "img_b_96x64.png")).convert("RGB")
multi = {"inputs": ["A:", "img_a_641x479.png", "B:", "img_b_96x64.png"],
         "ids": m.preprocess([[{"text": "A:"}, {"image": a}, {"text": "B:"}, {"image": b}]])["input_ids"][0].tolist() if False else None}
try:
    ids = m.preprocess([{"text": "A:", "image": [a, b]}])["input_ids"][0].tolist()
    multi["ids_text_two_images"] = ids
    multi["emb_text_two_images"] = m.encode({"text": "A:", "image": [a, b]}).tolist()
except Exception as e:
    print("two-image dict form failed:", repr(e)[:200], file=sys.stderr)
json.dump({"cases": meta, "multi": multi}, open(os.path.join(OUT, "ref_image.json"), "w"))
