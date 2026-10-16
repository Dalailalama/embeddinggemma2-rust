"""Reference audio path: log-mel features, audio-tower soft tokens, and final embeddings."""
import json, sys, os, wave, numpy as np, torch
from sentence_transformers import SentenceTransformer

MODEL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
hf = m[0].model
SR = 16000


def synth(seconds=None, n=None, seed=0):
    n = n or int(SR * seconds)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    f = 200 + 600 * t / t[-1]                                  # rising chirp
    x = 0.35 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    x += 0.2 * np.sin(2 * np.pi * 1250 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    x += 0.04 * rng.normal(size=n)
    x *= np.clip(np.minimum(t, t[-1] - t) * 20, 0, 1)         # fade in/out
    return np.clip(x, -1, 1)


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


cases = {"a_3p7s": synth(3.7, seed=1), "b_12345": synth(n=12345, seed=2), "c_9p3s": synth(9.3, seed=3), "d_0p5s": synth(0.5, seed=4)}
meta = []
for name, x in cases.items():
    path = os.path.join(OUT, f"{name}.wav")
    write_wav(path, x)
    arr = read_wav(path)                                        # exactly what Rust will decode
    feats = m.preprocess([{"audio": arr}])
    mask = feats["input_features_mask"][0].bool()
    mel = feats["input_features"][0][mask]
    with torch.no_grad():
        out = hf.get_audio_features(feats["input_features"], feats["input_features_mask"], return_dict=True)
    soft = out.pooler_output[0][out.attention_mask[0]]
    mel.numpy().astype("<f4").tofile(os.path.join(OUT, f"{name}.mel.bin"))
    soft.numpy().astype("<f4").tofile(os.path.join(OUT, f"{name}.soft.bin"))
    entry = {
        "name": name, "wav": path, "n_samples": int(arr.shape[0]), "frames": int(mel.shape[0]), "n_soft": int(soft.shape[0]),
        "ids_audio": feats["input_ids"][0].tolist(),
        "emb_audio": m.encode({"audio": arr}).tolist(),
        "ids_text_audio": m.preprocess([{"text": "a recording of", "audio": arr}])["input_ids"][0].tolist(),
        "emb_text_audio": m.encode({"text": "a recording of", "audio": arr}).tolist(),
    }
    meta.append(entry)
    print(name, "samples", arr.shape[0], "frames", entry["frames"], "soft", entry["n_soft"], file=sys.stderr)
json.dump({"cases": meta}, open(os.path.join(OUT, "ref_audio.json"), "w"))
