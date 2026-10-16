"""Extra reference cases: prompts placed before media, long audio (> 11.2 s, > 30 s), sub-1 fps video."""
import json, os, subprocess, sys, wave
import av, numpy as np, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers.video_processing_utils import VideoMetadata

MODEL, OUT, FFMPEG = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
vp = m[0].processor.video_processor
SR = 16000
res = {"prompt": "SearchQuery", "cases": []}


def add(name, inp, prompt_name=None, **extra):
    prompt = m.prompts[prompt_name] if prompt_name else None
    ids = m.preprocess([inp], prompt=prompt)["input_ids"][0].tolist()
    emb = m.encode(inp, prompt_name=prompt_name).tolist() if prompt_name else m.encode(inp).tolist()
    res["cases"].append({"name": name, "prompt_name": prompt_name, "ids": ids, "emb": emb, **extra})
    print(name, len(ids), "ids", file=sys.stderr)


# 1. The prompt goes before everything, media included (sentence-transformers system message).
img_path = os.path.abspath(os.path.join(OUT, "..", "image", "img_b_96x64.png"))
img = Image.open(img_path).convert("RGB")
add("prompt_image", {"image": img}, "SearchQuery", image=img_path)
add("prompt_text_image", {"text": "a red planet", "image": img}, "SearchQuery", image=img_path, text="a red planet", order="text,image")
add("prompt_image_text", {"image": img, "text": "a red planet"}, "SearchQuery", image=img_path, text="a red planet", order="image,text")

# 2. Long audio: 20 s (> the 11.2 s `audio_seq_length` in processor_config, which is not a cap) and 40 s (> 30 s truncation).
for secs in (20, 40):
    t = np.arange(SR * secs) / SR
    x = (0.3 * np.sin(2 * np.pi * (200 + 30 * t) * t) + 0.03 * np.random.default_rng(secs).normal(size=t.shape)).clip(-1, 1)
    path = os.path.abspath(os.path.join(OUT, f"a{secs}s.wav"))
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x * 32767).astype("<i2").tobytes())
    with wave.open(path) as w:
        arr = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    add(f"audio_{secs}s", {"audio": arr}, audio=path)

# 3. A 0.5 fps video: 1 fps sampling repeats every frame (decoded with PyAV, sampled by the processor).
mp4 = os.path.abspath(os.path.join(OUT, "v12_05fps.mp4"))
subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=1", "-frames:v", "12",
                "-r", "0.5", "-c:v", "libx264rgb", "-crf", "0", mp4], check=True)
with av.open(mp4) as c:
    st = c.streams.video[0]
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    fps = float(st.average_rate)
md = VideoMetadata(total_num_frames=len(frames), fps=fps, duration=len(frames) / fps)
idx = [int(i) for i in vp.sample_frames(md, fps=vp.fps, max_frames=vp.max_frames, overflow_strategy=vp.overflow_strategy)]
add("video_05fps", {"video": [frames[i] for i in idx]}, video=mp4, indices=idx, decoded=len(frames), fps=fps)
json.dump(res, open(os.path.join(OUT, "ref_extra.json"), "w"))
