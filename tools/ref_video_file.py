"""Reference for video *files*: decode with PyAV (transformers' `pyav` backend semantics),
sample at the processor's fps with its own `sample_frames`, then embed the sampled frames."""
import json, sys, os, av, torch
from sentence_transformers import SentenceTransformer
from transformers.video_processing_utils import VideoMetadata

MODEL, OUT = sys.argv[1], sys.argv[2]
m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
vp = m[0].processor.video_processor
cases = []
for name in ["vf40_10fps_rgb.mp4", "vf40_30fps_yuv420.mp4"]:
    path = os.path.abspath(os.path.join(OUT, name))
    with av.open(path) as c:
        st = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        fps = float(st.average_rate)
    md = VideoMetadata(total_num_frames=len(frames), fps=fps, duration=len(frames) / fps)
    idx = [int(i) for i in vp.sample_frames(md, fps=vp.fps, max_frames=vp.max_frames, overflow_strategy=vp.overflow_strategy)]
    sel = [frames[i] for i in idx]
    feats = m.preprocess([{"video": sel}])
    pv, pos = feats["pixel_values_videos"], feats["video_position_ids"]
    pv[(pos != -1).all(-1)].numpy().astype("<f4").tofile(os.path.join(OUT, f"{name}.patches.bin"))
    cases.append({"name": name, "path": path, "indices": idx, "n_frames": len(sel),
                  "ids_video": feats["input_ids"][0].tolist(), "emb_video": m.encode({"video": sel}).tolist()})
    print(name, "decoded", len(frames), "fps", fps, "sampled", idx, file=sys.stderr)
json.dump({"cases": cases}, open(os.path.join(OUT, "ref_video_file.json"), "w"))
