"""GPU vs GPU: PyTorch (CUDA) vs Rust egemma2 (CUDA) on the shared benchmark inputs.

usage: bench_gpu_compare.py MODEL_DIR TESTDATA_DIR CUDA_PYTHON EGEMMA2_EXE [ITERS]

Runs every configuration in its own process, one at a time, and samples total GPU memory with
nvidia-smi (per-process VRAM is not reported under Windows' WDDM driver), so "VRAM" below is the
peak rise of total used memory over the pre-run baseline -- the same measurement for both sides.
"""
import json, os, subprocess, sys, threading, time

MODEL, DATA, PY, EXE = sys.argv[1], sys.argv[2], sys.argv[3], os.path.abspath(sys.argv[4])
ITERS = sys.argv[5] if len(sys.argv) > 5 else "10"
HERE = os.path.dirname(os.path.abspath(__file__))


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def run(name, cmd, out_json):
    base = gpu_used_mb()
    peak = [base]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_used_mb())
            time.sleep(0.2)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    if p.returncode != 0:
        raise SystemExit(f"{name} failed:\n{p.stderr[-3000:]}")
    sys.stderr.write(f"--- {name} ({wall:.0f} s)\n{p.stderr}")
    res = json.load(open(out_json))
    res["_vram_rise_mb"] = peak[0] - base
    return res


rows = []
configs = [
    ("PyTorch CUDA fp32", [PY, os.path.join(HERE, "bench_ref_gpu.py"), MODEL, DATA, os.path.join(DATA, "bench_torch_fp32.json"), "fp32", ITERS], "bench_torch_fp32.json"),
    ("Rust CUDA fp32", [EXE, "--model-dir", MODEL, "--device", "cuda", "bench", "--data-dir", DATA, "--iters", ITERS, "--out", os.path.join(DATA, "bench_rust_fp32.json")], "bench_rust_fp32.json"),
    ("PyTorch CUDA tf32", [PY, os.path.join(HERE, "bench_ref_gpu.py"), MODEL, DATA, os.path.join(DATA, "bench_torch_tf32.json"), "tf32", ITERS], "bench_torch_tf32.json"),
    ("Rust CUDA tf32", [EXE, "--model-dir", MODEL, "--device", "cuda", "--tf32", "bench", "--data-dir", DATA, "--iters", ITERS, "--out", os.path.join(DATA, "bench_rust_tf32.json")], "bench_rust_tf32.json"),
    ("PyTorch CUDA bf16", [PY, os.path.join(HERE, "bench_ref_gpu.py"), MODEL, DATA, os.path.join(DATA, "bench_torch_bf16.json"), "bf16", ITERS], "bench_torch_bf16.json"),
    ("Rust CUDA bf16", [EXE, "--model-dir", MODEL, "--device", "cuda", "--bf16", "bench", "--data-dir", DATA, "--iters", ITERS, "--out", os.path.join(DATA, "bench_rust_bf16.json")], "bench_rust_bf16.json"),
]
results = {}
for name, cmd, out in configs:
    results[name] = run(name, cmd, os.path.join(DATA, out))

cases = ["short (17 tok)", "long (2208 tok)", "batch of 64 short docs", "image 641x479", "audio 9.3 s"]
names = list(results)
print("\nmedian ms (lower is better)")
print(f"{'workload':<24}" + "".join(f"{n:>20}" for n in names))
for c in cases:
    print(f"{c:<24}" + "".join(f"{results[n][c]['median_ms']:>20.1f}" for n in names))
print(f"{'VRAM rise (MB)':<24}" + "".join(f"{results[n]['_vram_rise_mb']:>20}" for n in names))
print("\nRust speed-up vs PyTorch at the same precision:")
for prec in ["fp32", "tf32", "bf16"]:
    t, r = results[f"PyTorch CUDA {prec}"], results[f"Rust CUDA {prec}"]
    print(f"  {prec}: " + ", ".join(f"{c.split(' (')[0]} {t[c]['median_ms'] / r[c]['median_ms']:.2f}x" for c in cases))
for n in names:
    a = results[n].get("_accuracy_vs_fp32_cpu")
    if a:
        print(f"{n} accuracy vs fp32 CPU reference: " + ", ".join(f"{k} cos={v['cos']:.6f}" for k, v in a.items()))
json.dump(results, open(os.path.join(DATA, "bench_gpu_compare.json"), "w"), indent=1)
