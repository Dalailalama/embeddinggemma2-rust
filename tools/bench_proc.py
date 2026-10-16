"""Cold start (launch -> one text embedding -> exit) and peak working set for both implementations.

Measures the whole process tree (a Windows venv python.exe is a launcher that spawns the real
interpreter as a child process).
"""
import os, subprocess, sys, time, statistics, psutil


def tree_peak(p, peaks):
    procs = [p]
    try:
        procs += p.children(recursive=True)
    except psutil.Error:
        pass
    for q in procs:
        try:
            mi = q.memory_info()
            peaks[q.pid] = max(peaks.get(q.pid, 0), getattr(mi, "peak_wset", mi.rss))
        except psutil.Error:
            pass


def run(cmd, reps=3):
    walls, peaks_mb = [], []
    for _ in range(reps):
        t = time.perf_counter()
        p = psutil.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        peaks = {}
        while p.poll() is None:
            tree_peak(p, peaks)
            time.sleep(0.005)
        walls.append(time.perf_counter() - t)
        peaks_mb.append(max(peaks.values(), default=0) / 2**20)
        assert p.returncode == 0, cmd
    return statistics.median(walls), max(peaks_mb)


model, py, exe = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"   # cpu or cuda
text = "task: search result | query: which planet is known as the red planet"
for name, cmd in [
    (f"PyTorch + sentence-transformers ({device})", [py, "tools/cold_py.py", model, device]),
    (f"Rust egemma2 ({device})", [exe, "--model-dir", model, "--device", device, "embed", "--text", text]),
]:
    wall, peak = run(cmd)
    print(f"{name:<34} cold start {wall:6.2f} s   peak working set {peak:7.0f} MB")
