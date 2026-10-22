"""Python client for `egemma2 serve`: the model stays loaded in one subprocess for all your requests.

Standard library only; numpy and Pillow are used when you pass their objects (or call `embed_numpy`), and
huggingface_hub, if installed, to download the model. `pip install egemma2` puts this module and the egemma2
binary it drives into the environment; the file also works on its own next to any egemma2 binary.

    from egemma2 import Egemma2

    with Egemma2(precision="bf16") as model:      # CUDA if available, else the CPU
        docs = model.embed(["first document", "second document"], prompt_name="Document")
        query = model.embed("what is in the documents?", prompt_name="SearchQuery")[0]
        pages = model.embed([Path("page1.png"), Path("page2.png")])    # files, by extension
        mixed = model.embed([{"text": "a photo of", "image": pil_image}])  # one interleaved input

Inputs: a `str` is text. A `pathlib.Path` is a file chosen by its extension (image, .wav audio, or a video
file / frame directory). Images can also be Pillow images, HxWx3 uint8 numpy arrays or encoded bytes. A dict
such as `{"text": ..., "image": ...}` is ONE input whose parts follow the dict's order (like
sentence-transformers); `{"parts": [...]}` takes any sequence of parts. Audio given as data is
`(samples, sample_rate)`; a video can be a list of frames, with `{"video": frames, "fps": 30}`.

The server is started with `egemma2 [options] serve` and talks JSON lines over its stdin/stdout; see the
README's "Server mode" section. For an HTTP server instead, run `egemma2 serve --http 127.0.0.1:8080` and use
any OpenAI-compatible client.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import threading
from array import array
from pathlib import Path

__all__ = ["Egemma2", "Egemma2Error", "find_egemma2_bin", "find_model_dir"]

MODEL_REPO = "google/embeddinggemma-2"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".wave"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


class Egemma2Error(RuntimeError):
    """The server rejected a request, or stopped."""


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _image(v):
    if isinstance(v, (str, os.PathLike)):
        s = os.fspath(v)
        return s if s.startswith("data:") else str(Path(s).resolve())
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"base64": _b64(bytes(v))}
    if hasattr(v, "convert") and hasattr(v, "tobytes"):  # Pillow image
        rgb = v.convert("RGB")
        return {"rgb": _b64(rgb.tobytes()), "width": rgb.width, "height": rgb.height}
    shape = getattr(v, "shape", None)
    if shape is not None and len(shape) == 3 and shape[2] == 3 and str(getattr(v, "dtype", "")) == "uint8":
        import numpy as np

        return {"rgb": _b64(np.ascontiguousarray(v).tobytes()), "width": int(shape[1]), "height": int(shape[0])}
    raise TypeError(f"unsupported image value: {type(v).__name__}")


def _audio(v):
    if isinstance(v, (str, os.PathLike)):
        return str(Path(v).resolve())
    if isinstance(v, (bytes, bytearray, memoryview)):  # an encoded .wav file
        return {"base64": _b64(bytes(v))}
    if isinstance(v, tuple) and len(v) == 2:  # (samples, sample_rate)
        samples, rate = v
        if hasattr(samples, "astype"):
            raw = samples.astype("<f4").tobytes()
        else:
            a = array("f", samples)
            if sys.byteorder == "big":
                a.byteswap()
            raw = a.tobytes()
        return {"pcm_f32": _b64(raw), "sample_rate": int(rate)}
    raise TypeError(f"unsupported audio value: {type(v).__name__} (use a path, wav bytes or (samples, sample_rate))")


def _video(v):
    if isinstance(v, (str, os.PathLike)):
        return str(Path(v).resolve())
    if isinstance(v, (list, tuple)):
        return [_image(f) for f in v]
    raise TypeError(f"unsupported video value: {type(v).__name__} (use a path or a list of frames)")


def _path_item(p):
    p = Path(p)
    ext = p.suffix.lower()
    if p.is_dir() or ext in VIDEO_EXTS:
        return {"video": _video(p)}
    if ext in AUDIO_EXTS:
        return {"audio": _audio(p)}
    if ext in IMAGE_EXTS:
        return {"image": _image(p)}
    raise ValueError(f"cannot tell what {p} is from its extension; pass {{'image': ...}}, {{'audio': ...}} or {{'video': ...}}")


_CONVERT = {"image": _image, "image_url": _image, "audio": _audio, "video": _video}


def _item(x):
    if isinstance(x, str):
        return x
    if isinstance(x, os.PathLike):
        return _path_item(x)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():  # dicts keep insertion order, which is the parts' order
            if k in _CONVERT:
                out[k] = _CONVERT[k](v)
            elif k in ("parts", "content"):
                out[k] = [_item(p) for p in v]
            else:
                out[k] = v
        return out
    try:
        return {"image": _image(x)}
    except TypeError:
        raise TypeError(f"unsupported input: {type(x).__name__}") from None


def _floats(v):
    if isinstance(v, str):
        a = array("f")
        a.frombytes(base64.b64decode(v))
        if sys.byteorder == "big":
            a.byteswap()
        return a
    return v


def find_egemma2_bin():
    """The egemma2 binary: $EGEMMA2_EXE, the one a `pip install egemma2` put next to this Python, or the PATH."""
    env = os.environ.get("EGEMMA2_EXE")
    if env:
        return env
    name = "egemma2" + (sysconfig.get_config_var("EXE") or (".exe" if os.name == "nt" else ""))
    dirs = [sysconfig.get_path("scripts")]
    user_scheme = getattr(sysconfig, "get_preferred_scheme", lambda _: "nt_user" if os.name == "nt" else "posix_user")("user")
    try:
        dirs.append(sysconfig.get_path("scripts", scheme=user_scheme))  # pip install --user
    except KeyError:
        pass
    dirs.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin"))  # pip install --target
    for d in dirs:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    found = shutil.which("egemma2")
    if found:
        return found
    raise FileNotFoundError("egemma2 not found: `pip install egemma2`, pass exe=..., set EGEMMA2_EXE or put it on PATH")


def find_model_dir():
    """The model directory: $EGEMMA2_MODEL_DIR, ./models/embeddinggemma-2, or a download of google/embeddinggemma-2
    into the Hugging Face cache (needs huggingface_hub: `pip install "egemma2[hub]"`; about 1.5 GB, once)."""
    env = os.environ.get("EGEMMA2_MODEL_DIR")
    if env:
        return env
    local = Path("models") / "embeddinggemma-2"
    if (local / "model.safetensors").is_file():
        return str(local.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise FileNotFoundError(
            "no model found: pass model_dir=..., set EGEMMA2_MODEL_DIR, install huggingface_hub to download it "
            f'automatically (pip install "egemma2[hub]"), or run: hf download {MODEL_REPO} --local-dir models/embeddinggemma-2'
        ) from None
    return snapshot_download(MODEL_REPO, allow_patterns=["*.json", "*.jinja", "model.safetensors"])


class Egemma2:
    """One `egemma2 serve` subprocess. Thread-safe: concurrent calls are sent one at a time."""

    def __init__(self, exe=None, model_dir=None, device="auto", precision=None, max_tokens=None, args=(), stderr=None, cwd=None):
        """
        exe: path to the egemma2 binary (default: see `find_egemma2_bin`).
        model_dir: the model directory (default: see `find_model_dir`).
        device: auto, cpu, cuda or cuda:N. precision: None (float32), "tf32" or "bf16" (GPU only).
        max_tokens: cut texts longer than this (default: no cut, like sentence-transformers).
        args: extra options for `egemma2 serve` (e.g. ["--prompt-name", "Document"]).
        stderr: where the server's log goes (default: this process's stderr).
        """
        exe = os.fspath(exe) if exe is not None else find_egemma2_bin()
        # A relative path such as target/release/egemma2 is not searched for by Windows' CreateProcess.
        exe = os.path.abspath(exe) if os.path.isfile(exe) else (shutil.which(exe) or exe)
        model_dir = os.path.abspath(os.fspath(model_dir)) if model_dir is not None else find_model_dir()
        cmd = [exe, "--device", device, "--model-dir", model_dir]
        if precision in ("bf16", "tf32"):
            cmd.append("--" + precision)
        elif precision not in (None, "fp32", "f32", "float32"):
            raise ValueError(f"precision must be None, 'tf32' or 'bf16' (got {precision!r})")
        if max_tokens is not None:
            cmd += ["--max-tokens", str(int(max_tokens))]
        cmd += ["serve", *args]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, cwd=cwd)
        self._lock = threading.Lock()
        self._next_id = 0

    def embed(self, inputs, *, prompt_name=None, prompt=None, dimensions=None):
        """Embed one input or a list of inputs; returns one list of floats (L2-normalised) per input.

        prompt_name: a sentence-transformers prompt such as "SearchQuery" or "Document" (or a literal
        `prompt` string), placed before the whole input. dimensions: Matryoshka size (768, 512, 256, 128).
        """
        single = isinstance(inputs, (str, dict, os.PathLike)) or not isinstance(inputs, (list, tuple))
        items = [_item(inputs)] if single else [_item(x) for x in inputs]
        if not items:
            return []
        req = {"input": items, "encoding_format": "base64"}
        if prompt_name is not None:
            req["prompt_name"] = prompt_name
        if prompt is not None:
            req["prompt"] = prompt
        if dimensions is not None:
            req["dimensions"] = int(dimensions)
        resp = self._call(req)
        return [list(_floats(d["embedding"])) for d in sorted(resp["data"], key=lambda d: d["index"])]

    def embed_numpy(self, inputs, **kwargs):
        """Like `embed`, as a float32 numpy array of shape (n, dim)."""
        import numpy as np

        return np.asarray(self.embed(inputs, **kwargs), dtype=np.float32)

    def _call(self, req):
        with self._lock:
            if self._proc.poll() is not None:
                raise Egemma2Error(f"the egemma2 server has exited (code {self._proc.returncode})")
            self._next_id += 1
            req["id"] = self._next_id
            try:
                self._proc.stdin.write(json.dumps(req).encode("utf-8") + b"\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except (BrokenPipeError, OSError) as e:
                raise Egemma2Error(f"the egemma2 server stopped: {e}") from None
        if not line:
            raise Egemma2Error(f"the egemma2 server stopped (exit code {self._proc.wait()}); see its log above")
        resp = json.loads(line)
        if "error" in resp:
            raise Egemma2Error(resp["error"]["message"])
        return resp

    def close(self):
        """Stop the server (it exits when its stdin closes)."""
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._proc.stdout is not None:
            self._proc.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
