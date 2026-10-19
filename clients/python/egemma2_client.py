"""Python client for `egemma2 serve`: the model stays loaded in one subprocess for all your requests.

Standard library only; numpy and Pillow are used when you pass their objects (or call `embed_numpy`).

    from egemma2_client import Egemma2

    with Egemma2(exe="egemma2", model_dir="models/embeddinggemma-2", precision="bf16") as model:
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
import threading
from array import array
from pathlib import Path

__all__ = ["Egemma2", "Egemma2Error"]

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


class Egemma2:
    """One `egemma2 serve` subprocess. Thread-safe: concurrent calls are sent one at a time."""

    def __init__(self, exe=None, model_dir=None, device="auto", precision=None, max_tokens=None, args=(), stderr=None, cwd=None):
        """
        exe: path to the egemma2 binary (default: $EGEMMA2_EXE, then `egemma2` on PATH).
        model_dir: the model directory (default: the binary's default, models/embeddinggemma-2).
        device: auto, cpu, cuda or cuda:N. precision: None (float32), "tf32" or "bf16" (GPU only).
        max_tokens: cut texts longer than this (default: no cut, like sentence-transformers).
        args: extra options for `egemma2 serve` (e.g. ["--prompt-name", "Document"]).
        stderr: where the server's log goes (default: this process's stderr).
        """
        exe = exe or os.environ.get("EGEMMA2_EXE") or shutil.which("egemma2")
        if not exe:
            raise FileNotFoundError("egemma2 not found: pass exe=..., set EGEMMA2_EXE or put it on PATH")
        exe = os.fspath(exe)
        # A relative path such as target/release/egemma2 is not searched for by Windows' CreateProcess.
        exe = os.path.abspath(exe) if os.path.isfile(exe) else (shutil.which(exe) or exe)
        cmd = [exe, "--device", device]
        if model_dir is not None:
            cmd += ["--model-dir", os.fspath(model_dir)]
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
