"""End-to-end test of egemma2_client against the PyTorch reference embeddings in testdata/ (CPU, float32).

    python -m unittest clients/python/test_egemma2_client.py

Needs a built binary (EGEMMA2_EXE, or target/release/egemma2) and the model in models/embeddinggemma-2.
Pillow and numpy are optional; the cases that need them are skipped without.
"""

import json
import math
import os
import sys
import unittest
import wave
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egemma2_client import Egemma2, Egemma2Error  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "testdata"
MODEL = ROOT / "models" / "embeddinggemma-2"
EXE = os.environ.get("EGEMMA2_EXE") or str(ROOT / "target" / "release" / ("egemma2.exe" if os.name == "nt" else "egemma2"))
TOL = 1e-5  # float32 CPU vs the PyTorch float32 CPU reference


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def max_diff(a, b):
    assert len(a) == len(b), (len(a), len(b))
    return max(abs(x - y) for x, y in zip(a, b))


@unittest.skipUnless(Path(EXE).exists() and (MODEL / "model.safetensors").exists(), "needs the egemma2 binary and the model")
class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = Egemma2(exe=EXE, model_dir=MODEL, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.m.close()

    def close(self, got, want, what):
        d = max_diff(got, want)
        self.assertLess(d, TOL, f"{what}: max|d| = {d:.2e}")

    def test_texts_match_reference(self):
        refs = load(DATA / "ref_text.json")
        for prompt in {r["prompt"] for r in refs}:
            group = [r for r in refs if r["prompt"] == prompt]
            got = self.m.embed([r["text"] for r in group], prompt_name=prompt)
            for g, r in zip(got, group):
                self.close(g, r["emb"], f"text with prompt {prompt}")
        r = refs[0]
        got = self.m.embed(r["text"], prompt_name=r["prompt"], dimensions=256)[0]
        self.close(got, r["emb256"], "Matryoshka 256")
        self.assertAlmostEqual(math.fsum(x * x for x in got), 1.0, places=5)

    def test_images_match_reference(self):
        cases = load(DATA / "image" / "ref_image.json")["cases"]
        a = cases[0]
        png = Path(a["png"])
        self.close(self.m.embed(png)[0], a["emb_image"], "image path")
        self.close(self.m.embed({"image": png.read_bytes()})[0], a["emb_image"], "image bytes")
        text_image, image_text = self.m.embed([{"text": "a photo of", "image": png}, {"image": png, "text": "a photo of"}])
        self.close(text_image, a["emb_text_image"], "text then image")
        self.close(image_text, a["emb_image_text"], "image then text (dict order)")
        got = self.m.embed([Path(c["png"]) for c in cases])
        for g, c in zip(got, cases):
            self.close(g, c["emb_image"], f"batched image {c['name']}")

    def test_pillow_and_numpy_images(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("needs Pillow and numpy")
        a = load(DATA / "image" / "ref_image.json")["cases"][0]
        img = Image.open(a["png"]).convert("RGB")
        self.close(self.m.embed(img)[0], a["emb_image"], "Pillow image (raw RGB)")
        arr = self.m.embed_numpy([np.asarray(img)])
        self.assertEqual(arr.shape, (1, 768))
        self.close(arr[0].tolist(), a["emb_image"], "numpy image")

    def test_audio_matches_reference(self):
        c = [c for c in load(DATA / "audio" / "ref_audio.json")["cases"] if c["name"] == "a_3p7s"][0]
        self.close(self.m.embed(Path(c["wav"]))[0], c["emb_audio"], "audio path")
        with wave.open(c["wav"]) as w:
            assert w.getsampwidth() == 2 and w.getnchannels() == 1
            raw, rate = w.readframes(w.getnframes()), w.getframerate()
        samples = [int.from_bytes(raw[i:i + 2], "little", signed=True) / 32768.0 for i in range(0, len(raw), 2)]
        self.close(self.m.embed({"audio": (samples, rate)})[0], c["emb_audio"], "audio samples")

    def test_errors_keep_the_server_alive(self):
        with self.assertRaises(Egemma2Error) as e:
            self.m.embed({"image": Path("no-such-file.png")})
        self.assertIn("no-such-file.png", str(e.exception))
        with self.assertRaises(Egemma2Error):
            self.m.embed("x", prompt_name="NoSuchPrompt")
        self.assertEqual(len(self.m.embed("still alive")[0]), 768)


if __name__ == "__main__":
    unittest.main()
