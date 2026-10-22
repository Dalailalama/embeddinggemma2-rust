"""EmbeddingGemma 2 in Rust: text, image, audio and video embeddings in one 768-d space, on CPU or CUDA.

`pip install egemma2` installs the `egemma2` command-line binary and this Python client for it:

    from egemma2 import Egemma2

    with Egemma2(precision="bf16") as model:
        vectors = model.embed(["a query", "a document"], prompt_name="Document")

See https://github.com/Dalailalama/embeddinggemma2-rust for the documentation.
"""

from .client import Egemma2, Egemma2Error, find_egemma2_bin, find_model_dir

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("egemma2")
except (ImportError, PackageNotFoundError):  # used from a source checkout
    __version__ = "0+unknown"

__all__ = ["Egemma2", "Egemma2Error", "find_egemma2_bin", "find_model_dir", "__version__"]
