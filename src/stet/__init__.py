"""Stet: a competition for agents that edit pretraining data."""

import os

# Must be set before PyTorch initializes cuBLAS so deterministic CUDA GEMMs work reliably.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .spec import HARNESS_VERSION as __version__

__all__ = ["__version__", "source_digest"]

# The files that decide a score. Their combined hash goes into every result, so that when
# two machines disagree the first question ("were they running the same code?") is answered.
SCORING_SOURCES = ("spec.py", "data.py", "script.py", "tok.py", "model.py", "train.py", "grade.py", "exam/items.py", "exam/score.py", "exam/sft.py")
_digest = None


def source_digest() -> str:
    global _digest
    if _digest is None:
        import hashlib
        from importlib import resources

        h = hashlib.sha256()
        for name in SCORING_SOURCES:
            h.update(name.encode())
            h.update(resources.files("stet").joinpath(*name.split("/")).read_bytes().replace(b"\r\n", b"\n"))
        _digest = h.hexdigest()
    return _digest
