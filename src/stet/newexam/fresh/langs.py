"""How much of a round's corpus is in each language: the shares its fresh items are drawn in.

fastText's published language identifier (lid.176, 176 languages, CC BY-SA 3.0) reads the start
of a seeded sample of the corpus's documents, weighted by length. The model file is pinned by its
sha256.
"""

from __future__ import annotations

from collections import defaultdict

from ...data import cache_dir, fetch
from ..rng import Rng

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
SAMPLE = 6000
READ = 1000  # characters of each sampled document
_model = None


def model():
    global _model
    if _model is None:
        import fasttext

        fasttext.FastText.eprint = lambda *a, **k: None  # it warns about a deprecated API on every load
        path = fetch([MODEL_URL], MODEL_SHA256, cache_dir() / "models" / "lid.176.ftz", "language identifier", quiet=True, max_bytes=4 << 20)
        _model = fasttext.load_model(str(path))
    return _model


def shares(corpus, seed: bytes, sample: int = SAMPLE) -> dict[str, float]:
    """Percent of the corpus's text in each language, from a seeded sample weighted by bytes."""
    n = len(corpus)
    pick = Rng(seed, "fresh/langs").sample(range(n), min(n, sample))
    m = model()
    weight: dict[str, float] = defaultdict(float)
    total = 0.0
    for i in pick:
        text = corpus.text(i)[:READ].replace("\n", " ")
        if not text.strip():
            continue
        (label,), _ = m.predict(text, k=1)
        size = float(corpus.size(i))
        weight[label.removeprefix("__label__")] += size
        total += size
    return {lang: round(100 * w / total, 4) for lang, w in sorted(weight.items(), key=lambda x: -x[1])} if total else {}
