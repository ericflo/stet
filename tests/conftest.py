import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_corpus(path: Path, docs: list[str]):
    path.mkdir(parents=True, exist_ok=True)
    blobs = [d.encode("utf-8") for d in docs]
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype("<u8")
    (path / "docs.bin").write_bytes(b"".join(blobs))
    offsets.tofile(path / "docs.idx")
    (path / "urls.txt").write_text("\n".join(f"http://example.test/{i}" for i in range(len(docs))) + "\n")
    (path / "meta.json").write_text(json.dumps({"n_docs": len(docs), "bytes": int(offsets[-1]), "sha256": "test"}))
    from stet.data import Corpus

    return Corpus(path)


@pytest.fixture
def corpus(tmp_path):
    docs = [
        "The quick brown fox jumps over the lazy dog. " * 40,  # 0: long prose
        "short",  # 1: tiny
        "Línea con acentos: áéíóú ñ. " * 30,  # 2: multibyte
        "Navigation | Home | About | Contact\n" + "A real paragraph about something worth reading. " * 30,  # 3
    ]
    return make_corpus(tmp_path / "corpus", docs)


class FakeEnc:
    """One token per whitespace-separated word; enough to test budgets without tiktoken."""

    def encode_ordinary_batch(self, texts):
        return [[(hash(w) % 8000) for w in t.split()] for t in texts]


@pytest.fixture
def enc():
    return FakeEnc()


EXAM_SENTENCE = "The committee met on a rainy Tuesday and agreed, after much argument, to postpone the vote."


@pytest.fixture(autouse=True)
def tiny_exam(monkeypatch):
    """Stand in for the season's exam pack, so tests need no download."""
    from stet import data
    from stet.spec import load_season

    cfg = load_season("s0").raw["decontam"]
    keys = data.keys_array(data.sentence_starts(EXAM_SENTENCE, cfg["key_bytes"], cfg["min_letters"]), cfg["key_bytes"])
    assert len(keys) == 1
    monkeypatch.setitem(data._exam_keys, "s0", keys)
    return keys
