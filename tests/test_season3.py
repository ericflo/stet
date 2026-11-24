"""Season 3 unit tests: contrastive choice scoring, contrastive fresh selection,
disjoint SFT probe, uint32 large-vocab scripts, top-half layer reset, cuda_linux_only,
and the Qwen3-0.6B vLLM reference reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stet import env
from stet.exam import gen, sft
from stet.exam.fresh import Doc
from stet.exam.fresh.select import select_contrastive
from stet.exam.gen.knowledge import TEMPLATES, KnowledgeMC, split_pool
from stet.exam.gen.logic import LogicMC
from stet.exam.items import Item
from stet.exam.rng import Rng
from stet.project import load_project
from stet.script import materialize
from stet.spec import load_season

ROOT = Path(__file__).resolve().parents[1]


def test_choice_item_serialization_and_validation():
    it = Item(
        kind="choice",
        corner="fresh/wikipedia/en/Q1/choice",
        fmt="prose",
        lang="en",
        source="wikipedia:en:1",
        ctx="First half of an article about bridges.",
        cont=" Second half continuing the bridge discussion.",
        wrongs=(
            " Second half about medieval pottery instead.",
            " Second half about ocean currents instead.",
        ),
    )
    again = Item.from_dict(it.to_dict())
    assert again == it
    assert again.id == it.id
    with pytest.raises(ValueError, match="wrong alternatives"):
        Item(kind="choice", corner="c", fmt="prose", lang="en", source="s", ctx="a", cont="b", wrongs=())


def test_select_contrastive_emits_only_choice_items_with_same_source_lang_distractors():
    docs = [
        Doc(
            source="wikipedia",
            id=f"en:{i}",
            url=f"https://en.wikipedia.org/wiki/Page_{i}",
            lang="en",
            license="CC BY-SA 4.0",
            published=1_800_000_000 + i,
            title=f"Topic {i}",
            text=(
                f"Paragraph one of topic {i} explains the early history in clear detail. "
                f"Researchers documented how topic {i} evolved across several decades of study. "
                f"Later developments in topic {i} introduced entirely new techniques and results. "
                f"Modern applications of topic {i} continue to grow in surprising directions."
            ),
            topic=f"Q{i}",
        )
        for i in range(12)
    ]
    docs += [
        Doc(
            source="wikipedia",
            id=f"de:{i}",
            url=f"https://de.wikipedia.org/wiki/Seite_{i}",
            lang="de",
            license="CC BY-SA 4.0",
            published=1_800_000_100 + i,
            title=f"Thema {i}",
            text=(
                f"Der erste Absatz zu Thema {i} beschreibt die Grundlagen sehr genau. "
                f"Forscher untersuchten die Entwicklung von Thema {i} ueber viele Jahre hinweg. "
                f"Spaetere Arbeiten zu Thema {i} brachten neue Methoden und Erkenntnisse hervor. "
                f"Heutige Anwendungen von Thema {i} zeigen deutliche Fortschritte im Alltag."
            ),
            topic=f"QD{i}",
        )
        for i in range(6)
    ]
    items = select_contrastive(docs, b"s3-seed", 20, {"reference"}, shares={"en": 60.0, "de": 40.0})
    assert len(items) > 0
    assert all(it.kind == "choice" for it in items)
    assert all(1 <= len(it.wrongs) <= 3 for it in items)
    # Every distractor must come from a different doc of the SAME language (never English mixed into German).
    for it in items:
        assert it.cont not in it.wrongs
        if it.lang == "de":
            assert all("Thema" in w for w in it.wrongs)
        else:
            assert all("topic" in w for w in it.wrongs)
    # Deterministic on same seed & docs (even when input order is reversed).
    assert select_contrastive(docs[::-1], b"s3-seed", 20, {"reference"}, shares={"en": 60.0, "de": 40.0}) == items


def test_knowledge_mc_and_sft_probe_have_zero_fact_overlap():
    # Even parity = exam, odd parity = SFT probe. Verify strict disjointness across all relations/langs.
    for fact in TEMPLATES:
        exam_pool, _ = split_pool(fact, "en", parity=0)
        sft_pool, _ = split_pool(fact, "en", parity=1)
        assert exam_pool and sft_pool
        exam_subjects = {s for s, _ in exam_pool}
        sft_subjects = {s for s, _ in sft_pool}
        assert exam_subjects.isdisjoint(sft_subjects), fact

    rng = Rng(b"round-seed", "test")
    exam_items = [KnowledgeMC.make(rng, corner) for corner in KnowledgeMC.corners()[:20]]
    assert exam_items and all(it.kind == "choice" and len(it.wrongs) == 3 for it in exam_items)
    assert all(it.cont not in it.wrongs for it in exam_items)

    logic_items = [LogicMC.make(rng, corner) for corner in LogicMC.corners()[:20]]
    assert logic_items and all(it.kind == "choice" and len(it.wrongs) >= 1 for it in logic_items)
    assert all(it.cont not in it.wrongs for it in logic_items)

    # SFT probe generates deterministic multilingual (prompt, completion) pairs without exam access.
    pairs = sft.build_sft_examples(b"round-seed", n=64)
    assert len(pairs) == 64
    assert pairs == sft.build_sft_examples(b"round-seed", n=64)
    assert pairs != sft.build_sft_examples(b"other-seed", n=64)
    assert all(isinstance(p, str) and isinstance(c, str) and p and c for p, c in pairs)


def test_s3_generated_exam_uses_contrastive_logic_and_knowledge():
    s3 = load_season("s3")
    items = gen.for_season(s3, b"s3-test-seed")
    parts = {it.corner.split("/", 1)[0] for it in items}
    assert "logic-mc" in parts and "knowledge-mc" in parts
    assert "logic" not in parts and "knowledge" not in parts
    for it in items:
        part = it.corner.split("/", 1)[0]
        if part in ("logic-mc", "knowledge-mc"):
            assert it.kind == "choice" and len(it.wrongs) >= 1


def test_uint32_script_materialization_preserves_large_vocab_token_ids(tmp_path):
    """Qwen3's 151,936 vocab exceeds uint16 (65535); materialize must use uint32 without wraparound."""
    from conftest import make_corpus

    from stet.spec import Scale

    class BigVocabEnc:
        n_vocab = 151_936

        def encode_ordinary(self, text: str) -> list[int]:
            # Map characters to token IDs well above 65535.
            return [70_000 + ord(c) for c in text]

        def encode_ordinary_batch(self, texts: list[str]) -> list[list[int]]:
            return [self.encode_ordinary(t) for t in texts]

    s3 = load_season("s3")
    tiny = Scale(name="tiny", corpus_units=1, train_tokens=512, n_layer=2, n_head=2, d_model=64, ctx=16, batch_size=2, lr=1e-3, edit_seconds=60)
    corpus = make_corpus(tmp_path / "corpus", ["Hello world! " * 20, "Multilingual test. " * 20])
    script = tmp_path / "script.jsonl"
    script.write_text("0\n1\n")
    tokens, stats = materialize(corpus, script, s3, tiny, enc=BigVocabEnc())
    assert tokens.dtype == np.uint32
    assert tokens.shape == (2, 16 * 16 + 1)
    assert int(tokens.max()) == s3.eot == 151_643
    assert int(tokens[0, 0]) == s3.eot
    assert int(tokens[0, 1]) == 70_000 + ord("H")
    assert stats.out_docs == 2


def test_choice_margin_tok_is_invariant_to_uniform_style_shift():
    """Adding a constant bias to all continuation logprobs cancels out in Right - best Wrong."""
    torch = pytest.importorskip("torch")
    from stet.exam import score as exam_score

    class FakeEnc:
        def encode_ordinary(self, text: str) -> list[int]:
            return [1 + (ord(c) % 30) for c in text]

    class BiasedLM(torch.nn.Module):
        def __init__(self, target_tok: int, style_shift: float):
            super().__init__()
            self.target_tok = target_tok
            self.style_shift = style_shift
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, ids):
            b, t = ids.shape
            logits = torch.full((b, t, 32), self.style_shift, dtype=torch.float32)
            logits[:, :, self.target_tok] = self.style_shift + 2.0
            return logits

    enc = FakeEnc()
    right_tok = enc.encode_ordinary("A")[0]
    it = Item(
        kind="choice",
        corner="fresh/wikipedia/en/Q1/choice",
        fmt="prose",
        lang="en",
        source="wikipedia:en:1",
        ctx="Context text.",
        cont="AAAA",
        wrongs=("BBBB", "CCCC"),
    )
    m0 = BiasedLM(right_tok, style_shift=0.0)
    m1 = BiasedLM(right_tok, style_shift=5.0)
    s0 = exam_score.sit(m0, [it], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    s1 = exam_score.sit(m1, [it], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    assert s0 > 0.0
    assert s0 == pytest.approx(s1, abs=1e-5)


def test_reset_top_layers_preserves_bottom_half_and_is_seed_deterministic():
    torch = pytest.importorskip("torch")
    from stet.train import _reset_top_layers

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.norm = torch.nn.LayerNorm(8)

    class TinyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([TinyBlock() for _ in range(4)])

    def make_ones():
        m = TinyHFModel()
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(1.0)
        return m

    m1, m2, m3 = make_ones(), make_ones(), make_ones()
    _reset_top_layers(m1, reset_frac=0.5, init_seed=42)
    _reset_top_layers(m2, reset_frac=0.5, init_seed=42)
    _reset_top_layers(m3, reset_frac=0.5, init_seed=99)

    # Bottom 2 layers untouched (still all 1.0).
    for idx in (0, 1):
        assert torch.all(m1.model.layers[idx].proj.weight == 1.0)
    # Top 2 layers reset and deterministic for same seed, distinct for different seed.
    for idx in (2, 3):
        assert not torch.all(m1.model.layers[idx].proj.weight == 1.0)
        assert torch.equal(m1.model.layers[idx].proj.weight, m2.model.layers[idx].proj.weight)
        assert not torch.equal(m1.model.layers[idx].proj.weight, m3.model.layers[idx].proj.weight)


def test_cuda_linux_only_packages_filtered_on_non_nvidia(monkeypatch):
    s3 = load_season("s3")
    cuda_only = set(s3.raw["environment"]["cuda_linux_only"])
    assert {"vllm", "liger-kernel"} <= cuda_only

    monkeypatch.setattr(env, "is_nvidia_linux", lambda: True)
    assert cuda_only <= set(env.expected(s3)["packages"])

    monkeypatch.setattr(env, "is_nvidia_linux", lambda: False)
    assert not (cuda_only & set(env.expected(s3)["packages"]))


def test_reader_example_pins_qwen3_06b_and_writes_fallback_first():
    s3 = load_season("s3")
    root = ROOT / "examples" / "reader"
    project = load_project(root)
    assert project.name == "reader" and project.entry == "python curator.py"
    assert project.assets and all(a.path.startswith("qwen3-0.6b/") for a in project.assets)
    assert sum(a.size for a in project.assets) <= s3.raw["assets"]["max_bytes"]
    assert len({a.url.split("/resolve/")[1].split("/")[0] for a in project.assets}) == 1
    src = (root / "curator.py").read_text()
    compile(src, "examples/reader/curator.py", "exec")
    main_src = src[src.index("def main()") :]
    assert main_src.index("task.write(") < main_src.index("_score_vllm(")
