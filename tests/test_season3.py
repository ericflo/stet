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
from stet.exam.gen.knowledge import LANGS, TEMPLATES, split_pool_with_ids
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
    # Partition must be by canonical Wikidata subject Q-ID across ALL 10 languages and ALL relations.
    all_exam_qids: set[str] = set()
    all_sft_qids: set[str] = set()
    for fact in TEMPLATES:
        for lang in LANGS:
            exam_pool, _ = split_pool_with_ids(fact, lang, parity=0)
            sft_pool, _ = split_pool_with_ids(fact, lang, parity=1)
            assert len(exam_pool) >= 4 and len(sft_pool) >= 4, (fact, lang)
            exam_qids = {qid for qid, _, _ in exam_pool}
            sft_qids = {qid for qid, _, _ in sft_pool}
            assert exam_qids.isdisjoint(sft_qids), (fact, lang)
            all_exam_qids |= exam_qids
            all_sft_qids |= sft_qids
    assert all_exam_qids and all_sft_qids
    assert all_exam_qids.isdisjoint(all_sft_qids)  # zero cross-language entity leakage

    # Verify concrete s3-p00000 exam and SFT probe share zero canonical Wikidata subject Q-IDs across all languages.
    from stet.spec import practice_round

    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    exam_items = [it for it in gen.for_season(s3, rnd.seed) if it.corner.startswith("knowledge-mc/")]
    assert exam_items and all(it.kind == "choice" and len(it.wrongs) == 3 for it in exam_items)
    assert all(it.cont not in it.wrongs for it in exam_items)
    # Build lookup from rendered prompt -> set of canonical Q-IDs in each split.
    exam_prompt_qids: dict[str, set[str]] = {}
    sft_prompt_qids: dict[str, set[str]] = {}
    for fact in TEMPLATES:
        for lang in LANGS:
            for qid, subj, _ in split_pool_with_ids(fact, lang, parity=0)[0]:
                exam_prompt_qids.setdefault(TEMPLATES[fact][lang].format(s=subj), set()).add(qid)
            for qid, subj, _ in split_pool_with_ids(fact, lang, parity=1)[0]:
                sft_prompt_qids.setdefault(TEMPLATES[fact][lang].format(s=subj), set()).add(qid)
    seen_exam_qids = {q for it in exam_items for q in exam_prompt_qids.get(it.ctx, ())}
    pairs = sft.build_sft_examples(rnd.seed, n=256)
    seen_sft_qids = {q for p, _ in pairs for q in sft_prompt_qids.get(p, ())}
    assert seen_exam_qids and seen_sft_qids
    assert seen_exam_qids.isdisjoint(seen_sft_qids)

    rng = Rng(b"round-seed", "test")
    logic_items = [LogicMC.make(rng, corner) for corner in LogicMC.corners()[:20]]
    assert logic_items and all(it.kind == "choice" and len(it.wrongs) >= 1 for it in logic_items)
    assert all(it.cont not in it.wrongs for it in logic_items)

    # SFT probe generates deterministic multilingual (prompt, completion) pairs without exam access.
    p64 = sft.build_sft_examples(b"round-seed", n=64)
    assert len(p64) == 64
    assert p64 == sft.build_sft_examples(b"round-seed", n=64)
    assert p64 != sft.build_sft_examples(b"other-seed", n=64)
    assert all(isinstance(p, str) and isinstance(c, str) and p and c for p, c in p64)


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


def test_reader_vllm_bounds_long_multilingual_prompts_and_handles_logprob_dataclass(monkeypatch):
    """Regression test for NVIDIA vLLM reader: long multilingual prompts must stay within
    max_model_len while preserving RUBRIC_SUFFIX, request digit logprob_token_ids, and read
    vLLM's Logprob dataclass without triggering eager float(entry) evaluation."""
    import dataclasses
    import importlib.util
    import sys
    import types

    torch = pytest.importorskip("torch")

    spec = importlib.util.spec_from_file_location("reader_curator_test", ROOT / "examples" / "reader" / "curator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    @dataclasses.dataclass
    class FakeLogprob:
        logprob: float
        rank: int = 1
        decoded_token: str = ""

    class FakeTok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # Realistic BPE-like lengths: ~4 ASCII chars/token (~85 tokens for RUBRIC_PREFIX),
            # while non-ASCII bytes are 1 token/byte so 900-char CJK/Cyrillic docs exceed 1,300 tokens.
            ids: list[int] = []
            buf = 0
            for ch in text:
                if ord(ch) < 128:
                    if ch in "12345":
                        if buf:
                            ids.append(500 + buf)
                            buf = 0
                        ids.append(ord(ch))
                    else:
                        buf += 1
                        if buf >= 4:
                            ids.append(500 + (ord(ch) % 200))
                            buf = 0
                else:
                    if buf:
                        ids.append(500 + buf)
                        buf = 0
                    ids.extend(1000 + b for b in ch.encode("utf-8"))
            if buf:
                ids.append(500 + buf)
            return ([1] + ids) if add_special_tokens else ids

    seen_calls: list[dict] = []

    class FakeSamplingParams:
        def __init__(self, *, max_tokens=1, temperature=0.0, logprobs=None, logprob_token_ids=None):
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.logprobs = logprobs
            self.logprob_token_ids = logprob_token_ids

    class FakeLLM:
        def __init__(self, **kwargs):
            self.max_model_len = kwargs["max_model_len"]
            self.tok = FakeTok()

        def get_tokenizer(self):
            return self.tok

        def generate(self, prompts, params, use_tqdm=False):
            seen_calls.append({"prompts": prompts, "params": params, "max_model_len": self.max_model_len})
            assert params.logprob_token_ids is not None and len(params.logprob_token_ids) == 5
            suffix = self.tok.encode(mod.RUBRIC_SUFFIX, add_special_tokens=False)
            prefix = self.tok.encode(mod.RUBRIC_PREFIX, add_special_tokens=True)
            outs = []
            for p in prompts:
                assert isinstance(p, dict) and "prompt_token_ids" in p
                ids = p["prompt_token_ids"]
                # Mirror vLLM v0.34.0 input_processor.py length check strictly.
                assert len(ids) + params.max_tokens <= self.max_model_len
                assert len(ids) <= mod.MAX_PROMPT_TOKENS
                assert ids[: len(prefix)] == prefix
                assert ids[-len(suffix) :] == suffix
                lps = {tid: FakeLogprob(logprob=-0.1 * (i + 1)) for i, tid in enumerate(params.logprob_token_ids)}
                outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(logprobs=[lps])]))
            return outs

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    fake_task = types.SimpleNamespace(device=lambda: "cuda", time_left=lambda: 600.0)
    cands = [
        (1.0, 0, 2000, "日本語の長文ドキュメントです。" * 80),  # > 1,300 UTF-8 bytes/tokens
        (0.9, 1, 1800, "Длинный русский текст с множеством слов. " * 40),
        (0.8, 2, 400, "Short clear English technical explanation."),
    ]
    scores: dict[int, float] = {}
    assert mod._score_vllm(fake_task, "/fake/model", cands, scores) is True
    assert len(scores) == len(cands)
    assert all(1.0 <= v <= 5.0 for v in scores.values())
    assert seen_calls


def test_large_vocab_exam_scoring_matches_full_ce_and_uses_token_budget_batches():
    """Large-vocab scoring must batch by padded token count (<= 2,048 tokens) and match naive
    full-tensor FP32 cross-entropy to numerical precision."""
    import types

    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from stet.exam import score as exam_score

    vocab = 65536
    seen_shapes: list[tuple[int, int]] = []

    class BigVocabLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab)
            self.emb = torch.nn.Embedding(vocab, 8)
            self.proj = torch.nn.Linear(8, vocab, bias=False)

        def forward(self, ids):
            seen_shapes.append(tuple(ids.shape))
            return self.proj(self.emb(ids))

    torch.manual_seed(7)
    model = BigVocabLM().eval()
    # Include the long sequence regime (782..1024) that previously OOM'd a 16 GiB grader at batch=8.
    lengths = [782, 793, 797, 866, 871, 909, 1010, 1024]
    seqs = [([1 + ((i * 13 + t) % 500) for t in range(L)], L - 20) for i, L in enumerate(lengths)]
    got = exam_score.nll(model, seqs, device=torch.device("cpu"), batch=64)

    # Every batch must respect the 2,048 padded-token cap (so T=1024 never runs at bs=8).
    assert all(b * t <= 2048 for b, t in seen_shapes), seen_shapes
    assert max(b for b, _ in seen_shapes) <= 2

    # Numerical equivalence against full-tensor FP32 cross-entropy on the first and last sequences.
    with torch.no_grad():
        for idx in (0, len(seqs) - 1):
            ids, first = seqs[idx]
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor([ids[1:]], dtype=torch.long)
            y[0, : first - 1] = -100
            logits = model(x)
            ref = float(F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none").sum())
            assert got[idx] == pytest.approx(ref, rel=1e-5, abs=1e-4)


def test_season2_v9_exam_is_frozen_bit_for_bit():
    """Season 2 replay must keep exact v9 shell generator hashes (including s2-p00014 write-read #6)."""
    from stet.exam.aggregate import plan_of, round_score
    from stet.exam.items import digest
    from stet.spec import practice_round

    s2 = load_season("s2")
    rnd14 = practice_round(s2, 14)
    s2_items = gen.for_season(s2, rnd14.seed)
    assert digest(s2_items) == "f2cd8de7eb205da5d8d8d73a2524f9c2374d7c9e975bac6fa6ced3e09b5fe6f1"
    wr = [it for it in s2_items if it.corner == "agent/en/tools/write-read"]
    assert wr[6].id == "5195455d2d05d5bf0e047a2d748613e82edefb69184f0d2f497fd826dbed0e5c"
    agg = round_score(s2_items, [0.1] * len(s2_items), plan=plan_of(s2))
    assert agg["score"] == pytest.approx(0.1)


def test_season2_practice_replay_aggregates_with_s2_plan_in_cli_and_gym(corpus, monkeypatch):
    """CLI _with_baseline and gym.Env(season='s2') must aggregate with plan_of(s2), not s3."""
    pytest.importorskip("torch")
    from stet import cli, data, gym
    from stet.spec import practice_round

    s2 = load_season("s2")
    rnd0 = practice_round(s2, 0)
    scale = s2.scale("nano")
    items = [
        Item(kind="answer", corner="agent/shell/ls", fmt="logs", lang="en", source="gen:agent", ctx="$ ls\na\n", cont="a"),
        Item(kind="answer", corner="logic/x", fmt="prose", lang="en", source="gen:logic", ctx="Q?", cont=" Yes"),
    ]
    monkeypatch.setattr("stet.pipeline.baseline", lambda *a, **k: {"exam": {"mean": -5.0, "item_scores": [-5.0, -5.0]}})
    monkeypatch.setattr(data, "round_exam", lambda *a, **k: items)
    res = {"valid": True, "exam": {"mean": -4.8, "item_scores": [-4.7, -4.9]}}
    cli._with_baseline(res, rnd0, scale, None, cli.Out(), skip=False)
    assert res["gain"] == pytest.approx(0.2) and set(res["pillars"]) == {"agents", "logic"}

    monkeypatch.setattr(data, "corpus", lambda *a, **k: corpus)
    monkeypatch.setattr(data, "materialize", lambda *a, **k: (corpus, items))
    monkeypatch.setattr(gym, "baseline", lambda *a, **k: {"exam": {"mean": -5.0, "item_scores": [-5.0, -5.0]}})
    monkeypatch.setattr(
        "stet.grade.grade",
        lambda *a, **k: {
            "valid": True,
            "invalid_reason": None,
            "script": {"out_docs": 1},
            "train": {"tokens": scale.train_tokens},
            "exam": {"items": 2, "mean": -4.8, "item_scores": [-4.7, -4.9]},
        },
    )
    e = gym.Env(season="s2", scale="nano", device="cpu")
    e.reset(0)
    reward, info = e.step([0])
    assert reward == pytest.approx(0.2) and info["gain_by_pillar"] == {"agents": pytest.approx(0.3), "logic": pytest.approx(0.1)}
