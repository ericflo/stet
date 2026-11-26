"""Season 3 unit tests: twin-context comprehension scoring, fresh context selection,
disjoint SFT probe, uint32 large-vocab scripts, hf_topreset FP32 master weights + SFT +
clean return, HF tokenizer control-token safety + SHA-256 verification, cuda_linux_only,
and the Qwen3-0.6B vLLM reference reader."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from stet import data, env
from stet.exam import gen, sft
from stet.exam.fresh import Doc
from stet.exam.fresh.select import select_context
from stet.exam.gen.knowledge import LANGS, TEMPLATES, split_pool_with_ids
from stet.exam.items import Item
from stet.project import load_project
from stet.script import materialize
from stet.spec import Scale, load_season, practice_round

ROOT = Path(__file__).resolve().parents[1]


def test_context_item_serialization_and_validation():
    it = Item(
        kind="context",
        corner="fresh/wikipedia/en/Q1/context",
        fmt="prose",
        lang="en",
        source="wikipedia:en:1",
        ctx="First half of an article about suspension bridges.",
        cont=" Second half continuing the bridge discussion.",
        wrongs=(
            "First half of an article about medieval pottery.",
            "First half of an article about ocean currents.",
        ),
    )
    again = Item.from_dict(it.to_dict())
    assert again == it
    assert again.id == it.id
    with pytest.raises(ValueError, match="wrong contexts"):
        Item(kind="context", corner="c", fmt="prose", lang="en", source="s", ctx="a", cont="b", wrongs=())
    with pytest.raises(ValueError, match="differ from the real context"):
        Item(kind="context", corner="c", fmt="prose", lang="en", source="s", ctx="same", cont="b", wrongs=("same",))


def test_select_context_emits_only_context_items_with_same_source_lang_and_clean_boundaries():
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
    # Unpunctuated single-sentence text must be skipped (no mid-sentence cuts).
    docs.append(
        Doc(
            source="wikipedia",
            id="en:nopunct",
            url="https://en.wikipedia.org/wiki/NoPunct",
            lang="en",
            license="CC BY-SA 4.0",
            published=1_800_000_999,
            title="NoPunct",
            text=" ".join(f"word{k}" for k in range(120)),
            topic="QNOPUNCT",
        )
    )
    items = select_context(docs, b"s3-seed", 20, {"reference"}, shares={"en": 60.0, "de": 40.0})
    assert len(items) > 0
    assert all(it.kind == "context" for it in items)
    assert all(1 <= len(it.wrongs) <= 3 for it in items)
    assert all(it.source != "wikipedia:en:nopunct" for it in items)
    # Every wrong opening must come from a different doc of the SAME language and matching boundary class.
    for it in items:
        assert it.ctx not in it.wrongs
        assert all(w.endswith("\n") == it.ctx.endswith("\n") for w in it.wrongs)
        if it.lang == "de":
            assert all("Thema" in w for w in it.wrongs)
        else:
            assert all("topic" in w for w in it.wrongs)
            assert it.cont.startswith(" ")  # prose BPE boundary space sits on cont
    # Deterministic on same seed & docs (even when input order is reversed).
    assert select_context(docs[::-1], b"s3-seed", 20, {"reference"}, shares={"en": 60.0, "de": 40.0}) == items


def test_knowledge_even_and_sft_probe_have_zero_fact_overlap():
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

    # Verify bundled s3-sft-ml.jsonl.zst loads 512 deterministic commercial-permissive pairs across 11 languages.
    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    assert gen.for_season(s3, rnd.seed) == []
    s3_pairs = sft.build_sft_examples(rnd.seed, n=512, season=s3)
    assert len(s3_pairs) == 512
    assert s3_pairs == sft.build_sft_examples(rnd.seed, n=512, season=s3)
    assert s3_pairs != sft.build_sft_examples(b"other-seed", n=512, season=s3)
    assert all(isinstance(p, str) and isinstance(c, str) and p and c for p, c in s3_pairs)

    # Synthetic fallback SFT generator (used when no pack is set) remains deterministic and disjoint from knowledge-even.
    p64 = sft.build_sft_examples(b"round-seed", n=64)
    assert len(p64) == 64
    assert p64 == sft.build_sft_examples(b"round-seed", n=64)
    assert p64 != sft.build_sft_examples(b"other-seed", n=64)
    assert all(isinstance(p, str) and isinstance(c, str) and p and c for p, c in p64)


def test_s3_is_100pct_fresh_and_offline_practice_pack_has_both_groups():
    from stet.exam.aggregate import plan_of, round_score
    from stet.exam.fresh.reason import load_practice_pack

    s3 = load_season("s3")
    assert gen.for_season(s3, b"s3-test-seed") == []
    pack = load_practice_pack(s3, b"s3-test-seed")
    assert len(pack) == 509
    assert pack == load_practice_pack(s3, b"s3-test-seed")
    assert pack != load_practice_pack(s3, b"other-seed")
    assert all(it.kind in ("context", "answer") for it in pack)
    assert not any(it.kind == "passage" for it in pack)
    # Scores cleanly under Season 3's 50/50 comprehension + reasoning plan.
    agg = round_score(pack, [0.05] * len(pack), plan=plan_of(s3))
    assert agg["score"] == pytest.approx(0.05)
    assert {"reference", "explain", "infer", "synthesis"} <= set(agg["pillars"])


def test_uint32_script_materialization_preserves_large_vocab_token_ids(tmp_path):
    """ERNIE-4.5's 103,424 vocab exceeds uint16 (65535); materialize must use uint32 without wraparound."""
    from conftest import make_corpus

    class BigVocabEnc:
        n_vocab = 103_424

        def encode_ordinary(self, text: str) -> list[int]:
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
    assert int(tokens.max()) == 70_000 + ord("w")
    assert int(tokens[0, 0]) == s3.eot == 2
    assert int(tokens[0, 1]) == 70_000 + ord("H")
    assert stats.out_docs == 2


def test_context_twin_mean_bpb_cancels_nontrivial_style_boost_and_zeros_uniform_lm():
    """Boosting the continuation's unconditional probability shifts raw answer bpb strongly,
    while twin-context (kind='context') cancels it to floating-point precision and still
    rewards true context-dependent prediction. A uniform LM scores exactly 0."""
    torch = pytest.importorskip("torch")
    from stet.exam import score as exam_score

    class FakeEnc:
        def encode_ordinary(self, text: str) -> list[int]:
            return [1 + (ord(c) % 30) for c in text]

    enc = FakeEnc()
    cont_tok = enc.encode_ordinary("A")[0]
    cue_tok = enc.encode_ordinary("R")[0]
    assert cont_tok != cue_tok

    class ContextAndStyleLM(torch.nn.Module):
        def __init__(self, style_boost: float, ctx_boost: float):
            super().__init__()
            self.style_boost = style_boost
            self.ctx_boost = ctx_boost
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, ids):
            b, t = ids.shape
            logits = torch.zeros((b, t, 32), dtype=torch.float32)
            # Non-trivial unconditional style boost ONLY on cont_tok (actually shifts softmax!).
            # Use log-space additive factors on softmax probabilities via exact logit shift:
            # Wait: softmax is non-linear across vocab when cue fires, so test exact unigram
            # shift where other tokens are scaled uniformly — or test pure unigram boost when
            # cue shifts an independent sub-distribution!
            logits[:, :, cont_tok] = self.style_boost
            has_cue = (ids == cue_tok).any(dim=1)
            if has_cue.any() and self.ctx_boost != 0.0:
                logits[has_cue, :, cont_tok] += self.ctx_boost
            return logits

    ctx_item = Item(
        kind="context",
        corner="fresh/wikipedia/en/Q1/context",
        fmt="prose",
        lang="en",
        source="wikipedia:en:1",
        ctx="RRRRRRRRRRRRRRRR",
        cont="AAAAAAAAAAAAAAAA",
        wrongs=("WWWWWWWWWWWWWWWW", "ZZZZZZZZZZZZZZZZ"),
    )
    ans_item = Item(
        kind="answer",
        corner="reading/x",
        fmt="prose",
        lang="en",
        source="gen:context",
        ctx="RRRRRRRRRRRRRRRR",
        cont="AAAAAAAAAAAAAAAA",
    )

    # 1. Uniform/dead LM scores exactly 0.0 on twin-context (never beats a real student!).
    dead = ContextAndStyleLM(style_boost=0.0, ctx_boost=0.0)
    assert exam_score.sit(dead, [ctx_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0] == pytest.approx(0.0, abs=1e-6)

    # 2. Pure unconditional style boost (no context comprehension) shifts raw answer score hugely
    #    (> +1.5 bits/byte!), yet scores EXACTLY 0.0 on twin-context comprehension!
    style_lo = ContextAndStyleLM(style_boost=0.0, ctx_boost=0.0)
    style_hi = ContextAndStyleLM(style_boost=4.0, ctx_boost=0.0)
    ans_lo = exam_score.sit(style_lo, [ans_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    ans_hi = exam_score.sit(style_hi, [ans_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    ctx_hi = exam_score.sit(style_hi, [ctx_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    assert ans_hi - ans_lo > 1.5  # proves style_boost genuinely moves continuation likelihood!
    assert ctx_hi == pytest.approx(0.0, abs=1e-6)  # twin-context cancels unconditional style completely!

    # 3. Real context comprehension (ctx_boost > 0) produces a strong positive twin-context score.
    smart = ContextAndStyleLM(style_boost=0.0, ctx_boost=3.0)
    assert exam_score.sit(smart, [ctx_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0] > 0.5


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
            self.model.norm = torch.nn.LayerNorm(8)

    def make_ones():
        m = TinyHFModel()
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(1.0)
            m.model.norm.weight.fill_(5.0)
        return m

    m1, m2, m3 = make_ones(), make_ones(), make_ones()
    _reset_top_layers(m1, reset_frac=0.5, init_seed=42)
    _reset_top_layers(m2, reset_frac=0.5, init_seed=42)
    _reset_top_layers(m3, reset_frac=0.5, init_seed=99)

    for idx in (0, 1):
        assert torch.all(m1.model.layers[idx].proj.weight == 1.0)
    assert torch.all(m1.model.norm.weight == 1.0)
    for idx in (2, 3):
        assert not torch.all(m1.model.layers[idx].proj.weight == 1.0)
        assert torch.equal(m1.model.layers[idx].proj.weight, m2.model.layers[idx].proj.weight)
        assert not torch.equal(m1.model.layers[idx].proj.weight, m3.model.layers[idx].proj.weight)


def test_season3_hf_topreset_train_loop_sft_fp32_norms_and_clean_return(monkeypatch, tmp_path):
    """End-to-end unit test of Season 3's hf_topreset trainer on a tiny HF-like model:
    FP32 master weights move 1D norm gains, all micro-batches contribute gradients,
    SFT probe runs and updates weights, warmup_steps scales LR, init_seed controls reset,
    and returned model is eval(), activation-checkpointing-disabled, with zero .grad tensors."""
    torch = pytest.importorskip("torch")
    import transformers

    from stet import train as tr

    class TinyRMSNorm(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(d))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

    class TinyLayer(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = TinyRMSNorm(d)
            self.fc = torch.nn.Linear(d, d, bias=False)

        def forward(self, x):
            return x + self.fc(torch.tanh(self.norm(x)))

    class TinyCausalLM(torch.nn.Module):
        def __init__(self, vocab=32, d=16, n_layers=4):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True, vocab_size=vocab)
            self.embed = torch.nn.Embedding(vocab, d)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([TinyLayer(d) for _ in range(n_layers)])
            self.lm_head = torch.nn.Linear(d, vocab, bias=False)
            self.ckpt_enabled = False
            g = torch.Generator(device="cpu").manual_seed(1234)
            with torch.no_grad():
                self.embed.weight.normal_(0.0, 0.05, generator=g)
                for layer in self.model.layers:
                    layer.fc.weight.normal_(0.0, 0.05, generator=g)
                self.lm_head.weight.normal_(0.0, 0.05, generator=g)

        def activation_checkpointing_enable(self, **kwargs):
            self.ckpt_enabled = True

        def activation_checkpointing_disable(self):
            self.ckpt_enabled = False

        def forward(self, ids):
            x = self.embed(ids)
            for layer in self.model.layers:
                x = layer(x)
            return types.SimpleNamespace(logits=self.lm_head(x))

    loaded_dtypes: list = []

    def fake_from_pretrained(path, *, local_files_only=False, dtype=None, attn_implementation=None):
        loaded_dtypes.append(dtype)
        return TinyCausalLM()

    class TinyEnc:
        n_vocab = 32

        def encode_ordinary(self, text: str) -> list[int]:
            return [1 + (ord(c) % 30) for c in text]

    monkeypatch.setattr(tr, "_fetch_hf_student", lambda cfg: tmp_path)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))

    s3 = load_season("s3")
    # Build a Season clone with tiny vocab/eot matching TinyCausalLM while keeping hf_topreset + sft.
    s3_tiny = dataclasses.replace(
        s3,
        raw={
            **s3.raw,
            "tokenizer": {**s3.raw["tokenizer"], "eot": 0, "vocab_size": 32},
            "sft": {**s3.raw["sft"], "examples": 16, "batch_size": 4, "micro_batch": 2, "ctx": 16, "lr": 1e-3, "warmup_steps": 1},
        },
    )
    scale_sft = Scale(
        name="tiny",
        corpus_units=1,
        train_tokens=4 * 16 * 3,  # batch_size=4, ctx=16, 3 optimizer steps, micro_batch=2 -> 2 micro-batches/step
        n_layer=4,
        n_head=2,
        d_model=16,
        ctx=16,
        batch_size=4,
        micro_batch=2,
        lr=1e-3,
        warmup_steps=2,
        edit_seconds=60,
        sft_steps=2,
    )
    scale_no_sft = dataclasses.replace(scale_sft, sft_steps=0)

    # Verify warmup_steps actually controls LR ramp.
    assert tr.lr_at(0, 32, 3e-4, 0.05, 0.05, warmup_steps=10) == pytest.approx(3e-5)
    assert tr.lr_at(9, 32, 3e-4, 0.05, 0.05, warmup_steps=10) == pytest.approx(3e-4)

    rng = np.random.default_rng(7)
    tokens_a = rng.integers(1, 31, size=(4, 16 * 3 + 1), dtype=np.uint32)
    tokens_b = tokens_a.copy()
    tokens_b[2:, :] = (tokens_b[2:, :] % 30) + 1  # change ONLY second micro-batch

    m_a, info_a = tr.train(tokens_a, scale_sft, s3_tiny, init_seed=42, device=torch.device("cpu"), enc=TinyEnc(), sft_seed=b"sft")
    m_b, _ = tr.train(tokens_b, scale_sft, s3_tiny, init_seed=42, device=torch.device("cpu"), enc=TinyEnc(), sft_seed=b"sft")
    m_nosft, info_nosft = tr.train(tokens_a, scale_no_sft, s3_tiny, init_seed=42, device=torch.device("cpu"), enc=TinyEnc(), sft_seed=b"sft")
    m_seed2, _ = tr.train(tokens_a, scale_sft, s3_tiny, init_seed=99, device=torch.device("cpu"), enc=TinyEnc(), sft_seed=b"sft")

    # FP32 master weights always loaded, norm gains moved, clean eval + zero .grad + ckpt disabled.
    assert loaded_dtypes and all(dt is torch.float32 for dt in loaded_dtypes)
    assert not m_a.training
    assert not m_a.ckpt_enabled
    assert all(p.grad is None for p in m_a.parameters())
    assert all(p.dtype == torch.float32 for p in m_a.parameters())
    assert any(not torch.allclose(layer.norm.weight, torch.ones_like(layer.norm.weight)) for layer in m_a.model.layers)

    # All micro-batches contributed, SFT updated weights, init_seed changed final weights.
    assert tr.weights_sha256(m_a) != tr.weights_sha256(m_b)
    assert tr.weights_sha256(m_a) != tr.weights_sha256(m_nosft)
    assert info_a["sft"]["steps"] == 2 and "sft" not in info_nosft
    assert tr.weights_sha256(m_a) != tr.weights_sha256(m_seed2)

    # Verify _backward_chunked_head produces exact gradients on a weight-tied large-vocab CausalLM.
    class TiedLM(torch.nn.Module):
        def __init__(self, vocab=65536, d=8):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab)
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(vocab, d)
            self.model.proj = torch.nn.Linear(d, d, bias=False)
            self.model.forward = lambda ids: types.SimpleNamespace(last_hidden_state=self.model.proj(self.model.embed_tokens(ids)))
            self.lm_head = torch.nn.Linear(d, vocab, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight  # weight tying!

        def forward(self, ids):
            return types.SimpleNamespace(logits=self.lm_head(self.model(ids).last_hidden_state))

    torch.manual_seed(11)
    m_ref = TiedLM()
    m_chk = TiedLM()
    m_chk.load_state_dict(m_ref.state_dict())
    m_chk.lm_head.weight = m_chk.model.embed_tokens.weight
    x_t = torch.randint(0, 1000, (4, 6), dtype=torch.long)
    y_t = torch.randint(0, 1000, (4, 6), dtype=torch.long)
    ref_loss = torch.nn.functional.cross_entropy(m_ref(x_t).logits.float().view(-1, 65536), y_t.reshape(-1)) / 2.0
    ref_loss.backward()
    chk_loss = tr._backward_chunked_head(m_chk, x_t, y_t, 2.0, torch.device("cpu"))
    assert float(chk_loss) == pytest.approx(float(ref_loss.detach()), rel=1e-5, abs=1e-6)
    assert torch.allclose(m_chk.model.proj.weight.grad, m_ref.model.proj.weight.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(m_chk.model.embed_tokens.weight.grad, m_ref.model.embed_tokens.weight.grad, atol=1e-5, rtol=1e-4)


def test_hf_tokenizer_encodes_control_tokens_as_ordinary_text_and_verifies_sha256(monkeypatch, tmp_path):
    """Web pages containing literal <|endoftext|>, <|im_start|>, or <tool_call> must never
    emit special/control token IDs in encode_ordinary, and tampered cached files must fail sha256."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split

    from stet import tok

    chars = sorted(set("abcdefghijklmnopqrstuvwxyz <|>_/:-.!?0123456789"))
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["[UNK]"] = len(vocab)
    t = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    t.pre_tokenizer = Split("", behavior="isolated")
    raw = json.loads(t.to_str())
    raw["added_tokens"] = [
        {"id": 151643, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 151644, "content": "<|im_start|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 151657, "content": "<tool_call>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
    ]
    tok_bytes = json.dumps(raw).encode()
    tok_sha = hashlib.sha256(tok_bytes).hexdigest()

    monkeypatch.setenv("STET_CACHE", str(tmp_path / "cache"))
    tok.load_encoding.cache_clear()
    tok_dir = tok._hf_model_dir("test/tiny-tok", "rev1")
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "tokenizer.json"
    tok_path.write_bytes(tok_bytes)

    s3 = load_season("s3")
    fake_season = dataclasses.replace(
        s3,
        raw={
            **s3.raw,
            "tokenizer": {
                "kind": "hf",
                "repo": "test/tiny-tok",
                "revision": "rev1",
                "file": "tokenizer.json",
                "url": "https://example.invalid/tokenizer.json",
                "sha256": tok_sha,
                "size": len(tok_bytes),
                "vocab_size": 151936,
                "eot": 151643,
            },
        },
    )
    enc = tok.load_encoding(fake_season)
    text = "hi <|endoftext|> <|im_start|> <tool_call> ok"
    ids = enc.encode_ordinary(text)
    batch_ids = enc.encode_ordinary_batch([text, "plain text"])
    assert ids == batch_ids[0]
    for forbidden in (151643, 151644, 151657):
        assert forbidden not in ids
    assert enc.decode(ids).replace(" ", "") == text.replace(" ", "")

    # Tampering with cached tokenizer.json must be caught by sha256 verification and unlinked.
    tok.load_encoding.cache_clear()
    tok_path.write_bytes(tok_bytes + b" ")
    with pytest.raises(data.Mismatch):
        tok.load_encoding(fake_season)
    assert not tok_path.exists()

    # Same verification holds for cached student files in _fetch_hf_student.
    torch = pytest.importorskip("torch")  # noqa: F841
    from stet.train import _fetch_hf_student

    cfg_dir = tok._hf_model_dir("test/tiny-student", "rev1")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.json"
    cfg_bytes = b'{"model_type":"qwen3"}'
    cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
    cfg_path.write_bytes(cfg_bytes + b"tampered")
    with pytest.raises(data.Mismatch):
        _fetch_hf_student({
            "repo": "test/tiny-student",
            "revision": "rev1",
            "files": [{"path": "config.json", "url": "https://example.invalid/config.json", "sha256": cfg_sha, "size": len(cfg_bytes)}],
        })
    assert not cfg_path.exists()


def test_cuda_linux_only_packages_filtered_on_non_nvidia(monkeypatch):
    s3 = load_season("s3")
    cuda_only = set(s3.raw["environment"]["cuda_linux_only"])
    also_cuda = set((s3.raw["environment"].get("also_cuda") or {}).keys())
    assert {"vllm", "liger-kernel"} <= cuda_only
    assert also_cuda and "triton" in also_cuda

    monkeypatch.setattr(env, "is_nvidia_linux", lambda: True)
    nvidia_pkgs = set(env.expected(s3)["packages"])
    assert (cuda_only | also_cuda) <= nvidia_pkgs

    monkeypatch.setattr(env, "is_nvidia_linux", lambda: False)
    non_nvidia_pkgs = set(env.expected(s3)["packages"])
    assert not ((cuda_only | also_cuda) & non_nvidia_pkgs)


def test_reader_example_pins_qwen3_06b_trailing_space_and_vllm_fallback(monkeypatch):
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
    assert "task.write(" not in main_src[: main_src.index("_score_vllm(")]
    assert "raise RuntimeError" in main_src

    # Load module and verify trailing-space rubric fix, overprovision/reserve, BYTES_PER_TOKEN, and vLLM failure fallback.
    spec = importlib.util.spec_from_file_location("reader_curator_fallback_test", root / "curator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.RUBRIC_SUFFIX.endswith(": "), "RUBRIC_SUFFIX must end with trailing space so next token is digit 1..5"
    assert 3.5 <= mod.BYTES_PER_TOKEN <= 4.0
    assert mod.CANDIDATE_OVERPROVISION >= 10.0 and mod.RESERVE_SECONDS >= 45
    assert 'VLLM_WORKER_MULTIPROC_METHOD' in src
    assert 'VLLM_WORKER_MULTIPROC_METHOD' in (ROOT / "src" / "stet" / "runner.py").read_text()

    torch = pytest.importorskip("torch")

    class BrokenLLM:
        def __init__(self, **kwargs):
            raise RuntimeError("CUDA OOM during vLLM engine init")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=BrokenLLM, SamplingParams=object))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    fake_task = types.SimpleNamespace(device=lambda: "cuda", time_left=lambda: 600.0)
    scores: dict[int, float] = {}
    assert mod._score_vllm(fake_task, "/fake/model", [(1.0, 0, 200, "Sample doc.")], scores) is False
    assert scores == {}


def test_reader_vllm_bounds_long_multilingual_prompts_and_handles_logprob_dataclass(monkeypatch):
    """Regression test for NVIDIA vLLM reader: long multilingual prompts must stay within
    max_model_len while preserving RUBRIC_SUFFIX, request digit logprob_token_ids, and read
    vLLM's Logprob dataclass without triggering eager float(entry) evaluation."""
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

    written: list[list[int]] = []
    fake_task = types.SimpleNamespace(
        device=lambda: "cuda",
        time_left=lambda: 600.0,
        train_tokens=5000,
        write=lambda lines: written.append(list(lines)),
    )
    cands = [
        (1.0, 0, 2000, "日本語の長文ドキュメントです。" * 80),
        (0.9, 1, 1800, "Длинный русский текст с множеством слов. " * 40),
        (0.8, 2, 400, "Short clear English technical explanation."),
    ]
    scores: dict[int, float] = {}
    assert mod._score_vllm(fake_task, "/fake/model", cands, scores) is True
    assert len(scores) == len(cands)
    assert all(1.0 <= v <= 5.0 for v in scores.values())
    assert seen_calls and written


def test_large_vocab_exam_scoring_matches_full_ce_and_uses_token_budget_batches():
    """Large-vocab scoring must batch by padded token count (<= 2,048 tokens) and match naive
    full-tensor FP32 cross-entropy to numerical precision."""
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
    lengths = [782, 793, 797, 866, 871, 909, 1010, 1024]
    seqs = [([1 + ((i * 13 + t) % 500) for t in range(L)], L - 20) for i, L in enumerate(lengths)]
    got = exam_score.nll(model, seqs, device=torch.device("cpu"), batch=64)

    assert all(b * t <= 2048 for b, t in seen_shapes), seen_shapes
    assert max(b for b, _ in seen_shapes) <= 2

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
    from stet import cli, gym

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


def test_season3_has_no_passage_items_and_v11_sessions_are_conditional_answers():
    """v11 shell sessions are conditional kind='answer', and Season 3's offline practice pack
    has zero whole-transcript passage items while Season 2 v9 stays frozen."""
    from stet.exam.aggregate import part_of, stratum_key
    from stet.exam.fresh.reason import load_practice_pack
    from stet.exam.gen import shell

    s3 = load_season("s3")
    pack = load_practice_pack(s3, b"s3-pack-seed")
    assert all(it.kind != "passage" for it in pack)
    v11_agent = gen.generate(b"seed-v11", per_corner=2, families=[shell], more={"agent": 4})
    assert all(it.kind != "passage" for it in v11_agent)
    sessions = [it for it in v11_agent if it.corner.endswith("/session")]
    assert len(sessions) == 16
    assert all(it.kind == "answer" and it.ctx and it.cont for it in sessions)
    assert {stratum_key(it) for it in v11_agent if part_of(it) == "agent" and it.lang == "en"} == {
        "logs/answer",
        "structured/answer",
    }


def test_sft_probe_excludes_exact_scored_exam_questions_including_booleans():
    """Item 2 (P2): Even small finite generators like 3-term Booleans (which previously collided on
    s3-p00004 and s3-p00031) must never overlap between the round's SFT probe and scored exam."""
    s3 = load_season("s3")
    for seed_i in (0, 4, 31):
        rnd = practice_round(s3, seed_i)
        exam_pairs = {(it.ctx, it.cont) for it in gen.for_season(s3, rnd.seed) if it.kind == "answer"}
        probe = sft.build_sft_examples(rnd.seed, n=512, season=s3)
        assert len(probe) == 512
        assert not (set(probe) & exam_pairs), f"overlap on s3-p{seed_i:05d}: {set(probe) & exam_pairs}"


def test_context_scoring_normalizes_by_scored_prefix_bytes_when_truncated(monkeypatch):
    """Item 3 (P2): When a long multilingual continuation exceeds the half-window token cap,
    twin-context scoring must divide by the byte length of the scored prefix, not the full text."""
    from stet.exam import score as exam_score

    # 1 char = 1 token mock encoding where each char is 3 UTF-8 bytes (Santali / Ol Chiki style).
    class ThreeByteCharEnc:
        def encode_ordinary(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

        def decode_bytes(self, ids: list[int]) -> bytes:
            return "".join(chr(i) for i in ids).encode("utf-8")

    # ctx=16 -> max continuation tokens = (16 + 1) // 2 = 8 tokens = 24 bytes (full cont is 20 chars = 60 bytes).
    full_cont = "ᱥ" * 20
    it = Item(
        kind="context",
        corner="fresh/wikipedia/sat/-/article-ctx",
        fmt="prose",
        lang="sat",
        source="wikipedia:1",
        ctx="Title\nOpening sentence.",
        cont=full_cont,
        wrongs=("Other\nWrong opening.",),
    )
    # Stub nll so wrong - real = LN2 * 24 nats -> should equal exactly 1.0 bit per scored byte (24 bytes),
    # NOT 24 / 60 = 0.4 bits per full-continuation byte!
    monkeypatch.setattr(exam_score, "nll", lambda model, seqs, device, batch=32: [10.0, 10.0 + exam_score.LN2 * 24.0])
    got = exam_score.sit(None, [it], ThreeByteCharEnc(), eot=0, ctx=16, device=None, floor=1)
    assert got[0] == pytest.approx(1.0)


def test_prose_paragraph_boundaries_accepted_and_separator_preserved():
    """Item 4 (P2): Sentence-ending punctuation followed by a paragraph newline must be accepted as
    a clean prose split and preserve '\\n' at the end of ctx (with no synthetic leading space on cont)."""
    from stet.exam.fresh import Doc
    from stet.exam.fresh.select import _split_continuation

    doc_nl = Doc(
        source="wikipedia-added",
        id="p1",
        lang="en",
        url="https://example.invalid/p1",
        license="CC-BY-SA",
        published=0,
        title="Example",
        text=(
            "First paragraph has one long sentence describing the subject in sufficient detail.\n"
            "Second paragraph continues the explanation with additional context and facts."
        ),
    )
    res_nl = _split_continuation(doc_nl)
    assert res_nl is not None
    _, fmt, ctx, cont = res_nl
    assert fmt == "prose"
    assert ctx.endswith(".\n")
    assert cont.startswith("Second paragraph") and not cont.startswith(" ")

    doc_sp = Doc(
        source="wikipedia-added",
        id="p2",
        lang="en",
        url="https://example.invalid/p2",
        license="CC-BY-SA",
        published=0,
        title="Example",
        text=(
            "First sentence has enough detail to exceed the opening minimum length cleanly. "
            "Second sentence continues on the same line after a regular space."
        ),
    )
    res_sp = _split_continuation(doc_sp)
    assert res_sp is not None
    _, fmt2, ctx2, cont2 = res_sp
    assert fmt2 == "prose"
    assert ctx2.endswith(".") and not ctx2.endswith(" ")
    assert cont2.startswith(" Second sentence")


def test_reader_vllm_memory_utilization_respects_sandbox_allowance_on_large_gpus():
    """Item 7 (P2): On 48 GiB and 80 GiB GPUs, vLLM's gpu_memory_utilization must stay inside
    Season 3's 22 GiB sandbox allowance rather than requesting 0.65 * physical VRAM."""
    root = ROOT / "examples" / "reader"
    spec = importlib.util.spec_from_file_location("reader_util_test", root / "curator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for total_gb in (24.0, 32.0, 48.0, 80.0):
        util = mod._vllm_gpu_memory_utilization(22.0, total_gb)
        assert 0.1 <= util <= 0.65
        assert util * total_gb <= 20.1, f"on {total_gb} GiB GPU, util={util} requests {util * total_gb:.1f} GiB > 20 GiB"
    assert mod._vllm_gpu_memory_utilization(22.0, 48.0) < 0.45
    assert mod._vllm_gpu_memory_utilization(0.0, 48.0) == pytest.approx(0.65)


def test_real_bpe_tokenizer_encodes_control_tokens_as_ordinary_text(monkeypatch, tmp_path):
    """Non-renumbered real byte-level BPE tokenizer test (matching Qwen3's BPE structure):
    literal <|endoftext|>, <|im_start|>, and <tool_call> in web text must never produce
    special/control token IDs and must round-trip identically."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from stet import tok

    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    bpe.train_from_iterator(
        ["Hello world! <|endoftext|> <|im_start|> <tool_call> plain web text\n" * 4],
        trainer=trainer,
    )
    raw = json.loads(bpe.to_str())
    raw["added_tokens"] = [
        {"id": 151643, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 151644, "content": "<|im_start|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 151657, "content": "<tool_call>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False},
    ]
    tok_bytes = json.dumps(raw).encode()
    tok_sha = hashlib.sha256(tok_bytes).hexdigest()

    monkeypatch.setenv("STET_CACHE", str(tmp_path / "cache"))
    tok.load_encoding.cache_clear()
    tok_dir = tok._hf_model_dir("test/bpe-tok", "rev1")
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "tokenizer.json").write_bytes(tok_bytes)

    s3 = load_season("s3")
    fake_season = dataclasses.replace(
        s3,
        raw={
            **s3.raw,
            "tokenizer": {
                "kind": "hf",
                "repo": "test/bpe-tok",
                "revision": "rev1",
                "file": "tokenizer.json",
                "url": "https://example.invalid/tokenizer.json",
                "sha256": tok_sha,
                "size": len(tok_bytes),
                "vocab_size": 151936,
                "eot": 151643,
            },
        },
    )
    enc = tok.load_encoding(fake_season)
    sample = "Web page quoting <|endoftext|>, <|im_start|>, and <tool_call> literally."
    ids = enc.encode_ordinary(sample)
    assert enc.encode_ordinary_batch([sample])[0] == ids
    for forbidden in (151643, 151644, 151657):
        assert forbidden not in ids
    assert enc.decode(ids) == sample


def test_select_context_blocks_reply_chain_continuation_leak():
    """In conversation threads, a later reply whose opening (question) is an earlier post's
    continuation (text) must never be paired as a wrong opening for that earlier post."""
    shared_reply = (
        "Switching to an immutable event log eliminated every race condition in our nightly rebuilds "
        "and cut tail latency in half across all regions."
    )
    docs = [
        Doc(
            source="hackernews",
            id="hn:parent",
            url="https://news.ycombinator.com/item?id=1",
            lang="en",
            license="public",
            published=1_800_000_000,
            title="Thread A",
            question="We spent two quarters chasing intermittent cache corruption across regional read replicas.",
            text=shared_reply,
            topic="hn:1",
        ),
        Doc(
            source="hackernews",
            id="hn:child_reply",
            url="https://news.ycombinator.com/item?id=2",
            lang="en",
            license="public",
            published=1_800_000_001,
            title="Thread A",
            question=shared_reply,
            text="Have you benchmarked compaction overhead when log segments grow past a few gigabytes?",
            topic="hn:2",
        ),
        Doc(
            source="hackernews",
            id="hn:unrelated",
            url="https://news.ycombinator.com/item?id=3",
            lang="en",
            license="public",
            published=1_800_000_002,
            title="Thread B",
            question="PostgreSQL logical replication slots require careful monitoring whenever downstream consumers fall behind.",
            text="Otherwise WAL segments accumulate on primary storage until disk alerts wake up the on-call engineer.",
            topic="hn:3",
        ),
    ]
    items = select_context(docs, b"reply-leak-seed", 10, {"discussion", "conversation"}, shares={"en": 100.0})
    parent_items = [it for it in items if it.source == "hackernews:hn:parent"]
    assert parent_items, "expected parent doc to produce a context item"
    for it in parent_items:
        for w in it.wrongs:
            assert it.cont.strip() not in w
            assert "immutable event log" not in w


def test_sft_probe_blocks_fewshot_final_question_and_worked_shot_collisions():
    """SFT probe must never share a few-shot final question or worked-shot Q+A with any exam item."""
    s3 = load_season("s3")
    for seed_i in (0, 4, 31):
        rnd = practice_round(s3, seed_i)
        exam_items = [it for it in gen.for_season(s3, rnd.seed) if it.kind == "answer"]
        exam_tails = {(sft._tail_question(it.ctx), it.cont.strip()) for it in exam_items}
        probe = sft.build_sft_examples(rnd.seed, n=512, season=s3)
        for p, c in probe:
            assert (sft._tail_question(p), c.strip()) not in exam_tails
            for block in p.split("\n\n")[:-1]:
                if "\n" in block:
                    q, a = block.rsplit("\n", 1)
                    if ":" in a:
                        prefix, ans = a.rsplit(":", 1)
                        assert (q + "\n" + prefix + ":", ans.strip()) not in exam_tails


def test_reader_fails_loudly_on_model_error_without_writing_unrated_fallback(monkeypatch, tmp_path):
    """If both vLLM and transformers fail to load, examples/reader must raise RuntimeError and
    leave no unrated fallback script.jsonl on disk."""
    from conftest import make_corpus

    root = ROOT / "examples" / "reader"
    spec = importlib.util.spec_from_file_location("reader_fail_loud_test", root / "curator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    corp_dir = tmp_path / "corpus"
    make_corpus(
        corp_dir,
        [
            "Clean well-written paragraph explaining distributed consensus algorithms in clear detail. " * 8
            for _ in range(6)
        ],
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assets_dir = tmp_path / "assets"
    (assets_dir / "qwen3-0.6b").mkdir(parents=True)
    task_json = tmp_path / "task.json"
    task_json.write_text(
        json.dumps(
            {
                "train_tokens": 2048,
                "cores": 1,
                "memory_gb": 16,
                "gpu_memory_gb": 22,
                "min_copy_bytes": 64,
                "assets": {"qwen3-0.6b": str(assets_dir / "qwen3-0.6b")},
            }
        )
    )
    monkeypatch.setenv("STET_CORPUS", str(corp_dir))
    monkeypatch.setenv("STET_TASK", str(task_json))
    monkeypatch.setenv("STET_OUT", str(out_dir))
    monkeypatch.setenv("STET_ASSETS", str(assets_dir))
    monkeypatch.setenv("STET_DEADLINE", "4102444800")

    monkeypatch.setattr(mod, "_score_vllm", lambda task, where, cands, scores: False)

    def broken_tf(task, where, cands, scores):
        raise RuntimeError("simulated model load failure")

    monkeypatch.setattr(mod, "_score_transformers", broken_tf)
    with pytest.raises(RuntimeError, match="simulated model load failure"):
        mod.main()
    assert not (out_dir / "script.jsonl").exists()

    # Even if transformers returns cleanly having rated zero documents, main() must fail loudly.
    monkeypatch.setattr(mod, "_score_transformers", lambda task, where, cands, scores: None)
    with pytest.raises(RuntimeError, match="rated zero candidates"):
        mod.main()
    assert not (out_dir / "script.jsonl").exists()


def test_trainer_sft_prompt_mask_cuda_ckpt_vram_gate_and_early_offload(monkeypatch, tmp_path):
    """Verify SFT prompt-token masking (-100 on prompt, loss only on completion), activation
    checkpointing enable/disable, 16 GB VRAM verifier gate, and CPU offload of held early students."""
    torch = pytest.importorskip("torch")
    from stet import client
    from stet import train as tr

    # 1. SFT prompt mask: mutating logits on prompt positions must not change _ce_loss, while
    #    mutating logits on completion positions must change _ce_loss.
    s3 = load_season("s3")
    logits = torch.zeros((2, 6, 32), dtype=torch.float32)
    targets = torch.tensor([[-100, -100, -100, 5, 6, 7], [-100, -100, 4, 5, -100, -100]], dtype=torch.long)
    base_loss = float(tr._ce_loss(logits, targets, ignore_index=-100))
    mut_prompt = logits.clone()
    mut_prompt[0, :3, :] = 10.0
    mut_prompt[1, :2, :] = -10.0
    assert float(tr._ce_loss(mut_prompt, targets, ignore_index=-100)) == pytest.approx(base_loss, abs=1e-6)
    mut_comp = logits.clone()
    mut_comp[0, 3, 5] = 8.0
    assert float(tr._ce_loss(mut_comp, targets, ignore_index=-100)) < base_loss - 0.1

    # 2. Activation checkpointing toggle helper calls enable(use_reentrant=False) and disable().
    calls: list[tuple[str, dict]] = []

    class CkptProbe(torch.nn.Module):
        def activation_checkpointing_enable(self, **kwargs):
            calls.append(("enable", kwargs))

        def activation_checkpointing_disable(self):
            calls.append(("disable", {}))

    probe = CkptProbe()
    tr._set_activation_checkpointing(probe, True)
    tr._set_activation_checkpointing(probe, False)
    assert calls == [("enable", {"activation_checkpointing_kwargs": {"use_reentrant": False}}), ("disable", {})]

    # 3. 16 GB VRAM gate in _verify rejects a 12 GB CUDA GPU for Season 3.
    monkeypatch.delenv("STET_DEV", raising=False)
    monkeypatch.setattr(client, "need_key", lambda: types.SimpleNamespace(handle="v", public="0" * 64, private="0" * 64))
    monkeypatch.setattr(client, "request", lambda method, path, **kw: {"scale": "standard"})
    monkeypatch.setattr(tr, "pick_device", lambda dev=None: types.SimpleNamespace(type="cuda"))
    monkeypatch.setattr(tr, "hardware_info", lambda dev: {"device": "cuda", "gpu": "RTX 3060", "vram_gb": 12.0, "backend": "cuda"})
    args = types.SimpleNamespace(
        season="s3",
        device="cuda",
        any_device=False,
        edits=False,
        no_sandbox=False,
        once=True,
        max_jobs=1,
        exit_when_idle=0,
        no_upgrade=True,
    )
    with pytest.raises(SystemExit) as exc:
        client._verify(args)
    assert exc.value.code == 1

    # 4. Early-grade CPU offload moves held student off GPU while waiting for the exam.
    import zstandard

    from stet import pipeline

    moved: list[str] = []

    class FakeModel:
        def zero_grad(self, set_to_none=True):
            moved.append("zero_grad")

        def to(self, dev):
            moved.append(f"to:{dev}")
            return self

    script_bytes = b"0\n"
    script_sha = hashlib.sha256(script_bytes).hexdigest()
    script_zst = zstandard.ZstdCompressor().compress(script_bytes)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: moved.append("empty_cache"))
    monkeypatch.setattr(pipeline, "train_baseline", lambda rnd, scale, dev, **kw: types.SimpleNamespace(model=FakeModel()))
    monkeypatch.setattr(pipeline, "train_round", lambda rnd, scale, path, **kw: types.SimpleNamespace(model=FakeModel()))
    monkeypatch.setattr(client, "request", lambda method, path, **kw: script_zst)
    rnd = practice_round(s3, 0)
    rnd_no_exam = dataclasses.replace(rnd, exam_digest=None)
    monkeypatch.setattr(client, "_job_round", lambda season, job, out: rnd_no_exam)
    bases: dict = {}
    out = types.SimpleNamespace(line=lambda *a, **kw: None, dim=lambda s: s, progress=lambda *a: None)
    pending = client._grade_job(
        {"id": 1, "scale": "nano", "round": rnd.id, "script": {"sha256": script_sha}, "baseline": True},
        s3,
        s3.scale("nano"),
        types.SimpleNamespace(type="cuda"),
        types.SimpleNamespace(),
        out,
        tmp_path / "work",
        bases=bases,
    )
    assert isinstance(pending, client.Pending)
    assert rnd.id in bases
    assert moved.count("to:cpu") == 2 and moved.count("empty_cache") == 2

    # 5. Restoring a held early-grade student moves it back to the grading device before sitting the exam.
    from stet import grade as gr

    restored: list[object] = []

    class RestoreModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def to(self, dev):
            restored.append(dev)
            return self

    monkeypatch.setattr("stet.exam.score.sit", lambda model, items, enc, eot, ctx, device, floor=16: [-1.0] * len(items))
    fake_cuda = types.SimpleNamespace(type="cuda")
    held = gr.Trained(result={}, trained=True, model=RestoreModel(), enc=types.SimpleNamespace(), device=fake_cuda)
    gr.sit_exam(held, rnd, s3.scale("nano"), [Item(kind="answer", corner="logic/x", fmt="prose", lang="en", source="g", ctx="Q", cont="A")])
    assert restored == [fake_cuda], "sit_exam did not move CPU-offloaded student back onto its CUDA device"

    # 6. Large-vocab SFT _ce_loss averages by valid completion tokens (not unweighted row means).
    big_logits = torch.zeros((2, 4, 65536), dtype=torch.float32)
    big_logits[0, 3, 1] = 10.0  # row 0 has 1 valid token with near-0 loss
    big_logits[1, :, 2] = -10.0  # row 1 has 4 valid tokens with higher loss
    big_targets = torch.tensor([[-100, -100, -100, 1], [2, 2, 2, 2]], dtype=torch.long)
    got_big = float(tr._ce_loss(big_logits, big_targets, ignore_index=-100))
    ref_big = float(torch.nn.functional.cross_entropy(big_logits.view(-1, 65536), big_targets.reshape(-1), ignore_index=-100))
    assert got_big == pytest.approx(ref_big, rel=1e-5, abs=1e-5)

    # 7. build_model loads FP32 master weights, disables activation checkpointing when
    # freeze_bottom=True (Season 3 ERNIE-4.5-0.3B), and enables it when freeze_bottom=False.
    import transformers

    cuda_loaded: list[tuple[object, bool]] = []

    class FakeCudaLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.ckpt = False

        def activation_checkpointing_enable(self, **kw):
            self.ckpt = True

        def to(self, dev):
            cuda_loaded.append((dev, self.ckpt))
            return self

    seen_cuda_dtypes: list = []
    monkeypatch.setattr(tr, "_fetch_hf_student", lambda cfg: tmp_path)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda path, *, local_files_only=False, dtype=None, attn_implementation=None: (seen_cuda_dtypes.append(dtype), FakeCudaLM())[1]),
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    tr.build_model(s3.scale("nano"), s3, init_seed=7, device=types.SimpleNamespace(type="cuda"))
    assert seen_cuda_dtypes == [torch.float32]
    assert cuda_loaded and cuda_loaded[0][1] is False  # freeze_bottom=True skips redundant checkpointing

    unfrozen_s3 = dataclasses.replace(s3, raw={**s3.raw, "student": {**s3.raw["student"], "freeze_bottom": False}})
    cuda_loaded.clear()
    tr.build_model(unfrozen_s3.scale("nano"), unfrozen_s3, init_seed=7, device=types.SimpleNamespace(type="cuda"))
    assert cuda_loaded and cuda_loaded[0][1] is True


def test_reason_module_selection_wrappers_anti_copy_and_decontam(tmp_path):
    """Season 3 fresh reasoning (Exam C v2): deterministic selection, zero-Q/A synthesis wrappers,
    angle-bracket cleaning, <50% 5-gram verbatim-copy rejection, and excerpt+response decontam."""
    from conftest import make_corpus

    from stet.exam.decontam import seen
    from stet.exam.fresh import Doc
    from stet.exam.fresh.reason import (
        _clean_angle,
        ngram_overlap,
        parse_generated_items,
        select_source_docs,
        wrap_prompt,
    )

    # 1. Deterministic selection stratified across kinds & languages (order-independent).
    docs = [
        Doc(
            source="wikipedia" if i % 2 == 0 else "ccnews",
            id=f"d{i}",
            url=f"https://example.org/{i}",
            lang="en" if i % 3 != 0 else "de",
            license="CC BY-SA 4.0",
            published=1_800_000_000 + i,
            title=f"Topic {i}",
            text=f"Detailed historical and technical passage number {i} explaining mechanisms and causes. " * 8,
            topic=f"T{i}",
        )
        for i in range(40)
    ]
    sel1 = select_source_docs(docs, b"seed-reason", target=12, shares={"en": 60.0, "de": 40.0})
    sel2 = select_source_docs(docs[::-1], b"seed-reason", target=12, shares={"en": 60.0, "de": 40.0})
    assert len(sel1) == 12 and [d["id"] for d in sel1] == [d["id"] for d in sel2]
    assert {d["lang"] for d in sel1} == {"en", "de"}

    # 2. Synthesis wrappers contain zero question marks or Q:/A: syntax.
    for w_idx in range(12):
        p_syn = wrap_prompt("A short factual excerpt about solar panels.", "The passage explains why.", "synthesis", w_idx)
        assert "?" not in p_syn and "Q:" not in p_syn and "A:" not in p_syn

    # 3. Angle-bracket cleaner + 5-gram overlap filter + parser.
    assert _clean_angle("<Why did it fail?>") == "Why did it fail?"
    assert _clean_angle("<<Response text>>") == "Response text"
    excerpt = (
        "The municipal grid operators installed lithium-iron-phosphate storage banks across four substations "
        "in order to absorb midday solar generation surges and stabilize evening peak voltage levels without "
        "relying on diesel peaking turbines."
    )
    assert ngram_overlap(excerpt, excerpt) > 0.9
    good_ans = (
        "Adding battery storage lets utility engineers capture surplus afternoon photovoltaic output and discharge it "
        "during high-demand twilight hours, keeping voltage stable while avoiding fossil-fuel backup generators."
    )
    assert ngram_overlap(good_ans, excerpt) < 0.2
    selected_rows = [
        {"source": "wikipedia:en:1", "id": "en:1", "lang": "en", "text": excerpt, "task_type": "explain", "wrapper_id": 0},
        {"source": "wikipedia:en:2", "id": "en:2", "lang": "en", "text": excerpt, "task_type": "infer", "wrapper_id": 1},
        {"source": "wikipedia:en:3", "id": "en:3", "lang": "en", "text": excerpt, "task_type": "synthesis", "wrapper_id": 2},
    ]
    raw_gens = [
        f" <Why were the storage banks added across substations?>\nRESPONSE: {good_ans}",
        f" What operational trade-off arises here?\nRESPONSE: {excerpt}",  # verbatim copy -> rejected!
        f" The passage summarizes the rationale behind the grid upgrade?\nRESPONSE: {good_ans}",  # '?' stripped on synthesis!
    ]
    parsed, dropped = parse_generated_items(selected_rows, raw_gens, max_ngram_overlap=0.5)
    parts_got = [it.source.split(":", 1)[0] for it in parsed]
    assert parts_got == ["explain", "synthesis"] and dropped.get("verbatim_copy") == 1
    assert all("<" not in it.ctx and ">" not in it.ctx for it in parsed)
    assert all("?" not in it.ctx and "Q:" not in it.ctx for it in parsed if it.source.startswith("synthesis:"))

    # 4. Decontamination checks BOTH excerpt (`ctx`) and response (`cont`) for reasoning items.
    s3 = load_season("s3")
    corpus_exc = make_corpus(tmp_path / "c_exc", ["Header\n" + excerpt + " Footer"])
    corpus_ans = make_corpus(tmp_path / "c_ans", ["Header\n" + good_ans + " Footer"])
    corpus_clean = make_corpus(tmp_path / "c_clean", ["Completely unrelated article about marine sponges."])
    assert seen(parsed, corpus_exc, s3) == {0, 1}
    assert seen(parsed, corpus_ans, s3) == {0, 1}
    assert seen(parsed, corpus_clean, s3) == set()


def test_ernie_byte_fallback_decode_bytes_and_freeze_bottom_and_chunked_equiv(tmp_path):
    """Verify SentencePiece Metaspace+ByteFallback decode_bytes, freeze_bottom freezing & weight
    immutability, and exact numerical equivalence of chunked lm_head in eval nll + SFT probe."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from stet import tok
    from stet import train as tr
    from stet.exam import score as exam_score

    # 1. SentencePiece Metaspace + ByteFallback exact byte decoding on truncated multi-byte UTF-8.
    class FakeSPDec:
        def __init__(self, vocab):
            self._v = vocab

        def id_to_token(self, tid):
            return self._v.get(tid)

        def decode(self, ids, skip_special_tokens=False):
            return ""

    # " hi 😀" where 😀 is UTF-8 F0 9F 98 80; truncated after first 2 bytes of emoji!
    sp_enc = tok._HFEncoding(
        None,
        FakeSPDec({10: "\u2581hi", 11: "\u2581", 12: "<0xF0>", 13: "<0x9F>"}),
        name="fake-ernie",
        byte_fallback=True,
    )
    assert sp_enc.decode_bytes([10, 11, 12, 13]) == b" hi \xf0\x9f"
    assert len(sp_enc.decode_bytes([10, 11, 12, 13])) == 6  # no U+FFFD 3-byte inflation!

    # 2. freeze_bottom freezes embed_tokens, lm_head, and bottom half of layers while top half trains.
    class TinyLayer(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = torch.nn.Linear(d, d)

        def forward(self, x):
            return x + self.lin(x)

    class TinyInner(torch.nn.Module):
        def __init__(self, v, d, n):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(v, d)
            self.layers = torch.nn.ModuleList([TinyLayer(d) for _ in range(n)])
            self.norm = torch.nn.LayerNorm(d)

        def forward(self, x):
            h = self.embed_tokens(x)
            for lyr in self.layers:
                h = lyr(h)
            return types.SimpleNamespace(last_hidden_state=self.norm(h))

    class TinyHFModel(torch.nn.Module):
        def __init__(self, v=65536, d=8, n=4):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=v)
            self.model = TinyInner(v, d, n)
            self.lm_head = torch.nn.Linear(d, v, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight  # tied like ERNIE-4.5

        def forward(self, x):
            return types.SimpleNamespace(logits=self.lm_head(self.model(x).last_hidden_state))

    torch.manual_seed(0)
    m = TinyHFModel(v=65536, d=8, n=4)
    emb_before = m.model.embed_tokens.weight.detach().clone()
    bot_before = m.model.layers[0].lin.weight.detach().clone()
    tr._reset_top_layers(m, 0.5, init_seed=42, freeze_bottom=True)
    assert not m.model.embed_tokens.weight.requires_grad
    assert not m.lm_head.weight.requires_grad
    assert all(not p.requires_grad for p in m.model.layers[:2].parameters())
    assert all(p.requires_grad for p in m.model.layers[2:].parameters())

    # 3. Chunked nll on large-vocab HF model matches naive full-logits CE to <1e-5.
    m.eval()
    seqs = [([10, 20, 30, 40, 50, 60], 3), ([5, 15, 25, 35, 45], 2)]
    got = exam_score.nll(m, seqs, device=torch.device("cpu"), batch=2)
    with torch.no_grad():
        for idx, (ids, first) in enumerate(seqs):
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor([ids[1:]], dtype=torch.long)
            y[0, : first - 1] = -100
            logits = m(x).logits.float()
            ref = float(F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none").sum())
            assert got[idx] == pytest.approx(ref, rel=1e-5, abs=1e-5)

    # 4. After one optimizer step on frozen model, frozen weights stay bit-identical while top layer updates.
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randint(0, 100, (2, 6))
    y = torch.randint(0, 100, (2, 6))
    tr._backward_chunked_head(m, x, y, 1.0, torch.device("cpu"))
    opt.step()
    assert torch.equal(m.model.embed_tokens.weight, emb_before)
    assert torch.equal(m.model.layers[0].lin.weight, bot_before)


def test_reader_per_language_quantile_normalization_prevents_non_english_starvation():
    """Verify examples/reader per-language quantile normalization prevents English-biased 1-5
    scores from starving Korean, Arabic, Russian, and Japanese during selection & cooldown."""
    root = ROOT / "examples" / "reader"
    spec = importlib.util.spec_from_file_location("reader_quantile_test", root / "curator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._detect_script_group("The quick brown fox and the dog of the hill in winter.") == "en"
    assert mod._detect_script_group("한국어 문서를 테스트하고 있습니다. 품질 평가 예시입니다.") == "hangul"
    assert mod._detect_script_group("日本語のテスト文章です。ひらがなとカタカナと漢字。") == "ja"
    assert mod._detect_script_group("Русский текст для проверки качества данных.") == "cyrillic"
    assert mod._detect_script_group("هذا نص عربي لاختبار التوازن بين اللغات المختلفة.") == "arabic"

    # Build 30 English docs rated 4.0..5.0 and 30 Korean docs rated 1.5..2.5 (biased English judge!).
    cands = []
    scores = {}
    for i in range(30):
        cands.append((0.9, i, 1000, "The English technical article and the analysis of systems in detail.", "en"))
        scores[i] = round(4.0 + i / 30.0, 2)
    for i in range(30):
        doc_id = 100 + i
        cands.append((0.9, doc_id, 1000, "한국어 기술 문서와 과학 분석 예제입니다.", "hangul"))
        scores[doc_id] = round(1.5 + i / 30.0, 2)

    fake_task = types.SimpleNamespace(train_tokens=5400)  # ~20,250 bytes -> picks ~20 of 60 docs
    selected = mod.build_script(fake_task, cands, scores)
    en_kept = sum(1 for d in selected if d < 100)
    ko_kept = sum(1 for d in selected if d >= 100)
    assert en_kept >= 9 and ko_kept >= 9 and abs(en_kept - ko_kept) <= 1, f"expected balanced ~10/10, got en={en_kept} ko={ko_kept}"
    # Cooldown band (last 25%) also contains both English and Korean top-quantile docs!
    last_band = selected[-5:]
    assert any(d < 100 for d in last_band) and any(d >= 100 for d in last_band)


def test_reason_cjk_and_short_copy_filter_and_decoupled_wrappers():
    """Finding 6: copy filter rejects wholly extractive answers in zh/ja/ko and <5-word strings
    while keeping genuine paraphrases; synthesis cycles across all 3 wrappers independently of task."""
    from stet.exam.fresh.reason import ngram_overlap, parse_generated_items, select_source_docs, wrap_prompt

    # 1. Whole-copy rejected and legitimate paraphrase kept in en, zh, ja, ko!
    fixtures = [
        (
            "en",
            "The bridge collapsed because winter ice jammed the central piers and scoured the riverbed underneath.",
            "The bridge collapsed because winter ice jammed the central piers and scoured the riverbed underneath.",
            "Severe seasonal ice buildup blocked the middle supports and washed away their foundation, causing structural failure.",
        ),
        (
            "zh",
            "由于冬季河面结冰堵塞了中央桥墩并冲刷了河床地基，这座古老的石桥最终在清晨发生坍塌，导致两岸交通中断数周。",
            "由于冬季河面结冰堵塞了中央桥墩并冲刷了河床地基，这座古老的石桥最终在清晨发生坍塌，导致两岸交通中断数周。",
            "严寒冰块阻挡了支撑结构并淘空底部泥沙，使建筑失去稳定而倾覆，附近通勤也随之受阻。",
        ),
        (
            "ja",
            "冬の流氷が中央の橋脚を塞ぎ河床の基礎を削り取ったため、古い石橋は早朝に崩落して両岸の交通が数週間遮断された。",
            "冬の流氷が中央の橋脚を塞ぎ河床の基礎を削り取ったため、古い石橋は早朝に崩落して両岸の交通が数週間遮断された。",
            "厳しい寒さによる氷の堆積が支柱まわりを侵食し、土台が弱まった結果として構造物が倒れ、通行止めが続いた。",
        ),
        (
            "ko",
            "겨울철 유빙이 중앙 교각을 막고 강바닥 기초를 깎아내리면서 오래된 돌다리가 새벽에 무너져 양쪽 교통이 몇 주간 끊겼다.",
            "겨울철 유빙이 중앙 교각을 막고 강바닥 기초를 깎아내리면서 오래된 돌다리가 새벽에 무너져 양쪽 교통이 몇 주간 끊겼다.",
            "한파로 쌓인 얼음이 기둥 주변 지반을 약화시켜 다리가 버티지 못하고 파손되었으며 주민 이동에도 큰 차질이 생겼다.",
        ),
    ]
    for lang, src, exact_copy, paraphrase in fixtures:
        assert ngram_overlap(exact_copy, src) > 0.50, f"failed to catch copy in {lang}"
        assert ngram_overlap(paraphrase, src) <= 0.50, f"falsely rejected paraphrase in {lang}"
        sel_copy = [{"source": "ccnews", "id": f"art-{lang}-1", "lang": lang, "text": src, "excerpt": src, "task_type": "explain", "wrapper_id": 0}]
        kept_copy, drop_copy = parse_generated_items(sel_copy, [f"Explain why the structure failed in {lang}?\nRESPONSE: {exact_copy}"])
        assert not kept_copy and drop_copy.get("verbatim_copy") == 1, (lang, kept_copy, drop_copy)

        kept_para, drop_para = parse_generated_items(sel_copy, [f"Explain why the structure failed in {lang}?\nRESPONSE: {paraphrase}"])
        assert len(kept_para) == 1 and not drop_para, (lang, drop_para)
        # Finding 7: harvest doc ID preserved on reasoning item source!
        assert kept_para[0].source == f"explain:ccnews:art-{lang}-1"

    # Short (<5 word) verbatim copy must also be caught rather than returning 0.0!
    assert ngram_overlap("central piers scoured", "The central piers scoured quickly.") == 1.0

    # 2. Synthesis wrapper_id must decouple from task_type so all 3 synthesis wrappers appear!
    docs = [
        Doc(
            source="ccnews",
            id=f"en:{i}",
            url=f"https://example.org/{i}",
            lang="en",
            license="ccnews",
            published=1_800_000_000 + i,
            title=f"Story {i}",
            text=f"Unique article {i} discusses industrial policy and regional supply chains across multiple sectors. " * 8,
            topic="",
        )
        for i in range(36)
    ]
    selected = select_source_docs(docs, b"wrap-seed", target=36)
    synth_wrappers = {d["wrapper_id"] % 3 for d in selected if d["task_type"] == "synthesis"}
    explain_wrappers = {d["wrapper_id"] % 6 for d in selected if d["task_type"] == "explain"}
    infer_wrappers = {d["wrapper_id"] % 6 for d in selected if d["task_type"] == "infer"}
    assert synth_wrappers == {0, 1, 2}, f"synthesis stuck on wrappers {synth_wrappers}"
    assert explain_wrappers == {0, 1, 2, 3, 4, 5}
    assert infer_wrappers == {0, 1, 2, 3, 4, 5}
    # Distinct rendered strings across the 3 synthesis wrappers
    rendered = {wrap_prompt("Body.", "Summary lead-in.", "synthesis", w) for w in (0, 1, 2)}
    assert len(rendered) == 3


def test_reason_shared_token_bounded_excerpt_endpoint_pinning_and_replay():
    """Finding 5: teacher and student share ONE token-bounded excerpt; teacher prompt never loses
    closing PROMPT: cue; endpoint verifies model/revision; replay produces identical item digest."""
    from stet.exam.fresh.reason import (
        _verify_endpoint_model,
        apply_token_budget,
        bound_excerpt,
        build_teacher_prompt,
        parse_generated_items,
        select_source_docs,
    )
    from stet.exam.items import digest

    class WordTok:
        """Simple deterministic mock tokenizer: 1 word = 1 token."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [i for i, _ in enumerate(text.split())]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(f"tok{i}" for i in ids)

    # 600-word dense doc capped to 380 tokens -> shared excerpt identical for teacher and student.
    long_text = " ".join(f"word{i}" for i in range(600))
    doc = Doc(
        source="ccnews",
        id="long-1",
        url="https://example.org/long",
        lang="en",
        license="ccnews",
        published=1_800_000_001,
        title="Long",
        text=long_text,
        topic="",
    )
    selected = select_source_docs([doc], b"replay-seed", target=1, max_excerpt_chars=2400)
    bounded = apply_token_budget(selected, WordTok(), max_tokens=100, max_chars=2400)
    ex = bounded[0]["excerpt"]
    assert len(ex.split()) == 100
    t_prompt = build_teacher_prompt(ex, "en", bounded[0]["task_type"])
    assert ex in t_prompt and t_prompt.endswith("\nPROMPT:")

    gen = ["Why does this process stabilize output?\nRESPONSE: By limiting each pass to a fixed span, downstream stages stay consistent and avoid silent cutoffs."]
    items1, _ = parse_generated_items(bounded, gen)
    items2, _ = parse_generated_items(bounded, gen)
    assert len(items1) == 1
    assert ex in items1[0].ctx  # student receives the EXACT same bounded excerpt as teacher!
    assert digest(items1) == digest(items2)  # deterministic replay digest

    # Bound excerpt without tokenizer still respects max_chars cleanly.
    assert len(bound_excerpt("a " * 2000, max_chars=500)) <= 500

    # Endpoint model/revision verification rejects unpinned overrides!
    import os

    old_m = os.environ.get("STET_EXAMINER_LLM_MODEL")
    old_r = os.environ.get("STET_EXAMINER_LLM_REVISION")
    try:
        os.environ["STET_EXAMINER_LLM_MODEL"] = "some-other/model"
        with pytest.raises(RuntimeError, match="does not match"):
            _verify_endpoint_model("http://127.0.0.1:9999/v1", "Qwen/Qwen3-8B", "rev123", {})
        os.environ.pop("STET_EXAMINER_LLM_MODEL", None)
        os.environ["STET_EXAMINER_LLM_REVISION"] = "wrong-rev"
        with pytest.raises(RuntimeError, match="does not match"):
            _verify_endpoint_model("http://127.0.0.1:9999/v1", "Qwen/Qwen3-8B", "rev123", {})
    finally:
        if old_m is None:
            os.environ.pop("STET_EXAMINER_LLM_MODEL", None)
        else:
            os.environ["STET_EXAMINER_LLM_MODEL"] = old_m
        if old_r is None:
            os.environ.pop("STET_EXAMINER_LLM_REVISION", None)
        else:
            os.environ["STET_EXAMINER_LLM_REVISION"] = old_r


def test_examine_enforces_min_pillar_items_and_regenerates_corrupt_cache(tmp_path, monkeypatch):
    """Finding 4: official assembly enforces min_pillar_items after decontamination, writes
    reasoning cache atomically, and regenerates corrupt or incomplete cache files."""
    from conftest import make_corpus

    from stet.exam import examine
    from stet.exam.fresh import write_snapshot
    from stet.spec import official_round

    s3 = load_season("s3")
    rnd = official_round(s3, 1, {"round": 10, "randomness": "ab" * 32}, {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 100})
    scale = s3.scale("nano")
    corpus = make_corpus(tmp_path / "corpus", ["Unrelated corpus text about gardening and weather. " * 20] * 4)
    monkeypatch.setattr(data, "corpus", lambda r, sc: corpus)

    # Small snapshot with 6 docs -> teacher returns only 1 reasoning item -> MUST fail coverage check!
    docs = [
        Doc(
            source="wikipedia",
            id=f"en:{i}",
            url=f"https://en.wikipedia.org/wiki/T{i}",
            lang="en",
            license="CC BY-SA 4.0",
            published=1_800_000_000 + i,
            title=f"Article {i}",
            text=f"Paragraph one of article {i} explains key facts clearly. Paragraph two expands the reasoning in detail. Paragraph three gives outcomes and trade-offs. " * 4,
            topic=f"Q{i}",
        )
        for i in range(6)
    ]
    snap = tmp_path / "1800000000.jsonl"
    write_snapshot(docs, snap, {"hour": 1_800_000_000})

    calls = {"n": 0}

    def fake_synth(d, seed, cfg, shares=None, log=print):
        calls["n"] += 1
        return [
            Item(
                kind="answer",
                corner="explain/en/wikipedia",
                fmt="prose",
                lang="en",
                source="explain:wikipedia:en:0",
                ctx="Excerpt one.\n\nWhy?\n",
                cont="A sufficiently detailed explanation in own words that answers the prompt cleanly.",
            )
        ]

    monkeypatch.setattr("stet.exam.fresh.reason.synthesize_reasoning", fake_synth)

    work = tmp_path / "work"
    # Official assembly refuses 1 surviving reasoning item (min_pillar_items=20)!
    with pytest.raises(RuntimeError, match="below minimum of 20"):
        examine.assemble(rnd, scale, [(1_800_000_000, snap)], work, log=lambda m: None, measure=False, synthesize=True)
    assert calls["n"] == 1

    # Corrupt / empty reasoning cache must NOT silently skip synthesis on retry!
    rcaches = list(work.glob("reasoning-*.jsonl.zst"))
    assert len(rcaches) == 1
    rcaches[0].write_bytes(b"")  # simulate interrupted write!
    with pytest.raises(RuntimeError, match="below minimum of 20"):
        examine.assemble(rnd, scale, [(1_800_000_000, snap)], work, log=lambda m: None, measure=False, synthesize=True)
    assert calls["n"] == 2  # regenerated instead of decoding empty cache as 0 items!


def test_practice_cache_tracks_synthesis_mode_and_third_party_notices_shipped(tmp_path, monkeypatch):
    """Other Finding A & Finding 7: practice_key & baseline_path separate STET_SYNTHESIZE_PRACTICE
    modes, and THIRD_PARTY_NOTICES.md ships in package data covering every SFT dataset source."""
    import importlib.resources as res
    import importlib.util

    import zstandard as zstd

    from stet.exam.fresh import write_snapshot

    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    scale = s3.scale("nano")

    # Put 3 harvested hours into a temp cache dir
    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    hdir = data.harvest_dir()
    hdir.mkdir(parents=True, exist_ok=True)
    for h in (1_800_000_000, 1_800_003_600, 1_800_007_200):
        write_snapshot([], hdir / f"{h}.jsonl", {"hour": h})

    has_torch = importlib.util.find_spec("torch") is not None
    pipeline = importlib.import_module("stet.pipeline") if has_torch else None

    monkeypatch.setattr("stet.exam.fresh.reason.can_synthesize_locally", lambda: True)
    monkeypatch.delenv("STET_SYNTHESIZE_PRACTICE", raising=False)
    k_off = data.practice_key(rnd)
    b_off = pipeline.baseline_path(rnd, scale, types.SimpleNamespace(type="cpu")) if pipeline else None

    monkeypatch.setenv("STET_SYNTHESIZE_PRACTICE", "1")
    k_on = data.practice_key(rnd)
    b_on = pipeline.baseline_path(rnd, scale, types.SimpleNamespace(type="cpu")) if pipeline else None
    assert k_off and k_on and k_off != k_on
    if pipeline is not None:
        assert b_off != b_on

    # Finding 7: THIRD_PARTY_NOTICES.md shipped inside stet.exam.data and covers every SFT source!
    notices = res.files("stet.exam").joinpath("data", "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    raw = zstd.ZstdDecompressor().decompress(
        res.files("stet.exam").joinpath("data", "s3-sft-ml.jsonl.zst").read_bytes(), max_output_size=64 << 20
    ).decode("utf-8")
    sources = {json.loads(ln)["source"] for ln in raw.splitlines() if ln.strip()}
    for src in sources:
        assert src in notices, f"missing notice for SFT source {src!r}"
    assert "CC BY-SA 4.0" in notices and "XStoryCloze" in notices and "s3-practice.jsonl.zst" in notices
