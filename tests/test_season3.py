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
    assert len(pack) == 508
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

    # 2. Pure context-insensitive style boost (ctx_boost = 0) shifts raw answer score hugely
    #    (> +1.5 bits/byte!), yet scores EXACTLY 0.0 on twin-context comprehension.
    style_lo = ContextAndStyleLM(style_boost=0.0, ctx_boost=0.0)
    style_hi = ContextAndStyleLM(style_boost=4.0, ctx_boost=0.0)
    ans_lo = exam_score.sit(style_lo, [ans_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    ans_hi = exam_score.sit(style_hi, [ans_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    ctx_hi = exam_score.sit(style_hi, [ctx_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    assert ans_hi - ans_lo > 1.5  # proves style_boost genuinely moves continuation likelihood!
    assert ctx_hi == pytest.approx(0.0, abs=1e-6)  # context-insensitive style predictor scores 0

    # 3. Real context comprehension (ctx_boost > 0) produces a strong positive twin-context score.
    smart = ContextAndStyleLM(style_boost=0.0, ctx_boost=3.0)
    smart_score = exam_score.sit(smart, [ctx_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    assert smart_score > 3.0

    # 4. Combined context + unconditional style boost: because softmax is nonlinear, changing
    #    unconditional token preferences DOES move a context-sensitive predictor's twin-context
    #    score (here saturation lowers it from ~3.65 to ~0.61 bpb rather than manufacturing gain).
    combined = ContextAndStyleLM(style_boost=4.0, ctx_boost=3.0)
    combined_score = exam_score.sit(combined, [ctx_item], enc, eot=0, ctx=64, device=torch.device("cpu"))[0]
    assert 0.0 < combined_score < 1.0 < smart_score


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

    # 3. 11 GB VRAM gate in _verify rejects an 8 GB CUDA GPU for Season 3 (12 GB floor).
    monkeypatch.delenv("STET_DEV", raising=False)
    monkeypatch.setattr(client, "need_key", lambda: types.SimpleNamespace(handle="v", public="0" * 64, private="0" * 64))
    monkeypatch.setattr(client, "request", lambda method, path, *a, **kw: {"scale": "standard"})
    monkeypatch.setattr(tr, "pick_device", lambda dev=None: types.SimpleNamespace(type="cuda"))
    monkeypatch.setattr(tr, "hardware_info", lambda dev: {"device": "cuda", "gpu": "RTX 3050", "vram_gb": 8.0, "backend": "cuda"})
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
    prompts_by_lang = {
        "en": "Why did the bridge collapse during the winter freeze?",
        "zh": "这座石桥为何在冬季清晨发生坍塌？",
        "ja": "この古い石橋が早朝に崩落した理由は何ですか？",
        "ko": "이 오래된 돌다리가 새벽에 무너진 이유는 무엇인가요?",
    }
    for lang, src, exact_copy, paraphrase in fixtures:
        assert ngram_overlap(exact_copy, src) > 0.50, f"failed to catch copy in {lang}"
        assert ngram_overlap(paraphrase, src) <= 0.50, f"falsely rejected paraphrase in {lang}"
        sel_copy = [{"source": "ccnews", "id": f"art-{lang}-1", "lang": lang, "text": src, "excerpt": src, "task_type": "explain", "wrapper_id": 0}]
        pr = prompts_by_lang[lang]
        kept_copy, drop_copy = parse_generated_items(sel_copy, [f"{pr}\nRESPONSE: {exact_copy}"])
        assert not kept_copy and drop_copy.get("verbatim_copy") == 1, (lang, kept_copy, drop_copy)

        kept_para, drop_para = parse_generated_items(sel_copy, [f"{pr}\nRESPONSE: {paraphrase}"])
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
    assert ex in t_prompt and t_prompt.endswith("\nPROMPT: ") and "PROMPT: ..." not in t_prompt

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

    def fake_synth(d, seed, cfg, shares=None, log=print, drops_path=None):
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
    # Official assembly drops sparse explain (1 < 20), leaving reasoning group empty -> fails loudly!
    with pytest.raises(RuntimeError, match="have no pillars meeting minimum of 20"):
        examine.assemble(rnd, scale, [(1_800_000_000, snap)], work, log=lambda m: None, measure=False, synthesize=True)
    assert calls["n"] == 1

    # Corrupt / empty reasoning cache must NOT silently skip synthesis on retry!
    rcaches = list(work.glob("reasoning-*.jsonl.zst"))
    assert len(rcaches) == 1
    rcaches[0].write_bytes(b"")  # simulate interrupted write!
    with pytest.raises(RuntimeError, match="have no pillars meeting minimum of 20"):
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

    monkeypatch.setattr("stet.exam.fresh.reason.can_synthesize_locally", lambda *a, **k: True)
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


def test_review6_filter_pillar_support_omits_sparse_pillar_and_refuses_empty_group():
    """Review 6 Finding 1: drop sparse/missing pillars (such as news=0) cleanly so their share
    goes to the rest of their group, and refuse ONLY when an entire declared group is empty."""
    from stet.exam import examine

    s3 = load_season("s3")
    items: list[Item] = []
    # Full reference, technical, conversation, explain, infer, synthesis (>=20 each); sparse news (5 < 20)
    for src, corner, kind in [
        ("wikipedia:en:0", "fresh/reference/en", "context"),
        ("github:0", "fresh/technical/en", "context"),
        ("hackernews:0", "fresh/conversation/en", "context"),
    ]:
        for i in range(20):
            items.append(Item(kind=kind, corner=corner, fmt="prose", lang="en", source=f"{src}:{i}", ctx=f"Open {i}.", cont=f" Cont {i}.", wrongs=("Wrong A.", "Wrong B.")))
    for i in range(5):
        items.append(Item(kind="context", corner="fresh/news/en", fmt="prose", lang="en", source=f"ccnews:n{i}", ctx=f"News {i}.", cont=f" End {i}.", wrongs=("Wrong N.",)))
    for rkind in ("explain", "infer", "synthesis"):
        for i in range(20):
            items.append(Item(kind="answer", corner=f"{rkind}/en/wikipedia", fmt="prose", lang="en", source=f"{rkind}:wikipedia:en:{i}", ctx=f"Ex {i}.\n\nLead {i}.\n", cont=f"Answer {i} explains mechanism thoroughly."))

    logs: list[str] = []
    kept = examine._filter_pillar_support(items, s3, "s3-o00001", logs.append)
    assert all(not it.source.startswith("ccnews:") for it in kept)
    assert len(kept) == 120
    assert any("omitting under-supported pillars" in m and "news=5" in m for m in logs)

    # Removing every reasoning item leaves the reasoning group empty -> must refuse!
    comp_only = [it for it in kept if it.kind == "context"]
    with pytest.raises(RuntimeError, match="exam group\\(s\\) reasoning have no pillars meeting minimum of 20"):
        examine._filter_pillar_support(comp_only, s3, "s3-o00001", lambda _: None)


def test_review6_teacher_prompt_parser_quality_and_min_keep_rate(monkeypatch):
    """Review 6 Findings 2 & 8: remove '...' placeholders, reject instruction echoes, prompt-copying
    answers, U+FFFD, mid-sentence truncations, wrong-language responses, and enforce min_keep_rate."""
    from stet.exam.fresh import reason

    # Prompts for all 3 task types must not contain literal '...' placeholder
    for task in ("explain", "infer", "synthesis"):
        p = reason.build_teacher_prompt("A real article discussing climate adaptation and urban grids.", "en", task)
        assert "PROMPT: ..." not in p and "RESPONSE: ..." not in p
        assert p.endswith("\nPROMPT: ")

    src_en = "Municipal transit agencies replaced diesel fleets with battery buses after depot chargers reduced overnight peak demand."
    doc_en = {"source": "ccnews", "id": "en-1", "lang": "en", "text": src_en, "excerpt": src_en, "task_type": "synthesis", "wrapper_id": 0}

    bad_cases = [
        ("placeholder", "...\nRESPONSE: Upgraded charging depots allowed municipal fleets to switch smoothly to electric vehicles."),
        ("echo", "Write one plain declarative lead-in sentence for this excerpt.\nRESPONSE: Upgraded charging depots allowed municipal fleets to switch smoothly to electric vehicles."),
        ("prompt_copy", "Upgraded charging depots let transit fleets replace diesel vehicles across the city.\nRESPONSE: Upgraded charging depots let transit fleets replace diesel vehicles across the city without disruption."),
        ("bad_utf8", "Summary of the fleet transition.\nRESPONSE: Upgraded charging depots allowed electric\ufffd fleets to cut overnight loads."),
        ("truncated", "Summary of the fleet transition.\nRESPONSE: Upgraded charging depots allowed municipal fleets to transition smoothly to battery buses while reducing"),
        ("wrong_script", "Summary of the fleet transition.\nRESPONSE: Ночные зарядные станции позволили городским автопаркам плавно перейти с дизельных автобусов на электрические."),
    ]
    if reason._langid is not None:
        bad_cases.append(
            ("wrong_lang", "Summary of the fleet transition.\nRESPONSE: Los depósitos de carga nocturna permitieron a las agencias municipales sustituir autobuses diésel por vehículos eléctricos con menor coste energético.")
        )
    for label, raw_out in bad_cases:
        kept, drops = reason.parse_generated_items([doc_en], [raw_out])
        assert not kept, f"{label} should have been rejected, got {kept}"
        assert sum(drops.values()) == 1, (label, drops)

    # Valid non-English CJK + Cyrillic responses with script-native sentence terminators pass!
    src_ja = "新しい鉄道網の整備により、地方都市から首都圏への通勤時間が大幅に短縮され、沿線の人口増加が続いている。"
    doc_ja = {"source": "wikipedia", "id": "ja-1", "lang": "ja", "text": src_ja, "excerpt": src_ja, "task_type": "explain", "wrapper_id": 1}
    ok_ja = "鉄道の高速化が沿線地域の人口動態に与えた影響を説明してください。\nRESPONSE: 移動に要する時間が減ったことで都市部での就労と郊外での居住を両立しやすくなり、移住者が増える結果につながった。"
    kept_ja, drops_ja = reason.parse_generated_items([doc_ja], [ok_ja])
    assert len(kept_ja) == 1 and not drops_ja, drops_ja

    # min_keep_rate enforcement fails loudly when any reasoning kind keeps <50% of drafts!
    docs = [
        Doc(
            source="ccnews",
            id=f"d-{i}",
            url=f"https://example.org/{i}",
            lang="en",
            license="ccnews",
            published=1_800_000_000 + i,
            title=f"Story {i}",
            text=f"Detailed article {i} examining agricultural irrigation sensors, reservoir telemetry, and seasonal rainfall patterns across valleys. " * 4,
            topic="",
        )
        for i in range(45)
    ]
    monkeypatch.setattr(reason, "_load_teacher_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(
        reason,
        "_generate_local_gpu",
        lambda prompts, *a, **k: ["...\nRESPONSE: bad" for _ in prompts],
    )
    cfg = {"model": "Qwen/Qwen3-8B", "revision": "rev", "items": 45, "min_keep_rate": 0.5}
    with pytest.raises(RuntimeError, match="keep rate"):
        reason.synthesize_reasoning(docs, b"keep-rate-seed", cfg, log=lambda _: None)


def test_review6_withheld_reasoning_published_and_reconstructible_and_fail_loud(monkeypatch):
    """Review 6 Finding 3: split_wrapped_ctx is a lossless inverse of all 9 wrappers; public_items
    publishes teacher prompt+cont+wrapper_id+excerpt_sha256 for withheld reasoning items;
    rebuild_withheld_reasoning verifies exact Item.id; tokenizer & empty /v1/models fail loudly."""
    from stet.exam import examine
    from stet.exam.fresh import reason

    excerpt = "Researchers measured glacier retreat across twelve valleys using radar altimetry.\nSecond paragraph details seasonal melt rates."
    prompt = "Why did melt rates accelerate after 2022?"
    cont = "Warmer summer nights prevented surface refreezing, allowing meltwater to lubricate basal ice flow."

    # 1. Lossless round-trip across ALL 6 explain/infer wrappers + ALL 3 synthesis wrappers!
    for task, n_wrap in (("explain", 6), ("infer", 6), ("synthesis", 3)):
        for wid in range(n_wrap):
            ctx = reason.wrap_prompt(excerpt, prompt, task, wid)
            got_ex, got_pr, got_wid = reason.split_wrapped_ctx(ctx, task)
            assert got_ex == excerpt and got_pr == prompt and got_wid == wid, (task, wid, got_ex, got_pr, got_wid)

    # 2. public_items publishes teacher text for non-OPEN reasoning items, and rebuild reproduces exact id!
    item = Item(
        kind="answer",
        corner="explain/en/ccnews",
        fmt="prose",
        lang="en",
        source="explain:ccnews:story-42",
        ctx=reason.wrap_prompt(excerpt, prompt, "explain", 2),
        cont=cont,
    )
    pub = examine.public_items([item])
    assert len(pub) == 1
    row = pub[0]
    assert row["withheld"] is True and row["withheld_excerpt"] is True
    assert row["prompt"] == prompt and row["cont"] == cont and row["wrapper_id"] == 2
    assert row["excerpt_sha256"] == hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    assert row["excerpt_chars"] == len(excerpt)
    assert excerpt not in json.dumps(row)  # third-party excerpt itself stays withheld!

    rebuilt = examine.rebuild_withheld_reasoning(row, excerpt)
    assert rebuilt == item and rebuilt.id == item.id
    with pytest.raises(ValueError, match="excerpt sha256 mismatch"):
        examine.rebuild_withheld_reasoning(row, excerpt + " tampered")

    # 3. _load_teacher_tokenizer fails loudly when transformers is missing or fails to load pinned tokenizer.
    if importlib.util.find_spec("transformers") is not None:
        import transformers as _tf

        monkeypatch.setattr(_tf.AutoTokenizer, "from_pretrained", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(RuntimeError, match="could not load pinned reasoning teacher tokenizer"):
        reason._load_teacher_tokenizer("Qwen/Qwen3-8B", "rev", required=True)

    # 4. _verify_endpoint_model rejects empty /v1/models list!
    class EmptyModelsResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"data": []}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: EmptyModelsResp())
    with pytest.raises(RuntimeError, match=r"serves \[\]"):
        reason._verify_endpoint_model("http://127.0.0.1:9999/v1", "Qwen/Qwen3-8B", "rev", {})


def test_review6_crawl_share_weighted_other_pool_and_source_selection():
    """Review 6 Findings 6 & 7: weighted 'other' pool prevents tiny micro-languages from outweighing
    larger pooled languages (ja/zh); select_source_docs enforces major-lang floor >=27, drops <0.05%
    micro-langs when major langs exist, and collapses Wikipedia variants into one bucket."""
    from stet.exam.aggregate import round_score
    from stet.exam.fresh.reason import select_source_docs

    # 1. Weighted 'other' pool: ja (5.53%) + zh (4.54%) at +0.20 vs 5 tiny langs (0.01-0.05%) at -0.40
    #    Under unweighted mean the 10 tiny items would drag 'other' strongly negative; under crawl-share
    #    weighting ja+zh hold >98% of the 'other' weight so 'other' stays near +0.19!
    shares = {"en": 42.0, "ja": 5.53, "zh": 4.54, "ca": 0.05, "ky": 0.01, "sw": 0.03}
    items = (
        [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source=f"explain:wikipedia:en:{i}", ctx=f"E{i}", cont=f"A{i}") for i in range(12)]
        + [Item(kind="answer", corner="explain/ja/wiki", fmt="prose", lang="ja", source=f"explain:wikipedia:ja:{i}", ctx=f"J{i}", cont=f"A{i}") for i in range(6)]
        + [Item(kind="answer", corner="explain/zh/wiki", fmt="prose", lang="zh", source=f"explain:wikipedia:zh:{i}", ctx=f"Z{i}", cont=f"A{i}") for i in range(6)]
        + [Item(kind="answer", corner=f"explain/{lg}/wiki", fmt="prose", lang=lg, source=f"explain:wikipedia:{lg}:{i}", ctx=f"T{lg}{i}", cont=f"A{i}") for lg in ("ca", "ky", "sw", "yo", "eo") for i in range(2)]
    )
    gains = [0.0] * 12 + [0.20] * 6 + [0.20] * 6 + [-0.40] * 10
    res = round_score(items, gains, shares=shares)
    # With ja (5.53%) + zh (4.54%) holding ~98% of the pooled "other" weight (vs 0.19% for 5 tiny
    # langs), "other" stays strongly positive (~+0.19) so overall explain score is > +0.03 rather
    # than being dragged negative by the 10 tiny-language items!
    assert res["score"] > 0.03, f"tiny languages drowned out ja/zh in 'other': {res}"

    # 2. select_source_docs: major langs (>=1%) get >=27 docs, <0.05% micro-langs excluded, wiki variants collapsed!
    docs: list[Doc] = []
    # 50 docs each in en (42%), ja (5.5%), zh (4.5%), plus 20 each in 10 micro-langs (0.01%)
    # and split across wikipedia, wikipedia-added, wikipedia-talk, ccnews, github, discourse
    sources_en = ["wikipedia", "wikipedia-added", "wikipedia-talk", "ccnews", "github", "discourse"]
    for i in range(180):
        src = sources_en[i % len(sources_en)]
        docs.append(Doc(source=src, id=f"en-{i}", url=f"https://example.org/en/{i}", lang="en", license="CC BY-SA 4.0", published=1_800_000_000 + i, title=f"EN {i}", text=f"English technical document number {i} covering protocols and architecture details. " * 6, topic=""))
    for lg in ("ja", "zh"):
        for i in range(40):
            docs.append(Doc(source="wikipedia", id=f"{lg}-{i}", url=f"https://example.org/{lg}/{i}", lang=lg, license="CC BY-SA 4.0", published=1_800_000_000 + i, title=f"{lg} {i}", text=f"Long multilingual passage {lg} {i} explaining mechanisms and context in depth. " * 6, topic=""))
    for lg in ("ky", "tt", "eu", "gl", "nn", "af"):
        for i in range(15):
            docs.append(Doc(source="wikipedia", id=f"{lg}-{i}", url=f"https://example.org/{lg}/{i}", lang=lg, license="CC BY-SA 4.0", published=1_800_000_000 + i, title=f"{lg} {i}", text=f"Micro language passage {lg} {i} discussing regional history and infrastructure. " * 6, topic=""))

    sel = select_source_docs(docs, b"lang-quota-seed", target=160, shares={"en": 42.0, "ja": 5.5, "zh": 4.5, "ky": 0.01, "tt": 0.01, "eu": 0.02, "gl": 0.02, "nn": 0.01, "af": 0.01})
    by_lang: dict[str, int] = {}
    en_wiki = 0
    en_non_wiki = 0
    for d in sel:
        by_lang[d["lang"]] = by_lang.get(d["lang"], 0) + 1
        if d["lang"] == "en":
            if d["source"].startswith("wikipedia"):
                en_wiki += 1
            else:
                en_non_wiki += 1
    assert by_lang.get("ja", 0) >= 27 and by_lang.get("zh", 0) >= 27, by_lang
    assert all(by_lang.get(lg, 0) == 0 for lg in ("ky", "tt", "eu", "gl", "nn", "af")), by_lang
    # Collapsed Wikipedia bucket must take ~1/4 of English docs (<=35%), NOT 3/6 (50%)!
    assert en_wiki <= en_non_wiki * 0.55, f"Wikipedia variants still dominated English selection: wiki={en_wiki}, non_wiki={en_non_wiki}"


def test_review6_reader_vllm_partial_exception_falls_back_to_transformers(monkeypatch):
    """Review 6 Smaller Point 1: if _score_vllm raises mid-batch after scoring some docs, it must
    return False (not True) so main() invokes _score_transformers on the remaining unrated docs."""
    torch = pytest.importorskip("torch")

    root = ROOT / "examples" / "reader"
    spec = importlib.util.spec_from_file_location("reader_partial_fallback_test", root / "curator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    @dataclasses.dataclass
    class FakeLogprob:
        logprob: float

    class FakeTok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(ch) for ch in text[:16]]

    class FlakyLLM:
        def __init__(self, **kwargs):
            self.calls = 0

        def get_tokenizer(self):
            return FakeTok()

        def generate(self, prompts, params, use_tqdm=False):
            self.calls += 1
            if self.calls == 1:
                lps = {tid: FakeLogprob(logprob=-0.1 * (i + 1)) for i, tid in enumerate(params.logprob_token_ids)}
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(logprobs=[lps])]) for _ in prompts]
            raise RuntimeError("transient CUDA failure on batch 2")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FlakyLLM, SamplingParams=FakeSamplingParams))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    fake_task = types.SimpleNamespace(device=lambda: "cuda", time_left=lambda: 600.0, train_tokens=2000, write=lambda lines: None)
    # 520 candidates -> batch 1 scores 512 docs, batch 2 raises -> must return False while keeping the 512 scores!
    cands = [(1.0, i, 200, f"Sample document {i} with clear prose.", "en") for i in range(520)]
    scores: dict[int, float] = {}
    ok = mod._score_vllm(fake_task, "/fake/model", cands, scores)
    assert ok is False, "partial mid-batch vLLM crash must return False so transformers finishes the rest"
    assert len(scores) == 512 and 0 in scores and 511 in scores and 512 not in scores


def test_review6_authoritative_init_sha256_matches_build_model(monkeypatch):
    """Review 6 Smaller Point 6: examine._round_init_sha256 is deterministic and matches
    train.build_model + train.weights_sha256 for the same round seed."""
    torch = pytest.importorskip("torch")
    from stet import train as tr
    from stet.exam import examine

    class TinyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4, 4))

    def fake_build(scale, season, init_seed=None, device=None):
        g = torch.Generator().manual_seed(int(init_seed or 0))
        m = TinyStudent()
        with torch.no_grad():
            m.w.copy_(torch.randn(4, 4, generator=g))
        return m

    monkeypatch.setattr(tr, "build_model", fake_build)
    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    scale = s3.scale("nano")

    h1 = examine._round_init_sha256(rnd, scale)
    h2 = examine._round_init_sha256(rnd, scale)
    expected = tr.weights_sha256(fake_build(scale, s3, rnd.init_seed, torch.device("cpu")))
    assert h1 == h2 == expected and len(h1) == 64


def test_review7_slim_result_carries_baseline_init_sha256():
    """Review 7 Finding 2: client.slim_result carries baseline.init_sha256 from the baseline run
    itself, both on valid grades and when the graded script is invalid."""
    from stet.client import slim_result

    base_init = "b" * 64
    exam_init = "a" * 64
    base = {
        "valid": True,
        "exam": {"mean": -2.5, "item_scores": [-2.5, -2.5]},
        "train": {"init_sha256": base_init},
    }
    valid_res = {
        "round": {"id": "s3-o00001"},
        "valid": True,
        "script": {"script_sha256": "1" * 64, "tokens_sha256": "2" * 64},
        "exam": {"mean": -2.4, "item_scores": [-2.4, -2.4]},
        "train": {"seconds": 10.0, "tokens_per_second": 1000.0, "final_train_loss": 2.4, "init_sha256": exam_init, "precision": "bf16", "curve": [[1, 3.0]]},
    }
    slim_valid = slim_result(valid_res, base)
    assert slim_valid["baseline"]["init_sha256"] == base_init
    assert slim_valid["train"]["init_sha256"] == exam_init

    invalid_res = {"round": {"id": "s3-o00001"}, "valid": False, "invalid_reason": "bad script"}
    slim_invalid = slim_result(invalid_res, base)
    assert slim_invalid["baseline"]["init_sha256"] == base_init
    assert "train" not in slim_invalid


def test_review7_official_replay_uses_recorded_language_shares(monkeypatch, tmp_path):
    """Review 7 Finding 5: official CLI replays pass the round's recorded language shares
    through `stet score` and `stet run` (`_with_baseline`) instead of falling back to SHARES."""
    from stet import cli, client, data
    from stet.exam.aggregate import plan_of, round_score
    from stet.exam.items import Item
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    rnd = official_round(
        s3,
        1,
        {"round": 123456, "randomness": "ab" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 100000},
    )
    rnd = dataclasses.replace(rnd, exam_digest="e" * 64)
    recorded_shares = {"en": 20.0, "ja": 80.0}
    client.fetch_round_shares(rnd, {"status": "settled", "languages": recorded_shares})
    assert data.round_shares(rnd) == recorded_shares

    items = (
        [Item(kind="answer", corner=f"explain/en/wiki/{i}", fmt="prose", lang="en", source=f"explain:wikipedia:en-{i}", ctx="Q", cont="A") for i in range(8)]
        + [Item(kind="answer", corner=f"explain/ja/wiki/{i}", fmt="prose", lang="ja", source=f"explain:wikipedia:ja-{i}", ctx="Q", cont="A") for i in range(8)]
    )
    base_scores = [-3.0] * len(items)
    # English improves by +0.10, Japanese regresses by -0.10 -> default SHARES (~43% en) vs recorded (20% en / 80% ja) differ sharply!
    exam_scores = [-2.9] * 8 + [-3.1] * 8

    monkeypatch.setattr(data, "round_exam", lambda r, sc: items)
    monkeypatch.setitem(
        sys.modules,
        "stet.pipeline",
        types.SimpleNamespace(baseline=lambda *a, **kw: {"valid": True, "exam": {"mean": -3.0, "item_scores": base_scores}}),
    )

    result = {"valid": True, "exam": {"mean": -3.0, "item_scores": exam_scores}}
    cli._with_baseline(result, rnd, s3.scale("nano"), None, cli.Out(quiet=True), skip=False)

    want_recorded = round_score(items, [a - b for a, b in zip(exam_scores, base_scores, strict=True)], shares=recorded_shares, plan=plan_of(s3))["score"]
    want_default = round_score(items, [a - b for a, b in zip(exam_scores, base_scores, strict=True)], shares=None, plan=plan_of(s3))["score"]
    assert abs(want_recorded - want_default) > 0.05
    assert result["gain"] == pytest.approx(round(want_recorded, 5), abs=1e-5)


def test_review7_practice_synthesis_on_then_off_ignores_reasoning_cache(monkeypatch, tmp_path):
    """Review 7 Finding 6: running practice with synthesis ON (populating the inner reasoning
    cache) followed by synthesis OFF must NOT load cached reasoning items into the OFF build."""
    from stet import data
    from stet.exam import examine
    from stet.exam.fresh import Doc, write_snapshot
    from stet.exam.items import Item

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    scale = s3.scale("nano")

    # Write a minimal harvested snapshot and fake corpus so assemble runs completely offline.
    snap = tmp_path / "1800000000.jsonl"
    docs = [
        Doc(source="wikipedia", id=f"w{i}", url=f"https://en.wikipedia.org/wiki/T{i}", lang="en", license="CC BY-SA 4.0",
            published=1_800_000_000 + i, title=f"Topic {i}",
            text=f"Paragraph one of article {i} introducing background details cleanly.\n\nParagraph two of article {i} continuing the explanation in clear prose with plenty of bytes.", topic="")
        for i in range(10)
    ]
    write_snapshot(docs, snap, {"hour": 1_800_000_000})

    fake_corpus = types.SimpleNamespace(meta={"sha256": "0" * 64})
    monkeypatch.setattr(data, "corpus", lambda r, sc: fake_corpus)
    monkeypatch.setattr("stet.exam.decontam.seen", lambda fresh, corpus, season, log=None: set())

    def fake_select_context(docs_in, seed, target, allowed, per_corner=2, shares=None):
        return [
            Item(kind="context", corner=f"fresh/wikipedia/en/c{i}", fmt="prose", lang="en",
                 source=f"wikipedia:w{i}", ctx=f"Opening {i}", cont=f" Continuation {i}", wrongs=(f"Wrong {i}",))
            for i in range(4)
        ]

    def fake_synth(docs_in, seed, r_cfg, shares=None, log=None, drops_path=None):
        return [
            Item(kind="answer", corner=f"{kind}/en/wikipedia", fmt="prose", lang="en",
                 source=f"{kind}:wikipedia:w{i}", ctx=f"Excerpt {i}\nQ?", cont=f" Answer {i}.")
            for kind in ("explain", "infer", "synthesis")
            for i in range(20)
        ]

    monkeypatch.setattr("stet.exam.fresh.select.select_context", fake_select_context)
    monkeypatch.setattr("stet.exam.fresh.reason.synthesize_reasoning", fake_synth)

    work_warm = tmp_path / "work_warm"
    work_cold = tmp_path / "work_cold"
    # Cold OFF build
    cold_off, _, _, _ = examine.assemble(rnd, scale, [(1_800_000_000, snap)], work_cold, log=lambda m: None, measure=False, synthesize=False, check_coverage=False)
    assert not any(it.source.startswith(("explain:", "infer:", "synthesis:")) for it in cold_off)

    # Warm ON build (populates inner reasoning cache) followed by OFF build on the same work dir
    warm_on, _, _, _ = examine.assemble(rnd, scale, [(1_800_000_000, snap)], work_warm, log=lambda m: None, measure=False, synthesize=True, check_coverage=False)
    assert sum(1 for it in warm_on if it.source.startswith(("explain:", "infer:", "synthesis:"))) == 60

    warm_off, _, _, _ = examine.assemble(rnd, scale, [(1_800_000_000, snap)], work_warm, log=lambda m: None, measure=False, synthesize=False, check_coverage=False)
    assert not any(it.source.startswith(("explain:", "infer:", "synthesis:")) for it in warm_off)
    assert [it.id for it in warm_off] == [it.id for it in cold_off]


def test_review7_prompt_language_and_foreign_script_leak_filter():
    """Review 7 Finding 9: quality checks reject mixed-language/code-switched teacher prompts
    (e.g., Thai prefix in Arabic prompt) and foreign non-Latin script leaks in prompts or responses,
    while allowing valid Japanese kanji+kana mixes."""
    from stet.exam.fresh import reason

    ar_excerpt = "تمكنت الجهات الرقابية من ضبط مصنع غير مرخص لتصنيع المخللات يفتقر إلى أبسط الاشتراطات الصحية والإنشائية."
    he_excerpt = "הדיונים המדיניים התמקדו בתוכנית עשרים הנקודות ובשאלת כוח הייצוב הבינלאומי ברצועת עזה."
    ru_excerpt = "Виноделие в Пьемонте является важной частью экономики региона и опирается на многовековые традиции производства."
    ja_excerpt = "日本の鉄道網は高い定時性と安全性を誇り、都市間の移動や地域経済の発展に大きく貢献している。"

    selected = [
        # 1. Exact Review 7 Thai-in-Arabic prompt bug -> MUST drop (wrong_prompt_lang)
        {"task_type": "explain", "source": "ccnews", "id": "ar-thai-pr", "lang": "ar", "wrapper_id": 0, "text": ar_excerpt, "excerpt": ar_excerpt},
        # 2. Han-in-Hebrew prompt leak -> MUST drop (wrong_prompt_lang)
        {"task_type": "explain", "source": "ccnews", "id": "he-han-pr", "lang": "he", "wrapper_id": 0, "text": he_excerpt, "excerpt": he_excerpt},
        # 3. Han-in-Russian response leak -> MUST drop (wrong_lang)
        {"task_type": "synthesis", "source": "wikipedia", "id": "ru-han-re", "lang": "ru", "wrapper_id": 0, "text": ru_excerpt, "excerpt": ru_excerpt},
        # 4. Pure English prompt on Arabic excerpt -> MUST drop (wrong_prompt_lang)
        {"task_type": "explain", "source": "ccnews", "id": "ar-en-pr", "lang": "ar", "wrapper_id": 0, "text": ar_excerpt, "excerpt": ar_excerpt},
        # 5. Clean Arabic prompt + response -> MUST pass
        {"task_type": "explain", "source": "ccnews", "id": "ar-clean", "lang": "ar", "wrapper_id": 0, "text": ar_excerpt, "excerpt": ar_excerpt},
        # 6. Clean Japanese kanji+kana prompt + response -> MUST pass
        {"task_type": "explain", "source": "wikipedia", "id": "ja-clean", "lang": "ja", "wrapper_id": 0, "text": ja_excerpt, "excerpt": ja_excerpt},
    ]
    generations = [
        "ทำไมการตรวจพบโรงงาน تصنيع المخللات غير المرخصة يعتبر خطراً على الصحة العامة؟\nRESPONSE: لأن الموقع غير مجهز بأي اشتراطات صحية أو إنشائية، مما يؤدي إلى تلوث المنتجات بالأتربة والحشرات.",
        "למה דרשו להתקדם עם תוכנית 20 הנקודות ול接纳 כוח הייצוב הבינלאומי?\nRESPONSE: המהלך נועד לייצב את המצב הביטחוני וליצור מסגרת ניהולית ברורה יותר לאזור כולו.",
        "Виноделие в Пьемонте играет важную роль в хозяйстве региона на протяжении многих поколений.\nRESPONSE: Отрасль обеспечивает значительные объёмы экспорта благодаря традициям и развитию знаменитых винных产区.",
        "Why does operating an unlicensed pickling facility pose a serious threat to public health?\nRESPONSE: لأن الموقع غير مجهز بأي اشتراطات صحية أو إنشائية، مما يؤدي إلى تلوث المنتجات بالأتربة والحشرات.",
        "لماذا يشكل تشغيل مصنع مخللات غير مرخص خطراً كبيراً على سلامة المستهلكين؟\nRESPONSE: لأن غياب التجهيزات الصحية والرقابة يعرض المنتجات للتلوث البيئي والبكتيري، مما يهدد صحة الأفراد.",
        "日本の鉄道網が地域社会と経済の発展を支える上で重要な理由は何ですか？\nRESPONSE: 正確な運行と高い安全基準がビジネスや観光の移動を円滑にし、広域的な経済活動を後押ししているためです。",
    ]

    kept, dropped = reason.parse_generated_items(selected, generations, check_lang=True)
    kept_sources = {it.source for it in kept}
    assert kept_sources == {"explain:ccnews:ar-clean", "explain:wikipedia:ja-clean"}, (kept_sources, dropped)
    assert dropped.get("wrong_prompt_lang") == 3, dropped
    assert dropped.get("wrong_lang") == 1, dropped


def test_review8_training_divergence_retries_once_then_raises_training_error(monkeypatch, tmp_path):
    """Review 8 Finding 3: a FloatingPointError in train() retries once on the same machine; if it
    recovers on attempt 2, grading succeeds; if it fails twice, raises TrainingError (NOT valid=False).
    ScriptError still returns valid=False immediately without retrying or training."""
    pytest.importorskip("torch")
    from stet import grade
    from stet.script import ScriptError

    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    scale = s3.scale("nano")
    corpus = types.SimpleNamespace(meta={"sha256": "a" * 64})
    script_path = tmp_path / "script.jsonl"
    script_path.write_text("0\n")

    fake_stats = types.SimpleNamespace(
        out_docs=1, out_bytes=100, tokens_one_pass=100, passes=1.0,
        to_dict=lambda c: {"script_sha256": "1" * 64, "tokens_sha256": "2" * 64, "out_docs": 1},
    )
    monkeypatch.setattr("stet.tok.load_encoding", lambda s: object())
    monkeypatch.setattr(grade, "materialize", lambda c, p, s, sc, e: ("toks", fake_stats))
    monkeypatch.setattr(grade, "hardware_info", lambda d: {"device": "cpu"})

    # 1. Transient divergence on attempt 1 recovers on attempt 2!
    calls = {"n": 0}

    def flaky_train(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FloatingPointError("SFT probe diverged at step 2")
        return object(), {"seconds": 1.0, "init_sha256": "3" * 64}

    monkeypatch.setattr(grade, "train", flaky_train)
    logs = []
    dev = types.SimpleNamespace(type="cpu")
    trained = grade.train_script(rnd, scale, corpus, script_path, device=dev, log=logs.append)
    assert trained.trained is True and calls["n"] == 2
    assert any("retrying once" in m for m in logs)

    # 2. Persistent divergence across both attempts raises TrainingError (never valid=False).
    calls["n"] = 0

    def always_diverges(*a, **kw):
        calls["n"] += 1
        raise FloatingPointError("SFT probe diverged at step 2")

    monkeypatch.setattr(grade, "train", always_diverges)
    with pytest.raises(grade.TrainingError, match="training failed after retry"):
        grade.train_script(rnd, scale, corpus, script_path, device=dev, log=lambda m: None)
    assert calls["n"] == 2

    # 3. Genuine ScriptError is decided BEFORE training, never retries, returns valid=False.
    calls["n"] = 0
    monkeypatch.setattr(grade, "materialize", lambda *a, **kw: (_ for _ in ()).throw(ScriptError("line 1: bad")))
    tr_bad = grade.train_script(rnd, scale, corpus, script_path, device=dev)
    assert tr_bad.trained is False and tr_bad.result["valid"] is False and "line 1: bad" in tr_bad.result["invalid_reason"]
    assert calls["n"] == 0


def test_review8_grade_hardware_backend_check():
    """Review 8 Finding 4: grade_backends distinguishes NVIDIA CUDA from AMD ROCm when device=='cuda'."""
    from stet.client import grade_hardware_ok, grade_hardware_requirement

    s_cuda_only = types.SimpleNamespace(raw={"grade_devices": ["cuda"], "grade_backends": ["cuda"]})
    assert grade_hardware_ok(s_cuda_only, "cuda", "cuda") is True
    assert grade_hardware_ok(s_cuda_only, "cuda", "rocm") is False
    assert grade_hardware_ok(s_cuda_only, "cuda", None) is False
    assert grade_hardware_ok(s_cuda_only, "cpu", None) is False
    assert grade_hardware_requirement(s_cuda_only) == "an NVIDIA (CUDA) GPU"

    s_both = types.SimpleNamespace(raw={"grade_devices": ["cuda"], "grade_backends": ["cuda", "rocm"]})
    assert grade_hardware_ok(s_both, "cuda", "cuda") is True
    assert grade_hardware_ok(s_both, "cuda", "rocm") is True
    assert grade_hardware_requirement(s_both) == "an NVIDIA (CUDA) or AMD (ROCm) GPU"

    s_legacy = types.SimpleNamespace(raw={"grade_devices": ["cuda", "cpu"]})
    assert grade_hardware_ok(s_legacy, "cuda", "rocm") is True
    assert grade_hardware_ok(s_legacy, "cpu", None) is True


def test_review8_reasoning_drops_saved_partial_keep_rate_and_cache_reuse(monkeypatch, tmp_path):
    """Review 8 Smaller items 2, 3, 4:
    - Dropped reasoning drafts are saved as JSONL beside the kept cache.
    - When one reasoning kind falls below min_keep_rate while others pass, only the bad kind is
      omitted instead of throwing away the whole build and spinning for 6 hours.
    - When ALL kinds fail min_keep_rate, FatalExamError is raised (not retried endlessly).
    - _reasoning_cache_usable reuses a valid cache where >=1 reasoning pillar meets min_pillar_items."""
    from stet.exam import examine
    from stet.exam.fresh import Doc, reason

    docs = [
        Doc(
            source="ccnews", id=f"d-{i}", url=f"https://example.org/{i}", lang="en", license="ccnews",
            published=1_800_000_000 + i, title=f"Story {i}",
            text=f"Detailed article {i} examining agricultural irrigation sensors, reservoir telemetry, and seasonal rainfall patterns across valleys. " * 4,
            topic="",
        )
        for i in range(60)
    ]
    monkeypatch.setattr(reason, "_load_teacher_tokenizer", lambda *a, **k: None)

    # 1. Partial failure: explain fails completely, infer & synthesis pass cleanly.
    def fake_gen_partial(prompts, *a, **k):
        out = []
        for p in prompts:
            if "why/how explanation question" in p:
                out.append("...\nRESPONSE: bad")
            else:
                out.append(
                    "Telemetry networks improve regional irrigation planning across seasonal droughts.\n"
                    "RESPONSE: Coordinated reservoir gauges and soil probes let district engineers adjust canal releases before crops suffer water stress."
                )
        return out

    monkeypatch.setattr(reason, "_generate_local_gpu", fake_gen_partial)
    drops_file = tmp_path / "reasoning-drops.jsonl"
    cfg = {"model": "Qwen/Qwen3-8B", "revision": "rev", "items": 60, "min_keep_rate": 0.5}
    kept = reason.synthesize_reasoning(docs, b"partial-seed", cfg, log=lambda _: None, drops_path=drops_file)
    kinds_kept = {it.source.split(":", 1)[0] for it in kept}
    assert "explain" not in kinds_kept and kinds_kept == {"infer", "synthesis"}
    assert drops_file.exists()
    drop_rows = [json.loads(ln) for ln in drops_file.read_text().splitlines() if ln]
    assert len(drop_rows) >= 15 and all(r["task_type"] == "explain" and r["reason"] == "prompt_len" for r in drop_rows)

    # _reasoning_cache_usable MUST accept this cache (2 of 3 pillars >= 20, 1 omitted)!
    s3 = load_season("s3")
    assert examine._reasoning_cache_usable(kept, s3) is True

    # 2. All kinds fail -> FatalExamError (non-retryable) AND drops file still written!
    monkeypatch.setattr(reason, "_generate_local_gpu", lambda prompts, *a, **k: ["...\nRESPONSE: bad" for _ in prompts])
    drops_all = tmp_path / "reasoning-drops-all.jsonl"
    with pytest.raises(reason.FatalExamError, match="every kind"):
        reason.synthesize_reasoning(docs, b"all-bad-seed", cfg, log=lambda _: None, drops_path=drops_all)
    assert len([ln for ln in drops_all.read_text().splitlines() if ln]) == 60


def test_review8_official_replay_public_subset_disclosure(monkeypatch, tmp_path, capsys):
    """Review 8 Smaller item 6: replaying a settled official round from exam-public-*.jsonl.zst
    discloses how many items were sat vs withheld so users understand why the gain differs from the board."""
    import zstandard

    from stet import cli, data
    from stet.exam import examine
    from stet.exam.items import Item, digest
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    rnd = official_round(
        s3, 1,
        {"round": 123456, "randomness": "ab" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 100000},
    )
    wiki = Item(kind="answer", corner="explain/en/wikipedia", fmt="prose", lang="en", source="explain:wikipedia:w1", ctx="Ex1\nQ1?\n", cont="Answer one.")
    news = Item(kind="answer", corner="explain/en/ccnews", fmt="prose", lang="en", source="explain:ccnews:n1", ctx="Ex2\nQ2?\n", cont="Answer two.")
    all_items = [wiki, news]
    dig = digest(all_items)
    rnd = dataclasses.replace(rnd, exam_digest=dig)

    pub_raw = "".join(json.dumps(r) + "\n" for r in examine.public_items(all_items)).encode()
    pub_path = data.exam_public_path(rnd, dig)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_bytes(zstandard.ZstdCompressor().compress(pub_raw))

    meta = data.round_exam_meta(rnd)
    assert meta == {"public_subset": True, "public_items": 1, "total_items": 2, "withheld_items": 1}

    out = cli.Out(quiet=False)
    res = {
        "valid": True,
        "round": {"kind": "official"},
        "scale": "standard",
        "script": {"out_docs": 10, "src_docs": 20, "kept_doc_frac": 0.5, "out_bytes": 1_000_000, "passes": 1.0, "truncated": False},
        "train": {"tokens": 1000, "seconds": 10, "tokens_per_second": 100, "final_train_loss": 2.5},
        "exam": {"items": 1, "mean": -2.0},
        "exam_subset": meta,
        "gain": 0.05,
        "pillars": {"explain": 0.05},
    }
    base = {"exam": {"mean": -2.05}}
    cli.report(out, res, base, s3)
    err = capsys.readouterr().err
    assert "1 of 2 items (public subset; 1 third-party items withheld)" in err
    assert "official board score uses all 2 items" in err


def test_review8_reasoning_only_score_diagnostic_comprehension_and_source_cap(monkeypatch, tmp_path):
    """Review 8 Finding 1 + Smaller item 1 + practice-with-harvest:
    - Season 3 scores 100% fresh reasoning (explain/infer/synthesis, 1/3 each), with comprehension at 0% diagnostic weight.
    - `select_context` caps any single source inside a kind when >= 2 sources exist, and `code_share=0.15` caps code in technical.
    - `_practice()` with harvested hours and synthesize=False appends bundled practice_pack reasoning items so local practice never scores 0.0.
    """
    from collections import Counter

    from stet import data
    from stet.exam import aggregate
    from stet.exam.fresh import Doc
    from stet.exam.fresh import select as fresh_select
    from stet.spec import practice_round

    s3 = load_season("s3")
    plan = aggregate.plan_of(s3)
    assert plan.groups == {"comprehension": 0.0, "reasoning": 1.0}
    assert plan.code_share == pytest.approx(0.15)
    assert s3.raw["grade_backends"] == ["cuda"]
    assert s3.raw["tolerance"]["standard"] == {"grade": 0.012, "baseline": 0.015, "run": 0.015}

    # Single-source cap inside select_context: 100 bluesky docs + 20 mastodon docs in "social"
    # With 2 sources and n=40, bluesky is capped at 20 (50%) while mastodon fills the other 20.
    docs = []
    for i in range(60):
        docs.append(
            Doc(
                source="bluesky", id=f"b{i}", url=f"https://bsky.app/{i}", lang="en", license="bluesky",
                published=1_800_000_000 + i, title=f"Blue {i}",
                text=f"First complete analytical sentence about observation {i} with distinct details and enough characters to split cleanly. "
                     f"Second distinct continuation sentence {i} providing a clear follow-up statement with plenty of context.",
                topic=f"tb{i}",
            )
        )
    for i in range(30):
        docs.append(
            Doc(
                source="mastodon", id=f"m{i}", url=f"https://masto.social/{i}", lang="en", license="mastodon",
                published=1_800_000_000 + i, title=f"Masto {i}",
                text=f"Opening remark on federated publishing system topic {i} written cleanly with multiple clauses for context. "
                     f"Continuation remark {i} describing the community response and follow-up outcomes in detail.",
                topic=f"tm{i}",
            )
        )
    chosen = fresh_select.select_context(docs, b"cap-seed", 40, {"social"}, per_corner=2, shares={"en": 100.0})
    assert len(chosen) == 40
    by_src = Counter(it.source.split(":", 1)[0] for it in chosen)
    assert by_src["bluesky"] == 20 and by_src["mastodon"] == 20

    # Local practice with harvested hours (synthesize=False) still includes bundled reasoning items
    # so `stet run` after `stet harvest` never scores 0.0!
    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    monkeypatch.delenv("STET_SYNTHESIZE_PRACTICE", raising=False)
    rnd = practice_round(s3, 0)
    scale = s3.scale("nano")
    # Mock assemble() to return comprehension-only items (as synthesize=False does)
    fake_comp = [
        Item(kind="context", corner="fresh/wikipedia/en/a", fmt="prose", lang="en", source="wikipedia:1",
             ctx="Alpha opening.", cont=" Alpha continuation.", wrongs=("Beta opening.",))
    ]
    from stet.exam import examine

    monkeypatch.setattr(examine, "assemble", lambda *a, **k: (fake_comp, 1, {"en": 100.0}, ["abc"]))
    hpath = tmp_path / "1800000000.jsonl"
    hpath.write_text("{}\n")
    items = data._practice(rnd, scale, [(1800000000, hpath)])
    parts = {aggregate.part_of(it) for it in items}
    assert "reference" in parts
    assert {"explain", "infer", "synthesis"} <= parts


def test_review9_explicit_season_round_resolution(monkeypatch, capsys):
    """Review 9 P2 #1: full round IDs (s2-p00000, s2-o00003) honor their encoded season,
    reject conflicting --season flags, and report clearly when historical official artifacts are unavailable."""
    from types import SimpleNamespace

    from stet import cli, client

    # Practice ID with explicit s2 season resolves to s2-p00000, not s3-p00000.
    args_p = SimpleNamespace(round="s2-p00000", season=None, scale=None)
    season, scale, rnd = cli.resolve_round(args_p)
    assert season.id == "s2" and rnd.id == "s2-p00000" and scale.name == "nano"

    # Plain numeric round still defaults to CURRENT_SEASON (s3).
    args_def = SimpleNamespace(round="0", season=None, scale=None)
    season_d, _, rnd_d = cli.resolve_round(args_def)
    assert season_d.id == "s3" and rnd_d.id == "s3-p00000"

    # Conflicting explicit --season is rejected.
    args_bad = SimpleNamespace(round="s2-p00000", season="s3", scale=None)
    with pytest.raises(SystemExit):
        cli.resolve_round(args_bad)
    assert "conflicts with --season" in capsys.readouterr().err

    # Official ID s2-o00003 queries s2-o00003 (not s3-o00003) and reports historical unavailability on 404.
    seen_paths = []

    def fake_request(method, path, *a, **k):
        seen_paths.append(path)
        raise client.ApiError(404, "no such round")

    monkeypatch.setattr(client, "request", fake_request)
    args_o = SimpleNamespace(round="s2-o00003", season=None, scale=None)
    with pytest.raises(SystemExit):
        cli.resolve_round(args_o)
    assert seen_paths == ["/api/v1/rounds/s2-o00003"]
    err = capsys.readouterr().err
    assert "s2-o00003" in err and "historical season s2" in err


def test_review9_public_and_full_exam_baseline_cache_isolation(monkeypatch, tmp_path):
    """Review 9 P2 #2: public-subset replay and full official exam never share a baseline cache entry
    in either transition (public -> full or full -> public)."""
    pytest.importorskip("torch")
    import json
    from types import SimpleNamespace

    import zstandard as zstd

    from stet import cli, data, pipeline
    from stet.exam.examine import public_items
    from stet.exam.items import digest as items_digest
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    p = s3.raw["pools"]["practice"]
    crawl = {"crawl": p["crawl"], "paths_sha256": p["paths_sha256"], "n_shards": int(p["n_shards"])}
    full_items = [
        Item(kind="context", corner="fresh/wikipedia/en/a", fmt="prose", lang="en", source="wikipedia:1",
             ctx="Alpha open.", cont=" Alpha cont.", wrongs=("Beta open.",)),
        Item(kind="context", corner="fresh/ccnews/en/b", fmt="prose", lang="en", source="ccnews:2",
             ctx="News open.", cont=" News cont.", wrongs=("Other news open.",)),
    ]
    full_sha = items_digest(full_items)
    rnd = official_round(s3, 7, {"round": 900007, "randomness": "77" * 32}, crawl)
    rnd = type(rnd)(**{**rnd.__dict__, "exam_digest": full_sha})

    pub_raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in public_items(full_items)).encode()
    full_raw = "\n".join(json.dumps(it.to_dict(), ensure_ascii=False) for it in full_items).encode()
    comp = zstd.ZstdCompressor()
    pub_path = data.exam_public_path(rnd, full_sha)
    full_path = data.exam_items_path(rnd, full_sha)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_bytes(comp.compress(pub_raw))

    dev = SimpleNamespace(type="cpu")
    sit_calls = []

    def fake_train(r, sc, d, **kw):
        return SimpleNamespace(device=dev)

    def fake_sit(trained, r, sc, exam):
        sha = items_digest(exam)
        sit_calls.append((len(exam), sha))
        return {
            "valid": True,
            "exam_sha256": sha,
            "exam": {"items": len(exam), "mean": -2.0 - 0.1 * len(exam), "item_scores": [-2.0] * len(exam)},
        }

    monkeypatch.setattr(pipeline, "train_baseline", fake_train)
    monkeypatch.setattr(pipeline, "sit_exam", fake_sit)

    # 1. Public subset only -> sits 1 item and caches under `-public.json`.
    b_pub = pipeline.baseline(rnd, scale, dev)
    assert b_pub["exam"]["items"] == 1 and b_pub["exam_sha256"] != full_sha
    assert len(sit_calls) == 1

    # 2. Full exam arrives -> MUST miss public cache and sit all 2 items!
    full_path.write_bytes(comp.compress(full_raw))
    b_full = pipeline.baseline(rnd, scale, dev)
    assert b_full["exam"]["items"] == 2 and b_full["exam_sha256"] == full_sha
    assert len(sit_calls) == 2

    # 3. Reverse transition (full exam removed, public subset remains) -> returns 1-item public baseline,
    #    NOT the 2-item full baseline!
    full_path.unlink()
    b_pub2 = pipeline.baseline(rnd, scale, dev)
    assert b_pub2["exam"]["items"] == 1 and b_pub2["exam_sha256"] != full_sha
    assert len(sit_calls) == 2  # cache hit on the separate public baseline file!

    # 4. Even if someone manually puts a wrong-digest file at a baseline path, _cached rejects it.
    wrong_path = pipeline.baseline_path(rnd, scale, dev, exam_sha=full_sha)
    assert pipeline._cached(wrong_path, want_exam_sha="0" * 64) is None
    assert pipeline._cached(wrong_path, want_exam_sha=full_sha, want_items=99) is None
    assert pipeline._cached(wrong_path, want_exam_sha=full_sha, want_items=2) is not None
    assert cli._has_writing(rnd)


def test_review9_pooled_languages_apply_tail_trim():
    """Review 9 P2 #3: pooled small languages apply the configured 10% tail trim consistently
    with single languages (equal-weight 19 zeros + one +0.1 -> 0.0) without letting tiny
    micro-languages crowd out larger pooled languages."""
    from stet.exam.aggregate import plan_of, round_score

    s3 = load_season("s3")
    plan = plan_of(s3)
    single = [Item(kind="answer", corner=f"fresh/explain/en/{i}", fmt="prose", lang="en", source=f"explain:{i}", ctx="Q", cont="A") for i in range(20)]
    pooled = [
        Item(kind="answer", corner=f"fresh/explain/{lang}/{i}", fmt="prose", lang=lang, source=f"explain:{i}", ctx="Q", cont="A")
        for i, lang in enumerate(["ca"] * 5 + ["sw"] * 5 + ["yo"] * 5 + ["eo"] * 5)
    ]
    gains = [0.0] * 19 + [0.1]
    s_single = round_score(single, gains, shares={"en": 100.0}, plan=plan)["score"]
    s_pooled = round_score(pooled, gains, shares={"ca": 1.0, "sw": 1.0, "yo": 1.0, "eo": 1.0}, plan=plan)["score"]
    assert s_single == pytest.approx(0.0, abs=1e-12)
    assert s_pooled == pytest.approx(0.0, abs=1e-12)

    # Skewed weights: 18 tiny micro-language items (-0.4) must NOT crowd out 2 heavy ja items (+0.2).
    skewed = (
        [Item(kind="answer", corner=f"fresh/explain/ky/{i}", fmt="prose", lang="ky", source=f"explain:ky:{i}", ctx="Q", cont="A") for i in range(6)]
        + [Item(kind="answer", corner=f"fresh/explain/yo/{i}", fmt="prose", lang="yo", source=f"explain:yo:{i}", ctx="Q", cont="A") for i in range(6)]
        + [Item(kind="answer", corner=f"fresh/explain/eo/{i}", fmt="prose", lang="eo", source=f"explain:eo:{i}", ctx="Q", cont="A") for i in range(6)]
        + [Item(kind="answer", corner=f"fresh/explain/ja/{i}", fmt="prose", lang="ja", source=f"explain:ja:{i}", ctx="Q", cont="A") for i in range(2)]
    )
    skewed_gains = [-0.4] * 18 + [0.2] * 2
    s_skew = round_score(skewed, skewed_gains, shares={"ky": 0.01, "yo": 0.01, "eo": 0.01, "ja": 5.53}, plan=plan)["score"]
    assert s_skew > 0.15, f"heavy pooled language ja should survive tail trim, got {s_skew}"


def test_review9_examine_follow_retries_transient_cancel_without_rebuilding(monkeypatch):
    """Review 9 P2 #4: a transient cancel POST failure after FatalExamError retains the reason
    and retries cancellation on subsequent polls without re-running the expensive exam build."""
    from types import SimpleNamespace

    from stet import cli, client
    from stet.exam.examine import FatalExamError

    build_calls = []
    cancel_calls = []
    polls = [0]

    class _StopLoop(BaseException):
        pass

    def fake_request(method, path, body=None, **kw):
        if method == "GET" and path.startswith("/api/v1/rounds"):
            polls[0] += 1
            if polls[0] > 3:
                raise _StopLoop()
            return {"rounds": [{"id": "s3-o00001", "status": "open", "exam_digest": ""}]}
        if method == "POST" and path == "/api/v1/admin/rounds/s3-o00001/cancel":
            cancel_calls.append(body)
            if len(cancel_calls) == 1:
                raise OSError("temporary network outage")
            return {"cancelled": True}
        raise AssertionError(f"unexpected request {method} {path}")

    def fake_examine_one(args, rid):
        build_calls.append(rid)
        raise FatalExamError("all reasoning kinds below min_keep_rate")

    monkeypatch.setattr(client, "need_key", lambda: SimpleNamespace())
    monkeypatch.setattr(client, "request", fake_request)
    monkeypatch.setattr(cli, "_examine_one", fake_examine_one)
    monkeypatch.setattr(cli, "_harvest_follow", lambda *a, **k: None)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    args = SimpleNamespace(season="s3", quiet=True)
    with pytest.raises(_StopLoop):
        cli._examine_follow(args)

    assert build_calls == ["s3-o00001"], f"exam build must run only once, got {build_calls}"
    assert len(cancel_calls) == 2, f"cancel must be retried after transient failure, got {cancel_calls}"
    assert "min_keep_rate" in cancel_calls[1]["reason"]


def test_review9_remote_examiner_requires_verified_revision(monkeypatch):
    """Review 9 P2 #5: remote `/v1/models` check rejects bare `Qwen/Qwen3-8B` without a
    revision-qualified model ID or verified revision metadata."""
    import io
    import json
    import urllib.request

    from stet.exam.fresh import reason

    repo = "Qwen/Qwen3-8B"
    rev = "b968826d9c46dd6066d109eabc6255188de91218"

    def mock_models(data_list):
        body = json.dumps({"data": data_list}).encode()

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: _Resp(body))

    monkeypatch.delenv("STET_EXAMINER_LLM_MODEL", raising=False)
    monkeypatch.delenv("STET_EXAMINER_LLM_REVISION", raising=False)

    # 1. Bare model ID without revision metadata MUST be rejected.
    mock_models([{"id": repo}])
    with pytest.raises(RuntimeError, match="without verified revision"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})

    # 2. Even with STET_EXAMINER_LLM_MODEL=Qwen/Qwen3-8B, bare ID without metadata MUST be rejected.
    monkeypatch.setenv("STET_EXAMINER_LLM_MODEL", repo)
    with pytest.raises(RuntimeError, match="without verified revision"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})
    monkeypatch.delenv("STET_EXAMINER_LLM_MODEL", raising=False)

    # 3. Bare model ID with WRONG revision metadata MUST be rejected.
    mock_models([{"id": repo, "revision": "0000000000000000"}])
    with pytest.raises(RuntimeError, match="reports revision"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})

    # 4. Revision-qualified deployment ID (`--served-model-name repo@rev`) is ACCEPTED.
    mock_models([{"id": f"{repo}@{rev}"}])
    assert reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {}) == f"{repo}@{rev}"

    # 5. Bare model ID with explicit matching revision metadata is ACCEPTED.
    mock_models([{"id": repo, "revision": rev}])
    assert reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {}) == repo

    # 6. Bare model ID with pinned HF snapshot path in `root` is ACCEPTED.
    mock_models([{"id": repo, "root": f"/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/{rev}"}])
    assert reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {}) == repo

    # 7. Gate 4 hardening: conflicting explicit revision cannot be bypassed by matching `root`.
    mock_models([{"id": repo, "revision": "deadbeef", "root": f"/cache/snapshots/{rev}"}])
    with pytest.raises(RuntimeError, match="reports revision"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})

    # 8. Prefix-colliding `snapshots/<rev>extra` in `root` MUST be rejected.
    mock_models([{"id": repo, "root": f"/cache/snapshots/{rev}extra"}])
    with pytest.raises(RuntimeError, match="without verified revision"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})

    # 9. Revision-qualified ID with conflicting revision metadata MUST report revision mismatch.
    mock_models([{"id": f"{repo}@{rev}", "revision": "deadbeef"}])
    with pytest.raises(RuntimeError, match="reports revision"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})

    # 10. STET_EXAMINER_LLM_MODEL=repo@rev when endpoint only serves verified bare repo reports missing ID.
    monkeypatch.setenv("STET_EXAMINER_LLM_MODEL", f"{repo}@{rev}")
    mock_models([{"id": repo, "revision": rev}])
    with pytest.raises(RuntimeError, match="does not include pinned model"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})
    monkeypatch.delenv("STET_EXAMINER_LLM_MODEL", raising=False)

    # 11. `"data": null` raises RuntimeError cleanly rather than TypeError.
    mock_models(None)
    with pytest.raises(RuntimeError, match=r"serves \[\]"):
        reason._verify_endpoint_model("http://127.0.0.1:8000/v1", repo, rev, {})


def test_inkling2_chat_teacher_config_and_endpoint(monkeypatch):
    """Season 3 pins thinkingmachines/Inkling-2-Mini (api='chat', reasoning_effort='low',
    max_new_tokens=1536, temperature=0.0), normalizes URLs & chat outputs without leaking CoT or
    control tokens, fails fast on 4xx & aborts workers on error, and requires STET_EXAMINER_LLM_URL."""
    import io
    import re
    import threading
    import urllib.error
    import urllib.request

    from stet.exam.fresh import reason

    s3 = load_season("s3")
    rcfg = s3.raw["exam"]["reasoning"]
    assert rcfg["model"] == "thinkingmachines/Inkling-2-Mini"
    assert rcfg["revision"] == "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
    assert rcfg["api"] == "chat"
    assert rcfg["reasoning_effort"] == "low"
    assert rcfg["max_new_tokens"] == 1536
    assert rcfg["temperature"] == 0.0

    # THIRD_PARTY_NOTICES.md credits Inkling-2-Mini under Apache License 2.0.
    notices = (Path(__file__).resolve().parents[1] / "src/stet/exam/data/THIRD_PARTY_NOTICES.md").read_text()
    assert "thinkingmachines/Inkling-2-Mini" in notices and "Apache License 2.0" in notices

    # 1. URL normalization handles whitespace, bare host, /v1, /v1/completions, and /v1/chat/completions.
    for raw_url in (
        "  http://127.0.0.1:8000  ",
        "http://127.0.0.1:8000/",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8000/v1/completions",
        "http://127.0.0.1:8000/v1/chat/completions",
    ):
        assert reason._base_v1_url(raw_url) == "http://127.0.0.1:8000/v1"

    # 2. Chat output normalization: strips valid wrappers & trailing Inkling blocks, and REJECTS
    #    any unclosed CoT, remaining control tokens, or raw preamble before PROMPT:.
    assert reason._normalize_chat_output("  PROMPT: Why?\nRESPONSE: Because.") == "Why?\nRESPONSE: Because."
    assert reason._normalize_chat_output("<think>secret</think>\nPROMPT: Why?\nRESPONSE: Because.") == "Why?\nRESPONSE: Because."
    assert (
        reason._normalize_chat_output(
            "<|content_thinking|>draft<|content_text|>PROMPT: Why?\nRESPONSE: Because.<|content_thinking|>more"
        )
        == "Why?\nRESPONSE: Because."
    )
    assert (
        reason._normalize_chat_output(
            "<|content_text|>PROMPT: Why?\nRESPONSE: Because.<|content_invoke_tool_json|>{}"
        )
        == "Why?\nRESPONSE: Because."
    )
    # `<|content_text|>` takes precedence over `</think>`, and any remaining `<think>`/`</think>` is rejected cleanly.
    assert reason._normalize_chat_output("<think>unclosed PROMPT: Why?\nRESPONSE: Because.") == ""
    assert reason._normalize_chat_output("<|content_thinking|>unclosed PROMPT: Why?\nRESPONSE: Because.") == ""
    assert reason._normalize_chat_output("Let me think carefully first. PROMPT: Why?\nRESPONSE: Because.") == ""
    assert reason._normalize_chat_output("<|content_text|>PROMPT: Why?\nRESPONSE: See </think> tag.") == ""
    assert reason._normalize_chat_output("") == ""

    # Parser-level defense-in-depth rejects control tokens and CoT preamble in `pr`/`resp`.
    sample_doc = [{
        "source": "ccnews", "id": "guard-1", "lang": "en", "task_type": "explain", "wrapper_id": 0,
        "text": "Municipal battery storage networks stabilize winter peak grid demand across suburban substations.",
        "excerpt": "Municipal battery storage networks stabilize winter peak grid demand across suburban substations.",
    }]
    kept_g, drops_g, _ = reason.parse_generated_items_detailed(
        sample_doc,
        ["Here is a question: PROMPT: Why?\nRESPONSE: Batteries absorb sudden spikes cleanly across regional grids every winter season."],
    )
    assert kept_g == [] and drops_g == {"meta_chatter": 1}
    kept_c, drops_c, _ = reason.parse_generated_items_detailed(
        sample_doc,
        ["Why do batteries help?\nRESPONSE: They absorb spikes cleanly <|mystery_tok|> across regional grids every winter season."],
    )
    assert kept_c == [] and drops_c == {"control_token": 1}

    # 3. can_synthesize_locally(season) honors passed season and requires STET_EXAMINER_LLM_URL for chat API.
    monkeypatch.delenv("STET_EXAMINER_LLM_URL", raising=False)
    assert reason.can_synthesize_locally(s3) is False
    assert reason.can_synthesize_locally() is False
    monkeypatch.setenv("STET_EXAMINER_LLM_URL", "http://127.0.0.1:8000/v1")
    assert reason.can_synthesize_locally(s3) is True

    # 4. synthesize_reasoning(api='chat') checks STET_EXAMINER_LLM_URL BEFORE loading tokenizer!
    monkeypatch.delenv("STET_EXAMINER_LLM_URL", raising=False)
    monkeypatch.setattr(
        reason,
        "_load_teacher_tokenizer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tokenizer must not be loaded before URL check")),
    )
    docs = [
        Doc(
            source="ccnews",
            id=f"ink-{i}",
            url=f"https://example.org/ink/{i}",
            lang="en",
            license="ccnews",
            published=1_800_000_000 + i,
            title=f"Story {i}",
            text=f"Detailed report {i} analyzing municipal battery storage grids, peak load shaving, and winter tariff resilience across districts. " * 4,
            topic="",
        )
        for i in range(6)
    ]
    with pytest.raises(RuntimeError, match="STET_EXAMINER_LLM_URL") as exc_info:
        reason.synthesize_reasoning(docs, b"ink-seed", dict(rcfg, items=6), log=lambda _: None)
    assert not isinstance(exc_info.value, reason.FatalExamError)

    # 5. Unsupported api value fails loudly with ValueError.
    with pytest.raises(ValueError, match="unsupported reasoning api"):
        reason.synthesize_reasoning(docs, b"ink-seed", {**rcfg, "api": "bogus"}, log=lambda _: None)

    # 6. End-to-end mocked /v1/chat/completions: verifies pinned model, sends reasoning_effort="low",
    #    ignores reasoning_content (never leaks CoT), preserves order, and retries transient error once.
    monkeypatch.setattr(reason, "_load_teacher_tokenizer", lambda *a, **k: None)
    repo = rcfg["model"]
    rev = rcfg["revision"]
    qualified = f"{repo}@{rev}"
    lock = threading.Lock()
    seen_payloads: list[dict] = []
    chat_calls = {"n": 0}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=30):
        url = req.full_url
        if url.endswith("/v1/models"):
            return _Resp(json.dumps({"data": [{"id": qualified}]}).encode())
        assert url == "http://127.0.0.1:8000/v1/chat/completions"
        body = json.loads(req.data.decode("utf-8"))
        with lock:
            seen_payloads.append(body)
            chat_calls["n"] += 1
            call_no = chat_calls["n"]
        if call_no == 1:
            raise OSError("transient blip")
        content_in = body["messages"][0]["content"]
        m = re.search(r"Detailed report (\d+)", content_in)
        tag = m.group(1) if m else "0"
        is_syn = "declarative topic sentence" in content_in
        prompt_line = (
            f"Municipal battery storage networks stabilize winter grid demand in report {tag}."
            if is_syn
            else f"How do municipal battery grids stabilize winter peak demand in report {tag}?"
        )
        reply = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "INTERNAL_COT_SECRET_DO_NOT_LEAK",
                        "content": f"PROMPT: {prompt_line}\nRESPONSE: Storing off-peak energy lets utilities absorb sudden winter spikes without firing costly backup turbines in case {tag}. Lower reserve margins reduce tariff volatility for households.",
                    },
                }
            ]
        }
        return _Resp(json.dumps(reply).encode())

    monkeypatch.setenv("STET_EXAMINER_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(reason.time, "sleep", lambda _: None)

    kept = reason.synthesize_reasoning(docs, b"ink-seed-ok", {**rcfg, "items": 6}, log=lambda _: None)
    assert len(kept) == 6
    assert len(seen_payloads) == 7  # 6 drafts + 1 transient retry
    for p in seen_payloads:
        assert p["model"] == qualified
        assert p["reasoning_effort"] == "low"
        assert p["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert p["temperature"] == 0.0
        assert p["max_tokens"] == 1536
        assert p["seed"] == 0
    for it in kept:
        assert "INTERNAL_COT_SECRET_DO_NOT_LEAK" not in it.ctx
        assert "INTERNAL_COT_SECRET_DO_NOT_LEAK" not in it.cont
        doc_num = it.source.rsplit("-", 1)[-1]
        assert f"case {doc_num}." in it.cont

    # 7. reasoning_effort=None omits effort keys, and `content: null` returns "" (cleanly dropped, no crash).
    null_payloads: list[dict] = []

    def null_content_urlopen(req, timeout=30):
        if req.full_url.endswith("/v1/models"):
            return _Resp(json.dumps({"data": [{"id": qualified}]}).encode())
        null_payloads.append(json.loads(req.data.decode("utf-8")))
        return _Resp(json.dumps({"choices": [{"message": {"role": "assistant", "content": None, "reasoning_content": "thought"}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", null_content_urlopen)
    outs_null = reason._generate_via_chat_endpoint(
        ["hello"], "http://127.0.0.1:8000/v1", repo, rev, reasoning_effort=None, log=lambda _: None
    )
    assert outs_null == [""]
    assert "reasoning_effort" not in null_payloads[0] and "chat_template_kwargs" not in null_payloads[0]

    # 8. HTTP 400 fails fast with ZERO retries and surfaces the server error body.
    http400_calls = {"n": 0}

    def http400_urlopen(req, timeout=30):
        if req.full_url.endswith("/v1/models"):
            return _Resp(json.dumps({"data": [{"id": qualified}]}).encode())
        with lock:
            http400_calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error": "bad reasoning_effort"}'))

    monkeypatch.setattr(urllib.request, "urlopen", http400_urlopen)
    with pytest.raises(RuntimeError, match="bad reasoning_effort") as exc_400:
        reason._generate_via_chat_endpoint(
            ["draft1", "draft2", "draft3"], "http://127.0.0.1:8000/v1", repo, rev, max_workers=1, log=lambda _: None
        )
    assert not isinstance(exc_400.value, reason.FatalExamError)
    assert http400_calls["n"] == 1  # zero retries AND remaining drafts aborted immediately!

    # 8b. Multi-worker concurrency (max_workers=2): draft 1's real HTTP 400 surfaces even if draft 0
    #     is sleeping in retry backoff (never masked by the synthetic abort message).
    d0_sleeping = threading.Event()

    def race_urlopen(req, timeout=30):
        if req.full_url.endswith("/v1/models"):
            return _Resp(json.dumps({"data": [{"id": qualified}]}).encode())
        msg_in = json.loads(req.data.decode("utf-8"))["messages"][0]["content"]
        if msg_in == "slow-draft-0":
            d0_sleeping.set()
            raise OSError("transient draft-0 blip")
        d0_sleeping.wait(timeout=2.0)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"root-cause-400"))

    monkeypatch.setattr(urllib.request, "urlopen", race_urlopen)
    with pytest.raises(RuntimeError, match="root-cause-400") as exc_race:
        reason._generate_via_chat_endpoint(
            ["slow-draft-0", "fail-draft-1", "queued-draft-2"],
            "http://127.0.0.1:8000/v1",
            repo,
            rev,
            max_workers=2,
            log=lambda _: None,
        )
    assert not isinstance(exc_race.value, reason.FatalExamError)

    # 9. Malformed /v1/chat/completions responses fail fast with retryable RuntimeError.
    for bad_body, match in (
        ({"choices": []}, "missing choices"),
        ({"choices": [{}]}, "missing message"),
        ({"choices": [{"message": {"content": 123}}]}, "content type"),
    ):
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=30, b=bad_body: _Resp(
                json.dumps({"data": [{"id": qualified}]} if req.full_url.endswith("/v1/models") else b).encode()
            ),
        )
        with pytest.raises(RuntimeError, match=match) as exc_bad:
            reason._generate_via_chat_endpoint(["draft"], "http://127.0.0.1:8000/v1", repo, rev, log=lambda _: None)
        assert not isinstance(exc_bad.value, reason.FatalExamError)

    # 10. Persistent transient network failure raises retryable RuntimeError (NEVER FatalExamError).
    def failing_urlopen(req, timeout=30):
        if req.full_url.endswith("/v1/models"):
            return _Resp(json.dumps({"data": [{"id": qualified}]}).encode())
        raise OSError("server down")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setattr(threading.Event, "wait", lambda self, timeout=None: self.is_set())
    with pytest.raises(RuntimeError, match="reasoning chat synthesis failed") as exc_fail:
        reason.synthesize_reasoning(docs[:1], b"ink-seed-fail", {**rcfg, "items": 1}, log=lambda _: None)
    assert not isinstance(exc_fail.value, reason.FatalExamError)


def test_control_token_filter_allows_code_pipe_operator_and_rejects_special_tokens():
    """Review 10 P3: _CONTROL_TOKEN_RE must not drop code reasoning answers containing |>
    (Elixir/F#/OCaml/R/Julia pipe operators) while still rejecting <|...|> and <think> tags."""
    from stet.exam.fresh import reason

    # Code pipe operator in answer MUST be accepted by both _CONTROL_TOKEN_RE and _clean_chat_output.
    ok_pipe = (
        "PROMPT: How does the pipeline transform the input list?\n"
        "RESPONSE: The expression `data |> Enum.filter(&(&1 > 0)) |> Enum.map(&(&1 * 2))` "
        "passes the collection through filtering before doubling each positive element."
    )
    cleaned = reason._normalize_chat_output(ok_pipe)
    assert cleaned and "|>" in cleaned and not reason._CONTROL_TOKEN_RE.search(cleaned)

    # Actual control/CoT tokens MUST still be rejected (including hyphenated special tokens).
    for bad in (
        "PROMPT: What happens?\nRESPONSE: Let me think <|custom_ctrl|> about this carefully first.",
        "PROMPT: What happens?\nRESPONSE: Done now.<|special_end|> Extra trailing commentary here.",
        "PROMPT: What happens?\nRESPONSE: Done now.<|end-of-text|> Extra trailing commentary here.",
        "PROMPT: What happens?\nRESPONSE: <think>internal reasoning</think> The final explanation is here.",
    ):
        assert reason._CONTROL_TOKEN_RE.search(bad), f"expected _CONTROL_TOKEN_RE to match: {bad!r}"
    assert reason._normalize_chat_output("PROMPT: Q?\nRESPONSE: text <|custom_ctrl|> more.") == ""
    assert reason._normalize_chat_output("PROMPT: Q?\nRESPONSE: <think>unfinished reasoning") == ""


def test_wait_for_exam_is_bounded_and_releases_pending_jobs_on_timeout(monkeypatch):
    """Review 10 P1 #1: wait_for_exam and early-grade Pending waits must enforce a bounded deadline
    instead of hanging indefinitely on a stalled examiner."""
    from stet import client

    rnd = types.SimpleNamespace(id="s3-o00009", exam_digest="")
    out = types.SimpleNamespace(line=lambda *a, **k: None, dim=lambda s: s, bad=lambda s: s)

    # 1. wait_for_exam raises ExamWaitTimeout (ApiError 409) once elapsed >= timeout.
    clock = {"t": 100.0}
    sleeps: list[float] = []
    monkeypatch.setattr(client, "exam_now", lambda r: None)
    monkeypatch.setattr(client.time, "monotonic", lambda: clock["t"])

    def fake_sleep(sec):
        sleeps.append(sec)
        clock["t"] += sec

    monkeypatch.setattr(client.time, "sleep", fake_sleep)
    with pytest.raises(client.ExamWaitTimeout) as exc:
        client.wait_for_exam(rnd, out, timeout=50.0)
    assert exc.value.status == 409
    assert client.EXAM_WAIT_TIMEOUT_REASON in str(exc.value)
    assert sum(sleeps) == pytest.approx(50.0)

    # 2. _exam_wait_limit reads STET_EXAM_WAIT_SECONDS with safe default EXAM_WAIT_MAX and rejects nan/inf/negative.
    monkeypatch.delenv("STET_EXAM_WAIT_SECONDS", raising=False)
    assert client._exam_wait_limit() == float(client.EXAM_WAIT_MAX)
    monkeypatch.setenv("STET_EXAM_WAIT_SECONDS", "1200")
    assert client._exam_wait_limit() == 1200.0
    for bad_env in ("nan", "inf", "-inf", "-5", "nope"):
        monkeypatch.setenv("STET_EXAM_WAIT_SECONDS", bad_env)
        assert client._exam_wait_limit() == float(client.EXAM_WAIT_MAX)


def test_review12_baseline_cache_isolates_rocm_from_nvidia_cuda(monkeypatch, tmp_path):
    """Review 12 P2 #3: NVIDIA CUDA grades must never reuse a ROCm or untagged CUDA baseline from
    a shared or migrated cache across baseline(), early-grade _sit(), optional-baseline lookup,
    and slim_result()."""
    torch = pytest.importorskip("torch")
    import json

    from stet import client, data, grade, pipeline, train
    from stet.exam.items import Item
    from stet.exam.items import digest as items_digest
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    p = s3.raw["pools"]["practice"]
    crawl = {"crawl": p["crawl"], "paths_sha256": p["paths_sha256"], "n_shards": int(p["n_shards"])}

    exam = [
        Item(kind="answer", corner="fresh/explain/en/1", fmt="prose", lang="en", source="explain", ctx="c1", cont="a" * 32),
        Item(kind="answer", corner="fresh/infer/en/2", fmt="prose", lang="en", source="infer", ctx="c2", cont="b" * 32),
    ]
    full_sha = items_digest(exam)
    rnd = official_round(s3, 7, {"round": 900007, "randomness": "77" * 32}, crawl)
    rnd = type(rnd)(**{**rnd.__dict__, "exam_digest": full_sha})
    monkeypatch.setattr(data, "round_exam", lambda r, sc: exam)

    cuda_dev = types.SimpleNamespace(type="cuda")
    nvidia_hw = {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0}
    rocm_hw = {"device": "cuda", "backend": "rocm", "gpu": "AMD Radeon 8060S", "vram_gb": 96.0}
    untagged_hw = {"device": "cuda", "gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0}

    # 1. Cache filenames separate NVIDIA CUDA (-cuda-) from AMD ROCm (-rocm-).
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert train.device_backend(cuda_dev) == "cuda"
    nv_path = pipeline.baseline_path(rnd, scale, cuda_dev)
    assert "-cuda-" in nv_path.name

    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising=False)
    assert train.device_backend(cuda_dev) == "rocm"
    rocm_path = pipeline.baseline_path(rnd, scale, cuda_dev)
    assert "-rocm-" in rocm_path.name and rocm_path != nv_path

    # Switch back to NVIDIA CUDA host and plant a migrated ROCm-labelled baseline AT the NVIDIA path!
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    rocm_cached = {
        "valid": True,
        "exam_sha256": full_sha,
        "hardware": rocm_hw,
        "train": {"init_sha256": "a" * 64},
        "exam": {"items": 2, "mean": -2.5, "item_scores": [-2.5, -2.5]},
    }
    nv_path.parent.mkdir(parents=True, exist_ok=True)
    nv_path.write_text(json.dumps(rocm_cached))

    # 2. _cached() MUST reject the ROCm-labelled file at the NVIDIA cache path.
    assert pipeline._cached(nv_path) is None
    assert pipeline._cached(nv_path, want_exam_sha=full_sha, want_items=2, want_backend="cuda") is None

    # Untagged legacy CUDA cache without explicit backend is ALSO rejected on NVIDIA CUDA.
    legacy_untagged = dict(rocm_cached, hardware=untagged_hw)
    nv_path.write_text(json.dumps(legacy_untagged))
    assert pipeline._cached(nv_path) is None

    # Put ROCm cache back at nv_path and verify baseline(), early-grade _sit(), and optional-baseline
    # _grade_job() all treat it as a cache miss instead of reusing AMD measurements.
    nv_path.write_text(json.dumps(rocm_cached))
    retrained = []
    monkeypatch.setattr(pipeline, "train_baseline", lambda r, sc, d, **kw: retrained.append("trained") or types.SimpleNamespace(device=cuda_dev))
    monkeypatch.setattr(
        pipeline,
        "sit_exam",
        lambda tr, r, sc, ex: {
            "valid": True,
            "exam_sha256": full_sha,
            "hardware": nvidia_hw,
            "train": {"init_sha256": "a" * 64},
            "exam": {"items": 2, "mean": -2.4, "item_scores": [-2.4, -2.4]},
        },
    )
    nv_base = pipeline.baseline(rnd, scale, cuda_dev)
    assert retrained == ["trained"] and nv_base["hardware"]["backend"] == "cuda"
    assert pipeline._cached(nv_path, want_exam_sha=full_sha, want_items=2, want_backend="cuda") is not None

    # Overwrite nv_path with ROCm cache again and verify _sit() (baseline=False) returns base=None!
    nv_path.write_text(json.dumps(rocm_cached))
    monkeypatch.setattr(client, "fetch_exam", lambda r, k: None)
    monkeypatch.setattr(grade, "sit_exam", lambda tr, r, sc, ex: {"valid": True, "hardware": nvidia_hw})
    pending = client.Pending({"id": 1, "baseline": False}, rnd, types.SimpleNamespace(model=None), cuda_dev)
    _, sit_base = client._sit(pending, {}, None, scale)
    assert sit_base is None, "early-grade _sit() must not reuse ROCm-labelled baseline on NVIDIA CUDA"

    # Optional-baseline lookup in _grade_job (baseline=False, exam already posted) also returns base=None!
    import hashlib

    import zstandard

    script_bytes = b'{"doc": 0}\n'
    sha = hashlib.sha256(script_bytes).hexdigest()
    monkeypatch.setattr(client, "_job_round", lambda season, job, out: rnd)
    monkeypatch.setattr(client, "request", lambda m, p, **kw: zstandard.ZstdCompressor().compress(script_bytes))
    monkeypatch.setattr(pipeline, "grade_round", lambda r, sc, p, **kw: {"valid": True, "hardware": nvidia_hw})
    out_stub = types.SimpleNamespace(line=lambda *a, **k: None, dim=lambda s: s, progress=lambda *a, **k: None)
    _, opt_base = client._grade_job({"id": 2, "baseline": False, "script": {"sha256": sha}}, s3, scale, cuda_dev, None, out_stub, tmp_path / "work")
    assert opt_base is None, "optional-baseline _grade_job() must not reuse ROCm-labelled baseline on NVIDIA CUDA"

    # 3. slim_result() rejects mismatched baseline hardware and attaches matching baseline.hardware.
    nvidia_grade = {
        "round": {"id": rnd.id},
        "valid": True,
        "hardware": nvidia_hw,
        "script": {"script_sha256": sha, "tokens_sha256": "2" * 64},
        "exam": {"mean": -2.3, "item_scores": [-2.3, -2.3]},
        "train": {"seconds": 10.0, "tokens_per_second": 1000.0, "final_train_loss": 2.3, "init_sha256": "a" * 64, "precision": "bf16", "curve": [[1, 3.0]]},
    }
    with pytest.raises(ValueError, match="backend"):
        client.slim_result(nvidia_grade, rocm_cached)
    with pytest.raises(ValueError, match="backend"):
        client.slim_result(nvidia_grade, legacy_untagged)
    slim_ok = client.slim_result(nvidia_grade, nv_base)
    assert slim_ok["baseline"]["hardware"] == nvidia_hw


def test_review13_slim_result_enforces_baseline_backend_without_torch():
    """Review 13 P1 #1: pure-Python baseline-backend enforcement in slim_result() runs in the
    lightweight no-Torch test suite as well as the Torch suite."""
    from stet import client

    nvidia_hw = {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0}
    rocm_hw = {"device": "cuda", "backend": "rocm", "gpu": "AMD Radeon 8060S", "vram_gb": 96.0}
    untagged_hw = {"device": "cuda", "gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0}
    grade_payload = {
        "round": {"id": "s3-o00007"},
        "valid": True,
        "hardware": nvidia_hw,
        "script": {"script_sha256": "1" * 64, "tokens_sha256": "2" * 64},
        "exam": {"mean": -2.3, "item_scores": [-2.3, -2.3]},
        "train": {"seconds": 10.0, "tokens_per_second": 1000.0, "final_train_loss": 2.3, "init_sha256": "a" * 64, "precision": "bf16", "curve": [[1, 3.0]]},
    }
    base_rocm = {"valid": True, "hardware": rocm_hw, "train": {"init_sha256": "a" * 64}, "exam": {"mean": -2.5, "item_scores": [-2.5, -2.5]}}
    base_untagged = {"valid": True, "hardware": untagged_hw, "train": {"init_sha256": "a" * 64}, "exam": {"mean": -2.5, "item_scores": [-2.5, -2.5]}}
    base_nv = {"valid": True, "hardware": nvidia_hw, "train": {"init_sha256": "a" * 64}, "exam": {"mean": -2.4, "item_scores": [-2.4, -2.4]}}

    with pytest.raises(ValueError, match="backend"):
        client.slim_result(grade_payload, base_rocm)
    with pytest.raises(ValueError, match="backend"):
        client.slim_result(grade_payload, base_untagged)
    slim_ok = client.slim_result(grade_payload, base_nv)
    assert slim_ok["baseline"]["hardware"] == nvidia_hw


def test_review13_decontam_catches_wrapper4_single_sentence_excerpt_copy():
    """Review 13 P2 #5: an exact single-sentence source excerpt in the corpus must be caught by
    decontam.seen() under every wrapper including wrapper 4 (`Context:\\n...`), for both newly
    generated items (with preserved metadata) and legacy items without metadata."""
    from stet.exam import decontam
    from stet.exam.fresh import reason

    s3 = load_season("s3")
    excerpt = "Coastal hydrologists documented rapid salinity intrusion across twelve estuarine aquifers after the autumn drought."
    prompt = "Why did salinity increase in those aquifers?"
    cont = "Reduced river discharge allowed denser tidal seawater to migrate farther inland through porous sand layers."

    class OneDocCorpus:
        def __len__(self):
            return 1

        def text(self, i: int) -> str:
            return excerpt

    items = []
    for task, n_wrap in (("explain", 6), ("infer", 6), ("synthesis", 3)):
        for wid in range(n_wrap):
            ctx = reason.wrap_prompt(excerpt, prompt, task, wid)
            # Authoritative metadata item (produced by parse_generated_items_detailed)
            items.append(
                Item(
                    kind="answer",
                    corner=f"{task}/en/ccnews",
                    fmt="prose",
                    lang="en",
                    source=f"{task}:ccnews:doc-{wid}",
                    ctx=ctx,
                    cont=cont,
                    prompt=prompt,
                    wrapper_id=wid,
                )
            )
            # Legacy item without explicit prompt/wrapper_id (e.g. archived draws)
            items.append(
                Item(
                    kind="answer",
                    corner=f"{task}/en/ccnews",
                    fmt="prose",
                    lang="en",
                    source=f"{task}:ccnews:legacy-{wid}",
                    ctx=ctx,
                    cont=cont,
                )
            )

    for it in items:
        spans = decontam.scored_texts(it)
        assert spans == [excerpt, cont], f"wrapper leaked prefix/prompt into scored_texts: {spans!r}"

    dropped = decontam.seen(items, OneDocCorpus(), s3)
    assert dropped == set(range(len(items))), f"missed exact excerpt copies: {set(range(len(items))) - dropped}"


def test_review13_public_items_preserves_excerpt_boundary_with_horizontal_rule():
    """Review 13 P2 #6: a third-party excerpt containing a Markdown horizontal rule (`\\n\\n---\\n`),
    `Context:\\n`, or bullet markers must never leak third-party prose into the published `prompt`
    or break `rebuild_withheld_reasoning` SHA verification."""
    from stet.exam import examine
    from stet.exam.farm import pack_items, unpack_items
    from stet.exam.fresh import reason

    tricky_excerpts = [
        "Opening section from third-party article about turbine blades.\n\n---\nSecond third-party paragraph after a Markdown horizontal rule.",
        "Context:\nLeading context marker inside the third-party article itself.\n\n• Bullet line inside the article.",
    ]
    prompt_syn = "Summarize the structural failure mode described in the passage."
    prompt_exp = "Why did the turbine blade fatigue crack propagate along the root?"
    resp = (
        "Cyclic thermal expansion concentrated bending stress near the blade root fillet until microscopic "
        "surface pits joined into a propagating fracture across the alloy grain boundary."
    )

    docs = []
    gens = []
    for ex_idx, ex in enumerate(tricky_excerpts):
        for task, pr, n_wrap in (("synthesis", prompt_syn, 3), ("explain", prompt_exp, 6), ("infer", prompt_exp, 6)):
            for wid in range(n_wrap):
                docs.append({
                    "id": f"doc-{ex_idx}-{task}-{wid}",
                    "source": "ccnews:2026-11-26",
                    "lang": "en",
                    "text": ex,
                    "excerpt": ex,
                    "task_type": task,
                    "wrapper_id": wid,
                })
                gens.append(f"{pr}\nRESPONSE: {resp}")

    items, dropped = reason.parse_generated_items(docs, gens, check_lang=False)
    assert not dropped and len(items) == len(docs), (len(items), dropped)

    # Round-trip through pack_items/unpack_items (reasoning cache & full exam storage).
    unpacked = unpack_items(pack_items(items))
    assert unpacked == items

    pub = examine.public_items(unpacked)
    assert len(pub) == len(items)
    for d, it, row in zip(docs, unpacked, pub, strict=True):
        expected_pr = prompt_syn if d["task_type"] == "synthesis" else prompt_exp
        expected_ex = d["excerpt"]
        assert row["withheld"] is True and row["withheld_excerpt"] is True
        assert row["prompt"] == expected_pr, f"leaked prose into prompt for {d['task_type']} w={d['wrapper_id']}: {row['prompt']!r}"
        assert row["wrapper_id"] == d["wrapper_id"]
        assert row["excerpt_sha256"] == hashlib.sha256(expected_ex.encode("utf-8")).hexdigest()
        assert row["excerpt_chars"] == len(expected_ex)
        assert "Second third-party paragraph" not in json.dumps(row)
        assert "Bullet line inside the article" not in json.dumps(row)
        rebuilt = examine.rebuild_withheld_reasoning(row, expected_ex)
        assert rebuilt == it and rebuilt.id == it.id


def test_review13_exam_timeout_is_round_scoped_across_two_rounds(monkeypatch, tmp_path):
    """Review 13 P2 #7: timing out waiting for Round A's exam must exclude only Round A from
    pre-exam grades (`skip_early_rounds=['s3-o00001']`, keeping `early=True`) so a grade-only
    worker still takes and finishes pre-exam grades for Round B."""
    import sys
    from types import SimpleNamespace

    from stet import bench, client, env, upgrade

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    monkeypatch.setenv("STET_EXAM_WAIT_SECONDS", "30")
    monkeypatch.setenv("STET_DEV", "1")
    monkeypatch.setattr(client, "need_key", lambda: SimpleNamespace(handle="grader-1", pub="pub", priv="priv"))
    monkeypatch.setattr(upgrade, "Upgrader", lambda *a, **k: SimpleNamespace(on=False, consider=lambda *a, **k: None, look=lambda: None, now=lambda: None))
    monkeypatch.setattr(env, "problems", lambda *a, **k: [])
    monkeypatch.setattr(bench, "saved", lambda: None)
    monkeypatch.setattr(client, "remember_announcement", lambda *a, **k: None)
    monkeypatch.setattr(client, "saved_pass", lambda *a, **k: None)
    monkeypatch.setattr(client, "tidy", lambda *a, **k: None)
    monkeypatch.setattr(client, "EXAM_POLL", 10)

    # Stub stet.train so this test runs cleanly even in the lightweight no-Torch venv.
    fake_train = types.ModuleType("stet.train")
    fake_train.pick_device = lambda *a, **k: SimpleNamespace(type="cuda", index=0)
    fake_train.hardware_info = lambda d: {"device": "cuda", "backend": "cuda", "gpu": "RTX 5090", "vram_gb": 32.0, "cuda_capability": "12.0", "bf16_native": True}
    monkeypatch.setitem(sys.modules, "stet.train", fake_train)

    clock = {"t": 1000.0}
    monkeypatch.setattr(client.time, "monotonic", lambda: clock["t"])

    def fake_sleep(sec):
        clock["t"] += sec

    monkeypatch.setattr(client.time, "sleep", fake_sleep)

    round_a_job = {
        "id": 101,
        "kind": "grade",
        "scale": "nano",
        "baseline": False,
        "expires_at": 9999999999,
        "round": {"id": "s3-o00001", "index": 1, "drawn_at": 1000, "exam_digest": ""},
        "editor": {"id": "ed1", "name": "ed1", "version": 1, "owner": "alice", "size": 10},
        "script": {"sha256": "1" * 64, "size": 10},
    }
    round_b_job = {
        "id": 202,
        "kind": "grade",
        "scale": "nano",
        "baseline": False,
        "expires_at": 9999999999,
        "round": {"id": "s3-o00002", "index": 2, "drawn_at": 2000, "exam_digest": ""},
        "editor": {"id": "ed2", "name": "ed2", "version": 1, "owner": "alice", "size": 10},
        "script": {"sha256": "2" * 64, "size": 10},
    }

    claims_seen: list[dict] = []
    releases: list[tuple[int, str]] = []
    submitted: list[int] = []
    state = {"phase": "A", "b_exam_posted": False}

    def fake_request(method, path, body=None, **kw):
        if method == "GET" and path == "/api/v1/season":
            return {"scale": "nano"}
        if method == "POST" and path == "/api/v1/jobs/claim":
            claims_seen.append(dict(body or {}))
            if len(claims_seen) > 20:
                raise AssertionError(f"too many claims: {claims_seen}")
            if state["phase"] == "A":
                state["phase"] = "A_wait"
                return {"job": round_a_job}
            if state["phase"] == "A_wait":
                return {"job": None, "retry_after": 10}
            if state["phase"] == "B_legacy_guard":
                # Simulate an older server ignoring skip_early_rounds and re-offering Round A pre-exam.
                state["phase"] = "B"
                return {"job": round_a_job}
            if state["phase"] == "B":
                # Round B offers ONLY pre-exam grades: requires early=True and Round B not skipped!
                if not body.get("early") or "s3-o00002" in (body.get("skip_early_rounds") or []):
                    return {"job": None, "retry_after": 10}
                state["phase"] = "B_wait"
                return {"job": round_b_job}
            return {"job": None, "retry_after": 10}
        if method == "POST" and path.endswith("/release"):
            jid = int(path.split("/")[4])
            releases.append((jid, (body or {}).get("reason", "")))
            if jid == 101 and state["phase"] == "A_wait":
                state["phase"] = "B_legacy_guard"
            return {"ok": True}
        if method == "POST" and path.endswith("/result"):
            jid = int(path.split("/")[4])
            submitted.append(jid)
            return {"status": "settled"}
        raise AssertionError(f"unexpected request: {method} {path}")

    def fake_exam_now(rnd):
        if rnd.id == "s3-o00002" and state["b_exam_posted"]:
            return SimpleNamespace(id="s3-o00002", exam_digest="e" * 64)
        return None

    def fake_grade_job(job, season, scale, device, key, out, work, bases=None):
        rid = job["round"]["id"]
        if rid == "s3-o00002":
            state["b_exam_posted"] = True
        return client.Pending(job, SimpleNamespace(id=rid, exam_digest=""), SimpleNamespace(model=None), device)

    def fake_sit(pending, bases, key, scale):
        return (
            {
                "round": {"id": pending.rnd.id},
                "valid": True,
                "hardware": {"device": "cuda", "backend": "cuda", "gpu": "RTX 5090"},
                "script": {"script_sha256": "2" * 64, "tokens_sha256": "3" * 64},
                "exam": {"mean": -2.0, "item_scores": [-2.0]},
                "train": {"seconds": 1.0, "tokens_per_second": 100.0, "final_train_loss": 2.0, "init_sha256": "a" * 64, "precision": "bf16", "curve": [[1, 2.0]]},
            },
            None,
        )

    monkeypatch.setattr(client, "request", fake_request)
    monkeypatch.setattr(client, "exam_now", fake_exam_now)
    monkeypatch.setattr(client, "_grade_job", fake_grade_job)
    monkeypatch.setattr(client, "_sit", fake_sit)

    args = SimpleNamespace(
        season="s3",
        device="cuda",
        any_device=True,
        edits=False,  # grade-only worker!
        once=False,
        max_jobs=1,
        exit_when_idle=None,
        auto_upgrade=False,
    )
    client._verify(args, organiser=False, me={"account": {"handle": "grader-1"}})

    # Round A timed out and released; subsequent claims kept early=True with skip_early_rounds=['s3-o00001']!
    assert (101, client.EXAM_WAIT_TIMEOUT_REASON) in releases
    post_timeout_claims = [c for c in claims_seen if "s3-o00001" in (c.get("skip_early_rounds") or [])]
    assert len(post_timeout_claims) >= 2, f"expected round-scoped skip_early_rounds claims, got {claims_seen}"
    assert all(c.get("early") is True for c in post_timeout_claims), f"early was disabled globally: {post_timeout_claims}"
    assert submitted == [202], f"expected Round B pre-exam grade 202 to complete and submit, got {submitted}"


def test_review14_prompt_excerpt_copy_filtered_and_redacted_on_publish():
    """Review 14 P2 #7: reasoning prompts that reproduce source excerpt prose (>50% 5-gram overlap)
    are dropped at generation time (`prompt_excerpt_copy`), and any legacy withheld third-party item
    whose prompt or response reproduces the excerpt is redacted in public_items() with SHA-256
    metadata while preserving exact Item.id and rebuild_withheld_reasoning() fidelity."""
    from stet.exam import examine
    from stet.exam.fresh import reason

    excerpt = (
        "Coastal rail operators announced overnight maintenance across three river bridges after "
        "inspection crews detected accelerated corrosion along the southern steel deck bearings "
        "during heavy winter storm tides."
    )
    copied_prompt = (
        "Why did coastal rail operators announce overnight maintenance across three river bridges "
        "after inspection crews detected accelerated corrosion along the southern steel deck bearings?"
    )
    clean_prompt = "Why are the river crossings closing overnight for urgent structural repairs?"
    clean_resp = (
        "Saltwater spray from winter storm surges accelerated rust on load-bearing deck hardware, "
        "requiring off-peak engineering work before commuter traffic resumes safely."
    )
    copied_resp = (
        "Coastal rail operators announced overnight maintenance across three river bridges after "
        "inspection crews detected accelerated corrosion along the southern steel deck bearings "
        "during heavy winter storm tides."
    )

    assert reason.prompt_copies_excerpt(copied_prompt, excerpt) is True
    assert reason.prompt_copies_excerpt(clean_prompt, excerpt) is False

    # 1. Generation-time filter drops items whose prompt reproduces the excerpt.
    docs = [
        {"id": "copy-1", "source": "ccnews:2026-11-26", "lang": "en", "text": excerpt, "excerpt": excerpt, "task_type": "explain", "wrapper_id": 0},
        {"id": "ok-1", "source": "ccnews:2026-11-26", "lang": "en", "text": excerpt, "excerpt": excerpt, "task_type": "explain", "wrapper_id": 0},
    ]
    gens = [
        f"{copied_prompt}\nRESPONSE: {clean_resp}",
        f"{clean_prompt}\nRESPONSE: {clean_resp}",
    ]
    kept, drops, _ = reason.parse_generated_items_detailed(docs, gens, check_lang=False)
    assert len(kept) == 1 and kept[0].source == "explain:ccnews:2026-11-26:ok-1"
    assert drops == {"prompt_excerpt_copy": 1}

    # 2. Publication-time defense-in-depth redacts copied prompt/cont on legacy withheld rows while
    #    preserving exact Item.id and verified reconstruction via rebuild_withheld_reasoning().
    legacy_copied_prompt_item = Item(
        kind="answer",
        corner="explain/en/ccnews",
        fmt="prose",
        lang="en",
        source="explain:ccnews:2026-11-26:legacy-copy",
        ctx=reason.wrap_prompt(excerpt, copied_prompt, "explain", 2),
        cont=clean_resp,
        prompt=copied_prompt,
        wrapper_id=2,
    )
    legacy_copied_cont_item = Item(
        kind="answer",
        corner="infer/en/ccnews",
        fmt="prose",
        lang="en",
        source="infer:ccnews:2026-11-26:legacy-cont",
        ctx=reason.wrap_prompt(excerpt, clean_prompt, "infer", 1),
        cont=copied_resp,
        prompt=clean_prompt,
        wrapper_id=1,
    )
    pub = examine.public_items([legacy_copied_prompt_item, legacy_copied_cont_item])

    row_p, row_c = pub[0], pub[1]
    assert row_p["id"] == legacy_copied_prompt_item.id
    assert row_p["withheld"] is True and row_p["withheld_excerpt"] is True
    assert row_p["withheld_prompt"] is True and "prompt" not in row_p
    assert row_p["prompt_sha256"] == hashlib.sha256(copied_prompt.encode("utf-8")).hexdigest()
    assert row_p["prompt_chars"] == len(copied_prompt)
    assert "southern steel deck bearings" not in json.dumps(row_p)
    with pytest.raises(ValueError, match="withheld prompt"):
        examine.rebuild_withheld_reasoning(row_p, excerpt)
    with pytest.raises(ValueError, match="prompt sha256 mismatch"):
        examine.rebuild_withheld_reasoning(row_p, excerpt, prompt="wrong prompt")
    rebuilt_p = examine.rebuild_withheld_reasoning(row_p, excerpt, prompt=copied_prompt)
    assert rebuilt_p == legacy_copied_prompt_item and rebuilt_p.id == legacy_copied_prompt_item.id

    assert row_c["id"] == legacy_copied_cont_item.id
    assert row_c["withheld_cont"] is True and "cont" not in row_c
    assert row_c["cont_sha256"] == hashlib.sha256(copied_resp.encode("utf-8")).hexdigest()
    assert "southern steel deck bearings" not in json.dumps(row_c)
    rebuilt_c = examine.rebuild_withheld_reasoning(row_c, excerpt, cont=copied_resp)
    assert rebuilt_c == legacy_copied_cont_item and rebuilt_c.id == legacy_copied_cont_item.id


def test_review14_bundled_practice_pack_uses_inkling2_and_passes_copy_filter():
    """Review 14 P3 #8: the bundled Season 3 practice pack ships Inkling-2-Mini CC-BY-SA Wikipedia
    reasoning items (matching the production examiner teacher), carries authoritative prompt/wrapper
    metadata with zero excerpt-copying prompts, and attributes Inkling-2-Mini in notices & README."""
    from stet.exam.farm import unpack_items
    from stet.exam.fresh import reason

    repo_root = Path(__file__).resolve().parents[1]
    pack_path = repo_root / "src/stet/exam/data/s3-practice.jsonl.zst"
    s3 = load_season("s3")
    raw = pack_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == s3.raw["exam"]["practice_pack"]["sha256"]

    items = unpack_items(raw)
    ctx_items = [it for it in items if it.kind == "context"]
    ans_items = [it for it in items if it.kind == "answer"]
    assert len(ctx_items) == 221
    assert len(ans_items) == 287
    by_task: dict[str, int] = {}
    for it in ans_items:
        task = it.source.split(":", 1)[0]
        by_task[task] = by_task.get(task, 0) + 1
        assert it.source.startswith(f"{task}:wikipedia"), f"practice pack must only use CC-BY-SA Wikipedia sources: {it.source}"
        assert isinstance(it.prompt, str) and it.prompt and isinstance(it.wrapper_id, int)
        excerpt = reason.extract_wrapped_excerpt(it.ctx, it.prompt, task, it.wrapper_id)
        assert excerpt is not None, f"could not extract excerpt for {it.id}"
        assert not reason.prompt_copies_excerpt(it.prompt, excerpt), f"prompt copies excerpt in {it.id}"
    assert by_task == {"explain": 96, "infer": 95, "synthesis": 96}

    notices = (repo_root / "src/stet/exam/data/THIRD_PARTY_NOTICES.md").read_text()
    assert "thinkingmachines/Inkling-2-Mini" in notices
    assert "Qwen3-8B" not in notices
    readme = (repo_root / "README.md").read_text()
    assert "Inkling-2-Mini" in readme and "STET_SYNTHESIZE_PRACTICE=1" in readme


def test_review15_examine_derives_season_from_round_id_and_rejects_conflict(monkeypatch):
    """Review 15 P2 #5: `stet examine --round s2-o00003` must target season s2 (not silently
    default to s3 and query /api/v1/rounds/s3-o00003), and an explicit conflicting `--season s3`
    must be rejected before any network call is made."""
    from types import SimpleNamespace

    from stet import cli, client

    fetched: list[tuple[str, int]] = []
    requested: list[tuple[str, str]] = []

    def fake_fetch_round(season, index):
        fetched.append((season.id, index))
        return SimpleNamespace(id=f"{season.id}-o{index:05d}", seed=b"\x00" * 32)

    def fake_request(method, path, body=None, **kw):
        requested.append((method, path))
        return {"round": {"id": path.rsplit("/", 1)[-1], "exam_digest": "a" * 64}}

    monkeypatch.setattr(client, "fetch_round", fake_fetch_round)
    monkeypatch.setattr(client, "request", fake_request)

    parser = cli.build_parser()

    # 1. `stet examine --round s2-o00003` (no --season) targets season s2, index 3!
    args_s2 = parser.parse_args(["examine", "--round", "s2-o00003", "--quiet"])
    assert args_s2.season is None
    cli.cmd_examine(args_s2)
    assert fetched == [("s2", 3)]
    assert requested == [("GET", "/api/v1/rounds/s2-o00003")]

    # 2. Explicit conflicting `--season s3` fails BEFORE fetching anything!
    fetched.clear()
    requested.clear()
    args_conflict = parser.parse_args(["examine", "--round", "s2-o00003", "--season", "s3", "--quiet"])
    with pytest.raises(SystemExit) as exc:
        cli.cmd_examine(args_conflict)
    assert exc.value.code != 0
    assert fetched == [] and requested == []

    # 3. Matching explicit `--season s2` succeeds cleanly.
    args_match = parser.parse_args(["examine", "--round", "s2-o00003", "--season", "s2", "--quiet"])
    cli.cmd_examine(args_match)
    assert fetched == [("s2", 3)]
    assert requested == [("GET", "/api/v1/rounds/s2-o00003")]


def test_review15_reasoning_cache_versioned_and_revalidated_on_upgrade(monkeypatch, tmp_path):
    """Review 15 P2 #6: upgrading generation/filter rules invalidates the reasoning cache stem
    (via REASONING_FILTER_VERSION) AND `_reasoning_cache_usable` revalidates every cached row
    against current filters so a previously-accepted prompt-copying row triggers clean regeneration."""
    import zstandard as zstd

    from stet.exam import examine
    from stet.exam.farm import pack_items
    from stet.exam.fresh import Doc, reason, write_snapshot
    from stet.spec import practice_round

    s3 = load_season("s3")
    r_cfg = s3.raw["exam"]["reasoning"]
    rnd = practice_round(s3, 0)
    scale = s3.scale("nano")

    excerpt = (
        "Coastal rail operators announced overnight maintenance across three river bridges after "
        "inspection crews detected accelerated corrosion along the southern steel deck bearings "
        "during heavy winter storm tides."
    )
    copied_prompt = (
        "Why did coastal rail operators announce overnight maintenance across three river bridges "
        "after inspection crews detected accelerated corrosion along the southern steel deck bearings?"
    )
    clean_prompt = "Why are the river crossings closing overnight for urgent structural repairs?"
    clean_resp = (
        "Saltwater spray from winter storm surges accelerated rust on load-bearing deck hardware, "
        "requiring off-peak engineering work before commuter traffic resumes safely."
    )

    bad_copied_item = Item(
        kind="answer",
        corner="explain/en/ccnews",
        fmt="prose",
        lang="en",
        source="explain:ccnews:2026-11-26:old-copied",
        ctx=reason.wrap_prompt(excerpt, copied_prompt, "explain", 0),
        cont=clean_resp,
        prompt=copied_prompt,
        wrapper_id=0,
    )
    bad_legacy_no_prompt = Item(
        kind="answer",
        corner="explain/en/ccnews",
        fmt="prose",
        lang="en",
        source="explain:ccnews:2026-11-26:old-no-meta",
        ctx=reason.wrap_prompt(excerpt, clean_prompt, "explain", 0),
        cont=clean_resp,
    )
    good_items = [
        Item(
            kind="answer",
            corner="explain/en/ccnews",
            fmt="prose",
            lang="en",
            source=f"explain:ccnews:2026-11-26:clean-{i}",
            ctx=reason.wrap_prompt(excerpt, clean_prompt, "explain", i % 6),
            cont=f"{clean_resp} Bridge section {i}.",
            prompt=clean_prompt,
            wrapper_id=i % 6,
        )
        for i in range(45)
    ]

    # 1. Direct revalidation rejects both prompt-copying rows and metadata-less legacy rows.
    assert reason.cached_item_passes_filter(bad_copied_item) is False
    assert reason.cached_item_passes_filter(bad_legacy_no_prompt) is False
    assert reason.cached_item_passes_filter(good_items[0]) is True
    assert examine._reasoning_cache_usable(good_items + [bad_copied_item], s3) is False
    assert examine._reasoning_cache_usable(good_items + [bad_legacy_no_prompt], s3) is False
    assert examine._reasoning_cache_usable(good_items, s3) is True

    # 2. Cache key differs from pre-versioned r_cfg-only digest, AND placing a poisoned pack
    #    at the exact versioned cache path forces regeneration on examine.assemble().
    snap_path = tmp_path / "2026-11-26-00.jsonl.zst"
    write_snapshot(
        [Doc(source="ccnews", id="d1", url="https://example.invalid/1", lang="en", license="ccnews", published=1_800_000_000, title="T", text=excerpt)],
        snap_path,
        {"slot": "2026-11-26-00"},
    )
    shas = [hashlib.sha256(snap_path.read_bytes()).hexdigest()]
    old_sig = hashlib.sha256(json.dumps(r_cfg, sort_keys=True).encode()).hexdigest()[:8]
    new_sig = hashlib.sha256(
        json.dumps({"filter_version": reason.REASONING_FILTER_VERSION, "cfg": r_cfg}, sort_keys=True).encode()
    ).hexdigest()[:8]
    assert new_sig != old_sig

    work = tmp_path / "work"
    work.mkdir()
    cur_stem = f"{examine.snapshot_name(shas)[:12]}-{rnd.seed.hex()[:8]}-{new_sig}"
    cur_cache = work / f"reasoning-{cur_stem}.jsonl.zst"
    cur_cache.write_bytes(pack_items(good_items + [bad_copied_item]))

    synth_calls = []

    def fake_synth(docs, seed, cfg, **kw):
        synth_calls.append(len(docs))
        return list(good_items)

    monkeypatch.setattr("stet.exam.fresh.reason.synthesize_reasoning", fake_synth)
    monkeypatch.setattr("stet.data.corpus", lambda *a, **k: tmp_path / "empty.jsonl.zst")
    (tmp_path / "empty.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(b""))
    monkeypatch.setattr("stet.exam.decontam.seen", lambda *a, **k: set())
    monkeypatch.setattr("stet.exam.fresh.select.select_context", lambda *a, **k: [])

    # First assemble: poisoned cache at cur_cache is rejected & regenerated!
    items1, _, _, _ = examine.assemble(rnd, scale, [(0, snap_path)], work, measure=False, synthesize=True, check_coverage=False, log=lambda _: None)
    assert len(synth_calls) == 1
    assert all(it.source != bad_copied_item.source for it in items1)

    # Second assemble: freshly written clean cache IS reused with zero extra synthesis calls!
    items2, _, _, _ = examine.assemble(rnd, scale, [(0, snap_path)], work, measure=False, synthesize=True, check_coverage=False, log=lambda _: None)
    assert len(synth_calls) == 1
    assert items2 == items1


def test_review16_practice_cache_invalidated_on_reasoning_filter_upgrade(monkeypatch, tmp_path):
    """Review 16 P2 #5: upgrading REASONING_FILTER_VERSION invalidates BOTH the synthesized
    practice exam cache and its shared baseline cache key, AND `_practice()` revalidates cached
    synthesized reasoning rows against `cached_item_passes_filter` before returning them."""
    import importlib.util

    from stet.exam.examine import snapshot_name
    from stet.exam.farm import pack_items
    from stet.exam.fresh import Doc, reason, write_snapshot

    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    scale = s3.scale("nano")

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    monkeypatch.setenv("STET_SYNTHESIZE_PRACTICE", "1")
    monkeypatch.setattr("stet.exam.fresh.reason.can_synthesize_locally", lambda *a, **k: True)

    excerpt = (
        "Municipal ferry operators suspended two harbor routes on Tuesday evening after sonar surveys "
        "detected submerged timber pilings blocking the southern approach channel during low tide."
    )
    copied_prompt = (
        "Why did municipal ferry operators suspend two harbor routes on Tuesday evening after sonar "
        "surveys detected submerged timber pilings blocking the southern approach channel?"
    )
    clean_prompt = "What hazard forced evening ferry closures along the harbor channel?"
    clean_resp = (
        "Underwater inspection identified sunken wooden dock pilings obstructing shallow-water navigation, "
        "prompting temporary suspensions until salvage crews clear the shipping lane."
    )

    hdir = data.harvest_dir()
    hdir.mkdir(parents=True, exist_ok=True)
    snap_paths = []
    for h in (1_800_000_000, 1_800_003_600, 1_800_007_200):
        sp = hdir / f"{h}.jsonl"
        write_snapshot(
            [Doc(source="ccnews", id=f"d-{h}", url=f"https://example.invalid/{h}", lang="en", license="ccnews", published=h, title="T", text=excerpt)],
            sp,
            {"hour": h},
        )
        snap_paths.append(sp)

    # 1. Versioned practice_key differs from pre-versioned `f"{base}:synth"` AND rolls over on bump.
    hours = data.practice_hours(rnd)
    base = snapshot_name([hashlib.sha256(p.read_bytes()).hexdigest() for _, p in hours])
    old_unversioned_key = hashlib.sha256(f"{base}:synth".encode()).hexdigest()
    cur_key = data.practice_key(rnd)
    assert cur_key != old_unversioned_key

    has_torch = importlib.util.find_spec("torch") is not None
    pipeline = importlib.import_module("stet.pipeline") if has_torch else None
    b_cur = pipeline.baseline_path(rnd, scale, types.SimpleNamespace(type="cpu")) if pipeline else None

    monkeypatch.setattr(reason, "REASONING_FILTER_VERSION", reason.REASONING_FILTER_VERSION + 1)
    bumped_key = data.practice_key(rnd)
    b_bumped = pipeline.baseline_path(rnd, scale, types.SimpleNamespace(type="cpu")) if pipeline else None
    assert bumped_key != cur_key
    if pipeline is not None:
        assert b_bumped != b_cur
    monkeypatch.setattr(reason, "REASONING_FILTER_VERSION", reason.REASONING_FILTER_VERSION - 1)

    bad_copied_item = Item(
        kind="answer",
        corner="explain/en/ccnews",
        fmt="prose",
        lang="en",
        source="explain:ccnews:2026-11-26:old-copied",
        ctx=reason.wrap_prompt(excerpt, copied_prompt, "explain", 0),
        cont=clean_resp,
        prompt=copied_prompt,
        wrapper_id=0,
    )
    clean_item = Item(
        kind="answer",
        corner="explain/en/ccnews",
        fmt="prose",
        lang="en",
        source="explain:ccnews:2026-11-26:clean-0",
        ctx=reason.wrap_prompt(excerpt, clean_prompt, "explain", 0),
        cont=clean_resp,
        prompt=clean_prompt,
        wrapper_id=0,
    )
    assert reason.cached_item_passes_filter(bad_copied_item) is False
    assert reason.cached_item_passes_filter(clean_item) is True

    # Place poisoned packs at BOTH the old pre-versioned path AND the current versioned path!
    rdir = data.round_dir(rnd, scale)
    rdir.mkdir(parents=True, exist_ok=True)
    old_cache = rdir / f"exam-practice-{rnd.season.digest[:8]}-{old_unversioned_key[:16]}.jsonl.zst"
    cur_cache = rdir / f"exam-practice-{rnd.season.digest[:8]}-{cur_key[:16]}.jsonl.zst"
    old_cache.write_bytes(pack_items([bad_copied_item]))
    cur_cache.write_bytes(pack_items([bad_copied_item]))

    assemble_calls = []

    def fake_assemble(*a, **k):
        assemble_calls.append(k.get("synthesize"))
        return ([clean_item], 1, {"en": 100.0}, [])

    monkeypatch.setattr("stet.exam.examine.assemble", fake_assemble)

    # First call rejects the poisoned cached row and re-assembles cleanly!
    got1 = data.round_exam(rnd, scale)
    assert assemble_calls == [True]
    assert got1 == [clean_item]

    # Second call reuses the freshly written clean cache without re-assembling!
    got2 = data.round_exam(rnd, scale)
    assert assemble_calls == [True]
    assert got2 == [clean_item]


def test_review16_wheel_bundles_upstream_license_texts(tmp_path):
    """Review 16 P2 #7: the built stet-cli wheel must distribute full upstream license texts
    for Apache 2.0 (Global-MMLU / OpenBookQA) and GSM8K's MIT copyright & permission notice."""
    import importlib.resources as res
    import shutil
    import subprocess
    import zipfile

    # 1. Verify bundled package resources directly.
    lic_dir = res.files("stet.exam").joinpath("data", "licenses")
    apache_text = lic_dir.joinpath("APACHE-2.0.txt").read_text(encoding="utf-8")
    gsm8k_text = lic_dir.joinpath("GSM8K-MIT.txt").read_text(encoding="utf-8")
    assert "Apache License" in apache_text and "Version 2.0, January 2004" in apache_text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in apache_text
    assert "Copyright (c) 2021 OpenAI" in gsm8k_text and "Permission is hereby granted, free of charge" in gsm8k_text

    # 2. Build the actual wheel into tmp_path and inspect its .dist-info/licenses/ archive entries.
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    built = False
    try:
        from hatchling.builders.wheel import WheelBuilder

        WheelBuilder(str(repo_root)).build_standard(str(out_dir))
        built = True
    except ImportError:
        uv_bin = shutil.which("uv")
        if uv_bin:
            subprocess.run(
                [uv_bin, "build", "--wheel", "--out-dir", str(out_dir), str(repo_root)],
                check=True,
                capture_output=True,
                text=True,
            )
            built = True
    assert built, "neither hatchling nor uv was available to build the wheel artifact"

    wheels = list(out_dir.glob("stet_cli-*.whl"))
    assert len(wheels) == 1, f"expected 1 built wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
        dist_licenses = {n: zf.read(n).decode("utf-8") for n in names if ".dist-info/licenses/" in n}
        assert any(n.endswith("THIRD_PARTY_NOTICES.md") for n in dist_licenses), f"missing notices: {dist_licenses.keys()}"
        apache_entries = [txt for n, txt in dist_licenses.items() if n.endswith("APACHE-2.0.txt")]
        gsm8k_entries = [txt for n, txt in dist_licenses.items() if n.endswith("GSM8K-MIT.txt")]
        assert len(apache_entries) == 1 and "Version 2.0, January 2004" in apache_entries[0]
        assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in apache_entries[0]
        assert len(gsm8k_entries) == 1 and "Copyright (c) 2021 OpenAI" in gsm8k_entries[0]
        assert "Permission is hereby granted, free of charge" in gsm8k_entries[0]


def test_review17_official_baseline_cache_bound_to_verifier_and_gpu_and_prefers_in_memory(monkeypatch, tmp_path):
    """Review 17 P1 #1: official baseline reuse MUST be bound to both the producing verifier key
    and the concrete GPU model, and early grading MUST prefer sitting the verifier's own
    already-trained baseline when the job requests one."""
    pytest.importorskip("torch")
    import torch

    from stet import cli, client, data, grade, keys, pipeline
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    rnd = official_round(
        s3,
        17,
        {"round": 123456, "randomness": "ab" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 100000},
    )
    exam = [
        Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source="explain:wikipedia:en:1", ctx="Q1", cont=" A1"),
        Item(kind="answer", corner="infer/en/wiki", fmt="prose", lang="en", source="infer:wikipedia:en:2", ctx="Q2", cont=" A2"),
    ]
    full_sha = data.exam_digest(exam)
    rnd = dataclasses.replace(rnd, exam_digest=full_sha)

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(data, "round_exam", lambda r, sc, **kw: exam)
    monkeypatch.setattr(client, "fetch_exam", lambda r, k: None)

    k1 = keys.create_key()
    k2 = keys.create_key()
    dev_5090 = types.SimpleNamespace(type="cuda", gpu="NVIDIA GeForce RTX 5090")
    dev_4090 = types.SimpleNamespace(type="cuda", gpu="NVIDIA GeForce RTX 4090")
    hw_5090 = {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0}
    hw_4090 = {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0}

    # 1. Cache paths separate different verifiers AND different GPU models on the same exam.
    p_k1_5090 = pipeline.baseline_path(rnd, scale, dev_5090, verifier=k1.public_hex)
    p_k2_5090 = pipeline.baseline_path(rnd, scale, dev_5090, verifier=k2.public_hex)
    p_k1_4090 = pipeline.baseline_path(rnd, scale, dev_4090, verifier=k1.public_hex)
    p_k2_4090 = pipeline.baseline_path(rnd, scale, dev_4090, verifier=k2.public_hex)
    assert len({p_k1_5090, p_k2_5090, p_k1_4090, p_k2_4090}) == 4

    sat_calls: list[tuple[str, str]] = []

    def fake_sit_exam(trained, r, sc, ex):
        tag = getattr(trained, "tag", "unknown")
        hw = hw_5090 if getattr(trained.device, "gpu", "") == "NVIDIA GeForce RTX 5090" else hw_4090
        score = -2.10 if hw is hw_5090 else -2.35
        sat_calls.append((tag, hw["gpu"]))
        return {
            "valid": True,
            "exam_sha256": full_sha,
            "hardware": hw,
            "train": {"init_sha256": "a" * 64},
            "exam": {"items": len(ex), "mean": score, "item_scores": [score] * len(ex)},
        }

    monkeypatch.setattr(pipeline, "sit_exam", fake_sit_exam)
    monkeypatch.setattr(grade, "sit_exam", fake_sit_exam)

    # 2. Verifier 1 (RTX 5090) sits its early grade and populates the shared STET_CACHE.
    bases1 = {rnd.id: types.SimpleNamespace(device=dev_5090, model=object(), tag="base-v1-5090")}
    p1 = client.Pending({"id": 1, "baseline": True}, rnd, types.SimpleNamespace(device=dev_5090, tag="cand-v1"), dev_5090)
    res1, base1 = client._sit(p1, bases1, k1, scale)
    assert base1["hardware"]["gpu"] == "NVIDIA GeForce RTX 5090" and base1["exam"]["mean"] == -2.10
    assert "verifier" not in base1
    assert p_k1_5090.exists()

    # 3. Verifier 2 (RTX 4090, different key) has its own trained baseline in memory AND baseline=True:
    #    MUST sit Verifier 2's own RTX 4090 baseline (NEVER reuse Verifier 1's RTX 5090 cache!).
    bases2 = {rnd.id: types.SimpleNamespace(device=dev_4090, model=object(), tag="base-v2-4090")}
    p2 = client.Pending({"id": 2, "baseline": True}, rnd, types.SimpleNamespace(device=dev_4090, tag="cand-v2"), dev_4090)
    res2, base2 = client._sit(p2, bases2, k2, scale)
    assert base2["hardware"]["gpu"] == "NVIDIA GeForce RTX 4090" and base2["exam"]["mean"] == -2.35
    assert ("base-v2-4090", "NVIDIA GeForce RTX 4090") in sat_calls
    assert p_k2_4090.exists()

    # 4. A third verifier with baseline=False on RTX 5090 MUST NOT borrow Verifier 1's cached baseline.
    k3 = keys.create_key()
    p3 = client.Pending({"id": 3, "baseline": False}, rnd, types.SimpleNamespace(device=dev_5090, tag="cand-v3"), dev_5090)
    _, base3 = client._sit(p3, {}, k3, scale)
    assert base3 is None

    # 5. Even if Verifier 1's cache file were copied onto Verifier 3's path, _cached rejects wrong verifier,
    #    and slim_result rejects mismatched GPU hardware as defense in depth.
    p_k3_5090 = pipeline.baseline_path(rnd, scale, dev_5090, verifier=k3.public_hex)
    p_k3_5090.write_text(p_k1_5090.read_text())
    assert pipeline._cached(p_k3_5090, want_exam_sha=full_sha, want_backend="cuda", want_gpu="NVIDIA GeForce RTX 5090", want_verifier=k3.public_hex) is None
    assert pipeline._cached(p_k1_5090, want_exam_sha=full_sha, want_backend="cuda", want_gpu="NVIDIA GeForce RTX 4090", want_verifier=k1.public_hex) is None

    with pytest.raises(ValueError, match="baseline GPU"):
        client.slim_result(
            {"round": {"id": rnd.id}, "valid": False, "hardware": hw_4090},
            base1,
        )

    # 6. Late _grade_job on Verifier 2 (different key) after exam posted does NOT reuse Verifier 1's file.
    trained_late: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "train_baseline",
        lambda r, sc, d, **kw: trained_late.append("v3") or types.SimpleNamespace(device=d, tag="late-v3"),
    )
    base_late = pipeline.baseline(rnd, scale, dev_5090, verifier=k3.public_hex)
    assert trained_late == ["v3"] and base_late["hardware"]["gpu"] == "NVIDIA GeForce RTX 5090"
    _ = cli


def test_review17_rejected_409_publication_preserves_accepted_exam_for_whole_exam(monkeypatch, tmp_path):
    """Review 17 P2 #2: overlapping exam builds sharing a round work directory stage each
    candidate under an immutable digest-addressed directory before posting and only promote to
    KEPT_ITEMS / KEPT_LANGUAGES after post() succeeds, so a 409 rejection never clobbers the
    accepted exam on disk and stages.whole_exam() always recovers the committed exam."""
    import base64

    from stet import client
    from stet.exam import examine, stages
    from stet.exam.farm import pack_items
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    rnd = official_round(
        s3,
        18,
        {"round": 123456, "randomness": "ab" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 100000},
    )

    items_a = [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source="explain:wikipedia:en:100", ctx="Ctx A", cont=" Ans A")]
    items_b = [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source="explain:wikipedia:en:200", ctx="Ctx B", cont=" Ans B")]
    digest_a = data.exam_digest(items_a)
    digest_b = data.exam_digest(items_b)
    assert digest_a != digest_b

    exam_a = {
        "round": rnd.id,
        "items_digest": digest_a,
        "items_file": base64.b64encode(pack_items(items_a)).decode(),
        "languages": {"en": 100.0},
        "table": [{"id": items_a[0].id}],
    }
    exam_b = {
        "round": rnd.id,
        "items_digest": digest_b,
        "items_file": base64.b64encode(pack_items(items_b)).decode(),
        "languages": {"en": 99.0},
        "table": [{"id": items_b[0].id}],
    }

    work = tmp_path / "examine" / rnd.id
    built_queue = iter([exam_a, exam_b])
    monkeypatch.setattr(examine, "build", lambda *a, **kw: next(built_queue))

    accepted_digest = {"val": None}

    def fake_post(ex, key):
        if accepted_digest["val"] is None:
            accepted_digest["val"] = ex["items_digest"]
            return {"ok": True}
        raise client.ApiError(409, "round already has an exam")

    monkeypatch.setattr(examine, "post", fake_post)

    # Build A succeeds and promotes A to KEPT_ITEMS / KEPT_LANGUAGES.
    examine.examine(rnd, scale, [], work, key=None, log=lambda *a: None)
    # Overlapping build B finishes later and receives 409 -> MUST NOT overwrite A!
    with pytest.raises(client.ApiError, match="409|already"):
        examine.examine(rnd, scale, [], work, key=None, log=lambda *a: None)

    rnd_settled = dataclasses.replace(rnd, exam_digest=digest_a)
    got_items, got_shares = stages.whole_exam(rnd_settled, tmp_path / "stages_work")
    assert got_items == items_a and got_shares == {"en": 100.0}

    # Even if KEPT_ITEMS were missing/corrupted after A staged & posted, whole_exam recovers A from candidate-<digest>.
    (work / examine.KEPT_ITEMS).unlink()
    got_items2, got_shares2 = stages.whole_exam(rnd_settled, tmp_path / "stages_work_2")
    assert got_items2 == items_a and got_shares2 == {"en": 100.0}


def test_review17_advancing_harvest_pins_practice_exam_and_checks_digests(monkeypatch, tmp_path):
    """Review 17 P2 #8:
    (a) `_practice()` MUST derive its cache filename from the passed `hours` rather than calling
        `practice_hours()` again, so an advancing harvest between selection and caching cannot
        store exam A under key B.
    (b) `cli._with_baseline()` MUST check both candidate and baseline `exam_sha256` against the
        pinned exam digest rather than subtracting scores from two disjoint equal-length exams."""
    from stet import cli

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s2 = load_season("s2")
    rnd = practice_round(s2, 3)
    scale = s2.scale("nano")

    h_dir = data.harvest_dir()
    h_dir.mkdir(parents=True, exist_ok=True)
    h1 = h_dir / "1700000000.jsonl"
    h2 = h_dir / "1700003600.jsonl"
    h1.write_text('{"hour":1700000000,"v":1}\n')

    items_a = [Item(kind="answer", corner="logic/a", fmt="prose", lang="en", source="gen:logic", ctx="A?", cont=" 1")]
    items_b = [Item(kind="answer", corner="logic/b", fmt="prose", lang="en", source="gen:logic", ctx="B?", cont=" 2")]
    sha_a = data.exam_digest(items_a)
    sha_b = data.exam_digest(items_b)
    assert sha_a != sha_b

    # Simulate harvest advancing right after round_exam() selected hours A=[(1700000000, h1)]:
    hours_a = data.practice_hours(rnd)
    key_a = data.practice_key(rnd, hours_a)
    h2.write_text('{"hour":1700003600,"v":2}\n')
    hours_b = data.practice_hours(rnd)
    key_b = data.practice_key(rnd, hours_b)
    assert key_a != key_b

    from stet.exam import examine

    def fake_assemble(r, sc, hrs, work, **kw):
        return (items_a if hrs == hours_a else items_b, 1, {"en": 100.0}, [])

    monkeypatch.setattr(examine, "assemble", fake_assemble)

    # Calling _practice(rnd, scale, hours_a) AFTER h2 appeared MUST write under key_a, NOT key_b!
    got_a = data._practice(rnd, scale, hours_a)
    assert got_a == items_a
    # Subsequent round_exam(rnd, scale) with hours_b selected MUST return items_b (never poisoned with items_a!).
    got_b = data.round_exam(rnd, scale)
    assert got_b == items_b

    # Disjoint equal-length candidate vs baseline digests MUST raise RuntimeError in _with_baseline!
    monkeypatch.setitem(
        sys.modules,
        "stet.pipeline",
        types.SimpleNamespace(
            baseline=lambda *a, **k: {
                "valid": True,
                "exam_sha256": sha_b,
                "exam": {"mean": -3.0, "item_scores": [-3.0]},
            }
        ),
    )
    res_a = {
        "valid": True,
        "exam_sha256": sha_a,
        "exam": {"mean": -2.0, "item_scores": [-2.0]},
    }
    with pytest.raises(RuntimeError, match="cannot compute gain"):
        cli._with_baseline(res_a, rnd, scale, None, cli.Out(quiet=True), skip=False, exam=items_a, practice_key=key_a)

    # (c) Direct pipeline.baseline() and gym.Env(split="practice").step() also pin practice hours/exam!
    if importlib.util.find_spec("torch") is not None:
        from stet import gym, pipeline

        # Advance harvest during train_baseline -> sit_baseline MUST still sit exam_b and cache under key_b!
        h3 = h_dir / "1700007200.jsonl"
        sat_exams: list[str] = []

        def fake_train_base(r, sc, d, **kw):
            h3.write_text('{"hour":1700007200,"v":3}\n')
            return types.SimpleNamespace(device=d)

        def fake_sit_base_exam(tr, r, sc, ex):
            sat_exams.append(data.exam_digest(ex))
            return {
                "valid": True,
                "exam_sha256": data.exam_digest(ex),
                "hardware": {"device": "cpu"},
                "exam": {"items": len(ex), "mean": -3.0, "item_scores": [-3.0] * len(ex)},
            }

        monkeypatch.setattr(pipeline, "train_baseline", fake_train_base)
        monkeypatch.setattr(pipeline, "sit_exam", fake_sit_base_exam)
        base_b = pipeline.baseline(rnd, scale, device=types.SimpleNamespace(type="cpu"))
        assert sat_exams == [sha_b] and base_b["exam_sha256"] == sha_b
        assert pipeline.baseline_path(rnd, scale, types.SimpleNamespace(type="cpu"), practice_key=key_b).exists()

        # gym.Env(split="practice").step() rejects mismatched baseline vs candidate digests.
        env = gym.Env(scale="nano", split="practice", device="cpu", season="s2")
        monkeypatch.setattr(data, "corpus", lambda *a, **k: types.SimpleNamespace(total_bytes=100, meta={"sha256": "0" * 64}, __len__=lambda self: 1))
        monkeypatch.setattr("stet.runner.task_info", lambda *a, **k: {"train_tokens": 1024})
        env.reset(3)
        monkeypatch.setattr(
            gym,
            "baseline",
            lambda *a, **k: {"valid": True, "exam_sha256": sha_a, "exam": {"mean": -3.0, "item_scores": [-3.0]}},
        )
        monkeypatch.setattr(
            "stet.grade.grade",
            lambda *a, **k: {
                "valid": True,
                "invalid_reason": None,
                "exam_sha256": sha_b,
                "script": {"out_docs": 1},
                "train": {"tokens": 1024},
                "exam": {"items": 1, "mean": -2.0, "item_scores": [-2.0]},
            },
        )
        with pytest.raises(RuntimeError, match="cannot compute reward"):
            env.step([0])


def test_review17_wikipedia_donor_sources_and_attribution_manifest_in_wheel(tmp_path):
    """Review 17 P2 #9:
    - `select_context()` preserves donor revision IDs in `Item.wrong_sources` (1:1 with `wrongs`).
    - `public_items()` preserves `wrong_sources` and per-passage Wikimedia revision/history URLs
      while `read_public_info()` still matches the round's exact `exam_digest`.
    - The built `stet-cli` wheel bundles `s3-practice.jsonl.zst` with `wrong_sources` on all 221
      comprehension items AND `WIKIPEDIA_ATTRIBUTION.jsonl` (both in `stet/exam/data/` and
      `.dist-info/licenses/`) covering 100% of primary, donor, and reasoning Wikipedia passages."""
    import shutil
    import subprocess
    import zipfile

    from stet.exam import examine
    from stet.exam.farm import unpack_items
    from stet.exam.fresh import Doc
    from stet.exam.fresh.select import select_context

    # 1. Selector preserves donor wrong_sources and public_items carries revision/history attribution.
    docs = [
        Doc(
            source="wikipedia",
            id=f"en:{1000 + i}",
            lang="en",
            title=f"Article {i}",
            url=f"https://en.wikipedia.org/wiki/Article_{i}",
            license="CC BY-SA 4.0",
            published=1_800_000_000 + i,
            text=(
                f"Opening paragraph number {i} describes the distinct geological history of basin {i} "
                f"with carefully measured sedimentary layers and mineral formations across the valley.\n\n"
                f"Continuation paragraph number {i} explains how later tectonic uplift reshaped region {i} "
                f"and altered downstream river courses over several million years."
            ),
        )
        for i in range(6)
    ]
    selected = select_context(docs, b"review17-seed", 4, {"reference"}, per_corner=10)
    assert len(selected) >= 4
    for it in selected:
        assert it.kind == "context"
        assert len(it.wrong_sources) == len(it.wrongs) and len(it.wrong_sources) > 0
        assert all(ws.startswith("wikipedia:en:") and ws != it.source for ws in it.wrong_sources)

    pub_rows = examine.public_items(selected)
    for it, row in zip(selected, pub_rows, strict=True):
        assert row["wrong_sources"] == list(it.wrong_sources)
        attr_by_src = {a["source"]: a for a in row["attribution"]}
        for sid in (it.source, *it.wrong_sources):
            assert sid in attr_by_src
            assert "oldid=" in attr_by_src[sid]["revision_url"]
            assert "action=history" in attr_by_src[sid]["history_url"]
    raw_pub = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pub_rows).encode("utf-8")
    kept, total = examine.read_public_info(raw_pub, data.exam_digest(selected))
    assert kept == selected and total == len(selected)

    # 2. Build actual wheel and verify s3-practice.jsonl.zst + WIKIPEDIA_ATTRIBUTION.jsonl inside it.
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    built = False
    try:
        from hatchling.builders.wheel import WheelBuilder

        WheelBuilder(str(repo_root)).build_standard(str(out_dir))
        built = True
    except ImportError:
        uv_bin = shutil.which("uv")
        if uv_bin:
            subprocess.run(
                [uv_bin, "build", "--wheel", "--out-dir", str(out_dir), str(repo_root)],
                check=True,
                capture_output=True,
                text=True,
            )
            built = True
    assert built, "neither hatchling nor uv was available to build the wheel artifact"

    wheels = list(out_dir.glob("stet_cli-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
        pack_blob = zf.read("stet/exam/data/s3-practice.jsonl.zst")
        data_manifest = zf.read("stet/exam/data/WIKIPEDIA_ATTRIBUTION.jsonl").decode("utf-8")
        lic_manifests = [zf.read(n).decode("utf-8") for n in names if ".dist-info/licenses/" in n and n.endswith("WIKIPEDIA_ATTRIBUTION.jsonl")]
        assert len(lic_manifests) == 1 and lic_manifests[0] == data_manifest

    manifest_rows = [json.loads(ln) for ln in data_manifest.splitlines() if ln.strip()]
    by_src = {r["source"]: r for r in manifest_rows}
    pack_items = unpack_items(pack_blob)
    ctx_items = [it for it in pack_items if it.kind == "context"]
    ans_items = [it for it in pack_items if it.kind == "answer"]
    assert len(ctx_items) == 221 and len(ans_items) == 287

    all_referenced: set[str] = set()
    for it in ctx_items:
        assert len(it.wrong_sources) == len(it.wrongs) and len(it.wrong_sources) > 0
        for sid in (it.source, *it.wrong_sources):
            all_referenced.add(sid)
            assert sid in by_src, f"missing attribution for {sid}"
            rec = by_src[sid]
            assert rec["title"] and rec["url"] and "oldid=" in rec["revision_url"] and "action=history" in rec["history_url"]
    for it in ans_items:
        norm = it.source.split(":", 1)[1]
        all_referenced.add(norm)
        assert norm in by_src, f"missing attribution for reasoning item {it.source}"
    assert set(by_src) == all_referenced


def test_review17_student_window_overflow_rejected_in_synthesis_and_cache():
    """Review 17 P3 #10: `parse_generated_items_detailed` and `cached_item_passes_filter` MUST
    reject any reasoning item whose `[eot] + encode(ctx) + encode(cont)` exceeds `seq_len + 1`
    tokens as `student_window_overflow`, while keeping items that fit."""
    from stet.exam.fresh import reason

    class CharEnc:
        def encode_ordinary(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    enc = CharEnc()
    short_ex = "Short excerpt about river sediment transport across coastal plains."
    long_ex = "Dense excerpt " + ("word " * 90)
    prompt = "Explain why downstream sediment deposition shifts after seasonal flooding."
    resp = "Seasonal flooding increases river velocity and carries coarser grains further downstream before slowing near the delta, where suspended particles settle and reshape alluvial channels."
    gen_txt = f"{prompt}\nRESPONSE: {resp}"

    selected = [
        {"source": "wikipedia", "id": "en:1", "lang": "en", "text": short_ex, "excerpt": short_ex, "task_type": "explain", "wrapper_id": 0},
        {"source": "wikipedia", "id": "en:2", "lang": "en", "text": long_ex, "excerpt": long_ex, "task_type": "explain", "wrapper_id": 0},
    ]
    # Window of 400 tokens fits the short item (~330 chars) and overflows the long item (~700+ chars).
    items, dropped, drop_records = reason.parse_generated_items_detailed(
        selected,
        [gen_txt, gen_txt],
        check_lang=False,
        student_window=(enc, 400),
    )
    assert len(items) == 1 and items[0].source == "explain:wikipedia:en:1"
    assert dropped == {"student_window_overflow": 1}
    assert drop_records[0]["reason"] == "student_window_overflow" and drop_records[0]["id"] == "en:2"

    # Without student_window, both parse; cached_item_passes_filter then rejects the overflowing item!
    both, _ = reason.parse_generated_items(selected, [gen_txt, gen_txt], check_lang=False)
    assert len(both) == 2
    assert reason.cached_item_passes_filter(both[0], student_window=(enc, 400)) is True
    assert reason.cached_item_passes_filter(both[1], student_window=(enc, 400)) is False


def test_review18_duplicate_items_different_shares_never_clobbers_accepted_shares(monkeypatch, tmp_path):
    """Review 18 P2 #4: staging MUST key by `(items_digest, shares_digest)`, a duplicate post
    (`posted=False` or HTTP 409) MUST NOT overwrite accepted `KEPT_LANGUAGES`, and
    `stages.whole_exam()` MUST honor authoritative `server_shares` and refuse ambiguous
    multi-share candidate dirs when server shares are omitted."""
    import base64

    from stet.exam import examine, stages
    from stet.exam.farm import pack_items
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    rnd = official_round(
        s3,
        19,
        {"round": 123457, "randomness": "cd" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "ef" * 32, "n_shards": 100000},
    )

    items = [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source="explain:wikipedia:en:100", ctx="Ctx A", cont=" Ans A")]
    digest = data.exam_digest(items)
    blob = base64.b64encode(pack_items(items)).decode()
    exam_first = {"round": rnd.id, "items_digest": digest, "items_file": blob, "languages": {"en": 60.0, "fr": 40.0}, "table": [{"id": items[0].id}]}
    exam_retry = {"round": rnd.id, "items_digest": digest, "items_file": blob, "languages": {"en": 55.0, "fr": 45.0}, "table": [{"id": items[0].id}]}

    work = tmp_path / "examine" / rnd.id
    built = iter([exam_first, exam_retry])
    monkeypatch.setattr(examine, "build", lambda *a, **kw: next(built))
    posts = iter([{"ok": True, "posted": True}, {"ok": True, "posted": False}])
    monkeypatch.setattr(examine, "post", lambda *a, **kw: next(posts))

    examine.examine(rnd, scale, [], work, key=None, log=lambda *a: None)
    examine.examine(rnd, scale, [], work, key=None, log=lambda *a: None)

    # Distinct candidate directories for the two share maps, while KEPT_LANGUAGES retains winner (60/40).
    d1 = examine.candidate_dir(work, digest, examine.shares_digest(exam_first["languages"]))
    d2 = examine.candidate_dir(work, digest, examine.shares_digest(exam_retry["languages"]))
    assert d1 != d2 and d1.is_dir() and d2.is_dir()
    assert json.loads((work / examine.KEPT_LANGUAGES).read_text()) == {"en": 60.0, "fr": 40.0}

    rnd_settled = dataclasses.replace(rnd, exam_digest=digest)
    got_items, got_shares = stages.whole_exam(rnd_settled, tmp_path / "w1")
    assert got_items == items and got_shares == {"en": 60.0, "fr": 40.0}

    # If KEPT_ITEMS is removed while two conflicting share candidates exist, whole_exam MUST
    # fail cleanly unless authoritative server_shares are passed.
    (work / examine.KEPT_ITEMS).unlink()
    with pytest.raises(stages.Stop, match="multiple candidate language shares"):
        stages.whole_exam(rnd_settled, tmp_path / "w2")

    got_items2, got_shares2 = stages.whole_exam(rnd_settled, tmp_path / "w3", server_shares={"en": 60, "fr": 40})
    assert got_items2 == items and got_shares2 == {"en": 60, "fr": 40}
    assert examine.shares_digest({"en": 60, "fr": 40}) == examine.shares_digest({"en": 60.0, "fr": 40.0})

    # Passing --shares alone (without --items) also resolves the ambiguous multi-share directory!
    shares_only = tmp_path / "explicit_shares.json"
    shares_only.write_text(json.dumps({"en": 60.0, "fr": 40.0}))
    got_items3, got_shares3 = stages.whole_exam(rnd_settled, tmp_path / "w4", shares_file=shares_only)
    assert got_items3 == items and got_shares3 == {"en": 60.0, "fr": 40.0}


def test_review18_official_tokenizer_failure_raises_retryable_error(monkeypatch, tmp_path):
    """Review 18 P2 #5: on official builds for a season with a pinned student tokenizer, failing to
    load that tokenizer MUST raise a retryable `RuntimeError` (not `FatalExamError` and not
    silently setting `student_window=None`) so overflowing items can never slip into the exam."""
    from stet import tok
    from stet.exam import examine
    from stet.exam.fresh import langs, reason
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    rnd = official_round(
        s3,
        20,
        {"round": 123458, "randomness": "ef" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "11" * 32, "n_shards": 100000},
    )

    monkeypatch.setattr(tok, "load_encoding", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tokenizer checksum mismatch")))
    monkeypatch.setattr(data, "corpus", lambda *a, **k: types.SimpleNamespace(total_bytes=0, meta={}, __len__=lambda self: 0))
    monkeypatch.setattr(langs, "shares", lambda *a, **k: {"en": 100.0})

    snap = tmp_path / "1700000000.jsonl"
    snap.write_text(
        json.dumps(
            {
                "source": "wikipedia",
                "id": "en:99",
                "lang": "en",
                "title": "Test",
                "url": "https://en.wikipedia.org/wiki/Test",
                "license": "CC BY-SA 4.0",
                "published": 1700000000,
                "text": "Enough prose to qualify as a document. " * 40,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="pinned student tokenizer") as exc_info:
        examine.assemble(rnd, scale, [(1700000000, snap)], tmp_path / "work", log=lambda *a: None)
    assert not isinstance(exc_info.value, reason.FatalExamError)


def test_review18_doctor_and_verify_agree_on_vram_and_native_bf16(monkeypatch, tmp_path, capsys):
    """Review 18 P2 #6: `stet doctor` and `stet verify` MUST share `grade_eligibility()` so both
    reject under-VRAM GPUs (e.g. 8 GB RTX 3070) and pre-Ampere GPUs lacking native bf16 (e.g.
    16 GB Tesla T4 SM 7.5), while accepting 12 GB+ Ampere/Ada/Blackwell GPUs."""
    from stet import bench, cli, client, env, runner

    s3 = load_season("s3")
    cases = [
        (
            {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 3070", "vram_gb": 8.0, "cuda_capability": "8.6", "bf16_native": True},
            False,
            "11 GB",
        ),
        (
            {"device": "cuda", "backend": "cuda", "gpu": "Tesla T4", "vram_gb": 16.0, "cuda_capability": "7.5", "bf16_native": False},
            False,
            "native bf16",
        ),
        (
            {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA Tesla T4", "vram_gb": 16.0},
            False,
            "native bf16",
        ),
        (
            {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "cuda_capability": "8.6", "bf16_native": True},
            True,
            "",
        ),
        (
            {"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0, "cuda_capability": "8.9", "bf16_native": True},
            True,
            "",
        ),
    ]
    for hw, want_ok, want_sub in cases:
        ok, why = client.grade_eligibility(s3, hw)
        assert ok is want_ok, f"grade_eligibility({hw}) -> ({ok}, {why!r}), want {want_ok}"
        if want_sub:
            assert want_sub in why, f"expected {want_sub!r} in {why!r}"

    # Verify cmd_doctor and _verify both reject 8 GB RTX 3070 and 16 GB Tesla T4.
    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    monkeypatch.delenv("STET_DEV", raising=False)
    for bad_hw, expect_msg in [
        ({"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 3070", "vram_gb": 8.0, "cuda_capability": "8.6", "bf16_native": True}, "11 GB"),
        ({"device": "cuda", "backend": "cuda", "gpu": "Tesla T4", "vram_gb": 16.0, "cuda_capability": "7.5", "bf16_native": False}, "native bf16"),
    ]:
        # _verify path
        fake_train = types.SimpleNamespace(
            pick_device=lambda d=None: types.SimpleNamespace(type="cuda"),
            hardware_info=lambda d, hw=bad_hw: hw,
        )
        monkeypatch.setitem(sys.modules, "stet.train", fake_train)
        monkeypatch.setattr(client, "need_key", lambda: types.SimpleNamespace(public_hex="0" * 64))
        monkeypatch.setattr(client, "request", lambda method, path, *a, **kw: {"season": "s3", "scale": "standard"})
        capsys.readouterr()
        args = types.SimpleNamespace(season="s3", device="cuda", any_device=False, upgrade=False)
        with pytest.raises(SystemExit) as se:
            client._verify(args)
        assert se.value.code == 1
        assert expect_msg in capsys.readouterr().err

        # cmd_doctor path: mock minimal torch + train so doctor runs cleanly even without Torch installed
        fake_torch = types.SimpleNamespace(
            __version__="2.9.1+cu128",
            cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i, g=bad_hw["gpu"]: g),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr("stet.spec.load_season", lambda name=None: s3)
        monkeypatch.setattr(runner, "bwrap_available", lambda: (True, "/usr/bin/bwrap"))
        monkeypatch.setattr(runner, "scope_available", lambda: (True, ""))
        monkeypatch.setattr(env, "problems", lambda *a, **k: [])
        monkeypatch.setattr(env, "summary", lambda *a, **k: "ok")
        monkeypatch.setattr(bench, "load_cached", lambda *a, **k: None, raising=False)
        capsys.readouterr()
        with pytest.raises(SystemExit) as doc_se:
            cli.cmd_doctor(types.SimpleNamespace(json=False, quiet=False))
        assert doc_se.value.code == 0
        cap = capsys.readouterr()
        doc_out = cap.out + cap.err
        assert "this machine can grade for the leaderboard" not in doc_out
        assert expect_msg in doc_out


def test_review18_synthesis_rejects_interior_question_in_parser_and_cache():
    """Review 18 P3 #8: synthesis prompts containing an interior question mark (e.g.
    'What changed? Describe the consequences.') MUST be dropped as `synthesis_question` in
    `parse_generated_items_detailed` and rejected by `cached_item_passes_filter`, while a pure
    trailing '?' normalizes cleanly without producing a double period."""
    from stet.exam.fresh import reason

    assert reason.REASONING_FILTER_VERSION >= 4
    ex = "Coastal marsh restoration alters tidal drainage channels and modifies local carbon burial rates over multi-decade intervals."
    resp = "Restored tidal drainage reconnects marsh flats to marine sediment supply, accelerating vertical accretion and stabilizing buried organic carbon against erosion."
    selected = [
        {"source": "wikipedia", "id": "en:10", "lang": "en", "text": ex, "excerpt": ex, "task_type": "synthesis", "wrapper_id": 2},
        {"source": "wikipedia", "id": "es:13", "lang": "es", "text": ex, "excerpt": ex, "task_type": "synthesis", "wrapper_id": 2},
        {"source": "wikipedia", "id": "en:11", "lang": "en", "text": ex, "excerpt": ex, "task_type": "synthesis", "wrapper_id": 2},
        {"source": "wikipedia", "id": "en:12", "lang": "en", "text": ex, "excerpt": ex, "task_type": "synthesis", "wrapper_id": 2},
    ]
    gens = [
        f"What changed? Describe the ecological consequences.\nRESPONSE: {resp}",
        f"¿Qué explica este mecanismo de restauración?\nRESPONSE: {resp}",
        f"Summarize the primary mechanism governing marsh stability?\nRESPONSE: {resp}",
        f"Outline the long-term carbon balance trade-off.?\nRESPONSE: {resp}",
    ]
    items, dropped, records = reason.parse_generated_items_detailed(selected, gens, check_lang=False)
    assert dropped == {"synthesis_question": 2}
    assert {r["id"] for r in records} == {"en:10", "es:13"}
    assert all(r["reason"] == "synthesis_question" for r in records)
    assert len(items) == 2
    assert items[0].source == "synthesis:wikipedia:en:11"
    assert items[0].prompt == "Summarize the primary mechanism governing marsh stability."
    assert "?" not in items[0].prompt
    assert items[1].source == "synthesis:wikipedia:en:12"
    assert items[1].prompt == "Outline the long-term carbon balance trade-off."
    assert not items[1].prompt.endswith("..")

    # cached_item_passes_filter must also reject a previously cached synthesis item with an interior '?'.
    bad_cached = Item(
        kind="answer",
        corner="synthesis/en/wiki",
        fmt="prose",
        lang="en",
        source="synthesis:wikipedia:en:99",
        ctx=reason.wrap_prompt(ex, "What changed? Describe the ecological consequences.", "synthesis", 2),
        cont=" " + resp,
        prompt="What changed? Describe the ecological consequences.",
        wrapper_id=2,
    )
    assert reason.cached_item_passes_filter(bad_cached) is False
    assert reason.cached_item_passes_filter(items[0]) is True


def test_review19_stale_shares_file_rejected_against_server_shares(monkeypatch, tmp_path):
    """Review 19 P2 #5: supplying a stale `--shares` file that disagrees with authoritative server
    shares MUST raise `stages.Stop` without mutating `work/<round>/languages.json` (preventing a
    +0.080 -> -0.080 score reversal), while still allowing `--shares` fallback when server shares
    are unavailable (historical rounds)."""
    from stet.exam import aggregate, examine, farm, stages
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    items = [
        Item(kind="answer", corner=f"explain/{lang}/wiki", fmt="prose", lang=lang, source=f"explain:wikipedia:{lang}:{i}", ctx=f"Ctx {lang} {i}", cont=f" Ans {lang} {i}")
        for lang in ("en", "de")
        for i in range(24)
    ]
    digest = data.exam_digest(items)
    rnd = dataclasses.replace(
        official_round(
            s3,
            21,
            {"round": 123459, "randomness": "aa" * 32},
            {"crawl": "CC-MAIN-2026-43", "paths_sha256": "bb" * 32, "n_shards": 100000},
        ),
        exam_digest=digest,
    )
    made = tmp_path / "examine" / rnd.id
    made.mkdir(parents=True, exist_ok=True)
    (made / examine.KEPT_ITEMS).write_bytes(farm.pack_items(items))
    (made / examine.KEPT_LANGUAGES).write_text(json.dumps({"en": 90.0, "de": 10.0}))

    work = tmp_path / "audit_work"
    got_items, got_shares = stages.whole_exam(rnd, work, server_shares={"en": 90.0, "de": 10.0})
    assert got_items == items and got_shares == {"en": 90.0, "de": 10.0}

    # Construct gains (+0.10 on en, -0.10 on de) -> +0.080 under 90/10, -0.080 under 10/90!
    gains = [0.10 if it.lang == "en" else -0.10 for it in items]
    plan = aggregate.plan_of(s3)
    score_auth = aggregate.round_score(items, gains, got_shares, plan)["score"]
    assert abs(score_auth - 0.080) < 1e-6

    # Supplying a stale 10/90 --shares file against authoritative 90/10 server_shares MUST fail
    # with Stop and leave work/<round>/languages.json untouched!
    stale = tmp_path / "stale_shares.json"
    stale.write_text(json.dumps({"en": 10.0, "de": 90.0}))
    with pytest.raises(stages.Stop, match="does not match"):
        stages.whole_exam(rnd, work, shares_file=stale, server_shares={"en": 90.0, "de": 10.0})
    assert json.loads((work / rnd.id / "languages.json").read_text()) == {"en": 90.0, "de": 10.0}

    # Matching --shares (even with int vs float JSON formatting) succeeds cleanly.
    matching = tmp_path / "matching_shares.json"
    matching.write_text(json.dumps({"en": 90, "de": 10}))
    _, got_match = stages.whole_exam(rnd, work, shares_file=matching, server_shares={"en": 90.0, "de": 10.0})
    assert got_match == {"en": 90.0, "de": 10.0}

    # Non-dict JSON in --shares raises stages.Stop cleanly instead of AttributeError.
    bad_shares = tmp_path / "bad_shares.json"
    bad_shares.write_text("[]")
    with pytest.raises(stages.Stop, match="JSON object"):
        stages.whole_exam(rnd, work, shares_file=bad_shares, server_shares={"en": 90.0, "de": 10.0})

    # Historical fallback (server_shares=None and no cached round_shares) still accepts explicit --shares.
    work_hist = tmp_path / "hist_work"
    _, got_hist = stages.whole_exam(rnd, work_hist, shares_file=stale, server_shares=None)
    assert got_hist == {"en": 10.0, "de": 90.0}


def test_review19_python_hardware_eligibility_matches_shared_season3_vectors():
    """Review 19 P2 #6: Python `client.grade_eligibility()` MUST agree with Go `gradesOnHardware()`
    on every vector in `server/testdata/grade_hardware_vectors_s3.json`."""
    from stet import client

    s3 = load_season("s3")
    vec_path = Path(__file__).resolve().parents[1] / "server" / "testdata" / "grade_hardware_vectors_s3.json"
    vectors = json.loads(vec_path.read_text())
    assert len(vectors) >= 10
    for vec in vectors:
        ok, why = client.grade_eligibility(s3, vec["hardware"])
        assert ok is vec["eligible"], f"vector {vec['name']} ({vec['hardware']}): got ({ok}, {why!r}), want {vec['eligible']}"


def test_review19_shared_baseline_retained_across_cache_error_and_recovered_once(monkeypatch, tmp_path):
    """Review 19 P2 #7:
    (a) When `sit_baseline` scores successfully and ONLY writing the baseline cache fails, the
        valid in-memory baseline is retained for job 1 and job 2, and persisted once storage recovers.
    (b) When the first baseline `sit_exam` fails once (clearing `entry.model = None` in `finally`),
        the second already-trained job recovers the baseline ONCE via `pipeline.baseline` instead
        of failing with 'no longer in memory', and subsequent jobs reuse that recovered baseline."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import os

    from stet import client, grade, pipeline
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    items = [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source="explain:wikipedia:en:1", ctx="Ctx", cont=" Ans")]
    exam_sha = data.exam_digest(items)
    rnd = dataclasses.replace(
        official_round(
            s3,
            22,
            {"round": 123460, "randomness": "cc" * 32},
            {"crawl": "CC-MAIN-2026-43", "paths_sha256": "dd" * 32, "n_shards": 100000},
        ),
        exam_digest=exam_sha,
    )
    key = types.SimpleNamespace(public_hex="a" * 64)
    dev = types.SimpleNamespace(type="cpu")

    monkeypatch.setattr(client, "fetch_exam", lambda *a, **k: None)
    monkeypatch.setattr(data, "round_exam", lambda *a, **k: items)

    def make_trained(tag: str):
        return grade.Trained(
            model=types.SimpleNamespace(tag=tag),
            enc=None,
            device=dev,
            result={"valid": False, "hardware": {"device": "cpu"}, "train": {"init_sha256": "1" * 64}},
            trained=True,
        )

    def fake_sit_exam(trained, r, sc, ex):
        try:
            if getattr(trained.model, "fail_once", False):
                trained.model.fail_once = False
                raise RuntimeError("transient baseline sit error")
            if trained.model is None:
                raise RuntimeError("model already cleared")
            res = dict(trained.result)
            res["valid"] = True
            res["exam_sha256"] = exam_sha
            res["exam"] = {"items": len(ex), "item_scores": [-2.0], "mean": -2.0}
            return res
        finally:
            trained.model = None

    monkeypatch.setattr(pipeline, "sit_exam", fake_sit_exam)
    monkeypatch.setattr(grade, "sit_exam", fake_sit_exam)

    # (a) Cache write fails during job 1 after baseline sit_exam succeeds -> _sit retains the valid
    #     in-memory measurement so BOTH job 1 and job 2 succeed without discarding trained models,
    #     and persists the cache file once storage recovers!
    orig_replace = os.replace
    fail_disk = {"on": True}

    def flaky_replace(src, dst):
        if fail_disk["on"]:
            raise OSError("No space left on device")
        return orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    bases_a = {rnd.id: make_trained("base-a")}
    p1 = client.Pending({"id": 1, "baseline": True}, rnd, make_trained("cand-1"), dev)
    p2 = client.Pending({"id": 2, "baseline": True}, rnd, make_trained("cand-2"), dev)

    res1, base1 = client._sit(p1, bases_a, key, scale)
    assert res1["valid"] and base1["valid"] and base1["exam_sha256"] == exam_sha
    assert isinstance(bases_a[rnd.id], client._SatBaseline) and bases_a[rnd.id].cached is False

    # Storage recovers before job 2 -> job 2 reuses retained in-memory baseline AND persists cache file!
    fail_disk["on"] = False
    res2, base2 = client._sit(p2, bases_a, key, scale)
    assert res2["valid"] and base2 == base1
    assert bases_a[rnd.id].cached is True
    bp = pipeline.baseline_path(rnd, scale, dev, exam_sha=exam_sha, verifier=key.public_hex, gpu="")
    assert bp.exists()

    # (b) First baseline sit_exam raises once (clearing entry.model = None in finally):
    #     job 2 recovers baseline ONCE via pipeline.baseline — EVEN IF disk cache write fails
    #     during recovery — and job 3 reuses that recovered _SatBaseline without poisoning bases!
    bp.unlink(missing_ok=True)
    recovered_trains = []
    monkeypatch.setattr(pipeline, "train_baseline", lambda *a, **k: recovered_trains.append(rnd.id) or make_trained("recovered-base"))
    bad_base = make_trained("base-b")
    bad_base.model.fail_once = True
    bases_b = {rnd.id: bad_base}
    q1 = client.Pending({"id": 10, "baseline": True}, rnd, make_trained("cand-10"), dev)
    q2 = client.Pending({"id": 11, "baseline": True}, rnd, make_trained("cand-11"), dev)
    q3 = client.Pending({"id": 12, "baseline": True}, rnd, make_trained("cand-12"), dev)

    with pytest.raises(RuntimeError, match="transient baseline sit error"):
        client._sit(q1, bases_b, key, scale)
    assert bad_base.model is None  # cleared by sit_exam's finally
    assert q1.trained.model is not None  # q1's own student untouched

    # Make disk full DURING recovery too -> recovery MUST still succeed in memory and NOT poison bases_b!
    fail_disk["on"] = True
    r2, b2 = client._sit(q2, bases_b, key, scale)
    assert r2["valid"] and b2["valid"] and b2["exam_sha256"] == exam_sha
    assert recovered_trains == [rnd.id]
    assert isinstance(bases_b[rnd.id], client._SatBaseline) and bases_b[rnd.id].cached is False

    # Storage recovers before job 3 -> job 3 reuses recovered _SatBaseline AND persists cache!
    fail_disk["on"] = False
    r3, b3 = client._sit(q3, bases_b, key, scale)
    assert r3["valid"] and b3 == b2
    assert recovered_trains == [rnd.id]
    assert bases_b[rnd.id].cached is True and bp.exists()

    # (c) If the single recovery attempt itself fails during training/sitting, a subsequent job does NOT retry recovery infinitely!
    bp.unlink(missing_ok=True)
    fail_rec_calls = []

    def failing_recover(r, sc, d, **kw):
        fail_rec_calls.append(r.id)
        raise RuntimeError("recovery failed")

    monkeypatch.setattr(pipeline, "baseline", failing_recover)
    bad_base2 = make_trained("base-c")
    bad_base2.model.fail_once = True
    bases_c = {rnd.id: bad_base2}
    with pytest.raises(RuntimeError, match="transient baseline sit error"):
        client._sit(client.Pending({"id": 20, "baseline": True}, rnd, make_trained("c20"), dev), bases_c, key, scale)
    with pytest.raises(RuntimeError, match="recovery failed"):
        client._sit(client.Pending({"id": 21, "baseline": True}, rnd, make_trained("c21"), dev), bases_c, key, scale)
    with pytest.raises(RuntimeError, match="no longer in memory"):
        client._sit(client.Pending({"id": 22, "baseline": True}, rnd, make_trained("c22"), dev), bases_c, key, scale)
    assert fail_rec_calls == [rnd.id]


def test_review20_reasoning_cache_shares_language_and_quality_checks_and_requires_langid(monkeypatch, tmp_path):
    """Review 20 P2 #6:
    (a) Language detection (`py3langid`) is a base dependency available for official synthesis.
    (b) Fresh parser and cached-item revalidator share the exact same per-item checks, so a
        French prompt paired with an English answer (or truncated/meta-chatter row) is rejected
        equally on fresh synthesis and on cached reload during a restarted build.
    (c) Official synthesis and official cache validation fail closed if `py3langid` is missing."""
    import zstandard as zstd

    from stet.exam import examine
    from stet.exam.farm import pack_items
    from stet.exam.fresh import Doc, reason, write_snapshot
    from stet.spec import official_round

    # Base packaging includes py3langid so lean examiner installs always have the language checker.
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '"py3langid>=0.4"' in pyproject
    assert reason._langid is not None

    s3 = load_season("s3")
    scale = s3.scale("nano")
    rnd = official_round(
        s3,
        25,
        {"round": 123465, "randomness": "ee" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "ff" * 32, "n_shards": 100000},
    )

    fr_excerpt = (
        "Les ingénieurs portuaires de Marseille ont modernisé les écluses du bassin nord afin de "
        "réduire les retards lors des grandes marées et de limiter la consommation électrique "
        "des pompes hydrauliques pendant les heures de pointe nocturnes."
    )
    fr_prompt = "Pourquoi la modernisation des écluses réduit-elle les retards maritimes ?"
    en_wrong_resp = (
        "Modernizing the lock gates speeds up vessel transit during extreme tides while lowering "
        "nighttime hydraulic pump electricity demand across the northern port basin."
    )
    fr_good_resp = (
        "La rénovation des vannes accélère le passage des navires lors des fortes marées tout en "
        "diminuant la puissance absorbée par les stations de pompage durant la nuit."
    )

    doc_sel = [
        {
            "source": "ccnews",
            "id": "fr-1",
            "lang": "fr",
            "text": fr_excerpt,
            "excerpt": fr_excerpt,
            "task_type": "explain",
            "wrapper_id": 0,
        }
    ]
    bad_gen = [f"{fr_prompt}\nRESPONSE: {en_wrong_resp}"]
    good_gen = [f"{fr_prompt}\nRESPONSE: {fr_good_resp}"]

    # 1. Fresh parser rejects French prompt + English answer as wrong_lang, and keeps valid French.
    kept_bad, drops_bad, _ = reason.parse_generated_items_detailed(doc_sel, bad_gen, require_langid=True)
    assert kept_bad == [] and drops_bad.get("wrong_lang") == 1
    kept_good, drops_good, _ = reason.parse_generated_items_detailed(doc_sel, good_gen, require_langid=True)
    assert len(kept_good) == 1 and not drops_good

    # 2. Construct a 20-row poisoned cached fixture (as if built earlier without language detector):
    #    cached_item_passes_filter and _reasoning_cache_usable MUST reject it!
    poisoned_rows = [
        Item(
            kind="answer",
            corner="explain/fr/ccnews",
            fmt="prose",
            lang="fr",
            source=f"explain:ccnews:fr-bad-{i}",
            ctx=reason.wrap_prompt(fr_excerpt, fr_prompt, "explain", i % 6),
            cont=f" {en_wrong_resp} Note {i}.",
            prompt=fr_prompt,
            wrapper_id=i % 6,
        )
        for i in range(25)
    ]
    good_rows = [
        Item(
            kind="answer",
            corner="explain/fr/ccnews",
            fmt="prose",
            lang="fr",
            source=f"explain:ccnews:fr-good-{i}",
            ctx=reason.wrap_prompt(fr_excerpt, fr_prompt, "explain", i % 6),
            cont=f" {fr_good_resp} Détail {i}.",
            prompt=fr_prompt,
            wrapper_id=i % 6,
        )
        for i in range(25)
    ]
    assert reason.cached_item_passes_filter(poisoned_rows[0], require_langid=True) is False
    assert reason.cached_item_passes_filter(good_rows[0], require_langid=True) is True
    assert examine._reasoning_cache_usable(poisoned_rows, s3, require_langid=True) is False
    assert examine._reasoning_cache_usable(good_rows, s3, require_langid=True) is True

    # Shared quality checks (truncated sentence, meta-chatter) are also enforced on cached rows.
    truncated_item = dataclasses.replace(good_rows[0], cont=f" {fr_good_resp.rstrip('.')}")
    chatter_item = dataclasses.replace(good_rows[0], cont=f" Here is the answer: {fr_good_resp}")
    assert reason.cached_item_passes_filter(truncated_item, require_langid=True) is False
    assert reason.cached_item_passes_filter(chatter_item, require_langid=True) is False

    # 3. Restarted official assemble() rejects the poisoned cache and regenerates cleanly.
    snap_path = tmp_path / "2026-11-27-00.jsonl.zst"
    write_snapshot(
        [Doc(source="ccnews", id="d1", url="https://example.invalid/fr", lang="fr", license="ccnews", published=1_800_000_000, title="T", text=fr_excerpt)],
        snap_path,
        {"slot": "2026-11-27-00"},
    )
    shas = [hashlib.sha256(snap_path.read_bytes()).hexdigest()]
    r_cfg = s3.raw["exam"]["reasoning"]
    rcfg_sig = hashlib.sha256(
        json.dumps({"filter_version": reason.REASONING_FILTER_VERSION, "cfg": r_cfg}, sort_keys=True).encode()
    ).hexdigest()[:8]
    work = tmp_path / "work"
    work.mkdir()
    rcache = work / f"reasoning-{examine.snapshot_name(shas)[:12]}-{rnd.seed.hex()[:8]}-{rcfg_sig}.jsonl.zst"
    rcache.write_bytes(pack_items(poisoned_rows))

    (tmp_path / "empty.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(b""))
    monkeypatch.setattr("stet.data.corpus", lambda *a, **k: tmp_path / "empty.jsonl.zst")
    monkeypatch.setattr("stet.exam.decontam.seen", lambda *a, **k: set())
    monkeypatch.setattr("stet.exam.fresh.select.select_context", lambda *a, **k: [])
    monkeypatch.setattr(examine, "_student_window", lambda *a, **k: None)

    synth_calls: list[dict] = []

    def fake_synth(docs, seed, cfg, **kw):
        synth_calls.append(kw)
        return list(good_rows)

    monkeypatch.setattr("stet.exam.fresh.reason.synthesize_reasoning", fake_synth)
    items, _, _, _ = examine.assemble(rnd, scale, [(0, snap_path)], work, measure=False, synthesize=True, check_coverage=False, log=lambda _: None)
    assert len(synth_calls) == 1 and synth_calls[0].get("require_langid") is True
    assert all(it.source.startswith("explain:ccnews:fr-good-") for it in items)

    # 4. If py3langid is unavailable, official synthesis, cache validation, and assemble() MUST fail closed.
    monkeypatch.setattr(reason, "_langid", None)
    with pytest.raises(RuntimeError, match="py3langid"):
        reason.parse_generated_items_detailed(doc_sel, bad_gen, require_langid=True)
    with pytest.raises(RuntimeError, match="py3langid"):
        reason.cached_item_passes_filter(good_rows[0], require_langid=True)
    with pytest.raises(RuntimeError, match="py3langid"):
        examine._reasoning_cache_usable(good_rows, s3, require_langid=True)
    with pytest.raises(RuntimeError, match="py3langid"):
        examine.assemble(rnd, scale, [(0, snap_path)], work, measure=False, synthesize=True, check_coverage=False, log=lambda _: None)


def test_review20_posted_exam_grade_job_retains_baseline_across_cache_write_error(monkeypatch, tmp_path):
    """Review 20 P2 #7: when the exam is ALREADY posted and ONLY the baseline cache rename fails
    with OSError, `_grade_job` MUST use the valid in-memory baseline for the current job, proceed
    to script download & grading, reuse that in-memory baseline on subsequent jobs without
    retraining, and persist the cache opportunistically once storage recovers."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import os
    import time

    import zstandard as zstd

    from stet import cli, client, grade, pipeline
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    items = [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source="explain:wikipedia:en:1", ctx="Ctx", cont=" Ans")]
    exam_sha = data.exam_digest(items)
    rnd = dataclasses.replace(
        official_round(
            s3,
            26,
            {"round": 123466, "randomness": "11" * 32},
            {"crawl": "CC-MAIN-2026-43", "paths_sha256": "22" * 32, "n_shards": 100000},
        ),
        exam_digest=exam_sha,
    )
    key = types.SimpleNamespace(public_hex="b" * 64, handle="tester")
    dev = types.SimpleNamespace(type="cpu")
    out = cli.Out(quiet=True)

    script_bytes = b'{"doc": 0}\n'
    script_sha = hashlib.sha256(script_bytes).hexdigest()
    compressed = zstd.ZstdCompressor().compress(script_bytes)

    train_calls: list[str] = []
    download_calls: list[str] = []
    grade_calls: list[bytes] = []

    monkeypatch.setattr(client, "_job_round", lambda season, job, out_: rnd)
    monkeypatch.setattr(client, "fetch_exam", lambda *a, **k: None)
    monkeypatch.setattr(data, "round_exam", lambda *a, **k: items)

    def fake_request(method, path, **kw):
        download_calls.append(path)
        return compressed

    def fake_train_baseline(r, sc, d, **kw):
        train_calls.append(r.id)
        return grade.Trained(
            model=types.SimpleNamespace(tag="base"),
            enc=None,
            device=dev,
            result={"valid": False, "hardware": {"device": "cpu"}, "train": {"init_sha256": "3" * 64}},
            trained=True,
        )

    def fake_sit_exam(trained, r, sc, ex):
        try:
            res = dict(trained.result)
            res["valid"] = True
            res["exam_sha256"] = exam_sha
            res["exam"] = {"items": len(ex), "item_scores": [-2.1], "mean": -2.1}
            return res
        finally:
            trained.model = None

    def fake_grade_round(r, sc, path, **kw):
        grade_calls.append(path.read_bytes())
        return {"valid": True, "exam_sha256": exam_sha, "exam": {"items": 1, "item_scores": [-1.9], "mean": -1.9}}

    monkeypatch.setattr(client, "request", fake_request)
    monkeypatch.setattr(pipeline, "train_baseline", fake_train_baseline)
    monkeypatch.setattr(pipeline, "sit_exam", fake_sit_exam)
    monkeypatch.setattr(pipeline, "grade_round", fake_grade_round)

    # Fail ONLY baseline cache rename (work dir remains writable).
    orig_replace = os.replace
    fail_base_cache = {"on": True}

    def flaky_replace(src, dst):
        if fail_base_cache["on"] and "baseline-" in str(dst):
            raise OSError("Read-only baseline cache mount")
        return orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    bases: dict = {}
    job1 = {"id": 101, "baseline": True, "script": {"sha256": script_sha}}
    job2 = {"id": 102, "baseline": True, "script": {"sha256": script_sha}}
    job3 = {"id": 103, "baseline": True, "script": {"sha256": script_sha}}

    # Job 1: baseline cache rename fails -> MUST NOT abort; completes script download + grading!
    res1, base1 = client._grade_job(job1, s3, scale, dev, key, out, tmp_path / "work1", bases=bases)
    assert res1["valid"] is True and base1["valid"] is True and base1["exam_sha256"] == exam_sha
    assert len(train_calls) == 1
    assert len(download_calls) == 1 and len(grade_calls) == 1
    assert isinstance(bases[rnd.id], client._SatBaseline) and bases[rnd.id].cached is False

    # Job 2 while cache rename STILL fails: reuses in-memory baseline (zero retrain!), finishes grading!
    res2, base2 = client._grade_job(job2, s3, scale, dev, key, out, tmp_path / "work2", bases=bases)
    assert res2["valid"] is True and base2 == base1
    assert len(train_calls) == 1
    assert len(download_calls) == 2 and len(grade_calls) == 2
    assert bases[rnd.id].cached is False

    # Job 3 once storage recovers: still zero retrain, AND opportunistically persists baseline cache!
    fail_base_cache["on"] = False
    res3, base3 = client._grade_job(job3, s3, scale, dev, key, out, tmp_path / "work3", bases=bases)
    assert res3["valid"] is True and base3 == base1
    assert len(train_calls) == 1
    assert len(download_calls) == 3 and len(grade_calls) == 3
    assert bases[rnd.id].cached is True
    bp = pipeline.baseline_path(rnd, scale, dev, exam_sha=exam_sha, verifier=key.public_hex, gpu="")
    assert bp.exists()

    # Standalone _grade_job(..., bases=None) also survives a baseline cache rename OSError!
    bp.unlink()
    fail_base_cache["on"] = True
    res4, base4 = client._grade_job(job1, s3, scale, dev, key, out, tmp_path / "work4", bases=None)
    assert res4["valid"] is True and base4["valid"] is True
    assert len(train_calls) == 2 and len(download_calls) == 4 and len(grade_calls) == 4

    # Gate 4 P1-1: verify HTTP 409 refusal in `_verify()` evicts the in-memory `_SatBaseline` in
    # `bases[round_id]` alongside `_forget_round` so the retry re-measures a fresh baseline instead
    # of immediately re-submitting the refused in-memory baseline!
    refusal_attempts = []
    submitted_baselines = []
    forgotten = []
    verify_jobs = [
        {"id": 201, "kind": "grade", "scale": "nano", "baseline": True, "expires_at": int(time.time()) + 3600,
         "round": rnd.describe(), "editor": {"id": "ed1", "name": "e", "version": 1, "owner": "a"},
         "script": {"sha256": script_sha, "size": len(script_bytes)}},
        {"id": 202, "kind": "grade", "scale": "nano", "baseline": True, "expires_at": int(time.time()) + 3600,
         "round": rnd.describe(), "editor": {"id": "ed1", "name": "e", "version": 1, "owner": "a"},
         "script": {"sha256": script_sha, "size": len(script_bytes)}},
    ]
    baseline_ver = {"n": 0}

    def fake_train_for_409(r, sc, d, **kw):
        baseline_ver["n"] += 1
        return grade.Trained(
            model=types.SimpleNamespace(tag=f"v{baseline_ver['n']}"),
            enc=None,
            device=dev,
            result={"valid": False, "hardware": {"device": "cpu"}, "train": {"init_sha256": f"{baseline_ver['n']}" * 64}},
            trained=True,
        )

    def fake_verify_req(method, path, body=None, **kw):
        if path == "/api/v1/season":
            return {"scale": "nano"}
        if path == "/api/v1/me":
            return {"account": {"handle": "tester"}, "verifier": {"validated": 0}}
        if path == "/api/v1/jobs/claim":
            return {"job": verify_jobs.pop(0)} if verify_jobs else {"job": None}
        if path.startswith("/api/v1/scripts/"):
            return compressed
        if path.endswith("/release"):
            return {}
        return {}

    def fake_report(path, payload, k, **kw):
        submitted_baselines.append(payload.get("baseline", {}).get("init_sha256"))
        if len(refusal_attempts) == 0:
            refusal_attempts.append(path)
            raise client.ApiError(409, "your unedited baseline differs from the round's")
        return {"status": "pending"}

    monkeypatch.setattr(pipeline, "train_baseline", fake_train_for_409)
    monkeypatch.setattr(client, "request", fake_verify_req)
    monkeypatch.setattr(client, "report", fake_report)
    monkeypatch.setattr(client, "_forget_round", lambda *a, **k: forgotten.append(a[0]["round"]["id"]))
    monkeypatch.setattr(client, "slim_result", lambda r, b: {"valid": True, "baseline": {"init_sha256": b["train"]["init_sha256"]}})
    monkeypatch.setattr(client, "need_key", lambda: key)
    monkeypatch.setattr(client, "note_pass", lambda *a, **k: None)
    monkeypatch.setattr(client, "saved_pass", lambda *a, **k: None)
    monkeypatch.setattr(client, "remember_announcement", lambda *a, **k: None)
    monkeypatch.setattr(client, "tidy", lambda *a, **k: None)
    monkeypatch.setattr(client, "config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(client.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr("stet.env.problems", lambda *a, **k: [])
    monkeypatch.setattr("stet.train.pick_device", lambda *a, **k: dev)
    monkeypatch.setattr("stet.train.hardware_info", lambda *a, **k: {"device": "cpu", "gpu": ""})
    monkeypatch.setattr("stet.upgrade.Upgrader", lambda *a, **k: types.SimpleNamespace(on=False, consider=lambda *a, **k: None, look=lambda: None, now=lambda: None))
    args = types.SimpleNamespace(once=False, max_jobs=1, edits=False, device="cpu", season="s3", no_sandbox=False, any_device=True, exit_when_idle=0, no_upgrade=True)
    client._verify(args)
    assert forgotten == [rnd.id]
    assert submitted_baselines == ["1" * 64, "2" * 64], f"409 refusal MUST evict in-memory _SatBaseline and re-measure! got {submitted_baselines}"


def test_review21_official_langid_runtime_error_fails_closed(monkeypatch, tmp_path):
    """Review 21 P2 #6: when py3langid IS installed but classify() raises an I/O / runtime error
    (or returns an empty detection), official reasoning parsing, cache validation, and assemble()
    MUST propagate RuntimeError instead of silently accepting wrong-language items or deleting a
    healthy cache file."""
    import hashlib
    import json

    from stet.exam import examine, farm
    from stet.exam.fresh import Doc, write_snapshot
    from stet.exam.fresh import reason as reason_mod
    from stet.spec import official_round

    assert reason_mod._langid is not None, "py3langid must be installed in base dependencies"

    fr_excerpt = (
        "Les ingénieurs ferroviaires de Lyon ont installé de nouveaux capteurs acoustiques "
        "le long des aiguillages afin de détecter les microfissures thermiques avant les "
        "périodes de grand froid hivernal et réduire les interruptions de service."
    )
    fr_prompt = "Pourquoi les capteurs acoustiques ont-ils été installés sur les aiguillages lyonnais ?"
    en_wrong_answer = (
        "Engineers mounted acoustic sensors along rail switches to catch thermal cracks before "
        "severe winter cold arrives, reducing unexpected service disruptions across the network."
    )
    doc_sel = [
        {
            "source": "ccnews",
            "id": f"fr-io-{i}",
            "lang": "fr",
            "text": fr_excerpt,
            "excerpt": fr_excerpt,
            "task_type": "explain",
            "wrapper_id": i % 6,
        }
        for i in range(20)
    ]
    raw_gens = [f"{fr_prompt}\nRESPONSE: {en_wrong_answer}" for _ in range(20)]

    # 1. Healthy detector rejects all 20 French-prompt/English-answer items as wrong_lang.
    items_ok, dropped_ok, _ = reason_mod.parse_generated_items_detailed(
        doc_sel, raw_gens, require_langid=True
    )
    assert items_ok == [] and dropped_ok.get("wrong_lang") == 20

    # Build a cached batch of those 20 bad items (bypassing check_lang during fixture setup).
    bad_cached, _ = reason_mod.parse_generated_items(doc_sel, raw_gens, check_lang=False)
    assert len(bad_cached) == 20

    # 2. Simulate installed classifier raising OSError on classify()!
    broken_langid = types.SimpleNamespace(
        rank=reason_mod._langid.rank,
        classify=lambda text: (_ for _ in ()).throw(OSError("model file I/O error")),
    )
    monkeypatch.setattr(reason_mod, "_langid", broken_langid)

    with pytest.raises(RuntimeError, match="py3langid language classification failed"):
        reason_mod._ensure_langid_available()
    with pytest.raises(RuntimeError, match="py3langid language classification failed"):
        reason_mod.parse_generated_items_detailed(doc_sel, raw_gens, require_langid=True)
    with pytest.raises(RuntimeError, match="py3langid language classification failed"):
        reason_mod.cached_item_passes_filter(bad_cached[0], require_langid=True)

    s3 = load_season("s3")
    with pytest.raises(RuntimeError, match="py3langid language classification failed"):
        examine._reasoning_cache_usable(bad_cached, s3, require_langid=True)

    # 3. Official assemble() must propagate RuntimeError AND preserve the cache file on disk
    # (not treat classify()'s OSError as a corrupt zstd cache and unlink it!).
    snap_path = tmp_path / "2026-11-27-00.jsonl.zst"
    snap_docs = [
        Doc(source="ccnews", id=f"fr-io-{i}", url="https://example.invalid/fr", lang="fr", license="ccnews", published=1_800_000_000, title=f"T{i}", text=fr_excerpt)
        for i in range(20)
    ]
    write_snapshot(snap_docs, snap_path, {"slot": "2026-11-27-00"})
    rnd = official_round(
        s3,
        31,
        {"round": 123499, "randomness": "33" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "44" * 32, "n_shards": 100000},
    )
    scale = s3.scale("nano")
    work = tmp_path / "work"
    work.mkdir()
    r_cfg = s3.raw["exam"]["reasoning"]
    rcfg_sig = hashlib.sha256(
        json.dumps({"filter_version": reason_mod.REASONING_FILTER_VERSION, "cfg": r_cfg}, sort_keys=True).encode()
    ).hexdigest()[:8]
    snap_sha = hashlib.sha256(snap_path.read_bytes()).hexdigest()
    rcache = work / f"reasoning-{examine.snapshot_name([snap_sha])[:12]}-{rnd.seed.hex()[:8]}-{rcfg_sig}.jsonl.zst"
    rcache.write_bytes(farm.pack_items(bad_cached))

    monkeypatch.setattr("stet.data.corpus", lambda *a, **k: types.SimpleNamespace(docs=[], meta={"sha256": "5" * 64}))
    monkeypatch.setattr(examine, "_student_window", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="py3langid language classification failed"):
        examine.assemble(rnd, scale, [(0, snap_path)], work, measure=False, synthesize=True, check_coverage=False, log=lambda _: None)
    assert rcache.exists(), "classify() failure must NOT unlink an intact reasoning cache file"

    # 4. Empty detection string under require_langid=True also fails closed.
    empty_langid = types.SimpleNamespace(rank=reason_mod._langid.rank, classify=lambda text: ("", 0.0))
    monkeypatch.setattr(reason_mod, "_langid", empty_langid)
    with pytest.raises(RuntimeError, match="empty language prediction"):
        reason_mod.parse_generated_items_detailed(doc_sel, raw_gens, require_langid=True)

    # 5. Non-official mode (require_langid=False) stays graceful when classify() raises.
    monkeypatch.setattr(reason_mod, "_langid", broken_langid)
    non_official_items, _, _ = reason_mod.parse_generated_items_detailed(
        doc_sel[:2], raw_gens[:2], require_langid=False
    )
    assert len(non_official_items) == 2


def test_review21_exam_scoring_deterministic_no_index_add_and_guard(monkeypatch):
    """Review 21 P3 #7: large-vocab `nll()` must never call GPU-nondeterministic `index_add_`,
    must produce bit-identical results matching dense FP32 CE under strict deterministic mode, and
    `sit_exam()` must wrap `sit()` in `_deterministic_cuda(use_det)` matching `train()`."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import contextlib

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from stet import grade
    from stet.exam import score as score_mod
    from stet.spec import official_round

    vocab = 65540
    hidden = 8
    torch.manual_seed(42)

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)

        def forward(self, x):
            return types.SimpleNamespace(last_hidden_state=self.emb(x))

    class _SplitLargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab)
            self.model = _Inner()
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

    class _DirectLargeModel(nn.Module):
        def __init__(self, split_m):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab)
            self._split = split_m

        def forward(self, x):
            h = self._split.model(x).last_hidden_state
            return types.SimpleNamespace(logits=self._split.lm_head(h))

    split_m = _SplitLargeModel().eval()
    direct_m = _DirectLargeModel(split_m).eval()

    # Four identical sequences (same shape as reviewer repro) plus a shorter varied sequence.
    base_seq = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3)
    seqs = [base_seq, base_seq, base_seq, base_seq, ([1, 5, 8, 2, 9], 2)]

    # Reference dense FP32 cross-entropy.
    expected = []
    with torch.no_grad():
        for ids, first in seqs:
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            y[: max(0, first - 1)] = -100
            logits = split_m.lm_head(split_m.model(x).last_hidden_state)[0].float()
            expected.append(float(F.cross_entropy(logits, y, ignore_index=-100, reduction="sum").item()))

    # Forbid index_add_ completely during nll() AND enforce strict deterministic algorithms!
    def _forbid_index_add(*a, **k):
        raise AssertionError("index_add_ must not be called during exam scoring")

    monkeypatch.setattr(torch.Tensor, "index_add_", _forbid_index_add)
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        dev = torch.device("cpu")
        runs_split = [score_mod.nll(split_m, seqs, dev, batch=4) for _ in range(5)]
        runs_direct = [score_mod.nll(direct_m, seqs, dev, batch=4) for _ in range(5)]
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)

    for r in runs_split[1:]:
        assert r == runs_split[0], "split large-vocab nll() must be bit-identical across runs"
    for r in runs_direct[1:]:
        assert r == runs_direct[0], "direct large-vocab nll() must be bit-identical across runs"
    assert runs_split[0][0] == runs_split[0][1] == runs_split[0][2] == runs_split[0][3]
    assert runs_direct[0][0] == runs_direct[0][1] == runs_direct[0][2] == runs_direct[0][3]
    for got_s, got_d, want in zip(runs_split[0], runs_direct[0], expected, strict=True):
        assert abs(got_s - want) < 1e-4
        assert abs(got_d - want) < 1e-4

    # Verify sit_exam() invokes _deterministic_cuda(True) on Season 3 hf_topreset CUDA devices
    # and _deterministic_cuda(False) on CPU devices.
    det_flags: list[bool] = []

    @contextlib.contextmanager
    def _spy_det(enabled: bool):
        det_flags.append(bool(enabled))
        yield

    monkeypatch.setattr(grade, "_deterministic_cuda", _spy_det)
    monkeypatch.setattr(score_mod, "sit", lambda *a, **k: [-1.5])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    s3 = load_season("s3")
    rnd = official_round(
        s3,
        32,
        {"round": 123500, "randomness": "66" * 32},
        {"crawl": "CC-MAIN-2026-43", "paths_sha256": "77" * 32, "n_shards": 100000},
    )
    exam = [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source="explain:wikipedia:en:1", ctx="C", cont=" A")]
    fake_cuda_model = types.SimpleNamespace(parameters=lambda: iter([]))
    grade.sit_exam(grade.Trained(result={}, trained=True, model=fake_cuda_model, enc=None, device=types.SimpleNamespace(type="cuda")), rnd, s3.scale("nano"), exam)
    grade.sit_exam(grade.Trained(result={}, trained=True, model=fake_cuda_model, enc=None, device=types.SimpleNamespace(type="cpu")), rnd, s3.scale("nano"), exam)
    assert det_flags == [True, False], f"expected _deterministic_cuda flags [True, False], got {det_flags}"


def test_review22_fetch_round_shares_prefers_server_response_over_stale_cache(monkeypatch, tmp_path):
    """Review 22 P2 #5: fetch_round_shares() MUST prefer the current server response's language
    shares and refresh the cache on disk rather than returning stale cached weights from another
    server sharing the same round ID and exam digest, falling back to cache only when the server
    supplies no weights."""
    import dataclasses

    from stet import client, data
    from stet.exam.aggregate import plan_of, round_score
    from stet.exam.items import Item, digest
    from stet.spec import load_season, official_round

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("STET_DEV", "1")
    monkeypatch.setattr(client, "verify_beacon", lambda *a, **k: None)

    s3 = load_season("s3")
    items = (
        [Item(kind="answer", corner="explain/en/wiki", fmt="prose", lang="en", source=f"explain:wikipedia:en:{i}", ctx=f"Ex {i}.", cont=f" Answer {i}.") for i in range(24)]
        + [Item(kind="answer", corner="explain/de/wiki", fmt="prose", lang="de", source=f"explain:wikipedia:de:{i}", ctx=f"Erkl {i}.", cont=f" Antwort {i}.") for i in range(24)]
    )
    exam_sha = digest(items)
    rnd = dataclasses.replace(
        official_round(
            s3,
            41,
            {"round": 123600, "randomness": "88" * 32},
            {"crawl": "CC-MAIN-2026-43", "paths_sha256": "99" * 32, "n_shards": 100000},
        ),
        exam_digest=exam_sha,
    )

    # Seed stale cached weights (10% en, 90% de) from another server sharing the cache directory.
    data.save_round_shares(rnd, {"en": 10.0, "de": 90.0})
    assert data.round_shares(rnd) == {"en": 10.0, "de": 90.0}

    # 1. Server returns 90% en / 10% de in GET /api/v1/rounds/{rid} -> fetch_round MUST refresh
    #    the cache so local scoring matches the server (+0.080 bpb, NOT stale -0.080 bpb!).
    round_doc = {**rnd.describe(), "status": "open", "languages": {"en": 90.0, "de": 10.0}}
    monkeypatch.setattr(client, "request", lambda method, path, **kw: {"round": round_doc} if path == f"/api/v1/rounds/{rnd.id}" else {})
    fetched = client.fetch_round(s3, 41)
    assert fetched.exam_digest == exam_sha
    assert data.round_shares(fetched) == {"en": 90.0, "de": 10.0}
    item_gains = [0.10] * 24 + [-0.10] * 24
    agg = round_score(items, item_gains, shares=data.round_shares(fetched), plan=plan_of(s3))
    assert agg["score"] == pytest.approx(0.080, abs=1e-6), f"expected +0.080 bpb with refreshed server weights, got {agg['score']}"

    # 2. Settled round omitting `languages` on /rounds/{rid} but returning updated weights on
    #    /rounds/{rid}/scores MUST also refresh stale cached weights.
    data.save_round_shares(rnd, {"en": 10.0, "de": 90.0})
    monkeypatch.setattr(
        client,
        "request",
        lambda method, path, **kw: {"languages": {"en": 75.0, "de": 25.0}} if path.endswith("/scores") else {},
    )
    got_settled = client.fetch_round_shares(rnd, {"status": "settled"})
    assert got_settled == {"en": 75.0, "de": 25.0}
    assert data.round_shares(rnd) == {"en": 75.0, "de": 25.0}

    # 3. When the server supplies no valid weights (or /scores fails), cached weights remain the fallback.
    monkeypatch.setattr(
        client,
        "request",
        lambda method, path, **kw: (_ for _ in ()).throw(client.ApiError(503, "down")),
    )
    got_fallback = client.fetch_round_shares(rnd, {"status": "settled"})
    assert got_fallback == {"en": 75.0, "de": 25.0}
    assert client.fetch_round_shares(rnd, {"status": "open", "languages": {"en": True, "de": -1.0}}) == {"en": 75.0, "de": 25.0}


def test_review24_select_context_handles_zero_percent_measured_language_shares():
    """Review 24 P2 #4: measured shares rounded to 0.0% (e.g. {'en': 100.0, 'de': 0.0}) MUST NOT
    raise ZeroDivisionError in _select_kind, whether all candidates for a kind are in zero-weight
    languages or positive-weight languages run out leaving only zero-weight candidates."""
    from stet.exam.fresh import Doc
    from stet.exam.fresh.select import select, select_context

    def _wiki_doc(lang: str, idx: int) -> Doc:
        if lang == "de":
            text = (
                f"Der Artikel Nummer {idx} beschreibt eine regionale Eisenbahnstrecke im Norden mit mehreren historischen Bahnhoefen und Bruecken. "
                f"Seit dem Umbau im vergangenen Jahr verkehren dort wieder regelmaessige Zuege zwischen den groesseren Staedten der Region. "
                f"Die neue Signaltechnik verbessert die Puenktlichkeit auf dem gesamten Streckenabschnitt Nummer {idx} deutlich."
            )
            title = f"Strecke {idx}"
        elif lang == "fr":
            text = (
                f"L article numero {idx} presente un pont ferroviaire historique qui traverse la vallee centrale sur plusieurs centaines de metres. "
                f"Apres des travaux de renovation complets, la circulation regionale a repris a cadence reguliere dans toute la province. "
                f"Les ingenieurs soulignent la robustesse de la structure metallique installee pour le troncon numero {idx}."
            )
            title = f"Pont {idx}"
        else:
            text = (
                f"Article number {idx} describes a coastal weather observatory that records wind pressure and ocean temperature throughout the year. "
                f"Researchers use its continuous archive to study seasonal storms and long-term shifts across the northern basin. "
                f"Recent instrument upgrades increased sampling precision across station sector number {idx} without interrupting daily logs."
            )
            title = f"Observatory {idx}"
        return Doc(
            source="wikipedia",
            id=f"{lang}-{idx}",
            url=f"https://{lang}.wikipedia.org/wiki/{idx}",
            lang=lang,
            license="CC-BY-SA-4.0",
            published="2026-07-01T12:00:00Z",
            topic=f"topic-{lang}-{idx}",
            title=title,
            text=text,
        )

    # 1. All-zero candidate pool for one kind: 6 German Wikipedia docs under {"en": 100.0, "de": 0.0}.
    de_docs = [_wiki_doc("de", i) for i in range(6)]
    all_zero_ctx = select_context(de_docs, b"r24-zero-all", 6, {"reference"}, per_corner=2, shares={"en": 100.0, "de": 0.0})
    assert len(all_zero_ctx) == 6
    assert all(it.lang == "de" for it in all_zero_ctx)
    # Plain select() uses _select_kind too and must also handle all-zero shares cleanly.
    all_zero_plain = select(de_docs, b"r24-zero-all", 6, {"reference"}, per_corner=2, shares={"en": 100.0, "de": 0.0})
    assert len(all_zero_plain) == 6

    # 2. Mixed pool where positive-weight English candidates run out leaving zero-weight German
    #    and French candidates.
    mixed_docs = [_wiki_doc("en", i) for i in range(2)] + [_wiki_doc("de", i) for i in range(4)] + [_wiki_doc("fr", i) for i in range(4)]
    mixed_ctx = select_context(
        mixed_docs,
        b"r24-zero-mixed",
        8,
        {"reference"},
        per_corner=2,
        shares={"en": 100.0, "de": 0.0, "fr": 0.0},
    )
    assert len(mixed_ctx) == 8
    langs = [it.lang for it in mixed_ctx]
    assert langs.count("en") == 2
    assert langs.count("de") == 3
    assert langs.count("fr") == 3


def test_review25_sparse_harvest_backfills_across_29_30_boundary_and_passes_coverage():
    """Review 25 P2 #4: adding a 30th above-cutoff English document to a sparse harvest with 300
    below-cutoff Welsh documents MUST NOT drop selection from 329 down to 30 (10/10/10) and cancel
    a viable round under `min_pillar_items=20`. Selection must smoothly backfill up to the pillar
    support floor without discontinuity and pass `_filter_pillar_support` on both sides of 29/30."""
    from collections import Counter

    from stet.exam.examine import _filter_pillar_support
    from stet.exam.fresh import Doc
    from stet.exam.fresh.reason import select_source_docs
    from stet.spec import load_season

    s3 = load_season("s3")
    shares = {"en": 50.0, "cy": 0.01}

    def _make_doc(lang: str, idx: int) -> Doc:
        if lang == "cy":
            text = (
                f"Mae erthygl rhif {idx} yn disgrifio rhwydwaith rheilffordd hanesyddol ar draws dyffrynnoedd y gogledd gyda nifer o bontydd cerrig a gorsafoedd lleol. "
                f"Wedi gwaith adfer manwl yn ystod y blynyddoedd diwethaf, mae trenau rheolaidd bellach yn cysylltu pentrefi gwledig a threfi marchnad bob awr. "
                f"Mae peirianwyr lleol yn nodi bod systemau signalau modern ar adran rhif {idx} wedi gwella dibynadwyedd teithiau dyddiol yn sylweddol ac wedi lleihau oedi, "
                f"tra bod cynghorau sir yn buddsoddi mewn llwybrau beicio a gwasanaethau bws cysylltiedig ar gyfer cymunedau gwledig y rhanbarth."
            )
            title = f"Rheilffordd {idx}"
        else:
            text = (
                f"Report number {idx} documents a coastal hydrology network measuring river sediment transport, tidal salinity shifts, and seasonal storm surges. "
                f"Over the past decade, automated sensor buoys installed along the estuary have recorded continuous hourly telemetry used by municipal flood planners. "
                f"Calibration logs from monitoring station {idx} confirm that updated pressure transducers maintain millimeter accuracy even during winter gales, "
                f"and independent verification teams compare buoy observations against satellite altimetry and harbor tide gauges every month."
            )
            title = f"Estuary {idx}"
        return Doc(
            source="wikipedia",
            id=f"{lang}-{idx}",
            url=f"https://{lang}.wikipedia.org/wiki/{idx}",
            lang=lang,
            license="CC-BY-SA-4.0",
            published="2026-07-01T12:00:00Z",
            topic=f"topic-{lang}-{idx}",
            title=title,
            text=text,
        )

    cy_docs = [_make_doc("cy", i) for i in range(300)]
    assert select_source_docs(cy_docs + [_make_doc("en", 0)], b"r25-zero-target", target=0, shares=shares) == []

    # Verify monotonicity and coverage across the 29/30 boundary and up through the 120-doc support floor.
    prev_total = 0
    prev_en = 0
    for n_en in (0, 29, 30, 59, 60, 119, 120, 121):
        docs = cy_docs + [_make_doc("en", i) for i in range(n_en)]
        sel = select_source_docs(docs, b"r25-sparse-boundary", target=640, shares=shares, min_pillar_items=20)
        assert len(sel) >= 120, f"n_en={n_en}: selected only {len(sel)} docs"
        assert len(sel) >= prev_total, f"n_en={n_en}: total dropped from {prev_total} to {len(sel)}"
        got_en = sum(1 for d in sel if d["lang"] == "en")
        assert got_en == n_en, f"n_en={n_en}: expected all {n_en} English docs kept, got {got_en}"
        assert got_en >= prev_en
        if n_en >= 120:
            assert all(d["lang"] == "en" for d in sel), f"n_en={n_en}: micro-language should not be used once support floor is met"
        kinds = Counter(d["task_type"] for d in sel)
        assert all(kinds[k] >= 38 for k in ("explain", "infer", "synthesis")), f"n_en={n_en}: uneven kinds {kinds}"

        # Pass through _filter_pillar_support treating every selected doc as a valid teacher item
        # (and even at 60% teacher acceptance!) -> all 3 reasoning pillars survive, never canceling!
        items_100 = [
            Item(
                kind="answer",
                corner=f"{d['task_type']}/{d['lang']}/wiki",
                fmt="prose",
                lang=d["lang"],
                source=f"{d['task_type']}:{d['source']}:{d['id']}",
                ctx=f"Prompt {i}",
                cont=f" Answer {i}.",
            )
            for i, d in enumerate(sel)
        ]
        kept_100 = _filter_pillar_support(items_100, s3, f"s3-o99-{n_en}")
        assert len(kept_100) == len(items_100)
        items_60 = [it for i, it in enumerate(items_100) if i % 5 < 3]
        kept_60 = _filter_pillar_support(items_60, s3, f"s3-o98-{n_en}")
        assert len(kept_60) == len(items_60)
        prev_total = len(sel)
        prev_en = got_en


def test_review26_public_replay_without_scored_pillars_reports_unavailable_gain(monkeypatch, tmp_path, capsys):
    """Review 26 P2 #4: when a public replay retains only zero-weight diagnostic pillars (e.g.
    20 openly licensed reference items while all 60 CCNews reasoning items are withheld), the CLI
    MUST record gain=None (not numeric 0.0), print 'unavailable' (not '+0.0000'), preserve the
    diagnostic pillar scores, and exclude that replay from multi-round gain summaries."""
    import json
    import sys
    import types

    import zstandard as zstd

    import stet
    from stet import cli, data
    from stet.exam.examine import public_items
    from stet.exam.items import digest as items_digest
    from stet.spec import official_round

    monkeypatch.setenv("STET_CACHE", str(tmp_path))
    s3 = load_season("s3")
    scale = s3.scale("nano")
    p = s3.raw["pools"]["practice"]
    crawl = {"crawl": p["crawl"], "paths_sha256": p["paths_sha256"], "n_shards": int(p["n_shards"])}

    # Build an 80-item exam passing pillar minimums: 20 per reasoning pillar from CCNews (withheld
    # in public replay) + 20 openly licensed Wikipedia reference items (0% diagnostic weight).
    full_items: list[Item] = []
    for pillar in ("explain", "infer", "synthesis"):
        for i in range(20):
            full_items.append(
                Item(
                    kind="answer",
                    corner=f"{pillar}/en/ccnews",
                    fmt="prose",
                    lang="en",
                    source=f"{pillar}:ccnews:{i}",
                    ctx=f"CCNews prompt {pillar} {i}",
                    cont=f" Answer {i}.",
                )
            )
    for i in range(20):
        full_items.append(
            Item(
                kind="context",
                corner="fresh/wikipedia/en/ref",
                fmt="prose",
                lang="en",
                source=f"wikipedia:{i}",
                ctx=f"Reference opening {i}.",
                cont=f" Reference continuation {i}.",
                wrongs=(f"Distractor {i}.",),
            )
        )

    full_sha = items_digest(full_items)
    rnd = official_round(s3, 26, {"round": 900026, "randomness": "26" * 32}, crawl)
    rnd = type(rnd)(**{**rnd.__dict__, "exam_digest": full_sha})

    pub_rows = public_items(full_items)
    pub_raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in pub_rows).encode()
    pub_path = data.exam_public_path(rnd, full_sha)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_bytes(zstd.ZstdCompressor().compress(pub_raw))

    pub_exam = data.round_exam(rnd, scale)
    assert len(pub_exam) == 20
    pub_sha = items_digest(pub_exam)

    # Candidate gains +0.500 bpb on every diagnostic reference item over baseline.
    base_result = {
        "valid": True,
        "exam_sha256": pub_sha,
        "exam": {"items": 20, "mean": -2.5, "item_scores": [-2.5] * 20},
    }
    fake_pipeline = types.SimpleNamespace(baseline=lambda *a, **k: base_result)
    monkeypatch.setitem(sys.modules, "stet.pipeline", fake_pipeline)
    monkeypatch.setattr(stet, "pipeline", fake_pipeline, raising=False)

    cand_result = {
        "valid": True,
        "round": {"id": rnd.id, "kind": rnd.kind},
        "scale": scale.name,
        "script": {"out_docs": 10, "src_docs": 10, "kept_doc_frac": 1.0, "out_bytes": 10000, "passes": 1.0, "truncated": False},
        "train": {"tokens": 1000, "seconds": 1.0, "tokens_per_second": 1000, "final_train_loss": 2.0},
        "exam_sha256": pub_sha,
        "exam": {"items": 20, "mean": -2.0, "item_scores": [-2.0] * 20},
    }
    out = cli.Out()
    base = cli._with_baseline(cand_result, rnd, scale, None, out, skip=False, exam=pub_exam)
    assert cand_result["gain"] is None, f"expected gain=None when no scored pillar survives, got {cand_result['gain']!r}"
    assert cand_result["pillars"] == pytest.approx({"reference": 0.5})
    assert cand_result["partial_exam"] == ["reasoning"]

    cli.report(out, cand_result, base, s3)
    rendered = capsys.readouterr().err
    assert "reference +0.500" in rendered
    assert "unavailable" in rendered
    assert "+0.0000" not in rendered

    # Multi-round summary MUST exclude the diagnostic-only replay instead of diluting mean with 0.0.
    scored_round = {"valid": True, "round": {"id": "s3-o00027"}, "gain": 0.4}
    summary_mixed = cli._summary(cli.Out(), [cand_result, scored_round])
    mixed_err = capsys.readouterr().err
    assert summary_mixed["rounds"] == ["s3-o00027"]
    assert summary_mixed["excluded_rounds"] == [rnd.id]
    assert summary_mixed["gains"] == [0.4]
    assert summary_mixed["mean_gain"] == pytest.approx(0.4)
    assert summary_mixed["se"] is None
    assert "+0.4000" in mixed_err

    # When ALL replays lack scored pillars, mean_gain is None and CLI prints unavailable.
    summary_all = cli._summary(cli.Out(), [cand_result])
    all_err = capsys.readouterr().err
    assert summary_all["mean_gain"] is None
    assert summary_all["se"] is None
    assert summary_all["gains"] == []
    assert summary_all["excluded_rounds"] == [rnd.id]
    assert "unavailable" in all_err
    assert "+0.0000" not in all_err

    # Gate 4 P2 #3: --no-baseline summaries MUST report "(baseline skipped)", NOT missing pillars!
    no_base_rounds = [
        {"valid": True, "round": {"id": "s3-p00000"}},
        {"valid": False, "round": {"id": "s3-p00001"}, "gain": 0.0},
    ]
    summary_nb = cli._summary(cli.Out(), no_base_rounds, no_baseline=True)
    nb_err = capsys.readouterr().err
    assert summary_nb == {"rounds": ["s3-p00000", "s3-p00001"], "gains": [], "mean_gain": None, "se": None}
    assert "baseline skipped" in nb_err and "no scored pillars" not in nb_err

    # Gate 4 P3 #6: report() with gain=None and empty pillars MUST NOT claim "diagnostic pillars shown above".
    empty_pillars = {**cand_result, "pillars": {}}
    cli.report(cli.Out(), empty_pillars, base, s3)
    empty_err = capsys.readouterr().err
    assert "unavailable" in empty_err and "diagnostic pillars shown above" not in empty_err


def test_review26_farm_memory_planner_uses_season_token_width():
    """Review 26 P3 #5: Season 3's 103,424-entry vocab allocates uint32 (4 B/token × 3 buffers =
    12 B/token) in script.materialize, so measure.memory_gb, stages.cost, and docs/OPERATING.md
    MUST use 12 B/token for s3 (~5.86 GiB at large-x4) and 6 B/token for uint16 seasons (s2)."""
    from pathlib import Path

    from stet.exam import measure, stages

    s2 = load_season("s2")
    s3 = load_season("s3")
    assert s2.vocab_size <= 65535 and s3.vocab_size > 65535
    assert measure.token_bytes(s2) == 6
    assert measure.token_bytes(s3) == 12
    assert measure.token_bytes(None) == 6

    std3 = s3.scale("standard")
    large3 = s3.scale("large")
    large3_x4 = measure.at_multiple(large3, 4)

    assert measure.memory_gb(std3, s3) == pytest.approx(12 * std3.train_tokens / 2**30)
    assert measure.memory_gb(large3_x4, s3) == pytest.approx(12 * large3_x4.train_tokens / 2**30)
    assert measure.memory_gb(large3_x4, s3) == pytest.approx(5.859375, abs=1e-4)
    assert measure.memory_gb(s2.scale("standard"), s2) == pytest.approx(6 * s2.scale("standard").train_tokens / 2**30)

    # stages.cost MUST pass season into measure.memory_gb.
    rows = stages.cost(s3, [(std3, 1), (large3_x4, 1)])
    assert rows[0]["memory_gb"] == pytest.approx(12 * std3.train_tokens / 2**30)
    assert rows[1]["memory_gb"] == pytest.approx(12 * large3_x4.train_tokens / 2**30)

    # docs/OPERATING.md table MUST match every row of stages.cost(s3, ...) across all columns.
    operating = (Path(__file__).resolve().parents[1] / "docs" / "OPERATING.md").read_text(encoding="utf-8")
    for name in ("standard", "large"):
        sc = s3.scale(name)
        for mult, suffix in ((0.5, "at ½×"), (1.0, ""), (2.0, "at 2×"), (4.0, "at 4×")):
            row = stages.cost(s3, [(measure.at_multiple(sc, mult), 1)])[0]
            label = f"{name} {suffix}".strip()
            want = f"| {label} | {row['tokens'] / 1e6:.1f}M | {row['hours_each']:.2f} | {row['hold_minutes']} | {row['memory_gb']:.2f} GB |"
            assert want in operating, f"missing/stale operating table row: {want}"
