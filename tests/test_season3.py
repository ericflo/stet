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

    # Verify concrete s3-p00000 exam and SFT probe share zero canonical Wikidata subject Q-IDs across all languages.
    s3 = load_season("s3")
    rnd = practice_round(s3, 0)
    exam_items = [it for it in gen.for_season(s3, rnd.seed) if it.corner.startswith("knowledge-even/")]
    assert exam_items and all(it.kind == "answer" and not it.wrongs for it in exam_items)
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

    # SFT probe generates deterministic multilingual (prompt, completion) pairs without exam access.
    p64 = sft.build_sft_examples(b"round-seed", n=64)
    assert len(p64) == 64
    assert p64 == sft.build_sft_examples(b"round-seed", n=64)
    assert p64 != sft.build_sft_examples(b"other-seed", n=64)
    assert all(isinstance(p, str) and isinstance(c, str) and p and c for p, c in p64)


def test_s3_generated_exam_uses_right_answer_logic_and_knowledge_even():
    s3 = load_season("s3")
    items = gen.for_season(s3, b"s3-test-seed")
    parts = {it.corner.split("/", 1)[0] for it in items}
    assert "logic" in parts and "knowledge-even" in parts
    assert "logic-mc" not in parts and "knowledge-mc" not in parts
    for it in items:
        part = it.corner.split("/", 1)[0]
        if part in ("logic", "knowledge-even"):
            assert it.kind == "answer" and not it.wrongs


def test_uint32_script_materialization_preserves_large_vocab_token_ids(tmp_path):
    """Qwen3's 151,936 vocab exceeds uint16 (65535); materialize must use uint32 without wraparound."""
    from conftest import make_corpus

    class BigVocabEnc:
        n_vocab = 151_936

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
    assert int(tokens.max()) == s3.eot == 151_643
    assert int(tokens[0, 0]) == s3.eot
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
    assert {"vllm", "liger-kernel"} <= cuda_only

    monkeypatch.setattr(env, "is_nvidia_linux", lambda: True)
    assert cuda_only <= set(env.expected(s3)["packages"])

    monkeypatch.setattr(env, "is_nvidia_linux", lambda: False)
    assert not (cuda_only & set(env.expected(s3)["packages"]))


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
    assert main_src.index("task.write(") < main_src.index("_score_vllm(")

    # Load module and verify trailing-space rubric fix, overprovision/reserve, and vLLM failure fallback.
    spec = importlib.util.spec_from_file_location("reader_curator_fallback_test", root / "curator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.RUBRIC_SUFFIX.endswith(": "), "RUBRIC_SUFFIX must end with trailing space so next token is digit 1..5"
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

    fake_task = types.SimpleNamespace(device=lambda: "cuda", time_left=lambda: 600.0)
    cands = [
        (1.0, 0, 2000, "日本語の長文ドキュメントです。" * 80),
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
