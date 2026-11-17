#!/usr/bin/env python3
"""Train a season's tokenizer: a small byte-level BPE.

Why not GPT-2's 50k vocabulary: at Stet's scales the student is tiny, and a 50k-way
softmax would be most of its compute and most of its parameters. A small vocabulary puts
the budget into the transformer, and gets the student past bag-of-words statistics on a
budget volunteers can afford.

Trained on an equal mix (by bytes) of raw pool text and practice-split exam passages, so it
is not alien to either. Output is tiktoken's format: `base64(token) rank` per line.

    python tools/train_tokenizer.py --vocab 8192 --out src/stet/seasons/s0-vocab.tiktoken
"""
import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import zstandard
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stet import data
from stet.spec import load_season


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192, help="total vocabulary including <|endoftext|>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mb", type=int, default=50, help="MB of text from each of the two halves")
    args = ap.parse_args()

    season = load_season("s0")
    limit = args.mb * 1_000_000

    def exam_texts():
        n = 0
        with open(data.exam_pack_path(season), "rb") as f:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
                it = json.loads(line)
                if it["split"] != "practice":
                    continue
                n += len(it["text"])
                yield it["text"]
                if n >= limit:
                    return

    def pool_texts():
        n = 0
        shard = season.pool("practice")[0]  # the first practice shard, in full, all languages
        for _, text in data.read_shard(data.shard_path(shard)):
            text = text[:20000]
            n += len(text.encode("utf-8", "replace"))
            yield text
            if n >= limit:
                return

    def both():
        yield from exam_texts()
        yield from pool_texts()

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab - 1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator(both(), trainer)

    u2b = {c: b for b, c in gpt2_bytes_to_unicode().items()}
    vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    lines = []
    seen = set()
    for token, rank in vocab:
        raw = bytes(u2b[c] for c in token)
        assert raw not in seen, f"duplicate token bytes {raw!r}"
        seen.add(raw)
        lines.append(f"{base64.b64encode(raw).decode()} {rank}")
    assert len(lines) == args.vocab - 1, len(lines)
    assert all(bytes([b]) in seen for b in range(256)), "every single byte must be a token"
    blob = ("\n".join(lines) + "\n").encode()
    args.out.write_bytes(blob)
    print(f"wrote {args.out}: {len(lines)} tokens + <|endoftext|> = {args.vocab}", file=sys.stderr)
    print(hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main()
