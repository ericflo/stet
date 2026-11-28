"""The season tokenizer: either a bundled tiktoken byte-level BPE (s1/s2) or a pinned
Hugging Face fast tokenizer (`tokenizers`, s3+).
"""

from __future__ import annotations

import base64
import hashlib
import re
from functools import cache
from importlib import resources

from .spec import Season

# The GPT-2 pre-tokenisation pattern, as tiktoken writes it.
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


def _gpt2_unicode_to_byte() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


_BYTE_DECODER = _gpt2_unicode_to_byte()


class _HFEncoding:
    """Thin adapter exposing tiktoken's encode_ordinary / encode_ordinary_batch over HF `tokenizers`.

    Uses a plain tokenizer with special `added_tokens` cleared for encoding so literal strings like
    `<|endoftext|>`, `</s>`, `<|im_start|>`, and `<tool_call>` in web pages are tokenized as
    ordinary text rather than turning into control tokens (matching tiktoken's `encode_ordinary`).
    """

    def __init__(self, enc_tok, dec_tok, name: str, *, byte_fallback: bool = False):
        self._enc = enc_tok
        self._dec = dec_tok
        self.name = name
        self._byte_fallback = byte_fallback

    def encode_ordinary(self, text: str) -> list[int]:
        return self._enc.encode(text, add_special_tokens=False).ids

    def encode_ordinary_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._enc.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        return self._dec.decode(ids, skip_special_tokens=False)

    def decode_bytes(self, ids: list[int]) -> bytes:
        """Recover exact UTF-8 byte length of a token slice without U+FFFD inflation.

        Handles both GPT-2 ByteLevel BPE (Qwen3) and SentencePiece Metaspace + ByteFallback
        (`<0xNN>` + `▁` -> space, ERNIE-4.5).
        """
        out = bytearray()
        if self._byte_fallback:
            for tid in ids:
                tok = self._dec.id_to_token(int(tid))
                if tok is None:
                    return self.decode(ids).encode("utf-8")
                if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                    try:
                        out.append(int(tok[3:5], 16))
                        continue
                    except ValueError:
                        pass
                out.extend(tok.replace("\u2581", " ").encode("utf-8"))
            return bytes(out)
        for tid in ids:
            tok = self._dec.id_to_token(int(tid))
            if tok is None:
                return self.decode(ids).encode("utf-8")
            for ch in tok:
                b = _BYTE_DECODER.get(ch)
                if b is None:
                    return self.decode(ids).encode("utf-8")
                out.append(b)
        return bytes(out)


def decoded_byte_len(enc, scored_ids: list[int], full_text: str, full_ids: list[int]) -> int:
    """Exact UTF-8 byte length of `scored_ids` (a prefix of `full_ids` encoding `full_text`)."""
    if len(scored_ids) >= len(full_ids):
        return len(full_text.encode("utf-8"))
    if hasattr(enc, "decode_bytes"):
        return len(enc.decode_bytes(scored_ids))
    if hasattr(enc, "decode"):
        return len(enc.decode(scored_ids).encode("utf-8"))
    return max(1, round(len(full_text.encode("utf-8")) * len(scored_ids) / max(1, len(full_ids))))


def _hf_model_dir(repo: str, revision: str):
    from .data import cache_dir

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "--", repo).strip("-")
    return cache_dir() / "models" / slug / revision


@cache
def load_encoding(season: Season):
    cfg = season.raw["tokenizer"]
    if cfg.get("kind") == "hf":
        import json

        from tokenizers import Tokenizer

        from .data import fetch

        dest = _hf_model_dir(cfg["repo"], cfg["revision"]) / cfg["file"]
        path = fetch(
            [cfg["url"]],
            cfg["sha256"],
            dest,
            label=f"{cfg['repo']}/{cfg['file']}",
            max_bytes=int(cfg.get("size") or (64 << 20)) + (1 << 20),
            verify=True,
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Strip every added token so literal control strings in web text (whether marked
        # special like `<|endoftext|>`/`</s>` or non-special like `<tool_call>`) encode as
        # ordinary text; ERNIE's digit tokens `0`..`9` are already in `model.vocab`.
        plain_tok = Tokenizer.from_str(json.dumps({**raw, "added_tokens": []}))
        full_tok = Tokenizer.from_file(str(path))
        dec_cfg = raw.get("decoder") or {}
        byte_fallback = dec_cfg.get("type") == "ByteFallback" or any(
            d.get("type") == "ByteFallback" for d in (dec_cfg.get("decoders") or [])
        )
        return _HFEncoding(plain_tok, full_tok, name=f"stet-{season.id}", byte_fallback=byte_fallback)

    import tiktoken

    blob = resources.files("stet.seasons").joinpath(cfg["file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != cfg["sha256"]:
        raise RuntimeError(f"tokenizer file {cfg['file']} does not match the season's sha256")
    ranks = {}
    for line in blob.splitlines():
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return tiktoken.Encoding(
        name=f"stet-{season.id}",
        pat_str=PAT_STR,
        mergeable_ranks=ranks,
        special_tokens={"<|endoftext|>": cfg["eot"]},
    )


# --- reading text nobody chose -------------------------------------------------------------------

# tiktoken's pattern matcher gives up on a run of about a million whitespace characters that
# is followed by something else. It gives up with a Rust panic, which Python sees as a
# BaseException: nothing written for an Exception catches it, and one such text, in a page
# of the corpus or in an entrant's script, would stop every machine that read it. So a text
# the matcher cannot read has its longest whitespace runs cut to this many characters and is
# read again. Every machine does the same, so they still agree on the tokens.
LONG_SPACE_CHARS = 65_536


class Unreadable(ValueError):
    """A text the tokenizer cannot read even with its whitespace cut short."""


def _panicked(e: BaseException) -> bool:
    return type(e).__name__ == "PanicException"


def encode_one(enc, text: str) -> list[int]:
    try:
        return enc.encode_ordinary(text)
    except BaseException as e:
        if not _panicked(e):
            raise
    cut = re.sub(rf"\s{{{LONG_SPACE_CHARS + 1},}}", lambda m: m.group(0)[:LONG_SPACE_CHARS], text)
    try:
        return enc.encode_ordinary(cut)
    except BaseException as e:
        if not _panicked(e):
            raise
        raise Unreadable("the tokenizer cannot read this text") from None


def encode_batch(enc, texts: list[str]) -> list[list[int]]:
    """The tokens of each text. Use this, not the encoder's own batch call, for any text
    that came from the corpus or from a script."""
    try:
        return enc.encode_ordinary_batch(texts)
    except BaseException as e:
        if not _panicked(e):
            raise
    return [encode_one(enc, t) for t in texts]
