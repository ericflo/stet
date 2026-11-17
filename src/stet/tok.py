"""The season tokenizer: a small byte-level BPE, loaded into tiktoken for speed.

A season ships its own vocabulary (see tools/train_tokenizer.py for why it is small).
Nothing here touches the network.
"""

from __future__ import annotations

import base64
import hashlib
from functools import cache
from importlib import resources

from .spec import Season

# The GPT-2 pre-tokenisation pattern, as tiktoken writes it.
PAT_STR = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


@cache
def load_encoding(season: Season):
    import tiktoken

    cfg = season.raw["tokenizer"]
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
    import re

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

