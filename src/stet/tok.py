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
