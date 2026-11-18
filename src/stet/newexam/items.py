"""An exam item, and what it says about itself."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

KINDS = ("passage", "answer", "pair")
# Formats an item can be in. A round is balanced across formats and kinds of task.
FORMATS = ("prose", "code", "dialogue", "table", "list", "math", "logs", "markup", "transcript", "verse", "structured")


@dataclass(frozen=True)
class Item:
    kind: str  # passage | answer | pair
    corner: str  # the finest unit: family/sub-kind/style..., at most a few items a round
    fmt: str  # one of FORMATS
    lang: str  # "en", "de", ..., or "code:python" for code
    source: str  # "gen:<generator>" or a fresh source's name
    text: str = ""  # passage
    ctx: str = ""  # answer: what the student is given
    cont: str = ""  # answer: the right answer, scored
    good: str = ""  # pair: the acceptable sentence
    bad: str = ""  # pair: its unacceptable twin

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown format {self.fmt!r}")
        if self.kind == "passage" and not self.text:
            raise ValueError("a passage item needs text")
        if self.kind == "answer" and not (self.ctx and self.cont):
            raise ValueError("an answer item needs a context and an answer")
        if self.kind == "pair" and not (self.good and self.bad and self.good != self.bad):
            raise ValueError("a pair item needs two different sentences")

    @property
    def family(self) -> str:
        """The generator or source family: the corner's first part."""
        return self.corner.split("/", 1)[0]

    @property
    def stratum(self) -> tuple[str, str]:
        """What a round balances across: the format and the kind of task."""
        return (self.fmt, self.kind)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v != ""}

    @classmethod
    def from_dict(cls, d: dict) -> Item:
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})

    @property
    def id(self) -> str:
        """A hash of everything the item says: two machines with the same item agree on it."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def digest(items: list[Item]) -> str:
    """One hash for a whole exam, in order."""
    h = hashlib.sha256()
    for it in items:
        h.update(it.id.encode())
    return h.hexdigest()
