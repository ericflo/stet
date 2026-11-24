"""An exam item, and what it says about itself."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

KINDS = ("passage", "answer", "pair", "context")
# Formats an item can be in. A round is balanced across formats and kinds of task.
FORMATS = ("prose", "code", "dialogue", "table", "list", "math", "logs", "markup", "transcript", "verse", "structured")


@dataclass(frozen=True)
class Item:
    kind: str  # passage | answer | pair | context
    corner: str  # the finest unit: family/sub-kind/style..., at most a few items a round
    fmt: str  # one of FORMATS
    lang: str  # "en", "de", ..., or "code:python" for code
    source: str  # "gen:<generator>" or a fresh source's name
    text: str = ""  # passage
    ctx: str = ""  # answer / context: what the student is given
    cont: str = ""  # answer / context: the right answer or continuation, scored
    good: str = ""  # pair: the acceptable sentence
    bad: str = ""  # pair: its unacceptable twin
    wrongs: tuple[str, ...] = ()  # context: matched wrong contexts from other documents of the same source & language

    def __post_init__(self):
        if isinstance(self.wrongs, list):
            object.__setattr__(self, "wrongs", tuple(self.wrongs))
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
        if self.kind == "context":
            if not (self.ctx and self.cont and self.wrongs):
                raise ValueError("a context item needs a context, a continuation, and wrong contexts")
            if any((not w) or w == self.ctx for w in self.wrongs):
                raise ValueError("every wrong context must be non-empty and differ from the real context")

    @property
    def family(self) -> str:
        """The generator or source family: the corner's first part."""
        return self.corner.split("/", 1)[0]

    @property
    def stratum(self) -> tuple[str, str]:
        """What a round balances across: the format and the kind of task."""
        return (self.fmt, self.kind)

    def to_dict(self) -> dict:
        out = {}
        for k, v in asdict(self).items():
            if v == "" or v == ():
                continue
            out[k] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> Item:
        kwargs = {}
        for k in cls.__dataclass_fields__:
            if k == "wrongs":
                kwargs[k] = tuple(d.get(k) or ())
            else:
                kwargs[k] = d.get(k, "")
        return cls(**kwargs)

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
