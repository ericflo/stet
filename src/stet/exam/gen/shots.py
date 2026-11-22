"""A family asked few-shot: a few solved examples of a task, then the task.

Asked plainly, arithmetic, unit conversions, algebra, word problems, program tracing and lookups
in structured data order curators no better than chance at any student size: a small student
cannot do them cold. Shown a few solved examples of the very same kind first, the question is
another one: can the student pick a pattern up from what is in front of it and carry it on? That
is learning in context, and a small student trained on good data may show it. Season 2 watches to
find out. Each family asked this way is a part of its own (arith-shots beside arith), so it is
sat, scored and published every round, and counts for nothing until a season names it.

The examples are items of the question's own corner, made by the family's own generator from
other draws of the seed: they are in exactly the question's format and differ from it in their
operands only, and none of them is the question. There are three to six of them, fewer when
more would not fit a student's window. A family whose examples share something long (a table
several questions are asked of) says so with a function `shared`.

A family that can say what a near miss is (`near_miss`) is also asked as pairs (arith-pairs): the
same examples and question, once with the right answer and once with one that is nearly right.
"""

from __future__ import annotations

from ..items import Item
from ..rng import Rng

SHOTS = (3, 6)  # how many examples a question is given, before any are dropped for room
# A question and its examples should fit the smallest student's window, 256 tokens. A generator
# may not depend on a season's tokenizer, so tokens are estimated from what a text is made of, in
# twentieths of a token: fitted to the season 2 tokenizer on these very items, where it is within
# about a tenth of the truth. The tests hold it to the real tokenizer.
ROOM = 232 * 20
SEP = "\n\n"


def cost(text: str) -> int:
    """About how many tokens a text takes, in twentieths. Whole numbers only, so that every
    machine drops the same examples."""
    n, in_word = 0, False
    for ch in text:
        letter = ch.isascii() and ch.isalpha()
        if letter:
            n += 5 if in_word else 15  # half a token for a word, a quarter for each of its letters
        elif ch == "\n":
            n += 32
        elif ch in " \t":
            n += 1
        elif not ch.isascii():
            n += 20 if ord(ch) >= 0x800 else 10  # Chinese and Japanese, or Cyrillic and accents
        elif ch.isdigit():
            n += 11
        else:
            n += 21
        in_word = letter
    return n


def examples(base, rng: Rng, corner: str) -> tuple[str, list[Item], Item, str]:
    """A question of a family's corner and the solved examples shown before it: what they all
    share (often nothing), the examples, the question, and what stands between them."""
    k = rng.between(*SHOTS)
    if hasattr(base, "shared"):
        head, shots, ask, sep = base.shared(rng, corner, k)
    else:
        head, sep, seen, made = "", SEP, set(), []
        for i in range(8 * (k + 1)):  # a corner with few questions to ask gives the examples it can
            it = base.make(rng.child(f"shot{i}"), corner)
            if it.ctx not in seen:
                seen.add(it.ctx)
                made.append(it)
            if len(made) == k + 1:
                break
        ask, shots = made[0], made[1:]
    while len(shots) > 1 and cost(_before(head, shots, ask, sep) + ask.cont) > ROOM:
        shots = shots[1:]
    return head, shots, ask, sep


def _before(head: str, shots: list[Item], ask: Item, sep: str) -> str:
    """Everything up to the answer: the examples with their answers, then the question."""
    return head + "".join(s.ctx + s.cont + sep for s in shots) + ask.ctx


class Shots:
    """A family's corners, each asked after examples of itself. It stands where a generator's
    module stands: a NAME, corners(), make(). Its corners are the family's under its own name
    (arith-shots/add/small/qa for arith/add/small/qa), so its items are a part of their own."""

    suffix = "shots"

    def __init__(self, base, only=lambda corner: True):
        self.base, self.NAME, self.only = base, f"{base.NAME}-{self.suffix}", only

    def corners(self) -> list[str]:
        return [self.NAME + c[len(self.base.NAME):] for c in self.base.corners() if self.only(c)]

    def of(self, corner: str) -> str:
        """The family's corner a corner of this one asks."""
        return self.base.NAME + corner[len(self.NAME):]

    def make(self, rng: Rng, corner: str) -> Item:
        head, shots, ask, sep = examples(self.base, rng, self.of(corner))
        return Item(kind="answer", corner=corner, fmt=ask.fmt, lang=ask.lang, source=f"gen:{self.NAME}",
                    ctx=_before(head, shots, ask, sep), cont=ask.cont)


class Pairs(Shots):
    """The same, as pairs: after the examples and the question, the right answer or a near miss.
    Both end with the line: without it 6 would always be likelier than 60, which begins with it."""

    suffix = "pairs"

    def make(self, rng: Rng, corner: str) -> Item:
        head, shots, ask, sep = examples(self.base, rng, self.of(corner))
        before = _before(head, shots, ask, sep)
        miss = self.base.near_miss(rng.child("miss"), self.of(corner), ask)
        return Item(kind="pair", corner=corner, fmt=ask.fmt, lang=ask.lang, source=f"gen:{self.NAME}",
                    good=f"{before}{ask.cont}\n", bad=f"{before}{miss}\n")
