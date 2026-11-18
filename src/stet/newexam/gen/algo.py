"""Small algorithmic tasks done by hand: sorting, counting, reversing, picking out."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import WORDS

NAME = "algo"
TASKS = ["sort-numbers", "sort-words", "reverse-word", "count-letter", "max", "min", "unique", "word-length",
         "uppercase", "word-count", "digit-sum", "last-word", "next-letter", "middle"]
STYLES = ["instruct", "qa"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def corners() -> list[str]:
    return [f"{NAME}/{t}/{s}" for t in TASKS for s in STYLES]


def _task(rng: Rng, t: str) -> tuple[str, str, str]:
    """(instruction, answer label, answer)"""
    nums = rng.sample(range(1, 100), rng.between(4, 7))
    words = rng.sample(WORDS, rng.between(3, 6))
    w = rng.choice(WORDS)
    if t == "sort-numbers":
        return f"Sort these numbers from smallest to largest: {', '.join(map(str, nums))}", "Sorted", ", ".join(map(str, sorted(nums)))
    if t == "sort-words":
        return f"Put these words in alphabetical order: {', '.join(words)}", "In order", ", ".join(sorted(words))
    if t == "reverse-word":
        return f'Write the word "{w}" backwards', "Backwards", w[::-1]
    if t == "count-letter":
        ch = rng.choice(sorted(set(w)))
        return f'How many times does the letter "{ch}" appear in "{w}"?', "Count", str(w.count(ch))
    if t == "max":
        return f"What is the largest of these numbers: {', '.join(map(str, nums))}?", "Largest", str(max(nums))
    if t == "min":
        return f"What is the smallest of these numbers: {', '.join(map(str, nums))}?", "Smallest", str(min(nums))
    if t == "unique":
        seq = [rng.choice(nums[:3]) for _ in range(7)]
        seen = []
        for x in seq:
            if x not in seen:
                seen.append(x)
        return f"Remove the repeats from this list, keeping the first of each: {', '.join(map(str, seq))}", "Without repeats", ", ".join(map(str, seen))
    if t == "word-length":
        return f'How many letters are in the word "{w}"?', "Letters", str(len(w))
    if t == "uppercase":
        return f'Write "{w}" in capital letters', "In capitals", w.upper()
    if t == "word-count":
        sent = " ".join(rng.sample(WORDS, rng.between(3, 9)))
        return f'How many words are in "{sent}"?', "Words", str(len(sent.split()))
    if t == "digit-sum":
        n = rng.between(100, 999_999)
        return f"What is the sum of the digits of {n}?", "Sum of digits", str(sum(map(int, str(n))))
    if t == "last-word":
        return f"What is the last word in this list: {', '.join(words)}?", "Last word", words[-1]
    if t == "next-letter":
        i = rng.below(25)
        return f'Which letter comes right after "{LETTERS[i]}" in the alphabet?', "Next letter", LETTERS[i + 1]
    if t == "middle":
        odd = nums[: len(nums) - (len(nums) + 1) % 2]
        return f"What is the middle number in this list: {', '.join(map(str, odd))}?", "Middle", str(odd[len(odd) // 2])
    raise ValueError(t)


def make(rng: Rng, corner: str) -> Item:
    _, t, style = corner.split("/")
    ask, label, ans = _task(rng, t)
    ctx = f"{ask}.\n{label}:" if style == "instruct" and not ask.endswith("?") else f"{ask}\n{label}:" if style == "instruct" else f"Q: {ask}\nA:"
    return Item(kind="answer", corner=corner, fmt="list", lang="en", source=f"gen:{NAME}", ctx=ctx, cont=f" {ans}")
