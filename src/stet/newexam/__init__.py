"""The exam: made after each draw, from things nobody can know before it.

A round's exam has two halves. *Generated* items are made by the generators in `gen/` from the
round's seed: facts in ten languages, reading in context, arithmetic and word problems,
programs and what they print, shell sessions, logic, structured lookups, grammar pairs.
*Fresh* items are human writing first published after the draw (`fresh/`). Every item belongs
to a corner (a topic, in a format, for a kind of task, in a language), and a round draws at
most a few items from any one corner.

Three kinds of item, all scored by log-likelihood, never by accuracy:

  passage  bits per byte of a passage
  answer   bits per byte of the right answer, given its question
  pair     how many bits the student prefers the good sentence of a minimal pair by

A curator's score is its paired gain over the unedited baseline on the same items: the mean of
seven pillars, each counting the same (`aggregate.py`). The organisers' examiner makes each
round's exam (`examine.py`); graders sit it (`score.py`). See
docs/research/exam-first-principles.md for why it is this exam.
"""

from .items import Item

__all__ = ["Item"]
