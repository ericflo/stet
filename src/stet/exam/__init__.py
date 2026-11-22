"""The exam: made after each draw, from things nobody can know before it.

A round's exam has two halves, and the season names every part of both (`aggregate.py`).
*Generated* items are made from the round's seed by the families of generators in `gen/` that
the season names: shell sessions, reading in context, text turned from one form into another,
logic, grammar pairs, numbers and small programs asked after worked examples, facts the web
knows. A family the season does not name is not generated.
*Fresh* items are human writing published in the day before the draw (`fresh/`), of the kinds of
writing the season names: a kind it does not name is not harvested or selected. Every item
belongs to a corner (a topic, in a format, for a kind of task, in a language), and a round
draws at most a few items from any one corner.

Three kinds of item, all scored by log-likelihood, never by accuracy:

  passage  bits per byte of a passage
  answer   bits per byte of the right answer, given its question
  pair     how many bits the student prefers the good sentence of a minimal pair by

A curator's score is its paired gain over the unedited baseline on the same items: each group
of pillars has the season's share of it, and inside a group every pillar counts the same
(`aggregate.py`). The organisers' examiner makes each
round's exam (`examine.py`); graders sit it (`score.py`). See
docs/research/exam-first-principles.md for why it is this exam.
"""

from .items import Item

__all__ = ["Item"]
