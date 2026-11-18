"""The next exam: made at each draw, from things nobody can know before it.

A round's exam has two halves. *Generated* items are made by the generators in `gen/` from the
round's seed: arithmetic and word problems, programs and what they print, shell sessions,
logic, structured lookups, grammar pairs. *Fresh* items are text first published after the
draw (not built yet). Every item belongs to a corner (a topic, in a format, for a kind of
task, in a language), and a round draws at most a few items from any one corner.

Three kinds of item, all scored by log-likelihood, never by accuracy:

  passage  bits per byte of a passage, as the season 0 exam scores its passages
  answer   bits per byte of the right answer, given its question
  pair     how many bits the student prefers the good sentence of a minimal pair by

A curator's score is its paired gain over the unedited baseline on the same items, balanced
across formats and kinds of task by a clipped, trimmed mean (`aggregate.py`).

See docs/research/exam-proposal.md. Nothing here decides an official score yet: it is the
shadow exam of season 0.
"""

from .items import Item

__all__ = ["Item"]
