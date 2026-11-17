"""A starter curator. Keep English prose, drop the rest, shuffle. Make it better.

Run it:      stet run
Submit it:   stet submit

You are given a corpus of raw web documents and a time limit. Write an edit script: one
line per output document, in the order the student should be trained on them.

    123                               keep document 123 whole
    {"d": 123, "k": [[0, 900]]}       keep byte ranges of document 123
    {"s": [[123, 0, 900], "text"]}    join copies, with a little text of your own
    {"t": "text"}                     a new document

Deleting, reordering and repeating are free. Your own text, and copies shorter than 64
bytes, are "ink": task.info["ink_tokens"] of it per round, which is very little. The stream is cut off at the training budget
(task.train_tokens) and repeated from the top if it is shorter, so keeping less means
training on it more often.
"""

import random

from stet.curator import Task

STOPWORDS = (" the ", " of ", " and ", " to ", " in ", " is ", " that ", " for ")


def looks_english(text: str) -> bool:
    sample = text[:3000]
    if len(sample) < 200:
        return False
    letters = sum(c.isascii() and (c.isalpha() or c == " ") for c in sample)
    if letters / len(sample) < 0.75:
        return False
    low = sample.lower()
    return sum(low.count(w) for w in STOPWORDS) >= max(4, len(sample) // 300)


def main():
    task = Task.from_env()  # task.corpus, task.train_tokens, task.time_left(), task.info
    corpus = task.corpus    # len(corpus), corpus.text(i), corpus.raw(i), corpus.size(i), corpus.urls[i]

    keep = [i for i, text in corpus if looks_english(text)]
    random.Random(0).shuffle(keep)

    # Write early; you can call task.write again with something better. If you run out of
    # time, the last complete script you wrote is the one that counts.
    task.write(keep)


if __name__ == "__main__":
    main()
