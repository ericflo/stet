"""A starter curator. Keep pages that read as writing, drop the rest, shuffle. Make it better.

Run it:      stet run        (once, first: stet harvest, so that practice rounds sit real writing)
Submit it:   stet submit

You are given a corpus of raw web pages and a time limit. Write an edit script: one line per
output document, in the order the student should be trained on them.

    123                               keep document 123 whole
    {"d": 123, "k": [[0, 900]]}       keep byte ranges of document 123
    {"s": [[123, 0, 900], [7, 40, 400]]}   join copies [doc, start, end] of any documents

A script is copies of the pages and holds no text of your own: a line with text in it makes
the script invalid. Deleting, reordering and repeating are free, and taking whole lines out
of a page is always allowed. Any other piece must be at least task.info["min_copy_bytes"]
(64) bytes long, or the script is invalid. The stream is cut off at the training budget
(task.train_tokens) and repeated from the top if it is shorter, so keeping less means
training on it more often.

The exam is three quarters writing people published this week, in every language of the
crawl by its share of it, and one quarter generated skills. Keeping one language only throws
away most of what the student is examined on.
"""

import random
import re
import unicodedata

from stet.curator import Task

# Letters of any script, spaces, and the marks that sit on letters (the vowel signs of Hindi,
# Thai and Arabic). A regular expression, because a round's pages are gigabytes and asking
# each character in turn is a hundred times slower.
MARKS = "".join(c for c in map(chr, range(0x300, 0x10000)) if unicodedata.category(c)[0] == "M")
WRITING = re.compile(rf"(?:[^\W\d_]|\s|[{re.escape(MARKS)}])+")


def looks_written(text: str) -> bool:
    """Mostly letters, of any script, and enough of them to say something."""
    sample = text[:3000]
    if len(sample) < 200:
        return False
    return len(WRITING.sub("", sample)) <= 0.25 * len(sample)


def main():
    task = Task.from_env()  # task.corpus, task.train_tokens, task.time_left(), task.info

    # task.map runs a function over every page, on every core you are given, and hands the
    # results back in page order. The pages are also at task.corpus: len(corpus),
    # corpus.text(i), corpus.raw(i), corpus.size(i), corpus.urls[i].
    keep = [i for i, ok in task.map(lambda i, text: looks_written(text)) if ok]
    random.Random(0).shuffle(keep)

    # Write early; you can call task.write again with something better. If you run out of
    # time, the last complete script you wrote is the one that counts.
    task.write(keep)


if __name__ == "__main__":
    main()
