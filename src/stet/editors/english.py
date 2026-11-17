"""Keep documents that look like English prose; drop the rest. Order unchanged.

The exam is English, and a large share of a raw crawl is not. This is the first thing
anyone would try, and the first rung of the ladder.
"""

from stet.editor import Task

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


def edit(task: Task):
    return [i for i, text in task.corpus if looks_english(text)]


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()
