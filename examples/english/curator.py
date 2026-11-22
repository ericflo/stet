"""Keep documents that look like English prose; drop the rest. Order unchanged.

More than half of a raw crawl is not English, and this throws all of it away. The exam does
not: three quarters of its score is human writing, each language counting by its share of
the round's corpus, and most of the skills quarter is asked in the web's main languages too.
So this gives up the writing and the skills in every other language, about half of the
score. It is the first thing anyone would try, and a yardstick for what that costs.
"""

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


def edit(task: Task):
    # task.map asks every page on every core the curator is given, and answers in page order.
    return [i for i, ok in task.map(lambda i, text: looks_english(text)) if ok]


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()
