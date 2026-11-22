"""Keep documents that look like English prose; drop the rest. Order unchanged.

More than half of a raw crawl is not English, and this throws all of it away. The exam does
not: three quarters of its score is human writing, each language counting by its share of
the round's corpus, and only the skills quarter is all in English. So this keeps what the
skills ask for and gives up the writing in every other language, about four tenths of the
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
    return [i for i, text in task.corpus if looks_english(text)]


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()
