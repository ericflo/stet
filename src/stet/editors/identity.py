"""The control: keep every document, in corpus order. This is the baseline, 1.00x by definition.

The corpus arrives in seeded random order, so the baseline student trains on a fair sample
of the round's raw data.
"""

from stet.editor import Task


def edit(task: Task):
    return range(len(task.corpus))


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()
