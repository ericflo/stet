"""Keep everything, in random order. Isolates what the corpus's own ordering costs."""

import random

from stet.editor import Task


def edit(task: Task):
    order = list(range(len(task.corpus)))
    random.Random(0).shuffle(order)
    return order


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()
