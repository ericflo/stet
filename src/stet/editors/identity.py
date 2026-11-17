"""The control: keep every document, in the order it arrived. This scores 1.00x by definition."""

from stet.editor import Task


def edit(task: Task):
    return range(len(task.corpus))


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()
