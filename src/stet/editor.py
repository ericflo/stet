"""What an editor sees. Import this from your editor, or read the same files yourself.

An editor is any program. It is started with these environment variables:

    STET_TASK      path to task.json (budgets, rules, what the exam covers)
    STET_CORPUS    directory holding the raw corpus (see stet.data for the layout)
    STET_OUT       directory to write script.jsonl into
    STET_SCRATCH   a writable scratch directory
    STET_DEADLINE  unix time at which the editor is killed

Write script.jsonl early and overwrite it as you improve it: if you run out of time, the
last complete script you wrote is the one that counts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .data import Corpus
from .script import write_script


class Task:
    def __init__(self, info: dict, corpus_dir: str, out_dir: str, scratch_dir: str, deadline: float):
        self.info = info
        self.corpus = Corpus(corpus_dir)
        self.out = Path(out_dir)
        self.scratch = Path(scratch_dir)
        self.deadline = deadline

    @classmethod
    def from_env(cls) -> "Task":
        info = json.loads(Path(os.environ["STET_TASK"]).read_text())
        return cls(
            info,
            os.environ["STET_CORPUS"],
            os.environ["STET_OUT"],
            os.environ.get("STET_SCRATCH", os.environ["STET_OUT"]),
            float(os.environ.get("STET_DEADLINE", time.time() + info.get("time_limit", 600))),
        )

    @property
    def train_tokens(self) -> int:
        return int(self.info["train_tokens"])

    @property
    def ink_frac(self) -> float:
        return float(self.info["ink_frac"])

    def time_left(self) -> float:
        return self.deadline - time.time()

    def write(self, lines) -> None:
        """Write the edit script (atomically; safe to call more than once)."""
        self.out.mkdir(parents=True, exist_ok=True)
        write_script(self.out / "script.jsonl", lines)
