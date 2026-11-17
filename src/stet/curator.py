"""What a curator sees. Import this from your curator, or read the same files yourself.

A curator is any program. It is started with these environment variables:

    STET_TASK      path to task.json (budgets, rules, what the exam covers)
    STET_CORPUS    directory holding the raw corpus (see stet.data for the layout)
    STET_OUT       directory to write script.jsonl into
    STET_SCRATCH   a writable scratch directory (memory-backed: it counts as memory)
    STET_ASSETS    directory holding the files the entry named in [[assets]], read-only
    STET_DEADLINE  unix time at which the curator is killed

Write script.jsonl early and overwrite it as you improve it: if you run out of time, the
last complete script you wrote is the one that counts.

Every curator is given the same machine wherever it runs: task.cores CPU cores,
task.memory_gb of memory and task.gpu_memory_gb of GPU memory, on an NVIDIA or an AMD GPU.
PyTorch calls both "cuda". There is no network.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .data import Corpus
from .script import write_script


class Task:
    def __init__(self, info: dict, corpus_dir: str, out_dir: str, scratch_dir: str, deadline: float, assets_dir: str = ""):
        self.info = info
        self.corpus = Corpus(corpus_dir)
        self.out = Path(out_dir)
        self.scratch = Path(scratch_dir)
        self.deadline = deadline
        self.assets = Path(assets_dir) if assets_dir else None

    @classmethod
    def from_env(cls) -> Task:
        info = json.loads(Path(os.environ["STET_TASK"]).read_text())
        return cls(
            info,
            os.environ["STET_CORPUS"],
            os.environ["STET_OUT"],
            os.environ.get("STET_SCRATCH", os.environ["STET_OUT"]),
            float(os.environ.get("STET_DEADLINE", time.time() + info.get("time_limit", 600))),
            os.environ.get("STET_ASSETS", ""),
        )

    @property
    def train_tokens(self) -> int:
        return int(self.info["train_tokens"])

    @property
    def ink_frac(self) -> float:
        return float(self.info["ink_frac"])

    def time_left(self) -> float:
        return self.deadline - time.time()

    def _machine(self, key: str) -> float:
        return float((self.info.get("machine") or {}).get(key) or 0)

    @property
    def cores(self) -> int:
        """CPU cores this curator may use. The same number on every machine that runs it."""
        return int(self._machine("cores")) or (os.cpu_count() or 1)

    @property
    def memory_gb(self) -> float:
        """Memory this curator may use, scratch included. 0 if this run has no limit."""
        return self._machine("memory_gb")

    @property
    def gpu_memory_gb(self) -> float:
        """GPU memory this curator may use. 0 if this run has no limit."""
        return self._machine("gpu_memory_gb")

    def asset(self, path: str) -> Path:
        """A file or directory this entry named in [[assets]]: task.asset("smollm2-135m")
        is the directory holding every asset whose path starts with smollm2-135m/."""
        if self.assets is None:
            raise FileNotFoundError("this run has no assets directory (STET_ASSETS is not set)")
        p = self.assets / path
        if not p.exists():
            named = ", ".join(self.info.get("assets") or []) or "none"
            raise FileNotFoundError(f"no asset at {path!r}. stet.toml names: {named}")
        return p

    def device(self) -> str:
        """Where to put a model: "cuda" when there is a GPU (NVIDIA and AMD alike), else "cpu".

        Every machine that runs curators for the leaderboard has a GPU. Your own may not,
        and a curator that asks this still runs there, slowly.
        """
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def write(self, lines) -> None:
        """Write the edit script (atomically; safe to call more than once)."""
        self.out.mkdir(parents=True, exist_ok=True)
        write_script(self.out / "script.jsonl", lines)
