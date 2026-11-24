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

A loop over the corpus runs on one of those cores. task.map runs it on all of them:

    def keep(i, text):  # in a worker process, once for every page
        return looks_written(text)

    task.write([i for i, ok in task.map(keep) if ok])
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import random
import signal
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .data import Corpus
from .script import write_script

# --- one function over every page, on every core --------------------------------------------------
#
# Task.map, below, is what a curator calls. The pages are cut into chunks, the same chunks
# whatever the number of cores, and each chunk is one job: a worker process is sent its number,
# reads the pages itself (the corpus is memory-mapped, and a forked process shares the map), and
# sends back what the function returned for each. The parent hands the results on in page
# order, holding the chunks that came back before their turn.

CHUNK = 256  # pages in a job
IN_HAND = 2  # jobs a worker has been sent and not answered: one to do, one to go straight on to
AHEAD = 4  # jobs per worker that may be finished or under way beyond the one whose turn it is
# Where a process can be forked and left running without trouble. Curators are run on Linux;
# anywhere else the same function runs in the one process and gives the same results.
FORK = sys.platform.startswith("linux")


class WorkerError(RuntimeError):
    """A worker process of Task.map died, or raised something that could not be brought back."""


def _chunk(fn, corpus: Corpus, docs: list[int] | None, a: int, b: int) -> list:
    """What fn returns for each page of one chunk: pages a to b, or docs[a:b].

    Random numbers start from the chunk's place, not from wherever the process that runs it
    had got to: a function that draws some draws the same ones on any number of cores.
    """
    random.seed(a)
    np.random.seed(a % 2**32)
    return [fn(i, corpus.text(i)) for i in (range(a, b) if docs is None else docs[a:b])]


def _here(fn, corpus: Corpus, docs: list[int] | None, a: int, b: int) -> list:
    """One chunk in this process, whose own random numbers go on from where they were."""
    state = random.getstate(), np.random.get_state()
    try:
        return _chunk(fn, corpus, docs, a, b)
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])


def _whole(e: BaseException) -> BaseException | None:
    """The exception, if it can be sent to another process, or None."""
    try:
        pickle.loads(pickle.dumps(e))
    except Exception:
        return None
    return e


def _worker(fn, corpus: Corpus, docs: list[int] | None, jobs, results, others) -> None:
    """A worker's whole life: jobs in, results out, until the parent closes the pipe."""
    for end in others:  # the parent's ends of the other workers' pipes, which a fork copies
        end.close()
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is the parent's to answer, and it stops its workers
    # Everything inherited is shared with the parent until one of them writes to it. A
    # collection writes to every object it visits: it is told to leave these alone.
    gc.freeze()
    while True:
        try:
            k, a, b = jobs.recv()
        except (EOFError, OSError):
            return  # no more jobs, or no parent
        try:
            try:
                results.send((k, _chunk(fn, corpus, docs, a, b), None))
            except Exception as e:  # what fn raised, or a result that cannot be sent
                results.send((k, None, (_whole(e), traceback.format_exc())))
        except OSError:
            return


def _died(proc) -> str:
    proc.join(5)
    code = proc.exitcode
    if code is not None and code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        why = f"was killed by {name}" + (" (the kernel does that to a process when the curator is over its memory)" if -code == signal.SIGKILL else "")
    else:
        why = "stopped before its work was done" + (f" (exit code {code})" if code is not None else "")
    return f"a worker process of task.map {why}"


def _map(fn: Callable[[int, str], Any], corpus: Corpus, docs: Iterable[int] | None, cores: int, chunk: int) -> Iterator[tuple[int, Any]]:
    if docs is not None:
        docs = [int(i) for i in docs]
    n = len(corpus) if docs is None else len(docs)
    chunk = max(1, int(chunk))
    spans = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]

    def numbered(k: int, out: list):
        a, b = spans[k]
        return zip(range(a, b) if docs is None else docs[a:b], out, strict=True)

    crew: list[list] = []  # a worker: [process, jobs to it, results from it, jobs it has in hand]
    finished = False
    try:
        if FORK and min(int(cores), len(spans)) > 1:
            import multiprocessing
            from multiprocessing.connection import wait

            # Forked, not started afresh: fn may be any function, a lambda or a closure, and
            # what this process has loaded is there for it, shared. Pipes, and no locks or
            # queues: nothing here needs /dev/shm, a semaphore or a socket.
            ctx = multiprocessing.get_context("fork")
            for _ in range(min(int(cores), len(spans))):
                jobs_r, jobs_w = ctx.Pipe(duplex=False)
                results_r, results_w = ctx.Pipe(duplex=False)
                others = [end for w in crew for end in (w[1], w[2])] + [jobs_w, results_r]
                proc = ctx.Process(target=_worker, args=(fn, corpus, docs, jobs_r, results_w, others), daemon=True)
                try:
                    proc.start()
                except OSError:  # no more processes to be had: the ones there are do the work
                    for end in (jobs_r, jobs_w, results_r, results_w):
                        end.close()
                    break
                # Only the worker holds its end now, so its pipe closes when it dies.
                jobs_r.close()
                results_w.close()
                crew.append([proc, jobs_w, results_r, 0])
        if not crew:  # one core, one chunk, or no fork: here, with the same seeds and the same results
            for k, (a, b) in enumerate(spans):
                yield from numbered(k, _here(fn, corpus, docs, a, b))
            finished = True
            return
        back: dict[int, tuple] = {}  # chunks that came back before their turn
        sent = turn = 0
        while turn < len(spans):
            for depth in range(1, IN_HAND + 1):  # every worker one job before any worker two
                for w in crew:
                    if w[3] < depth and sent < len(spans) and sent - turn < AHEAD * len(crew):
                        try:
                            w[1].send((sent, *spans[sent]))
                        except OSError:
                            raise WorkerError(_died(w[0])) from None
                        w[3] += 1
                        sent += 1
            for end in wait([w[2] for w in crew]):
                w = next(w for w in crew if w[2] is end)
                try:
                    k, out, error = end.recv()
                except (EOFError, OSError):
                    raise WorkerError(_died(w[0])) from None
                w[3] -= 1
                back[k] = (out, error)
            while turn in back:
                out, error = back.pop(turn)
                if error is not None:
                    # Raised when its pages' turn comes, after every page before them: as a
                    # loop in one process would have raised it.
                    e, where = error
                    told = WorkerError(f"in a worker process of task.map:\n{where}")
                    if e is None:
                        raise told
                    raise e from told
                yield from numbered(turn, out)
                turn += 1
        finished = True
    finally:
        # However this ends (the last page, an exception, a loop that stopped reading), no
        # worker is left behind. One with nothing left to do is let go; one still working is
        # stopped, since nobody will read what it is working out.
        for w in crew:
            w[1].close()
            if not finished:
                w[0].kill()
        for w in crew:
            w[0].join(5 if finished else None)
            if w[0].is_alive():
                w[0].kill()
                w[0].join()
            w[2].close()


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

    def map(
        self, fn: Callable[[int, str], Any], docs: Iterable[int] | None = None, *, cores: int | None = None, chunk: int = CHUNK
    ) -> Iterator[tuple[int, Any]]:
        """fn(i, text) for every page, on every core this curator is given. Yields (i, result)
        in page order, as `for i, text in task.corpus` yields the pages.

            def keep(i, text):  # in a worker process, once for every page
                return looks_written(text)

            task.write([i for i, ok in task.map(keep) if ok])

        The pages are worked through by task.cores worker processes, a few hundred pages at a
        time, and the results are handed on in the order of the pages whichever was finished
        first. So whatever depends on order (which of two copies of a page came first, how a
        tie falls) is decided in the loop that reads the results, exactly as a loop over
        task.corpus decides it: a curator moved onto task.map writes the script it wrote before.

        The workers are forked from this process when the first result is asked for:

          - What is loaded by then (a model, a table) is theirs to use and is shared, not
            copied: its memory is counted once. What a worker loads for itself is in memory
            once for each worker. All of it comes out of task.memory_gb.
          - fn runs in another process. What it changes (a global, a set of pages seen) stays
            there: return what you need, and keep the tally in the loop that reads it. A result
            is pickled to bring it back, so return numbers, strings, lists and the like.
          - fn should depend on the page and nothing else. Each chunk of pages starts with
            `random` and `numpy.random` seeded by where it stands among the pages, so a
            function that draws random numbers draws the same ones on any number of cores, and
            this process's own are untouched. A generator of your own, seed from the page number.
          - No GPU in fn: a forked process cannot use the one this process has opened. And
            map before PyTorch is loaded and used, where you can: it runs threads of its own,
            and a worker forked while one of them holds a lock can wait on that lock for ever.

        `docs` is the pages to go through, in the order to go through them, if not all of them.
        `cores` is how many processes to use, if fewer than task.cores. `chunk` is how many
        pages a worker takes at a time: fewer when a page takes long, or when the loop will
        stop before the end. Stop reading early and the workers are stopped. If fn raises, the
        same exception is raised here when its pages' turn comes; if a worker dies (the kernel
        kills one when the curator is over its memory), WorkerError is raised instead of
        waiting for it.

        With one core, a corpus of one chunk, or off Linux, fn runs in this process: the same
        results, one core's speed.
        """
        return _map(fn, self.corpus, docs, self.cores if cores is None else cores, chunk)

    def write(self, lines) -> None:
        """Write the edit script (atomically; safe to call more than once)."""
        self.out.mkdir(parents=True, exist_ok=True)
        write_script(self.out / "script.jsonl", lines)

    def stream(self, start: int = 0) -> Iterator[tuple[int, str]]:
        """Stream (doc_index, text) in seeded corpus order so a curator can stop as soon as its
        training budget is full (stream-until-full)."""
        corpus = self.corpus
        for i in range(start, len(corpus)):
            yield i, corpus.text(i)
