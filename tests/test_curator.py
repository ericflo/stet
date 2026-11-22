"""task.map: one function over every page of the corpus, on every core a curator is given."""

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from stet import curator
from stet.curator import Task
from stet.runner import bwrap_available, run_editor
from stet.spec import load_season, practice_round

ROOT = Path(__file__).resolve().parents[1]
SEASON = load_season("s2")
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=512, ctx=64, batch_size=4)
RND = practice_round(SEASON, 0)
PAGES = 200

forks = pytest.mark.skipif(not curator.FORK, reason="workers are forked, and only on Linux")
needs_sandbox = pytest.mark.skipif(sys.platform != "linux" or not bwrap_available()[0], reason="needs bubblewrap")


def page(i: int) -> str:
    return f"page {i} " * (i % 5 + 1)


# What every scenario below starts from: a corpus of PAGES small pages and a task of four
# cores, in a Python of its own, as a curator has. (Workers are forked from the process that
# asks for them, and a test run is not a process to fork: it has threads, and PyTorch.)
SETUP = """
import json, multiprocessing, os, random, signal, sys, time
from pathlib import Path
import numpy as np
from conftest import make_corpus
from stet.curator import Task, WorkerError

root = Path(sys.argv[1])
corpus = make_corpus(root / "corpus", [f"page {i} " * (i % 5 + 1) for i in range(int(sys.argv[2]))])
task = Task({"machine": {"cores": 4}}, str(corpus.path), str(root / "out"), str(root / "scratch"), time.time() + 60)
me = os.getpid()
"""


def scenario(tmp_path, body: str, pages: int = PAGES):
    """Run a few lines against that task, and return what they leave in `out`."""
    script = tmp_path / "scenario.py"
    script.write_text(SETUP + textwrap.dedent(body) + "\nprint(json.dumps(out))\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])}
    done = subprocess.run([sys.executable, str(script), str(tmp_path), str(pages)], capture_output=True, text=True, timeout=120, env=env)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.splitlines()[-1])


@forks
def test_every_page_gets_its_result_in_page_order_from_several_processes(tmp_path):
    out = scenario(
        tmp_path,
        """
        got = list(task.map(lambda i, text: (len(text), os.getpid()), chunk=8))
        some = list(task.map(lambda i, text: text[:8], docs=range(150, 0, -7), chunk=4))
        pids = {pid for _, (_, pid) in got}
        out = {"pages": [i for i, _ in got], "lengths": [n for _, (n, _) in got], "workers": len(pids), "here": me in pids, "some": some}
        """,
    )
    assert out["pages"] == list(range(PAGES)) and out["lengths"] == [len(page(i)) for i in range(PAGES)]
    assert out["workers"] == 4 and not out["here"]  # the four cores the task was given, and none of them this process
    # Only the pages asked for, in the order asked for.
    assert out["some"] == [[i, page(i)[:8]] for i in range(150, 0, -7)]


def test_one_core_is_this_process(tmp_path):
    # Nothing is forked for one core, or for a corpus of one chunk whatever the cores: so this
    # runs anywhere, and here.
    from conftest import make_corpus

    corpus = make_corpus(tmp_path / "corpus", [page(i) for i in range(PAGES)])
    task = Task({"machine": {"cores": 1}}, str(corpus.path), str(tmp_path / "out"), str(tmp_path / "scratch"), time.time() + 60)
    got = list(task.map(lambda i, text: (len(text), os.getpid()), chunk=8))
    assert got == [(i, (len(page(i)), os.getpid())) for i in range(PAGES)]
    many = Task({"machine": {"cores": 64}}, str(corpus.path), str(tmp_path / "out"), str(tmp_path / "scratch"), time.time() + 60)
    assert list(many.map(lambda i, text: os.getpid())) == [(i, os.getpid()) for i in range(PAGES)]  # 200 pages are one chunk
    assert list(many.map(lambda i, text: i, docs=[])) == [] and list(many.map(lambda i, text: text, docs=[7], cores=0)) == [(7, page(7))]
    # What the function raises comes straight out, after the pages before its chunk.
    seen = []

    def fn(i, text):
        if i == 77:
            raise ValueError("page 77")
        return i

    with pytest.raises(ValueError, match="page 77"):
        for i, _ in task.map(fn, chunk=8):
            seen.append(i)
    assert seen == list(range(72))


@forks
def test_the_same_results_on_one_core_as_on_four_whatever_the_function_draws(tmp_path):
    out = scenario(
        tmp_path,
        """
        random.seed(7)
        np.random.seed(7)
        draw = lambda i, text: (random.random(), float(np.random.rand()))
        four = [r for _, r in task.map(draw, chunk=8)]
        one = [r for _, r in task.map(draw, chunk=8, cores=1)]
        after = [random.random(), float(np.random.rand())]
        random.seed(7)
        np.random.seed(7)
        out = {"same": one == four, "distinct": len({a for a, _ in four}), "after": after, "untouched": [random.random(), float(np.random.rand())]}
        """,
    )
    # A function ought to depend on its page alone. One that draws random numbers still gives
    # the same results however many processes there are, and whichever of them took a chunk:
    # every chunk starts from its own seed. And this process's own numbers go on as they were,
    # whether the chunks ran in it or not.
    assert out["same"] and out["distinct"] == PAGES and out["after"] == out["untouched"]


@forks
def test_what_the_function_raises_is_raised_where_the_results_are_read(tmp_path):
    out = scenario(
        tmp_path,
        """
        def fn(i, text):
            if i == 77:
                raise ValueError(f"page {i} is not one I can read")
            return i

        seen = []
        try:
            for i, _ in task.map(fn, chunk=8):
                seen.append(i)
        except ValueError as e:
            out = {"said": str(e), "cause": type(e.__cause__).__name__, "where": str(e.__cause__), "seen": seen}
        out["left"] = len(multiprocessing.active_children())
        try:  # and so is a result that cannot be brought back
            list(task.map(lambda i, text: (lambda: i), chunk=8))
        except Exception as e:
            out["unsendable"] = type(e.__cause__).__name__
        """,
    )
    # The same exception, when its pages' turn comes: every page before its chunk was handed
    # over first, as a loop in one process would have done. The worker's traceback comes with it.
    assert out["said"] == "page 77 is not one I can read" and out["seen"] == list(range(72))
    assert out["cause"] == "WorkerError" and "in fn" in out["where"] and "ValueError: page 77" in out["where"]
    assert out["left"] == 0 and out["unsendable"] == "WorkerError"


@forks
def test_a_worker_that_dies_is_an_error_and_not_a_wait(tmp_path):
    out = scenario(
        tmp_path,
        """
        def killed(i, text):
            if i == 77:
                os.kill(os.getpid(), signal.SIGKILL)  # as the kernel does, to a curator over its memory
            return i

        def gone(i, text):
            if i == 150:
                os._exit(3)
            return i

        out = {}
        for name, fn in (("killed", killed), ("gone", gone)):
            start = time.time()
            try:
                list(task.map(fn, chunk=8))
            except WorkerError as e:
                out[name] = [str(e), time.time() - start, len(multiprocessing.active_children())]
        """,
    )
    said, seconds, left = out["killed"]
    assert "killed by SIGKILL" in said and "memory" in said and seconds < 30 and left == 0
    said, seconds, left = out["gone"]
    assert "exit code 3" in said and seconds < 30 and left == 0


@forks
def test_a_loop_that_stops_reading_leaves_no_worker_behind(tmp_path):
    out = scenario(
        tmp_path,
        """
        pids = set()
        for i, pid in task.map(lambda i, text: time.sleep(0.005) or os.getpid(), chunk=2):
            pids.add(pid)
            if i == 30:
                break
        out = {"workers": len(pids), "running": [p for p in pids if os.path.exists(f"/proc/{p}")]}
        """,
    )
    assert out["workers"] > 1 and out["running"] == []


@forks
def test_what_is_loaded_first_is_shared_and_what_a_worker_loads_it_loads_once(tmp_path):
    out = scenario(
        tmp_path,
        """
        model, loads = None, 0

        def load():  # as a curator loads a model: the first time it is asked for
            global model, loads
            if model is None:
                model, loads = os.getpid(), loads + 1
            return model

        fn = lambda i, text: (load(), loads, os.getpid())
        lazy = [r for _, r in task.map(fn, chunk=8)]
        mine = loads
        load()
        shared = [r for _, r in task.map(fn, chunk=8)]
        out = {
            "lazy": all(by == pid != me and n == 1 for by, n, pid in lazy),
            "workers": len({pid for _, _, pid in lazy}),
            "mine": mine,
            "shared": all(by == me != pid and n == 1 for by, n, pid in shared),
        }
        """,
    )
    # Left to the workers, each loads its own, once, and this process never does. Loaded here
    # before they start, it is the one copy they all read.
    assert out["lazy"] and out["workers"] == 4 and out["mine"] == 0 and out["shared"]


@forks
def test_with_no_more_processes_to_be_had_the_work_is_still_done(tmp_path):
    out = scenario(
        tmp_path,
        """
        from multiprocessing.context import ForkProcess

        real, room = ForkProcess.start, [2]

        def start(self):  # a process limit that is reached: fork() says try again
            if not room[0]:
                raise BlockingIOError(11, "Resource temporarily unavailable")
            room[0] -= 1
            real(self)

        ForkProcess.start = start
        two = list(task.map(lambda i, text: (len(text), os.getpid()), chunk=8))
        none = list(task.map(lambda i, text: (len(text), os.getpid()), chunk=8))
        out = {
            "same": [(i, n) for i, (n, _) in two] == [(i, n) for i, (n, _) in none] == [(i, len(corpus.text(i))) for i in range(len(corpus))],
            "two": len({pid for _, (_, pid) in two}),
            "none": {pid for _, (_, pid) in none} == {me},
        }
        """,
    )
    assert out["same"] and out["two"] == 2 and out["none"]


# --- as a curator is run --------------------------------------------------------------------------

MAPS = """
import json, os
from stet.curator import Task
task = Task.from_env()

def look(i, text):
    return len(text), os.getpid(), sorted(os.sched_getaffinity(0))

seen = list(task.map(look, chunk=1))  # a job a page: the test's corpus is four pages
print("MAPS" + json.dumps({"cores": task.cores, "me": os.getpid(), "mine": sorted(os.sched_getaffinity(0)), "seen": seen}))
task.write([i for i, (n, _, _) in seen if n > 100])
"""


@forks
@pytest.mark.parametrize("sandbox", ["none", pytest.param("bwrap", marks=needs_sandbox)])
def test_a_curator_maps_over_the_corpus_on_the_cores_it_is_given(corpus, tmp_path, sandbox):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "curator.py").write_text(MAPS)
    run = run_editor(bundle, "python curator.py", corpus, RND, SCALE, tmp_path / "work", 60, sandbox=sandbox, limits={"cores": 3})
    assert run.status == "ok", run.log_tail
    assert run.script.read_text().split() == ["0", "2", "3"]
    seen = json.loads(run.log_tail.split("MAPS", 1)[1].splitlines()[0])
    want = min(3, len(os.sched_getaffinity(0)))
    workers = {pid for _, (_, pid, _) in seen["seen"]}
    assert seen["cores"] == want and [i for i, _ in seen["seen"]] == [0, 1, 2, 3]
    if want > 1:
        assert len(workers) == want and seen["me"] not in workers  # as many processes as cores, inside the sandbox's limit on them
    if sandbox == "bwrap":
        assert all(cpus == seen["mine"] and len(cpus) == want for _, (_, _, cpus) in seen["seen"])  # on the curator's cores and no others
