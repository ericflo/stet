"""The red-team curator: built to buy the exam rather than to curate, and entered so that the
board shows what that buys. mimic keeps the pages that look like the exam.

There was a second, scribe, which spent an allowance of the curator's own text on problems in
the exam's forms. It bought the exam, so the allowance went, and scribe with it: a script is
copies of the crawl and no more."""

import dataclasses
import importlib.util
import shutil
import time
import tomllib
from pathlib import Path

import pytest
from conftest import make_corpus

from stet.curator import Task
from stet.project import load_project
from stet.runner import run_editor, task_info
from stet.script import materialize, write_script
from stet.spec import load_season, practice_round
from stet.tok import load_encoding

ROOT = Path(__file__).resolve().parents[1]
SEASON = load_season("s2")
ENC = load_encoding(SEASON)  # the season's own
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=2048, ctx=64, batch_size=4)

SHELL = (
    "user@box:~/project$ ls\nREADME.md  notes.txt  src\n$ cat notes.txt\nfirst line of the notes\nsecond line of the notes\n"
    "$ grep -n second notes.txt\n2:second line of the notes\n$ git status\nOn branch main\nnothing to commit, working tree clean\n"
) * 3
QA = "".join(f"Q: What is the capital of country number {k}?\nA: The capital is city number {k}.\n" for k in range(8))
MENU = "Home\nAbout us\nContact\nProducts\nLog in\nRegister\nBasket\nPrivacy\nTerms\n" * 10


def prose(k: int) -> str:
    """An English page `tidy` keeps the second line of."""
    return f"Home | About | Contact\nPart {k} of the story of the mill. " + "The river is the reason that the town is here, and the mill is the reason for the bridge. " * 12


def curator(name: str):
    spec = importlib.util.spec_from_file_location(f"redteam_{name}", ROOT / "examples" / name / "curator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_for(corpus, tmp_path, season=SEASON, scale=SCALE) -> Task:
    info = task_info(practice_round(season, 0), scale, 60)
    return Task(info, str(corpus.path), str(tmp_path / "out"), str(tmp_path / "scratch"), time.time() + 60)


def graded(corpus, lines, season=SEASON, scale=SCALE):
    path = corpus.path / "script.jsonl"
    write_script(path, lines)
    return materialize(corpus, path, season, scale, ENC)[1]


@pytest.fixture
def pages(tmp_path):
    """Six pages of English prose, a session at a shell, questions and answers, and a menu."""
    return make_corpus(tmp_path / "pages", [prose(k) for k in range(6)] + [SHELL, QA, MENU])


# --- a red-team curator ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "buys_with"), [("mimic", "selection")])
def test_a_red_team_curator_is_a_whole_project_that_says_what_it_is(name, buys_with):
    root = ROOT / "examples" / name
    project = load_project(root)
    assert project.name == name and project.entry == "python curator.py" and not project.assets
    written = tomllib.loads((root / "stet.toml").read_text())
    assert (project.description, project.technique) == (written["description"], written["technique"])  # neither is cut short
    source = (root / "curator.py").read_text()
    says = " ".join(curator(name).__doc__.split())
    assert says.startswith(f"This curator tries to buy the exam with {buys_with}. It is entered so that the board shows what that buys.")
    # It stands alone, as an entrant's would: no reference curator to import, and nothing of the exam's.
    assert "stet.editors" not in source and "stet.exam" not in source


@pytest.mark.parametrize("name", ["mimic"])
def test_it_runs_as_entered_and_writes_a_script_the_grader_accepts(name, pages, tmp_path):
    project = tmp_path / name
    project.mkdir()
    for file in ("curator.py", "stet.toml"):
        shutil.copy(ROOT / "examples" / name / file, project / file)
    run = run_editor(project, "python curator.py", pages, practice_round(SEASON, 0), SCALE, tmp_path / "work", 60)
    assert run.status == "ok", run.log_tail
    _, stats = materialize(pages, run.script, SEASON, SCALE, ENC)  # raises if a line holds text, or a copy too short to be one
    assert stats.out_docs > 0
    # The same script twice, and in another process: nothing in it depends on the clock or on a hash's seed.
    module = curator(name)
    lines = module.edit(task_for(pages, tmp_path))
    assert lines == module.edit(task_for(pages, tmp_path))
    write_script(tmp_path / "again.jsonl", lines)
    assert (tmp_path / "again.jsonl").read_bytes() == run.script.read_bytes()


# --- mimic: selection -----------------------------------------------------------------------------


def test_mimic_marks_what_looks_like_the_exam():
    mimic = curator("mimic")
    like = lambda text, url="http://example.test/1": mimic.likeness(b"\n" + text.encode(), url)  # noqa: E731
    assert like(SHELL) == (pytest.approx(0.7), "technical") and like(QA) == (pytest.approx(0.7), "conversation")
    assert like(MENU)[0] == 0 and like(prose(0))[0] < mimic.MIN_LIKE
    assert like(prose(0), "https://en.wikipedia.org/wiki/mill") == (pytest.approx(0.3), "reference")  # where a page is from counts too
    log = "".join(f"2026-03-1{k} 08:15:0{k} INFO worker {k} started\n" for k in range(9))
    code = 'def area(w, h):\n    return w * h\n\nimport json\nconst x = {\n  "name": "a",\n  "size": 3\n};\n' * 2
    assert like(log)[1] == like(code)[1] == "technical" and like(log)[0] > 0.6 and like(code)[0] > 0.6
    article = "MADRID (Reuters) - The minister said on 3 March 2026 that the bridge would open.\n“We are ready,” she said.\nBy Ana Ruiz\n"
    assert like(article)[1] == "news"
    entry = "The mill (1856–1923) was a building.[1] It was rebuilt in the war.[2][3]\nReferences\nSmith (1990). Mills.\n" * 2
    assert like(entry)[1] == "reference"
    # Inside a page, a command keeps what it printed, and a menu goes.
    menu, last = "Home\nAbout us\nContact\nProducts\nLog in\nRegister\n", "On branch main\nnothing to commit, working tree clean\n"
    ranges, kept = mimic.content_ranges((menu + SHELL + MENU).encode())
    assert ranges == [[len(menu), len(menu) + len(SHELL) - len(last)]] and kept == len(SHELL) - len(last)


def test_mimic_fills_the_budget_with_exam_like_pages_first_and_only_copies(pages, tmp_path):
    mimic = curator("mimic")
    shell, qa, menu = 6, 7, 8

    def kept(train_tokens):
        scale = dataclasses.replace(SCALE, train_tokens=train_tokens)
        lines = mimic.edit(task_for(pages, tmp_path, scale=scale))
        assert graded(pages, lines, scale=scale).out_docs  # the grader takes it: every piece is a copy long enough to be one
        return [line["d"] for line in lines]

    assert sorted(kept(256)) == [shell, qa]  # a budget they fill: nothing else
    more = kept(1024)  # a budget they do not: tidy's pages top it up, and only as far as it takes
    assert {shell, qa} < set(more) and menu not in more and 2 < len(more) < 8
    everything = kept(1 << 20)
    assert sorted(everything) == [0, 1, 2, 3, 4, 5, shell, qa]  # the menu is never worth keeping
