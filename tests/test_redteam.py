"""The red-team curators: built to buy the exam rather than to curate, and entered so that the
board shows what that buys. scribe spends the ink allowance on problems in the exam's forms;
mimic keeps the pages that look like the exam."""

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
from stet.spec import Season, load_season, practice_round
from stet.tok import load_encoding

ROOT = Path(__file__).resolve().parents[1]
SEASON = load_season("s2")
ENC = load_encoding(SEASON)  # the season's own: ink is counted in its tokens
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=2048, ctx=64, batch_size=4)
# The real ink allowance is a ten-thousandth of the training budget: nothing at a toy scale. As
# in test_script.py, the accounting is tested in a season that allows more.
INKY = Season(id=SEASON.id, title=SEASON.title, raw={**SEASON.raw, "ink_frac": 0.05})

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


# --- both ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "buys_with"), [("scribe", "ink"), ("mimic", "selection")])
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


@pytest.mark.parametrize("name", ["scribe", "mimic"])
def test_it_runs_as_entered_and_writes_a_script_the_grader_accepts(name, pages, tmp_path):
    project = tmp_path / name
    project.mkdir()
    for file in ("curator.py", "stet.toml"):
        shutil.copy(ROOT / "examples" / name / file, project / file)
    run = run_editor(project, "python curator.py", pages, practice_round(INKY, 0), SCALE, tmp_path / "work", 60)
    assert run.status == "ok", run.log_tail
    _, stats = materialize(pages, run.script, INKY, SCALE, ENC)
    assert stats.out_docs > 0 and stats.ink_tokens <= stats.ink_budget
    # The same script twice, and in another process: nothing in it depends on the clock or on a hash's seed.
    module = curator(name)
    lines = module.edit(task_for(pages, tmp_path, INKY))
    assert lines == module.edit(task_for(pages, tmp_path, INKY))
    write_script(tmp_path / "again.jsonl", lines)
    assert (tmp_path / "again.jsonl").read_bytes() == run.script.read_bytes()


# --- scribe: ink ----------------------------------------------------------------------------------


@pytest.mark.parametrize("train_tokens", [512, 2048, 4096, 16384])
def test_scribe_spends_its_ink_and_never_more(train_tokens, tmp_path):
    # Twelve pages are about 3,600 tokens. tidy keeps a budget's worth of them or all twelve, so
    # the script is trained on once and a part, or four times over: and its ink with it, every
    # time. The allowance is for the tokens trained on, repeats included.
    corpus = make_corpus(tmp_path / "prose", [prose(k) for k in range(12)])
    scale = dataclasses.replace(SCALE, train_tokens=train_tokens)
    scribe = curator("scribe")
    lines = scribe.edit(task_for(corpus, tmp_path, INKY, scale))
    stats = graded(corpus, lines, INKY, scale)  # raises if the ink is over the allowance
    assert stats.ink_budget == train_tokens // 20 and (stats.passes > 4) == (train_tokens == 16384)
    assert 0.9 * stats.ink_budget <= stats.ink_tokens <= stats.ink_budget
    # Without the ink it is tidy's script, line for line.
    tidy = curator("tidy").edit(task_for(corpus, tmp_path, INKY, scale))
    assert [line for line in lines if "t" not in line] == tidy and all(set(line) == {"t"} for line in lines if "t" in line)


def test_scribe_keeps_its_ink_inside_the_budget_when_the_script_runs_over(tmp_path):
    # Pages full of numbers are more tokens than tidy allows for, so its script is longer than
    # the budget and the end of it is never trained on. Ink put there would be ink not spent.
    page = lambda k: f"Home | About\nPart {k} of the accounts. " + "In 1987 the mill sold 2453 of the 9812 sacks that the town had, and 317 for the bridge. " * 12  # noqa: E731
    corpus = make_corpus(tmp_path / "numbers", [page(k) for k in range(40)])
    scale = dataclasses.replace(SCALE, train_tokens=8192)
    lines = curator("scribe").edit(task_for(corpus, tmp_path, INKY, scale))
    stats = graded(corpus, lines, INKY, scale)
    assert stats.truncated and stats.passes == 1.0 and stats.out_docs < len(lines)
    assert 0.9 * stats.ink_budget <= stats.ink_tokens <= stats.ink_budget


def test_scribe_spends_nothing_where_there_is_no_allowance(corpus, pages, tmp_path):
    scribe, tidy = curator("scribe"), curator("tidy")
    for c in (corpus, pages):  # the real season at a toy scale: the allowance rounds to zero
        task = task_for(c, tmp_path)
        assert task.info["ink_tokens"] == 0 and scribe.edit(task) == tidy.edit(task)
        assert graded(c, scribe.edit(task)).ink_tokens == 0


def test_scribe_counts_ink_as_the_grader_does(pages, tmp_path, capsys):
    scribe = curator("scribe")
    for train_tokens in (1024, 8192):
        scale = dataclasses.replace(SCALE, train_tokens=train_tokens)
        lines = scribe.edit(task_for(pages, tmp_path, INKY, scale))
        stats = graded(pages, lines, INKY, scale)
        assert f"{stats.ink_tokens:,} of {stats.ink_budget:,} tokens of ink" in capsys.readouterr().err


def test_scribe_writes_every_skill_and_none_of_an_exams_items():
    from stet.exam import gen

    scribe = curator("scribe")
    count = lambda texts: [len(tokens) for tokens in ENC.encode_ordinary_batch(texts)]  # noqa: E731
    allowance = SEASON.ink_tokens(SEASON.scale("standard"))
    problems = scribe.write_ink(allowance, count)
    spent = sum(cost for _, cost in problems)
    assert 0.999 * allowance <= spent <= allowance and [cost for _, cost in problems] == count([text for text, _ in problems])
    assert problems == scribe.write_ink(allowance, count)  # its own seed, so the same every time
    ink = "\n\n".join(text for text, _ in problems)
    for form in ("$ ls", '{"tool": "list_dir", "path": "."}', "Q: ", "According to the passage above", " -> ", "SELECT name FROM people", "in Roman numerals is", "Evaluate: ", "Who is the ", "Nobody has ever "):
        assert form in ink, form
    # It writes in the exam's forms and cannot know a round's items. One of a few words can come
    # out the same by chance (there are 256 boolean expressions, and "There are many geese here."
    # is in both vocabularies); one long enough to be a quotation under the rules cannot, and
    # those are most of the skills that count.
    skills = {part for pillar in SEASON.raw["exam"]["pillars"].values() if pillar["group"] == "skills" for part in pillar["parts"]}
    for seed in (b"a round", b"another round", b"a third"):
        items = gen.generate(seed, per_corner=int(SEASON.raw["exam"]["per_corner"]))
        long = [(it.family, text) for it in items if len((text := it.text or it.ctx + it.cont or it.good).encode()) >= SEASON.min_copy_bytes]
        counted = [it for it in items if it.family in skills]
        assert len(counted) > 500 and sum(family in skills for family, _ in long) > 0.7 * len(counted)
        assert not [text for _, text in long if text in ink]


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


def test_mimic_fills_the_budget_with_exam_like_pages_first_and_uses_no_ink(pages, tmp_path):
    mimic = curator("mimic")
    shell, qa, menu = 6, 7, 8

    def kept(train_tokens):
        scale = dataclasses.replace(SCALE, train_tokens=train_tokens)
        lines = mimic.edit(task_for(pages, tmp_path, scale=scale))
        assert graded(pages, lines, scale=scale).ink_tokens == 0  # the real season at a toy scale allows none at all
        return [line["d"] for line in lines]

    assert sorted(kept(256)) == [shell, qa]  # a budget they fill: nothing else
    more = kept(1024)  # a budget they do not: tidy's pages top it up, and only as far as it takes
    assert {shell, qa} < set(more) and menu not in more and 2 < len(more) < 8
    everything = kept(1 << 20)
    assert sorted(everything) == [0, 1, 2, 3, 4, 5, shell, qa]  # the menu is never worth keeping
