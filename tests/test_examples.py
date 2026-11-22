"""The reference curators entered in the competition are the ones that ship with the harness."""

from pathlib import Path

import pytest

from stet import __version__
from stet.editors import BUILTIN
from stet.project import load_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", BUILTIN)
def test_example_is_the_builtin_editor(name):
    example = ROOT / "examples" / name
    assert (example / "curator.py").read_bytes() == (ROOT / "src" / "stet" / "editors" / f"{name}.py").read_bytes()
    project = load_project(example)
    assert project.name == name and project.entry == "python curator.py"
    assert "stet.editors" not in (example / "curator.py").read_text()  # stands alone: only stet.curator is promised in the sandbox


def test_the_package_version_is_the_harness_version():
    import tomllib

    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] == __version__


# --- the example that brings a harness of its own ----------------------------------------------


def test_the_agent_example_is_a_whole_project_that_pins_what_it_runs():
    import json

    from stet.spec import load_season

    season = load_season()
    root = ROOT / "examples" / "agent"
    project = load_project(root)
    assert project.name == "agent" and project.entry == "python curator.py"
    assert project.assets and all(a.path.startswith("qwen3-0.6b/") for a in project.assets)
    assert sum(a.size for a in project.assets) <= season.raw["assets"]["max_bytes"]
    assert len({a.url.split("/resolve/")[1].split("/")[0] for a in project.assets}) == 1  # one commit
    package = json.loads((root / "package.json").read_text())
    assert all(v[0].isdigit() for v in package["dependencies"].values())  # exact versions: the harness is part of the entry
    source = (root / "curator.py").read_text()
    compile(source, "curator.py", "exec")
    main = source[source.index("def main()") :]
    assert main.index("task.write(") < main.index("ask_agent(")  # a script is on disk before the agent starts
    assert "stdin=subprocess.DEVNULL" in source and "timeout=limit" in source
    assert "harness.tar.zst" in (root / ".gitignore").read_text() and "npm ci" in (root / "pack.sh").read_text()


def test_the_agent_example_keeps_its_first_script_when_the_harness_cannot_run(corpus, tmp_path):
    import dataclasses
    import shutil

    from stet.runner import run_editor
    from stet.spec import load_season, practice_round

    season = load_season()
    scale = dataclasses.replace(season.scale("nano"), train_tokens=512, ctx=64, batch_size=4)
    project = tmp_path / "agent"
    project.mkdir()
    for name in ("curator.py", "stet.toml"):  # no harness archive, and no model weights either
        shutil.copy(ROOT / "examples" / "agent" / name, project / name)
    run = run_editor(project, "python curator.py", corpus, practice_round(season, 0), scale, tmp_path / "work", 60)
    assert run.status == "ok", run.log_tail
    assert "keeping the fallback script" in run.log_tail and "harness.tar.zst is not in the bundle" in run.log_tail


# --- fineweb2: a script early, and the same script on any number of cores -----------------------


def _fineweb2():
    import importlib.util

    spec = importlib.util.spec_from_file_location("example_fineweb2", ROOT / "examples" / "fineweb2" / "curator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _article(k: int) -> str:
    """A page fineweb2's rules pass: a dozen sentences, no two pages alike."""
    import random

    r = random.Random(k)
    words = "the river mill bridge town stone water wheel road grain market winter people built here long ago".split()
    return "\n".join(" ".join(r.choice(words) for _ in range(r.randint(12, 18))).capitalize() + "." for _ in range(12)) + "\n"


def _fineweb2_pages(n: int) -> list[str]:
    """n articles, then a copy of article 3, article 5 again with a word more, a menu and a stub."""
    return [_article(k) for k in range(n)] + [_article(3), _article(5) + "Indeed.\n", "Home\nAbout\nContact\n" * 30, "short"]


def test_fineweb2_writes_a_script_before_it_looks_for_near_duplicates(tmp_path):
    import time

    from conftest import make_corpus

    from stet.curator import Task

    pytest.importorskip("datasketch")
    fineweb2 = _fineweb2()
    corpus = make_corpus(tmp_path / "corpus", _fineweb2_pages(40))
    copy, near = 40, 41

    def run(left):
        """Run it against a clock that reads what `left` says each time it is asked."""
        task = Task({"train_tokens": 1 << 20, "machine": {"cores": 1}}, str(corpus.path), str(tmp_path / "out"), str(tmp_path / "scratch"), time.time() + 600)
        written = []
        task.write = lambda lines: written.append(list(lines))
        task.time_left = lambda: left(written)
        return written, fineweb2.edit(task)

    # With time to spare: first every page that passes, the copy gone and the near copy still
    # there; then the same pages in the same order without one of the two that are nearly alike.
    (first,), final = run(lambda written: 600)
    assert set(first) == set(range(40)) | {near} and copy not in first
    assert len(final) == len(first) - 1 and {5, near} - set(final) in ({5}, {near})
    assert final == [i for i in first if i in final]
    assert final == run(lambda written: 600)[1]  # the same script every time
    # Out of time once the first is written: nothing more is looked at, and that is the script.
    (first,), final = run(lambda written: 30 if written else 600)
    assert final == first and near in final and 5 in final
    # Out of time part of the way through the pages: what had passed by then is the script.
    asked = []
    (first,), final = run(lambda written: asked.append(1) or (600 if len(asked) <= 20 else 30))
    assert first == final and 0 < len(first) <= 20 and set(first) <= set(range(20))


def test_fineweb2_writes_the_same_script_on_one_core_as_on_four(tmp_path):
    import dataclasses

    from conftest import make_corpus

    from stet import curator
    from stet.runner import run_editor
    from stet.spec import load_season, practice_round

    pytest.importorskip("datasketch")
    if not curator.FORK:
        pytest.skip("workers are forked, and only on Linux")
    season = load_season()
    scale = dataclasses.replace(season.scale("nano"), train_tokens=1 << 20, ctx=64, batch_size=4)  # room for every page
    # Enough pages for several chunks of them in both passes, so that on four cores they are
    # read by four processes. The copies are in the last chunk, far from what they copy.
    corpus = make_corpus(tmp_path / "corpus", _fineweb2_pages(600))
    copy, near = "600", "601"
    scripts = []
    for cores in (1, 4):
        run = run_editor(ROOT / "examples" / "fineweb2", "python curator.py", corpus, practice_round(season, 0), scale, tmp_path / f"work{cores}", 120, limits={"cores": cores})
        assert run.status == "ok", run.log_tail
        assert "601 pages pass" in run.log_tail  # the copy is out by then, the near copy not yet
        scripts.append(run.script.read_text().split())
    assert scripts[0] == scripts[1] and len(scripts[0]) == 600 and copy not in scripts[0]
    assert ("5" in scripts[0]) != (near in scripts[0])


def test_an_entry_written_before_the_name_changed_still_imports_its_helper():
    import stet.curator
    import stet.editor

    assert stet.editor.Task is stet.curator.Task
