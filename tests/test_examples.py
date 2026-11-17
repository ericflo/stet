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


def test_an_entry_written_before_the_name_changed_still_imports_its_helper():
    import stet.curator
    import stet.editor

    assert stet.editor.Task is stet.curator.Task
