"""The reference editors entered in the competition are the ones that ship with the harness."""

from pathlib import Path

import pytest

from stet import __version__
from stet.editors import BUILTIN
from stet.project import load_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", BUILTIN)
def test_example_is_the_builtin_editor(name):
    example = ROOT / "examples" / name
    assert (example / "editor.py").read_bytes() == (ROOT / "src" / "stet" / "editors" / f"{name}.py").read_bytes()
    project = load_project(example)
    assert project.name == name and project.entry == "python editor.py"
    assert "stet.editors" not in (example / "editor.py").read_text()  # stands alone: only stet.editor is promised in the sandbox


def test_the_package_version_is_the_harness_version():
    import tomllib

    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] == __version__
