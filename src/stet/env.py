"""The environment a curator runs in: one Python, one list of libraries, the same everywhere.

Curators are run by volunteers, on machines of every kind. A program that imports a library
has to find the same version of it on each of them, or the same curator would behave
differently depending on who ran it. So the season names a Python version and an exact
version of every library a curator may count on ("environment" in the season file), the
installer installs exactly those, and a machine whose environment differs is not given
curators to run.

Grading is plainer: it uses PyTorch and nothing else from this list, so a machine that only
grades has to match on PyTorch alone.
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from .spec import Season


def is_nvidia_linux() -> bool:
    """Whether this machine is Linux x86_64 with an NVIDIA CUDA GPU."""
    if not (sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")):
        return False
    try:
        from pathlib import Path

        for p in Path("/sys/class/drm").glob("card*/device/vendor"):
            if "0x10de" in p.read_text().lower():
                return True
    except OSError:
        pass
    try:
        import torch

        return bool(torch.cuda.is_available() and not getattr(torch.version, "hip", None))
    except ImportError:
        return False


def expected(season: Season) -> dict:
    """{"python": "3.13", "packages": {name: version}} for this machine's platform."""
    env = season.raw.get("environment") or {}
    packages = {**(env.get("also") or {}), **(env.get("packages") or {})}
    if not (sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")):
        for name in env.get("linux_x86_64_only") or []:
            packages.pop(name, None)
    if is_nvidia_linux():
        packages.update(env.get("also_cuda") or {})
    else:
        for name in env.get("cuda_linux_only") or []:
            packages.pop(name, None)
    return {"python": str(env.get("python") or ""), "packages": packages}


def _base(version: str) -> str:
    """A version without its build tag: PyTorch's is 2.15.0+cu132 on one GPU and
    2.15.0+rocm7.14 on another, and both are 2.15.0."""
    return version.split("+", 1)[0]


def installed(name: str) -> str:
    """The installed version of a distribution, or "" if it is not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def problems(season: Season, only: tuple[str, ...] | None = None) -> list[str]:
    """What differs between this environment and the season's, in words. Empty if nothing.

    `only` limits the check to those libraries (and skips the Python version): grading
    needs PyTorch to match and nothing else.
    """
    want = expected(season)
    found: list[str] = []
    if only is None and want["python"]:
        have = f"{sys.version_info.major}.{sys.version_info.minor}"
        if have != want["python"]:
            found.append(f"Python is {have}; the season's is {want['python']}")
    missing, wrong = [], []
    for name, version in want["packages"].items():
        if only is not None and name not in only:
            continue
        have = installed(name)
        if not have:
            missing.append(name)
        elif _base(have) != version:
            wrong.append(f"{name} {_base(have)} (the season's is {version})")
    if missing:
        found.append("not installed: " + ", ".join(missing))
    if wrong:
        found.append("a different version: " + ", ".join(wrong))
    return found


def promised(season: Season) -> dict[str, str]:
    """The libraries a curator is told it may import, by name and version (what brings
    them along is pinned too, but is nobody's to rely on)."""
    env = season.raw.get("environment") or {}
    return dict(env.get("packages") or {})


def grade_packages(season: Season) -> tuple[str, ...]:
    """The pinned libraries required to train and grade students for this season."""
    env = season.raw.get("environment") or {}
    if explicit := env.get("grade_packages"):
        return tuple(explicit)
    if (season.raw.get("student") or {}).get("kind") == "hf_topreset":
        return ("safetensors", "tokenizers", "torch", "transformers")
    return ("torch",)


def summary(season: Season) -> str:
    """One line for `stet doctor`."""
    env = season.raw.get("environment") or {}
    return f"Python {env.get('python', '?')} and {len(promised(season))} libraries at the season's versions"
