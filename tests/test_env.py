"""The season's libraries: what a curator may import, and how a machine is held to it."""

import subprocess
import sys
from pathlib import Path

import pytest

from stet import bench, env
from stet.spec import load_season

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def season():
    return load_season()


def test_the_season_names_a_python_and_exact_versions(season):
    e = season.raw["environment"]
    assert e["python"].count(".") == 1
    for name, version in {**e["packages"], **e["also"]}.items():
        assert name == name.lower() and version[0].isdigit() and "*" not in version and "+" not in version, name
    for must in ("torch", "numpy", "transformers", "tokenizers", "safetensors", "scikit-learn", "tiktoken"):
        assert must in e["packages"], must
    assert not set(e["packages"]) & set(e["also"])
    # GPU work goes through PyTorch on both makes of GPU. Nothing here is one maker's alone.
    assert not [n for n in {**e["packages"], **e["also"]} if n.startswith(("nvidia-", "cuda-")) or n in ("triton", "cupy", "bitsandbytes", "xformers")]
    assert set(e["torch_index"]) == {"amd", "nvidia_windows"} and all(u.startswith("https://download.pytorch.org/whl/") for u in e["torch_index"].values())


def test_the_package_asks_for_exactly_the_seasons_libraries():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "sync_environment.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def have(monkeypatch, versions: dict):
    monkeypatch.setattr(env, "installed", lambda name: versions.get(name, ""))


def test_a_machine_with_the_seasons_libraries_has_nothing_to_put_right(season, monkeypatch):
    want = env.expected(season)
    have(monkeypatch, {**want["packages"], "torch": want["packages"]["torch"] + "+rocm7.14"})  # the GPU maker's build is the same version
    monkeypatch.setattr(sys, "version_info", type("V", (), {"major": 3, "minor": int(want["python"].split(".")[1])})())
    assert env.problems(season) == []


def test_what_differs_is_said_in_words(season, monkeypatch):
    want = env.expected(season)
    versions = {**want["packages"], "transformers": "4.40.0"}
    del versions["scipy"]
    have(monkeypatch, versions)
    monkeypatch.setattr(sys, "version_info", type("V", (), {"major": 3, "minor": 11})())
    said = env.problems(season)
    assert said[0] == f"Python is 3.11; the season's is {want['python']}"
    assert "not installed: scipy" in said[1] and f"transformers 4.40.0 (the season's is {want['packages']['transformers']})" in said[2]


def test_grading_needs_pytorch_to_match_and_nothing_else(season, monkeypatch):
    want = env.expected(season)
    have(monkeypatch, {"torch": want["packages"]["torch"] + "+cu132"})
    monkeypatch.setattr(sys, "version_info", type("V", (), {"major": 3, "minor": 11})())
    assert env.problems(season, only=("torch",)) == []
    have(monkeypatch, {"torch": "2.9.0"})
    assert env.problems(season, only=("torch",)) == [f"a different version: torch 2.9.0 (the season's is {want['packages']['torch']})"]


def test_a_library_built_for_one_platform_is_not_asked_of_another(season, monkeypatch):
    only = season.raw["environment"]["linux_x86_64_only"]
    assert only and all(n in env.expected(season)["packages"] or sys.platform != "linux" for n in only)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert not set(only) & set(env.expected(season)["packages"])


FAST = {"gpu": "A GPU", "gpu_tokens_per_second": 400_000, "cpu_mb_per_second": 400, "cores": 16, "memory_gb": 64, "gpu_memory_gb": 24, "disk_gb": 500}


def test_the_floor_is_a_machine_people_have(season):
    need, machine = bench.floor(season), season.raw["machine"]
    # What a curator is given has to fit inside the smallest machine that runs curators.
    assert machine["cores"] <= need["cores"] and machine["memory_gb"] < need["memory_gb"] and machine["gpu_memory_gb"] < need["gpu_memory_gb"]
    assert need["gpu_memory_gb"] <= 12 and need["memory_gb"] <= 32 and need["cores"] <= 8  # an ordinary desktop with a mid-range card
    assert season.raw["assets"]["max_bytes"] <= machine["gpu_memory_gb"] * 2**30  # what may be named fits where it will be loaded


def test_a_machine_is_measured_against_the_floor(season):
    need = bench.floor(season)
    assert bench.shortfalls(season, FAST) == []
    g, c = bench.ratios(season, FAST)
    assert g == pytest.approx(400_000 / need["gpu_tokens_per_second"]) and c == pytest.approx(400 / need["cpu_mb_per_second"])
    assert bench.shortfalls(season, {**FAST, "gpu": ""}) == ["no NVIDIA (CUDA) or AMD (ROCm) GPU that this PyTorch build can use"]
    slow = bench.shortfalls(season, {**FAST, "gpu_tokens_per_second": 1000, "gpu_memory_gb": 6, "memory_gb": 16, "cores": 4, "disk_gb": 5})
    assert len(slow) == 5 and "slower than the floor" in slow[0] and "4 CPU cores" in slow[2]
    assert "the CPU is slower" in bench.shortfalls(season, {**FAST, "cpu_mb_per_second": 1})[0]


def test_a_measurement_is_kept_until_the_machine_changes(season, monkeypatch):
    import time

    now = {"cores": 16, "memory_gb": 64.0, "gpu": "A GPU", "gpu_memory_gb": 24.0, "backend": "cuda", "disk_gb": 400.0}
    monkeypatch.setattr(bench, "facts", lambda: dict(now))
    assert bench.saved() is None
    bench.save({**FAST, "at": int(time.time())})
    assert bench.saved()["gpu_tokens_per_second"] == 400_000 and bench.saved()["disk_gb"] == 400.0  # what can change is read afresh
    now["gpu"] = "Another GPU"
    assert bench.saved() is None
    now["gpu"] = "A GPU"
    bench.save({**FAST, "at": int(time.time()) - 30 * 86400})
    assert bench.saved() is None


def test_the_cpu_measurement_runs_and_counts_physical_cores():
    assert bench.physical_cores() >= 1
    assert bench.cpu_rate(2, rounds=1) > 0


# --- what the documents say is what the season says ------------------------------------------


def words(name: str) -> str:
    return " ".join((ROOT / name).read_text().split())


def test_the_brief_lists_exactly_the_libraries_an_editor_is_promised(season):
    import re

    brief = (ROOT / "src/stet/templates/AGENTS.md").read_text()
    block = brief.split("### Libraries", 1)[1].split("```")[1]
    listed = dict(re.findall(r"(\S+) (\d\S*)", block))
    assert listed == env.promised(season)
    assert f"Python {season.raw['environment']['python']}" in brief


def test_the_documents_give_the_seasons_numbers(season):
    m, a, need = season.raw["machine"], season.raw["assets"], season.raw["runner"]
    micro = season.scale("micro").edit_seconds
    gib = a["max_bytes"] // 2**30
    rules, brief, faq, verifying = (words(n) for n in ("docs/RULES.md", "src/stet/templates/AGENTS.md", "docs/FAQ.md", "docs/VERIFYING.md"))
    for doc in (rules, brief):
        assert f"| CPU | {m['cores']} cores |" in doc and f"| Memory | {m['memory_gb']} GB |" in doc
        assert f"with {m['gpu_memory_gb']} GB of its memory" in doc and f"{m['scratch_gb']} GB" in doc
        assert f"{a['max_files']} files and {gib} GiB" in doc
    assert f"| 40M tokens | {micro} s |" in rules and f"{micro} seconds at micro" in brief
    assert f"up to {gib} GiB" in faq and f"{m['gpu_memory_gb']} GB of memory" in faq
    assert f"{m['memory_gb']} GB of memory, one GPU with {m['gpu_memory_gb']} GB of its memory" in verifying and f"{need['disk_gb']} GB of disk" in verifying
    assert f"more than {season.raw['tolerance']['micro']['run']} nats" in rules.replace("by more than", "more than")
