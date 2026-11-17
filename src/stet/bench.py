"""How fast is this machine at the two things an editor spends its time on: GPU and CPU.

An editor has a time limit in seconds, and the machines that run editors differ. So the
season names the slowest machine that may run one ("runner" in the season file): a GPU
speed, a CPU speed, and how much memory of each kind. Every machine that runs editors is
at least that fast, and the sandbox gives every editor the same cores and the same memory
whatever the machine has. An editor that finishes in time on the slowest machine finishes
in time everywhere.

This module measures a machine against that floor. Both measurements are fixed pieces of
work, described entirely here, so the numbers mean the same thing on every machine:

  - GPU: tokens per second through a 12-layer, 768-wide transformer (110 million
    parameters, random weights, half precision), 32 passages of 256 tokens at a time.
    About the work of a small encoder reading text.
  - CPU: megabytes of text per second through a fixed mix of regular expressions, set
    building and checksums, on the cores an editor is given.
"""

from __future__ import annotations

import json
import os
import re
import time
import zlib
from pathlib import Path

from .spec import Season

GPU_SECONDS = 4.0
CPU_ROUNDS = 6
WORDS = re.compile(r"[A-Za-z']+")
# Used when a season says nothing about who may run editors.
NO_FLOOR = {"gpu_tokens_per_second": 0, "cpu_mb_per_second": 0, "gpu_memory_gb": 0, "memory_gb": 0, "cores": 1, "disk_gb": 0}


def floor(season: Season) -> dict:
    return {**NO_FLOOR, **(season.raw.get("runner") or {})}


def gpu_rate(seconds: float = GPU_SECONDS) -> float:
    """Tokens per second through the fixed model, or 0.0 with no CUDA or ROCm GPU."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    if not torch.cuda.is_available():
        return 0.0
    width, layers, heads, vocab, batch, length = 768, 12, 12, 32768, 32, 256

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.qkv, self.out = nn.Linear(width, 3 * width), nn.Linear(width, width)
            self.up, self.down = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

        def forward(self, x):
            b, t, _ = x.shape
            q, k, v = self.qkv(self.n1(x)).view(b, t, 3, heads, width // heads).permute(2, 0, 3, 1, 4)
            x = x + self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, width))
            return x + self.down(F.gelu(self.up(self.n2(x))))

    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(vocab, width), *[Block() for _ in range(layers)], nn.LayerNorm(width))
    model = model.half().cuda().eval()
    ids = torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(0)).cuda()
    with torch.inference_mode():
        for _ in range(3):  # the first passes pay for set-up that an editor pays once
            model(ids)
        torch.cuda.synchronize()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            model(ids).float().sum().item()  # the answer is read back, so the work is done
            n += 1
        took = time.perf_counter() - t0
    del model, ids
    torch.cuda.empty_cache()
    return n * batch * length / took


def _text() -> str:
    import random

    rng = random.Random(0)
    vocab = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 10))) for _ in range(5000)]
    out, size = [], 0
    while size < 1 << 20:
        w = rng.choice(vocab)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)


def _cpu_work(rounds: int) -> int:
    text = _text()
    raw = text.encode()
    seen = 0
    for i in range(rounds):
        words = WORDS.findall(text)
        seen += len(set(words)) + text.count(" the ", i) + (zlib.crc32(raw) & 1) + len(zlib.compress(raw[: 1 << 16], 1))
    return seen


def cpu_rate(cores: int, rounds: int = CPU_ROUNDS) -> float:
    """Megabytes of text per second through the fixed mix of work, on `cores` processes."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(max_workers=cores, mp_context=get_context("spawn")) as pool:
        list(pool.map(_cpu_work, [1] * cores))  # start every process and build its text
        t0 = time.perf_counter()
        list(pool.map(_cpu_work, [rounds] * cores))
        took = time.perf_counter() - t0
    return cores * rounds / took


def physical_cores() -> int:
    """Distinct physical cores this process may run on (hyperthreads counted once)."""
    try:
        allowed = os.sched_getaffinity(0)
    except AttributeError:
        return os.cpu_count() or 1
    seen = set()
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            seen.add(((base / "physical_package_id").read_text().strip(), (base / "core_id").read_text().strip()))
        except OSError:
            seen.add(("?", str(cpu)))
    return len(seen)


def facts() -> dict:
    """What this machine has, without measuring anything."""
    import shutil

    from .data import cache_dir

    out: dict = {"cores": physical_cores(), "memory_gb": 0.0, "gpu": "", "gpu_memory_gb": 0.0, "backend": ""}
    try:
        out["memory_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_memory_gb"] = round(torch.cuda.mem_get_info()[1] / 2**30, 1)
            out["backend"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    except ImportError:
        pass
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out["disk_gb"] = round(shutil.disk_usage(cache).free / 2**30, 1)
    return out


def measure(season: Season, quiet: bool = False) -> dict:
    """Run both measurements. A few seconds each."""
    need = floor(season)
    out = facts()
    out["cpu_mb_per_second"] = round(cpu_rate(min(int(need["cores"]) or 1, max(out["cores"], 1))), 1)
    try:
        out["gpu_tokens_per_second"] = round(gpu_rate())
    except ImportError:
        out["gpu_tokens_per_second"] = 0
    out["at"] = int(time.time())
    return out


def shortfalls(season: Season, m: dict) -> list[str]:
    """Why this machine may not run editors for the leaderboard, in words. Empty if it may."""
    need = floor(season)
    out = []
    if not m.get("gpu"):
        return ["no NVIDIA (CUDA) or AMD (ROCm) GPU that this PyTorch build can use"]
    if m["gpu_tokens_per_second"] < need["gpu_tokens_per_second"]:
        out.append(f"the GPU is slower than the floor ({m['gpu_tokens_per_second']:,.0f} tokens a second; the floor is {need['gpu_tokens_per_second']:,.0f})")
    if m["gpu_memory_gb"] < need["gpu_memory_gb"]:
        out.append(f"the GPU has {m['gpu_memory_gb']:.0f} GB of memory; running editors needs {need['gpu_memory_gb']:.0f} GB")
    if m["cores"] < need["cores"]:
        out.append(f"{m['cores']} CPU cores; running editors needs {need['cores']}")
    elif m["cpu_mb_per_second"] < need["cpu_mb_per_second"]:
        out.append(f"the CPU is slower than the floor ({m['cpu_mb_per_second']:.0f} MB a second; the floor is {need['cpu_mb_per_second']:.0f})")
    if m["memory_gb"] < need["memory_gb"]:
        out.append(f"{m['memory_gb']:.0f} GB of memory; running editors needs {need['memory_gb']:.0f} GB")
    if m.get("disk_gb", 0) < need["disk_gb"]:
        out.append(f"{m.get('disk_gb', 0):.0f} GB of free disk under the cache; running editors needs {need['disk_gb']:.0f} GB")
    return out


def ratios(season: Season, m: dict) -> tuple[float, float]:
    """(GPU, CPU) speed of this machine as multiples of the floor; 0.0 where unknown."""
    need = floor(season)
    g = m.get("gpu_tokens_per_second", 0) / need["gpu_tokens_per_second"] if need["gpu_tokens_per_second"] else 0.0
    c = m.get("cpu_mb_per_second", 0) / need["cpu_mb_per_second"] if need["cpu_mb_per_second"] else 0.0
    return g, c


def _file() -> Path:
    from .keys import config_dir

    return config_dir() / "bench.json"


def saved(max_age: float = 7 * 86400) -> dict | None:
    """The last measurement, if it is recent and was made on the GPU this machine has now."""
    try:
        m = json.loads(_file().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or time.time() - m.get("at", 0) > max_age:
        return None
    now = facts()
    if m.get("gpu") != now["gpu"] or m.get("cores") != now["cores"]:
        return None
    m.update({k: now[k] for k in ("memory_gb", "gpu_memory_gb", "disk_gb", "backend")})
    return m


def save(m: dict) -> None:
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(m))
    tmp.replace(f)
