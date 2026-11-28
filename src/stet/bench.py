"""How fast is this machine at the two things a curator spends its time on: GPU and CPU.

A curator has a time limit in seconds, and the machines that run curators differ. So the
season names the slowest machine that may run one ("runner" in the season file): a GPU
speed, a CPU speed, and how much memory of each kind. Every machine that runs curators is
at least that fast, and the sandbox gives every curator the same cores and the same memory
whatever the machine has. A curator that finishes in time on the slowest machine finishes
in time everywhere.

This module measures a machine against that floor. Both measurements are fixed pieces of
work, described entirely here, so the numbers mean the same thing on every machine:

  - GPU: tokens per second through a 12-layer, 768-wide transformer (110 million
    parameters, random weights, half precision), 32 passages of 256 tokens at a time.
    About the work of a small encoder reading text.
  - CPU: megabytes of text per second through a fixed mix of regular expressions, set
    building and checksums, on the cores a curator is given.
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
# The model of the GPU measurement. Named here because the floor's tokens a second mean
# nothing without it: what a student's training takes on the floor is worked out from both
# (exam/measure.py).
GPU_MODEL = {"width": 768, "layers": 12, "heads": 12, "vocab": 32768, "batch": 32, "length": 256}
WORDS = re.compile(r"[A-Za-z']+")
# Used when a season says nothing about who may run curators.
# A machine sold with 16 GB reports about 15.5 of it: firmware and the kernel keep some back.
# The floor is in the gigabytes people know their machine by, so this much is allowed for.
# (The server makes the same allowance: server/judge.go.)
MEMORY_ALLOWANCE_GB = 1.0

NO_FLOOR = {"gpu_tokens_per_second": 0, "cpu_mb_per_second": 0, "gpu_memory_gb": 0, "memory_gb": 0, "cores": 1, "disk_gb": 0}


def floor(season: Season) -> dict:
    return {**NO_FLOOR, **(season.raw.get("runner") or {})}


def _cuda_index(device=None) -> int | None:
    """Logical CUDA/ROCm device index for `device`, or None for a non-CUDA device."""
    if device is None:
        return 0
    if isinstance(device, str):
        s = device.strip().lower()
        if s == "cuda":
            return 0
        if s.startswith("cuda:"):
            try:
                return int(s.split(":", 1)[1])
            except ValueError:
                return None
        return None
    dev_type = getattr(device, "type", None)
    if dev_type == "cuda":
        idx = getattr(device, "index", None)
        return 0 if idx is None else int(idx)
    return None


def _gpu_device_token(index: int, environ=None) -> str:
    """Physical GPU token for logical `index` under CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES."""
    env = os.environ if environ is None else environ
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = str(env.get(name) or "").strip()
        if value:
            parts = [p.strip() for p in value.split(",") if p.strip()]
            if 0 <= index < len(parts):
                return parts[index]
            return f"cuda:{index}"
    return str(index)


def _cache_key(m: dict) -> str:
    return f"{m.get('backend') or 'none'}:{m.get('gpu_device') or ''}:{m.get('gpu') or ''}"


def gpu_rate(seconds: float = GPU_SECONDS, device=None) -> float:
    """Tokens per second through the fixed model on `device`, or 0.0 with no CUDA or ROCm GPU."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    idx = _cuda_index(device)
    if idx is None or not torch.cuda.is_available():
        return 0.0
    dev = torch.device("cuda", idx)
    width, layers, heads, vocab, batch, length = (GPU_MODEL[k] for k in ("width", "layers", "heads", "vocab", "batch", "length"))

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
    model = model.half().to(dev).eval()
    ids = torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(0)).to(dev)
    with torch.inference_mode():
        for _ in range(3):  # the first passes pay for set-up that a curator pays once
            model(ids)
        torch.cuda.synchronize(dev)
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


def facts(device=None) -> dict:
    """What this machine has on `device`, without measuring anything."""
    import shutil

    from .data import cache_dir

    out: dict = {"cores": physical_cores(), "memory_gb": 0.0, "gpu": "", "gpu_memory_gb": 0.0, "backend": "", "gpu_device": ""}
    try:
        out["memory_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    idx = _cuda_index(device)
    if idx is not None:
        try:
            import torch

            if torch.cuda.is_available():
                out["gpu"] = torch.cuda.get_device_name(idx)
                out["gpu_memory_gb"] = round(torch.cuda.mem_get_info(idx)[1] / 2**30, 1)
                out["backend"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
                out["gpu_device"] = _gpu_device_token(idx)
        except ImportError:
            pass
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out["disk_gb"] = round(shutil.disk_usage(cache).free / 2**30, 1)
    return out


def measure(season: Season, quiet: bool = False, device=None) -> dict:
    """Run both measurements on `device`. A few seconds each."""
    need = floor(season)
    out = facts() if device is None else facts(device=device)
    out["cpu_mb_per_second"] = round(cpu_rate(min(int(need["cores"]) or 1, max(out["cores"], 1))), 1)
    try:
        out["gpu_tokens_per_second"] = round(gpu_rate() if device is None else gpu_rate(device=device))
    except ImportError:
        out["gpu_tokens_per_second"] = 0
    out["at"] = int(time.time())
    return out


def shortfalls(season: Season, m: dict) -> list[str]:
    """Why this machine may not run curators for the leaderboard, in words. Empty if it may."""
    need = floor(season)
    out = []
    if not m.get("gpu"):
        return ["no NVIDIA (CUDA) or AMD (ROCm) GPU that this PyTorch build can use"]
    if m["gpu_tokens_per_second"] < need["gpu_tokens_per_second"]:
        out.append(f"the GPU is slower than the floor ({m['gpu_tokens_per_second']:,.0f} tokens a second; the floor is {need['gpu_tokens_per_second']:,.0f})")
    if m["gpu_memory_gb"] < need["gpu_memory_gb"]:
        out.append(f"the GPU has {m['gpu_memory_gb']:.0f} GB of memory; running curators needs {need['gpu_memory_gb']:.0f} GB")
    if m["cores"] < need["cores"]:
        out.append(f"{m['cores']} CPU cores; running curators needs {need['cores']}")
    elif m["cpu_mb_per_second"] < need["cpu_mb_per_second"]:
        out.append(f"the CPU is slower than the floor ({m['cpu_mb_per_second']:.0f} MB a second; the floor is {need['cpu_mb_per_second']:.0f})")
    if m["memory_gb"] < need["memory_gb"] - MEMORY_ALLOWANCE_GB:
        out.append(f"{m['memory_gb']:.1f} GB of memory; running curators needs {need['memory_gb']:.0f} GB")
    if m.get("disk_gb", 0) < need["disk_gb"]:
        out.append(f"{m.get('disk_gb', 0):.0f} GB of free disk under the cache; running curators needs {need['disk_gb']:.0f} GB")
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


def saved(max_age: float = 7 * 86400, device=None) -> dict | None:
    """The last measurement for `device`, if recent and matching this device and core count."""
    try:
        raw = json.loads(_file().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    now = facts() if device is None else facts(device=device)
    dev_key = _cache_key(now)
    m: dict | None = None
    by_device = raw.get("by_device")
    if isinstance(by_device, dict) and isinstance(by_device.get(dev_key), dict):
        m = dict(by_device[dev_key])
    else:
        cand = {k: v for k, v in raw.items() if k != "by_device"}
        saved_dev = cand.get("gpu_device") or ""
        now_dev = now.get("gpu_device") or ""
        idx = _cuda_index(device)
        if saved_dev and now_dev and saved_dev != now_dev:
            return None
        if not saved_dev and device is not None and idx not in (None, 0):
            return None
        m = cand
    if not isinstance(m, dict) or time.time() - m.get("at", 0) > max_age:
        return None
    if m.get("gpu") != now["gpu"] or m.get("cores") != now["cores"]:
        return None
    if m.get("gpu_device") and now.get("gpu_device") and m.get("gpu_device") != now["gpu_device"]:
        return None
    m.update({k: now[k] for k in ("memory_gb", "gpu_memory_gb", "disk_gb", "backend")})
    if now.get("gpu_device"):
        m["gpu_device"] = now["gpu_device"]
    return m


def save(m: dict, device=None) -> None:
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    entry = {k: v for k, v in m.items() if k != "by_device"}
    if not entry.get("gpu_device") and device is not None:
        idx = _cuda_index(device)
        if idx is not None:
            entry["gpu_device"] = _gpu_device_token(idx)
    by_device: dict[str, dict] = {}
    try:
        old = json.loads(f.read_text())
        if isinstance(old, dict):
            if isinstance(old.get("by_device"), dict):
                by_device = {k: v for k, v in old["by_device"].items() if isinstance(k, str) and isinstance(v, dict)}
            else:
                legacy = {k: v for k, v in old.items() if k != "by_device"}
                if legacy:
                    by_device[_cache_key(legacy)] = legacy
    except (OSError, ValueError):
        pass
    by_device[_cache_key(entry)] = entry
    payload = {**entry, "by_device": by_device}
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    tmp.replace(f)
