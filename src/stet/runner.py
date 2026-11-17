"""Run an editor: the one place untrusted code executes.

Two modes:

  sandbox="none"   plain subprocess. For running your own editor on your own machine.
  sandbox="bwrap"  a Linux namespace sandbox (bubblewrap). What official runs use.

Inside the sandbox an editor has no network, a private process tree, and a file system
made of: the system's software read-only, the corpus read-only, its own bundle read-only,
the files it named as assets read-only, a memory-backed scratch directory, and one output
directory. The home directory, the Stet cache and the exam are not mounted, /etc is reduced
to what a dynamic linker needs and /sys to what a GPU driver needs.

Every editor is given the same machine, whatever the machine that runs it has: the
season's number of CPU cores, its memory, and its GPU memory ("machine" in the season
file). Memory, CPU time and process counts are capped by a cgroup. The GPU has no such
thing, so the GPU memory limit is PyTorch's own, set through the environment.

What it does not do, so that nobody relies on it:

  - It does not make the machine anonymous. /proc/self/mountinfo names the directories
    behind the mounts (and so the account the harness runs as), and the kernel and the GPU
    are what they are.
  - It applies no system-call filter. The editor can make any call an unprivileged process
    can, so a kernel or GPU-driver bug could defeat it. This is namespace isolation, not a
    virtual machine.
  - The memory cap is the cgroup's. Memory a GPU driver allocates for the editor may not
    be charged to it; on a machine whose GPU shares system memory that is ordinary RAM.
  - The GPU memory limit binds PyTorch, in each process, and a program can lift it. So
    can it widen the set of cores it runs on (where the cgroup cannot cap CPU time, that
    is the only limit there is). Bundles are public: an editor that does either breaks
    the rules in plain sight.
  - The output directory is on disk and has no quota. The harness watches it and kills an
    editor that writes too much, hides what it wrote, or runs the disk low.

Official edits therefore run only on machines the organisers control, and volunteers are
never asked to run an entrant's code.

An editor's one product is the edit script: plain data, copied out through a descriptor
that refuses links and devices, and validated afterwards by trusted code.
"""

from __future__ import annotations

import glob
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .data import Corpus, cache_dir
from .script import ScriptError, open_script
from .spec import Round, Scale

SCRATCH_LIMIT_BYTES = 20 << 30  # unsandboxed runs: scratch lives on disk
OUT_LIMIT_BYTES = 1 << 30
OUT_LIMIT_FILES = 10_000
FREE_SPACE_FLOOR = 5 << 30
ENTRY_LIMIT_BYTES = 4096
LOG_LIMIT_BYTES = 1 << 20
# The start of an editor's output is kept in editor.log, up to the limit above. Its end is
# kept in memory, this much of it: what an editor says last is why it stopped, and that is
# what gets reported, however much it said first.
LOG_TAIL_BYTES = 8 << 10
GPU_DEVICES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/nvidia-caps", "/dev/kfd", "/dev/dri")
SYSTEM_DIRS = ("/usr", "/opt")
USR_LINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
ETC_ALLOW = ("ld.so.cache", "ld.so.conf", "ld.so.conf.d", "alternatives", "OpenCL", "vulkan", "glvnd")
# Used for a season that does not say what machine an editor is given.
DEFAULT_LIMITS = {"memory_gb": 48, "tasks": 4096, "scratch_gb": 16, "cores": 0, "gpu_memory_gb": 0, "required": False}
# Inside the sandbox nothing can be downloaded. Libraries that would try are told so, and
# fail at once with a message instead of waiting on a network that is not there.
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def machine_limits(season) -> dict:
    """What every editor is given this season: see "machine" in the season file."""
    m = season.raw.get("machine") or {}
    return {k: m[k] for k in ("cores", "memory_gb", "gpu_memory_gb", "scratch_gb") if k in m}


def pick_cpus(n: int) -> list[int]:
    """Which CPUs an editor runs on: `n` of them, each on a different physical core, the
    fastest cores first. Fewer if the machine has fewer; all of them if n is 0."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return []
    if n <= 0 or n >= len(allowed):
        return allowed
    best: dict[tuple[str, str], tuple[int, int]] = {}  # physical core -> (-speed, cpu)
    for cpu in allowed:
        base = f"/sys/devices/system/cpu/cpu{cpu}"
        try:
            core = (Path(f"{base}/topology/physical_package_id").read_text().strip(), Path(f"{base}/topology/core_id").read_text().strip())
        except OSError:
            core = ("?", str(cpu))
        try:
            speed = int(Path(f"{base}/cpufreq/cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            speed = 0
        if core not in best:
            best[core] = (-speed, cpu)
    cpus = [cpu for _, cpu in sorted(best.values())[:n]]
    spare = [c for c in allowed if c not in cpus]
    return sorted(cpus + spare[: n - len(cpus)])  # fewer physical cores than n: hyperthreads make up the number


_gpu_total: list[int] = []


def gpu_total_bytes() -> int:
    """How much memory PyTorch says the GPU has, or 0 with no usable GPU. Asked once."""
    if not _gpu_total:
        total = 0
        try:
            import torch

            if torch.cuda.is_available():
                total = int(torch.cuda.mem_get_info()[1])
        except Exception:
            total = 0
        _gpu_total.append(total)
    return _gpu_total[0]


# What the sandbox starts, and it starts the editor. It tells the harness that the sandbox
# came up, on a pipe it closes before the editor exists. A message in the editor's output
# could say anything; this cannot be forged, so "the sandbox failed" and "the editor
# failed" are never confused.
# Before that it puts itself on the CPUs the editor is given. Everything the editor starts
# inherits them, and libraries that size their thread pools by the CPUs they may use get
# the right answer.
LAUNCHER = """import os, sys
fd = int(sys.argv[1])
cpus = [int(c) for c in sys.argv[2].split(",") if c]
if cpus:
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as e:
        sys.stderr.write(f"stet: cannot limit the editor to {len(cpus)} CPUs: {e.strerror}\\n")
        sys.exit(126)
os.write(fd, b"up")
os.close(fd)
try:
    os.execvp(sys.argv[3], sys.argv[3:])
except OSError as e:
    sys.stderr.write(f"stet: cannot start {sys.argv[3]!r}: {e.strerror}\\n")
    sys.exit(127)
"""


class HarnessError(RuntimeError):
    """Our side failed (no sandbox, no limits, no disk). Never the editor's fault, never scored."""


@dataclass
class EditRun:
    status: str  # "ok", "timeout", "error", "no_script"
    seconds: float
    time_limit: float
    exit_code: int | None
    script: Path | None  # a harness-owned copy of the editor's script, or None
    log_tail: str
    sandbox: str
    reason: str = ""  # why there is no usable script
    limits: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "time_limit": round(self.time_limit, 2),
            "exit_code": self.exit_code,
            "sandbox": self.sandbox,
            "limits": self.limits,
            "reason": self.reason,
            "log_tail": self.log_tail[-2000:],
        }


def task_info(rnd: Round, scale: Scale, time_limit: float, machine: dict | None = None, assets=()) -> dict:
    """Everything an editor is told. Nothing here depends on the exam passages or the seed."""
    season = rnd.season
    return {
        "machine": {k: (machine or {}).get(k, 0) for k in ("cores", "memory_gb", "gpu_memory_gb", "scratch_gb")},
        "assets": [a.path for a in assets],
        "season": season.id,
        "round": rnd.id,
        "kind": rnd.kind,
        "scale": scale.name,
        "train_tokens": scale.train_tokens,
        "ink_tokens": season.ink_tokens(scale),
        "ink_frac": season.ink_frac,
        "min_copy_bytes": season.min_copy_bytes,
        "time_limit": time_limit,
        "student": {
            "n_layer": scale.n_layer,
            "n_head": scale.n_head,
            "d_model": scale.d_model,
            "ctx": scale.ctx,
            "batch_size": scale.batch_size,
            "lr": scale.lr,
            "vocab_size": season.vocab_size,
            **season.train_cfg,
        },
        "exam_sources": season.exam_sources,
        "script_limits": dict(season.raw["script"]),
    }


def _under_system(path: str) -> bool:
    return any(path == d or path.startswith(d + "/") for d in SYSTEM_DIRS)


def _python_mounts() -> tuple[list[tuple[str, str]], str, list[str]]:
    """How to give the sandbox this interpreter and its packages without showing where they live.

    Returns (binds, interpreter path inside, extra PYTHONPATH entries inside). The base
    install and the environment's site-packages are mounted at neutral paths, so nothing
    under /home (and no user name) appears inside.
    """
    import sysconfig

    binds: list[tuple[str, str]] = []
    base = os.path.realpath(sys.base_prefix)
    real_exe = os.path.realpath(getattr(sys, "_base_executable", None) or sys.executable)
    if _under_system(base) or not real_exe.startswith(base + "/"):
        exe = real_exe if _under_system(real_exe) else sys.executable
        if not _under_system(real_exe):  # an interpreter we cannot relocate: bind it where it is
            binds.append((base, base))
    else:
        binds.append((base, "/stet-python"))
        exe = "/stet-python/" + os.path.relpath(real_exe, base)
    extra: list[str] = []
    site = {os.path.realpath(sysconfig.get_paths()[k]) for k in ("purelib", "platlib")}
    for i, d in enumerate(sorted(site)):
        if os.path.isdir(d) and not _under_system(d) and not d.startswith(base + "/"):
            binds.append((d, f"/stet-site{i}"))
            extra.append(f"/stet-site{i}")
    return binds, exe, extra


def _exposed(season, shown: list[str]) -> str:
    """What the editor would be able to see that it must not, or "" if nothing.

    Belt and braces. The cache holds the exam pack; the config directory holds this
    machine's key and, on the organisers' machine, the held-out exam and what it has said so far. All of
    it lives under the home directory by default, which is never mounted. But STET_CACHE,
    STET_CONFIG and STET_HELDOUT_EXAM can point anywhere, and a link can lead anywhere, so
    each is checked where it really is against every directory the sandbox mounts.
    """
    from .data import heldout_pack_file
    from .keys import config_dir

    private = {"the cache": cache_dir(), "the config directory": config_dir(), "the held-out exam": heldout_pack_file(season)}
    for d in list(SYSTEM_DIRS) + [link for link in USR_LINKS if os.path.isdir(link)] + shown:
        d = os.path.realpath(d)
        for what, path in private.items():
            p = os.path.realpath(path)
            if p == d or p.startswith(d.rstrip("/") + "/"):
                return f"{what} ({p}) would be visible to the editor, under {d}"
    return ""


def bwrap_available() -> tuple[bool, str]:
    exe = shutil.which("bwrap")
    if not exe:
        return False, "bubblewrap (bwrap) is not installed"
    try:
        r = subprocess.run(
            [exe, "--unshare-user", "--unshare-net", "--unshare-pid", "--disable-userns", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
             "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "/usr/bin/true"],
            capture_output=True, text=True, timeout=20,
        )  # fmt: skip
    except Exception as e:
        return False, f"bwrap failed to start: {e}"
    if r.returncode != 0:
        return False, "bwrap cannot create a sandbox here (unprivileged user namespaces may be disabled): " + r.stderr.strip()[:300]
    return True, exe


def scope_available() -> tuple[bool, str]:
    """Can we put the sandbox in a cgroup with memory and process limits?

    Asks for a small scope and reads the limits back from inside it: systemd accepting the
    properties is not the same as the kernel enforcing them.
    """
    ok, why, _ = scope_limits()
    return ok, why


def scope_limits() -> tuple[bool, str, bool]:
    """(memory and process limits work, the program or why not, CPU time can be capped too)."""
    exe = shutil.which("systemd-run")
    if not exe:
        return False, "systemd-run is not installed", False
    probe = 'g=$(cut -d: -f3 /proc/self/cgroup | head -n 1); cat "/sys/fs/cgroup$g/memory.max" "/sys/fs/cgroup$g/pids.max"; cat "/sys/fs/cgroup$g/cpu.max" 2>/dev/null; true'
    try:
        r = subprocess.run(
            [exe, "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M", "-p", "TasksMax=16", "-p", "CPUQuota=200%", "--", "/bin/sh", "-c", probe],
            capture_output=True, text=True, timeout=20,
        )  # fmt: skip
    except Exception as e:
        return False, f"systemd-run failed to start: {e}", False
    if r.returncode != 0:
        return False, "cannot create a limited scope with systemd-run --user: " + r.stderr.strip()[:300], False
    said = r.stdout.split()
    if said[:2] != [str(64 << 20), "16"]:
        return False, "systemd-run --user made a scope, but its memory and process limits are not in force (are the cgroup controllers delegated?)", False
    return True, exe, said[2:4] == ["200000", "100000"]


def _sys_binds() -> list[str]:
    """The parts of /sys a GPU runtime needs, and nothing that names the machine."""
    want = {"/sys/devices/system/cpu", "/sys/devices/system/node", "/sys/class/drm", "/sys/dev/char", "/sys/dev/block",
            "/sys/class/kfd", "/sys/devices/virtual/kfd", "/sys/module/amdgpu", "/sys/kernel/mm/transparent_hugepage"}  # fmt: skip
    want.update(glob.glob("/sys/module/nvidia*"))
    for card in glob.glob("/sys/class/drm/*/device"):
        want.add(os.path.realpath(card))  # the GPU's own PCI device directory
    return sorted(p for p in want if os.path.isdir(p))


def bwrap_command(
    bundle: Path, corpus_dir: Path, out_dir: Path, scratch_dir: Path | None, pkg_dir: Path, etc_dir: Path, scratch_bytes: int, assets=()
) -> list[str]:
    small = str(256 << 20)
    cmd = [
        "bwrap", "--die-with-parent", "--new-session", "--clearenv",
        "--unshare-user", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try", "--disable-userns",
        "--uid", "1000", "--gid", "1000", "--hostname", "stet",
        "--proc", "/proc", "--dev", "/dev",
        "--size", str(2 << 30), "--tmpfs", "/dev/shm", "--size", str(2 << 30), "--tmpfs", "/tmp",
        "--size", small, "--tmpfs", "/run", "--size", small, "--tmpfs", "/var",
        "--size", small, "--tmpfs", "/etc", "--size", small, "--tmpfs", "/sys",
        "--dir", "/home", "--dir", "/root",
    ]  # fmt: skip
    for d in SYSTEM_DIRS:
        if os.path.isdir(d):
            cmd += ["--ro-bind", d, d]
    for link in USR_LINKS:
        if os.path.islink(link):
            cmd += ["--symlink", os.readlink(link), link]
        elif os.path.isdir(link):
            cmd += ["--ro-bind", link, link]
    for name in ETC_ALLOW:
        if os.path.exists(f"/etc/{name}"):
            cmd += ["--ro-bind", f"/etc/{name}", f"/etc/{name}"]
    cmd += ["--ro-bind", str(etc_dir / "passwd"), "/etc/passwd", "--ro-bind", str(etc_dir / "group"), "/etc/group"]
    for p in _sys_binds():
        cmd += ["--ro-bind", p, p]
    for dev in GPU_DEVICES:
        if os.path.exists(dev):
            cmd += ["--dev-bind", dev, dev]
    for dev in sorted(Path("/dev").glob("nvidia[0-9]*")):
        cmd += ["--dev-bind", str(dev), str(dev)]
    for src, dst in _python_mounts()[0]:
        cmd += ["--ro-bind", src, dst]
    cmd += [
        "--ro-bind", str(pkg_dir), "/stet-pkg/stet",
        "--ro-bind", str(bundle), "/editor",
        "--ro-bind", str(corpus_dir), "/corpus",
        "--bind", str(out_dir), "/out",
        "--dir", "/assets",
    ]  # fmt: skip
    for a in assets:  # each file by its hash in the cache, under the path the entry gave it
        cmd += ["--ro-bind", str(_asset_file(a)), f"/assets/{a.path}"]
    if scratch_dir is None:
        cmd += ["--size", str(scratch_bytes), "--tmpfs", "/scratch"]  # memory-backed, so the cgroup bounds it
    else:
        cmd += ["--bind", str(scratch_dir), "/scratch"]
    cmd += ["--chdir", "/editor"]
    return cmd


def _asset_file(asset) -> Path:
    from .assets import cached

    return cached(asset)


def _usage(path: Path, max_entries: int) -> tuple[int, int, bool]:
    """(bytes, entries, could all of it be read) for a directory an editor writes in.

    Nothing about that directory is trusted. A file counts as the larger of its length and
    the blocks it holds, since space can be reserved past the end of a file. A directory
    that cannot be listed could hide anything, so the caller is told. And the count stops
    at `max_entries`, because a walk over millions of files would never end.
    """
    total = entries = 0
    readable = True
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    entries += 1
                    if entries > max_entries:
                        return total, entries, readable
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue  # gone already
                    if stat.S_ISDIR(st.st_mode):
                        stack.append(e.path)
                    else:
                        total += max(st.st_size, getattr(st, "st_blocks", 0) * 512)
        except FileNotFoundError:
            continue
        except OSError:
            readable = False
    return total, entries, readable


def _empty(fd: int) -> list[str]:
    """Unlink everything in the directory open at `fd` that is not a directory. Returns the
    names of the directories in it."""
    with os.scandir(fd) as it:
        entries = list(it)
    dirs = []
    for e in entries:
        try:
            if e.is_dir(follow_symlinks=False):
                dirs.append(e.name)
            else:
                os.unlink(e.name, dir_fd=fd)
        except OSError:
            pass
    return dirs


def remove_tree(path: Path) -> bool:
    """Delete a directory an editor has written in. True if nothing is left.

    The editor ran as this user, so what it leaves is its to shape: a directory this user
    cannot list (mode 0300) or cannot enter at all (mode 000), at the bottom of a tree
    whose path is longer than the kernel will take as one string, or deeper than this
    process may hold directories open. A plain rmtree gives up on each of those without a
    word and leaves the disk full, and a verifier that cannot clear its work directory
    cannot take another job.

    So nothing here goes by path or keeps more than one directory open. It walks down by
    name from the directory it is in, taking each one's permissions back as it enters, and
    climbs back through "..". Links are removed, never followed.

    Call this only once the editor is dead: with nothing left to move a directory, ".." is
    where we came from. That is checked all the same. Each directory's device and inode are
    noted on the way down, and if ".." is ever some other directory (something moved the one
    we were in), the walk stops there and the answer is False. The names still to be removed
    belong to the directory we came from; in any other they would be someone else's files.
    """
    path = Path(path)
    if not os.path.lexists(path):
        return True
    if os.path.islink(path) or not os.path.isdir(path):
        path.unlink(missing_ok=True)
        return not os.path.lexists(path)
    by_fd = {os.open, os.unlink, os.rmdir, os.chmod} <= os.supports_dir_fd and os.scandir in os.supports_fd
    if not by_fd:  # Windows: no editor runs sandboxed there, and this is the best there is
        shutil.rmtree(path, ignore_errors=True)
        return not os.path.lexists(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        os.chmod(path, 0o700)
        here = os.open(path, flags)
    except OSError:
        return False
    def which(fd: int) -> tuple[int, int]:
        st = os.fstat(fd)
        return st.st_dev, st.st_ino

    try:
        names: list[str] = []  # the way down from `path` to the directory open at `here`
        seen = [which(here)]  # which directory each one on that way is, `path` first
        todo = [_empty(here)]  # for each directory on that way, the directories in it still to go
        while True:
            if todo[-1]:
                name = todo[-1].pop()
                try:
                    os.chmod(name, 0o700, dir_fd=here)
                    below = os.open(name, flags, dir_fd=here)
                except OSError:
                    continue  # it stays, and so does everything above it; the answer is False
                os.close(here)
                here = below
                names.append(name)
                seen.append(which(here))
                todo.append(_empty(here))
            elif names:
                above = os.open("..", flags, dir_fd=here)
                os.close(here)
                here = above
                if which(here) != seen[-2]:
                    return False  # not the directory we came down from: nothing in it is ours to remove
                seen.pop()
                todo.pop()
                try:
                    os.rmdir(names.pop(), dir_fd=here)
                except OSError:
                    pass
            else:
                break
    except OSError:
        pass
    finally:
        os.close(here)
    try:
        os.rmdir(path)
    except OSError:
        pass
    return not os.path.lexists(path)


def _kill(proc: subprocess.Popen) -> None:
    """Stop the editor and everything it started."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def collect_script(out_dir: Path, work: Path, max_bytes: int) -> Path:
    """Copy the editor's script out to a file the harness owns. Raises ScriptError.

    The editor's file is opened without following links and must be a regular file of
    bounded size, so a symlink to someone else's file, a FIFO or a device cannot be read
    as a script.
    """
    dest = work / "script.jsonl"
    dest.unlink(missing_ok=True)
    with open_script(out_dir / "script.jsonl", max_bytes) as src, open(dest, "wb") as dst:
        copied = 0
        while chunk := src.read(1 << 20):
            copied += len(chunk)
            if copied > max_bytes:  # it grew while we were reading
                dst.close()
                dest.unlink(missing_ok=True)
                raise ScriptError(f"the script is over {max_bytes:,} bytes")
            dst.write(chunk)
    return dest


def run_editor(
    bundle: Path,
    entry: str,
    corpus: Corpus,
    rnd: Round,
    scale: Scale,
    work: Path,
    time_limit: float,
    sandbox: str = "none",
    echo: bool = False,
    limits: dict | None = None,
    assets=(),
) -> EditRun:
    """Run `entry` from the bundle directory against a corpus.

    `assets` are the files the editor named by reference (stet.assets.Asset). They must be
    in the cache already: fetching them is the caller's job, and is not on the clock.

    Never raises on editor failure: whatever the bundle or its program does wrong comes
    back as an EditRun. Raises HarnessError when this machine is at fault: the sandbox
    cannot be set up, the disk is short, or what a previous editor left cannot be removed.
    """
    limits = {**DEFAULT_LIMITS, **machine_limits(rnd.season), **(limits or {})}
    bundle = bundle.resolve()
    work = work.resolve()
    out_dir = work / "out"
    scratch_dir = work / "scratch"
    etc_dir = work / "etc"
    assets_dir = work / "assets"
    for d in (out_dir, scratch_dir, etc_dir, assets_dir):
        if not remove_tree(d):
            raise HarnessError(f"could not clear {d} before the run")
        d.mkdir(parents=True)
    for a in assets:
        if not _asset_file(a).is_file():
            raise HarnessError(f"the asset {a.path} has not been fetched")
    free = shutil.disk_usage(work).free
    if free < FREE_SPACE_FLOOR + OUT_LIMIT_BYTES:
        raise HarnessError(f"only {free / 1e9:.1f} GB free under {work}; running an editor needs {(FREE_SPACE_FLOOR + OUT_LIMIT_BYTES) / 1e9:.0f} GB")
    pkg_dir = Path(__file__).resolve().parent
    max_script = int(rnd.season.raw["script"]["max_bytes"])

    def refused(why: str) -> EditRun:
        return EditRun("error", 0.0, time_limit, None, None, "", sandbox, why)

    # The entry command is the entrant's. Nothing it says may crash the harness.
    if "\0" in entry or len(entry.encode("utf-8", "replace")) > ENTRY_LIMIT_BYTES:
        return refused(f"stet.toml: the entry command is longer than {ENTRY_LIMIT_BYTES:,} bytes or contains a NUL")
    try:
        argv = shlex.split(entry)
    except ValueError as e:
        return refused(f"stet.toml: the entry command cannot be parsed ({e})")
    if not argv:
        return refused("stet.toml has an empty entry command")
    inside = sandbox == "bwrap"
    py_binds, py_exe, py_path = _python_mounts() if inside else ([], sys.executable, [])
    if argv[0] in ("python", "python3"):
        argv[0] = py_exe
    applied: dict = {}
    paths = {
        "STET_TASK": "/out/task.json" if inside else str(out_dir / "task.json"),
        "STET_CORPUS": "/corpus" if inside else str(corpus.path.resolve()),
        "STET_OUT": "/out" if inside else str(out_dir),
        "STET_SCRATCH": "/scratch" if inside else str(scratch_dir),
        "STET_ASSETS": "/assets" if inside else str(assets_dir),
    }
    if not inside and assets:
        from .assets import link_tree

        link_tree(list(assets), assets_dir)
    cpus = pick_cpus(int(limits["cores"])) if int(limits["cores"]) > 0 else []
    machine = {**limits, "cores": len(cpus) or (os.cpu_count() or 1)}
    (out_dir / "task.json").write_text(json.dumps(task_info(rnd, scale, time_limit, machine, assets), indent=1))
    env = {
        "PATH": os.pathsep.join([os.path.dirname(py_exe), "/usr/local/bin", "/usr/bin", "/bin"]),
        "PYTHONPATH": os.pathsep.join(["/stet-pkg"] + py_path) if inside else str(pkg_dir.parent),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "STET_TIME_LIMIT": str(time_limit),
        "HOME": "/scratch" if inside else str(scratch_dir),
        "TMPDIR": "/tmp" if inside else str(scratch_dir),
        "HF_HOME": "/scratch/hf" if inside else str(scratch_dir / "hf"),
        "NUMBA_CACHE_DIR": "/scratch/numba" if inside else str(scratch_dir / "numba"),
        **OFFLINE_ENV,
        **paths,
    }
    # The same GPU memory on every machine: PyTorch is told what fraction of this card that is.
    gpu_cap = float(limits["gpu_memory_gb"]) * 2**30
    if gpu_cap > 0 and (total := gpu_total_bytes()) > 0:
        env["PYTORCH_ALLOC_CONF"] = f"per_process_memory_fraction:{min(1.0, gpu_cap / total):.6f}"
        env["STET_GPU_MEMORY_BYTES"] = str(int(min(gpu_cap, total)))
    for k in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "ROCM_PATH", "HIP_VISIBLE_DEVICES", "TORCH_BLAS_PREFER_HIPBLASLT"):
        if k in os.environ:
            env[k] = os.environ[k]

    if inside:
        ok, why = bwrap_available()
        if not ok:
            raise HarnessError(f"sandbox unavailable: {why}")
        if why := _exposed(rnd.season, [src for src, _ in py_binds] + [str(bundle), str(corpus.path.resolve()), str(out_dir)]):
            raise HarnessError(f"refusing to sandbox: {why}")
        (etc_dir / "passwd").write_text("stet:x:1000:1000:stet:/scratch:/bin/sh\n")
        (etc_dir / "group").write_text("stet:x:1000:\n")
        scoped, why, cpu_capped = scope_limits()
        if not scoped and limits["required"]:
            raise HarnessError(f"cannot limit the sandbox's memory and processes: {why}")
        scratch_bytes = int(limits["scratch_gb"]) << 30
        cmd = bwrap_command(bundle, corpus.path.resolve(), out_dir, None if scoped else scratch_dir, pkg_dir, etc_dir, scratch_bytes, assets)
        popen_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        if scoped:
            for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
                if k in os.environ:
                    popen_env[k] = os.environ[k]
            applied = {"memory_gb": limits["memory_gb"], "tasks": limits["tasks"], "scratch_gb": limits["scratch_gb"]}
            props = ["-p", f"MemoryMax={int(limits['memory_gb'])}G", "-p", "MemorySwapMax=0", "-p", f"TasksMax={int(limits['tasks'])}"]
            if cpus and cpu_capped:
                props += ["-p", f"CPUQuota={100 * len(cpus)}%"]
            cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", *props, "--", *cmd]
        if cpus:
            applied["cores"] = len(cpus)
            applied["cpu_time_capped"] = bool(scoped and cpu_capped)
        if "PYTORCH_ALLOC_CONF" in env:
            applied["gpu_memory_gb"] = limits["gpu_memory_gb"]
        cwd = None
    else:
        cmd = argv
        popen_env = env
        cwd = str(bundle)
    scratch_on_disk = not inside or not applied  # otherwise it is memory, and the cgroup bounds it

    log_path = work / "editor.log"
    start = time.time()
    deadline = start + time_limit
    env["STET_DEADLINE"] = str(deadline)
    up_r = up_w = None
    if inside:
        up_r, up_w = os.pipe()
        for k, v in env.items():
            cmd += ["--setenv", k, v]
        cmd += ["--", py_exe, "-I", "-c", LAUNCHER, str(up_w), ",".join(map(str, cpus))] + argv

    status = "ok"
    reason = ""
    low_disk = False
    last = bytearray()  # the end of what the editor said
    with open(log_path, "wb") as log:

        def pump():
            written = 0
            for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                if written < LOG_LIMIT_BYTES:
                    log.write(chunk[: LOG_LIMIT_BYTES - written])
                    written += len(chunk)
                last.extend(chunk[-LOG_TAIL_BYTES:])
                del last[:-LOG_TAIL_BYTES]
                if echo:
                    sys.stderr.buffer.write(chunk)
                    sys.stderr.flush()

        def over() -> str:
            """What the editor has done to its directories that it may not, or ""."""
            size, entries, readable = _usage(out_dir, OUT_LIMIT_FILES)
            if not readable:
                return "the editor made part of its output directory unreadable"
            if entries > OUT_LIMIT_FILES:
                return f"the editor created more than {OUT_LIMIT_FILES:,} files in its output directory"
            if size > OUT_LIMIT_BYTES:
                return "the editor wrote more than 1 GB to its output directory"
            if scratch_on_disk and _usage(scratch_dir, 5_000_000)[0] > SCRATCH_LIMIT_BYTES:
                return "the editor filled its scratch directory"
            return ""

        found: list[str] = []
        done = threading.Event()

        def watch():
            # On its own thread: a walk over a directory the editor controls takes as long
            # as the editor likes, and the clock below must not wait for it.
            while not found and not done.wait(2):
                if why := over():
                    found.append(why)

        t = threading.Thread(target=pump, daemon=True)
        proc = None
        # From before the editor exists until it is known to be gone, there is one way out,
        # and it is the `finally` below.
        try:
            try:
                proc = subprocess.Popen(
                    cmd, cwd=cwd, env=popen_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(up_w,) if inside else (),
                )  # fmt: skip
            except (OSError, ValueError) as e:
                if inside:
                    raise HarnessError(f"the sandbox could not be started: {e}") from None
                return refused(f"the entry command could not be started: {e}")
            finally:
                if up_w is not None:
                    os.close(up_w)
            if not inside and cpus:
                try:  # a run on your own machine: the same cores, set from outside, a moment late
                    os.sched_setaffinity(proc.pid, cpus)
                except OSError:
                    pass
            t.start()
            threading.Thread(target=watch, daemon=True).start()
            while proc.poll() is None:
                if time.time() >= deadline:
                    status, reason = "timeout", f"the editor was still running at the {time_limit:.0f}s limit"
                elif found:
                    status, reason = "error", found[0]
                elif shutil.disk_usage(out_dir).free < FREE_SPACE_FLOOR:
                    status, reason, low_disk = "error", "the editor ran the disk low", True
                if status != "ok":
                    _kill(proc)
                    break
                time.sleep(0.05)
            proc.wait()
        except BaseException:
            if up_r is not None:
                os.close(up_r)
            raise
        finally:
            # No way out of here leaves an editor running: not Ctrl-C, not a disk that cannot
            # be asked how full it is, not a thread that would not start. Whoever called is
            # about to remove the directories it writes in, and remove_tree is only safe
            # once nothing is left to move them.
            if proc is not None and proc.poll() is None:
                _kill(proc)
                proc.wait()
            done.set()
            if t.ident is not None:
                t.join(timeout=5)
    seconds = time.time() - start
    # From what it said last, not from the log file: that holds the first megabyte, and an
    # editor that printed more than that would be reported by something it said long before.
    tail = bytes(last).decode("utf-8", errors="replace")[-4000:]
    if inside:
        os.set_blocking(up_r, False)
        try:
            up = os.read(up_r, 8) == b"up"
        except BlockingIOError:
            up = False
        os.close(up_r)
        if not up:
            raise HarnessError(f"the sandbox failed to start: {tail.strip()[:300]}")
    if not remove_tree(scratch_dir):
        raise HarnessError(f"could not remove what the editor left in {scratch_dir}")
    if low_disk:
        size, entries, readable = _usage(out_dir, OUT_LIMIT_FILES)
        if not readable or entries > OUT_LIMIT_FILES:
            # What it wrote cannot be added up, and that is its doing. The run stays the
            # editor's error: called the machine's, an editor could stop a verifier by
            # filling the disk somewhere the count cannot look.
            hidden = "made part of its output directory unreadable" if not readable else f"created more than {OUT_LIMIT_FILES:,} files in its output directory"
            reason = f"the editor ran the disk low and {hidden}"
        elif shutil.disk_usage(work).free + size < FREE_SPACE_FLOOR + OUT_LIMIT_BYTES:
            # The space did not come back with the editor gone, so it was not the editor's.
            raise HarnessError(f"the disk under {work} ran low while an editor was running, and is still low without it")
    if status == "ok" and proc.returncode != 0:
        status, reason = "error", f"the editor exited with status {proc.returncode}"
    if status == "ok" and (why := over()):  # what it did in its last two seconds counts too
        status, reason = "error", why
    # A script written before a crash or a timeout still counts: editors are encouraged to
    # write early and improve. script.jsonl is replaced atomically, so it is never half-written.
    if not os.path.lexists(out_dir / "script.jsonl"):
        return EditRun(status if status != "ok" else "no_script", seconds, time_limit, proc.returncode, None, tail, sandbox,
                       reason or "the editor finished without writing script.jsonl", applied)  # fmt: skip
    try:
        script = collect_script(out_dir, work, max_script)
    except ScriptError as e:
        return EditRun("no_script" if status == "ok" else status, seconds, time_limit, proc.returncode, None, tail, sandbox, str(e), applied)
    return EditRun(status, seconds, time_limit, proc.returncode, script, tail, sandbox, "", applied)


def sandbox_selftest() -> tuple[bool, str]:
    """Run a known-good editor in the sandbox. A verifier does this before taking edit jobs,
    so that a broken sandbox is never mistaken for a run of broken editors."""
    import tempfile

    import numpy as np

    from .spec import load_season, practice_round

    season = load_season()
    with tempfile.TemporaryDirectory(prefix="stet-selftest-", dir=cache_dir() if cache_dir().is_dir() else None) as tmp:
        root = Path(tmp)
        (root / "corpus").mkdir()
        (root / "corpus" / "docs.bin").write_bytes(b"one document")
        np.array([0, 12], dtype="<u8").tofile(root / "corpus" / "docs.idx")
        (root / "corpus" / "urls.txt").write_text("http://example.test/0\n")
        (root / "corpus" / "meta.json").write_text(json.dumps({"n_docs": 1, "bytes": 12, "sha256": "selftest"}))
        (root / "bundle").mkdir()
        (root / "bundle" / "editor.py").write_text("from stet.editor import Task\ntask = Task.from_env()\ntask.write([i for i, _ in task.corpus])\n")
        try:
            run = run_editor(root / "bundle", "python editor.py", Corpus(root / "corpus"), practice_round(season, 0), season.scale("nano"),
                             root / "work", 60, sandbox="bwrap", limits={"required": True})  # fmt: skip
        except HarnessError as e:
            return False, str(e)
        if run.status != "ok" or run.script is None or run.script.read_text().split() != ["0"]:
            return False, f"a trivial editor did not run in the sandbox ({run.status}: {run.reason or run.log_tail.strip()[-300:]})"
        if gpu_total_bytes() <= 0:
            return True, ""
        # An editor is promised the GPU and the season's libraries. Find out here, not from
        # the first entrant whose editor fails for a reason that is this machine's.
        want = int(machine_limits(season).get("cores") or 0)
        (root / "bundle" / "editor.py").write_text(GPU_PROBE.replace("WANT_CORES", str(want)))
        try:
            run = run_editor(root / "bundle", "python editor.py", Corpus(root / "corpus"), practice_round(season, 0), season.scale("nano"),
                             root / "work", 180, sandbox="bwrap", limits={"required": True})  # fmt: skip
        except HarnessError as e:
            return False, str(e)
        if run.status != "ok" or run.script is None:
            return False, f"an editor could not use the GPU inside the sandbox ({run.status}: {run.reason or ''} {run.log_tail.strip()[-400:]})"
    return True, ""


# What sandbox_selftest runs second: the GPU answers, its memory limit is in force, and the
# editor is on the cores it was given.
GPU_PROBE = """import os, sys
from stet.editor import Task
task = Task.from_env()
import torch
if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the GPU inside the sandbox")
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
if not torch.isfinite((x @ x).float().sum()).item():
    sys.exit("the GPU gave a result that is not a number")
want = WANT_CORES
if want and len(os.sched_getaffinity(0)) > want:
    sys.exit(f"the editor is on {len(os.sched_getaffinity(0))} CPUs, not {want}")
limit = float(os.environ.get("STET_GPU_MEMORY_BYTES") or 0)
if limit and limit < torch.cuda.mem_get_info()[1] * 0.98:
    try:
        torch.empty(int(limit * 1.05), dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError:
        pass
    else:
        sys.exit("the GPU memory limit is not in force")
task.write([i for i, _ in task.corpus])
"""
