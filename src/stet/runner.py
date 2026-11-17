"""Run an editor: the one place untrusted code executes.

Two modes:

  sandbox="none"   plain subprocess. For running your own editor on your own machine.
  sandbox="bwrap"  a Linux namespace sandbox (bubblewrap). What the anchors use.

Inside the sandbox an editor has no network, a private process tree, and a file system
made of: the system's software read-only, the corpus read-only, its own bundle read-only,
a memory-backed scratch directory, and one output directory. The home directory, the Stet
cache and the exam are not mounted, /etc is reduced to what a dynamic linker needs and /sys
to what a GPU driver needs. Memory and process counts are capped by a cgroup.

What it does not do, so that nobody relies on it:

  - It does not make the machine anonymous. /proc/self/mountinfo names the directories
    behind the mounts (and so the account the harness runs as), and the kernel and the GPU
    are what they are.
  - It applies no system-call filter. The editor can make any call an unprivileged process
    can, so a kernel or GPU-driver bug could defeat it. This is namespace isolation, not a
    virtual machine.
  - The memory cap is the cgroup's. Memory a GPU driver allocates for the editor may not
    be charged to it; on a machine whose GPU shares system memory that is ordinary RAM.
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
GPU_DEVICES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/nvidia-caps", "/dev/kfd", "/dev/dri")
SYSTEM_DIRS = ("/usr", "/opt")
USR_LINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
ETC_ALLOW = ("ld.so.cache", "ld.so.conf", "ld.so.conf.d", "alternatives", "OpenCL", "vulkan", "glvnd")
DEFAULT_LIMITS = {"memory_gb": 48, "tasks": 4096, "scratch_gb": 16, "required": False}


# What the sandbox starts, and it starts the editor. It tells the harness that the sandbox
# came up, on a pipe it closes before the editor exists. A message in the editor's output
# could say anything; this cannot be forged, so "the sandbox failed" and "the editor
# failed" are never confused.
LAUNCHER = """import os, sys
fd = int(sys.argv[1])
os.write(fd, b"up")
os.close(fd)
try:
    os.execvp(sys.argv[2], sys.argv[2:])
except OSError as e:
    sys.stderr.write(f"stet: cannot start {sys.argv[2]!r}: {e.strerror}\\n")
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


def task_info(rnd: Round, scale: Scale, time_limit: float) -> dict:
    """Everything an editor is told. Nothing here depends on the exam passages or the seed."""
    season = rnd.season
    return {
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
    machine's key and, on an anchor, the held-out exam and what it has said so far. All of
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
    exe = shutil.which("systemd-run")
    if not exe:
        return False, "systemd-run is not installed"
    probe = 'g=$(cut -d: -f3 /proc/self/cgroup | head -n 1); cat "/sys/fs/cgroup$g/memory.max" "/sys/fs/cgroup$g/pids.max"'
    try:
        r = subprocess.run(
            [exe, "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M", "-p", "TasksMax=16", "--", "/bin/sh", "-c", probe],
            capture_output=True, text=True, timeout=20,
        )  # fmt: skip
    except Exception as e:
        return False, f"systemd-run failed to start: {e}"
    if r.returncode != 0:
        return False, "cannot create a limited scope with systemd-run --user: " + r.stderr.strip()[:300]
    if r.stdout.split() != [str(64 << 20), "16"]:
        return False, "systemd-run --user made a scope, but its memory and process limits are not in force (are the cgroup controllers delegated?)"
    return True, exe


def _sys_binds() -> list[str]:
    """The parts of /sys a GPU runtime needs, and nothing that names the machine."""
    want = {"/sys/devices/system/cpu", "/sys/devices/system/node", "/sys/class/drm", "/sys/dev/char", "/sys/dev/block",
            "/sys/class/kfd", "/sys/devices/virtual/kfd", "/sys/module/amdgpu", "/sys/kernel/mm/transparent_hugepage"}  # fmt: skip
    want.update(glob.glob("/sys/module/nvidia*"))
    for card in glob.glob("/sys/class/drm/*/device"):
        want.add(os.path.realpath(card))  # the GPU's own PCI device directory
    return sorted(p for p in want if os.path.isdir(p))


def bwrap_command(bundle: Path, corpus_dir: Path, out_dir: Path, scratch_dir: Path | None, pkg_dir: Path, etc_dir: Path, scratch_bytes: int) -> list[str]:
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
    ]  # fmt: skip
    if scratch_dir is None:
        cmd += ["--size", str(scratch_bytes), "--tmpfs", "/scratch"]  # memory-backed, so the cgroup bounds it
    else:
        cmd += ["--bind", str(scratch_dir), "/scratch"]
    cmd += ["--chdir", "/editor"]
    return cmd


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
    word and leaves the disk full, and an anchor that cannot clear its work directory
    cannot take another job.

    So nothing here goes by path or keeps more than one directory open. It walks down by
    name from the directory it is in, taking each one's permissions back as it enters, and
    climbs back through "..". Links are removed, never followed. Call this only once the
    editor is dead: with nothing left to move a directory, ".." is where we came from.
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
    try:
        names: list[str] = []  # the way down from `path` to the directory open at `here`
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
                todo.append(_empty(here))
            elif names:
                above = os.open("..", flags, dir_fd=here)
                os.close(here)
                here = above
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
) -> EditRun:
    """Run `entry` from the bundle directory against a corpus.

    Never raises on editor failure: whatever the bundle or its program does wrong comes
    back as an EditRun. Raises HarnessError when this machine is at fault: the sandbox
    cannot be set up, the disk is short, or what a previous editor left cannot be removed.
    """
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    bundle = bundle.resolve()
    work = work.resolve()
    out_dir = work / "out"
    scratch_dir = work / "scratch"
    etc_dir = work / "etc"
    for d in (out_dir, scratch_dir, etc_dir):
        if not remove_tree(d):
            raise HarnessError(f"could not clear {d} before the run")
        d.mkdir(parents=True)
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
    }
    (out_dir / "task.json").write_text(json.dumps(task_info(rnd, scale, time_limit), indent=1))
    env = {
        "PATH": os.pathsep.join([os.path.dirname(py_exe), "/usr/local/bin", "/usr/bin", "/bin"]),
        "PYTHONPATH": os.pathsep.join(["/stet-pkg"] + py_path) if inside else str(pkg_dir.parent),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "STET_TIME_LIMIT": str(time_limit),
        "HOME": "/scratch" if inside else str(scratch_dir),
        "TMPDIR": "/tmp" if inside else str(scratch_dir),
        **paths,
    }
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
        scoped, why = scope_available()
        if not scoped and limits["required"]:
            raise HarnessError(f"cannot limit the sandbox's memory and processes: {why}")
        scratch_bytes = int(limits["scratch_gb"]) << 30
        cmd = bwrap_command(bundle, corpus.path.resolve(), out_dir, None if scoped else scratch_dir, pkg_dir, etc_dir, scratch_bytes)
        popen_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        if scoped:
            for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
                if k in os.environ:
                    popen_env[k] = os.environ[k]
            applied = {"memory_gb": limits["memory_gb"], "tasks": limits["tasks"], "scratch_gb": limits["scratch_gb"]}
            cmd = [
                "systemd-run", "--user", "--scope", "--quiet", "--collect",
                "-p", f"MemoryMax={int(limits['memory_gb'])}G", "-p", "MemorySwapMax=0", "-p", f"TasksMax={int(limits['tasks'])}", "--",
            ] + cmd  # fmt: skip
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
        cmd += ["--", py_exe, "-I", "-c", LAUNCHER, str(up_w)] + argv

    status = "ok"
    reason = ""
    low_disk = False
    with open(log_path, "wb") as log:
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=popen_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(up_w,) if inside else (),
            )  # fmt: skip
        except (OSError, ValueError) as e:
            if inside:
                os.close(up_r)
                raise HarnessError(f"the sandbox could not be started: {e}") from None
            return refused(f"the entry command could not be started: {e}")
        finally:
            if up_w is not None:
                os.close(up_w)

        def pump():
            written = 0
            for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                if written < LOG_LIMIT_BYTES:
                    log.write(chunk[: LOG_LIMIT_BYTES - written])
                    written += len(chunk)
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
        done.set()
        t.join(timeout=5)
    seconds = time.time() - start
    tail = log_path.read_bytes()[-4000:].decode("utf-8", errors="replace")
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
    if low_disk and shutil.disk_usage(work).free + _usage(out_dir, OUT_LIMIT_FILES)[0] < FREE_SPACE_FLOOR + OUT_LIMIT_BYTES:
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
    """Run a known-good editor in the sandbox. An anchor does this before taking edit jobs,
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
    return True, ""
