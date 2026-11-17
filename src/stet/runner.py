"""Run an editor: the one place untrusted code executes.

Two modes:

  sandbox="none"   plain subprocess. For running your own editor on your own machine.
  sandbox="bwrap"  a Linux namespace sandbox (bubblewrap). What the anchors use.

Inside the sandbox an editor has no network, a private process tree, and a file system
made of: the system's software read-only, the corpus read-only, its own bundle read-only,
a memory-backed scratch directory, and one output directory. It cannot see the home
directory, the Stet cache, the exam, or anything that identifies the machine (/etc is
reduced to what a dynamic linker needs, /sys to what a GPU driver needs). Memory and
process counts are capped by a cgroup.

Its one product is the edit script: plain data, copied out through a descriptor that
refuses links and devices, and validated afterwards by trusted code.

This is namespace isolation, not a virtual machine. A kernel or GPU-driver bug could defeat
it. Official edits therefore run only on machines the organisers control, and volunteers
are never asked to run an entrant's code.
"""

from __future__ import annotations

import glob
import json
import os
import shlex
import shutil
import signal
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
FREE_SPACE_FLOOR = 5 << 30
LOG_LIMIT_BYTES = 1 << 20
GPU_DEVICES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/nvidia-caps", "/dev/kfd", "/dev/dri")
SYSTEM_DIRS = ("/usr", "/opt")
USR_LINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
ETC_ALLOW = ("ld.so.cache", "ld.so.conf", "ld.so.conf.d", "alternatives", "OpenCL", "vulkan", "glvnd")
DEFAULT_LIMITS = {"memory_gb": 48, "tasks": 4096, "scratch_gb": 16, "required": False}


class HarnessError(RuntimeError):
    """Our side failed (no sandbox, no limits). Never the editor's fault, never scored."""


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
    """Can we put the sandbox in a cgroup with memory and process limits?"""
    exe = shutil.which("systemd-run")
    if not exe:
        return False, "systemd-run is not installed"
    try:
        r = subprocess.run(
            [exe, "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M", "-p", "TasksMax=16", "--", "/usr/bin/true"],
            capture_output=True, text=True, timeout=20,
        )  # fmt: skip
    except Exception as e:
        return False, f"systemd-run failed to start: {e}"
    if r.returncode != 0:
        return False, "cannot create a limited scope with systemd-run --user: " + r.stderr.strip()[:300]
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


def _du(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


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

    Never raises on editor failure. Raises HarnessError if the sandbox cannot be set up.
    """
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    bundle = bundle.resolve()
    work = work.resolve()
    out_dir = work / "out"
    scratch_dir = work / "scratch"
    etc_dir = work / "etc"
    for d in (out_dir, scratch_dir, etc_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    pkg_dir = Path(__file__).resolve().parent
    max_script = int(rnd.season.raw["script"]["max_bytes"])
    argv = shlex.split(entry)
    if not argv:
        return EditRun("error", 0.0, time_limit, None, None, "", sandbox, "stet.toml has an empty entry command")
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
        # Belt and braces: the cache holds the exam pack. It lives under the home directory by
        # default (never bound), but STET_CACHE can point anywhere.
        cache = str(cache_dir().resolve())
        if any(cache == d or cache.startswith(d + "/") for d in SYSTEM_DIRS + tuple(src for src, _ in py_binds)):
            raise HarnessError(f"refusing to sandbox: the cache {cache} would be visible to the editor")
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

    log_path = work / "editor.log"
    start = time.time()
    deadline = start + time_limit
    env["STET_DEADLINE"] = str(deadline)
    if inside:
        for k, v in env.items():
            cmd += ["--setenv", k, v]
        cmd += ["--"] + argv

    status = "ok"
    reason = ""
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=popen_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

        def pump():
            written = 0
            for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                if written < LOG_LIMIT_BYTES:
                    log.write(chunk[: LOG_LIMIT_BYTES - written])
                    written += len(chunk)
                if echo:
                    sys.stderr.buffer.write(chunk)
                    sys.stderr.flush()

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        last_du = 0.0
        while proc.poll() is None:
            now = time.time()
            if now >= deadline:
                status, reason = "timeout", f"the editor was still running at the {time_limit:.0f}s limit"
            elif now - last_du > 2:
                last_du = now
                if _du(out_dir) > OUT_LIMIT_BYTES:
                    status, reason = "error", "the editor wrote more than 1 GB to its output directory"
                elif (not inside or not applied) and _du(scratch_dir) > SCRATCH_LIMIT_BYTES:
                    status, reason = "error", "the editor filled its scratch directory"
                elif shutil.disk_usage(out_dir).free < FREE_SPACE_FLOOR:
                    status, reason = "error", "the disk ran low while the editor was running"
            if status != "ok":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.05)
        proc.wait()
        t.join(timeout=5)
    seconds = time.time() - start
    tail = log_path.read_bytes()[-4000:].decode("utf-8", errors="replace")
    shutil.rmtree(scratch_dir, ignore_errors=True)
    if inside and proc.returncode not in (0, None) and seconds < 5 and tail.lstrip().startswith(("bwrap:", "Failed to start transient scope", "systemd-run")):
        raise HarnessError(f"the sandbox failed to start: {tail.strip()[:300]}")
    if status == "ok" and proc.returncode != 0:
        status, reason = "error", f"the editor exited with status {proc.returncode}"
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
