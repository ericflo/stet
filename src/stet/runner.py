"""Run an editor: the one place untrusted code executes.

Two modes:

  sandbox="none"   plain subprocess. For running your own editor on your own machine.
  sandbox="bwrap"  a Linux namespace sandbox (bubblewrap). What verifiers use.

Inside the sandbox an editor has no network, sees an allowlist of system directories
read-only, cannot see the home directory, the Stet cache or the exam, and can write only
to its output and scratch directories. Its one product is the edit script: plain data that
trusted code validates afterwards.

This is namespace isolation, not a virtual machine. A kernel or GPU-driver bug could defeat
it. Verifiers should use a dedicated account or machine.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .data import Corpus, cache_dir
from .spec import Round, Scale

SCRATCH_LIMIT_BYTES = 20 << 30
LOG_LIMIT_BYTES = 1 << 20
GPU_DEVICES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/nvidia-caps", "/dev/kfd", "/dev/dri")
SYSTEM_DIRS = ("/usr", "/etc", "/opt", "/sys")
USR_LINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")


@dataclass
class EditRun:
    status: str  # "ok", "timeout", "error", "no_script"
    seconds: float
    time_limit: float
    exit_code: int | None
    script: Path | None
    log_tail: str
    sandbox: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "time_limit": round(self.time_limit, 2),
            "exit_code": self.exit_code,
            "sandbox": self.sandbox,
            "log_tail": self.log_tail[-2000:],
        }


def task_info(rnd: Round, scale: Scale, time_limit: float) -> dict:
    """Everything an editor is told. Nothing here depends on the exam passages."""
    season = rnd.season
    return {
        "season": season.id,
        "round": rnd.id,
        "scale": scale.name,
        "train_tokens": scale.train_tokens,
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
    }


def _python_roots() -> list[str]:
    """Directories the interpreter needs: the venv and the base install."""
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix}
    # A venv's interpreter is often a symlink into somewhere else entirely (uv, pyenv).
    roots.add(str(Path(sys.executable).resolve().parents[1]))
    return sorted(r for r in roots if os.path.isdir(r))


def bwrap_available() -> tuple[bool, str]:
    exe = shutil.which("bwrap")
    if not exe:
        return False, "bubblewrap (bwrap) is not installed"
    try:
        r = subprocess.run(
            [exe, "--unshare-user", "--unshare-net", "--unshare-pid", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
             "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "/usr/bin/true"],
            capture_output=True, text=True, timeout=20,
        )  # fmt: skip
    except Exception as e:
        return False, f"bwrap failed to start: {e}"
    if r.returncode != 0:
        return False, "bwrap cannot create a sandbox here (unprivileged user namespaces may be disabled): " + r.stderr.strip()[:300]
    return True, exe


def bwrap_command(bundle: Path, corpus_dir: Path, out_dir: Path, scratch_dir: Path, pkg_dir: Path) -> list[str]:
    cmd = [
        "bwrap", "--die-with-parent", "--new-session",
        "--unshare-user", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
        "--hostname", "stet",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--tmpfs", "/tmp", "--tmpfs", "/run", "--tmpfs", "/var",
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
    for dev in GPU_DEVICES:
        if os.path.exists(dev):
            cmd += ["--dev-bind", dev, dev]
    for dev in sorted(Path("/dev").glob("nvidia[0-9]*")):
        cmd += ["--dev-bind", str(dev), str(dev)]
    for root in _python_roots():
        cmd += ["--ro-bind", root, root]
    cmd += [
        "--ro-bind", str(pkg_dir), "/stet-pkg/stet",
        "--ro-bind", str(bundle), "/editor",
        "--ro-bind", str(corpus_dir), "/corpus",
        "--bind", str(out_dir), "/out",
        "--bind", str(scratch_dir), "/scratch",
        "--chdir", "/editor",
        "--setenv", "HOME", "/scratch",
        "--setenv", "TMPDIR", "/tmp",
    ]  # fmt: skip
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
) -> EditRun:
    """Run `entry` from the bundle directory against a corpus. Never raises on editor failure."""
    bundle = bundle.resolve()
    out_dir = (work / "out").resolve()
    scratch_dir = (work / "scratch").resolve()
    for d in (out_dir, scratch_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    pkg_dir = Path(__file__).resolve().parent
    argv = shlex.split(entry)
    if not argv:
        return EditRun("error", 0.0, time_limit, None, None, "stet.toml has an empty entry command", sandbox)
    if argv[0] in ("python", "python3"):
        argv[0] = sys.executable

    inside = sandbox == "bwrap"
    paths = {
        "STET_TASK": "/out/task.json" if inside else str(out_dir / "task.json"),
        "STET_CORPUS": "/corpus" if inside else str(corpus.path.resolve()),
        "STET_OUT": "/out" if inside else str(out_dir),
        "STET_SCRATCH": "/scratch" if inside else str(scratch_dir),
    }
    (out_dir / "task.json").write_text(json.dumps(task_info(rnd, scale, time_limit), indent=1))
    env = {
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin"]),
        "PYTHONPATH": "/stet-pkg" if inside else str(pkg_dir.parent),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "STET_TIME_LIMIT": str(time_limit),
        **paths,
    }
    for k in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "ROCM_PATH", "HIP_VISIBLE_DEVICES", "TORCH_BLAS_PREFER_HIPBLASLT"):
        if k in os.environ:
            env[k] = os.environ[k]
    if not inside:
        env["HOME"] = str(scratch_dir)
        env["TMPDIR"] = str(scratch_dir)

    if inside:
        ok, why = bwrap_available()
        if not ok:
            return EditRun("error", 0.0, time_limit, None, None, f"sandbox unavailable: {why}", sandbox)
        cmd = bwrap_command(bundle, corpus.path.resolve(), out_dir, scratch_dir, pkg_dir)
        # Belt and braces: the cache holds the exam pack. It lives under the home directory by
        # default (never bound), but STET_CACHE can point anywhere.
        cache = str(cache_dir().resolve())
        if any(cache == d or cache.startswith(d + "/") for d in SYSTEM_DIRS + tuple(_python_roots())):
            return EditRun("error", 0.0, time_limit, None, None, f"refusing to sandbox: the cache {cache} would be visible to the editor", sandbox)
        for k, v in env.items():
            cmd += ["--setenv", k, v]
        cmd += ["--"] + argv
        popen_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        cwd = None
    else:
        cmd = argv
        popen_env = env
        cwd = str(bundle)

    log_path = work / "editor.log"
    start = time.time()
    deadline = start + time_limit
    env_deadline = str(deadline)
    if inside:
        cmd[cmd.index("--") : cmd.index("--")] = ["--setenv", "STET_DEADLINE", env_deadline]
    else:
        popen_env["STET_DEADLINE"] = env_deadline

    status = "ok"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=popen_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

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
                status = "timeout"
            elif now - last_du > 5:
                last_du = now
                if _du(scratch_dir) + _du(out_dir) > SCRATCH_LIMIT_BYTES:
                    status = "error"
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
    script = out_dir / "script.jsonl"
    if status == "ok" and proc.returncode != 0:
        status = "error"
    # A script written before a crash or a timeout still counts: editors are encouraged to
    # write early and improve. script.jsonl is replaced atomically, so it is never half-written.
    if not script.exists():
        return EditRun(status if status != "ok" else "no_script", seconds, time_limit, proc.returncode, None, tail, sandbox)
    return EditRun(status, seconds, time_limit, proc.returncode, script, tail, sandbox)
