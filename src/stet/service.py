"""`stet service`: keep a verifier running on a machine nobody is watching.

`stet verify` in a terminal stops when the terminal closes or the machine restarts. This
writes a systemd user unit that starts it with the machine, starts it again if it stops,
and keeps its log. With automatic upgrades (stet.upgrade) that is a machine its owner can
forget about.

Linux with systemd only: it is where curators are run, and where a GPU box left alone
usually is. Elsewhere `stet service` says so and what to do instead.

Nothing here needs root. The unit lives in the user's own systemd directory; the one step
that may ask for a password is `loginctl enable-linger`, which lets a user's services run
while they are logged out, and when it is refused the command says what that means.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from . import upgrade

DEFAULT_NAME = "stet-verify"
KEEP = ("STET_CONFIG", "STET_API", "STET_AUTO_UPGRADE", "STET_ASSET_CACHE_GB", "STET_CACHE", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES", "HF_HOME",
        "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")  # fmt: skip
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,40}$")


class ServiceError(Exception):
    pass


def unit_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser() / "systemd" / "user"


def unit_path(name: str) -> Path:
    return unit_dir() / f"{name}.service"


def log_path(name: str) -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or "~/.local/state").expanduser() / "stet" / f"{name}.log"


def supported() -> tuple[bool, str]:
    if not sys.platform.startswith("linux"):
        return False, "`stet service` is for Linux with systemd. Here, leave `stet verify` running in a terminal, or start it from your system's own scheduler."
    if shutil.which("systemctl") is None:
        return False, "there is no systemctl on this machine. Leave `stet verify` running under whatever keeps your programs up."
    return True, ""


def quote(value: str) -> str:
    """One word of a systemd command line or Environment= value. % is systemd's own escape."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def unit_text(stet: str, name: str = DEFAULT_NAME, edits: bool = False, no_upgrade: bool = False, device: str | None = None, environ=None) -> str:
    """The unit file. Everything this machine's owner set in the shell that changes what a
    verifier does (which account, which GPU, where its caches are) is carried into it,
    because a service starts with none of the shell's environment."""
    environ = os.environ if environ is None else environ
    words = [stet, "verify"]
    if edits:
        words.append("--edits")
    if no_upgrade:
        words.append("--no-upgrade")
    if device:
        words += ["--device", device]
    path = []
    for d in (os.path.dirname(stet), os.path.dirname(upgrade.find_uv() or ""), "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"):
        if d and d not in path:
            path.append(d)
    env = [f"Environment={quote('PYTHONUNBUFFERED=1')}", f"Environment={quote('PATH=' + ':'.join(path))}"]
    for k in KEEP:
        if environ.get(k):
            env.append(f"Environment={quote(f'{k}={environ[k]}')}")
    log = str(log_path(name)).replace("%", "%%")
    what = "runs entrants' curators in a sandbox and grades edit scripts" if edits else "grades edit scripts"
    return "\n".join(
        [
            "[Unit]",
            f"Description=Stet verifier: {what}",
            "Documentation=https://www.stetrun.com/verify",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "ExecStart=" + " ".join(quote(w) for w in words),
            *env,
            # A verifier told to stop hands its job back before it goes: give it time to.
            "TimeoutStopSec=90",
            "Restart=always",
            "RestartSec=60",
            f"StandardOutput=append:{log}",
            f"StandardError=append:{log}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    done = subprocess.run(["systemctl", "--user", *args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    if check and done.returncode != 0:
        said = (done.stderr or done.stdout).strip().splitlines()
        raise ServiceError(f"systemctl --user {' '.join(args)}: {said[-1] if said else 'failed'}")
    return done


def linger() -> tuple[bool, str]:
    """Let this user's services run while nobody is logged in. (on, what to tell them)."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if shutil.which("loginctl") is None or not user:
        return False, "Could not tell whether it keeps running after you log out. If it does not: sudo loginctl enable-linger $USER"
    try:
        shown = subprocess.run(["loginctl", "show-user", user, "-p", "Linger"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
        if "Linger=yes" in shown.stdout:
            return True, ""
        done = subprocess.run(["loginctl", "enable-linger", user], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False, f"Could not turn on lingering. To keep it running while you are logged out: sudo loginctl enable-linger {user}"
    if done.returncode == 0:
        return True, ""
    return False, f"It will stop when you log out. To keep it running, once: sudo loginctl enable-linger {user}"


def install(name: str = DEFAULT_NAME, edits: bool = False, no_upgrade: bool = False, device: str | None = None, start: bool = True) -> dict:
    ok, why = supported()
    if not ok:
        raise ServiceError(why)
    if not NAME_RE.match(name):
        raise ServiceError("a service's name is lower-case letters, digits and hyphens")
    stet = upgrade.launcher()
    if not stet:
        raise ServiceError("cannot tell where `stet` is installed. Run it by its full path, or put it on your PATH.")
    stet = os.path.abspath(stet)
    path = unit_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_path(name).parent.mkdir(parents=True, exist_ok=True)
    text = unit_text(stet, name, edits, no_upgrade, device)
    tmp = path.with_suffix(".service.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    systemctl("daemon-reload")
    systemctl("enable", f"{name}.service")
    if start:
        systemctl("restart", f"{name}.service")
    lingering, note = linger()
    return {"unit": str(path), "log": str(log_path(name)), "stet": stet, "linger": lingering, "note": note, "started": start}


def uninstall(name: str = DEFAULT_NAME) -> bool:
    ok, why = supported()
    if not ok:
        raise ServiceError(why)
    path = unit_path(name)
    if not path.exists():
        return False
    systemctl("disable", "--now", f"{name}.service", check=False)
    path.unlink()
    systemctl("daemon-reload", check=False)
    return True


def status(name: str = DEFAULT_NAME) -> dict:
    ok, why = supported()
    if not ok:
        raise ServiceError(why)
    path = unit_path(name)
    if not path.exists():
        return {"installed": False}
    active = systemctl("is-active", f"{name}.service", check=False).stdout.strip() or "unknown"
    enabled = systemctl("is-enabled", f"{name}.service", check=False).stdout.strip() or "unknown"
    return {"installed": True, "active": active, "enabled": enabled, "unit": str(path), "log": str(log_path(name)), "tail": tail(name, 8)}


def tail(name: str = DEFAULT_NAME, lines: int = 40) -> list[str]:
    path = log_path(name)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


LOG_LIMIT = 8 << 20  # bytes a verifier's log may reach
LOG_KEEP = 1 << 20  # and what is kept of it then: the newest


def trim_log(fd: int = 1, limit: int = LOG_LIMIT, keep: int = LOG_KEEP) -> bool:
    """Cut a log this process is appending to back to its newest part. True if it was cut.

    A service's output is appended to one file for as long as the service is installed,
    and a verifier left alone for a year would leave a log nobody rotates. Only a file
    opened for appending is touched: writing goes on at the new end of it. Anything else
    (a terminal, a pipe, a file being written from a fixed place) is left alone.
    """
    try:
        import fcntl

        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size <= limit or not fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_APPEND:
            return False
        with open(f"/proc/self/fd/{fd}", "rb") as f:
            f.seek(st.st_size - keep)
            tail = f.read()
        tail = tail[tail.find(b"\n") + 1 :]  # from the start of a line
        sys.stdout.flush()
        sys.stderr.flush()
        os.ftruncate(fd, 0)
        os.write(fd, b"(older lines were cleared to keep this log small)\n" + tail)
        return True
    except (ImportError, OSError, ValueError):
        return False

