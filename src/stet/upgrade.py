"""Keeping an unattended verifier current.

Someone who starts `stet verify` and leaves the machine alone should not have to come back
every time the harness changes. So a verifier that holds no job asks now and then whether a
newer stet is out, and if one is, runs the site's installer (the same one that installed it)
and starts itself again with the same arguments.

It is on by default, and off with `stet verify --no-upgrade` or STET_AUTO_UPGRADE=0.

What it trusts: the site, exactly as the first `curl ... | sh` did, and nothing more. The
installer is fetched over TLS from the address the API gives as its site. A machine whose
owner wants to read each release first turns this off and upgrades by hand.

What it will not do: upgrade while a job is held; upgrade a checkout or any install it did
not make (it only knows how to redo `uv tool install`); or try again at once after a
failure. The installer, told it is upgrading, fetches and builds everything in a scratch
environment before it touches the install that works, so a download that fails half-way
leaves the old version running.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__

CHECK_EVERY = 15 * 60  # seconds between looks at what the server has, while nothing is held
RETRY_AFTER = 30 * 60  # after an upgrade that failed
INSTALL_TIMEOUT = 45 * 60  # a few GB on a slow line
INSTALLER_LIMIT = 256 << 10
OFF = ("0", "false", "no", "off")


def parse(version: str) -> tuple[int, ...]:
    """1.2.3 as (1, 2, 3). Anything after the numbers (a local tag, rc1) is ignored."""
    out = []
    for part in str(version).split("+", 1)[0].split("."):
        digits = ""
        for c in part:
            if not c.isdigit():
                break
            digits += c
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def newer(theirs: str, ours: str = __version__) -> bool:
    a, b = parse(theirs), parse(ours)
    return bool(a) and bool(b) and a > b


def enabled(args=None) -> bool:
    if args is not None and getattr(args, "no_upgrade", False):
        return False
    if os.environ.get("STET_AUTO_UPGRADE", "").strip().lower() in OFF:
        return False
    return os.environ.get("STET_DEV") != "1"


def find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for guess in ("~/.local/bin/uv", "~/.cargo/bin/uv"):  # a service's PATH is often shorter than a shell's
        p = Path(guess).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def launcher() -> str | None:
    """The `stet` this process was started as: what to start again."""
    first = sys.argv[0] if sys.argv else ""
    if first and os.path.isabs(first) and os.path.isfile(first) and os.access(first, os.X_OK):
        return first
    return shutil.which(first) if first and os.sep not in first else (shutil.which("stet") or None)


def can() -> tuple[bool, str]:
    """Whether this install is one the installer made and can make again."""
    if os.name != "posix":
        return False, "automatic upgrades are for Linux and macOS"
    if not (Path(sys.prefix) / "uv-receipt.toml").is_file():
        return False, "this stet was not installed by the installer (a checkout, or another tool's environment)"
    if find_uv() is None:
        return False, "uv is not on the PATH"
    if shutil.which("sh") is None:
        return False, "there is no sh"
    if launcher() is None:
        return False, "cannot tell how this stet was started"
    return True, ""


def torch_index(season_env: dict) -> str:
    """The wheel index this install's PyTorch came from, when that can be told from it, so
    that an upgrade keeps the build for this machine's GPU. Empty lets the installer choose."""
    try:
        from importlib import metadata

        tag = metadata.version("torch").partition("+")[2]
    except Exception:
        return ""
    index = season_env.get("torch_index") or {}
    if tag.startswith("rocm"):
        return str(index.get("amd") or "")
    return ""


def has_torch() -> bool:
    try:
        from importlib import metadata

        metadata.version("torch")
        return True
    except Exception:
        return False


def is_grade_only(prefix: str | Path | None = None) -> bool:
    """Whether this uv tool install was created as a lean grade-only install (`stet-cli[train]`),
    so upgrades preserve the selected installation role even when STET_GRADE_ONLY is not kept in
    the environment."""
    receipt = Path(prefix if prefix is not None else sys.prefix) / "uv-receipt.toml"
    if not receipt.is_file():
        return False
    try:
        import tomllib

        data = tomllib.loads(receipt.read_text(encoding="utf-8"))
    except Exception:
        return False
    reqs = (data.get("tool") or {}).get("requirements") or []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        if str(req.get("name") or "").lower().replace("_", "-") != "stet-cli":
            continue
        extras = {str(e).strip().lower() for e in (req.get("extras") or [])}
        return "train" in extras and "run" not in extras
    return False


def fetch_installer(site: str) -> bytes:
    # The host is compared whole: "http://localhost.example" and "http://127.0.0.1@example" are not this machine.
    try:
        at = urllib.parse.urlsplit(site)
        scheme, host = at.scheme, at.hostname
    except ValueError:
        scheme, host = "", None
    if not (scheme == "https" and host) and not (scheme == "http" and host in ("localhost", "127.0.0.1")):
        raise OSError(f"will not fetch an installer from {site!r}: not https")
    from .client import USER_AGENT

    req = urllib.request.Request(site.rstrip("/") + "/install.sh", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        blob = r.read(INSTALLER_LIMIT + 1)
    if len(blob) > INSTALLER_LIMIT or not blob.startswith(b"#!/bin/sh"):
        raise OSError("what the site sent is not the installer")
    return blob


def install(site: str, season_env: dict | None = None, log=None) -> tuple[bool, str]:
    """Run the site's installer over this install. (worked, the end of what it said)."""
    say = log or (lambda m: None)
    try:
        script = fetch_installer(site)
    except (OSError, ValueError) as e:
        return False, f"could not fetch the installer ({e})"
    env = dict(os.environ)
    uv = find_uv()
    if uv:
        env["PATH"] = os.pathsep.join([os.path.dirname(uv), env.get("PATH", "")])
    env["STET_UPGRADE"] = "1"
    if not has_torch():
        env["STET_LIGHT"] = "1"
    else:
        if not env.get("STET_GRADE_ONLY") and is_grade_only():
            env["STET_GRADE_ONLY"] = "1"
        if not env.get("STET_TORCH_INDEX") and (index := torch_index(season_env or {})):
            env["STET_TORCH_INDEX"] = index
    with tempfile.TemporaryDirectory(prefix="stet-upgrade-") as tmp:
        path = Path(tmp) / "install.sh"
        path.write_bytes(script)
        say("running the installer (a few minutes if libraries changed) ...")
        try:
            done = subprocess.run(["sh", str(path)], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", timeout=INSTALL_TIMEOUT)  # fmt: skip
        except subprocess.TimeoutExpired:
            return False, f"the installer was still running after {INSTALL_TIMEOUT // 60} minutes"
        except OSError as e:
            return False, f"the installer could not be run ({e})"
    tail = " | ".join(line.strip() for line in done.stdout.strip().splitlines()[-6:])[-600:]
    return done.returncode == 0, tail


def installed_version(path: str) -> str:
    """What the stet at `path` says it is, asked afresh: this process is still the old one."""
    try:
        said = subprocess.run([path, "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ""
    words = said.stdout.split()
    return words[-1] if said.returncode == 0 and words else ""


def restart(path: str) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(path, [path, *sys.argv[1:]])


class Upgrader:
    """What a long-running command holds to keep itself current. Call `look` whenever no
    job is held; call `now` when the server has said this version is too old to serve."""

    def __init__(self, args, season_env: dict | None = None, say=None, fetch_info=None):
        self.on = enabled(args)
        self.season_env = season_env or {}
        self.say = say or (lambda m: None)
        self.fetch_info = fetch_info
        self.checked = time.monotonic()  # the caller has just read the season: no need to ask again at once
        self.failed_at: float | None = None
        self.told = False

    def _explain_once(self, why: str) -> None:
        if not self.told:
            self.told = True
            self.say(why)

    def consider(self, info: dict) -> bool:
        """Upgrade and start again if `info` (the season, as the server gives it) names a
        newer stet. Returns False when there is nothing to do or it could not be done; does
        not return at all when it worked."""
        cli = info.get("cli") if isinstance(info, dict) else None
        theirs = str((cli or {}).get("version") or "")
        if not newer(theirs):
            return False
        if not self.on:
            self._explain_once(f"stet {theirs} is out (this is {__version__}). Automatic upgrades are off: curl -fsSL {info.get('site') or 'https://www.stetrun.com'}/install.sh | sh")
            return False
        ok, why = can()
        if not ok:
            self._explain_once(f"stet {theirs} is out (this is {__version__}), and it cannot be installed automatically here: {why}. Upgrade by hand.")
            return False
        if self.failed_at is not None and time.monotonic() - self.failed_at < RETRY_AFTER:
            return False
        self.say(f"stet {theirs} is out (this is {__version__}). Upgrading; turn this off with --no-upgrade.")
        worked, tail = install(str(info.get("site") or ""), self.season_env, self.say)
        path = launcher()
        have = installed_version(path) if worked and path else ""
        if not worked or not path or not newer(have):
            self.failed_at = time.monotonic()
            self.say(f"the upgrade did not work ({tail or 'the installer said nothing'}). Carrying on as {__version__}; trying again in {RETRY_AFTER // 60} minutes.")
            return False
        self.say(f"upgraded to {have}. Starting again.")
        restart(path)
        return True  # only reached when restart is replaced in a test

    def look(self) -> bool:
        """Ask the server what is current, if it has been a while. For when nothing is held."""
        if self.fetch_info is None or time.monotonic() - self.checked < CHECK_EVERY:
            return False
        self.checked = time.monotonic()
        try:
            info = self.fetch_info()
        except Exception:
            return False  # the server is away; the next look will ask again
        return self.consider(info if isinstance(info, dict) else {})

    def now(self) -> bool:
        """The server has refused this version. Ask what is current and upgrade to it."""
        if self.fetch_info is None:
            return False
        self.checked = time.monotonic()
        try:
            info = self.fetch_info()
        except Exception:
            return False
        return self.consider(info if isinstance(info, dict) else {})
