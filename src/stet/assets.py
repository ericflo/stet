"""Files an editor needs that are too big to ship in its bundle: model weights, by reference.

A bundle is small (50 MB), so that it can be read by a person and passed around. What does
not fit, an editor names in stet.toml, and every machine that runs it fetches the same bytes:

    [[assets]]
    path = "smollm2-135m/model.safetensors"
    url = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/<commit>/model.safetensors"
    sha256 = "<64 hex characters>"
    size = 269060552

The rules, and why:

  - The file is public and anyone can download it without an account. Volunteers run
    editors, and a volunteer has no login to give.
  - The address names a commit, not a branch, and the entry names the file's hash. What is
    scored is what was submitted: a model cannot be swapped under an entry afterwards.
  - Files are fetched before the editor's clock starts and are read-only inside the
    sandbox, at $STET_ASSETS/<path> (task.asset("<path>")). The sandbox has no network.
  - A season bounds how many files and how many bytes an editor may name.

`stet asset add <repository>` writes these tables for you. The same checks are made by the
server when an entry is submitted (server/bundle.go); keep the two in step.
"""

from __future__ import annotations

import errno
import fnmatch
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .spec import Season

# One host for now: files there are content-addressed, pinned by commit, and served
# without an account. Another host is a change to the season, not to an entry.
URL_RE = re.compile(
    r"^https://huggingface\.co/(?:datasets/)?[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
    r"/resolve/[0-9a-f]{40}/[A-Za-z0-9][A-Za-z0-9._/-]{0,400}$"
)
PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_PATH = 200
FIELDS = ("path", "url", "sha256", "size")

# Used when a season says nothing: none are allowed.
NO_ASSETS = {"max_files": 0, "max_bytes": 0}

# What `stet asset add` leaves out unless asked: other frameworks' copies of the same
# weights, and files no program reads.
SKIP = (
    "*.bin", "*.h5", "*.msgpack", "*.onnx", "*.ot", "*.gguf", "*.pt", "*.pth", "*.ckpt", "*.tflite", "*.mlmodel",
    "*.md", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", ".gitattributes", ".gitignore", "LICENSE*", "NOTICE*",
    "onnx/*", "openvino/*", "runs/*", "logs/*", "coreml/*", "flax_model*", "tf_model*", "rust_model*", "*.npz",
)  # fmt: skip


class AssetError(ValueError):
    """An assets table that breaks a rule. The message is for the entrant."""


class Unavailable(Exception):
    """A file an entry names is not there to be had: taken down, made private, or no longer
    the bytes the entry says. The server checked every file when the entry was submitted,
    so this is something that happened to the entry's files since, and it is the entry's
    failure. A host that is merely away is not this: that is data.FetchError."""


@dataclass(frozen=True)
class Asset:
    path: str
    url: str
    sha256: str
    size: int


def limits(season: Season) -> dict:
    return {**NO_ASSETS, **(season.raw.get("assets") or {})}


def parse(raw, season: Season) -> list[Asset]:
    """Check the [[assets]] tables of a manifest and return them. Raises AssetError."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(a, dict) for a in raw):
        raise AssetError("stet.toml: assets must be written as [[assets]] tables")
    lim = limits(season)
    if raw and not lim["max_files"]:
        raise AssetError("stet.toml: this season does not allow [[assets]]")
    if len(raw) > lim["max_files"]:
        raise AssetError(f"stet.toml: {len(raw)} assets; the limit is {lim['max_files']}")
    out: list[Asset] = []
    seen: dict[str, str] = {}
    total = 0
    for i, a in enumerate(raw, 1):
        where = f"stet.toml: asset {i}"
        for k in a:
            if k not in FIELDS:
                raise AssetError(f"{where}: unknown key {str(k)[:40]!r} (an asset has path, url, sha256 and size)")
        for k in FIELDS:
            if k not in a:
                raise AssetError(f"{where}: missing {k}")
        path, url, sha, size = a["path"], a["url"], a["sha256"], a["size"]
        if not isinstance(path, str) or not isinstance(url, str) or not isinstance(sha, str):
            raise AssetError(f"{where}: path, url and sha256 must be strings")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise AssetError(f"{where}: size must be the file's length in bytes")
        parts = path.split("/")
        if len(path) > MAX_PATH or not all(PART_RE.match(p) for p in parts):
            raise AssetError(
                f"{where}: path {path[:60]!r} must be a relative path of plain names "
                "(letters, digits, dot, dash and underscore, none starting with a dot)"
            )
        if not URL_RE.match(url) or ".." in url or "//" in url[8:]:
            raise AssetError(
                f"{where}: the url must be a Hugging Face file pinned to a commit: "
                "https://huggingface.co/<owner>/<repository>/resolve/<40-character commit>/<file>"
            )
        if not SHA_RE.match(sha):
            raise AssetError(f"{where}: sha256 must be 64 lowercase hex characters")
        key = path.lower()
        for other in seen:
            if key == other or key.startswith(other + "/") or other.startswith(key + "/"):
                raise AssetError(f"{where}: path {path!r} collides with {seen[other]!r}")
        seen[key] = path
        total += size
        out.append(Asset(path, url, sha, size))
    if total > lim["max_bytes"]:
        raise AssetError(
            f"stet.toml: the assets come to {total / 2**30:.2f} GiB; the limit is {lim['max_bytes'] / 2**30:.0f} GiB"
        )
    return out


def to_toml(assets: list[Asset]) -> str:
    """The [[assets]] tables, as they are written into stet.toml."""
    out = []
    for a in assets:
        out.append(f'[[assets]]\npath = "{a.path}"\nurl = "{a.url}"\nsha256 = "{a.sha256}"\nsize = {a.size}\n')
    return "\n".join(out)


# --- the cache ----------------------------------------------------------------------------------


def cache() -> Path:
    from .data import cache_dir

    return cache_dir() / "assets"


def cached(asset: Asset) -> Path:
    """Where this file lives on this machine, by its hash. Two editors that name the same
    file share one copy."""
    return cache() / asset.sha256


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fetch(assets: list[Asset], quiet: bool = False, verify: bool = False) -> None:
    """Download whichever of these files this machine does not have, checking each hash.

    `verify` reads the copies this machine already has and checks them again. A run for
    the leaderboard does: a copy that has changed on disk since it was fetched would make
    this machine's run of an editor differ from everyone else's, and nobody could say why.
    """
    from .data import FetchError, Mismatch, TooBig, fetch, sweep_stale

    # What a verifier killed in the middle of a download left behind: up to 4 GiB each,
    # which nothing else would ever clear or count against the cache's limit.
    sweep_stale(cache(), "*.part*")
    for a in assets:
        dest = cached(a)
        if dest.exists():
            if not verify or (dest.stat().st_size == a.size and _sha256(dest) == a.sha256):
                os.utime(dest)  # most recently used: the last to be cleared away
                continue
            dest.unlink(missing_ok=True)  # not the file it was: fetched again below
        try:
            fetch([a.url], a.sha256, dest, label=a.path, quiet=quiet, max_bytes=a.size)
        except TooBig:
            raise Unavailable(f"the asset {a.path} is larger than its entry says") from None
        except FetchError as e:
            last = getattr(e, "last", None)
            if isinstance(last, urllib.error.HTTPError) and last.code in (401, 403, 404, 410):
                raise Unavailable(f"the asset {a.path} is no longer public at its address (the host answers {last.code})") from None
            if isinstance(last, Mismatch):
                raise Unavailable(f"the asset {a.path} is not the bytes its entry names") from None
            raise
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EROFS):
                raise  # this machine's disk, not the host's fault: waiting will not help
            raise FetchError(f"could not fetch the asset {a.path}: {e}") from None
        if dest.stat().st_size != a.size:
            dest.unlink(missing_ok=True)
            raise Unavailable(f"the asset {a.path} is not the size its entry says")
        dest.chmod(0o444)


def prune(keep: list[Asset] = (), limit_bytes: int | None = None) -> int:
    """Clear the least recently used files away until the cache is under its limit.

    A machine that runs editors for others sees many models. The limit is 40 GB unless
    STET_ASSET_CACHE_GB says otherwise; what the current job needs is never cleared.
    """
    if limit_bytes is None:
        limit_bytes = int(float(os.environ.get("STET_ASSET_CACHE_GB") or 40) * 2**30)
    d = cache()
    if not d.is_dir():
        return 0
    needed = {a.sha256 for a in keep}
    files = [(p.stat().st_mtime, p.stat().st_size, p) for p in d.iterdir() if p.is_file() and SHA_RE.match(p.name)]
    total = sum(size for _, size, _ in files)
    removed = 0
    for _, size, p in sorted(files):
        if total <= limit_bytes:
            break
        if p.name in needed:
            continue
        p.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed


def link_tree(assets: list[Asset], dest: Path) -> Path:
    """Lay the files out under `dest` by their paths, as links into the cache. For runs
    outside the sandbox; inside it each file is mounted read-only at the same path."""
    for a in assets:
        p = dest / a.path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or p.exists():
            p.unlink()
        p.symlink_to(cached(a))
    dest.mkdir(parents=True, exist_ok=True)
    return dest


# --- looking a repository up --------------------------------------------------------------------

HUB = "https://huggingface.co"


def _get_json(url: str):
    from .data import USER_AGENT

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AssetError("that repository needs an account to read. Assets must be public: volunteers have no login to give") from None
            if e.code == 404:
                raise AssetError("no such repository, revision or file on Hugging Face") from None
            last = e
        except (OSError, ValueError) as e:
            last = e
        time.sleep(1 + attempt)
    raise AssetError(f"could not reach Hugging Face: {last}")


def lookup(repo: str, revision: str = "", include: list[str] | None = None, under: str = "") -> tuple[str, list[Asset], list[str]]:
    """Find a repository's files at a pinned commit.

    Returns (commit, assets, skipped). `include` is a list of glob patterns; without it,
    everything except other frameworks' copies of the weights and files no program reads.
    `under` is the directory the files appear in for the editor (the repository's name,
    lowercased, unless given).
    """
    kind = "datasets" if repo.startswith("datasets/") else "models"
    name = repo.removeprefix("datasets/").strip("/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise AssetError("name the repository as owner/name, as in its Hugging Face address")
    info = _get_json(f"{HUB}/api/{kind}/{name}" + (f"/revision/{urllib.parse.quote(revision, safe='')}" if revision else ""))
    commit = info.get("sha") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AssetError("Hugging Face did not say which commit that is")
    if info.get("gated") or info.get("private"):
        raise AssetError("that repository is gated or private. Assets must be public: volunteers have no login to give")
    tree: list[dict] = []
    url = f"{HUB}/api/{kind}/{name}/tree/{commit}?recursive=true&limit=1000"
    page = _get_json(url)
    if not isinstance(page, list):
        raise AssetError("Hugging Face did not list that repository's files")
    if len(page) >= 1000:
        # The listing comes a thousand files at a time, and a model is a dozen. Half a
        # repository listed would look like all of it.
        raise AssetError("that repository has more than a thousand files, too many to list. Write the [[assets]] tables for the files the editor needs by hand")
    tree.extend(page)
    prefix = "datasets/" if kind == "datasets" else ""
    under = (under or name.split("/")[1].lower()).strip("/")
    assets: list[Asset] = []
    skipped: list[str] = []
    for f in sorted(tree, key=lambda f: f.get("path", "")):
        if f.get("type") != "file":
            continue
        path = f["path"]
        if include:
            if not any(fnmatch.fnmatch(path, pat) for pat in include):
                continue
        elif any(fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path.rsplit("/", 1)[-1], pat) for pat in SKIP):
            skipped.append(path)
            continue
        lfs = f.get("lfs") or {}
        sha, size = lfs.get("oid") or "", int(lfs.get("size") or f.get("size") or 0)
        a = Asset(f"{under}/{path}", f"{HUB}/{prefix}{name}/resolve/{commit}/{path}", sha if SHA_RE.match(sha) else "", size)
        if not URL_RE.match(a.url) or ".." in a.url or "//" in a.url[8:] or not all(PART_RE.match(p) for p in a.path.split("/")) or len(a.path) > MAX_PATH:
            skipped.append(path)  # a name the rules do not allow (a space, a leading dot): it cannot be an asset
            continue
        assets.append(a)
    return commit, assets, skipped


def fill_hashes(assets: list[Asset], quiet: bool = False) -> list[Asset]:
    """Small files are kept in git, which names them by a hash of its own. Fetch those to
    learn their sha256; large files already have theirs. What is fetched goes into the
    cache, so it is not fetched again to run the editor."""
    from .data import fetch

    out = []
    cache().mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(assets):
        if a.sha256:
            out.append(a)
            continue
        part = cache() / f".new{os.getpid()}-{i}"
        part.unlink(missing_ok=True)
        try:
            fetch([a.url], "", part, label=a.path, quiet=quiet, max_bytes=max(a.size, 1) + (1 << 20))
            found = Asset(a.path, a.url, _sha256(part), part.stat().st_size)
            if cached(found).exists():
                part.unlink()
            else:
                part.chmod(0o444)
                part.replace(cached(found))
        finally:
            part.unlink(missing_ok=True)
        out.append(found)
    return out
