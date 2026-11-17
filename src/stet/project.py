"""An editor project on disk: stet.toml, and how it is packed into a bundle.

A bundle is a deterministic tar of the project directory, zstd-compressed. Its sha256 is
the editor's identity: the leaderboard, the verifiers and the audit trail all refer to
exactly those bytes.
"""

from __future__ import annotations

import hashlib
import io
import math
import posixpath
import re
import tarfile
import tomllib
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import zstandard

MAX_BUNDLE_BYTES = 50 << 20
# The server refuses a bundle past either of these, so the client does too, on both sides:
# when packing one and when unpacking someone else's.
MAX_UNPACKED_BYTES = 400 << 20
MAX_BUNDLE_FILES = 2000
MAX_PATH_BYTES = 255  # one file's path inside the bundle
MAX_PATHS_BYTES = 64 << 10  # every file's path, added up: the file list is kept with the editor for good
MAX_MANIFEST_BYTES = 64 << 10  # stet.toml is five short fields
MAX_ENTRY_BYTES = 200
# Never packed. A bundle is public, so anything whose name starts with a dot stays home too:
# .env, .git, .venv, an editor's settings, a tool's cache.
EXCLUDE_DIRS = {"venv", "__pycache__", "node_modules", "runs"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")
# Every key the server reads from stet.toml. It matches them without regard to case, so
# `Name` is `name` to the server and nothing at all to this client: see load_project.
MANIFEST_FIELDS = ("name", "entry", "description", "technique", "parent", "license", "assets")


class ProjectError(ValueError):
    pass


@dataclass
class Project:
    path: Path
    name: str
    entry: str
    description: str = ""
    technique: str = ""
    parent: str = ""
    license: str = "MIT"
    assets: list = dc_field(default_factory=list)  # of stet.assets.Asset: files fetched by reference

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "entry": self.entry,
            "description": self.description,
            "technique": self.technique,
            "parent": self.parent,
            "license": self.license,
        }


def _clip(text: str, limit: int) -> str:
    """The first `limit` bytes of text, cut between characters. (The server measures in bytes.)"""
    return text.encode("utf-8", "replace")[:limit].decode("utf-8", "ignore")


def _float(text: str) -> float:
    """A TOML float as the server reads one: a number too large for 64 bits is refused, not
    rounded up to infinity. (`inf` itself is a float, and is let through.)"""
    value = float(text)
    if math.isinf(value) and "inf" not in text:
        raise ValueError(f"the number {text[:40]} does not fit in a 64-bit float")
    return value


def _whole_numbers_fit(cfg: dict) -> None:
    """Refuse an integer the server cannot hold, wherever in the file it is. Python reads an
    integer of any size; the server reads 64 bits and calls anything larger a file it
    cannot parse."""
    todo: list = [cfg]
    while todo:
        value = todo.pop()
        if isinstance(value, dict):
            todo.extend(value.values())
        elif isinstance(value, list):
            todo.extend(value)
        elif isinstance(value, int) and not isinstance(value, bool) and not -(1 << 63) <= value < (1 << 63):
            raise ProjectError(f"stet.toml is not valid TOML: the integer {str(value)[:40]} does not fit in 64 bits")


def load_project(path: str | Path) -> Project:
    """Read a project's stet.toml. Whatever the server would refuse in one is refused here,
    so that `stet run` and `stet submit --dry-run` find it first. So is what the server
    would read differently from this client."""
    path = Path(path).resolve()
    toml = path / "stet.toml"
    if toml.is_symlink():
        # Links are never packed, so the bundle would arrive without one.
        raise ProjectError("stet.toml is a symbolic link. Links are not packed into a bundle, so it would be submitted without a stet.toml: make it a plain file.")
    if not toml.exists():
        raise ProjectError(f"{path} has no stet.toml. Run `stet init` to create an editor project.")
    if not toml.is_file():
        raise ProjectError("stet.toml is not a plain file")
    if toml.stat().st_size > MAX_MANIFEST_BYTES:
        raise ProjectError(f"stet.toml is over {MAX_MANIFEST_BYTES >> 10} KB. It holds a name, a command and a few lines about the editor.")
    try:
        cfg = tomllib.loads(toml.read_text(encoding="utf-8"), parse_float=_float)
    except (ValueError, RecursionError) as e:  # not TOML, not UTF-8, or a number nothing can hold
        raise ProjectError(f"stet.toml is not valid TOML: {e}") from None
    _whole_numbers_fit(cfg)
    for key in cfg:
        for field in MANIFEST_FIELDS:
            if key != field and key.casefold() == field:
                # The server matches field names without regard to case, and this client
                # does not. Left alone, the two would read different manifests from one file.
                raise ProjectError(f"stet.toml: the key {key[:40]!r} must be written {field}. Field names are lowercase, and the server would take this one for {field}.")
    for field in ("name", "entry", "description", "technique", "parent", "license"):
        if not isinstance(cfg.get(field, ""), str):
            raise ProjectError(f'stet.toml: {field} must be text in quotes, e.g. {field} = "..."')
    name = cfg.get("name", "").strip()
    if not NAME_RE.match(name):
        raise ProjectError("stet.toml: name must be 3-40 characters of lowercase letters, digits and hyphens")
    entry = cfg.get("entry", "").strip()
    if not entry:
        raise ProjectError('stet.toml: entry is required, e.g. entry = "python editor.py"')
    if len(entry.encode("utf-8", "replace")) > MAX_ENTRY_BYTES or any(ch in entry for ch in "\0\n\r"):
        raise ProjectError(f"stet.toml: entry is one command line of at most {MAX_ENTRY_BYTES} characters")
    from . import assets as assets_
    from .spec import load_season

    try:
        named = assets_.parse(cfg.get("assets"), load_season())
    except assets_.AssetError as e:
        raise ProjectError(str(e)) from None
    description, technique = _clip(cfg.get("description", "").strip(), 300), _clip(cfg.get("technique", "").strip(), 120)
    # These two are shown on the site and printed in other people's terminals.
    if not (description + technique).isprintable():
        raise ProjectError("stet.toml: description and technique are one line of ordinary text each")
    return Project(
        path=path,
        name=name,
        entry=entry,
        description=description,
        technique=technique,
        parent=cfg.get("parent", "").strip(),
        license=cfg.get("license", "MIT").strip(),
        assets=named,
    )


def skip_reason(rel: Path, is_dir: bool = False) -> str:
    """Why a path (relative to the project) is left out of a bundle, or "" if it is not.

    It mirrors leftOut in server/bundle.go, which refuses a bundle holding any such path:
    the names in EXCLUDE_DIRS are names of directories, so a plain file called `runs` is
    packed like any other.
    """
    parts = rel.parts
    for i, part in enumerate(parts):
        if part.startswith("."):
            return "its name starts with a dot"
        if part in EXCLUDE_DIRS and (is_dir or i < len(parts) - 1):
            return f"it is under {part}/"
    if rel.name.endswith(EXCLUDE_SUFFIXES):
        return "it is compiled Python"
    return ""


def survey(project: Project) -> tuple[list[Path], list[tuple[str, str]]]:
    """What a bundle of this project holds, and what it leaves out and why.

    Returns (files to pack, [(relative path, reason)]). A directory that is left out is
    listed once, not file by file.
    """
    keep: list[Path] = []
    skipped: list[tuple[str, str]] = []

    def walk(directory: Path):
        for p in sorted(directory.iterdir(), key=lambda q: q.name):
            rel = p.relative_to(project.path)
            why = skip_reason(rel, is_dir=p.is_dir() and not p.is_symlink())
            if not why and p.is_symlink():
                why = "it is a link"
            if why:
                skipped.append((rel.as_posix() + ("/" if p.is_dir() and not p.is_symlink() else ""), why))
            elif p.is_dir():
                walk(p)
            elif p.is_file():
                keep.append(p)
            else:
                skipped.append((rel.as_posix(), "it is not a plain file"))

    walk(project.path)
    return keep, skipped


def bundle_files(project: Project) -> list[Path]:
    return survey(project)[0]


def pack(project: Project) -> tuple[bytes, str]:
    """Return (bundle bytes, editor id). Same files in, same bytes out."""
    files = bundle_files(project)
    if len(files) > MAX_BUNDLE_FILES:
        raise ProjectError(f"{len(files):,} files; a bundle holds at most {MAX_BUNDLE_FILES:,}. `stet submit --dry-run` lists them.")
    too_big = ProjectError(f"the files add up to more than {MAX_UNPACKED_BYTES >> 20} MB, which is the most a bundle holds before compression")
    if sum(p.stat().st_size for p in files) > MAX_UNPACKED_BYTES:
        raise too_big
    names = [p.relative_to(project.path).as_posix() for p in files]
    for name in names:
        if len(name.encode("utf-8", "surrogateescape")) > MAX_PATH_BYTES:
            raise ProjectError(f"a file's path inside a bundle is at most {MAX_PATH_BYTES} bytes; this one is longer: {name[:80]}...")
    if sum(len(name.encode("utf-8", "surrogateescape")) for name in names) > MAX_PATHS_BYTES:
        raise ProjectError(
            f"the paths of the {len(files):,} files add up to more than {MAX_PATHS_BYTES >> 10} KB. "
            "Put many small files into one archive that the editor opens, or shorten their names."
        )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p, name in zip(files, names, strict=True):
            info = tarfile.TarInfo(name=name)
            data = p.read_bytes()
            info.size = len(data)
            info.mode = 0o755 if p.stat().st_mode & 0o100 else 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    # The limit is on the tar, headers and padding included: that is what the server and
    # a verifier measure when they unpack it.
    if buf.tell() > MAX_UNPACKED_BYTES:
        raise too_big
    blob = zstandard.ZstdCompressor(level=10).compress(buf.getvalue())
    if len(blob) > MAX_BUNDLE_BYTES:
        raise ProjectError(
            f"bundle is {len(blob) / 1e6:.1f} MB; the limit is {MAX_BUNDLE_BYTES >> 20} MB. "
            "Model weights and other large files do not go in the bundle: put them on Hugging Face and name them "
            "with `stet asset add <owner>/<repository>`. They are fetched before the editor starts."
        )
    return blob, hashlib.sha256(blob).hexdigest()


def unzstd(blob: bytes, limit: int) -> bytes:
    """Decompress, stopping one byte past `limit`: a small download cannot expand without bound.

    Raises zstandard.ZstdError for anything that is not whole zstd frames from end to end.

    zstandard's conveniences each get one of these wrong. Its one-shot decompress()
    believes the size a frame declares, whatever limit it is given. Its stream reader stops
    after the first frame unless told otherwise, so a file of two frames (pzstd writes
    those) would pass for its first; and it reports a frame that is cut short as the end of
    the data, so half a script would pass for a script. So this goes frame by frame, a
    little input at a time, and insists that the last frame ends where the data does.
    """
    dctx = zstandard.ZstdDecompressor()
    parts: list[bytes] = []
    got, at, end = 0, 0, len(blob)
    while at < end:
        frame = dctx.decompressobj()
        while not frame.eof:
            if at >= end:
                raise zstandard.ZstdError("the zstd data is cut short")
            chunk = frame.decompress(blob[at : at + 1024])  # 1 KB in is at most about 32 MB out
            at = min(end, at + 1024)
            parts.append(chunk)
            got += len(chunk)
            if got > limit:
                return b"".join(parts)[: limit + 1]
        at -= len(frame.unused_data)  # the start of the next frame, if there is one
    return b"".join(parts)


def unpack(blob: bytes, dest: Path) -> list[str]:
    """Extract a bundle someone else made. Raises ProjectError for anything but plain files.

    Nothing in a bundle is trusted: not its size (it is read up to a limit and no further),
    not its headers (a sparse member can claim a terabyte of holes), and not its paths.
    Every file lands inside `dest`, and a name `pack` would never have packed (.git/config,
    .envrc, anything under a dot) is not written at all. Returns the names left out.
    """
    dest = dest.resolve()
    if len(blob) > MAX_BUNDLE_BYTES:
        raise ProjectError(f"the bundle is over {MAX_BUNDLE_BYTES >> 20} MB")
    try:
        raw = unzstd(blob, MAX_UNPACKED_BYTES)
    except zstandard.ZstdError:
        raise ProjectError("the bundle is not zstd-compressed") from None
    if len(raw) > MAX_UNPACKED_BYTES:
        raise ProjectError(f"the bundle unpacks to more than {MAX_UNPACKED_BYTES >> 20} MB")
    dest.mkdir(parents=True, exist_ok=True)
    left_out: list[str] = []
    seen: set[str] = set()
    count = total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            for info in tar:
                target = (dest / info.name).resolve()
                if not info.isreg() or info.sparse is not None or target == dest or not target.is_relative_to(dest):
                    raise ProjectError(f"the bundle contains an entry that is not a plain file: {info.name[:80]!r}")
                # One name, one file, written one way. Of two files with a name the later
                # would be the one that runs, and the site shows only the first: what
                # everyone reads and what a verifier runs must be the same bytes.
                if posixpath.normpath(info.name) != info.name or info.name in seen:
                    raise ProjectError(f"the bundle names a file twice, or by a roundabout path: {info.name[:80]!r}")
                seen.add(info.name)
                count += 1
                total += info.size
                if count > MAX_BUNDLE_FILES:
                    raise ProjectError(f"the bundle has more than {MAX_BUNDLE_FILES:,} files")
                if total > MAX_UNPACKED_BYTES:
                    raise ProjectError(f"the bundle unpacks to more than {MAX_UNPACKED_BYTES >> 20} MB")
                if skip_reason(target.relative_to(dest)):
                    left_out.append(info.name[:200])
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    dst = open(target, "wb")
                except (FileExistsError, NotADirectoryError, IsADirectoryError):
                    raise ProjectError(f"the bundle uses one name for a file and for a directory: {info.name[:80]!r}") from None
                with tar.extractfile(info) as src, dst:
                    left = info.size
                    while left > 0 and (chunk := src.read(min(1 << 20, left))):
                        dst.write(chunk)
                        left -= len(chunk)
                target.chmod(0o755 if info.mode & 0o100 else 0o644)
    except tarfile.TarError:
        raise ProjectError("the bundle is not a tar archive, or is damaged") from None
    return left_out
