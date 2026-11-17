"""An editor project on disk: stet.toml, and how it is packed into a bundle.

A bundle is a deterministic tar of the project directory, zstd-compressed. Its sha256 is
the editor's identity: the leaderboard, the verifiers and the audit trail all refer to
exactly those bytes.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import zstandard

MAX_BUNDLE_BYTES = 50 << 20
MAX_ASSET_BYTES = 2 << 30
EXCLUDE_DIRS = {".git", ".hg", ".stet", ".venv", "venv", "__pycache__", "node_modules", "runs", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".DS_Store")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")


class ProjectError(ValueError):
    pass


@dataclass
class Asset:
    url: str
    sha256: str
    path: str


@dataclass
class Project:
    path: Path
    name: str
    entry: str
    description: str = ""
    technique: str = ""
    parent: str = ""
    license: str = "MIT"
    assets: list[Asset] = field(default_factory=list)

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "entry": self.entry,
            "description": self.description,
            "technique": self.technique,
            "parent": self.parent,
            "license": self.license,
            "assets": [a.__dict__ for a in self.assets],
        }


def load_project(path: str | Path) -> Project:
    path = Path(path).resolve()
    toml = path / "stet.toml"
    if not toml.exists():
        raise ProjectError(f"{path} has no stet.toml. Run `stet init` to create an editor project.")
    try:
        cfg = tomllib.loads(toml.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ProjectError(f"stet.toml is not valid TOML: {e}") from None
    name = str(cfg.get("name", "")).strip()
    if not NAME_RE.match(name):
        raise ProjectError("stet.toml: name must be 3-40 characters of lowercase letters, digits and hyphens")
    entry = str(cfg.get("entry", "")).strip()
    if not entry:
        raise ProjectError('stet.toml: entry is required, e.g. entry = "python editor.py"')
    assets = []
    for a in cfg.get("assets", []) or []:
        url, sha, rel = str(a.get("url", "")), str(a.get("sha256", "")).lower(), str(a.get("path", ""))
        if not url.startswith("https://"):
            raise ProjectError(f"stet.toml: asset url must be https ({url!r})")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ProjectError(f"stet.toml: asset {url} needs a sha256")
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise ProjectError(f"stet.toml: asset path must be relative and inside the project ({rel!r})")
        assets.append(Asset(url, sha, rel))
    return Project(
        path=path,
        name=name,
        entry=entry,
        description=str(cfg.get("description", "")).strip()[:300],
        technique=str(cfg.get("technique", "")).strip()[:120],
        parent=str(cfg.get("parent", "")).strip(),
        license=str(cfg.get("license", "MIT")).strip(),
        assets=assets,
    )


def bundle_files(project: Project) -> list[Path]:
    asset_paths = {(project.path / a.path).resolve() for a in project.assets}
    out = []
    for p in sorted(project.path.rglob("*")):
        rel = p.relative_to(project.path)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.is_symlink() or not p.is_file():
            continue
        if p.name.endswith(EXCLUDE_SUFFIXES) or p.resolve() in asset_paths:
            continue
        out.append(p)
    return out


def pack(project: Project) -> tuple[bytes, str]:
    """Return (bundle bytes, editor id). Same files in, same bytes out."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in bundle_files(project):
            info = tarfile.TarInfo(name=p.relative_to(project.path).as_posix())
            data = p.read_bytes()
            info.size = len(data)
            info.mode = 0o755 if p.stat().st_mode & 0o100 else 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    blob = zstandard.ZstdCompressor(level=10).compress(buf.getvalue())
    if len(blob) > MAX_BUNDLE_BYTES:
        raise ProjectError(
            f"bundle is {len(blob) / 1e6:.1f} MB; the limit is {MAX_BUNDLE_BYTES >> 20} MB. "
            "Put weights and other large files in [[assets]] (fetched by URL and sha256)."
        )
    return blob, hashlib.sha256(blob).hexdigest()


def unpack(blob: bytes, dest: Path) -> None:
    """Extract a bundle. Refuses anything that is not a plain file inside dest."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    raw = zstandard.ZstdDecompressor().decompress(blob, max_output_size=8 * MAX_BUNDLE_BYTES)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        for info in tar:
            target = (dest / info.name).resolve()
            if not info.isfile() or not str(target).startswith(str(dest) + "/"):
                raise ProjectError(f"bundle contains an unsafe entry: {info.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(info) as src:
                target.write_bytes(src.read())
            target.chmod(0o755 if info.mode & 0o100 else 0o644)
