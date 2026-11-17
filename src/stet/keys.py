"""Your identity: an Ed25519 key in your config directory.

There are no passwords and no browser logins. The key signs every request that needs to
be attributed to you: submitting an editor, claiming a verification job, reporting a
result. Back the file up if your standing matters to you; it cannot be recovered.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def config_dir() -> Path:
    base = os.environ.get("STET_CONFIG") or os.path.join(os.environ.get("XDG_CONFIG_HOME") or "~/.config", "stet")
    return Path(base).expanduser()


def key_path() -> Path:
    return config_dir() / "key.json"


class Key:
    def __init__(self, private: Ed25519PrivateKey, handle: str = ""):
        self._private = private
        self.handle = handle

    @property
    def public_hex(self) -> str:
        raw = self._private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return raw.hex()

    def sign(self, message: bytes) -> str:
        return self._private.sign(message).hex()

    def headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        """Signature headers for one request. `path` includes the query string."""
        now = str(int(time.time()))
        message = "\n".join(["stet-v1", method.upper(), path, now, hashlib.sha256(body).hexdigest()]).encode()
        return {"X-Stet-Key": self.public_hex, "X-Stet-Time": now, "X-Stet-Sig": self.sign(message)}

    def save(self) -> None:
        path = key_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self._private.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"private": raw.hex(), "public": self.public_hex, "handle": self.handle}, f, indent=1)
        tmp.replace(path)


def load_key() -> Key | None:
    path = key_path()
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return Key(Ed25519PrivateKey.from_private_bytes(bytes.fromhex(d["private"])), d.get("handle", ""))


def create_key() -> Key:
    return Key(Ed25519PrivateKey.generate())
