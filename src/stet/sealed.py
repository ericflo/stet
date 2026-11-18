"""A file sealed to one account's key: only that account's machine can open it.

The organisers' machines pass the season's held-out exam between themselves this way. The
server and its file store hold a copy for each of those machines, and can open none of
them.

An account's key is an Ed25519 signing key. The same key, read as a point on Curve25519,
is one that a secret can be agreed with (as libsodium's sealed boxes do), so nothing new
has to be made, registered or carried from one machine to another:

    sealed = magic | ephemeral X25519 public key | nonce | ChaCha20-Poly1305(file)

with the cipher's key drawn by HKDF-SHA256 from the X25519 secret shared between a fresh
ephemeral key and the recipient's.
"""

from __future__ import annotations

import hashlib
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"stet-sealed-1\n"
P = 2**255 - 19


class SealError(ValueError):
    """Not a sealed file, or not one this key opens."""


def x25519_public(ed_public: bytes) -> bytes:
    """The Curve25519 form of an Ed25519 public key: u = (1 + y) / (1 - y)."""
    if len(ed_public) != 32:
        raise SealError("a public key is 32 bytes")
    y = int.from_bytes(ed_public, "little") & ((1 << 255) - 1)
    if y >= P or y == 1:
        raise SealError("not a public key a file can be sealed to")
    return ((1 + y) * pow(1 - y, -1, P) % P).to_bytes(32, "little")


def x25519_private(ed_private: Ed25519PrivateKey) -> X25519PrivateKey:
    """The Curve25519 form of an Ed25519 private key: the scalar it signs with."""
    seed = ed_private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    h = bytearray(hashlib.sha512(seed).digest()[:32])
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    return X25519PrivateKey.from_private_bytes(bytes(h))


def _cipher(shared: bytes, ephemeral: bytes, recipient: bytes) -> ChaCha20Poly1305:
    return ChaCha20Poly1305(HKDF(hashes.SHA256(), 32, salt=None, info=MAGIC + ephemeral + recipient).derive(shared))


def seal(plain: bytes, ed_public: bytes) -> bytes:
    """Seal `plain` so that only the holder of this Ed25519 public key's private half opens it."""
    recipient = x25519_public(ed_public)
    ephemeral = X25519PrivateKey.generate()
    eph_pub = ephemeral.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    try:
        shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient))
    except ValueError as e:  # a point of small order: no key anyone holds
        raise SealError(f"not a public key a file can be sealed to ({e})") from None
    nonce = os.urandom(12)
    return MAGIC + eph_pub + nonce + _cipher(shared, eph_pub, recipient).encrypt(nonce, plain, MAGIC)


def unseal(blob: bytes, ed_private: Ed25519PrivateKey) -> bytes:
    """Open what was sealed to this key. Raises SealError for anything else."""
    head = len(MAGIC)
    if len(blob) < head + 32 + 12 + 16 or blob[:head] != MAGIC:
        raise SealError("not a sealed file")
    eph_pub, nonce, body = blob[head : head + 32], blob[head + 32 : head + 44], blob[head + 44 :]
    mine = x25519_private(ed_private)
    recipient = mine.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    try:
        shared = mine.exchange(X25519PublicKey.from_public_bytes(eph_pub))
        return _cipher(shared, eph_pub, recipient).decrypt(nonce, body, MAGIC)
    except (InvalidTag, ValueError):
        raise SealError("this file was not sealed to this machine's key, or it is damaged") from None
