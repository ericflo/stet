"""The held-out exam goes between the organisers' machines sealed to each one's key."""

import hashlib
import os
import stat
import types

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stet import client, data, sealed
from stet.keys import Key
from stet.spec import load_season

RAW = (serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(*RAW)


def test_a_sealed_file_opens_for_the_key_it_was_sealed_to_and_no_other():
    mine, yours = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    secret = os.urandom(5000)
    blob = sealed.seal(secret, public(mine))
    assert sealed.unseal(blob, mine) == secret
    assert secret[:64] not in blob and blob.startswith(sealed.MAGIC)
    assert sealed.seal(secret, public(mine)) != blob  # a fresh key each time: two copies say nothing about each other
    with pytest.raises(sealed.SealError, match="not sealed to this machine's key"):
        sealed.unseal(blob, yours)
    for damaged in (blob[:-1], blob[:40] + bytes([blob[40] ^ 1]) + blob[41:], blob[:-10] + bytes(10)):
        with pytest.raises(sealed.SealError):
            sealed.unseal(damaged, mine)
    for not_sealed in (b"", secret, sealed.MAGIC, b"x" * 200):
        with pytest.raises(sealed.SealError, match="not a sealed file"):
            sealed.unseal(not_sealed, mine)
    with pytest.raises(sealed.SealError):
        sealed.seal(secret, b"short")


def test_an_accounts_signing_key_and_its_sealing_key_are_one_key():
    # The public half is worked out from the public key alone, the private half from the
    # private key alone. They have to be the two halves of one key, for every key.
    for _ in range(50):
        key = Ed25519PrivateKey.generate()
        assert sealed.x25519_public(public(key)) == sealed.x25519_private(key).public_key().public_bytes(*RAW)


class Server:
    """The three endpoints, as the server keeps them: a copy per organisers' key, unread."""

    def __init__(self, keys: dict[str, Key]):
        self.keys, self.kept, self.asked = keys, {}, []

    def request(self, method, path, body=None, key=None, raw=False, timeout=120, limit=0):
        self.asked.append((method, path))
        if (method, path) == ("GET", "/api/v1/admin/sealed"):
            return {"organisers": [{"key": k.public_hex, "handle": h, "sealed": k.public_hex in self.kept} for h, k in self.keys.items()]}
        if (method, path) == ("GET", "/api/v1/admin/sealed/mine"):
            if key.public_hex not in self.kept:
                raise client.ApiError(404, "no copy of the held-out exam is sealed to this machine's key yet")
            return self.kept[key.public_hex]
        if method == "PUT" and path.startswith("/api/v1/admin/sealed/"):
            self.kept[path.rsplit("/", 1)[1]] = bytes(body)
            return {}
        raise AssertionError(f"unexpected request: {method} {path}")


@pytest.fixture
def two_machines(tmp_path, monkeypatch):
    """Two organisers' machines and a season with a held-out exam; the first machine holds the pack."""
    season = load_season("s0")
    pack = os.urandom(20_000)
    monkeypatch.setitem(season.raw, "heldout_exam", {"sha256": hashlib.sha256(pack).hexdigest()})
    keys = {"one": Key(Ed25519PrivateKey.generate(), "one"), "two": Key(Ed25519PrivateKey.generate(), "two")}
    server = Server(keys)
    monkeypatch.setattr(client, "request", server.request)
    monkeypatch.setattr(client, "_shared_at", 0.0)

    def at(name):  # this process is now that machine
        monkeypatch.setenv("STET_CONFIG", str(tmp_path / name))
        monkeypatch.setattr(data, "_heldout_pack", {})
        return keys[name]

    one = at("one")
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "heldout-s0.jsonl.zst").write_bytes(pack)
    out = types.SimpleNamespace(lines=[], dim=lambda s: s)
    out.line = lambda tag, text: out.lines.append(text)
    return types.SimpleNamespace(season=season, pack=pack, server=server, at=at, one=one, two=keys["two"], out=out, dir=tmp_path)


def test_one_organisers_machine_shares_the_exam_and_the_other_fetches_it(two_machines):
    m = two_machines
    # A machine without the pack has nothing to share and nothing is there for it yet.
    two = m.at("two")
    assert client.share_heldout(two, m.season, m.out, every=0) == 0
    assert client.fetch_heldout(two, m.season, m.out, wait=False) is False
    assert data.heldout_pack_path(m.season) is None

    # The machine that holds it seals a copy to every organisers' key that has none.
    one = m.at("one")
    assert client.share_heldout(one, m.season, m.out, every=0) == 2
    assert set(m.server.kept) == {one.public_hex, two.public_hex}
    for blob in m.server.kept.values():  # what the server keeps says nothing it could read
        assert blob.startswith(sealed.MAGIC) and m.pack[:64] not in blob and m.pack[-64:] not in blob
    assert "shared the held-out exam with two" in " ".join(m.out.lines)
    assert client.share_heldout(one, m.season, m.out, every=0) == 0  # and not twice
    m.server.asked.clear()
    assert client.share_heldout(one, m.season, m.out) == 0 and m.server.asked == []  # nor asked about on every poll

    # The other fetches the copy sealed to it, checks it is the season's pack, and keeps it to itself.
    two = m.at("two")
    assert client.fetch_heldout(two, m.season, m.out, wait=False) is True
    dest = m.dir / "two" / "heldout-s0.jsonl.zst"
    assert dest.read_bytes() == m.pack and data.heldout_pack_path(m.season) == dest
    if os.name != "nt":
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600
    assert not list(dest.parent.glob("*.tmp*"))


def test_nothing_is_put_in_place_that_is_not_the_seasons_exam_sealed_to_this_machine(two_machines, capsys):
    m = two_machines
    one, two = m.one, m.at("two")
    dest = m.dir / "two" / "heldout-s0.jsonl.zst"
    # A copy sealed to another machine, handed to this one.
    m.server.kept[two.public_hex] = sealed.seal(m.pack, bytes.fromhex(one.public_hex))
    assert client.fetch_heldout(two, m.season, m.out, wait=False) is False and not dest.exists()
    assert "not sealed to this machine's key" in capsys.readouterr().err
    # Something else, sealed to this machine by whoever could reach the server.
    m.server.kept[two.public_hex] = sealed.seal(b"an exam of somebody's own choosing", bytes.fromhex(two.public_hex))
    assert client.fetch_heldout(two, m.season, m.out, wait=False) is False and not dest.exists()
    assert "not the held-out exam the season committed to" in capsys.readouterr().err
    # The pack itself, unsealed: not taken either.
    m.server.kept[two.public_hex] = m.pack
    assert client.fetch_heldout(two, m.season, m.out, wait=False) is False and not dest.exists()


def test_a_machine_waits_for_its_copy_when_it_is_left_running(two_machines, monkeypatch):
    m = two_machines
    two = m.at("two")
    naps = []

    def nap(seconds):
        naps.append(seconds)
        if len(naps) == 3:  # meanwhile the other machine, which is running, has noticed the new key
            m.server.kept[two.public_hex] = sealed.seal(m.pack, bytes.fromhex(two.public_hex))

    monkeypatch.setattr(client.time, "sleep", nap)
    assert client.fetch_heldout(two, m.season, m.out) is True
    assert len(naps) == 3 and (m.dir / "two" / "heldout-s0.jsonl.zst").read_bytes() == m.pack
    assert sum("does not hold the season's held-out exam" in line for line in m.out.lines) == 1  # said once, not every minute


def test_a_machine_does_not_wait_for_what_waiting_will_not_bring(two_machines, monkeypatch):
    m = two_machines
    two = m.at("two")
    monkeypatch.setattr(client.time, "sleep", lambda s: pytest.fail("waited"))
    for status, why in ((403, "organisers only"), (426, "this copy of stet is out of date")):
        monkeypatch.setattr(client, "request", lambda *a, status=status, why=why, **k: (_ for _ in ()).throw(client.ApiError(status, why)))
        assert client.fetch_heldout(two, m.season, m.out) is False
