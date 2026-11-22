"""A verifier left running keeps itself current, between jobs, unless it is told not to."""

import argparse
import sys

import pytest

from stet import __version__, upgrade


def ahead(by: int = 1) -> str:
    major, minor, patch = (list(upgrade.parse(__version__)) + [0, 0, 0])[:3]
    return f"{major}.{minor}.{patch + by}"


def info(version: str) -> dict:
    return {"site": "https://site.test", "cli": {"version": version, "wheel": "https://api.test/files/stet.whl"}}


@pytest.fixture
def installer(monkeypatch):
    """An install made by the installer, an installer that works, and a restart that is recorded."""
    seen = argparse.Namespace(installs=[], restarts=[], said=[], version=ahead(), ok=True)
    monkeypatch.delenv("STET_DEV", raising=False)
    monkeypatch.delenv("STET_AUTO_UPGRADE", raising=False)
    monkeypatch.setattr(upgrade, "can", lambda: (True, ""))
    monkeypatch.setattr(upgrade, "launcher", lambda: "/home/u/.local/bin/stet")
    monkeypatch.setattr(upgrade, "install", lambda site, env=None, log=None: (seen.installs.append(site), (seen.ok, "the end of what it said"))[1])
    monkeypatch.setattr(upgrade, "installed_version", lambda path: seen.version)
    monkeypatch.setattr(upgrade, "restart", lambda path: seen.restarts.append(path))
    return seen


def args(**k):
    return argparse.Namespace(**{"no_upgrade": False, **k})


def test_versions_compare_as_numbers_and_ignore_what_follows_them():
    assert upgrade.newer("0.3.10", "0.3.9") and upgrade.newer("1.0", "0.99.99") and upgrade.newer("0.4.0+rocm", "0.3.9")
    assert not upgrade.newer("0.3.1", "0.3.1") and not upgrade.newer("0.3.0", "0.3.1")
    assert not upgrade.newer("", "0.3.1") and not upgrade.newer("latest", "0.3.1")  # nothing to compare is not newer


def test_a_newer_stet_is_installed_and_the_verifier_starts_again(installer):
    up = upgrade.Upgrader(args(), say=installer.said.append)
    assert up.consider(info(ahead())) is True
    assert installer.installs == ["https://site.test"] and installer.restarts == ["/home/u/.local/bin/stet"]
    assert "--no-upgrade" in " ".join(installer.said)  # it says how to turn this off, where the owner will read it


def test_the_same_or_an_older_version_changes_nothing(installer):
    up = upgrade.Upgrader(args(), say=installer.said.append)
    assert up.consider(info(__version__)) is False and up.consider(info("0.0.1")) is False and up.consider({}) is False
    assert installer.installs == [] and installer.restarts == [] and installer.said == []


@pytest.mark.parametrize("how", ["flag", "env", "dev"])
def test_it_can_be_turned_off_and_then_only_says_a_version_is_out(installer, monkeypatch, how):
    if how == "env":
        monkeypatch.setenv("STET_AUTO_UPGRADE", "0")
    if how == "dev":
        monkeypatch.setenv("STET_DEV", "1")
    up = upgrade.Upgrader(args(no_upgrade=how == "flag"), say=installer.said.append)
    assert up.consider(info(ahead())) is False and up.consider(info(ahead())) is False
    assert installer.installs == [] and installer.restarts == []
    assert len(installer.said) == 1 and "install.sh" in installer.said[0]  # said once, with what to run by hand


def test_an_install_the_installer_did_not_make_is_left_alone(installer, monkeypatch):
    monkeypatch.setattr(upgrade, "can", lambda: (False, "this stet was not installed by the installer"))
    up = upgrade.Upgrader(args(), say=installer.said.append)
    assert up.consider(info(ahead())) is False
    assert installer.installs == [] and "by hand" in installer.said[0]


def test_an_upgrade_that_fails_keeps_the_old_version_running_and_waits_before_trying_again(installer, monkeypatch):
    installer.ok = False
    clock = [1000.0]
    monkeypatch.setattr(upgrade.time, "monotonic", lambda: clock[0])
    up = upgrade.Upgrader(args(), say=installer.said.append)
    assert up.consider(info(ahead())) is False and installer.restarts == []
    assert "did not work" in installer.said[-1] and __version__ in installer.said[-1]
    assert up.consider(info(ahead())) is False and len(installer.installs) == 1  # not again at once
    clock[0] += upgrade.RETRY_AFTER + 1
    installer.ok = True
    assert up.consider(info(ahead())) is True and len(installer.installs) == 2 and installer.restarts


def test_an_installer_that_says_it_worked_but_left_the_old_version_is_not_restarted_into(installer):
    installer.version = __version__  # nothing changed on disk: starting again would upgrade again, for ever
    up = upgrade.Upgrader(args(), say=installer.said.append)
    assert up.consider(info(ahead())) is False and installer.restarts == []


def test_it_looks_only_now_and_then_and_a_server_that_is_away_is_not_an_error(installer, monkeypatch):
    clock, asked = [0.0], []
    monkeypatch.setattr(upgrade.time, "monotonic", lambda: clock[0])

    def fetch():
        asked.append(clock[0])
        if len(asked) == 1:
            raise OSError("away")
        return info(ahead())

    up = upgrade.Upgrader(args(), say=installer.said.append, fetch_info=fetch)
    assert up.look() is False and asked == []  # the caller has just read the season
    clock[0] += upgrade.CHECK_EVERY + 1
    assert up.look() is False and len(asked) == 1  # away: carry on
    assert up.look() is False and len(asked) == 1  # and not asked again at once
    clock[0] += upgrade.CHECK_EVERY + 1
    assert up.look() is True and installer.restarts
    assert up.now() is True and len(asked) == 3  # refused for its version: ask whatever the clock says


def test_the_installer_is_told_it_is_upgrading_and_keeps_this_machines_build_of_pytorch(monkeypatch, tmp_path):
    ran = {}

    def run(cmd, env=None, **k):
        ran["cmd"], ran["env"], ran["script"] = cmd, env, open(cmd[1], "rb").read()
        return argparse.Namespace(returncode=0, stdout="Installing...\nDone. Next:\n")

    monkeypatch.setattr(upgrade, "fetch_installer", lambda site: b"#!/bin/sh\necho hi\n")
    monkeypatch.setattr(upgrade.subprocess, "run", run)
    monkeypatch.setattr(upgrade, "find_uv", lambda: "/opt/uv/bin/uv")
    monkeypatch.setattr(upgrade, "has_torch", lambda: True)
    monkeypatch.setattr(upgrade, "torch_index", lambda env: "https://download.test/rocm")
    monkeypatch.delenv("STET_TORCH_INDEX", raising=False)
    ok, tail = upgrade.install("https://site.test", {})
    assert ok and "Done" in tail and ran["cmd"][0] == "sh" and ran["script"].startswith(b"#!/bin/sh")
    assert ran["env"]["STET_UPGRADE"] == "1" and ran["env"]["STET_TORCH_INDEX"] == "https://download.test/rocm"
    assert ran["env"]["PATH"].startswith("/opt/uv/bin") and "STET_LIGHT" not in ran["env"]
    monkeypatch.setattr(upgrade, "has_torch", lambda: False)  # a light install stays light
    upgrade.install("https://site.test", {})
    assert ran["env"]["STET_LIGHT"] == "1"


def test_an_installer_is_only_fetched_over_tls_and_only_if_it_is_one(monkeypatch):
    with pytest.raises(OSError, match="not https"):
        upgrade.fetch_installer("http://site.test")

    class Answer:
        def __init__(self, body):
            self.body = body

        def read(self, n):
            return self.body[:n]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(upgrade.urllib.request, "urlopen", lambda req, timeout=0: Answer(b"<html>not found</html>"))
    with pytest.raises(OSError, match="not the installer"):
        upgrade.fetch_installer("https://site.test")
    monkeypatch.setattr(upgrade.urllib.request, "urlopen", lambda req, timeout=0: Answer(b"#!/bin/sh\nset -eu\n"))
    assert upgrade.fetch_installer("https://site.test").startswith(b"#!/bin/sh")
    # plain http is for this machine alone, and a name that only begins like it is another machine
    for mine in ("http://localhost", "http://localhost:8000/", "http://127.0.0.1:8000"):
        assert upgrade.fetch_installer(mine).startswith(b"#!/bin/sh")
    for other in ("http://localhost.attacker.example", "http://127.0.0.1.attacker.example", "http://localhost@attacker.example",
                  "http://127.0.0.10", "ftp://localhost", "https://", "localhost"):  # fmt: skip
        with pytest.raises(OSError, match="not https"):
            upgrade.fetch_installer(other)


def test_a_checkout_is_not_something_the_installer_made():
    if (upgrade.Path(sys.prefix) / "uv-receipt.toml").is_file():
        pytest.skip("these tests are running from an installed stet")
    ok, why = upgrade.can()
    assert not ok and "installer" in why
