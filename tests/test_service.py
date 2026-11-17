"""`stet service`: the unit a volunteer's machine is left running under."""

import subprocess

import pytest

from stet import service


def test_the_unit_runs_verify_with_what_was_asked_for(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    text = service.unit_text("/home/someone/.local/bin/stet", edits=True, environ={})
    assert 'ExecStart="/home/someone/.local/bin/stet" "verify" "--edits"' in text
    assert "Restart=always" in text and "WantedBy=default.target" in text
    assert f"StandardOutput=append:{tmp_path / 'state' / 'stet' / 'stet-verify.log'}" in text
    assert "--no-upgrade" not in text  # upgrades are on unless the owner says otherwise
    assert "--no-upgrade" in service.unit_text("/x/stet", no_upgrade=True, environ={})
    plain = service.unit_text("/x/stet", environ={})
    assert 'ExecStart="/x/stet" "verify"\n' in plain and "sandbox" not in plain


def test_the_unit_carries_the_account_and_the_gpu_the_shell_chose():
    """A service starts with none of the shell's environment: a second verifier on a second
    GPU must keep its own account and its own card."""
    text = service.unit_text("/x/stet", environ={"STET_CONFIG": "/home/a/second", "CUDA_VISIBLE_DEVICES": "1", "HOME": "/home/a", "SECRET": "no"})
    assert 'Environment="STET_CONFIG=/home/a/second"' in text
    assert 'Environment="CUDA_VISIBLE_DEVICES=1"' in text
    assert "SECRET" not in text and "HOME=" not in text
    assert 'Environment="PATH=/x:' in text  # the installed stet, then uv, then the system's


def test_a_percent_or_a_quote_in_a_path_cannot_become_a_systemd_directive():
    text = service.unit_text('/odd/100%/st"et', environ={"STET_CONFIG": "/a b/%h"})
    assert 'ExecStart="/odd/100%%/st\\"et" "verify"' in text
    assert 'Environment="STET_CONFIG=/a b/%%h"' in text


def test_install_writes_the_unit_and_starts_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(service, "supported", lambda: (True, ""))
    monkeypatch.setattr(service.upgrade, "launcher", lambda: "/home/someone/.local/bin/stet")
    monkeypatch.setattr(service, "systemctl", lambda *a, check=True: calls.append(a) or subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(service, "linger", lambda: (False, "It will stop when you log out."))
    done = service.install(edits=True)
    unit = tmp_path / "config" / "systemd" / "user" / "stet-verify.service"
    assert unit.read_text().count("ExecStart=") == 1 and '"--edits"' in unit.read_text()
    assert calls == [("daemon-reload",), ("enable", "stet-verify.service"), ("restart", "stet-verify.service")]
    assert done["note"] and not done["linger"]  # the owner is told when it will not outlive their login
    assert service.status()["installed"]
    assert service.uninstall() and not unit.exists()
    assert not service.uninstall()


def test_a_second_verifier_gets_a_unit_of_its_own(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(service, "supported", lambda: (True, ""))
    monkeypatch.setattr(service.upgrade, "launcher", lambda: "/x/stet")
    monkeypatch.setattr(service, "systemctl", lambda *a, check=True: subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(service, "linger", lambda: (True, ""))
    service.install("stet-verify-2")
    assert (tmp_path / "systemd" / "user" / "stet-verify-2.service").exists()
    with pytest.raises(service.ServiceError):
        service.install("../etc/evil")


def test_where_there_is_no_systemd_it_says_what_to_do_instead(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "darwin")
    ok, why = service.supported()
    assert not ok and "stet verify" in why
    with pytest.raises(service.ServiceError):
        service.install()


def test_the_logs_are_the_end_of_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert service.tail() == []
    (tmp_path / "stet").mkdir()
    (tmp_path / "stet" / "stet-verify.log").write_text("".join(f"line {i}\n" for i in range(100)))
    assert service.tail(lines=3) == ["line 97", "line 98", "line 99"]
