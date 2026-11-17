import dataclasses
import json
import os
import sys
import time

import pytest

from stet.runner import bwrap_available, collect_script, run_editor, scope_available
from stet.script import ScriptError
from stet.spec import load_season, practice_round

SEASON = load_season("s0")
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=512, ctx=64, batch_size=4)
RND = practice_round(SEASON, 0)


def editor(tmp_path, body: str):
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "editor.py").write_text(body)
    return bundle


WRITE_ALL = """
import json, os
task = json.load(open(os.environ["STET_TASK"]))
n = json.load(open(os.path.join(os.environ["STET_CORPUS"], "meta.json")))["n_docs"]
with open(os.path.join(os.environ["STET_OUT"], "script.jsonl"), "w") as f:
    for i in range(n):
        f.write(f"{i}\\n")
print("kept", n, "of", task["train_tokens"])
"""


def test_an_editor_runs_and_its_script_is_copied_out(corpus, tmp_path):
    run = run_editor(editor(tmp_path, WRITE_ALL), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 30)
    assert run.status == "ok" and run.script == (tmp_path / "work" / "script.jsonl").resolve()
    assert run.script.read_text().split() == ["0", "1", "2", "3"]
    assert "kept 4" in run.log_tail and "seed" not in json.dumps(json.load(open(tmp_path / "work" / "out" / "task.json")))


def test_no_script_a_crash_and_a_timeout(corpus, tmp_path):
    run = run_editor(editor(tmp_path, "print('nothing')"), "python editor.py", corpus, RND, SCALE, tmp_path / "w1", 30)
    assert run.status == "no_script" and run.script is None and "script.jsonl" in run.reason
    run = run_editor(editor(tmp_path, "raise SystemExit(3)"), "python editor.py", corpus, RND, SCALE, tmp_path / "w2", 30)
    assert run.status == "error" and run.exit_code == 3
    # A script written before the deadline still counts when the editor is killed at it.
    slow = WRITE_ALL + "\nimport time\ntime.sleep(60)\n"
    run = run_editor(editor(tmp_path, slow), "python editor.py", corpus, RND, SCALE, tmp_path / "w3", 2)
    assert run.status == "timeout" and run.script is not None and run.seconds < 10


@pytest.mark.skipif(os.name == "nt", reason="POSIX only")
def test_a_script_that_points_somewhere_else_is_not_read(corpus, tmp_path):
    secret = tmp_path / "secret.jsonl"
    secret.write_text("3\n")
    body = f"import os\nos.symlink({str(secret)!r}, os.path.join(os.environ['STET_OUT'], 'script.jsonl'))\n"
    run = run_editor(editor(tmp_path, body), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 30)
    assert run.script is None and run.status == "no_script"
    body = "import os\nos.mkfifo(os.path.join(os.environ['STET_OUT'], 'script.jsonl'))\n"
    run = run_editor(editor(tmp_path, body), "python editor.py", corpus, RND, SCALE, tmp_path / "work2", 30)
    assert run.script is None and "regular file" in run.reason
    out = tmp_path / "out"
    out.mkdir()
    (out / "script.jsonl").write_bytes(b"0\n" * 1000)
    with pytest.raises(ScriptError):
        collect_script(out, tmp_path, 100)


SNOOP = """
import json, os, socket, subprocess
seen = {}
for p in ("/etc/machine-id", "/etc/hostname", "/etc/hosts", "/etc/shadow", os.path.expanduser("~/.ssh"), "/home", "/sys/class/net", "/sys/class/dmi/id/product_name"):
    try:
        seen[p] = os.listdir(p) if os.path.isdir(p) else open(p).read()[:80]
    except OSError as e:
        seen[p] = None
seen["hostname"] = socket.gethostname()
seen["uid"] = os.getuid()
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2); seen["net"] = True
except OSError:
    seen["net"] = False
try:
    seen["userns"] = subprocess.run(["unshare", "-U", "true"], capture_output=True).returncode == 0
except OSError:
    seen["userns"] = False
seen["env"] = sorted(os.environ)
print("SNOOP" + json.dumps(seen))
with open(os.path.join(os.environ["STET_OUT"], "script.jsonl"), "w") as f:
    f.write("0\\n")
open("/scratch/ok", "w").write("x")
"""


@pytest.mark.skipif(sys.platform != "linux" or not bwrap_available()[0], reason="needs bubblewrap")
def test_the_sandbox_hides_the_machine(corpus, tmp_path):
    run = run_editor(editor(tmp_path, SNOOP), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 60, sandbox="bwrap")
    assert run.status == "ok", run.log_tail
    seen = json.loads(run.log_tail.split("SNOOP", 1)[1].splitlines()[0])
    assert seen["/etc/machine-id"] is None and seen["/etc/hostname"] is None and seen["/etc/hosts"] is None and seen["/etc/shadow"] is None
    assert seen["/home"] == [] and seen["/sys/class/net"] is None and seen["/sys/class/dmi/id/product_name"] is None
    assert seen["hostname"] == "stet" and seen["uid"] == 1000 and seen["net"] is False and seen["userns"] is False
    assert not [k for k in seen["env"] if k.startswith(("XDG", "DBUS", "SSH", "USER", "LOGNAME"))]
    assert run.script is not None and run.script.read_text() == "0\n"
    if scope_available()[0]:
        assert run.limits.get("memory_gb")


HOG = """
import os
with open(os.path.join(os.environ["STET_OUT"], "script.jsonl"), "w") as f:
    f.write("0\\n")
blocks = []
while True:
    blocks.append(bytearray(256 << 20))
"""


@pytest.mark.skipif(sys.platform != "linux" or not bwrap_available()[0] or not scope_available()[0], reason="needs bubblewrap and systemd-run --user")
def test_the_sandbox_caps_memory(corpus, tmp_path):
    run = run_editor(editor(tmp_path, HOG), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 120, sandbox="bwrap", limits={"memory_gb": 1, "required": True})
    assert run.status == "error" and run.seconds < 60  # killed by the cgroup, not by the deadline
    assert run.script is not None  # what it wrote before dying still counts


# --- whose fault a failure is ---------------------------------------------------------------
#
# An anchor hands a job back when the machine failed and reports it when the editor did. If
# an editor could make its own failure look like the machine's, its job would be retried
# for ever and nobody else's would run.

needs_sandbox = pytest.mark.skipif(sys.platform != "linux" or not bwrap_available()[0], reason="needs bubblewrap")


def test_a_bad_entry_command_is_the_editors_error(corpus, tmp_path):
    bundle = editor(tmp_path, WRITE_ALL)
    for n, entry in enumerate(['python "editor.py', "no-such-program-xyz editor.py", "python editor.py " + "x" * 5000, "./editor.py"]):
        run = run_editor(bundle, entry, corpus, RND, SCALE, tmp_path / f"w{n}", 30)
        assert run.status == "error" and run.script is None, entry
        assert "entry command" in run.reason, run.reason


@needs_sandbox
def test_in_the_sandbox_a_bad_entry_or_a_forged_message_is_still_the_editors_error(corpus, tmp_path):
    bundle = editor(tmp_path, "import sys\nprint('bwrap: execvp python: No such file or directory')\nsys.exit(1)\n")
    run = run_editor(bundle, "python editor.py", corpus, RND, SCALE, tmp_path / "w1", 30, sandbox="bwrap")
    assert run.status == "error" and run.exit_code == 1 and "bwrap:" in run.log_tail
    for n, entry in enumerate(["node index.js", "./editor.py"]):  # not installed; not executable
        run = run_editor(bundle, entry, corpus, RND, SCALE, tmp_path / f"v{n}", 30, sandbox="bwrap")
        assert run.status == "error" and run.exit_code == 127 and "cannot start" in run.log_tail, run


@needs_sandbox
def test_a_sandbox_that_does_not_start_is_the_machines_error(corpus, tmp_path, monkeypatch):
    from stet import runner

    real = runner.bwrap_command
    monkeypatch.setattr(runner, "bwrap_command", lambda *a: [*real(*a), "--ro-bind", "/no/such/directory", "/nowhere"])
    with pytest.raises(runner.HarnessError, match="failed to start"):
        run_editor(editor(tmp_path, WRITE_ALL), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 30, sandbox="bwrap")


@pytest.mark.skipif(os.name == "nt", reason="POSIX only")
def test_nothing_private_is_mounted_where_an_editor_could_read_it(corpus, tmp_path, monkeypatch):
    from stet import runner

    shown = [str(tmp_path / "site-packages"), str(tmp_path / "corpus")]  # directories the sandbox would mount
    assert runner._exposed(SEASON, shown) == ""  # the cache, the key and the held-out exam are all elsewhere
    monkeypatch.setenv("STET_HELDOUT_EXAM", str(tmp_path / "site-packages" / "data" / "heldout-s0.jsonl.zst"))
    assert "the held-out exam" in runner._exposed(SEASON, shown)
    monkeypatch.delenv("STET_HELDOUT_EXAM")
    monkeypatch.setenv("STET_CONFIG", "/usr/share/stet")
    assert "the config directory" in runner._exposed(SEASON, shown)
    # A link is followed to where it leads.
    os.symlink(tmp_path / "corpus", tmp_path / "elsewhere")
    monkeypatch.setenv("STET_CONFIG", str(tmp_path / "elsewhere" / "stet"))
    assert "the config directory" in runner._exposed(SEASON, shown)
    monkeypatch.setenv("STET_CONFIG", str(tmp_path / "stet-config"))
    monkeypatch.setenv("STET_CACHE", str(tmp_path / "corpus"))
    assert "the cache" in runner._exposed(SEASON, shown)
    if sys.platform == "linux" and bwrap_available()[0]:  # and nothing is started in that case
        with pytest.raises(runner.HarnessError, match="refusing to sandbox: the cache"):
            run_editor(editor(tmp_path, WRITE_ALL), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 30, sandbox="bwrap")


def test_a_full_disk_is_the_machines_error(corpus, tmp_path, monkeypatch):
    from stet import runner

    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 30})())
    with pytest.raises(runner.HarnessError, match="free"):
        run_editor(editor(tmp_path, WRITE_ALL), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 30)


# --- the output directory ---------------------------------------------------------------------

HIDE = """
import os, time
out = os.environ["STET_OUT"]
with open(os.path.join(out, "script.jsonl"), "w") as f:
    f.write("0\\n")
os.mkdir(os.path.join(out, "x"), 0o300)  # writable, not listable
with open(os.path.join(out, "x", "hoard"), "wb") as f:
    f.truncate(8 << 20)
time.sleep(60)
"""

LITTER = """
import os, time
out = os.environ["STET_OUT"]
for i in range(12000):
    open(os.path.join(out, f"f{i}"), "w").close()
time.sleep(60)
"""


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions that apply to this user")
def test_an_editor_cannot_hide_what_it_writes_and_it_can_be_removed(corpus, tmp_path):
    from stet.runner import remove_tree

    work = tmp_path / "work"
    run = run_editor(editor(tmp_path, HIDE), "python editor.py", corpus, RND, SCALE, work, 30)
    assert run.status == "error" and run.seconds < 15
    assert run.script is not None  # what it wrote before being stopped still counts
    assert (work / "out" / "x").exists() and remove_tree(work) and not work.exists()


def _dig(root, names, bottom_mode=None):
    """Make a chain of directories under root by name alone, as an editor would with mkdir and
    chdir: no path string ever has to hold the whole way down. A file goes at the bottom."""
    fd = os.open(root, os.O_RDONLY)
    for name in names:
        os.mkdir(name, dir_fd=fd)
        below = os.open(name, os.O_RDONLY, dir_fd=fd)
        os.close(fd)
        fd = below
    os.mkdir("vault", dir_fd=fd)
    vault = os.open("vault", os.O_RDONLY, dir_fd=fd)
    os.close(os.open("hoard", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=vault))
    os.symlink("/etc", "elsewhere", dir_fd=vault)
    os.close(vault)
    if bottom_mode is not None:
        os.chmod("vault", bottom_mode, dir_fd=fd)
    os.close(fd)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions that apply to this user")
def test_whatever_shape_an_editor_leaves_its_directory_in_it_can_be_removed(tmp_path):
    from stet.runner import remove_tree

    # Twenty names of 255 bytes: the way down is longer than any path the kernel accepts,
    # and the directory at the end of it can be neither listed nor entered.
    long = tmp_path / "long"
    long.mkdir()
    _dig(long, ["d" * 255] * 20, bottom_mode=0)
    with pytest.raises(OSError):
        os.stat(str(long) + ("/" + "d" * 255) * 20)  # ENAMETOOLONG: nothing gets there by path
    assert remove_tree(long) and not long.exists()

    # Deeper than this process may hold directories open, or Python may recurse.
    deep = tmp_path / "deep"
    deep.mkdir()
    _dig(deep, ["d"] * 3000)
    assert remove_tree(deep) and not deep.exists()

    # Links go; what they point at stays. So does a file where a directory was expected.
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "thesis.txt").write_text("three years")
    mixed = tmp_path / "mixed"
    (mixed / "a" / "b").mkdir(parents=True)
    os.symlink(keep, mixed / "a" / "link")
    (mixed / "a" / "b").chmod(0o300)
    assert remove_tree(mixed) and (keep / "thesis.txt").exists()
    os.symlink(keep, tmp_path / "top")
    assert remove_tree(tmp_path / "top") and (keep / "thesis.txt").exists()
    assert remove_tree(tmp_path / "never-there")


def test_an_editor_cannot_bury_the_harness_in_files(corpus, tmp_path):
    run = run_editor(editor(tmp_path, LITTER), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 30)
    assert run.status == "error" and "files" in run.reason and run.seconds < 20


def test_the_clock_does_not_wait_for_the_accounting(corpus, tmp_path, monkeypatch):
    from stet import runner

    def slow(path, max_entries):  # a walk over a directory the editor has stuffed
        time.sleep(30)
        return 0, 0, True

    monkeypatch.setattr(runner, "_usage", slow)
    run = run_editor(editor(tmp_path, WRITE_ALL + "\nimport time\ntime.sleep(60)\n"), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 3)
    assert run.status == "timeout" and run.seconds < 8


def test_usage_counts_what_a_file_claims_and_notices_what_it_cannot_read(tmp_path):
    from stet.runner import _usage

    (tmp_path / "a").mkdir()
    with open(tmp_path / "a" / "sparse", "wb") as f:
        f.truncate(5 << 20)
    (tmp_path / "b").write_bytes(b"x" * 1000)
    size, entries, readable = _usage(tmp_path, 100)
    assert size >= (5 << 20) + 1000 and entries == 3 and readable
    assert _usage(tmp_path, 2)[1] == 3  # stopped counting
    if os.name != "nt" and os.geteuid() != 0:
        (tmp_path / "a").chmod(0o300)
        assert _usage(tmp_path, 100)[2] is False
        (tmp_path / "a").chmod(0o700)


@pytest.mark.skipif(sys.platform != "linux" or not bwrap_available()[0] or not scope_available()[0], reason="needs bubblewrap and systemd-run --user")
def test_the_sandbox_self_test_passes_here(tmp_path, monkeypatch):
    from stet.runner import sandbox_selftest

    monkeypatch.setenv("STET_CACHE", str(tmp_path / "cache"))
    assert sandbox_selftest() == (True, "")
