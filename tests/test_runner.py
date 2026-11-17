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


# --- an editor does not outlive its run, and what it leaves is removed with care -----------------

LINGER = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(os.path.join(os.environ["STET_OUT"], "pids.tmp"), "w") as f:
    f.write(f"{os.getpid()} {child.pid}")
os.rename(os.path.join(os.environ["STET_OUT"], "pids.tmp"), os.path.join(os.environ["STET_OUT"], "pids"))
time.sleep(120)
"""


def _gone(pid: int) -> bool:
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:  # killed, and waiting for whoever adopted it to notice
            if open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except OSError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(os.name == "nt" or not os.path.isdir("/proc/self"), reason="needs process groups and /proc")
@pytest.mark.parametrize("trouble", [KeyboardInterrupt(), OSError(5, "Input/output error")], ids=["ctrl-c", "a disk that cannot be asked"])
def test_no_way_out_of_a_run_leaves_the_editor_running(corpus, tmp_path, monkeypatch, trouble):
    from stet import runner

    work = tmp_path / "work"
    pids = work / "out" / "pids"
    real = runner.shutil.disk_usage

    def disk_usage(path):
        if pids.exists():
            raise trouble  # in the middle of the run, with the editor and its child alive
        return real(path)

    monkeypatch.setattr(runner.shutil, "disk_usage", disk_usage)
    with pytest.raises(type(trouble)):
        run_editor(editor(tmp_path, LINGER), "python editor.py", corpus, RND, SCALE, work, 60)
    monkeypatch.undo()
    # Whoever called is about to remove the work directory. Nothing is left to write in it.
    for pid in map(int, pids.read_text().split()):
        assert _gone(pid)
    assert runner.remove_tree(work) and not work.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX only")
def test_a_directory_moved_while_it_is_being_removed_takes_nothing_else_with_it(tmp_path, monkeypatch):
    from stet import runner

    # root/a holds b (with c inside) and x. Somewhere else entirely there is another x,
    # with something in it that is not ours.
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    (root / "a" / "b" / "c").mkdir(parents=True)
    (root / "a" / "x").mkdir()
    (root / "a" / "x" / "ours.txt").write_text("")
    (elsewhere / "x").mkdir(parents=True)
    (elsewhere / "x" / "thesis.txt").write_text("three years")
    real, calls = runner._empty, []

    def empty(fd):
        calls.append(fd)
        if len(calls) == 4:  # root, a, b, and now c: while we are down here, b is moved away
            os.rename(root / "a" / "b", elsewhere / "b")
        return sorted(real(fd), reverse=True)  # b before x

    monkeypatch.setattr(runner, "_empty", empty)
    # ".." from b is no longer a. The names still to do there ("b" itself, then "x") are
    # a's; used in the directory we have landed in, "x" would be someone else's.
    assert runner.remove_tree(root) is False
    assert (elsewhere / "x" / "thesis.txt").read_text() == "three years" and (elsewhere / "b").is_dir()
    assert (root / "a" / "x" / "ours.txt").exists()  # it stopped; it did not guess
    monkeypatch.undo()
    assert runner.remove_tree(root) and not root.exists()  # and with nothing moving, it finishes


CHATTY = """
import os, sys
line = "progress " + "." * 110 + "\\n"
for i in range(20000):  # about 2.4 MB, more than the log keeps
    sys.stdout.write(line)
sys.stdout.flush()
raise RuntimeError("the real reason: the tokenizer file is missing")
"""


def test_what_an_editor_said_last_is_what_is_reported(corpus, tmp_path):
    from stet import runner

    run = run_editor(editor(tmp_path, CHATTY), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 60)
    assert run.status == "error" and run.exit_code == 1
    # The log file holds the first megabyte. The reason it stopped came after that.
    log = (tmp_path / "work" / "editor.log").read_bytes()
    assert len(log) == runner.LOG_LIMIT_BYTES and b"the real reason" not in log and log.startswith(b"progress")
    assert "RuntimeError: the real reason: the tokenizer file is missing" in run.log_tail and "Traceback" in run.log_tail
    assert len(run.log_tail) <= 4000 and "the real reason" in run.to_dict()["log_tail"]
    # A quiet editor's output is all there, as before.
    run = run_editor(editor(tmp_path, "print('one line')"), "python editor.py", corpus, RND, SCALE, tmp_path / "w2", 60)
    assert run.log_tail == "one line\n"


WAIT = """
import os, time
out = os.environ["STET_OUT"]
%s
open(os.path.join(out, "started"), "w").close()
time.sleep(120)
"""


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions that apply to this user")
def test_a_low_disk_with_output_that_cannot_be_measured_is_the_editors_error(corpus, tmp_path, monkeypatch):
    from stet import runner

    real = runner.shutil.disk_usage
    for n, (body, hidden) in enumerate((("", False), ('os.mkdir(os.path.join(out, "x"), 0o300)', True))):
        work = tmp_path / f"work{n}"
        started = work / "out" / "started"
        # Plenty of room until the editor is running; none from then on, and none after it is gone.
        monkeypatch.setattr(runner.shutil, "disk_usage", lambda p, started=started: type("U", (), {"free": 1 << 20})() if started.exists() else real(p))
        if not hidden:
            # Nothing of the editor's accounts for it: the machine's disk is full, and no editor is blamed.
            with pytest.raises(runner.HarnessError, match="still low without it"):
                run_editor(editor(tmp_path, WAIT % body), "python editor.py", corpus, RND, SCALE, work, 60)
        else:
            # What the editor wrote cannot be added up, because the editor made it so. That
            # is not the machine's failure, and an anchor does not stop for it.
            run = run_editor(editor(tmp_path, WAIT % body), "python editor.py", corpus, RND, SCALE, work, 60)
            assert run.status == "error" and run.reason == "the editor ran the disk low and made part of its output directory unreadable"
        monkeypatch.undo()
        assert runner.remove_tree(work)


# --- the same machine for every editor ---------------------------------------------------------
#
# Volunteers' machines differ. What an editor is given does not: the season's cores, memory
# and GPU memory, the files it named, and no network to fetch anything else.

LOOK = """
import json, os
from stet.editor import Task
task = Task.from_env()
seen = {"cpus": len(os.sched_getaffinity(0)), "cores": task.cores, "memory_gb": task.memory_gb, "gpu_memory_gb": task.gpu_memory_gb,
        "offline": os.environ.get("HF_HUB_OFFLINE"), "alloc": os.environ.get("PYTORCH_ALLOC_CONF", ""), "assets": task.info["assets"],
        "weights": task.asset("smol/model.bin").read_bytes().decode(), "dir": sorted(p.name for p in task.asset("smol").iterdir())}
try:
    open(task.asset("smol/model.bin"), "ab").write(b"x"); seen["writable"] = True
except OSError:
    seen["writable"] = False
try:
    task.asset("smol/missing"); seen["missing"] = "found"
except FileNotFoundError as e:
    seen["missing"] = str(e)
print("LOOK" + json.dumps(seen))
task.write([0])
"""


def an_asset(blob: bytes, path: str = "smol/model.bin"):
    import hashlib

    from stet import assets

    a = assets.Asset(path, "https://huggingface.co/o/n/resolve/" + "a" * 40 + "/f", hashlib.sha256(blob).hexdigest(), len(blob))
    assets.cache().mkdir(parents=True, exist_ok=True)
    if not assets.cached(a).exists():
        assets.cached(a).write_bytes(blob)
        assets.cached(a).chmod(0o444)
    return a


@pytest.mark.parametrize("sandbox", ["none", pytest.param("bwrap", marks=needs_sandbox)])
def test_an_editor_is_given_its_assets_and_the_seasons_machine(corpus, tmp_path, monkeypatch, sandbox):
    from stet import runner

    if sys.platform != "linux":
        pytest.skip("cores are assigned on Linux")
    monkeypatch.setattr(runner, "_gpu_total", [16 << 30])  # a 16 GB card, whatever this machine has
    a, b = an_asset(b"the weights"), an_asset(b"{}", "smol/config.json")
    limits = {"cores": 2, "memory_gb": 4, "gpu_memory_gb": 8}
    run = run_editor(editor(tmp_path, LOOK), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 60, sandbox=sandbox, limits=limits, assets=[a, b])
    assert run.status == "ok", run.log_tail
    seen = json.loads(run.log_tail.split("LOOK", 1)[1].splitlines()[0])
    want = min(2, len(os.sched_getaffinity(0)))
    assert seen["cpus"] == want and seen["cores"] == want and seen["memory_gb"] == 4 and seen["gpu_memory_gb"] == 8
    assert seen["offline"] == "1" and seen["alloc"] == "per_process_memory_fraction:0.500000"
    assert seen["weights"] == "the weights" and seen["dir"] == ["config.json", "model.bin"] and seen["assets"] == ["smol/model.bin", "smol/config.json"]
    assert seen["writable"] is False and "no asset at 'smol/missing'" in seen["missing"] and "smol/model.bin" in seen["missing"]
    if sandbox == "bwrap" and scope_available()[0]:
        assert run.limits["cores"] == want and run.limits["gpu_memory_gb"] == 8


def test_an_asset_that_was_never_fetched_is_the_machines_error(corpus, tmp_path):
    from stet import assets
    from stet.runner import HarnessError

    missing = assets.Asset("smol/model.bin", "https://huggingface.co/o/n/resolve/" + "a" * 40 + "/f", "b" * 64, 5)
    with pytest.raises(HarnessError, match="has not been fetched"):
        run_editor(editor(tmp_path, WRITE_ALL), "python editor.py", corpus, RND, SCALE, tmp_path / "work", 60, assets=[missing])


def test_the_cores_an_editor_gets_are_different_physical_cores():
    from stet.runner import pick_cpus

    if sys.platform != "linux":
        pytest.skip("Linux only")
    allowed = sorted(os.sched_getaffinity(0))
    assert pick_cpus(0) == allowed and pick_cpus(10**6) == allowed
    two = pick_cpus(2)
    assert len(two) == min(2, len(allowed)) and set(two) <= set(allowed)

    def core(cpu):
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
        try:
            return open(base + "physical_package_id").read(), open(base + "core_id").read()
        except OSError:
            return ("?", cpu)

    physical = {core(c) for c in allowed}
    if len(physical) >= 2:
        assert len({core(c) for c in two}) == 2


def test_the_season_says_what_machine_an_editor_is_given():
    from stet.runner import machine_limits

    m = machine_limits(SEASON)
    assert set(m) == {"cores", "memory_gb", "gpu_memory_gb", "scratch_gb"} and all(v > 0 for v in m.values())
