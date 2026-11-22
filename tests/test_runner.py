import dataclasses
import json
import os
import sys
import time

import pytest

from stet.runner import bwrap_available, collect_script, run_editor, scope_available
from stet.script import ScriptError
from stet.spec import load_season, practice_round

SEASON = load_season("s2")
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=512, ctx=64, batch_size=4)
RND = practice_round(SEASON, 0)


def editor(tmp_path, body: str):
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "curator.py").write_text(body)
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
    run = run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 30)
    assert run.status == "ok" and run.script == (tmp_path / "work" / "script.jsonl").resolve()
    assert run.script.read_text().split() == ["0", "1", "2", "3"]
    assert "kept 4" in run.log_tail and "seed" not in json.dumps(json.load(open(tmp_path / "work" / "out" / "task.json")))


def test_no_script_a_crash_and_a_timeout(corpus, tmp_path):
    run = run_editor(editor(tmp_path, "print('nothing')"), "python curator.py", corpus, RND, SCALE, tmp_path / "w1", 30)
    assert run.status == "no_script" and run.script is None and "script.jsonl" in run.reason
    run = run_editor(editor(tmp_path, "raise SystemExit(3)"), "python curator.py", corpus, RND, SCALE, tmp_path / "w2", 30)
    assert run.status == "error" and run.exit_code == 3
    # A script written before the deadline still counts when the curator is killed at it.
    slow = WRITE_ALL + "\nimport time\ntime.sleep(60)\n"
    run = run_editor(editor(tmp_path, slow), "python curator.py", corpus, RND, SCALE, tmp_path / "w3", 2)
    assert run.status == "timeout" and run.script is not None and run.seconds < 10


@pytest.mark.skipif(os.name == "nt", reason="POSIX only")
def test_a_script_that_points_somewhere_else_is_not_read(corpus, tmp_path):
    secret = tmp_path / "secret.jsonl"
    secret.write_text("3\n")
    body = f"import os\nos.symlink({str(secret)!r}, os.path.join(os.environ['STET_OUT'], 'script.jsonl'))\n"
    run = run_editor(editor(tmp_path, body), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 30)
    assert run.script is None and run.status == "no_script"
    body = "import os\nos.mkfifo(os.path.join(os.environ['STET_OUT'], 'script.jsonl'))\n"
    run = run_editor(editor(tmp_path, body), "python curator.py", corpus, RND, SCALE, tmp_path / "work2", 30)
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
    run = run_editor(editor(tmp_path, SNOOP), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 60, sandbox="bwrap")
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
    run = run_editor(editor(tmp_path, HOG), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 120, sandbox="bwrap", limits={"memory_gb": 1, "required": True})
    assert run.status == "error" and run.seconds < 60  # killed by the cgroup, not by the deadline
    assert run.script is not None  # what it wrote before dying still counts


# --- whose fault a failure is ---------------------------------------------------------------
#
# An anchor hands a job back when the machine failed and reports it when the curator did. If
# a curator could make its own failure look like the machine's, its job would be retried
# for ever and nobody else's would run.

needs_sandbox = pytest.mark.skipif(sys.platform != "linux" or not bwrap_available()[0], reason="needs bubblewrap")

FILL_SCRATCH = """
import time
with open("/scratch/big", "wb") as f:
    f.write(bytes(4 << 20))
time.sleep(60)
"""


@needs_sandbox
def test_a_scratch_directory_on_disk_is_watched_whatever_else_was_applied(corpus, tmp_path, monkeypatch):
    # Where no cgroup can be had the sandbox's scratch is a directory on disk. Cores are pinned
    # all the same, and that some limit was applied does not mean scratch is in bounded memory.
    from stet import runner

    monkeypatch.setattr(runner, "scope_limits", lambda: (False, "no systemd here", False))
    monkeypatch.setattr(runner, "SCRATCH_LIMIT_BYTES", 1 << 20)
    run = run_editor(editor(tmp_path, FILL_SCRATCH), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 40, sandbox="bwrap", limits={"cores": 1})
    assert run.limits.get("cores") == 1 and "memory_gb" not in run.limits
    assert run.status == "error" and "scratch" in run.reason and run.seconds < 30


def test_a_bad_entry_command_is_the_editors_error(corpus, tmp_path):
    bundle = editor(tmp_path, WRITE_ALL)
    for n, entry in enumerate(['python "curator.py', "no-such-program-xyz curator.py", "python curator.py " + "x" * 5000, "./editor.py"]):
        run = run_editor(bundle, entry, corpus, RND, SCALE, tmp_path / f"w{n}", 30)
        assert run.status == "error" and run.script is None, entry
        assert "entry command" in run.reason, run.reason


@needs_sandbox
def test_in_the_sandbox_a_bad_entry_or_a_forged_message_is_still_the_editors_error(corpus, tmp_path):
    bundle = editor(tmp_path, "import sys\nprint('bwrap: execvp python: No such file or directory')\nsys.exit(1)\n")
    run = run_editor(bundle, "python curator.py", corpus, RND, SCALE, tmp_path / "w1", 30, sandbox="bwrap")
    assert run.status == "error" and run.exit_code == 1 and "bwrap:" in run.log_tail
    for n, entry in enumerate(["no-such-program index.js", "./editor.py"]):  # not installed; not executable
        run = run_editor(bundle, entry, corpus, RND, SCALE, tmp_path / f"v{n}", 30, sandbox="bwrap")
        assert run.status == "error" and run.exit_code == 127 and "cannot start" in run.log_tail, run


@needs_sandbox
def test_a_sandbox_that_does_not_start_is_the_machines_error(corpus, tmp_path, monkeypatch):
    from stet import runner

    real = runner.bwrap_command
    monkeypatch.setattr(runner, "bwrap_command", lambda *a: [*real(*a), "--ro-bind", "/no/such/directory", "/nowhere"])
    with pytest.raises(runner.HarnessError, match="failed to start"):
        run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 30, sandbox="bwrap")


@pytest.mark.skipif(os.name == "nt", reason="POSIX only")
def test_nothing_private_is_mounted_where_an_editor_could_read_it(corpus, tmp_path, monkeypatch):
    from stet import runner

    shown = [str(tmp_path / "site-packages"), str(tmp_path / "corpus")]  # directories the sandbox would mount
    assert runner._exposed(SEASON, shown) == ""  # the cache and the key are elsewhere
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
            run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 30, sandbox="bwrap")


def test_a_full_disk_is_the_machines_error(corpus, tmp_path, monkeypatch):
    from stet import runner

    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 30})())
    with pytest.raises(runner.HarnessError, match="free"):
        run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 30)


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
    run = run_editor(editor(tmp_path, HIDE), "python curator.py", corpus, RND, SCALE, work, 30)
    assert run.status == "error" and run.seconds < 15
    assert run.script is not None  # what it wrote before being stopped still counts
    assert (work / "out" / "x").exists() and remove_tree(work) and not work.exists()


def _dig(root, names, bottom_mode=None):
    """Make a chain of directories under root by name alone, as a curator would with mkdir and
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
    run = run_editor(editor(tmp_path, LITTER), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 30)
    assert run.status == "error" and "files" in run.reason and run.seconds < 20


def test_the_clock_does_not_wait_for_the_accounting(corpus, tmp_path, monkeypatch):
    from stet import runner

    def slow(path, max_entries):  # a walk over a directory the curator has stuffed
        time.sleep(30)
        return 0, 0, True

    monkeypatch.setattr(runner, "_usage", slow)
    run = run_editor(editor(tmp_path, WRITE_ALL + "\nimport time\ntime.sleep(60)\n"), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 3)
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
    from stet.runner import node_dir, sandbox_selftest

    monkeypatch.setenv("STET_CACHE", str(tmp_path / "cache"))
    if node_dir() is None:  # a checkout without the season's Node: not a machine that may run curators
        ok, why = sandbox_selftest()
        assert not ok and "Node.js" in why
    else:
        assert sandbox_selftest() == (True, "")


# --- a curator does not outlive its run, and what it leaves is removed with care -----------------

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
            raise trouble  # in the middle of the run, with the curator and its child alive
        return real(path)

    monkeypatch.setattr(runner.shutil, "disk_usage", disk_usage)
    with pytest.raises(type(trouble)):
        run_editor(editor(tmp_path, LINGER), "python curator.py", corpus, RND, SCALE, work, 60)
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

    run = run_editor(editor(tmp_path, CHATTY), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 60)
    assert run.status == "error" and run.exit_code == 1
    # The log file holds the first megabyte. The reason it stopped came after that.
    log = (tmp_path / "work" / "editor.log").read_bytes()
    assert len(log) == runner.LOG_LIMIT_BYTES and b"the real reason" not in log and log.startswith(b"progress")
    assert "RuntimeError: the real reason: the tokenizer file is missing" in run.log_tail and "Traceback" in run.log_tail
    assert len(run.log_tail) <= 4000 and "the real reason" in run.to_dict()["log_tail"]
    # A quiet curator's output is all there, as before.
    run = run_editor(editor(tmp_path, "print('one line')"), "python curator.py", corpus, RND, SCALE, tmp_path / "w2", 60)
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
        # Plenty of room until the curator is running; none from then on, and none after it is gone.
        monkeypatch.setattr(runner.shutil, "disk_usage", lambda p, started=started: type("U", (), {"free": 1 << 20})() if started.exists() else real(p))
        if not hidden:
            # Nothing of the curator's accounts for it: the machine's disk is full, and no curator is blamed.
            with pytest.raises(runner.HarnessError, match="still low without it"):
                run_editor(editor(tmp_path, WAIT % body), "python curator.py", corpus, RND, SCALE, work, 60)
        else:
            # What the curator wrote cannot be added up, because the curator made it so. That
            # is not the machine's failure, and an anchor does not stop for it.
            run = run_editor(editor(tmp_path, WAIT % body), "python curator.py", corpus, RND, SCALE, work, 60)
            assert run.status == "error" and run.reason == "the curator ran the disk low and made part of its output directory unreadable"
        monkeypatch.undo()
        assert runner.remove_tree(work)


# --- the same machine for every curator ---------------------------------------------------------
#
# Volunteers' machines differ. What a curator is given does not: the season's cores, memory
# and GPU memory, the files it named, and no network to fetch anything else.

LOOK = """
import json, os
from stet.curator import Task
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
    run = run_editor(editor(tmp_path, LOOK), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 60, sandbox=sandbox, limits=limits, assets=[a, b])
    assert run.status == "ok", run.log_tail
    seen = json.loads(run.log_tail.split("LOOK", 1)[1].splitlines()[0])
    want = min(2, len(os.sched_getaffinity(0)))
    assert seen["cpus"] == want and seen["cores"] == want and seen["memory_gb"] == 4 and seen["gpu_memory_gb"] == 8
    assert seen["offline"] == "1" and seen["alloc"] == "per_process_memory_fraction:0.500000"
    assert seen["weights"] == "the weights" and seen["dir"] == ["config.json", "model.bin"] and seen["assets"] == ["smol/model.bin", "smol/config.json"]
    assert seen["writable"] is False and "no asset at 'smol/missing'" in seen["missing"] and "smol/model.bin" in seen["missing"]
    if sandbox == "bwrap" and scope_available()[0]:
        assert run.limits["cores"] == want and run.limits["gpu_memory_gb"] == 8


HARNESS = """
import json, os, subprocess, sys, threading, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from stet.curator import Task
task = Task.from_env()

class Model(BaseHTTPRequestHandler):  # stands in for a model served to a harness
    def do_POST(self):
        said = self.rfile.read(int(self.headers["content-length"]))
        body = json.dumps({"heard": said.decode()}).encode()
        self.send_response(200); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a):
        pass

server = HTTPServer(("127.0.0.1", 0), Model)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/v1/x"
seen = {}
# A harness in another process, in another language, talking to the model over loopback.
js = "fetch(process.argv[1], {method: 'POST', body: 'keep or cut?'}).then(r => r.json()).then(j => console.log(process.version + ' ' + j.heard))"
node = subprocess.run(["node", "-e", js, url], capture_output=True, text=True, timeout=60)
seen["node"] = (node.stdout + node.stderr).strip()
py = subprocess.run([sys.executable, "-c", "import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], data=b'hello', timeout=10).read().decode())", url], capture_output=True, text=True, timeout=60)
seen["python"] = (py.stdout + py.stderr).strip()
try:
    urllib.request.urlopen("http://1.1.1.1/", timeout=3); seen["outside"] = "reached"
except OSError as e:
    seen["outside"] = "no"
print("HARNESS" + json.dumps(seen))
task.write([0])
"""


@pytest.mark.parametrize("sandbox", ["none", pytest.param("bwrap", marks=needs_sandbox)])
def test_an_editor_may_run_a_harness_of_its_own_against_a_model_it_serves_itself(corpus, tmp_path, sandbox):
    # The shape of an agent: a model served on loopback, and a harness in another process
    # (here Node, the season's own) that talks to it. No network is needed for either.
    from stet.runner import node_dir

    if node_dir() is None:
        pytest.skip("the season's Node.js is not installed here")
    run = run_editor(editor(tmp_path, HARNESS), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 120, sandbox=sandbox)
    assert run.status == "ok", run.log_tail
    seen = json.loads(run.log_tail.split("HARNESS", 1)[1].splitlines()[0])
    assert seen["node"] == "v24.19.0 keep or cut?", seen  # the season's Node, whatever /usr/bin holds
    assert json.loads(seen["python"]) == {"heard": "hello"}
    if sandbox == "bwrap":
        assert seen["outside"] == "no"


def test_the_seasons_node_is_found_where_the_sandbox_shows_it(monkeypatch):
    from stet import runner

    real = runner.node_dir()
    if real is None:
        pytest.skip("the season's Node.js is not installed here")
    site = os.path.dirname(os.path.dirname(real))
    assert runner.node_dir([(site, "/stet-site0")]) == "/stet-site0/nodejs_wheel/bin"
    assert runner.node_dir([("/somewhere/else", "/stet-site0")]) is None  # not mounted: not promised


def test_an_asset_that_was_never_fetched_is_the_machines_error(corpus, tmp_path):
    from stet import assets
    from stet.runner import HarnessError

    missing = assets.Asset("smol/model.bin", "https://huggingface.co/o/n/resolve/" + "a" * 40 + "/f", "b" * 64, 5)
    with pytest.raises(HarnessError, match="has not been fetched"):
        run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 60, assets=[missing])


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


# --- one GPU, with its memory there to be had -----------------------------------------------------


def test_an_editor_is_shown_one_gpu_whatever_the_machine_has():
    from stet.runner import one_gpu

    assert one_gpu({}) == {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}
    assert one_gpu({}, index=1) == {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"}  # the card this machine grades on
    assert one_gpu({"CUDA_VISIBLE_DEVICES": "2,3"}) == {"CUDA_VISIBLE_DEVICES": "2"}  # the operator's choice, cut to one
    assert one_gpu({"HIP_VISIBLE_DEVICES": " 1 "}) == {"HIP_VISIBLE_DEVICES": "1"}  # and never both, or the second would index into the first


SHOW_GPUS = """
import os
from stet.curator import Task
print("GPUS", os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("HIP_VISIBLE_DEVICES"))
Task.from_env().write([0])
"""


@needs_sandbox
def test_in_the_sandbox_the_editor_is_told_to_use_one_gpu(corpus, tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    run = run_editor(editor(tmp_path, SHOW_GPUS), "python curator.py", corpus, RND, SCALE, tmp_path / "work", 60, sandbox="bwrap")
    assert run.status == "ok" and "GPUS 0 0" in run.log_tail, run.log_tail


def test_memory_that_is_taken_is_this_machines_trouble_not_the_curators(corpus, tmp_path, monkeypatch):
    from stet import runner
    from stet.runner import HarnessError

    # A machine at the floor with a browser open: the limit would let a curator have what
    # the machine cannot give, and the kernel would kill it for the machine's shortage.
    monkeypatch.setattr(runner, "memory_free_bytes", lambda: 9 << 30)
    limits = {"memory_gb": 12, "gpu_memory_gb": 0, "required": True}
    with pytest.raises(HarnessError, match="9.0 GB of memory free and a curator is promised 12"):
        run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "w1", 60, limits=limits)
    # A run of your own (`stet run`) is not refused: it is your machine and your curator.
    run = run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "w2", 60, limits={"memory_gb": 12})
    assert run.status == "ok"


def test_the_gpu_as_windows_wsl_shows_it_is_passed_into_the_sandbox(tmp_path, monkeypatch):
    from stet import runner

    real = os.path.exists
    monkeypatch.setattr(runner.os.path, "exists", lambda p: p == "/dev/dxg" or (not str(p).startswith("/dev/") and real(p)))
    (tmp_path / "etc").mkdir()
    cmd = runner.bwrap_command(tmp_path, tmp_path, tmp_path, None, tmp_path, tmp_path / "etc", 1 << 30)
    at = cmd.index("/dev/dxg")
    assert cmd[at - 1 : at + 2] == ["--dev-bind", "/dev/dxg", "/dev/dxg"]


def test_a_gpu_whose_memory_is_taken_is_this_machines_trouble_not_the_editors(corpus, tmp_path, monkeypatch):
    from stet import runner
    from stet.runner import HarnessError

    monkeypatch.setattr(runner, "_gpu_total", [10 << 30])
    monkeypatch.setattr(runner, "gpu_free_bytes", lambda index=0: 6 << 30)  # the verifier, or a game, holds the rest
    limits = {"gpu_memory_gb": 8, "required": True}
    with pytest.raises(HarnessError, match="6.0 GB of memory free and a curator is promised 8.0"):
        run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "w1", 60, limits=limits)
    monkeypatch.setattr(runner, "gpu_free_bytes", lambda index=0: int(7.8 * 2**30))  # all but what the drivers keep: enough
    monkeypatch.setattr(runner, "scope_available", lambda: (True, ""))
    run = run_editor(editor(tmp_path, WRITE_ALL), "python curator.py", corpus, RND, SCALE, tmp_path / "w2", 60, limits={"gpu_memory_gb": 8})
    assert run.status == "ok"
