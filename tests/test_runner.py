import dataclasses
import json
import os
import sys

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
