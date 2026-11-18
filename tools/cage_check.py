#!/usr/bin/env python3
"""Try the container sandbox from inside, the way a curator that wanted out would.

Run it as root in a container that runs stet and nothing else (a rented machine), after
`stet doctor` says the container sandbox is there:

    python tools/cage_check.py            # or: python -m ... with stet installed

Each case is a small curator run with the sandbox's limits (made small, so that the
memory and process cases are quick), and what it was able to do is checked from outside.
It exits 0 when every case held, and prints one line a case either way.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from stet import cage
from stet.data import Corpus, cache_dir
from stet.runner import isolation_probe, run_editor, sandbox_selftest
from stet.spec import load_season, practice_round

WRITE = "from stet.curator import Task\ntask = Task.from_env()\n"
DONE = "task.write([i for i, _ in task.corpus])\n"

CASES = {
    # name: (curator body, limits, time limit, what must be true afterwards)
    "memory": (
        "import numpy as np\nblocks = []\nwhile True:\n    blocks.append(np.ones(1 << 27, dtype=np.uint8))\n",
        {"memory_gb": 1}, 60, "stopped for memory",
    ),
    "fork bomb": (
        "import os, time\nfor _ in range(10000):\n    try:\n        os.fork()\n    except OSError:\n        pass\ntime.sleep(2)\n" + DONE,
        {"tasks": 64}, 30, "no processes left",
    ),
    "left running": (
        "import os, subprocess\nsubprocess.Popen(['sleep', '1000'], start_new_session=True)\n"
        "if os.fork() == 0:\n    os.setsid()\n    if os.fork() == 0:\n        import time\n        time.sleep(1000)\n    os._exit(0)\n" + DONE,
        {}, 30, "no processes left",
    ),
    "leftovers": (
        "import os, ctypes\nopen('/tmp/left', 'w').write('x')\nopen('/dev/shm/left', 'w').write('x')\nos.makedirs('/var/tmp/left/deep', exist_ok=True)\n"
        "libc = ctypes.CDLL(None)\nassert libc.shmget(0, 4096, 0o1600) >= 0\n" + DONE,
        {}, 30, "nothing left behind",
    ),
    "secrets": (
        "import os, sys\nbad = []\nfor p in ('/root', '/proc/1/environ', '/proc/1/mem', '/proc/1/root/root'):\n    try:\n"
        "        os.listdir(p) if os.path.isdir(p) else open(p, 'rb').read(1)\n        bad.append(p)\n    except OSError:\n        pass\n"
        "if bad:\n    sys.exit('read ' + ' '.join(bad))\n" + DONE,
        {}, 30, "ok",
    ),
    "privileges": (
        "import ctypes, os, subprocess, sys\nlibc = ctypes.CDLL(None, use_errno=True)\nbad = []\n"
        "if libc.ptrace(16, 1, 0, 0) == 0:\n    bad.append('ptrace')\n"
        "r = subprocess.run(['su', '-c', 'id -u'], capture_output=True, text=True, input='')\nif r.stdout.strip() == '0':\n    bad.append('su')\n"
        "if os.path.exists('/usr/bin/sudo'):\n    r = subprocess.run(['sudo', '-n', 'id', '-u'], capture_output=True, text=True)\n    if r.stdout.strip() == '0':\n        bad.append('sudo')\n"
        "if bad:\n    sys.exit('got ' + ' '.join(bad))\n" + DONE,
        {}, 30, "ok",
    ),
    "time limit": ("import time\nwhile True:\n    time.sleep(1)\n", {}, 3, "timed out, no processes left"),
}


def setup(root: Path) -> Corpus:
    import numpy as np

    (root / "corpus").mkdir()
    (root / "corpus" / "docs.bin").write_bytes(b"one document")
    np.array([0, 12], dtype="<u8").tofile(root / "corpus" / "docs.idx")
    (root / "corpus" / "urls.txt").write_text("http://example.test/0\n")
    (root / "corpus" / "meta.json").write_text(json.dumps({"n_docs": 1, "bytes": 12, "sha256": "check"}))
    (root / "bundle").mkdir()
    return Corpus(root / "corpus")


def leftovers() -> list[str]:
    found = [str(p) for d in cage.SHARED if d.is_dir() for p in d.iterdir() if cage._owned(p)]
    try:
        shm = Path("/proc/sysvipc/shm").read_text().splitlines()
        head = shm[0].split()
        found += [f"shm {f['shmid']}" for f in (dict(zip(head, line.split(), strict=False)) for line in shm[1:]) if int(f["uid"]) == cage.UID]
    except (OSError, IndexError, KeyError):
        pass
    return found


def main() -> int:
    ok, why = cage.available()
    if not ok:
        print(f"no container sandbox here: {why}")
        return 2
    season = load_season()
    rnd, scale = practice_round(season, 0), season.scale("nano")
    failed = 0
    with tempfile.TemporaryDirectory(prefix="stet-cage-check-", dir=cache_dir() if cache_dir().is_dir() else None) as tmp:
        root = Path(tmp)
        corpus = setup(root)
        for name, (body, limits, seconds, want) in CASES.items():
            (root / "bundle" / "curator.py").write_text(WRITE + body)
            t0 = time.time()
            run = run_editor(root / "bundle", "python curator.py", corpus, rnd, scale, root / "work", seconds, sandbox="container", limits=limits)
            left_procs, left_files = cage.pids(), leftovers()
            said = f"{run.status}: {run.reason or run.log_tail.strip()[-200:]}"
            held = {
                "stopped for memory": run.status == "error" and "memory" in run.reason and not left_procs,
                "no processes left": not left_procs,
                "nothing left behind": run.status == "ok" and not left_files and not left_procs,
                "ok": run.status == "ok",
                "timed out, no processes left": run.status == "timeout" and not left_procs,
            }[want]
            failed += not held
            print(f"{'held ' if held else 'FAILED'}  {name:13} {time.time() - t0:5.1f}s  {said[:160]}" + (f"  left: {left_procs[:5]} {left_files[:5]}" if not held else ""))
        (root / "bundle" / "curator.py").write_text(isolation_probe())
        run = run_editor(root / "bundle", "python curator.py", corpus, rnd, scale, root / "work", 90, sandbox="container")
        failed += run.status != "ok"
        print(f"{'held ' if run.status == 'ok' else 'FAILED'}  {'network, files':13}        {run.status}: {run.reason or run.log_tail.strip()[-300:]}")
    ok, why = sandbox_selftest("container")
    failed += not ok
    print(f"{'held ' if ok else 'FAILED'}  {'self-test':13}        {why or 'a curator runs, with Node, loopback, the GPU and its limits'}")
    print("every case held" if not failed else f"{failed} cases did not hold")
    return 1 if failed else 0


if __name__ == "__main__":
    os.umask(0o022)
    sys.exit(main())
