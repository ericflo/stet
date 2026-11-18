"""An agent at work: shell sessions and JSON tool calls in a small simulated project, with the
results worked out here. The student predicts what comes back, or reads a whole session."""

from __future__ import annotations

import json

from ..items import Item
from ..rng import Rng
from .lexicon import NAMES, PLACES, WORDS

NAME = "agent"
COMMANDS = ["ls", "ls-dir", "cat", "wc", "head", "grep", "cd-pwd", "echo-math", "mkdir-ls"]
TOOLS = ["ls", "ls-dir", "cat", "wc", "grep"]


def corners() -> list[str]:
    out = [f"{NAME}/shell/{c}" for c in COMMANDS] + [f"{NAME}/tools/{c}" for c in TOOLS]
    return out + [f"{NAME}/shell/session", f"{NAME}/tools/session"]


class Project:
    def __init__(self, rng: Rng):
        self.user = rng.choice(NAMES).lower()
        self.name = rng.choice(WORDS)
        self.cwd = f"/home/{self.user}/{self.name}"
        notes = [" ".join(rng.sample(WORDS, rng.between(2, 5))) for _ in range(rng.between(3, 6))]
        levels = ["INFO", "INFO", "WARN", "ERROR"]
        log = []
        for _ in range(rng.between(4, 8)):
            lvl = rng.choice(levels)
            log.append(f"2026-0{rng.between(1, 9)}-1{rng.below(10)} 0{rng.below(10)}:{rng.between(10, 59)}:{rng.between(10, 59)} {lvl} {rng.choice(WORDS)} {rng.choice(['started', 'stopped', 'failed', 'retrying', 'ok'])}")
        if not any(" ERROR " in line for line in log):
            log[rng.below(len(log))] = log[0].split(" INFO ")[0].split(" WARN ")[0] + f" ERROR {rng.choice(WORDS)} failed"
        rows = ["name,city,count"] + [f"{rng.choice(NAMES)},{rng.choice(PLACES)},{rng.between(1, 99)}" for _ in range(rng.between(2, 5))]
        fn = rng.choice(WORDS)
        self.files = {
            "README.md": f"# {self.name.title()}\n\nA small tool for {rng.choice(WORDS)} {rng.choice(WORDS)}s.",
            "notes.txt": "\n".join(notes),
            f"src/{fn}.py": f"def {fn}(x):\n    return x * {rng.between(2, 9)}",
            "src/main.py": f"from {fn} import {fn}\n\nprint({fn}({rng.between(1, 20)}))",
            "logs/app.log": "\n".join(log),
            f"data/{rng.choice(WORDS)}.csv": "\n".join(rows),
        }

    def ls(self, d: str = "") -> list[str]:
        pre = f"{d}/" if d else ""
        out = set()
        for p in self.files:
            if p.startswith(pre):
                out.add(p[len(pre):].split("/")[0])
        return sorted(out, key=str.lower)

    def dirs(self) -> list[str]:
        return sorted({p.split("/")[0] for p in self.files if "/" in p})

    def some_file(self, rng: Rng) -> str:
        return rng.choice(sorted(self.files))


def _shell(p: Project, rng: Rng, c: str) -> tuple[str, str]:
    if c == "ls":
        return "ls", "  ".join(p.ls())
    if c == "ls-dir":
        d = rng.choice(p.dirs())
        return f"ls {d}", "  ".join(p.ls(d))
    if c == "cat":
        f = rng.choice(["notes.txt", "README.md", "logs/app.log"] + [x for x in p.files if x.endswith(".csv")])
        return f"cat {f}", p.files[f]
    if c == "wc":
        f = rng.choice(["notes.txt", "logs/app.log"])
        return f"wc -l {f}", f"{len(p.files[f].splitlines())} {f}"
    if c == "head":
        f, n = rng.choice(["notes.txt", "logs/app.log"]), rng.between(1, 3)
        return f"head -n {n} {f}", "\n".join(p.files[f].splitlines()[:n])
    if c == "grep":
        return "grep ERROR logs/app.log", "\n".join(line for line in p.files["logs/app.log"].splitlines() if "ERROR" in line)
    if c == "cd-pwd":
        d = rng.choice(p.dirs())
        return f"cd {d} && pwd", f"{p.cwd}/{d}"
    if c == "echo-math":
        a, b = rng.between(2, 99), rng.between(2, 99)
        return f"echo $(({a} * {b}))", str(a * b)
    if c == "mkdir-ls":
        d = rng.choice(["build", "dist", "tmp", "out"])
        return f"mkdir {d} && ls", "  ".join(sorted(p.ls() + [d], key=str.lower))
    raise ValueError(c)


def _tool(p: Project, rng: Rng, c: str) -> tuple[dict, object]:
    if c == "ls":
        return {"tool": "list_dir", "path": "."}, p.ls()
    if c == "ls-dir":
        d = rng.choice(p.dirs())
        return {"tool": "list_dir", "path": d}, p.ls(d)
    if c == "cat":
        f = rng.choice(["notes.txt", "README.md"])
        return {"tool": "read_file", "path": f}, p.files[f]
    if c == "wc":
        f = rng.choice(["notes.txt", "logs/app.log"])
        return {"tool": "count_lines", "path": f}, len(p.files[f].splitlines())
    if c == "grep":
        return {"tool": "search", "pattern": "ERROR", "path": "logs/app.log"}, [line for line in p.files["logs/app.log"].splitlines() if "ERROR" in line]
    raise ValueError(c)


def _history(p: Project, rng: Rng, style: str, n: int, avoid: str) -> str:
    pool = [c for c in (COMMANDS if style == "shell" else TOOLS) if c != avoid and c not in ("cd-pwd", "mkdir-ls")]
    out = []
    for c in rng.sample(pool, n):
        if style == "shell":
            cmd, res = _shell(p, rng, c)
            out.append(f"$ {cmd}\n{res}\n")
        else:
            call, res = _tool(p, rng, c)
            out.append(json.dumps(call) + "\n" + json.dumps({"result": res}) + "\n")
    return "".join(out)


def make(rng: Rng, corner: str) -> Item:
    _, style, c = corner.split("/")
    p = Project(rng)
    head = f"{p.user}@{p.name}" if style == "shell" else f"# agent log: {p.name}"
    if c == "session":
        text = (f"{head}\n" if style == "tools" else "") + _history(p, rng, style, 4, "")
        return Item(kind="passage", corner=corner, fmt="logs", lang="en", source=f"gen:{NAME}", text=text.rstrip("\n"))
    before = _history(p, rng, style, rng.between(1, 2), c)
    if style == "shell":
        cmd, res = _shell(p, rng, c)
        return Item(kind="answer", corner=corner, fmt="logs", lang="en", source=f"gen:{NAME}", ctx=f"{before}$ {cmd}\n", cont=res)
    call, res = _tool(p, rng, c)
    full = json.dumps({"result": res})
    cut = len('{"result": ')
    return Item(kind="answer", corner=corner, fmt="structured", lang="en", source=f"gen:{NAME}",
                ctx=f"{head}\n{before}{json.dumps(call)}\n{full[:cut].rstrip()}", cont=" " + full[cut:])
