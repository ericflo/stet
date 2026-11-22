"""An agent at work: shell sessions and JSON tool calls in a small simulated project, with the
results worked out here. The student predicts what comes back, or reads a whole session.

Every answer follows from the session so far: a file is listed, shown or written before anything
is asked about it, so what comes back can be worked out and cannot be guessed or learned by heart.
The project is the round's own: its directories, its files' names and what is in them are drawn
from the seed, half of the words coined for the purpose (lexicon.coin), so no listing or file is
the same from one round to the next.

Commands and tool calls are the same in every country, so they are not translated. What differs is
what the files hold. Beside the English projects there is one for each of the exam's other
languages, whose notes are written in that language and its script, and the questions asked of it
are the two whose answer is those notes: a file shown whole after its two halves, and a file read
back after a line is added. Such an item counts as its notes' language; every other as English,
for what comes back is file names, paths, counts and log lines.
"""

from __future__ import annotations

import json

from ..items import Item
from ..rng import Rng
from .lexicon import LANGS, NAMES, NOUNS, PLACES, WORDS, coin

NAME = "agent"
COMMANDS = ["ls", "ls-dir", "cat", "wc", "head", "grep", "cd-pwd", "mkdir-ls", "rm-ls", "append-cat"]
TOOLS = ["ls", "ls-dir", "cat", "wc", "grep", "write-read"]
DIRS = ["src", "data", "logs", "docs", "tests", "scripts", "config", "assets", "lib", "notes", "tools", "build"]
LEVELS = ["INFO", "INFO", "WARN", "ERROR", "DEBUG"]
EVENTS = ["started", "stopped", "failed", "retrying", "ok", "timed out", "connected", "closed"]


def corners() -> list[str]:
    out = [f"{NAME}/en/shell/{c}" for c in COMMANDS] + [f"{NAME}/en/tools/{c}" for c in TOOLS]
    out += [f"{NAME}/en/shell/session", f"{NAME}/en/tools/session"]
    for lang in LANGS[1:]:
        out += [f"{NAME}/{lang}/shell/cat", f"{NAME}/{lang}/tools/write-read"]
    return out


def _word(rng: Rng, lang: str = "en") -> str:
    """A word, coined half the time. Names of files and projects are always in Latin letters."""
    return coin(rng, lang) if rng.chance(1, 2) else rng.choice(NOUNS[lang] if lang != "en" else WORDS)


class Project:
    """A few directories of small files, and the commands an agent runs in it."""

    def __init__(self, rng: Rng, lang: str = "en"):
        self.lang = lang  # what its notes are written in
        self.user = rng.choice(NAMES).lower()
        self.name = _word(rng)
        self.home = self.cwd = f"/home/{self.user}/{self.name}"
        dirs = rng.sample(DIRS, rng.between(2, 3))
        self.files: dict[str, str] = {}
        self.empty: set[str] = set()  # directories with nothing in them
        self.files[rng.choice(["README.md", "NOTES.md", "TODO.md"])] = f"# {self.name.title()}\n\nA small tool for {_word(rng)} {_word(rng)}s."
        self.text = self._add(rng, "", ".txt", "\n".join(self.line(rng) for _ in range(rng.between(3, 5))))
        log = []
        for _ in range(rng.between(4, 7)):
            log.append(f"2026-0{rng.between(1, 9)}-1{rng.below(10)} 0{rng.below(10)}:{rng.between(10, 59)}:{rng.between(10, 59)} {rng.choice(LEVELS)} {_word(rng)} {rng.choice(EVENTS)}")
        self.level = rng.choice(sorted({line.split()[2] for line in log}))  # a level the log has
        self.log = self._add(rng, dirs[0], ".log", "\n".join(log))
        rows = ["name,city,count"] + [f"{rng.choice(NAMES)},{rng.choice(PLACES)},{rng.between(1, 99)}" for _ in range(rng.between(2, 4))]
        self._add(rng, dirs[1], ".csv", "\n".join(rows))
        fn = _word(rng)
        self.files[f"{dirs[-1]}/{fn}.py"] = f"def {fn}(x):\n    return x * {rng.between(2, 9)}"
        for _ in range(rng.between(0, 2)):
            self._add(rng, rng.choice(dirs), rng.choice([".md", ".txt", ".json", ".sh"]), _word(rng))

    def line(self, rng: Rng) -> str:
        """A line of the project's notes: a few words, in its language."""
        return " ".join(_word(rng, self.lang) for _ in range(rng.between(2, 4)))

    def _add(self, rng: Rng, d: str, ext: str, content: str) -> str:
        while (path := (f"{d}/" if d else "") + _word(rng) + ext) in self.files:
            pass
        self.files[path] = content
        return path

    def ls(self, d: str = "") -> list[str]:
        pre = f"{d}/" if d else ""
        out = {p[len(pre):].split("/")[0] for p in self.files if p.startswith(pre)}
        if not d:
            out |= self.empty
        return sorted(out, key=str.lower)

    def dirs(self) -> list[str]:
        return sorted({p.split("/")[0] for p in self.files if "/" in p})

    def shown(self) -> list[str]:
        """Files whose every line a session can show: more than one line, so there is something to ask."""
        return [p for p in sorted(self.files) if len(self.files[p].splitlines()) > 2]


def _steps(p: Project, rng: Rng, c: str) -> tuple[list[tuple[str, dict, object]], tuple[str, dict, object]]:
    """What an agent does before the question, so that its answer follows, and the question: each a
    shell command, the same as a tool call, and what comes back."""

    def ls(d=""):
        return (f"ls {d}".strip(), {"tool": "list_dir", "path": d or "."}, p.ls(d))

    def cat(f):
        return (f"cat {f}", {"tool": "read_file", "path": f}, p.files[f])

    if c in ("ls", "mkdir-ls", "rm-ls"):
        before = [ls()]
        loose = [x for x in p.ls() if x in p.files and x != p.text]  # a file nothing later asks about
        if loose and (c == "rm-ls" or (c == "ls" and rng.chance(1, 2))):
            f = rng.choice(loose)
            del p.files[f]
            act = (f"rm {f}", {"tool": "delete_file", "path": f}, "")
        else:
            d = rng.choice([x for x in DIRS + ["dist", "tmp", "out"] if x not in p.ls()])
            p.empty.add(d)
            act = (f"mkdir {d}", {"tool": "make_dir", "path": d}, "")
        if c == "mkdir-ls":
            return before, (f"{act[0]} && ls", act[1], p.ls())
        return before + [act], ls()
    if c == "ls-dir":
        d = rng.choice(p.dirs())
        before = [ls(d)]
        f = f"{d}/{_word(rng)}{rng.choice(['.txt', '.md', '.json'])}"
        p.files.setdefault(f, "")  # touch leaves a file that is there as it was
        return before + [(f"touch {f}", {"tool": "write_file", "path": f, "content": ""}, "")], ls(d)
    if c == "cat":
        f = rng.choice(p.shown()) if p.lang == "en" else p.text
        k = rng.between(1, len(p.files[f].splitlines()) - 1)
        head = (f"head -n {k} {f}", {"tool": "read_file", "path": f, "lines": k}, "\n".join(p.files[f].splitlines()[:k]))
        tail_n = len(p.files[f].splitlines()) - k
        tail = (f"tail -n {tail_n} {f}", {"tool": "read_file", "path": f, "from_line": k + 1}, "\n".join(p.files[f].splitlines()[k:]))
        return [head, tail], cat(f)
    if c in ("append-cat", "write-read"):
        f = rng.choice([p.text, p.log]) if p.lang == "en" else p.text
        line = p.line(rng)
        before = [cat(f)]
        p.files[f] += "\n" + line
        return before + [(f'echo "{line}" >> {f}', {"tool": "append_file", "path": f, "content": line}, "")], cat(f)
    if c == "wc":
        f = rng.choice(p.shown())
        n = len(p.files[f].splitlines())
        return [cat(f)], (f"wc -l {f}", {"tool": "count_lines", "path": f}, n)
    if c == "head":
        f = rng.choice(p.shown())
        k = rng.between(1, len(p.files[f].splitlines()) - 1)
        return [cat(f)], (f"head -n {k} {f}", {"tool": "read_file", "path": f, "lines": k}, "\n".join(p.files[f].splitlines()[:k]))
    if c == "grep":
        hits = [line for line in p.files[p.log].splitlines() if f" {p.level} " in line]
        return [cat(p.log)], (f"grep {p.level} {p.log}", {"tool": "search", "pattern": p.level, "path": p.log}, hits)
    if c == "cd-pwd":
        d = rng.choice(p.dirs())
        return [("pwd", {"tool": "working_dir"}, p.cwd), ls()], (f"cd {d} && pwd", {"tool": "change_dir", "path": d}, f"{p.cwd}/{d}")
    raise ValueError(c)


def _shell(step: tuple[str, dict, object]) -> tuple[str, str]:
    cmd, _, res = step
    if isinstance(res, list):
        res = "\n".join(res) if cmd.startswith("grep") else "  ".join(res)
    elif isinstance(res, int):
        res = f"{res} {cmd.split()[-1]}"
    return cmd, res


def _json(v) -> str:
    """As a tool writes it: every script as it is read, not as \\u escapes."""
    return json.dumps(v, ensure_ascii=False)


def _lines(steps, style: str) -> str:
    out = []
    for step in steps:
        if style == "shell":
            cmd, res = _shell(step)
            out.append(f"$ {cmd}\n" + (f"{res}\n" if res else ""))
        else:
            out.append(_json(step[1]) + "\n" + _json({"result": step[2]}) + "\n")
    return "".join(out)


# An answer's session and the answer together, in bytes. The student reads 512 tokens at a
# time, and a session it cannot see the start of may ask for what it was never shown. These
# sessions come to 0.71 tokens a byte at most, so 640 bytes is under the window with room.
LIMIT = 640


def make(rng: Rng, corner: str) -> Item:
    """One item for a corner: the first of a few draws that fits the student's window, the
    later ones without the aside."""
    it = _make(rng, corner, True)
    for again in range(1, 7):
        if it.kind != "answer" or len((it.ctx + it.cont).encode()) <= LIMIT:
            break
        it = _make(rng.child(f"shorter{again}"), corner, again < 2)
    return it


def _make(rng: Rng, corner: str, asides: bool) -> Item:
    _, lang, style, c = corner.split("/")
    p = Project(rng, lang)
    head = f"{p.user}@{p.name}:~/{p.name}\n" if style == "shell" else f"# agent log: {p.name}\n"
    pool = COMMANDS if style == "shell" else TOOLS
    if c == "session":
        steps = []
        for k in rng.sample(pool, 3):
            before, ask = _steps(p, rng, k)
            steps += before + [ask]
        return Item(kind="passage", corner=corner, fmt="logs", lang="en", source=f"gen:{NAME}", text=(head + _lines(steps, style)).rstrip("\n"))
    # Something else first, half the time, so that the answer is not always in the line before.
    aside = []
    if rng.chance(1, 2) and asides:
        b, a = _steps(p, rng, rng.choice([k for k in ("wc", "head", "grep") if k != c]))
        aside = b + [a]
    before, ask = _steps(p, rng, c)
    if style == "shell":
        cmd, res = _shell(ask)
        return Item(kind="answer", corner=corner, fmt="logs", lang=lang, source=f"gen:{NAME}",
                    ctx=f"{head}{_lines(aside + before, style)}$ {cmd}\n", cont=res)
    full = _json({"result": ask[2]})
    cut = len('{"result": ')
    return Item(kind="answer", corner=corner, fmt="structured", lang=lang, source=f"gen:{NAME}",
                ctx=f"{head}{_lines(aside + before, style)}{_json(ask[1])}\n{full[:cut].rstrip()}", cont=" " + full[cut:])
