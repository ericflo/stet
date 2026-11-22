"""An agent decides: how to run a harness of your own, with a model of your own, in a curator.

A small language model is served inside the sandbox on loopback, and a real agent harness
(pi, https://pi.dev, run by the machine's Node) is pointed at it. The agent is shown what
this round's corpus looks like and makes one decision with its tools; the rest of the
edit is ordinary code. Nothing here needs the network.

  1. Keep English prose, as `tidy` does for every language, and write that script straight
     away, so there is always one. (English only to keep the example short: the exam is in
     every language of the crawl.)
  2. Lay out a small workspace: a few numbers about the round and a few sample pages.
  3. Serve the model named in stet.toml with `transformers serve`, an OpenAI-compatible
     server that understands tool calls.
  4. Unpack the harness (harness.tar.zst: see pack.sh) and run it against that server,
     with a time limit. It reads the workspace and writes one word to choice.txt.
  5. Write the script again, as strict as the agent said.

This is a starting point, not a strong entry. The model is 0.6B parameters and was never
trained for this job: in our trials it made the tool call every time, and what it chose
followed the facts only loosely. So the decision is one of three, and all three are
sensible. An entry would bring a model trained for its harness. What to copy from it:

  - The harness is yours. `entry` is any command; this one is Python that starts two more
    programs. A harness in JavaScript runs on the machine's Node 24, which is on the PATH.
  - Ship a harness as one archive in your bundle and unpack it to /scratch. A bundle
    holds 2,000 files, and `node_modules/` is never packed as it is.
  - Loopback works; nothing else does. Serve on 127.0.0.1 and point the harness there.
  - Give every step a time limit taken from task.time_left(), and keep a script on disk
    before the agent starts. An agent that hangs must cost you nothing but its time.
  - An agent is not deterministic. Your entry is run twice, by two people, and its gain is
    the mean. Bound what the agent can decide so that either answer is one you can live
    with, and check everything it writes before you use it.
"""

import hashlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from stet.curator import Task

SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")
STOPWORDS = (" the ", " of ", " and ", " to ", " in ", " is ", " that ", " for ")
SELLS = (" buy ", " price", " shipping", " cart", " sale", " click ", " subscribe", " cookies")
BYTES_PER_TOKEN = 3.7
MODEL = "qwen3-0.6b"  # the directory under /assets, from stet.toml
PI = "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
RESERVE_SECONDS = 20  # the agent is stopped this long before the limit
# What the agent chooses between: the least prose a page may be, and whether pages that
# read like a shop are dropped outright.
RECIPES = {"loose": (0.0, False), "normal": (0.35, False), "strict": (0.55, True)}

SYSTEM = "You are a careful assistant that edits a dataset. You act by calling tools. /no_think"
JOB = (
    "Here are facts about a sample of web pages:\n{stats}\n"
    "The files page1.txt, page2.txt and page3.txt in this directory hold three of the pages, if you want to read one.\n"
    "Decide how strictly the pages should be filtered before a language model is trained on them: "
    "loose if more than 80 percent are mostly prose; strict if fewer than 30 percent are mostly prose or more than 40 percent are shop-like; normal otherwise.\n"
    "Call the write tool once, with path choice.txt and content set to exactly one word: loose, normal or strict."
)


def looks_english(text: str) -> bool:
    sample = text[:3000]
    if len(sample) < 200:
        return False
    letters = sum(c.isascii() and (c.isalpha() or c == " ") for c in sample)
    if letters / len(sample) < 0.75:
        return False
    low = sample.lower()
    return sum(low.count(w) for w in STOPWORDS) >= max(4, len(sample) // 300)


def prose_ranges(text: str, min_copy: int = 64) -> tuple[list[list[int]], int]:
    """Byte ranges of the lines worth keeping, and how many bytes they hold."""
    ranges: list[list[int]] = []
    pos = 0
    for line in text.split("\n"):
        n = len(line.encode("utf-8")) + 1
        body = line.strip()
        if len(body) >= 60 and body.count(" ") + 1 >= 10 and SENTENCE_END.search(body + " ") is not None:
            if ranges and ranges[-1][1] == pos:
                ranges[-1][1] = pos + n
            else:
                ranges.append([pos, pos + n])
        pos += n
    ranges = [r for r in ranges if r[1] - r[0] >= min_copy + 4]
    return ranges, sum(b - a for a, b in ranges)


def candidates(task: Task) -> tuple[list[dict], dict]:
    """The pages worth considering, best first, and a few numbers about the round."""
    corpus = task.corpus
    min_copy = int(task.info.get("min_copy_bytes", 64))
    seen: set[bytes] = set()
    out, pages, english, repeats = [], 0, 0, 0
    for i, text in corpus:
        pages += 1
        if not looks_english(text):
            continue
        english += 1
        ranges, kept = prose_ranges(text, min_copy)
        if kept < 400:
            continue
        key = hashlib.blake2b(text[:2000].encode("utf-8", "replace"), digest_size=8).digest()
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        size = corpus.size(i)
        ranges = [[a, min(b, size)] for a, b in ranges if a < size and min(b, size) - a >= min_copy]
        low = text[:4000].lower()
        density = kept / max(1, len(text.encode("utf-8")))
        sells = sum(low.count(w) for w in SELLS)
        explains = sum(low.count(w) for w in (" because ", " therefore ", " however ", " for example ", " which ", " whereas "))
        score = density + min(1.0, kept / 3000) + min(explains, 8) * 0.08 - min(sells, 8) * 0.1
        out.append({"d": i, "k": ranges, "bytes": kept, "density": density, "sells": sells, "score": score, "head": text[:700]})
    out.sort(key=lambda c: (-c["score"], c["d"]))
    n = max(1, len(out))
    stats = {
        "pages": pages,
        "english_percent": round(100 * english / max(1, pages)),
        "repeated_percent": round(100 * repeats / max(1, english)),
        "mostly_prose_percent": round(100 * sum(c["density"] >= 0.55 for c in out) / n),
        "shop_like_percent": round(100 * sum(c["sells"] >= 3 for c in out) / n),
    }
    return out, stats


def choose(task: Task, cands: list[dict], recipe: str) -> list[dict]:
    least, no_shops = RECIPES[recipe]
    budget = task.train_tokens * BYTES_PER_TOKEN
    chosen, used = [], 0
    for strict in (True, False):  # if the recipe leaves the budget short, fill it from the rest
        for c in cands:
            if used >= budget:
                break
            passes = c["density"] >= least and not (no_shops and c["sells"] >= 3)
            if passes == strict:
                chosen.append({"d": c["d"], "k": c["k"]})
                used += c["bytes"]
    random.Random(0).shuffle(chosen)
    return chosen


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(task: Task, port: int) -> subprocess.Popen:
    """The model, behind an OpenAI-compatible API on loopback. Returns once it answers."""
    log = open(task.scratch / "serve.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "transformers.cli.transformers", "serve", str(task.asset(MODEL)), "--host", "127.0.0.1", "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    while task.time_left() > RESERVE_SECONDS and proc.poll() is None:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("the model server did not come up: " + (task.scratch / "serve.log").read_text()[-600:])


def unpack_harness(task: Task) -> Path:
    import zstandard

    archive = Path(__file__).with_name("harness.tar.zst")
    if not archive.exists():
        raise RuntimeError("harness.tar.zst is not in the bundle: run ./pack.sh before `stet run` or `stet submit`")
    where = task.scratch / "harness"
    with open(archive, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z, tarfile.open(fileobj=z, mode="r|") as tar:
        tar.extractall(where, filter="tar")
    return where


def ask_agent(task: Task, stats: dict, cands: list[dict]) -> str:
    """Run the harness on the workspace and return what it chose. Raises if it chose nothing usable."""
    work = task.scratch / "work"
    work.mkdir(parents=True, exist_ok=True)
    facts = "".join(f"- {k.replace('_', ' ')}: {v}\n" for k, v in stats.items())
    for n, c in enumerate(cands[:: max(1, len(cands) // 3)][:3], 1):  # a good page, a middling one, a poor one
        (work / f"page{n}.txt").write_text(c["head"])
    harness = unpack_harness(task)
    port = free_port()
    server = serve(task, port)
    try:
        home = task.scratch / "pi"
        home.mkdir(exist_ok=True)
        model = {"id": str(task.asset(MODEL)), "name": "local", "contextWindow": 8192, "maxTokens": 1024}
        provider = {"baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions", "apiKey": "local",
                    "compat": {"supportsDeveloperRole": False}, "models": [model]}  # fmt: skip
        (home / "models.json").write_text(json.dumps({"providers": {"local": provider}}))
        env = {**os.environ, "PI_CODING_AGENT_DIR": str(home), "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
        cmd = ["node", str(harness / PI), "--print", "--offline", "--no-session", "--no-extensions", "--no-skills", "--no-context-files", "--no-mcp",
               "--provider", "local", "--model", model["id"], "--tools", "read,write", "--system-prompt", SYSTEM, JOB.format(stats=facts)]  # fmt: skip
        limit = max(5.0, min(180.0, task.time_left() - RESERVE_SECONDS))
        # Standard input is closed: given a pipe, pi waits to read a prompt from it.
        said = subprocess.run(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=limit)
        print(f"the harness exited {said.returncode}: {(said.stdout.strip() or said.stderr.strip())[-300:]!r}", file=sys.stderr)
    finally:
        server.kill()
    # Whatever the agent wrote is read as data, and only one of three words is accepted.
    choice = (work / "choice.txt").read_text(errors="replace")[:200].strip().lower() if (work / "choice.txt").exists() else ""
    word = re.sub(r"[^a-z]", "", choice.split()[0]) if choice else ""
    if word not in RECIPES:
        raise RuntimeError(f"the agent chose nothing usable (choice.txt: {choice[:60]!r})")
    return word


def main():
    task = Task.from_env()
    cands, stats = candidates(task)
    task.write(choose(task, cands, "normal"))  # the fallback, written before anything can go wrong
    print(f"{len(cands):,} candidates, {stats}, {task.time_left():.0f}s left", file=sys.stderr)
    try:
        recipe = ask_agent(task, stats, cands)
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f"keeping the fallback script: {str(e)[:500]}", file=sys.stderr)
        return
    print(f"the agent chose {recipe!r}, {task.time_left():.0f}s left", file=sys.stderr)
    task.write(choose(task, cands, recipe))


if __name__ == "__main__":
    main()
