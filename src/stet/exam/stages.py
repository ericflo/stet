"""The organisers' measurements on settled rounds, one command a stage: the unedited curve
(`stet curve`), the scale audit (`stet audit`) and the finals (`stet finals`).

The three are one procedure, shared here:

    plan      who and what: which curators, which sizes, how many trainings, what they cost
    scripts   build each size's corpus on this machine and run the curators on it, in the sandbox
    farm      post the students for the organisers' machines to train (farm.py), and follow them
    report    collect what came back and add it up (measure.py; `table`, for the finals: finals.py)

A curator's script is made for one corpus, and a round's corpus is a different one at each
size. So a bigger student needs scripts made on the bigger corpus: that is the `scripts` stage,
the only one that runs anybody's code and the only one that needs a corpus on this machine. It
is run by hand, on a machine of the organisers' that meets the floor, with nothing else running.

Everything a stage makes is kept under one directory (`--out`), by round and size, and nothing
is made twice: a stage that was stopped is run again and carries on. Students trained for one
measurement serve the next (the curve's 1x point is the audit's unedited student, when both sat
the same items), because the server names a batch by its round, items and size.

The exam here is the round's whole exam, with the text of sources that are never published.
The directory is private to the organisers; `stet finals table` writes what may be published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ..spec import CURRENT_SEASON, load_season, parse_round_id
from . import farm, finals, measure
from .aggregate import is_generated, item_table
from .farm import UNEDITED
from .items import Item, digest

MOST_TRAININGS = 9  # of one script in one batch: the farm names the further ones #2 to #9 (server/farm.go)
FARM_TRIES = 3  # machines a student is handed to before the server leaves it undone
SAMPLE_DOCS = 20_000  # documents read to estimate a corpus's tokens
NEAR_THE_END = 0.9  # of that estimate: a budget past this may have reached the end of the corpus, and the training is asked
PATIENCE = 72 * 3600  # how long a stage follows its students before it says what is missing and stops


class Stop(Exception):
    """A stage cannot go on, and says why in words for whoever is running it."""


def say(text: str = "") -> None:
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _read(path: Path):
    return json.loads(path.read_text())


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def root_of(args) -> Path:
    return Path(getattr(args, "out", None) or os.environ.get("STET_MEASUREMENTS") or "~/stet-measurements").expanduser()


# --- the server ------------------------------------------------------------------------------------


def patiently(call, waits=(5, 15, 30, 60, 120)):
    """A call to the server, tried again through a moment without one (a restart, a roll, the
    network): a stage makes dozens of them, and follows its students for hours."""
    from ..client import ApiError

    for wait in (*waits, None):
        try:
            return call()
        except ApiError as e:
            if wait is None or e.status not in (0, 429, 502, 503, 504):
                raise
        time.sleep(wait)


def ask(method: str, path: str, body=None, **kw):
    from .. import client

    return patiently(lambda: client.request(method, path, body, **kw))


def settled_round(season, round_id: str):
    """A settled official round, checked against the beacon as any round is, with what the
    server says of it. The farm trains students on settled rounds only."""
    from ..client import round_from_api

    try:
        sid, kind, index = parse_round_id(round_id)
    except ValueError as e:
        raise Stop(str(e)) from None
    if kind != "official" or sid != season.id:
        raise Stop(f"{round_id} is not an official round of season {season.id}")
    d = ask("GET", f"/api/v1/rounds/{season.id}-o{index:05d}")
    r = d.get("round") or {}
    if r.get("status") != "settled" or not r.get("exam_digest"):
        raise Stop(f"{round_id} has not settled: the organisers' students are trained on settled rounds only")
    try:
        return round_from_api(season, r), d
    except ValueError as e:  # its seed does not follow from the beacon: not a round to measure anything on
        raise Stop(str(e)) from None


def own_scale(season, d: dict):
    """The size a round was run at."""
    return season.scale((d.get("round") or {}).get("scale") or season.official_scale)


# --- the round's exam ------------------------------------------------------------------------------


def _items_of(blob: bytes) -> list[Item]:
    if blob[:4] == b"\x28\xb5\x2f\xfd":  # zstd, as the examiner keeps them
        return farm.unpack_items(blob)
    return [Item.from_dict(json.loads(line)) for line in blob.decode().split("\n") if line.strip()]


def whole_exam(rnd, work: Path, items_file=None, shares_file=None) -> tuple[list[Item], dict]:
    """A settled round's whole exam and the languages of its corpus, as its score used them.

    The server gives out a settled round's published exam only, which leaves out the text of
    most of the writing. The whole one is where it was made: the examiner keeps it beside its
    harvest. A round examined before it did (Season 1's) has only the harvest there, and its
    exam is not made again from that: its items and languages are brought as files. Found
    once, it is kept under `work`. Whatever its source, it must hash to the digest the round
    committed to."""
    from ..data import cache_dir
    from . import examine

    kept, kept_shares = work / rnd.id / "exam.jsonl.zst", work / rnd.id / "languages.json"
    if not (kept.exists() and kept_shares.exists()):
        made = cache_dir() / "examine" / rnd.id
        if items_file and shares_file:
            items, shares = _items_of(Path(items_file).read_bytes()), _read(Path(shares_file))
        elif (made / examine.KEPT_ITEMS).exists() and (made / examine.KEPT_LANGUAGES).exists():
            items, shares = _items_of((made / examine.KEPT_ITEMS).read_bytes()), _read(made / examine.KEPT_LANGUAGES)
        elif (made / "fresh.jsonl").exists():
            raise Stop(f"{rnd.id}: the examiner kept its harvest here ({made}) but not its whole exam, as it did before Season 2, and an exam is "
                       "not made again from a harvest: bring its items and languages here and name them with --items and --shares")
        else:
            raise Stop(f"{rnd.id}: its whole exam is not on this machine. It is kept where the round was examined ({made}): "
                       "run this there, or bring its items and languages here and name them with --items and --shares")
        if digest(items) != rnd.exam_digest:
            raise Stop(f"{rnd.id}: those items are not the exam the round committed to ({rnd.exam_digest[:12]})")
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_bytes(farm.pack_items(items))
        _write(kept_shares, shares)
    items, shares = _items_of(kept.read_bytes()), _read(kept_shares)
    if digest(items) != rnd.exam_digest:
        raise Stop(f"{rnd.id}: the exam kept at {kept} is not the one the round committed to; remove it and run this again")
    return items, shares


def corpus_tokens(corpus, enc, sample: int = SAMPLE_DOCS) -> int:
    """About how many tokens a corpus is, as the trainer counts them (each document, and the
    end-of-text token before it), from its first `sample` documents. The corpus is written in a
    seeded shuffle, so they are a fair sample of it."""
    from ..tok import encode_batch

    n = min(len(corpus), sample)
    tokens = size = 0
    for i in range(0, n, 256):
        texts = [corpus.text(j) for j in range(i, min(n, i + 256))]
        tokens += sum(len(t) + 1 for t in encode_batch(enc, texts))
        size += sum(corpus.size(j) for j in range(i, min(n, i + 256)))
    return tokens if n == len(corpus) else round(tokens * corpus.total_bytes / max(1, size))


def prepare(rnd, scale, work: Path, log=say) -> dict:
    """Build a size's corpus on this machine, and note what the later stages need of it so that
    they need no corpus: how many documents it has (the unedited script, and so its hash,
    follows) and about how many tokens (how far a training budget reaches into it)."""
    path = work / rnd.id / scale.name / "corpus.json"
    if path.exists():
        return _read(path)
    from .. import data
    from ..tok import load_encoding

    log(f"{rnd.id}: the {scale.name} corpus ...")
    corpus = data.corpus(rnd, scale)
    with tempfile.TemporaryDirectory(prefix="stet-measure-") as tmp:
        unedited = farm.unedited_script(Path(tmp) / "unedited.jsonl", len(corpus))
    info = {"round": rnd.id, "scale": scale.name, "n_docs": len(corpus), "bytes": corpus.total_bytes, "sha256": corpus.meta.get("sha256"),
            "tokens": corpus_tokens(corpus, load_encoding(rnd.season)), "unedited": unedited}
    _write(path, info)
    log(f"{rnd.id}: {scale.name}: {info['n_docs']:,} documents, {info['bytes'] / 1e9:.1f} GB, about {info['tokens'] / 1e6:,.0f}M tokens")
    return info


def prepared(rnd_id: str, size: str, work: Path) -> dict:
    path = work / rnd_id / size / "corpus.json"
    if not path.exists():
        raise Stop(f"{rnd_id}: the {size} corpus has not been built for this yet: run the scripts stage, on a machine that can build it")
    return _read(path)


def _seen_path(rnd_id: str, size: str, items: list[Item], work: Path) -> Path:
    return work / rnd_id / size / f"seen-{digest(items)[:12]}.json"


def check_exam(rnd, scale, items: list[Item], work: Path, log=say) -> set[str]:
    """The exam's fresh items that this size's corpus already holds, by id. A round's exam was
    made against the corpus of the round's own size; a bigger one is more of the same crawl, and
    may hold text the first did not. Such an item would be a question the student had seen."""
    path = _seen_path(rnd.id, scale.name, items, work)
    if path.exists():
        return set(_read(path))
    from .. import data
    from .decontam import seen

    fresh = [it for it in items if not is_generated(it)]
    hit = seen(fresh, data.corpus(rnd, scale), rnd.season, log=log)
    ids = sorted(fresh[i].id for i in hit)
    _write(path, ids)
    if ids:
        log(f"{rnd.id}: {len(ids)} of the exam's items are in the {scale.name} corpus, and are left out")
    return set(ids)


def exam_at(rnd_id: str, items: list[Item], sizes: list[str], own: str, work: Path) -> list[Item]:
    """The items students of these sizes sit: the round's exam, less what a bigger corpus holds.
    Every size sits the same ones, so that the sizes are compared item for item."""
    drop: set[str] = set()
    for size in sizes:
        if size == own:
            continue
        path = _seen_path(rnd_id, size, items, work)
        if not path.exists():
            raise Stop(f"{rnd_id}: the exam has not been checked against the {size} corpus yet: run the scripts stage, on a machine that can build it")
        drop |= set(_read(path))
    return [it for it in items if it.id not in drop]


# --- curators' scripts at a size -------------------------------------------------------------------


def label_of(e: dict) -> str:
    """A curator as a file name: its owner, its name and its version, with a full stop between
    them. A handle and a name are lowercase letters, digits and hyphens (the server's handleRe
    and editorNameRe), so no two curators share one: joined by hyphens, foo-bar/baz and
    foo/bar-baz did, and one was given the script kept for the other."""
    return ".".join(re.sub(r"[^a-z0-9]+", "-", str(part).lower()).strip("-") for part in (e.get("owner"), e.get("name"), f"v{e.get('version')}"))


def name_of(e: dict) -> str:
    return f"{e.get('owner')}/{e.get('name')}@v{e.get('version')}"


def _edit(*a, **kw):
    from ..pipeline import edit_round

    return edit_round(*a, **kw)


def make_script(who: dict, rnd, scale, sdir: Path, sandbox: str, time_limit: float | None, key, gpu: int = 0, log=say) -> dict:
    """Run one curator on this size's corpus, on this machine, as a verifier would (the same
    sandbox, the size's own time limit), and keep the script it writes. A curator that fails is
    noted as a failure and not run again: by the rules that run scores as the unedited corpus."""
    from ..client import MACHINE_ERRNOS, SANDBOXES
    from ..project import MAX_BUNDLE_BYTES, load_project, unpack
    from ..runner import remove_tree

    note = sdir / f"{who['label']}.json"
    if note.exists():
        return _read(note)
    if sandbox not in SANDBOXES and os.environ.get("STET_DEV") != "1":
        raise Stop("refusing to run an entrant's curator outside the sandbox")
    blob = ask("GET", f"/api/v1/editors/{who['id']}/bundle", key=key, raw=True, timeout=600, limit=MAX_BUNDLE_BYTES)
    if _sha(blob) != who["id"]:
        raise Stop(f"{who['label']}: the bundle the server sent does not match the curator's id")
    limit = float(time_limit or scale.edit_seconds)
    run = sdir / "runs" / who["label"]
    remove_tree(run)
    run.mkdir(parents=True)
    rec: dict = {"label": who["label"], "editor": who["id"], "scale": scale.name, "script": None}
    try:
        try:
            unpack(blob, run / "bundle")
            project = load_project(run / "bundle")
        except Exception as e:
            if isinstance(e, MemoryError) or (isinstance(e, OSError) and e.errno in MACHINE_ERRNOS):
                raise
            rec.update(status="error", seconds=0.0, time_limit=limit, sandbox=sandbox, reason=f"the bundle could not be loaded: {e}"[:300])
        else:
            edit = _edit(run / "bundle", project.entry, rnd, scale, run, sandbox=sandbox, log=lambda m: None,
                         limits={"required": sandbox in SANDBOXES}, assets=project.assets, time_limit=limit, verify_assets=True, gpu=gpu)
            rec.update(edit.to_dict())
            if edit.script is not None:
                raw = Path(edit.script).read_bytes()
                (sdir / f"{who['label']}.jsonl").write_bytes(raw)
                rec.update(script=_sha(raw), bytes=len(raw))
    finally:
        remove_tree(run)
    _write(note, rec)
    said = f"script in {rec['seconds']:.0f}s of {limit:.0f}s, {rec['bytes'] / 1e6:.1f} MB" if rec["script"] else f"no script ({rec['status']}: {rec.get('reason')})"
    log(f"  {who['label']} · {scale.name}: {said}")
    return rec


def script_at(who: dict, size: str, own: str, rnd_id: str, work: Path) -> tuple[str | None, Path | None]:
    """A curator's script at a size: (its sha256, the file here), or (None, None) where it wrote
    none. At the round's own size it is the script the round scored, which the server has."""
    if size == own and who.get("script"):
        return who["script"], None
    sdir = work / rnd_id / size / "scripts"
    note = sdir / f"{who['label']}.json"
    if not note.exists():
        raise Stop(f"{rnd_id}: {who['label']} has not been run on the {size} corpus yet: run the scripts stage")
    sha = _read(note).get("script")
    return (sha, sdir / f"{who['label']}.jsonl") if sha else (None, None)


# --- students: posted, followed, read back ---------------------------------------------------------


def refs_of(script: str, n: int) -> list[str]:
    """What the farm calls the trainings of one script: its name, then #2, #3, ..."""
    return [script] + [f"{script}#{k}" for k in range(2, n + 1)]


def batch_dir(rnd_id: str, size: str, items: list[Item], work: Path) -> Path:
    return work / rnd_id / size / f"farm-{digest(items)[:12]}"


def wanted_refs(students: dict[str, int], lost) -> list[str]:
    """The trainings wanted, by the names the farm knows them by: of each script, its first
    names the server has not given up on. A student three machines failed at is never handed
    out again under that name, so the next name takes its place."""
    return [r for script, n in students.items() for r in [x for x in refs_of(script, MOST_TRAININGS) if x not in lost][:n]]


def post(rnd, scale, items: list[Item], students: dict[str, int], files: dict[str, Path], bdir: Path, key, spec: dict | None = None, lease: int = 0) -> dict:
    """Post a size's students: for each script ("unedited", or a sha256) how many trainings of
    it are wanted. Scripts made here (`files`) are sent first; a round's own are the server's
    already. Only students that are not back are posted, so posting again adds what is new,
    replaces what the server gave up on, and trains nothing twice."""
    import zstandard

    note = bdir / "batch.json"
    had = _read(note) if note.exists() else {}
    # Another measurement may keep its students here (the curve's 1x point and an audit's
    # unedited student are the same trainings): what either wants stays wanted.
    students = {**had.get("students", {}), **{s: max(n, had.get("students", {}).get(s, 0)) for s, n in students.items()}}
    if max(students.values()) > MOST_TRAININGS:
        raise Stop(f"the farm trains one script at most {MOST_TRAININGS} times in a batch")
    rec = {"batch": had.get("batch"), "round": rnd.id, "scale": scale.name, "spec": spec, "lease": lease, "items": len(items), "items_digest": digest(items),
           "students": students, "sent": had.get("sent", []), "lost": had.get("lost", [])}
    todo = [r for r in wanted_refs(students, rec["lost"]) if not (bdir / f"{r}.json").exists()]
    for sha in sorted({r.split("#")[0] for r in todo}):
        if sha in files and sha not in rec["sent"]:
            ask("PUT", f"/api/v1/admin/farm/scripts/{sha}", zstandard.ZstdCompressor(level=10).compress(files[sha].read_bytes()), key=key, timeout=900)
            rec["sent"] = sorted({*rec["sent"], sha})
            _write(note, rec)
    if todo:
        rec["batch"] = patiently(lambda: farm.post_batch(rnd, scale, items, todo, key, spec=spec, lease=lease))
    _write(note, rec)
    return rec


def collect(bdir: Path, key, only: dict[str, int] | None = None) -> dict:
    """One look at a batch. Every student that has come back is kept, once, as it came. The
    rest are counted (those of `only`, when the batch is shared with another measurement):
    still to come, or to be posted again because the server gave up on them or on the batch."""
    from ..client import ApiError

    path = bdir / "batch.json"
    note = _read(path)
    lost = set(note.get("lost", []))

    def have(ref: str) -> bool:
        return (bdir / f"{ref}.json").exists()

    state = {"wanted": 0, "done": 0, "left": 0, "under_way": 0, "unposted": 0, "forgotten": False}
    tasks: dict[str, dict] = {}
    try:
        b = ask("GET", f"/api/v1/admin/farm/batches/{note['batch']}", key=key, timeout=300, limit=256 << 20)
    except ApiError as e:
        if e.status != 404:
            raise
        # The server keeps its newest batches only. What was collected is here; the rest is
        # posted again, in a batch where every name starts afresh.
        state["forgotten"], lost = True, set()
    else:
        now = time.time()
        for t in b["tasks"]:
            ref, d = t["script"], t.get("done")
            if ref.split("#")[0] not in note["students"]:
                continue
            t["held"] = bool(t.get("holder") and t.get("until", 0) > now)
            tasks[ref] = t
            if d and not have(ref):
                valid, got = bool(d.get("valid")), t.get("item_scores")
                if not valid or (isinstance(got, list) and len(got) == note["items"]):
                    _write(bdir / f"{ref}.json", {"script": ref, "valid": valid, "reason": d.get("reason"), "by": d.get("by"), "hardware": d.get("hardware"),
                                                  "kept_doc_frac": d.get("kept_doc_frac"), "train_seconds": d.get("train_seconds"), "item_scores": got if valid else None})
            if have(ref):
                lost.discard(ref)  # a machine thought lost that finished after all
            elif not d and not t["held"] and t.get("tries", 0) >= FARM_TRIES:
                lost.add(ref)
    if lost != set(note.get("lost", [])):
        _write(path, {**note, "lost": sorted(lost)})
    for ref in wanted_refs(only or note["students"], lost):
        state["wanted"] += 1
        if have(ref):
            state["done"] += 1
        elif ref not in tasks:
            state["unposted"] += 1
        else:
            state["left"] += 1
            state["under_way"] += tasks[ref]["held"]
    return state


def follow(batches: list[tuple[Path, dict | None]], key, log=say, patience: float = PATIENCE, wait: bool = True) -> bool:
    """Look at the batches (each with the students of it that are waited for, or None for all)
    until every student is back or given up, keeping each as it arrives. True when all are back.
    Stopped at any time, it is run again and carries on."""
    t0, said = time.time(), None
    while True:
        states = [collect(b, key, only) for b, only in batches]
        done, wanted, left = (sum(s[k] for s in states) for k in ("done", "wanted", "left"))
        now = f"{done} of {wanted} trained" + (f", {sum(s['under_way'] for s in states)} under way" if left else "")
        if now != said:
            log(f"  {now} [{time.time() - t0:.0f}s]")
            said = now
        if not left or not wait or time.time() - t0 > patience:
            for s, (b, _) in zip(states, batches, strict=True):
                if s["unposted"]:
                    why = "the server no longer has the batch" if s["forgotten"] else "three machines failed at each"
                    log(f"  {b.parent.name}: {s['unposted']} students were not trained ({why}). Run this stage again to post them again.")
            return done == wanted
        time.sleep(farm.POLL)


def trained(bdir: Path, script: str) -> list[dict]:
    """Every training of a script that came back, in the order they are named."""
    out = []
    for ref in refs_of(script, MOST_TRAININGS):
        path = bdir / f"{ref}.json"
        if path.exists():
            out.append(_read(path))
    return out


def scores(bdir: Path, script: str) -> list[list[float]]:
    return [t["item_scores"] for t in trained(bdir, script) if t.get("valid") and t.get("item_scores")]


def refused(bdir: Path, script: str) -> str | None:
    """Why the trainer refused a script, if it did: one that breaks a rule at this size."""
    return next((t.get("reason") or "not a script" for t in trained(bdir, script) if not t.get("valid")), None)


def read_twice(passes: float, bdir: Path) -> bool:
    """Whether the unedited students kept here read some of the corpus twice, and so are not
    "that many times the data". The estimate of the corpus's tokens says whether the budget
    fits. Near the end, the trainings say whether it did: a student that drew on every document
    reached the end."""
    used_up = any((t.get("kept_doc_frac") or 0) >= 0.9999 for t in trained(bdir, UNEDITED) if t.get("valid"))
    return passes > 1 or (used_up and passes > NEAR_THE_END)


# --- what a plan costs -----------------------------------------------------------------------------


def cost(season, rows: list[tuple]) -> list[dict]:
    """For each (scale, trainings): what the trainings take on the season's floor."""
    out = []
    for scale, n in rows:
        each = measure.train_seconds(season, scale) / 3600
        out.append({"scale": scale.name, "tokens": scale.train_tokens, "trainings": n, "hours_each": each, "hours": n * each,
                    "hold_minutes": (measure.hold_seconds(season, scale) or measure.SERVER_HOLD) // 60, "memory_gb": measure.memory_gb(scale)})
    return out


def say_cost(season, rows: list[dict], speed: float = 1.0) -> None:
    from ..client import table

    floor = season.raw.get("runner") or {}
    say(table([[r["scale"], f"{r['tokens'] / 1e6:,.0f}M", str(r["trainings"]), f"{r['hours_each'] / speed:.2f}", f"{r['hours'] / speed:.1f}",
                str(r["hold_minutes"]), f"{r['memory_gb']:.1f}"] for r in rows],
              ["students", "tokens", "trainings", "GPU hours each", "GPU hours", "held (min)", "memory (GB)"], right=(1, 2, 3, 4, 5, 6)))
    total = sum(r["hours"] for r in rows) / speed
    where = "on the slowest GPU that may run curators" if speed == 1 else f"on a GPU {speed:g} times the floor"
    say(f"{total:.1f} GPU hours {where} (the season's floor: {floor.get('gpu_tokens_per_second', 0):,} tokens a second in `stet bench`).")
    if speed == 1:
        say("A faster card takes less, in about the proportion of its `stet bench` to the floor's: --speed 2 shows a card twice the floor.")
    long = max((r["hold_minutes"] for r in rows), default=0)
    if long > 60:
        say(f"The longest students are held for up to {long} minutes: they go only to machines set aside for them (stet-cloud --long-farm {long}).")
    short = float(floor.get("memory_gb") or 0)
    for r in rows:
        if short and r["memory_gb"] > short - 4:
            say(f"{r['scale']} needs about {r['memory_gb']:.0f} GB of memory for its tokens: more than a machine at the floor ({short:g} GB) has to spare.")


def say_noise(season, size: str, trainings: int) -> None:
    sd = ((season.raw.get("noise") or {}).get(size) or {}).get("gain_sd")
    if sd:
        say(f"{size}: training again moves a gain by about ±{sd:.3f}; with {trainings} trainings a curator's gain is known to about ±{sd / math.sqrt(trainings):.4f}.")
    else:
        say(f"{size}: the season has no noise figure for this size yet; the report measures it from the repeated trainings.")


# --- stet curve ------------------------------------------------------------------------------------


def curve_points(season, scale, multiples: list[float], at_one: int, elsewhere: int, corpus_tokens_: int) -> list[dict]:
    """The students of a curve: the unedited corpus at each multiple of the budget."""
    out = []
    for m in sorted({1.0, *multiples}):
        s = measure.at_multiple(scale, m)
        out.append({"multiple": m, "scale": s, "spec": None if m == 1 else measure.spec_of(s), "lease": measure.hold_seconds(season, s),
                    "trainings": at_one if m == 1 else elsewhere, "passes": measure.passes(s, corpus_tokens_)})
    return out


def multiple_text(m: float) -> str:
    return f"{m:g}x"


def gain_text(v) -> str:
    return "" if v is None else f"{v:+.4f}"


def reference_line(key: str, ref: list[list[float]]) -> str:
    return f'"{key}": ' + json.dumps([[int(m) if float(m).is_integer() else m, int(v) if v == 0 else v] for m, v in ref])


def say_curve(title: str, points: list[dict], key: str) -> None:
    from ..client import table

    rows = []
    for p in points:
        note = ""
        if p.get("repeats"):
            note = f"read some of the corpus twice ({p['passes']:.2f} of it by the estimate): not {multiple_text(p['multiple'])} the data, and left out below"
        elif p.get("passes", 0) > NEAR_THE_END:
            note = f"{100 * p['passes']:.0f}% of the corpus"
        n = p.get("trainings", p.get("rounds"))
        rows.append([multiple_text(p["multiple"]), str(n), "0" if p["multiple"] == 1 else gain_text(p["gain"]), "" if p.get("sd") is None else f"±{p['sd']:.4f}", note])
    say(title)
    say(table(rows, ["budget", "trainings" if "trainings" in points[0] else "rounds", "gain (bits per byte)", "spread", ""], right=(0, 1, 2, 3)))
    used = [p for p in points if not p.get("repeats")]
    if len(used) > 1:
        say("per doubling of the data: " + " · ".join(f"{multiple_text(s['from'])} to {multiple_text(s['to'])} {s['per_doubling']:+.4f}" for s in measure.slopes(used)))
        if not measure.rises(used):
            say("More data did not always do better here: this curve cannot be read as a multiplier. Train its points again, or average more rounds.")
        say('for "reference_curves" in the season file (and server/season.json):')
        sys.stdout.write(reference_line(key, measure.as_reference(used)) + "\n")
        sys.stdout.flush()


def curve_of(rnd_id: str, scale, points: list[dict], sat: list[Item], shares: dict, work: Path, plan=None) -> dict | None:
    """A round's curve from the students that are back, or None while there is no 1x student."""
    got = {p["multiple"]: (batch_dir(rnd_id, p["scale"].name, sat, work), p) for p in points}
    each = {m: scores(b, UNEDITED) for m, (b, _) in got.items()}
    if not each.get(1.0):
        return None
    out = measure.curve(item_table(sat, plan=plan), shares, {m: v for m, v in each.items() if v}, plan=plan)
    for row in out:
        bdir, p = got[row["multiple"]]
        row.update(tokens=p["scale"].train_tokens, passes=p["passes"], repeats=read_twice(p["passes"], bdir),
                   kept_doc_frac=[t.get("kept_doc_frac") for t in trained(bdir, UNEDITED) if t.get("valid")])
    used = [r for r in out if not r["repeats"]]
    missing = [p["multiple"] for p in points if len(each.get(p["multiple"], [])) < p["trainings"]]
    return {"round": rnd_id, "scale": scale.name, "key": f"official/{scale.name}", "items": len(sat), "items_digest": digest(sat), "points": out,
            "slopes": measure.slopes(used), "reference": measure.as_reference(used), "rises": measure.rises(used), "missing": missing}


def cmd_curve(args):
    from ..client import need_key

    season, work = load_season(args.season), root_of(args)
    if args.average:
        return _curve_average(args, season, work)
    key = need_key()
    for rid in args.rounds:
        rnd, d = settled_round(season, rid)
        own = own_scale(season, d)
        scale = season.scale(args.scale or own.name)
        info = prepare(rnd, scale, work)
        points = curve_points(season, scale, args.multiples, args.repeats, args.repeats_elsewhere, info["tokens"])
        say(f"{rnd.id} · the unedited curve · {scale.name} · a corpus of about {info['tokens'] / 1e6:,.0f}M tokens")
        say_cost(season, cost(season, [(p["scale"], p["trainings"]) for p in points]), args.speed)
        for p in points:
            if p["passes"] > 1:
                say(f"{multiple_text(p['multiple'])} is {p['scale'].train_tokens / 1e6:,.0f}M tokens, more than the corpus has: that student would read it "
                    f"{p['passes']:.2f} times, which is not {multiple_text(p['multiple'])} the data. It is trained and shown, marked, and is no point of the curve.")
            elif p["passes"] > NEAR_THE_END:
                say(f"{multiple_text(p['multiple'])} is {p['scale'].train_tokens / 1e6:,.0f}M tokens, {100 * p['passes']:.0f}% of the corpus by this estimate: "
                    "whether it reached the end is read from the training itself.")
        if args.plan:
            continue
        items, shares = whole_exam(rnd, work, args.items, args.shares)
        if scale.name != own.name:
            check_exam(rnd, scale, items, work)
        sat = exam_at(rnd.id, items, [scale.name], own.name, work)
        batches = []
        for p in points:
            bdir = batch_dir(rnd.id, p["scale"].name, sat, work)
            post(rnd, p["scale"], sat, {UNEDITED: p["trainings"]}, {}, bdir, key, spec=p["spec"], lease=p["lease"])
            batches.append((bdir, {UNEDITED: p["trainings"]}))
        say(f"{rnd.id}: {sum(p['trainings'] for p in points)} students of the unedited corpus, sitting {len(sat):,} items")
        follow(batches, key, wait=not args.no_wait)
        from .aggregate import plan_of

        out = curve_of(rnd.id, scale, points, sat, shares, work, plan=plan_of(season))
        if out is None:
            say(f"{rnd.id}: no 1x student is back yet, and the curve is measured from it. Run this again later.")
            continue
        _write(work / rnd.id / f"curve-{scale.name}.json", out)
        if out["missing"]:
            say(f"{rnd.id}: still to come: {', '.join(multiple_text(m) for m in out['missing'])}. What follows is what is back; run this again for the rest.")
        say_curve(f"{rnd.id} · {scale.name} · gain over the 1x unedited student (spread: between trainings)", out["points"], out["key"])


def _curve_average(args, season, work: Path) -> None:
    scale = season.scale(args.scale or season.official_scale)
    curves, missing = [], []
    for rid in args.rounds:
        path = work / rid / f"curve-{scale.name}.json"
        if not path.exists():
            missing.append(rid)
            continue
        curves.append([p for p in _read(path)["points"] if not p.get("repeats")])
    if missing:
        say(f"no {scale.name} curve measured yet for: {', '.join(missing)} (stet curve <round> --scale {scale.name})")
    if not curves:
        raise Stop("no curve to average")
    points = measure.average(curves)
    out = {"scale": scale.name, "key": f"official/{scale.name}", "rounds": [r for r in args.rounds if r not in missing], "points": points,
           "reference": measure.as_reference(points), "rises": measure.rises(points)}
    _write(work / f"curve-{scale.name}.json", out)
    say_curve(f"{len(curves)} rounds · {scale.name} · gain over the 1x unedited student, mean of the rounds (spread: between rounds)", points, out["key"])


# --- stet audit ------------------------------------------------------------------------------------


def next_size(season, own: str) -> str | None:
    """The season's next size of student up from this one."""
    by = sorted(season.raw["scales"], key=lambda s: season.scale(s).train_tokens)
    bigger = by[by.index(own) + 1 :]
    return bigger[0] if bigger else None


def _audit_plan(args, season, work: Path, key) -> None:
    rnd, d = settled_round(season, args.round)
    own = own_scale(season, d).name
    sizes = args.sizes or [own] + ([next_size(season, own)] if next_size(season, own) else [])
    if sizes[0] != own:
        raise Stop(f"{rnd.id} was run at {own}: the sizes audited start there ({own},{','.join(s for s in sizes if s != own)})")
    entries = []
    for e in d.get("entries") or []:
        sha = (e.get("script") or {}).get("sha256")
        if not e.get("valid") or e.get("void") or not sha:
            continue
        entries.append({"label": label_of(e), "name": name_of(e), "id": e["editor"], "script": sha, "gain": e.get("gain")})
    if args.entries:
        entries = [e for e in entries if e["label"] in args.entries or e["name"] in args.entries]
    if len(entries) < 3:
        raise Stop(f"{rnd.id} has {len(entries)} entries with a script: an order of fewer than three says nothing")
    plan = {"round": rnd.id, "exam_digest": rnd.exam_digest, "sizes": sizes, "trainings": args.trainings, "baselines": args.baselines, "entries": entries}
    _write(work / rnd.id / "audit.json", plan)
    say(f"{rnd.id} · audit · {' and '.join(sizes)} · {len(entries)} curators, each script trained {args.trainings} times, the unedited corpus {args.baselines}")
    for e in entries:
        say(f"  {e['name']}  {gain_text(e['gain'])}")
    say_cost(season, cost(season, [(season.scale(s), len(entries) * args.trainings + args.baselines) for s in sizes]), args.speed)
    for s in sizes:
        say_noise(season, s, args.trainings)
    for s in sizes[1:]:
        say(f"Running {len(entries)} curators on the {s} corpus takes up to {len(entries) * season.scale(s).edit_seconds / 3600:.1f} hours on this machine "
            f"({season.scale(s).edit_seconds}s each), and building the corpus comes first.")
    say(f"Next: stet audit scripts {rnd.id}" + (f"   (here, by hand: it builds the corpora and runs the curators at {', '.join(sizes[1:])})" if len(sizes) > 1 else ""))


def _audit(work: Path, round_id: str) -> dict:
    path = work / round_id / "audit.json"
    if not path.exists():
        raise Stop(f"{round_id} has no audit planned: stet audit plan {round_id}")
    return _read(path)


def _audit_scripts(args, season, work: Path, key) -> None:
    plan = _audit(work, args.round)
    rnd, _ = settled_round(season, plan["round"])
    own = plan["sizes"][0]
    items, _ = whole_exam(rnd, work, args.items, args.shares)
    for size in plan["sizes"]:
        scale = season.scale(size)
        prepare(rnd, scale, work)
        if size == own:
            continue  # the scripts the round scored are the server's already
        check_exam(rnd, scale, items, work)
        made = [make_script(e, rnd, scale, work / rnd.id / size / "scripts", args.sandbox, args.time_limit, key, gpu=args.gpu) for e in plan["entries"]]
        say(f"{rnd.id} · {size}: {sum(1 for m in made if m['script'])} of {len(made)} curators wrote a script")
    say(f"Next: stet audit farm {rnd.id}")


def _audit_batches(plan: dict, season, work: Path) -> tuple[list[Item], dict, list[tuple]]:
    """What an audit trains, from what is on disk: the items sat, the corpus's languages, and for
    each size (the scale, where its students are kept, each script's trainings, the scripts to
    send, and which curator has which script)."""
    rid, own = plan["round"], plan["sizes"][0]
    kept = work / rid / "exam.jsonl.zst"
    if not kept.exists():
        raise Stop(f"{rid}: its exam is not here yet: run the scripts stage")
    items, shares = _items_of(kept.read_bytes()), _read(work / rid / "languages.json")
    sat = exam_at(rid, items, plan["sizes"], own, work)
    at = {size: {e["name"]: script_at(e, size, own, rid, work) for e in plan["entries"]} for size in plan["sizes"]}
    same = {size: prepared(rid, size, work)["unedited"] for size in plan["sizes"]}
    # A curator with no script at one size, or one that keeps the corpus as it is, has no place
    # in an order between sizes: none of its students is trained.
    out_of_it = {e["name"] for e in plan["entries"] if any(at[size][e["name"]][0] in (None, same[size]) for size in plan["sizes"])}
    out = []
    for size in plan["sizes"]:
        unedited = same[size]
        students, files, who = {UNEDITED: plan["baselines"]}, {}, {}
        for e in plan["entries"]:
            sha, path = at[size][e["name"]]
            who[e["name"]] = UNEDITED if sha == unedited else sha  # a script that keeps the corpus as it is, is the unedited student
            if sha and sha != unedited and e["name"] not in out_of_it:
                students[sha] = plan["trainings"]
                if path:
                    files[sha] = path
        out.append((season.scale(size), batch_dir(rid, size, sat, work), students, files, who))
    return sat, shares, out


def _audit_farm(args, season, work: Path, key) -> None:
    plan = _audit(work, args.round)
    rnd, _ = settled_round(season, plan["round"])
    sat, _, batches = _audit_batches(plan, season, work)
    for scale, bdir, students, files, _ in batches:
        b = post(rnd, scale, sat, students, files, bdir, key, lease=measure.hold_seconds(season, scale))
        say(f"{rnd.id} · {scale.name}: {sum(students.values())} students, sitting {len(sat):,} items (batch {b['batch']})")
    if follow([(b[1], b[2]) for b in batches], key, wait=not args.no_wait):
        say(f"Next: stet audit report {rnd.id}")


def _ratio(v) -> str:
    return "" if v is None else f"{v:.1f}"


def _rho(v) -> str:
    return "" if v is None else f"{v:+.2f}"


def _measure_name(m: str) -> str:
    return "the score" if m == measure.SCORE else m.removeprefix(measure.PILLAR)


def say_audit(rep: dict) -> None:
    from ..client import table

    sizes = rep["sizes"]
    say(f"{rep['round']} · audit · {' and '.join(sizes)} · {rep['items']:,} items" + (f" ({rep['dropped']} left out: a bigger corpus holds them)" if rep.get("dropped") else ""))
    order = {s: {n: i + 1 for i, n in enumerate(sorted((n for n, g in rep["curators"].items() if s in g), key=lambda n: -rep["curators"][n][s]))} for s in sizes}
    rows = [[n] + [c for s in sizes for c in ((gain_text(g[s]), str(order[s][n])) if s in g else ("", ""))] for n, g in sorted(rep["curators"].items(), key=lambda x: -x[1].get(sizes[0], -9e9))]
    say(table(rows, ["curator"] + [c for s in sizes for c in (s, "rank")], right=tuple(range(1, 1 + 2 * len(sizes)))))
    for n, why in sorted(rep.get("left_out", {}).items()):
        say(f"  left out: {n}: {why}")
    for pair, rho in rep["measures"][measure.SCORE]["carried"].items():
        a, b = pair.split(">")
        say(f"The order of curators carried from {a} to {b} at {_rho(rho) or 'too few curators to say'} (rank agreement: 1 is the same order).")
    rows = []
    for m, row in rep["measures"].items():
        cells = [_measure_name(m)]
        for s in sizes:
            st = row["sizes"].get(s) or {}
            cells += ["" if st.get("noise") is None else f"{st['noise']:.4f}", "" if st.get("spread") is None else f"{st['spread']:.4f}", _ratio(st.get("spread_over_noise"))]
        cells += [_rho(v) for v in row["carried"].values()] or [""] * (len(sizes) - 1)
        rows.append(cells)
    head = [""] + [c for s in sizes for c in (f"{s} noise", "spread", "over noise")] + [f"carried to {s}" for s in sizes[1:]]
    say(table(rows, head, right=tuple(range(1, len(head)))))
    say("One round's order rests on a few curators. Every round audited, as one: stet audit report --over <rounds>.")


def _audit_report(args, season, work: Path, key) -> None:
    from ..client import ApiError

    if args.over:
        return _audit_over(args.over, work)
    plan = _audit(work, args.round)
    sat, shares, batches = _audit_batches(plan, season, work)
    for _, bdir, *_ in batches:
        if (bdir / "batch.json").exists() and key is not None:
            try:
                collect(bdir, key)
            except ApiError as e:
                say(f"the server could not be asked for what is new ({e}); this is what was collected before")
    from .aggregate import plan_of

    table_, stats, why = item_table(sat, plan=plan_of(season)), {}, {}
    for scale, bdir, _, _, who in batches:
        unedited = scores(bdir, UNEDITED)
        if not unedited:
            raise Stop(f"{plan['round']}: no {scale.name} student of the unedited corpus is back yet: there is nothing to measure from")
        curators = {}
        for name, sha in who.items():
            if sha == UNEDITED:
                why.setdefault(name, []).append((0, f"its {scale.name} script keeps the corpus as it is: it is the unedited student"))
            elif sha is None:
                why.setdefault(name, []).append((1, f"it wrote no script at {scale.name}"))
            elif refused(bdir, sha):
                why.setdefault(name, []).append((2, f"its {scale.name} script breaks a rule: {refused(bdir, sha)}"))
            elif scores(bdir, sha):
                curators[name] = scores(bdir, sha)
            else:
                why.setdefault(name, []).append((3, f"its {scale.name} student is not back yet"))
        stats[scale.name] = curators, unedited
    left_out = {name: min(w)[1] for name, w in why.items()}  # the reason that would hold whatever else came back
    if sum(1 for e in plan["entries"] if e["name"] not in left_out) < 3:
        raise Stop(f"{plan['round']}: fewer than three curators have a student back at every size: there is no order to report on yet")
    # An order is compared between sizes, so a curator counts where it has a student at every size.
    for scale, *_ in batches:
        curators, unedited = stats[scale.name]
        stats[scale.name] = measure.size_stats(table_, shares, unedited, {n: s for n, s in curators.items() if n not in left_out})
    rep = measure.audit(stats)
    whole = len(_items_of((work / plan["round"] / "exam.jsonl.zst").read_bytes()))
    rep.update(round=plan["round"], exam_digest=plan["exam_digest"], items=len(sat), items_digest=digest(sat), dropped=whole - len(sat), left_out=left_out,
               trainings={s: stats[s]["trainings"] for s in stats}, unedited={s: stats[s]["unedited"] for s in stats})
    _write(work / plan["round"] / "audit-report.json", rep)
    say_audit(rep)


def _audit_over(rounds: list[str], work: Path) -> None:
    """Several rounds' audit reports as one. A round has a handful of curators, and the rank
    agreement of a handful says little: over the rounds, the score's and each pillar's spread
    over noise is pooled, and how far the order carried between sizes is averaged."""
    from ..client import table

    reports, missing = [], []
    for rid in rounds:
        path = work / rid / "audit-report.json"
        (reports.append(_read(path)) if path.exists() else missing.append(rid))
    if missing:
        say(f"no audit report yet for: {', '.join(missing)}")
    if not reports:
        raise Stop("no audit report to pool")
    pooled = measure.pool(reports)
    pooled["over"] = [r["round"] for r in reports]
    _write(work / "audit-over.json", pooled)
    say(f"rounds audited, {len(reports)}: {', '.join(pooled['over'])}")
    rows = []
    for m, row in pooled["measures"].items():
        sizes = " · ".join(f"{s} {_ratio(st['spread_over_noise'])}" for s, st in row["sizes"].items())
        carried = " · ".join(f"{p.replace('>', ' to ')} {_rho(c['mean'])} ({c['rounds']})" for p, c in row["carried"].items())
        rows.append([_measure_name(m), str(row["rounds"]), sizes, carried])
    say(table(rows, ["", "rounds", "spread over noise", "order carried (rounds)"], right=(1,)))


def cmd_audit(args):
    from ..client import need_key

    season, work = load_season(args.season), root_of(args)
    if args.stage != "report" or not args.over:
        if not args.round:
            raise Stop(f"name the round: stet audit {args.stage} <round>")
    key = need_key() if not (args.stage == "report" and (args.over or args.offline)) else None
    {"plan": _audit_plan, "scripts": _audit_scripts, "farm": _audit_farm, "report": _audit_report}[args.stage](args, season, work, key)


# --- stet finals -----------------------------------------------------------------------------------


def fetch_beacon(season, number: int) -> str:
    """One round of the public beacon, from drand itself, checked as a round's is: the
    randomness is the hash of the signature."""
    import urllib.request

    from ..data import USER_AGENT

    cfg = season.raw["beacon"]
    if season.beacon_time(number) > time.time():
        raise Stop(f"drand round {number} is emitted at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(season.beacon_time(number)))}: the lot is drawn after that")
    last: Exception | None = None
    for relay in cfg["relays"]:
        try:
            req = urllib.request.Request(f"{relay}/{cfg['chain_hash']}/public/{number}", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r:
                got = json.loads(r.read(1 << 16))
            randomness = str(got["randomness"]).lower()
            if int(got["round"]) != number or _sha(bytes.fromhex(got["signature"])) != randomness:
                raise ValueError("the beacon it returned does not add up")
            return randomness
        except Exception as e:  # this relay has told us nothing; try the next
            last = e
    raise Stop(f"no drand relay gave a readable answer about round {number} ({last}); try again, or give the value: --lot-beacon {number}:<randomness>")


def finals_dir(work: Path, season, rehearsal: bool = False) -> Path:
    """Where a season's finals are kept. A rehearsal is kept apart, and cannot be taken for them."""
    return work / (f"finals-{season.id}" + ("-rehearsal" if rehearsal else ""))


def _freeze(args, season) -> dict:
    """The board as it stands, and who it sends to the finals."""
    board = ask("GET", "/api/v1/leaderboard").get("editors") or []
    # Whose curators are the organisers' own is said on each row of the board. It is not worked out
    # from the list of verifiers: that list is cut at 200, most work first, and the organisers'
    # accounts are counted as having done none, so with enough volunteers they are not on it.
    organisers = set(args.organisers or [])
    if all(isinstance(r.get("organisers"), bool) for r in board):
        organisers |= {r["owner"] for r in board if r["organisers"]}
    elif not organisers:
        raise Stop("the server's board does not say which curators are the organisers' own, so they cannot be told from entrants': "
                   "name the organisers' accounts with --organisers")
    rounds = ask("GET", "/api/v1/rounds?limit=30").get("rounds") or []
    # The newest crawl anyone could have seen by now: the one the server has adopted for its next
    # draw, or the newest a round was drawn from. Crawls are named by year and week, so the
    # greatest name is the newest.
    known = [((ask("GET", "/api/v1/season").get("crawl") or {}).get("crawl") or "")] + [r.get("crawl_id") or "" for r in rounds]
    try:
        chosen = finals.choose(board, organisers)
    except ValueError as e:  # a board without what the finalists are taken by
        raise Stop(str(e)) from None
    if len(chosen["finalists"]) < 2:
        raise Stop("fewer than two entrants are ranked: there is nobody to hold finals between")
    scale = season.scale(args.scale or finals.SCALE).name
    if scale != finals.SCALE and not args.rehearsal:
        raise Stop(f"the finals are decided at {finals.SCALE}: another size is for a rehearsal (--rehearsal)")
    return {"season": season.id, "frozen_at": int(time.time()), "frozen_round": next((r.get("id") for r in rounds if r.get("status") == "settled"), None),
            "crawl": max(known),
            "organisers": sorted(organisers), "board": [finals.slim(r) for r in board], **chosen, "scale": scale, "trainings": finals.TRAININGS,
            "baselines": finals.BASELINES, "multiples": list(finals.MULTIPLES), "lot": None, "draws": [], "rounds": [], "own": {}, "rehearsal": bool(args.rehearsal)}


def who_runs(plan: dict) -> list[dict]:
    """Every curator the finals run, with its role: the finalists, the lot's pick, the yardsticks."""
    out = [{**r, "role": finals.FINALIST} for r in plan["finalists"]]
    if plan.get("lot") and plan["lot"].get("pick"):
        out.append({**plan["lot"]["pick"], "role": finals.LOT})
    out += [{**r, "role": finals.YARDSTICK} for r in plan["yardsticks"]]
    return [{**r, "label": label_of(r), "name": name_of(r)} for r in out]


def _finals_plan(args, season, work: Path, key) -> None:
    from .. import client
    from ..data import FetchError

    path = finals_dir(work, season, args.rehearsal) / "plan.json"
    # Who is in the finals is decided once, by the first run: the board moves on, the finals do not.
    plan = _read(path) if path.exists() else _freeze(args, season)
    if args.draws:
        for b in args.draws:
            if season.draw_slot(b) is None:
                raise Stop(f"drand round {b} is not on the season's draw grid: no round can be drawn from it")
            if season.beacon_time(b) <= plan["frozen_at"] and not plan["rehearsal"]:
                raise Stop(f"drand round {b} was emitted before the finalists were frozen: the finals' draws are announced beforehand")
        plan["draws"] = sorted(args.draws)
    if args.lot_beacon:
        number, _, given = args.lot_beacon.partition(":")
        number = int(number)
        if plan.get("lot") and plan["lot"]["beacon"]["round"] != number and not plan["rehearsal"]:
            raise Stop(f"the lot was drawn from drand round {plan['lot']['beacon']['round']}: it is not drawn again from another")
        if season.beacon_time(number) <= plan["frozen_at"] and not plan["rehearsal"]:
            raise Stop(f"drand round {number} was emitted before the finalists were frozen: the lot's beacon round is named beforehand")
        beacon = {"round": number, "randomness": given.lower() or fetch_beacon(season, number)}
        checked = True
        if given:
            try:
                client.verify_beacon(season, beacon)  # a value typed in is checked against drand, as a round's is
            except FetchError:
                checked = False  # drand could not be asked; the plan says so, and the value is anyone's to check
            except ValueError as e:
                raise Stop(str(e)) from None
        plan["lot"] = {"beacon": beacon, "checked": checked, "pick": finals.draw_lot(season.id, plan["pool"], beacon)}
    if args.rounds:
        if len(args.rounds) != finals.ROUNDS and not plan["rehearsal"]:
            raise Stop(f"the finals are {finals.ROUNDS} rounds")
        own = {}
        for rid in args.rounds:
            rnd, d = settled_round(season, rid)
            if rnd.id in own:  # however it is spelt: two of one round would pass for the finals' two, and be one
                raise Stop(f"{rnd.id} is named twice: the finals' rounds are different rounds")
            own[rnd.id] = own_scale(season, d).name
            if plan["rehearsal"]:
                continue
            if rnd.beacon["round"] not in plan["draws"]:
                raise Stop(f"{rid} was drawn from drand round {rnd.beacon['round']}, which was not announced for the finals (stet finals plan --draws ...)")
            if not rnd.crawl["crawl"] > plan["crawl"]:
                raise Stop(f"{rid} is drawn from {rnd.crawl['crawl']}, which is no newer than the newest crawl there was when the finalists were frozen "
                           f"({plan['crawl']}): the finals wait for a crawl published after that")
        plan["rounds"], plan["own"] = list(own), own
    _write(path, plan)
    say_finals_plan(plan, season, args.speed)


def say_finals_plan(plan: dict, season, speed: float = 1.0) -> None:
    say(f"{season.title} finals" + (" · a rehearsal" if plan["rehearsal"] else "") + f" · the board as it stood at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(plan['frozen_at']))}"
        + (f", after {plan['frozen_round']}" if plan.get("frozen_round") else ""))
    say("finalists:")
    for i, r in enumerate(plan["finalists"]):
        say(f"  {i + 1}. {name_of(r)}  (rank {r.get('rank')}, {r.get('rounds')} rounds" + (f", at least {r['lower']:+.4f}" if r.get("lower") is not None else "") + ")")
    say(f"the lot is drawn among {len(plan['pool'])}: " + (", ".join(name_of(r) for r in plan["pool"]) or "nobody ranked 9th to 16th has run five rounds"))
    lot = plan.get("lot")
    if lot:
        say(f"  drand round {lot['beacon']['round']} ({lot['beacon']['randomness'][:16]}…" + ("" if lot.get("checked") else ", not checked against drand")
            + f") picks: {name_of(lot['pick']) if lot['pick'] else 'nobody'}")
    elif plan["pool"]:
        say("  not drawn yet: name a beacon round that has not been emitted, announce it, and then: stet finals plan --lot-beacon <round>")
    say("yardsticks, run beside them and given no place: " + (", ".join(name_of(r) for r in plan["yardsticks"]) or "none"))
    if plan["draws"]:
        say("the finals' rounds are drawn from drand rounds " + ", ".join(f"{b} ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(season.beacon_time(b)))})" for b in plan["draws"]))
    if plan["rounds"]:
        say(f"rounds: {', '.join(plan['rounds'])}")
    scale = season.scale(plan["scale"])
    n = len(plan["finalists"]) + bool(plan["pool"]) + len(plan["yardsticks"])
    held = len(plan["rounds"]) or finals.ROUNDS
    rows = [(scale, held * (n * plan["trainings"] + plan["baselines"]))] + [(measure.at_multiple(scale, m), held) for m in plan["multiples"]]
    say(f"{held} rounds at {scale.name}: {n} curators, each script trained {plan['trainings']} times, the unedited corpus {plan['baselines']}, and its curve once:")
    say_cost(season, cost(season, rows), speed)
    say(f"Running {n} curators on {held} {scale.name} corpora takes up to {held * n * scale.edit_seconds / 3600:.1f} hours on this machine "
        f"({scale.edit_seconds}s each: stet finals scripts), and building each corpus comes first.")
    say_noise(season, scale.name, held * plan["trainings"])


def _finals(args, work: Path, season) -> tuple[dict, Path]:
    """The finals' plan, once it is whole, and where the finals are kept."""
    fdir = finals_dir(work, season, args.rehearsal)
    again = "stet finals plan" + (" --rehearsal" if args.rehearsal else "")
    if not (fdir / "plan.json").exists():
        raise Stop(f"the finals are not planned: {again}")
    plan = _read(fdir / "plan.json")
    if not plan["rounds"]:
        raise Stop(f"the finals' rounds are not named yet: {again} --rounds <round> <round>")
    if plan["pool"] and not plan.get("lot"):
        raise Stop(f"the lot has not been drawn: {again} --lot-beacon <round>")
    return plan, fdir


def _finals_scripts(args, season, work: Path, key) -> None:
    plan, _ = _finals(args, work, season)
    scale = season.scale(plan["scale"])
    for rid in plan["rounds"]:
        rnd, _ = settled_round(season, rid)
        own = season.scale(plan["own"][rid])
        items, _ = whole_exam(rnd, work, args.items, args.shares)
        prepare(rnd, scale, work)
        if scale.name != own.name:
            check_exam(rnd, scale, items, work)
        made = [make_script(w, rnd, scale, work / rid / scale.name / "scripts", args.sandbox, args.time_limit, key, gpu=args.gpu) for w in who_runs(plan)]
        say(f"{rid} · {scale.name}: {sum(1 for m in made if m['script'])} of {len(made)} curators wrote a script")
    say("Next: stet finals farm" + (" --rehearsal" if args.rehearsal else ""))


def _finals_batches(plan: dict, season, work: Path) -> list[dict]:
    """What the finals train, a round at a time, from what is on disk."""
    scale = season.scale(plan["scale"])
    out = []
    for rid in plan["rounds"]:
        kept = work / rid / "exam.jsonl.zst"
        if not kept.exists():
            raise Stop(f"{rid}: its exam is not here yet: run the scripts stage")
        items, shares = _items_of(kept.read_bytes()), _read(work / rid / "languages.json")
        sat = exam_at(rid, items, [scale.name], plan["own"][rid], work)
        info = prepared(rid, scale.name, work)
        students, files, who = {UNEDITED: plan["baselines"]}, {}, {}
        for w in who_runs(plan):
            # Always the script made here: the finals run every curator themselves, whatever the round scored.
            sha, path = script_at(w, scale.name, "", rid, work)
            who[w["name"]] = UNEDITED if sha == info["unedited"] else sha
            if sha and sha != info["unedited"]:
                students[sha], files[sha] = plan["trainings"], path
        points = [p for p in curve_points(season, scale, plan["multiples"], plan["baselines"], 1, info["tokens"]) if p["multiple"] != 1]
        for p in points:
            p["bdir"] = batch_dir(rid, p["scale"].name, sat, work)
        out.append({"round": rid, "sat": sat, "whole": len(items), "shares": shares, "scale": scale, "bdir": batch_dir(rid, scale.name, sat, work), "students": students,
                    "files": files, "who": who, "points": points})
    return out


def _finals_farm(args, season, work: Path, key) -> None:
    plan, _ = _finals(args, work, season)
    batches = []
    for b in _finals_batches(plan, season, work):
        rnd, _ = settled_round(season, b["round"])
        got = post(rnd, b["scale"], b["sat"], b["students"], b["files"], b["bdir"], key, lease=measure.hold_seconds(season, b["scale"]))
        say(f"{rnd.id} · {b['scale'].name}: {sum(b['students'].values())} students, sitting {len(b['sat']):,} items (batch {got['batch']})")
        batches.append((b["bdir"], b["students"]))
        for p in b["points"]:
            if p["passes"] > 1:
                say(f"{rnd.id}: {multiple_text(p['multiple'])} would read the corpus {p['passes']:.2f} times: it is trained, marked, and is no point of the curve")
            post(rnd, p["scale"], b["sat"], {UNEDITED: p["trainings"]}, {}, p["bdir"], key, spec=p["spec"], lease=p["lease"])
            batches.append((p["bdir"], {UNEDITED: p["trainings"]}))
    if follow(batches, key, wait=not args.no_wait):
        say("Next: stet finals table" + (" --rehearsal" if args.rehearsal else ""))


def _place(p) -> str:
    return "" if p is None else str(p)


def say_finals_table(out: dict) -> None:
    from ..client import table
    from ..score import format_multiplier

    say(f"{out['title']} · {out['scale']} · {', '.join(out['rounds'])}" + (" · a rehearsal, not the finals" if out.get("rehearsal") else ""))
    rows = []
    for r in out["rows"]:
        role = {finals.LOT: "drawn by lot", finals.YARDSTICK: "yardstick"}.get(r["role"], "")
        m = format_multiplier((r["multiplier"], r["capped"])) if r["multiplier"] is not None else ""
        rows.append([_place(r["place"]) + ("=" if r["shares_place_with"] else ""), r["name"], gain_text(r["gain"]), "" if r["se"] is None else f"{r['se']:.4f}", m,
                     " ".join(gain_text(g) for g in r["rounds"]), role + ("; " if role and r.get("note") else "") + (r.get("note") or "")])
    say(table(rows, ["place", "curator", "gain", "std error", "worth", "by round", ""], right=(2, 3, 4)))
    if out["se_difference"] is not None:
        say(f"Two curators whose gains differ by less than {out['tie'] * out['se_difference']:.4f} ({out['tie']:g} standard errors of a difference) share a place.")
    lot = out.get("lot")
    if lot:
        if lot["ahead_of"]:
            say(f"The lot's pick, {lot['name']}, placed {lot['place']}, ahead of {len(lot['ahead_of'])} of the finalists the rounds chose: {', '.join(lot['ahead_of'])}.")
        else:
            say(f"The lot's pick, {lot['name']}, placed {lot['place']}, ahead of no finalist" + (f" and level with {', '.join(lot['level_with'])}" if lot["level_with"] else "") + ".")
    if out["curve"] is None:
        say("No multiplier: the unedited curve's students are not back.")
    elif not out["curve_rises"]:
        say("More data did not always do better on the finals' curve: the multipliers here cannot be trusted.")
    for n in out.get("curve_notes", []):
        say(f"  {n}")
    if out.get("missing"):
        say("This table is not final. Still to come:")
        for m in out["missing"]:
            say(f"  {m}")
        if out.get("waiting"):
            say(f"  and not in the table at all until a student of theirs is back: {', '.join(out['waiting'])}")


def _finals_table(args, season, work: Path, key) -> None:
    from ..client import ApiError
    from .examine import public_items

    plan, fdir = _finals(args, work, season)
    runs = who_runs(plan)
    rounds, missing, notes, waiting = [], [], {}, set()
    for b in _finals_batches(plan, season, work):
        if key is not None:
            try:
                for d in [b["bdir"]] + [p["bdir"] for p in b["points"]]:
                    if (d / "batch.json").exists():
                        collect(d, key)
            except ApiError as e:
                say(f"the server could not be asked for what is new ({e}); this is what was collected before")
        unedited = scores(b["bdir"], UNEDITED)
        if not unedited:
            raise Stop(f"{b['round']}: no student of the unedited corpus is back yet: there is nothing to measure from")
        if len(unedited) < plan["baselines"]:
            missing.append(f"{b['round']}: {plan['baselines'] - len(unedited)} of {plan['baselines']} trainings of the unedited corpus")
        students = {}
        for w in runs:
            name, sha = w["name"], b["who"][w["name"]]
            if sha == UNEDITED:
                students[name] = unedited  # a script that keeps the corpus as it is: the unedited student
            elif sha is None:
                notes.setdefault(name, []).append(f"no script in {b['round']}: 0 there")
            elif refused(b["bdir"], sha):
                notes.setdefault(name, []).append(f"its script in {b['round']} breaks a rule ({refused(b['bdir'], sha)}): 0 there")
            else:
                students[name] = scores(b["bdir"], sha)
                if len(students[name]) < plan["trainings"]:
                    missing.append(f"{b['round']}: {plan['trainings'] - len(students[name])} of {plan['trainings']} trainings of {name}")
                if not students[name]:
                    waiting.add(name)  # no student of its script is back: that is no gain yet, and not a gain of 0
        curve = {}
        for p in b["points"]:
            got = scores(p["bdir"], UNEDITED)
            if not got:
                missing.append(f"{b['round']}: the unedited corpus at {multiple_text(p['multiple'])}")
            elif read_twice(p["passes"], p["bdir"]):
                notes.setdefault("curve", []).append(f"{b['round']}: the student at {multiple_text(p['multiple'])} read some of the corpus twice, and is no point of the curve")
            else:
                curve[p["multiple"]] = got
        from .aggregate import plan_of

        rounds.append({"round": b["round"], "table": item_table(b["sat"], plan=plan_of(season)), "shares": b["shares"], "unedited": unedited, "students": students, "curve": curve})
        # What is published of a round: its exam as a settled round's is, and how every student did on every item.
        pub = fdir / "published" / b["round"]
        _write(pub / "scores.json", {"round": b["round"], "scale": plan["scale"], "items": [it.id for it in b["sat"]], "left_out": b["whole"] - len(b["sat"]),
                                     "languages": b["shares"], "unedited": unedited, "students": students, "scripts": {w["name"]: b["who"][w["name"]] for w in runs},
                                     "curve": {multiple_text(m): v for m, v in curve.items()}})
        (pub / "exam.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_items(b["sat"])))
    out = finals.table(rounds, {w["name"]: w["role"] for w in runs if w["name"] not in waiting})
    for r in out["rows"]:
        if notes.get(r["name"]):
            r["note"] = "; ".join(notes[r["name"]])
    out.update(title=f"{season.title} finals", season=season.id, scale=plan["scale"], rounds=plan["rounds"], rehearsal=plan["rehearsal"], complete=not missing,
               missing=missing, waiting=sorted(waiting), curve_notes=notes.get("curve", []), frozen_at=plan["frozen_at"], lot_beacon=(plan.get("lot") or {}).get("beacon"))
    _write(fdir / "table.json", out)
    shutil.copyfile(fdir / "plan.json", fdir / "published" / "plan.json")
    _write(fdir / "published" / "table.json", out)
    say_finals_table(out)
    say(f"To publish: {fdir / 'published'} (the plan, the table, each round's exam as a settled round's is published, and every student's score on every "
        f"item), and the scripts themselves: {work}/<round>/{plan['scale']}/scripts.")


def cmd_finals(args):
    from ..client import need_key

    season, work = load_season(args.season), root_of(args)
    key = None if (args.stage == "table" and args.offline) else need_key()
    {"plan": _finals_plan, "scripts": _finals_scripts, "farm": _finals_farm, "table": _finals_table}[args.stage](args, season, work, key)


# --- the commands ----------------------------------------------------------------------------------


def _stopping(fn):
    def run(args):
        try:
            return fn(args)
        except Stop as e:
            sys.stderr.write(f"stet: {e}\n")
            sys.exit(1)

    return run


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x]


def _names(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def add_commands(sub) -> None:
    """The organisers' commands: not listed in the help, like `stet examine`."""

    def common(p, scripts: bool = False):
        p.add_argument("--out", help="where everything is kept (default: ~/stet-measurements, or STET_MEASUREMENTS)")
        p.add_argument("--season", default=CURRENT_SEASON, help=argparse.SUPPRESS)
        p.add_argument("--speed", type=float, default=1.0, help="show GPU hours for a card this many times the season's floor (default: the floor)")
        p.add_argument("--items", help="the round's whole exam, where this machine did not examine it (JSON lines, or zstd)")
        p.add_argument("--shares", help="with --items: the languages of the round's corpus (the examiner's languages.json)")
        p.add_argument("--no-wait", action="store_true", help="post the students and take what is back, without following them")
        if scripts:
            p.add_argument("--sandbox", choices=["bwrap", "container"], default="bwrap", help="scripts: what the curators run in")
            p.add_argument("--time-limit", type=float, default=None, help="scripts: seconds a curator is given (default: the size's own limit)")
            p.add_argument("--gpu", type=int, default=0, help="scripts: the GPU the curators are shown (default: 0)")
            p.add_argument("--offline", action="store_true", help="the last stage: add up what was collected, without asking the server for more")

    p = sub.add_parser("curve", help=argparse.SUPPRESS)  # organisers: the unedited corpus at multiples of the budget
    p.add_argument("rounds", nargs="+", help="settled official rounds")
    p.add_argument("--scale", help="the size of student (default: the round's own)")
    p.add_argument("--multiples", type=_floats, default=[0.5, 1.0, 2.0, 4.0], help="multiples of the training budget (default: 0.5,1,2,4)")
    p.add_argument("--repeats", type=int, default=2, help="trainings at 1x, which every point is measured from (default: 2)")
    p.add_argument("--repeats-elsewhere", type=int, default=1, help="trainings at each other multiple (default: 1)")
    p.add_argument("--plan", action="store_true", help="say what would be trained and what it costs, and post nothing")
    p.add_argument("--average", action="store_true", help="the mean of the curves already measured for these rounds")
    common(p)
    p.set_defaults(fn=_stopping(cmd_curve))

    p = sub.add_parser("audit", help=argparse.SUPPRESS)  # organisers: a settled round trained again one size up
    p.add_argument("stage", choices=["plan", "scripts", "farm", "report"])
    p.add_argument("round", nargs="?", help="a settled official round")
    p.add_argument("--sizes", type=_names, help="plan: the sizes of student, the round's own first (default: it and the next one up)")
    p.add_argument("--trainings", type=int, default=2, help="plan: trainings of each script at each size (default: 2)")
    p.add_argument("--baselines", type=int, default=farm.BASELINES, help=f"plan: trainings of the unedited corpus at each size (default: {farm.BASELINES})")
    p.add_argument("--entries", type=_names, help="plan: only these curators (owner/name@vN), instead of every entry of the round")
    p.add_argument("--over", nargs="+", metavar="ROUND", help="report: the reports of these rounds as one, for the score and each pillar")
    common(p, scripts=True)
    p.set_defaults(fn=_stopping(cmd_audit))

    p = sub.add_parser("finals", help=argparse.SUPPRESS)  # organisers: the season's finals
    p.add_argument("stage", choices=["plan", "scripts", "farm", "table"])
    p.add_argument("--organisers", type=_names, help="plan: the organisers' handles, whose curators are yardsticks (as well as those the server's board marks as theirs)")
    p.add_argument("--lot-beacon", metavar="ROUND[:RANDOMNESS]", help="plan: draw the lot from this drand round (its value is fetched from drand unless given)")
    p.add_argument("--draws", type=int, nargs="+", metavar="DRAND_ROUND", help="plan: the drand rounds announced for the finals' draws")
    p.add_argument("--rounds", nargs="+", metavar="ROUND", help="plan: the finals' rounds, once they have settled")
    p.add_argument("--rehearsal", action="store_true", help="every stage: a rehearsal on any settled rounds, kept apart and marked as one in everything it writes")
    p.add_argument("--scale", help="plan, with --rehearsal: the size of student to rehearse at (the finals' own is large)")
    common(p, scripts=True)
    p.set_defaults(fn=_stopping(cmd_finals))
