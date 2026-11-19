"""Students trained for the organisers on their own machines: finals, and measurements.

A season's finals train the top curators' students at a size larger than rounds use, and the
exam's calibration trains students again and again. The organisers' machine posts them as a
batch: the items to sit, and the scripts whose students it wants, at one of the season's
scales or one built for the purpose (`post_batch`). Every organisers' machine running `stet
verify`, rented ones included, takes one when it has no official work, trains it, sits it,
and sends back how it did on each item (`work_one`). The organisers' machine collects the
results and adds them up (`farm_round`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from .aggregate import gains, round_score
from .items import Item, digest

UNEDITED = "unedited"
POLL = 60  # seconds between looks at a batch
BASELINES = 4  # trainings of the unedited corpus averaged into a round's reference: the least steady student of all


def pack_items(items: list[Item]) -> bytes:
    import zstandard

    raw = "".join(json.dumps(it.to_dict(), ensure_ascii=False) + "\n" for it in items).encode()
    return zstandard.ZstdCompressor(level=10).compress(raw)


def unpack_items(blob: bytes, limit: int = 256 << 20) -> list[Item]:
    import zstandard

    with zstandard.ZstdDecompressor().stream_reader(blob) as r:
        raw = r.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("a batch's items are too large")
    return [Item.from_dict(json.loads(line)) for line in raw.decode().splitlines() if line.strip()]


# --- a machine of the organisers', training one ---------------------------------------------------


def scale_of(season, name: str, spec: dict | None):
    """The scale a task names: one of the season's, or one built from its spec (the scale
    check and finals), starting from the season's standard scale."""
    if not spec:
        return season.scale(name)
    base = season.scale("standard")
    fields = {f.name for f in dataclasses.fields(base)} - {"name"}
    unknown = set(spec) - fields
    if unknown:
        raise ValueError(f"a scale this harness does not know how to build: {sorted(unknown)}")
    return dataclasses.replace(base, name=name, **{k: type(getattr(base, k))(v) for k, v in spec.items()})


def _fetch_script(sha: str, season, dest: Path, key=None) -> Path:
    import zstandard

    from ..client import ApiError, request

    limit = int(season.raw["script"]["max_bytes"])
    try:
        blob = request("GET", f"/api/v1/scripts/{sha}", raw=True, timeout=600, limit=limit + (1 << 20))
    except ApiError as e:
        if e.status not in (403, 404):
            raise
        # No round's entry: a reference curator's, or an entry's at a scale of its own.
        blob = request("GET", f"/api/v1/farm/scripts/{sha}", key=key, raw=True, timeout=600, limit=limit + (1 << 20))
    with zstandard.ZstdDecompressor().stream_reader(blob) as r:
        raw = r.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"script {sha[:12]} does not match its hash")
    dest.write_bytes(raw)
    return dest


def _send(where: str, body: dict, key, waits=(5, 15, 30, 60, 120)) -> None:
    """Send a student's result, waiting out a server that is busy for a moment: the student took
    minutes to train, and is not thrown away for want of one answer."""
    from ..client import ApiError, request

    for wait in (*waits, None):
        try:
            request("POST", where, body, key=key, timeout=300)
            return
        except ApiError as e:
            if wait is None or not (e.status == 0 or e.status == 429 or e.status >= 500):
                raise
        time.sleep(wait)


FETCH_WAITS = (60, 120, 240, 480, 600)  # a crawl host that says "slow down" is waited out, not given up on


def _corpus(data, rnd, scale, log):
    """The round's corpus, waiting out a public data host that is busy: a student held is not
    handed back (and a try spent) because Common Crawl asked a dozen machines to slow down."""
    for wait in (*FETCH_WAITS, None):
        try:
            return data.corpus(rnd, scale)
        except data.FetchError as e:
            if wait is None:
                raise
            log(f"{rnd.id}: {e}; trying again in {wait // 60} min")
            time.sleep(wait)


def work_one(task: dict, season, key, device, log=print) -> None:
    """Train and sit the student a task names, and send back how it did."""
    from .. import data
    from ..client import request, round_from_api
    from ..runner import release_gpu_memory
    from ..tok import load_encoding
    from .score import sit

    where = f"/api/v1/farm/batches/{task['batch']}/tasks/{int(task['index'])}"
    try:
        rnd = round_from_api(season, request("GET", f"/api/v1/rounds/{task['round']}")["round"])
        scale = scale_of(season, task["scale"], task.get("scale_spec"))
        script_ref = task["script"].split("#")[0]  # "#2": the same student, trained again
        items = unpack_items(request("GET", f"/api/v1/farm/batches/{task['batch']}/items", key=key, raw=True, timeout=600, limit=64 << 20))
        if len(items) != task["items"] or digest(items) != task["items_digest"]:
            raise ValueError("the batch's items do not match their digest")
        corpus = _corpus(data, rnd, scale, log)
        enc = load_encoding(season)
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="stet-farm-") as tmp:
            script = Path(tmp) / "script.jsonl"
            if script_ref == UNEDITED:
                unedited_script(script, len(corpus))
            else:
                _fetch_script(script_ref, season, script, key=key)
            meta, model = train_student(rnd, scale, corpus, enc, script, device)
        body = dict(meta)
        body["reason"] = body.pop("invalid_reason", "")
        if model is not None:
            body["item_scores"] = [round(s, 6) for s in sit(model, items, enc, season.eot, scale.ctx, device)]
            del model
        release_gpu_memory()
        try:
            from ..train import hardware_info

            body["hardware"] = str((hardware_info(device) or {}).get("gpu") or device)[:200]
        except Exception:
            body["hardware"] = str(device)
        _send(where, body, key)
        said = "trained" if meta["valid"] else f"not a script: {meta.get('invalid_reason')}"
        log(f"{task['round']} · {task['script'][:12]}: {said}, {len(items)} items sat [{time.time() - t0:.0f}s]")
    except BaseException:
        try:
            request("POST", where + "/release", {}, key=key)
        except Exception:
            pass
        raise


def turn(season, key, device, log=print) -> bool:
    """Ask for a student to train, and train it. False when there is none."""
    from ..client import request

    # A machine set aside for bigger students says how long it will give one; any other is
    # handed only those that take less than an hour.
    ask = {}
    if os.environ.get("STET_FARM_MAX_MINUTES", "").isdigit():
        ask["max_minutes"] = int(os.environ["STET_FARM_MAX_MINUTES"])
    task = (request("POST", "/api/v1/farm/claim", ask, key=key) or {}).get("task")
    if not isinstance(task, dict):
        return False
    work_one(task, season, key, device, log=log)
    return True


# --- the organisers' machine, farming a round out --------------------------------------------------


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def unedited_script(path: Path, n_docs: int) -> str:
    """Write the script that keeps the corpus as it is; returns its sha256."""
    from ..script import write_script

    write_script(path, range(n_docs))
    return _sha(path)


def train_student(rnd, scale, corpus, enc, script: Path, device):
    """Train the student on what a script keeps: (what it says, the model, or None when the
    script is not one)."""
    from ..script import ScriptError, materialize
    from ..train import train

    meta: dict = {"valid": False}
    try:
        tokens, stats = materialize(corpus, script, rnd.season, scale, enc)
    except ScriptError as e:
        meta["invalid_reason"] = str(e)
        return meta, None
    model, tr = train(tokens, scale, rnd.season, rnd.init_seed, device)
    meta.update(valid=True, kept_doc_frac=stats.to_dict(corpus)["kept_doc_frac"], train_seconds=round(tr["seconds"], 1))
    return meta, model


def write_items(items: list[Item], path: Path) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def summarize(rnd, scale, items: list[Item], base: dict, students: dict[str, dict], shares: dict[str, float] | None = None) -> dict:
    """What a batch came to: each script's score on the items over the unedited student's."""
    from . import gen

    rows = []
    for name, r in students.items():
        row = {"name": name, "valid": r["valid"]}
        if r["valid"]:
            agg = round_score(items, gains(r["item_scores"], base["item_scores"]), shares=shares)
            row.update(score=round(agg["score"], 5), pillars={k: v["score"] for k, v in agg["pillars"].items()}, families=agg["families"])
        else:
            row["reason"] = r.get("invalid_reason")
        rows.append(row)
    return {"round": rnd.id, "scale": scale.name, "items": len(items), "items_digest": digest(items), "generator_version": gen.VERSION,
            "entries": rows}


def post_batch(rnd, scale, items: list[Item], scripts: list[str], key, spec: dict | None = None, lease: int = 0) -> str:
    import base64

    from ..client import request
    from .gen import VERSION

    body = {"round": rnd.index, "scale": scale.name, "items_digest": digest(items), "items": len(items), "generator_version": VERSION,
            "scripts": scripts, "items_file": base64.b64encode(pack_items(items)).decode()}
    if spec:
        body["scale_spec"], body["lease"] = spec, lease
    return request("POST", "/api/v1/admin/farm/batches", body, key=key, timeout=600)["batch"]


def farm_round(rnd, scale, scripts: dict[str, str], out_dir: Path, items: list[Item], n_docs: int, key,
               shares: dict[str, float] | None = None, log=print, patience: float = 12 * 3600, baselines: int = BASELINES,
               spec: dict | None = None, lease: int = 0) -> dict:
    """Have the organisers' machines train a round's students (scripts: name -> sha256), keep
    what they send in out_dir as a local run would, and add it up.

    The unedited corpus is trained `baselines` times and its scores averaged: its student
    varies between trainings several times as much as a curated corpus's does, and every
    gain is measured from it. Each training is kept beside the average (`<sha>#k`)."""
    from ..client import request

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = digest(items)[:12]
    write_items(items, out_dir / "items.jsonl")
    if shares:
        _write(out_dir / "languages.json", shares)
    with tempfile.TemporaryDirectory(prefix="stet-farm-") as tmp:
        base_sha = unedited_script(Path(tmp) / "unedited.jsonl", n_docs)
    # An entry whose script keeps the corpus as it is (identity) is the unedited corpus's student.
    scripts = {name: (UNEDITED if sha == base_sha else sha) for name, sha in scripts.items()}
    names = {sha: name for name, sha in scripts.items() if sha != UNEDITED}
    names[UNEDITED] = "(unedited)"
    local = {UNEDITED: base_sha, **{sha: sha for sha in scripts.values() if sha != UNEDITED}}
    repeats = [UNEDITED] + [f"{UNEDITED}#{k}" for k in range(2, max(1, baselines) + 1)]
    batch = post_batch(rnd, scale, items, [*repeats, *sorted(set(scripts.values()) - {UNEDITED})], key, spec=spec, lease=lease)
    log(f"{rnd.id}: {len(names)} students posted for the organisers' machines to train (batch {batch})")
    t0, said = time.time(), None
    while True:
        b = request("GET", f"/api/v1/admin/farm/batches/{batch}", key=key, timeout=300, limit=256 << 20)
        tasks = b["tasks"]
        for t in tasks:
            d = t.get("done")
            if not d or t["script"].split("#")[0] not in local:  # not done, or a task this run did not ask for
                continue
            sha16 = local[t["script"].split("#")[0]][:16]
            if t["script"].startswith(UNEDITED):  # one of the reference's trainings, kept apart
                sha16 += "#" + (t["script"].split("#")[1] if "#" in t["script"] else "1")
            meta = {"name": names[t["script"].split("#")[0]], "valid": bool(d["valid"]), "farmed_by": d.get("by"), "hardware": d.get("hardware")}
            if d["valid"]:
                meta.update(kept_doc_frac=d.get("kept_doc_frac"), train_seconds=d.get("train_seconds"))
                _write(out_dir / f"{sha16}-{tag}.json", t["item_scores"])
            else:
                meta["invalid_reason"] = d.get("reason")
            _write(out_dir / f"{sha16}.json", meta)
        done = sum(1 for t in tasks if t.get("done"))
        # Still to come: not done, and either to be tried again or being tried now.
        left = [t for t in tasks if not t.get("done") and (t.get("tries", 0) < 3 or (t.get("holder") and t.get("until", 0) > time.time()))]
        now = f"{done} of {len(tasks)} trained" + (f", {sum(1 for t in left if t.get('holder'))} under way" if left else "")
        if now != said:
            log(f"  {now} [{time.time() - t0:.0f}s]")
            said = now
        if not left or time.time() - t0 > patience:
            break
        time.sleep(POLL)

    def student(sha: str) -> dict | None:
        p = out_dir / f"{local[sha][:16]}.json"
        if not p.exists():
            return None
        meta = json.loads(p.read_text())
        s = out_dir / f"{local[sha][:16]}-{tag}.json"
        if meta["valid"]:
            meta["item_scores"] = json.loads(s.read_text())
        return meta

    trained = []
    for k in range(1, len(repeats) + 1):
        p = out_dir / f"{base_sha[:16]}#{k}.json"
        if p.exists() and (m := json.loads(p.read_text()))["valid"]:
            m["item_scores"] = json.loads((out_dir / f"{base_sha[:16]}#{k}-{tag}.json").read_text())
            trained.append(m)
    if not trained:
        raise RuntimeError(f"{rnd.id}: the unedited corpus's student was not trained, so nothing can be added up")
    n = len(trained)
    base = {"name": "(unedited)", "valid": True, "trainings": n, "hardware": sorted({str(m.get("hardware")) for m in trained}),
            "item_scores": [round(sum(col) / n, 6) for col in zip(*(m["item_scores"] for m in trained), strict=True)]}
    _write(out_dir / f"{base_sha[:16]}.json", {k: v for k, v in base.items() if k != "item_scores"})
    _write(out_dir / f"{base_sha[:16]}-{tag}.json", base["item_scores"])
    log(f"  the unedited corpus: {n} trainings averaged")
    got = {name: m for name, sha in scripts.items() if (m := student(sha)) is not None}
    for name, sha in scripts.items():
        if sha == UNEDITED:
            got[name] = {**base, "name": name}
    if len(got) < len(scripts):
        log(f"  {len(scripts) - len(got)} students were not trained: left out")
    summary = summarize(rnd, scale, items, base, got, shares)
    _write(out_dir / "summary.json", summary)
    return summary
