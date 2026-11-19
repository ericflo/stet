"""The next exam's students, farmed out to the organisers' machines.

A round's shadow needs one student per entry and one for the unedited corpus. The organisers'
machine posts them as a batch: the round's next-exam items, and the scripts whose students it
wants (`post_batch`). Every organisers' machine running `stet verify`, rented ones included,
takes one when it has no official work, trains it, sits it on both exams, and sends back how
it did on each item (`work_one`). The organisers' machine collects the results into the same
files a local run keeps, so `stet shadow` and tools/next_exam_report.py read them alike, and
adds them up (`farm_round`).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

from .items import Item, digest

UNEDITED = "unedited"
POLL = 60  # seconds between looks at a batch


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


def _fetch_script(sha: str, season, dest: Path) -> Path:
    import zstandard

    from ..client import request

    limit = int(season.raw["script"]["max_bytes"])
    blob = request("GET", f"/api/v1/scripts/{sha}", raw=True, timeout=600, limit=limit + (1 << 20))
    with zstandard.ZstdDecompressor().stream_reader(blob) as r:
        raw = r.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"script {sha[:12]} does not match its hash")
    dest.write_bytes(raw)
    return dest


def work_one(task: dict, season, key, device, log=print) -> None:
    """Train and sit the student a task names, and send back how it did."""
    from .. import data
    from ..client import request, round_from_api
    from ..runner import release_gpu_memory
    from ..tok import load_encoding
    from .score import sit
    from .shadow import train_student, unedited_script

    where = f"/api/v1/shadow/batches/{task['batch']}/tasks/{int(task['index'])}"
    try:
        rnd = round_from_api(season, request("GET", f"/api/v1/rounds/{task['round']}")["round"])
        scale = season.scale(task["scale"])
        items = unpack_items(request("GET", f"/api/v1/shadow/batches/{task['batch']}/items", key=key, raw=True, timeout=600, limit=64 << 20))
        if len(items) != task["items"] or digest(items) != task["items_digest"]:
            raise ValueError("the batch's items do not match their digest")
        corpus, exam = data.materialize(rnd, scale)
        enc = load_encoding(season)
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="stet-shadow-") as tmp:
            script = Path(tmp) / "script.jsonl"
            if task["script"] == UNEDITED:
                unedited_script(script, len(corpus))
            else:
                _fetch_script(task["script"], season, script)
            meta, model = train_student(rnd, scale, corpus, exam, enc, script, device)
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
        request("POST", where, body, key=key, timeout=300)
        said = f"season exam {meta['exam_loss']:.4f}" if meta["valid"] else f"not a script: {meta.get('invalid_reason')}"
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

    task = (request("POST", "/api/v1/shadow/claim", {}, key=key) or {}).get("task")
    if not isinstance(task, dict):
        return False
    work_one(task, season, key, device, log=log)
    return True


# --- the organisers' machine, farming a round out --------------------------------------------------


def post_batch(rnd, scale, items: list[Item], scripts: list[str], key) -> str:
    import base64

    from ..client import request
    from .gen import VERSION

    body = {"round": rnd.index, "scale": scale.name, "items_digest": digest(items), "items": len(items), "generator_version": VERSION,
            "scripts": scripts, "items_file": base64.b64encode(pack_items(items)).decode()}
    return request("POST", "/api/v1/admin/shadow/batches", body, key=key, timeout=600)["batch"]


def farm_round(rnd, scale, scripts: dict[str, str], out_dir: Path, items: list[Item], n_docs: int, key,
               shares: dict[str, float] | None = None, log=print, patience: float = 12 * 3600) -> dict:
    """Have the organisers' machines train a round's students (scripts: name -> sha256), keep
    what they send in out_dir as a local run would, and add it up."""
    from ..client import request
    from .shadow import _write, summarize, unedited_script, write_items

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = digest(items)[:12]
    write_items(items, out_dir / "items.jsonl")
    if shares:
        _write(out_dir / "languages.json", shares)
    with tempfile.TemporaryDirectory(prefix="stet-shadow-") as tmp:
        base_sha = unedited_script(Path(tmp) / "unedited.jsonl", n_docs)
    # An entry whose script keeps the corpus as it is (identity) is the unedited corpus's student.
    scripts = {name: (UNEDITED if sha == base_sha else sha) for name, sha in scripts.items()}
    names = {sha: name for name, sha in scripts.items() if sha != UNEDITED}
    names[UNEDITED] = "(unedited)"
    local = {UNEDITED: base_sha, **{sha: sha for sha in scripts.values() if sha != UNEDITED}}
    batch = post_batch(rnd, scale, items, [UNEDITED, *sorted(set(scripts.values()) - {UNEDITED})], key)
    log(f"{rnd.id}: {len(names)} students posted for the organisers' machines to train (batch {batch})")
    t0, said = time.time(), None
    while True:
        b = request("GET", f"/api/v1/admin/shadow/batches/{batch}", key=key, timeout=300, limit=256 << 20)
        tasks = b["tasks"]
        for t in tasks:
            d = t.get("done")
            if not d or t["script"] not in local:  # not done, or a task this run did not ask for
                continue
            sha16 = local[t["script"]][:16]
            meta = {"name": names[t["script"]], "valid": bool(d["valid"]), "farmed_by": d.get("by"), "hardware": d.get("hardware")}
            if d["valid"]:
                meta.update(exam_loss=d["exam_loss"], by_source=d.get("by_source") or {}, kept_doc_frac=d.get("kept_doc_frac"),
                            train_seconds=d.get("train_seconds"))
                _write(out_dir / f"{sha16}-{tag}.json", t["item_scores"])
            else:
                meta["invalid_reason"] = d.get("reason")
            _write(out_dir / f"{sha16}.json", meta)
        done = sum(1 for t in tasks if t.get("done"))
        left = [t for t in tasks if not t.get("done") and t.get("tries", 0) < 3]
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

    base = student(UNEDITED)
    if base is None or not base["valid"]:
        raise RuntimeError(f"{rnd.id}: the unedited corpus's student was not trained, so nothing can be added up")
    got = {name: m for name, sha in scripts.items() if (m := student(sha)) is not None}
    for name, sha in scripts.items():
        if sha == UNEDITED:
            got[name] = {**base, "name": name}
    if len(got) < len(scripts):
        log(f"  {len(scripts) - len(got)} students were not trained: left out")
    summary = summarize(rnd, scale, items, base, got, shares)
    _write(out_dir / "summary.json", summary)
    return summary
