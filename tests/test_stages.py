"""The organisers' stages from end to end (exam/stages.py): `stet curve`, `stet audit` and
`stet finals`, as their commands are typed.

Nothing here is real but the code under test. The server, with its board, its rounds and its
farm, is a class below; the machines of the farm are a function from a script to scores; the
corpora are a few made-up documents; and a curator's run writes a line saying who it was."""

import base64
import hashlib
import json
import re
import time
import types

import pytest
import zstandard
from conftest import FakeEnc, make_corpus

from stet import cli, client, data
from stet.exam import examine, farm, finals, measure, stages
from stet.exam.items import Item, digest
from stet.runner import EditRun
from stet.spec import Scale, Season, load_season, official_round

SEASON = load_season()
# Students small enough that a corpus of a few hundred words is several budgets long.
TINY = {
    "standard": Scale(name="standard", corpus_units=8, train_tokens=100, n_layer=2, n_head=2, d_model=64, ctx=16, batch_size=2, lr=0.001, edit_seconds=60),
    "large": Scale(name="large", corpus_units=32, train_tokens=250, n_layer=2, n_head=2, d_model=64, ctx=16, batch_size=2, lr=0.0008, edit_seconds=120),
}
QUOTED = "The river authority approved a new flood barrier for the lower town after three wet winters in a row."
CORPUS_TOKENS = {"standard": 400, "large": 1000}  # FakeEnc: a token a word, and one before each document


def _items() -> list[Item]:
    """An exam of eighteen: news written the day before the draw, and generated logic of two sorts."""
    news = [Item(kind="passage", corner=f"fresh/ccnews/en/-/{i}", fmt="prose", lang="en", source="ccnews:test",
                 text=QUOTED if i == 0 else f"Report number {i} of the day, which no crawl made before it could hold: the council met and adjourned {i} times over.")
            for i in range(6)]
    chains = [Item(kind="answer", corner=f"logic/chain/{i}", fmt="prose", lang="en", source="gen:logic", ctx=f"If {i} then", cont=f" {i + 1}.") for i in range(6)]
    orders = [Item(kind="answer", corner=f"logic/order/{i}", fmt="prose", lang="en", source="gen:logic", ctx=f"Q{i}", cont=f" A{i}.") for i in range(6)]
    return news + chains + orders


def _slot(t: int) -> tuple[int, int]:
    """The first draw of the season's grid after t: (the drand round it is drawn from, when)."""
    grid = int(SEASON.raw["draw_grid_seconds"])
    slot = (t // grid + 1) * grid
    return SEASON.beacon_round_at(slot), slot


def _bundle(owner: str) -> tuple[str, bytes]:
    blob = f"the bundle of {owner}".encode()
    return hashlib.sha256(blob).hexdigest(), blob


def _official_script(owner: str) -> tuple[str, bytes]:
    raw = (json.dumps({"curator": owner, "size": "standard"}) + "\n").encode()
    return hashlib.sha256(raw).hexdigest(), raw


class Server:
    """The server, as far as the stages see it, and behind it the farm's machines: asked about
    a batch, it has trained every student that nothing here holds back."""

    def __init__(self):
        self.rounds: dict[str, dict] = {}
        self.board: list[dict] = []
        self.scripts: dict[str, bytes] = {}  # the rounds' own, and those sent to the farm
        self.sent: list[str] = []  # scripts sent to the farm, in order
        self.batches: dict[str, dict] = {}
        self.posts: list[tuple[str, list[str]]] = []  # (scale, the students asked for) of every post
        self.gains: dict[str, dict[str, float]] = {}  # size -> curator -> its gain there
        self.curve: dict[str, float] = {"0p5": -0.04, "2": 0.035, "4": 0.06, "8": 0.07}  # the unedited corpus's gain at other budgets
        self.asleep: set[str] = set()  # students no machine has taken yet
        self.failing: set[str] = set()  # students three machines failed at
        self.forgetful = False
        self.adopted = "CC-MAIN-2026-45"  # the crawl its next draw will use: published, and no round drawn from it yet

    def add_round(self, index: int, items: list[Item], entries=(), at: int = 1_795_000_200, crawl: str = "CC-MAIN-2026-43") -> str:
        number, slot = _slot(at)
        rnd = official_round(SEASON, index, {"round": number, "randomness": f"{index:02x}" * 32}, {"crawl": crawl, "paths_sha256": "cd" * 32, "n_shards": 100_000})
        rows = []
        for owner, gain, valid in entries:
            sha, raw = _official_script(owner)
            self.scripts[sha] = raw
            rows.append({"round": rnd.id, "editor": _bundle(owner)[0], "name": "c", "version": 1, "owner": owner, "valid": valid, "gain": gain,
                         **({"script": {"sha256": sha}} if valid else {})})
        self.rounds[rnd.id] = {"round": {**rnd.describe(), "status": "settled", "scale": "standard", "draw_at": slot, "exam_digest": digest(items), "crawl_id": crawl},
                               "entries": rows}
        # The examiner kept the whole exam where it made it.
        made = data.cache_dir() / "examine" / rnd.id
        made.mkdir(parents=True, exist_ok=True)
        examine.keep({"items_file": base64.b64encode(farm.pack_items(items)).decode(), "languages": {"en": 100.0}}, made)
        return rnd.id

    def student(self, batch: dict, ref: str) -> tuple[dict, list[float] | None]:
        script, _, k = ref.partition("#")
        k = int(k or 1)
        size, _, multiple = batch["scale"].partition("-x")
        tokens = int(batch["spec"]["train_tokens"]) if batch["spec"] else TINY[size].train_tokens
        done = {"by": "rented-one", "hardware": "a made-up card", "valid": True, "kept_doc_frac": min(1.0, tokens / CORPUS_TOKENS[size]), "train_seconds": 12.5}
        if script == farm.UNEDITED:
            gain = self.curve[multiple] if multiple else (0.002, -0.002, 0.001, -0.001)[(k - 1) % 4]
        else:
            said = json.loads(self.scripts[script])
            if said.get("breaks"):
                return {"by": "rented-one", "valid": False, "reason": "line 3: this line holds text"}, None
            gain = self.gains[size][said["curator"]] + (0.001, -0.001)[(k - 1) % 2]
        return done, [round(0.5 + 0.01 * i + gain, 6) for i in range(len(batch["items"]))]

    def request(self, method, path, body=None, key=None, raw=False, timeout=0, limit=0):
        if path.startswith("/api/v1/rounds/"):
            if path.rsplit("/", 1)[1] not in self.rounds:
                raise client.ApiError(404, "no such round")
            return self.rounds[path.rsplit("/", 1)[1]]
        if path.startswith("/api/v1/rounds?"):
            return {"rounds": [r["round"] for r in reversed(list(self.rounds.values()))]}
        if path == "/api/v1/season":
            return {"season": SEASON.id, "crawl": {"crawl": self.adopted, "paths_sha256": "cd" * 32, "n_shards": 100_000}}
        if path == "/api/v1/leaderboard":
            return {"editors": self.board}
        if path.startswith("/api/v1/editors/"):
            owner = next(r["owner"] for r in self.board + [e for d in self.rounds.values() for e in d["entries"]] if r.get("id", r.get("editor")) == path.split("/")[4])
            return _bundle(owner)[1]
        assert key == "the organisers' key", path  # everything else is the organisers' alone
        if method == "PUT" and path.startswith("/api/v1/admin/farm/scripts/"):
            got = zstandard.ZstdDecompressor().decompress(body, max_output_size=1 << 20)
            assert hashlib.sha256(got).hexdigest() == path.rsplit("/", 1)[1]
            self.scripts[path.rsplit("/", 1)[1]] = got
            self.sent.append(path.rsplit("/", 1)[1])
            return {"ok": True}
        if method == "POST" and path == "/api/v1/admin/farm/batches":
            spec = json.dumps({k: float(v) for k, v in sorted((body.get("scale_spec") or {}).items())}) if body.get("scale_spec") else ""
            bid = f"{body['round']:08d}-{hashlib.sha256((body['items_digest'] + chr(0) + body['scale'] + chr(0) + spec).encode()).hexdigest()[:12]}"
            assert re.fullmatch(r"[a-z][a-z0-9-]{0,23}", body["scale"]) and body["scripts"] and 0 <= body.get("lease", 0) <= 86400
            items = farm.unpack_items(base64.b64decode(body["items_file"]))
            assert digest(items) == body["items_digest"] and len(items) == body["items"]
            b = self.batches.setdefault(bid, {"scale": body["scale"], "spec": body.get("scale_spec"), "lease": body.get("lease", 0), "items": items, "tasks": []})
            for ref in body["scripts"]:
                assert re.fullmatch(r"(unedited|[0-9a-f]{64})(#[2-9])?", ref), ref
                assert ref.split("#")[0] == farm.UNEDITED or ref.split("#")[0] in self.scripts, "a script the server was never sent"
                if ref not in [t["script"] for t in b["tasks"]]:
                    b["tasks"].append({"script": ref, "tries": 0})
            self.posts.append((body["scale"], list(body["scripts"])))
            return {"batch": bid, "tasks": len(b["tasks"])}
        if method == "GET" and path.startswith("/api/v1/admin/farm/batches/"):
            b = self.batches.get(path.rsplit("/", 1)[1])
            if b is None or self.forgetful:
                raise client.ApiError(404, "no such batch")
            for t in b["tasks"]:
                if t["script"] in self.failing:
                    t["tries"] = 3
                elif not t.get("done") and t["script"] not in self.asleep:
                    t["tries"] += 1
                    t["done"], scores = self.student(b, t["script"])
                    if scores is not None:
                        t["item_scores"] = scores
            return {"batch": path.rsplit("/", 1)[1], "tasks": [dict(t) for t in b["tasks"]]}
        raise AssertionError(f"{method} {path}")

    def tasks(self, scale: str) -> list[str]:
        return sorted(t["script"] for b in self.batches.values() if b["scale"] == scale for t in b["tasks"])


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A server, corpora and curators, and the places the stages keep things."""
    server = Server()
    filler = " ".join(["word"] * 99)
    corpora = {"standard": make_corpus(tmp_path / "standard", [filler] * 4),
               # More of the same crawl, and in it a page that one of the exam's news items quotes.
               "large": make_corpus(tmp_path / "large", [filler] * 9 + [QUOTED + " " + " ".join(["word"] * (99 - len(QUOTED.split())))])}
    real = Season.scale
    monkeypatch.setattr(Season, "scale", lambda self, name: TINY.get(name) or real(self, name))
    monkeypatch.setattr(data, "corpus", lambda rnd, scale, quiet=False: corpora[scale.name])
    monkeypatch.setattr("stet.tok.load_encoding", lambda season: FakeEnc())
    monkeypatch.setattr(client, "request", server.request)
    monkeypatch.setattr(client, "need_key", lambda: "the organisers' key")
    monkeypatch.setattr(client, "verify_beacon", lambda season, beacon: None)  # drand is not asked
    monkeypatch.setattr("stet.project.unpack", lambda blob, dest: dest.mkdir(parents=True))
    monkeypatch.setattr("stet.project.load_project", lambda path: types.SimpleNamespace(entry="curator.py", assets=()))
    ran: list[tuple] = []
    odd: dict[str, str] = {}  # curators that do something other than write a script: owner -> what

    def edit(bundle, entry, rnd, scale, work, sandbox="none", time_limit=None, gpu=0, **kw):
        owner = bundle.parent.name.split(".")[0]
        ran.append((owner, rnd.id, scale.name, sandbox, time_limit, kw["limits"]))
        if odd.get(owner) in ("nothing", f"nothing in {rnd.id}"):
            return EditRun("no_script", 3.0, time_limit, 0, None, "", sandbox, reason="the curator wrote no script")
        path = work / "script.jsonl"
        if odd.get(owner) == "same":
            farm.unedited_script(path, len(corpora[scale.name]))
        else:
            path.write_text(json.dumps({"curator": owner, "size": scale.name, "round": rnd.id, **({"breaks": True} if odd.get(owner) == "breaks" else {})}) + "\n")
        return EditRun("ok", 3.0, time_limit, 0, path, "", sandbox)

    monkeypatch.setattr(stages, "_edit", edit)
    return types.SimpleNamespace(server=server, out=tmp_path / "kept", ran=ran, odd=odd, corpora=corpora)


def run(world, *argv) -> int:
    return cli.main([*argv, "--out", str(world.out)])


def read(path) -> dict:
    return json.loads(path.read_text())


# --- the curve -------------------------------------------------------------------------------------


def test_a_corpus_is_counted_in_tokens_from_a_sample_of_it(world):
    enc = FakeEnc()
    assert stages.corpus_tokens(world.corpora["standard"], enc) == 400 and stages.corpus_tokens(world.corpora["large"], enc) == 1000
    # From its first documents alone, scaled by bytes: every document here is as long as every other.
    assert stages.corpus_tokens(world.corpora["standard"], enc, sample=2) == 400


def test_the_curve_is_planned_farmed_and_printed_for_the_season_file(world, capsys):
    rid = world.server.add_round(7, _items())
    # A plan says what it would cost, and which budgets the corpus cannot fill, and posts nothing.
    assert run(world, "curve", rid, "--multiples", "0.5,1,2,4,8", "--plan") == 0
    said = capsys.readouterr()
    assert not world.server.posts and said.out == ""
    assert "GPU hours" in said.err and "standard-x0p5" in said.err and "standard-x8" in said.err
    assert "that student would read it 2.00 times" in said.err  # 8x: 800 tokens of a corpus of 400
    assert "4x is" in said.err and "100% of the corpus" in said.err  # 400 of 400: the training will say whether it reached the end

    assert run(world, "curve", rid, "--multiples", "0.5,1,2,4,8") == 0
    said = capsys.readouterr()
    # Each budget is a batch of its own: the round's own scale as it is, the others as specs with only the budget changed.
    assert world.server.posts == [("standard-x0p5", ["unedited"]), ("standard", ["unedited", "unedited#2"]), ("standard-x2", ["unedited"]),
                                  ("standard-x4", ["unedited"]), ("standard-x8", ["unedited"])]
    specs = {b["scale"]: b["spec"] for b in world.server.batches.values()}
    assert specs["standard"] is None and specs["standard-x2"] == measure.spec_of(measure.at_multiple(TINY["standard"], 2))
    assert specs["standard-x0p5"]["train_tokens"] == 50 and all(len(b["items"]) == 18 for b in world.server.batches.values())
    # 4x read the whole corpus (the training said so) and 8x read it twice: neither is that many times the data.
    assert said.out == '"official/standard": [[0.5, 0.04], [1, 0], [2, -0.035]]\n'
    assert "per doubling of the data: 0.5x to 1x +0.0400 · 1x to 2x +0.0350" in said.err
    kept = read(world.out / rid / "curve-standard.json")
    assert [(p["multiple"], p["trainings"], p["repeats"]) for p in kept["points"]] == [(0.5, 1, False), (1.0, 2, False), (2.0, 1, False), (4.0, 1, True), (8.0, 1, True)]
    assert [p["gain"] for p in kept["points"]] == pytest.approx([-0.04, 0.0, 0.035, 0.06, 0.07])
    assert kept["reference"] == [[0.5, 0.04], [1.0, 0.0], [2.0, -0.035]] and kept["missing"] == [] and kept["rises"] and kept["key"] == "official/standard"
    assert kept["points"][3]["kept_doc_frac"] == [1.0] and kept["points"][2]["passes"] == pytest.approx(0.5)

    # Run again, it trains nothing again and says the same.
    posts = len(world.server.posts)
    assert run(world, "curve", rid, "--multiples", "0.5,1,2,4,8") == 0
    assert len(world.server.posts) == posts and capsys.readouterr().out == said.out


def test_curves_of_several_rounds_are_averaged(world, capsys):
    first = world.server.add_round(7, _items())
    second = world.server.add_round(8, _items()[1:], at=1_795_100_000)
    assert run(world, "curve", first, "--multiples", "0.5,1,2") == 0
    world.server.curve["2"] = 0.045
    assert run(world, "curve", second, "--multiples", "1,2", "--repeats", "4") == 0
    assert ("standard", ["unedited", "unedited#2", "unedited#3", "unedited#4"]) in world.server.posts
    capsys.readouterr()
    assert run(world, "curve", first, second, "s3-o00009", "--average") == 0
    said = capsys.readouterr()
    assert said.out == '"official/standard": [[0.5, 0.04], [1, 0], [2, -0.04]]\n'
    assert "no standard curve measured yet for: s3-o00009" in said.err and "2 rounds" in said.err
    mean = {p["multiple"]: p for p in read(world.out / "curve-standard.json")["points"]}
    assert mean[2.0]["rounds"] == 2 and mean[2.0]["sd"] == pytest.approx(0.01 / 2**0.5) and mean[0.5]["rounds"] == 1


def test_a_round_whose_exam_is_not_here_stops_with_where_it_is(world, capsys, tmp_path):
    items = _items()
    rid = world.server.add_round(7, items)
    made = data.cache_dir() / "examine" / rid
    (made / examine.KEPT_ITEMS).unlink()
    with pytest.raises(SystemExit):
        run(world, "curve", rid)
    assert "its whole exam is not on this machine" in capsys.readouterr().err and not world.server.posts
    # A round of the season before, of which the examiner kept its harvest only: an exam is not made again from one.
    (made / "fresh.jsonl").write_text("")
    with pytest.raises(SystemExit):
        run(world, "curve", rid)
    said = capsys.readouterr().err
    assert "kept its harvest here" in said and "name them with --items and --shares" in said and "Traceback" not in said and not world.server.posts
    # Brought from the examiner's machine as files: taken when they are the round's exam, refused when they are not.
    (tmp_path / "languages.json").write_text(json.dumps({"en": 100.0}))
    (tmp_path / "other.jsonl").write_text("".join(json.dumps(it.to_dict()) + "\n" for it in items[1:]))
    with pytest.raises(SystemExit):
        run(world, "curve", rid, "--items", str(tmp_path / "other.jsonl"), "--shares", str(tmp_path / "languages.json"))
    assert "not the exam the round committed to" in capsys.readouterr().err and not world.server.posts
    (tmp_path / "exam.jsonl").write_text("".join(json.dumps(it.to_dict()) + "\n" for it in items))
    assert run(world, "curve", rid, "--multiples", "1", "--items", str(tmp_path / "exam.jsonl"), "--shares", str(tmp_path / "languages.json")) == 0
    assert world.server.posts == [("standard", ["unedited", "unedited#2"])]
    # A round that has not settled has no students trained on it.
    world.server.rounds[rid]["round"]["status"] = "open"
    with pytest.raises(SystemExit):
        run(world, "curve", rid)
    assert "has not settled" in capsys.readouterr().err


# --- students: shared, lost, forgotten -------------------------------------------------------------


def _round(world, index: int = 7):
    rid = world.server.add_round(index, _items())
    rnd, d = stages.settled_round(SEASON, rid)
    return rnd, _items()


def test_two_measurements_share_a_batch_and_nothing_is_trained_twice(world):
    rnd, items = _round(world)
    bdir = stages.batch_dir(rnd.id, "standard", items, world.out)
    key = "the organisers' key"
    stages.post(rnd, TINY["standard"], items, {farm.UNEDITED: 2}, {}, bdir, key)
    assert stages.follow([(bdir, {farm.UNEDITED: 2})], key)
    # An audit of the same round wants four of the unedited student: it posts the two it lacks.
    sha, raw = _official_script("e1")
    world.server.scripts[sha], world.server.gains["standard"] = raw, {"e1": 0.03}
    stages.post(rnd, TINY["standard"], items, {farm.UNEDITED: 4, sha: 2}, {}, bdir, key)
    assert world.server.posts[-1] == ("standard", ["unedited#3", "unedited#4", sha, f"{sha}#2"])
    world.server.asleep = {sha, f"{sha}#2"}
    # The curve waits for its own two, which are back; the audit for all six, and two are not.
    assert stages.collect(bdir, key, {farm.UNEDITED: 2}) == {"wanted": 2, "done": 2, "left": 0, "under_way": 0, "unposted": 0, "forgotten": False}
    assert stages.collect(bdir, key)["left"] == 2 and not stages.follow([(bdir, None)], key, wait=False)
    # The curve, run again, still wants what the audit wants of this batch, and posts nothing that is back.
    stages.post(rnd, TINY["standard"], items, {farm.UNEDITED: 2}, {}, bdir, key)
    assert read(bdir / "batch.json")["students"] == {farm.UNEDITED: 4, sha: 2} and world.server.posts[-1] == ("standard", [sha, f"{sha}#2"])
    world.server.asleep = set()
    assert stages.follow([(bdir, None)], key)
    assert len(stages.scores(bdir, farm.UNEDITED)) == 4 and len(stages.scores(bdir, sha)) == 2 and len(stages.scores(bdir, sha)[0]) == 18
    assert stages.trained(bdir, sha)[0]["by"] == "rented-one" and stages.refused(bdir, sha) is None


def test_a_student_three_machines_failed_at_is_posted_again_under_the_next_name(world):
    rnd, items = _round(world)
    bdir = stages.batch_dir(rnd.id, "standard", items, world.out)
    key = "the organisers' key"
    world.server.failing = {"unedited#2"}
    stages.post(rnd, TINY["standard"], items, {farm.UNEDITED: 2}, {}, bdir, key)
    said: list[str] = []
    assert not stages.follow([(bdir, None)], key, log=said.append)  # one is back; the other never will be, and it does not wait for it
    assert read(bdir / "batch.json")["lost"] == ["unedited#2"] and "three machines failed" in said[-1]
    stages.post(rnd, TINY["standard"], items, {farm.UNEDITED: 2}, {}, bdir, key)
    assert world.server.posts[-1] == ("standard", ["unedited#3"])
    assert stages.follow([(bdir, None)], key) and [t["script"] for t in stages.trained(bdir, farm.UNEDITED)] == ["unedited", "unedited#3"]


def test_a_batch_the_server_has_forgotten_is_posted_again_less_what_was_collected(world):
    rnd, items = _round(world)
    bdir = stages.batch_dir(rnd.id, "standard", items, world.out)
    key = "the organisers' key"
    world.server.asleep = {"unedited#2", "unedited#3"}
    stages.post(rnd, TINY["standard"], items, {farm.UNEDITED: 3}, {}, bdir, key)
    assert stages.collect(bdir, key)["done"] == 1
    world.server.forgetful, world.server.asleep = True, set()
    got = stages.collect(bdir, key)
    assert got["forgotten"] and got["done"] == 1 and got["unposted"] == 2 and got["left"] == 0
    world.server.forgetful = False
    stages.post(rnd, TINY["standard"], items, {farm.UNEDITED: 3}, {}, bdir, key)
    assert world.server.posts[-1] == ("standard", ["unedited#2", "unedited#3"])
    assert stages.follow([(bdir, None)], key) and len(stages.scores(bdir, farm.UNEDITED)) == 3


def test_a_long_student_is_held_for_long_enough_whatever_scale_it_is(monkeypatch):
    """The season's own large scale has no spec, and still takes longer than the server's usual hold."""
    sent = []
    monkeypatch.setattr(client, "request", lambda method, path, body=None, **kw: sent.append(body) or {"batch": "b"})
    rnd = official_round(SEASON, 7, {"round": _slot(1_795_000_200)[0], "randomness": "ab" * 32}, {"crawl": "CC-MAIN-2026-43", "paths_sha256": "cd" * 32, "n_shards": 9})
    farm.post_batch(rnd, SEASON.scale("large"), _items(), ["unedited"], "key", lease=24000)
    farm.post_batch(rnd, SEASON.scale("standard"), _items(), ["unedited"], "key")
    assert sent[0]["lease"] == 24000 and "scale_spec" not in sent[0] and "lease" not in sent[1]
    # And a machine reads the corpus of the season's scale of that size, whatever the batch's scale is called.
    double = measure.at_multiple(SEASON.scale("large"), 2)
    assert farm.corpus_scale(SEASON, double) == SEASON.scale("large") and farm.corpus_scale(SEASON, SEASON.scale("micro")) == SEASON.scale("micro")
    odd = Scale(name="odd", corpus_units=3, train_tokens=1, n_layer=1, n_head=1, d_model=8, ctx=8, batch_size=1, lr=0.1, edit_seconds=1)
    assert farm.corpus_scale(SEASON, odd) is odd


# --- the audit -------------------------------------------------------------------------------------

ENTRIES = [("e1", 0.04, True), ("e2", 0.03, True), ("e3", 0.02, True), ("e4", 0.01, True), ("e5", 0.05, True), ("e6", 0.06, True), ("e7", 0.0, True),
           ("e8", 0.0, False)]  # e8 wrote no script in the round


def test_the_audit_trains_a_round_at_two_sizes_and_says_whether_the_order_carried(world, capsys):
    rid = world.server.add_round(7, _items(), ENTRIES)
    world.server.gains = {"standard": {"e1": 0.04, "e2": 0.03, "e3": 0.02, "e4": 0.01, "e5": 0.05, "e6": 0.06, "e7": 0.0},
                          "large": {"e1": 0.08, "e2": 0.06, "e3": 0.02, "e4": 0.04}}  # e3 and e4 change places one size up
    world.odd.update(e5="nothing", e6="breaks", e7="same")

    assert run(world, "audit", "plan", rid) == 0
    said = capsys.readouterr().err
    plan = read(world.out / rid / "audit.json")
    assert plan["sizes"] == ["standard", "large"] and [e["name"] for e in plan["entries"]] == [f"e{i}/c@v1" for i in range(1, 8)]
    assert "7 curators, each script trained 2 times, the unedited corpus 4" in said and "GPU hours" in said and "stet audit scripts" in said
    with pytest.raises(SystemExit):
        run(world, "audit", "farm", rid)  # nothing is posted before the curators have been run one size up
    assert "run the scripts stage" in capsys.readouterr().err and not world.server.posts

    assert run(world, "audit", "scripts", rid) == 0
    said = capsys.readouterr().err
    # Each curator is run once, on the larger corpus only, in the sandbox, with that size's own time limit.
    assert [(r[0], r[2], r[3], r[4], r[5]) for r in world.ran] == [(f"e{i}", "large", "bwrap", 120.0, {"required": True}) for i in range(1, 8)]
    assert "large: 6 of 7 curators wrote a script" in said and "1 of the exam's items are in the large corpus" in said
    scripts = world.out / rid / "large" / "scripts"
    assert read(scripts / "e5.c.v1.json")["status"] == "no_script" and not (scripts / "e5.c.v1.jsonl").exists()
    assert json.loads((scripts / "e1.c.v1.jsonl").read_text()) == {"curator": "e1", "size": "large", "round": rid}
    assert read(world.out / rid / "large" / "corpus.json")["tokens"] == 1000
    assert not any((scripts / "runs").iterdir())  # nothing of a curator's run is kept but its script
    assert run(world, "audit", "scripts", rid) == 0 and len(world.ran) == 7  # run again, it runs nobody again

    assert run(world, "audit", "farm", rid) == 0
    said = capsys.readouterr().err
    own = {o: _official_script(o)[0] for o in ("e1", "e2", "e3", "e4", "e6")}
    made = {o: hashlib.sha256((scripts / f"{o}.c.v1.jsonl").read_bytes()).hexdigest() for o in ("e1", "e2", "e3", "e4", "e6")}
    # A curator with no script one size up (e5), or one that keeps the corpus as it is (e7), is trained at neither size.
    assert world.server.tasks("standard") == sorted(["unedited", "unedited#2", "unedited#3", "unedited#4"] + [r for s in own.values() for r in (s, f"{s}#2")])
    assert world.server.tasks("large") == sorted(["unedited", "unedited#2", "unedited#3", "unedited#4"] + [r for s in made.values() for r in (s, f"{s}#2")])
    assert sorted(world.server.sent) == sorted(made.values())  # the round's own scripts are the server's already
    assert all(len(b["items"]) == 17 and QUOTED not in [it.text for it in b["items"]] for b in world.server.batches.values())  # both sizes sit the same items
    assert "28 of 28 trained" in said and f"stet audit report {rid}" in said

    assert run(world, "audit", "report", rid) == 0
    said = capsys.readouterr().err
    rep = read(world.out / rid / "audit-report.json")
    assert rep["items"] == 17 and rep["dropped"] == 1 and rep["sizes"] == ["standard", "large"]
    assert sorted(rep["curators"]) == ["e1/c@v1", "e2/c@v1", "e3/c@v1", "e4/c@v1"]
    assert rep["curators"]["e4/c@v1"] == {"standard": pytest.approx(0.01), "large": pytest.approx(0.04)}
    score = rep["measures"]["score"]
    assert score["carried"] == {"standard>large": pytest.approx(0.8)}
    assert score["sizes"]["standard"]["noise"] == pytest.approx(0.001 * 2**0.5) and score["sizes"]["large"]["spread"] == pytest.approx(0.02 * (5 / 3) ** 0.5)
    # The score and each pillar of the exam, and nothing else: every item sat is in one of them.
    assert set(rep["measures"]) == {"score", "pillar:news", "pillar:logic"} and all(set(m) == {"sizes", "carried"} for m in rep["measures"].values())
    assert rep["measures"]["pillar:logic"]["carried"] == {"standard>large": pytest.approx(0.8)}
    assert rep["left_out"] == {"e5/c@v1": "it wrote no script at large", "e6/c@v1": "its large script breaks a rule: line 3: this line holds text",
                               "e7/c@v1": "its large script keeps the corpus as it is: it is the unedited student"}
    assert rep["unedited"] == {"standard": 4, "large": 4} and rep["trainings"]["large"]["e1/c@v1"] == 2
    assert "The order of curators carried from standard to large at +0.80" in said and "left out: e5/c@v1: it wrote no script at large" in said
    assert "logic" in said and "(1 left out: a bigger corpus holds them)" in said and "stet audit report --over" in said

    # The rounds audited so far as one: a round has too few curators for its order to say much alone.
    assert run(world, "audit", "report", "--over", rid, "s3-o00099") == 0
    said = capsys.readouterr().err
    over = read(world.out / "audit-over.json")
    assert over["over"] == [rid] and over["rounds"] == 1 and set(over["measures"]) == set(rep["measures"])
    assert over["measures"]["score"]["carried"] == {"standard>large": {"rounds": 1, "mean": pytest.approx(0.8)}}
    assert "no audit report yet for: s3-o00099" in said and f"rounds audited, 1: {rid}" in said and "standard to large +0.80 (1)" in said
    # With what was collected, the report needs no server.
    world.server.forgetful = True
    assert run(world, "audit", "report", rid, "--offline") == 0 and read(world.out / rid / "audit-report.json")["measures"] == rep["measures"]


def test_an_audit_needs_three_curators_and_starts_at_the_rounds_own_size(world, capsys):
    rid = world.server.add_round(7, _items(), ENTRIES[:2])
    with pytest.raises(SystemExit):
        run(world, "audit", "plan", rid)
    assert "an order of fewer than three says nothing" in capsys.readouterr().err
    rid = world.server.add_round(8, _items(), ENTRIES, at=1_795_100_000)
    with pytest.raises(SystemExit):
        run(world, "audit", "plan", rid, "--sizes", "large")
    assert "was run at standard" in capsys.readouterr().err
    # At the round's own size alone, an audit needs no curator run and no corpus but the round's: how steady is the order?
    world.server.gains = {"standard": {f"e{i}": 0.01 * i for i in range(1, 8)}}
    assert run(world, "audit", "plan", rid, "--sizes", "standard", "--entries", "e1/c@v1,e2/c@v1,e3/c@v1,e4/c@v1", "--trainings", "3") == 0
    assert run(world, "audit", "scripts", rid) == 0 and not world.ran
    assert run(world, "audit", "farm", rid) == 0 and len(world.server.tasks("standard")) == 4 + 4 * 3
    assert run(world, "audit", "report", rid) == 0
    rep = read(world.out / rid / "audit-report.json")
    assert rep["sizes"] == ["standard"] and rep["items"] == 18 and rep["measures"]["score"]["carried"] == {} and len(rep["curators"]) == 4


def test_no_two_curators_share_a_label_and_so_a_kept_script():
    """A label is the file a curator's script is kept under. Hyphens are in handles and in names,
    so a label joined by them was one for foo-bar/baz and foo/bar-baz."""
    a, b = {"owner": "foo-bar", "name": "baz", "version": 1}, {"owner": "foo", "name": "bar-baz", "version": 1}
    assert (stages.label_of(a), stages.label_of(b)) == ("foo-bar.baz.v1", "foo.bar-baz.v1")
    # Nor a name that ends as a version does, with another's version.
    assert stages.label_of({"owner": "foo", "name": "bar-v1", "version": 2}) == "foo.bar-v1.v2" != stages.label_of({"owner": "foo", "name": "bar", "version": 12})
    handles = ["a-b", "a", "b-c", "ab", "a-b-c", "c", "b"]
    curators = [{"owner": o, "name": n, "version": v} for o in handles for n in handles for v in (1, 2, 11)]
    assert len({stages.label_of(c) for c in curators}) == len(curators)
    assert all(re.fullmatch(r"[a-z0-9-]+\.[a-z0-9-]+\.v[0-9]+", stages.label_of(c)) for c in curators)  # still a name to read, and a file's


# --- the finals ------------------------------------------------------------------------------------


def _board(server: Server) -> None:
    def row(rank, owner, rounds=9):
        return {"id": _bundle(owner)[0], "name": "c", "version": 1, "owner": owner, "github": owner, "rank": rank, "rounds": rounds, "rating": 1 - rank / 100,
                "se": 0.005, "lower": 1 - rank / 100 - 1.28 * 0.005, "organisers": owner == "stet"}

    owners = ["f1", "f2", "stet", "f3", "f4", "f5", "f6", "f7", "f8", "q1", "q2", "q3"]
    server.board = [row(i + 1, o, rounds=3 if o == "q3" else 9) for i, o in enumerate(owners)]
    server.gains["large"] = {"f1": 0.08, "f2": 0.07, "f3": 0.06, "f4": 0.05, "f5": 0.04, "f6": 0.03, "f7": 0.02, "f8": 0.019, "q1": 0.055, "q2": 0.055, "stet": 0.045}


def test_the_finals_from_the_board_to_the_table(world, capsys):
    server = world.server
    _board(server)
    old = server.add_round(7, _items())  # a round of the season, on the crawl in use when the finalists are frozen
    now = int(time.time())

    assert run(world, "finals", "plan") == 0
    said = capsys.readouterr().err
    fdir = world.out / "finals-s3"
    plan = read(fdir / "plan.json")
    assert [r["owner"] for r in plan["finalists"]] == [f"f{i}" for i in range(1, 9)] and [r["owner"] for r in plan["yardsticks"]] == ["stet"]
    assert [r["owner"] for r in plan["pool"]] == ["q1", "q2"]  # q3 has run three rounds
    # The frozen board is published with what the finalists were taken by: each row's cautious rating.
    assert len(plan["board"]) == 12 and all(r["lower"] == pytest.approx(r["rating"] - 1.28 * r["se"]) for r in plan["board"]) and "at least +0.9836" in said
    assert plan["organisers"] == ["stet"] and plan["frozen_round"] == old and now <= plan["frozen_at"] <= now + 60
    assert plan["crawl"] == "CC-MAIN-2026-45"  # the newest there was: adopted for the next draw, though the last round used an older one
    assert "the lot is drawn among 2: q1/c@v1, q2/c@v1" in said and "not drawn yet" in said and "yardsticks, run beside them and given no place: stet/c@v1" in said
    assert "GPU hours" in said and "large-x4" in said
    with pytest.raises(SystemExit):
        run(world, "finals", "scripts")
    assert "the finals' rounds are not named yet" in capsys.readouterr().err

    # The draws are announced: two beacon rounds of the season's grid that have not been emitted.
    (b1, t1), (b2, t2) = _slot(now + 86400), _slot(now + 2 * 86400)
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--draws", str(b1 + 1), str(b2))
    assert "not on the season's draw grid" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--draws", str(_slot(now - 86400)[0]), str(b2))
    assert "emitted before the finalists were frozen" in capsys.readouterr().err
    assert run(world, "finals", "plan", "--draws", str(b2), str(b1)) == 0 and read(fdir / "plan.json")["draws"] == [b1, b2]

    # The lot: a beacon round named beforehand, and anyone's to work out again.
    lot = SEASON.beacon_round_at(now + 3600)
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--lot-beacon", f"{SEASON.beacon_round_at(now - 3600)}:{'ab' * 32}")
    assert "the lot's beacon round is named beforehand" in capsys.readouterr().err
    assert run(world, "finals", "plan", "--lot-beacon", f"{lot}:{'AB' * 32}") == 0
    plan = read(fdir / "plan.json")
    pick = finals.draw_lot("s3", plan["pool"], {"round": lot, "randomness": "ab" * 32})
    assert plan["lot"] == {"beacon": {"round": lot, "randomness": "ab" * 32}, "checked": True, "pick": pick} and pick["owner"] in ("q1", "q2")
    assert f"picks: {pick['owner']}/c@v1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--lot-beacon", f"{lot + 1}:{'cd' * 32}")
    assert "it is not drawn again from another" in capsys.readouterr().err
    assert [r["owner"] for r in read(fdir / "plan.json")["finalists"]] == [f"f{i}" for i in range(1, 9)]  # frozen: the board is not asked again

    # The rounds: drawn from the announced beacon rounds, on a crawl published since.
    stale = server.add_round(20, _items(), at=t1 - 1, crawl="CC-MAIN-2026-45")  # a crawl anyone could have read before entries closed
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--rounds", stale, old)
    assert "no newer than the newest crawl there was when the finalists were frozen (CC-MAIN-2026-45)" in capsys.readouterr().err
    r1, r2 = server.add_round(21, _items(), at=t1 - 1, crawl="CC-MAIN-2026-47"), server.add_round(22, _items()[2:], at=t2 - 1, crawl="CC-MAIN-2026-47")
    unannounced = server.add_round(23, _items(), at=t2 + 86400, crawl="CC-MAIN-2026-47")
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--rounds", r1, unannounced)
    assert "was not announced for the finals" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--rounds", r1)
    assert "the finals are 2 rounds" in capsys.readouterr().err
    for again in (r1, "s3-o21"):  # one round named twice is not two rounds, however it is spelt
        with pytest.raises(SystemExit):
            run(world, "finals", "plan", "--rounds", r1, again)
        assert f"{r1} is named twice" in capsys.readouterr().err and read(fdir / "plan.json")["rounds"] == []
    assert run(world, "finals", "plan", "--rounds", r1, r2) == 0
    assert read(fdir / "plan.json")["rounds"] == [r1, r2] and read(fdir / "plan.json")["own"] == {r1: "standard", r2: "standard"}

    # Every curator is run on each round's large corpus, here, in the sandbox. f6 writes nothing in the second round.
    world.odd["f6"] = f"nothing in {r2}"
    capsys.readouterr()
    assert run(world, "finals", "scripts") == 0
    said = capsys.readouterr().err
    runners = [f"f{i}" for i in range(1, 9)] + [pick["owner"], "stet"]
    assert [(r[0], r[1], r[2], r[3], r[4]) for r in world.ran] == [(o, rid, "large", "bwrap", 120.0) for rid in (r1, r2) for o in runners]
    assert f"{r1} · large: 10 of 10 curators wrote a script" in said and f"{r2} · large: 9 of 10 curators wrote a script" in said

    # The farm: f2's students of the first round are slow to come back, and the table says it is not final.
    f2 = hashlib.sha256((world.out / r1 / "large" / "scripts" / "f2.c.v1.jsonl").read_bytes()).hexdigest()
    server.asleep = {f2, f"{f2}#2"}
    assert run(world, "finals", "farm", "--no-wait") == 0
    said = capsys.readouterr().err
    assert "50 of 52 trained" in said and "stet finals table" not in said  # 2 x (4 + 10 x 2 + 3 of the curve), less f6's two, and f2's are asleep
    larges = [b for b in server.batches.values() if b["scale"] == "large"]
    assert sorted(len(b["tasks"]) for b in larges) == [22, 24] and all(b["lease"] == 0 and b["spec"] is None for b in larges)
    assert sorted({b["scale"] for b in server.batches.values()}) == ["large", "large-x0p5", "large-x2", "large-x4"]
    assert len(server.sent) == 19 and {len(b["items"]) for b in server.batches.values()} == {17, 16}  # the first exam's 18 less the one the large corpus holds, and the second's 16
    assert run(world, "finals", "table") == 0
    said = capsys.readouterr().err
    early = read(fdir / "table.json")
    assert not early["complete"] and early["waiting"] == ["f2/c@v1"] and "f2/c@v1" not in [r["name"] for r in early["rows"]]
    assert "This table is not final" in said and f"{r1}: 2 of 2 trainings of f2/c@v1" in said

    server.asleep = set()
    assert run(world, "finals", "farm") == 0
    assert "52 of 52 trained" in capsys.readouterr().err
    assert run(world, "finals", "table") == 0
    said = capsys.readouterr().err
    out = read(fdir / "table.json")
    rows = {r["name"].split("/")[0]: r for r in out["rows"]}
    lot_name = pick["owner"]
    assert out["complete"] and out["missing"] == [] and out["rounds"] == [r1, r2] and out["scale"] == "large" and not out["rehearsal"]
    # Every script trained twice a round, 0.001 either side of its gain, and nobody moved between
    # the rounds: a difference is uncertain by 0.001, and gains within 0.002 share a place.
    assert out["se_difference"] == pytest.approx(0.001)
    assert {o: r["gain"] for o, r in rows.items()} == pytest.approx(
        {"f1": 0.08, "f2": 0.07, "f3": 0.06, lot_name: 0.055, "f4": 0.05, "stet": 0.045, "f5": 0.04, "f7": 0.02, "f8": 0.019, "f6": 0.015})
    assert {o: r["place"] for o, r in rows.items()} == {"f1": 1, "f2": 2, "f3": 3, lot_name: 4, "f4": 5, "stet": None, "f5": 6, "f7": 7, "f8": 7, "f6": 9}
    assert rows["f7"]["shares_place_with"] == ["f8/c@v1"] and rows["stet"]["role"] == "yardstick" and rows[lot_name]["role"] == "lot"
    assert rows["f6"]["rounds"] == pytest.approx([0.03, 0.0]) and rows["f6"]["note"] == f"no script in {r2}: 0 there" and rows["f6"]["trainings"] == [2, 0]
    assert out["lot"] == {"name": f"{lot_name}/c@v1", "place": 4, "ahead_of": [f"f{i}/c@v1" for i in (4, 5, 7, 8, 6)], "level_with": []}
    # The finals' own curve stops where the corpus does: the student at 4x read all of it.
    assert out["curve"] == [[0.5, 0.04], [1.0, 0.0], [2.0, -0.035]] and len(out["curve_notes"]) == 2 and "read some of the corpus twice" in out["curve_notes"][0]
    assert (rows["f1"]["multiplier"], rows["f1"]["capped"]) == (2.0, True) and rows["f7"]["multiplier"] == pytest.approx(2 ** (0.02 / 0.035))
    assert f"The lot's pick, {lot_name}/c@v1, placed 4, ahead of 5 of the finalists the rounds chose" in said
    assert "Two curators whose gains differ by less than 0.0020 (2 standard errors of a difference) share a place." in said and "7=" in said

    # What is published: the plan, the table, each round's exam as a settled round's is, and every score.
    pub = fdir / "published"
    assert read(pub / "table.json") == out and read(pub / "plan.json") == read(fdir / "plan.json")
    exam = [json.loads(line) for line in (pub / r1 / "exam.jsonl").read_text().splitlines()]
    assert len(exam) == 17 and sum(1 for row in exam if row.get("withheld")) == 5 and all("text" not in row for row in exam if row.get("withheld"))
    scores = read(pub / r1 / "scores.json")
    assert len(scores["items"]) == 17 and scores["left_out"] == 1 and len(scores["unedited"]) == 4 and sorted(scores["curve"]) == ["0.5x", "2x"]
    assert sorted(scores["students"]) == sorted(r["name"] for r in out["rows"]) and all(len(s) == 2 and len(s[0]) == 17 for s in scores["students"].values())
    assert scores["scripts"]["f2/c@v1"] == f2 and read(pub / r2 / "scores.json")["scripts"]["f6/c@v1"] is None
    assert QUOTED not in (pub / r1 / "exam.jsonl").read_text() + (pub / r1 / "scores.json").read_text()


def test_a_rehearsal_is_kept_apart_and_says_what_it_is(world, capsys):
    server = world.server
    _board(server)
    server.gains["standard"] = server.gains["large"]
    rid = server.add_round(7, _items())
    with pytest.raises(SystemExit):
        run(world, "finals", "plan", "--scale", "standard")
    assert "another size is for a rehearsal" in capsys.readouterr().err and not (world.out / "finals-s3").exists()
    # Any settled round will do, at any size, with a beacon round that has been emitted already. An account named by hand
    # is the organisers' as well as those the board marks: f1's curator is a yardstick beside stet's.
    lot = SEASON.beacon_round_at(int(time.time()) - 3600)
    assert run(world, "finals", "plan", "--rehearsal", "--scale", "standard", "--organisers", "f1", "--rounds", rid, "--lot-beacon", f"{lot}:{'ab' * 32}") == 0
    plan = read(world.out / "finals-s3-rehearsal" / "plan.json")
    assert plan["rehearsal"] and plan["scale"] == "standard" and plan["rounds"] == [rid] and [r["owner"] for r in plan["yardsticks"]] == ["f1", "stet"]
    assert [r["owner"] for r in plan["finalists"]] == ["f2", "f3", "f4", "f5", "f6", "f7", "f8", "q1"] and [r["owner"] for r in plan["pool"]] == ["q2"]
    assert "a rehearsal" in capsys.readouterr().err and not (world.out / "finals-s3").exists()
    with pytest.raises(SystemExit):
        run(world, "finals", "scripts")  # the finals themselves are not planned
    capsys.readouterr()
    assert run(world, "finals", "scripts", "--rehearsal") == 0 and {r[2] for r in world.ran} == {"standard"} and len(world.ran) == 11
    assert run(world, "finals", "farm", "--rehearsal") == 0
    assert run(world, "finals", "table", "--rehearsal") == 0
    out = read(world.out / "finals-s3-rehearsal" / "table.json")
    assert out["rehearsal"] and out["complete"] and out["lot"]["name"] == "q2/c@v1" and "a rehearsal, not the finals" in capsys.readouterr().err
    assert [r["name"] for r in out["rows"] if r["role"] == "yardstick"] == ["f1/c@v1", "stet/c@v1"] and len(out["rows"]) == 11


def test_the_organisers_curators_are_those_the_board_marks_and_the_list_of_verifiers_is_not_asked(world, capsys):
    """The list of verifiers is cut at 200 rows, and the organisers' accounts sort last on it: the
    fake server here has no such list at all, and would fail the test if it were asked for."""
    server = world.server
    _board(server)
    server.add_round(7, _items())
    server.board[0]["owner"], server.board[0]["organisers"] = "stet-two", True  # a second account of theirs, top of the board
    assert run(world, "finals", "plan", "--rehearsal") == 0
    plan = read(world.out / "finals-s3-rehearsal" / "plan.json")
    assert plan["organisers"] == ["stet", "stet-two"] and [r["owner"] for r in plan["yardsticks"]] == ["stet-two", "stet"]
    assert [r["owner"] for r in plan["finalists"]] == ["f2", "f3", "f4", "f5", "f6", "f7", "f8", "q1"]  # none of them theirs
    capsys.readouterr()
    # A board that does not say (an older server's) is not guessed at: the accounts are named, or nothing is planned.
    for r in server.board:
        del r["organisers"]
    with pytest.raises(SystemExit):
        run(world, "finals", "plan")
    assert "does not say which curators are the organisers' own" in capsys.readouterr().err and not (world.out / "finals-s3").exists()
    assert run(world, "finals", "plan", "--organisers", "stet,stet-two") == 0
    assert [r["owner"] for r in read(world.out / "finals-s3" / "plan.json")["yardsticks"]] == ["stet-two", "stet"]


def test_the_lot_is_not_drawn_from_a_beacon_round_that_has_not_been_emitted():
    with pytest.raises(stages.Stop, match="the lot is drawn after that"):
        stages.fetch_beacon(SEASON, SEASON.beacon_round_at(int(time.time()) + 3600))  # and no relay is asked
