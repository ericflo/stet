"""Model weights by reference: what an entry may name, and how a machine comes to hold it."""

import hashlib
import os
import time
import tomllib

import pytest

from stet import assets
from stet.assets import Asset, AssetError
from stet.project import ProjectError, load_project
from stet.spec import load_season

COMMIT = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
URL = f"https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/{COMMIT}/model.safetensors"
SHA = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
GOOD = {"path": "smol/model.safetensors", "url": URL, "sha256": SHA, "size": 269060552}


@pytest.fixture
def season():
    return load_season()


def test_a_well_formed_table_is_taken_as_written(season):
    assert assets.parse([GOOD], season) == [Asset("smol/model.safetensors", URL, SHA, 269060552)]
    assert assets.parse(None, season) == [] and assets.parse([], season) == []
    dataset = {**GOOD, "url": f"https://huggingface.co/datasets/owner/name/resolve/{COMMIT}/data/train.bin"}
    assert assets.parse([dataset], season)[0].url.startswith("https://huggingface.co/datasets/")


@pytest.mark.parametrize(
    "change, why",
    [
        ({"url": URL.replace(COMMIT, "main")}, "pinned to a commit"),  # a branch can move
        ({"url": URL.replace("https://", "http://")}, "pinned to a commit"),
        ({"url": URL.replace("huggingface.co", "example.test")}, "pinned to a commit"),
        ({"url": URL + "?download=true"}, "pinned to a commit"),
        ({"url": URL.replace("/model.safetensors", "/../../other/model.safetensors")}, "pinned to a commit"),
        ({"url": URL.replace("/model.safetensors", "//model.safetensors")}, "pinned to a commit"),
        ({"sha256": SHA.upper()}, "64 lowercase hex"),
        ({"sha256": SHA[:40]}, "64 lowercase hex"),
        ({"size": 0}, "length in bytes"),
        ({"size": True}, "length in bytes"),
        ({"size": 1.5}, "length in bytes"),
        ({"size": "269060552"}, "length in bytes"),
        ({"path": "/etc/passwd"}, "relative path of plain names"),
        ({"path": "../up"}, "relative path of plain names"),
        ({"path": "smol/.hidden"}, "relative path of plain names"),
        ({"path": "smol//model"}, "relative path of plain names"),
        ({"path": "smol/mo del"}, "relative path of plain names"),
        ({"path": "a/" * 120 + "b"}, "relative path of plain names"),
        ({"path": 7}, "must be strings"),
        ({"mirror": "https://example.test/x"}, "unknown key 'mirror'"),
    ],
)
def test_each_rule_is_enforced(season, change, why):
    with pytest.raises(AssetError, match=why):
        assets.parse([{**GOOD, **change}], season)


def test_a_missing_field_and_a_wrong_shape_are_refused(season):
    for field in ("path", "url", "sha256", "size"):
        with pytest.raises(AssetError, match=f"missing {field}"):
            assets.parse([{k: v for k, v in GOOD.items() if k != field}], season)
    for shape in (5, "none", {}, [1], [[GOOD]]):
        with pytest.raises(AssetError, match=r"\[\[assets\]\] tables"):
            assets.parse(shape, season)


def test_two_assets_cannot_claim_one_place(season):
    for a, b in (("smol/model", "smol/model"), ("smol/Model", "smol/model"), ("smol", "smol/model"), ("smol/model/x", "smol/model")):
        with pytest.raises(AssetError, match="collides"):
            assets.parse([{**GOOD, "path": a}, {**GOOD, "path": b}], season)
    assert len(assets.parse([{**GOOD, "path": "smol/model"}, {**GOOD, "path": "smol/model2"}], season)) == 2


def test_the_season_bounds_how_many_and_how_much(season):
    lim = assets.limits(season)
    assert lim["max_files"] >= 8 and lim["max_bytes"] >= 2 << 30  # room for a small language model
    many = [{**GOOD, "path": f"m/{i}", "size": 1} for i in range(lim["max_files"] + 1)]
    with pytest.raises(AssetError, match=f"the limit is {lim['max_files']}"):
        assets.parse(many, season)
    big = [{**GOOD, "path": "a", "size": lim["max_bytes"]}, {**GOOD, "path": "b", "size": 1}]
    with pytest.raises(AssetError, match="GiB; the limit is"):
        assets.parse(big, season)
    assert len(assets.parse([{**GOOD, "path": "a", "size": lim["max_bytes"]}], season)) == 1


def test_a_season_that_says_nothing_allows_none(season, monkeypatch):
    monkeypatch.setitem(season.raw, "assets", None)
    assert assets.parse([], season) == []
    with pytest.raises(AssetError, match="does not allow"):
        assets.parse([GOOD], season)


def test_what_is_written_into_a_manifest_reads_back_the_same(season, tmp_path):
    items = assets.parse([GOOD, {**GOOD, "path": "smol/config.json", "size": 704}], season)
    (tmp_path / "stet.toml").write_text('name = "a-curator"\nentry = "python curator.py"\n\n' + assets.to_toml(items))
    assert tomllib.loads((tmp_path / "stet.toml").read_text())["assets"][0] == GOOD
    assert load_project(tmp_path).assets == items
    (tmp_path / "stet.toml").write_text('name = "a-curator"\nentry = "python curator.py"\n[[assets]]\npath = "x"\n')
    with pytest.raises(ProjectError, match="asset 1: missing url"):
        load_project(tmp_path)


def blob_asset(path: str, blob: bytes) -> Asset:
    return Asset(path, URL, hashlib.sha256(blob).hexdigest(), len(blob))


def test_a_file_is_fetched_once_checked_and_kept_by_its_hash(monkeypatch):
    from stet import data

    blob = b"weights" * 100
    a = blob_asset("smol/model.safetensors", blob)
    calls = []

    def fetch(urls, sha256, dest, label="", quiet=False, max_bytes=0, gzipped=False):
        calls.append((urls, sha256, max_bytes))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        return dest

    monkeypatch.setattr(data, "fetch", fetch)
    assets.fetch([a], quiet=True)
    assert calls == [([URL], a.sha256, len(blob))]  # never more bytes than the entry said
    assert assets.cached(a).read_bytes() == blob and not (assets.cached(a).stat().st_mode & 0o222)
    assets.fetch([a, blob_asset("other/name", blob)], quiet=True)  # the same bytes under another name: one copy
    assert len(calls) == 1


def test_a_file_that_is_not_the_size_its_entry_says_is_not_kept(monkeypatch):
    from stet import data

    a = Asset("smol/model", URL, SHA, 1000)

    def fetch(urls, sha256, dest, **k):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"short")
        return dest

    monkeypatch.setattr(data, "fetch", fetch)
    with pytest.raises(assets.Unavailable, match="not the size"):  # the entry's mistake, not the network's
        assets.fetch([a], quiet=True)
    assert not assets.cached(a).exists()


def test_the_cache_lets_go_of_what_was_used_longest_ago_and_never_of_what_is_needed(tmp_path):
    import os

    d = assets.cache()
    d.mkdir(parents=True)
    made = []
    for i in range(4):
        a = blob_asset(f"m/{i}", bytes([i]) * 1000)
        assets.cached(a).write_bytes(bytes([i]) * 1000)
        os.utime(assets.cached(a), (1000 + i, 1000 + i))
        made.append(a)
    (d / "notes.txt").write_text("not ours to remove")
    assert assets.prune(keep=[made[0]], limit_bytes=2500) == 2
    assert [assets.cached(a).exists() for a in made] == [True, False, False, True] and (d / "notes.txt").exists()
    assert assets.prune(limit_bytes=10**9) == 0


def test_outside_the_sandbox_the_files_are_laid_out_by_their_paths(tmp_path):
    a, b = blob_asset("smol/model.safetensors", b"w"), blob_asset("smol/sub/config.json", b"{}")
    assets.cache().mkdir(parents=True)
    for x, blob in ((a, b"w"), (b, b"{}")):
        assets.cached(x).write_bytes(blob)
    root = assets.link_tree([a, b], tmp_path / "assets")
    assert (root / "smol/model.safetensors").read_bytes() == b"w" and (root / "smol/sub/config.json").read_bytes() == b"{}"
    assets.link_tree([a], tmp_path / "assets")  # again, over what is there


TREE = [
    {"type": "file", "path": ".gitattributes", "size": 1519},
    {"type": "file", "path": "README.md", "size": 6340},
    {"type": "file", "path": "config.json", "size": 704},
    {"type": "file", "path": "model.safetensors", "size": 135, "lfs": {"oid": SHA, "size": 269060552}},
    {"type": "file", "path": "pytorch_model.bin", "size": 135, "lfs": {"oid": "b" * 64, "size": 269000000}},
    {"type": "file", "path": "onnx/model.onnx", "size": 135, "lfs": {"oid": "c" * 64, "size": 540000000}},
    {"type": "directory", "path": "onnx"},
]


def test_a_repository_is_pinned_to_the_commit_it_is_at(monkeypatch):
    asked = []

    def get(url):
        asked.append(url)
        return TREE if "/tree/" in url else {"sha": COMMIT, "gated": False, "private": False}

    monkeypatch.setattr(assets, "_get_json", get)
    commit, found, skipped = assets.lookup("HuggingFaceTB/SmolLM2-135M")
    assert commit == COMMIT and f"/tree/{COMMIT}?" in asked[1]
    assert [(a.path, a.size, a.sha256) for a in found] == [("smollm2-135m/config.json", 704, ""), ("smollm2-135m/model.safetensors", 269060552, SHA)]
    assert found[1].url == URL and set(skipped) == {".gitattributes", "README.md", "pytorch_model.bin", "onnx/model.onnx"}
    _, only, _ = assets.lookup("HuggingFaceTB/SmolLM2-135M", include=["*.bin"], under="weights")
    assert [a.path for a in only] == ["weights/pytorch_model.bin"]
    assets.lookup("HuggingFaceTB/SmolLM2-135M", revision="refs/pr/1")
    assert asked[-2].endswith("/revision/refs%2Fpr%2F1")


def test_a_repository_that_needs_an_account_is_refused(monkeypatch):
    monkeypatch.setattr(assets, "_get_json", lambda url: {"sha": COMMIT, "gated": "manual"})
    with pytest.raises(AssetError, match="volunteers have no login"):
        assets.lookup("meta-llama/Llama-3.2-1B")
    with pytest.raises(AssetError, match="owner/name"):
        assets.lookup("just-a-name")


def test_the_example_that_uses_a_model_names_files_the_season_allows(season):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples" / "reader"
    project = load_project(root)
    assert len(project.assets) >= 2 and all(a.path.startswith("qwen3-0.6b/") for a in project.assets)
    assert sum(a.size for a in project.assets) < 2 << 30  # a 0.6B model: this is the pattern, not the limit
    assert len({a.url.split("/resolve/")[1].split("/")[0] for a in project.assets}) == 1  # one commit
    source = (root / "curator.py").read_text()
    compile(source, "curator.py", "exec")
    assert 'task.asset("qwen3-0.6b")' in source and "task.device()" in source
    main = source[source.index("def main()") :]
    assert main.index("task.write(") < main.index("_score_vllm(")  # a script is written before the model is even loaded


# --- when a file cannot be had ------------------------------------------------------------------


def test_a_copy_that_changed_on_disk_is_fetched_again_for_an_official_run(monkeypatch):
    from stet import data

    blob = b"weights" * 100
    a = blob_asset("smol/model.safetensors", blob)
    assets.cache().mkdir(parents=True)
    assets.cached(a).write_bytes(b"x" * len(blob))  # the right length, the wrong bytes
    calls = []

    def fetch(urls, sha256, dest, **k):
        calls.append(dest)
        dest.write_bytes(blob)
        return dest

    monkeypatch.setattr(data, "fetch", fetch)
    assets.fetch([a], quiet=True)  # a practice run takes the cache's word
    assert not calls
    assets.fetch([a], quiet=True, verify=True)
    assert len(calls) == 1 and assets.cached(a).read_bytes() == blob
    assets.fetch([a], quiet=True, verify=True)  # and a good copy is left alone
    assert len(calls) == 1


def test_a_file_that_is_gone_is_the_entrys_failure_and_a_host_that_is_away_is_not(monkeypatch):
    import urllib.error

    from stet import data

    a = Asset("smol/model", URL, SHA, 1000)

    def failing(last):
        def fetch(urls, sha256, dest, **k):
            err = data.FetchError(f"could not fetch: {last}")
            err.last = last
            raise err

        return fetch

    for code in (401, 403, 404, 410):
        monkeypatch.setattr(data, "fetch", failing(urllib.error.HTTPError(URL, code, "no", None, None)))
        with pytest.raises(assets.Unavailable, match=f"no longer public at its address .*{code}"):
            assets.fetch([a], quiet=True)
    monkeypatch.setattr(data, "fetch", failing(data.Mismatch("sha256 mismatch")))
    with pytest.raises(assets.Unavailable, match="not the bytes its entry names"):
        assets.fetch([a], quiet=True)
    for last in (urllib.error.HTTPError(URL, 503, "busy", None, None), urllib.error.HTTPError(URL, 429, "slow down", None, None), TimeoutError("timed out")):
        monkeypatch.setattr(data, "fetch", failing(last))
        with pytest.raises(data.FetchError):
            assets.fetch([a], quiet=True)


def test_a_run_whose_asset_is_gone_is_a_failed_run_not_a_broken_machine(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # the pipeline trains, so it imports the trainer
    from stet import data, pipeline
    from stet.spec import practice_round

    season = load_season()
    monkeypatch.setattr(data, "corpus", lambda rnd, scale: type("C", (), {"total_bytes": 0, "__len__": lambda self: 0})())
    monkeypatch.setattr(assets, "fetch", lambda items, **k: (_ for _ in ()).throw(assets.Unavailable("the asset smol/model is no longer public at its address (the host answers 404)")))
    monkeypatch.setattr(pipeline, "run_editor", lambda *a, **k: pytest.fail("the curator was run without its files"))
    said = []
    run = pipeline.edit_round(tmp_path, "python curator.py", practice_round(season, 0), season.scale("nano"), tmp_path / "work", assets=[Asset("smol/model", URL, SHA, 5)], log=said.append)
    assert run.status == "error" and run.script is None and "no longer public" in run.reason and run.seconds == 0
    assert any("curator not run" in m for m in said)


def test_names_the_rules_do_not_allow_are_left_out_of_a_lookup(monkeypatch):
    tree = [
        {"type": "file", "path": "config.json", "size": 10},
        {"type": "file", "path": "my model.safetensors", "size": 10, "lfs": {"oid": SHA, "size": 10}},
        {"type": "file", "path": "sub/.hidden.json", "size": 10},
    ]
    monkeypatch.setattr(assets, "_get_json", lambda url: tree if "/tree/" in url else {"sha": COMMIT})
    _, found, skipped = assets.lookup("owner/name", include=["*"])
    assert [a.path for a in found] == ["name/config.json"] and set(skipped) == {"my model.safetensors", "sub/.hidden.json"}
    monkeypatch.setattr(assets, "_get_json", lambda url: [{"type": "file", "path": f"f{i}", "size": 1} for i in range(1000)] if "/tree/" in url else {"sha": COMMIT})
    with pytest.raises(AssetError, match="more than a thousand files"):
        assets.lookup("owner/name")


def test_small_files_are_hashed_as_they_arrive_and_kept(monkeypatch):
    from stet import data

    blob = b'{"hidden_size": 576}'

    def fetch(urls, sha256, dest, **k):
        dest.write_bytes(blob)
        return dest

    monkeypatch.setattr(data, "fetch", fetch)
    big = Asset("smol/model.safetensors", URL, SHA, 269060552)
    small = Asset("smol/config.json", URL.replace("model.safetensors", "config.json"), "", 704)
    out = assets.fill_hashes([big, small], quiet=True)
    assert out[0] == big and out[1].sha256 == hashlib.sha256(blob).hexdigest() and out[1].size == len(blob)
    assert assets.cached(out[1]).read_bytes() == blob and not [p for p in assets.cache().iterdir() if p.name.startswith(".new")]


def test_a_full_disk_is_this_machines_trouble_not_the_networks(monkeypatch):
    """A download that fails because the disk is full will fail the same way in a minute.
    Called the network's fault, the verifier would hold the job, and the round, for half
    an hour, and never notice that the trouble is its own."""
    import errno

    from stet import data

    a = Asset("smol/model", URL, SHA, 1000)

    def full(urls, sha256, dest, label="", quiet=False, max_bytes=0, gzipped=False):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(data, "fetch", full)
    with pytest.raises(OSError) as e:
        assets.fetch([a], quiet=True)
    assert not isinstance(e.value, data.FetchError) and e.value.errno == errno.ENOSPC

    def dropped(urls, sha256, dest, label="", quiet=False, max_bytes=0, gzipped=False):
        raise ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")

    monkeypatch.setattr(data, "fetch", dropped)  # the network's, as before
    with pytest.raises(data.FetchError):
        assets.fetch([a], quiet=True)


def test_what_a_killed_download_left_is_cleared_away(monkeypatch):
    """Killed outright in the middle of a 4 GiB file, a verifier leaves the part behind.
    Nothing else would remove it, or count it against the cache's limit."""
    from stet import data

    blob = b"weights" * 100
    a = blob_asset("smol/model.safetensors", blob)
    d = assets.cache()
    d.mkdir(parents=True, exist_ok=True)
    dead = d / f"{'ab' * 32}.part999999999"  # no process has this id
    mine = d / f"{'cd' * 32}.part{os.getpid()}"  # a download this process is making now
    fresh = d / f"{'ef' * 32}.part999999998"  # written a moment ago: its process may be starting
    for p in (dead, mine, fresh):
        p.write_bytes(b"half a model")
    old = time.time() - 2 * data.STALE_AFTER
    os.utime(dead, (old, old))
    os.utime(mine, (old, old))

    def fetch(urls, sha256, dest, label="", quiet=False, max_bytes=0, gzipped=False):
        dest.write_bytes(blob)
        return dest

    monkeypatch.setattr(data, "fetch", fetch)
    assets.fetch([a], quiet=True)
    assert not dead.exists() and mine.exists() and fresh.exists()
    assert assets.cached(a).read_bytes() == blob
