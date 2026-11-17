"""Bundles: what goes into one, and what is allowed to come out of one somebody else made."""

import hashlib
import io
import os
import tarfile

import pytest
import zstandard

from stet import project
from stet.project import ProjectError, load_project, pack, survey, unpack


def make(tmp_path, files: dict[str, str]):
    root = tmp_path / "proj"
    for name, text in {"stet.toml": 'name = "an-editor"\nentry = "python editor.py"\n', "editor.py": "print('hi')\n", **files}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return load_project(root)


def bundle(members: list[tarfile.TarInfo | tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for m in members:
            if isinstance(m, tuple):
                info = tarfile.TarInfo(m[0])
                info.size = len(m[1])
                tar.addfile(info, io.BytesIO(m[1]))
            else:
                tar.addfile(m)
    return zstandard.ZstdCompressor().compress(buf.getvalue())


def test_a_bundle_holds_the_project_and_not_what_lives_beside_it(tmp_path):
    p = make(tmp_path, {
        ".env": "API_KEY=secret", ".git/config": "[core]", ".claude/notes.md": "x", ".gitignore": ".stet/\n",
        "runs/best/model.pt": "w", "venv/lib/x.py": "x", "pkg/__pycache__/m.pyc": "x", "pkg/m.py": "x = 1\n", "weights/model.bin": "w",
    })  # fmt: skip
    os.symlink(p.path / "editor.py", p.path / "link.py")
    files, skipped = survey(p)
    assert [f.relative_to(p.path).as_posix() for f in files] == ["editor.py", "pkg/m.py", "stet.toml", "weights/model.bin"]
    left = dict(skipped)
    assert set(left) == {".env", ".git/", ".claude/", ".gitignore", "runs/", "venv/", "pkg/__pycache__/", "link.py"}
    assert "dot" in left[".env"] and "link" in left["link.py"]
    blob, editor_id = pack(p)
    assert pack(p) == (blob, editor_id)  # same files in, same bytes out
    assert unpack(blob, tmp_path / "out") == []
    assert sorted(f.relative_to(tmp_path / "out").as_posix() for f in (tmp_path / "out").rglob("*") if f.is_file()) == ["editor.py", "pkg/m.py", "stet.toml", "weights/model.bin"]


def test_a_bundle_that_is_too_big_says_what_can_be_done_about_it(tmp_path, monkeypatch):
    p = make(tmp_path, {"blob.bin": "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(400))})  # does not compress much
    monkeypatch.setattr(project, "MAX_BUNDLE_BYTES", 2000)
    with pytest.raises(ProjectError) as e:
        pack(p)
    assert "stet asset add" in str(e.value) and "fetched before the editor starts" in str(e.value)
    monkeypatch.setattr(project, "MAX_BUNDLE_FILES", 2)
    with pytest.raises(ProjectError, match="at most 2"):
        pack(p)


def test_unpacking_stops_at_the_limit_whatever_the_bundle_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "MAX_UNPACKED_BYTES", 1 << 20)
    bomb = zstandard.ZstdCompressor().compress(b"\0" * (8 << 20))  # the frame declares its size; a few hundred bytes
    assert len(bomb) < 2000
    with pytest.raises(ProjectError, match="unpacks to more"):
        unpack(bomb, tmp_path / "a")
    # A member can claim any length without holding it: a sparse file is all holes.
    sparse = tarfile.TarInfo("big")
    sparse.pax_headers = {"GNU.sparse.major": "0", "GNU.sparse.minor": "1", "GNU.sparse.map": "0,0", "GNU.sparse.size": str(1 << 40), "GNU.sparse.name": "big"}
    with pytest.raises(ProjectError, match="not a plain file"):
        unpack(bundle([sparse]), tmp_path / "b")
    assert not (tmp_path / "b" / "big").exists()
    monkeypatch.setattr(project, "MAX_BUNDLE_FILES", 3)
    with pytest.raises(ProjectError, match="more than 3 files"):
        unpack(bundle([(f"f{i}", b"x") for i in range(4)]), tmp_path / "c")


def test_unpacking_writes_only_plain_files_inside_the_destination(tmp_path):
    link = tarfile.TarInfo("evil")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    for n, members in enumerate(([link], [("../outside", b"x")], [("/abs/olute", b"x")], [("a/../../outside", b"x")])):
        with pytest.raises(ProjectError):
            unpack(bundle(members), tmp_path / f"d{n}")
    assert not (tmp_path / "outside").exists()
    for junk in (b"not zstd", zstandard.ZstdCompressor().compress(b"not a tar" * 100)):
        with pytest.raises(ProjectError):
            unpack(junk, tmp_path / "junk")


def test_unpacking_leaves_out_what_no_bundle_should_hold(tmp_path):
    blob = bundle([("stet.toml", b'name = "x-y"\nentry = "python e.py"\n'), (".git/config", b"[core]\nfsmonitor = rm -rf ~"), (".envrc", b"x"), ("sub/.vscode/tasks.json", b"{}"), ("e.py", b"")])
    assert unpack(blob, tmp_path / "f") == [".git/config", ".envrc", "sub/.vscode/tasks.json"]
    assert sorted(p.name for p in (tmp_path / "f").iterdir()) == ["e.py", "stet.toml"]


def test_a_manifest_that_cannot_be_read_is_a_project_error(tmp_path):
    for n, text in enumerate(("x = " + "[" * 5000 + "]" * 5000, "name = \x00", "not toml at all [")):
        root = tmp_path / f"p{n}"
        root.mkdir()
        (root / "stet.toml").write_text(text)
        with pytest.raises(ProjectError):
            load_project(root)


# What the server would refuse, or would read differently from this client. Each of these
# was put to the server's own TOML decoder: see the note in the test below.
OK = 'name = "an-editor"\nentry = "python editor.py"\n'
NOT_AS_THE_SERVER_READS_IT = {
    # An integer of any size is an integer to Python. The server reads 64 bits.
    "does not fit in 64 bits": [OK + "budget = 9223372036854775808\n", OK + "[tool.mine]\nseeds = [1, [2, {n = -9223372036854775809}]]\n", OK + "mask = 0xFFFFFFFFFFFFFFFFFF\n"],
    "not valid TOML": [OK + "big = " + "9" * 5000 + "\n", OK + "rate = 1e400\n", OK + "rate = -1.5E+999\n"],
    # The server matches a field's name whatever its case. `Name` beside `name` is two
    # answers to one question, and which the server takes is not this client's to say.
    "'Name' must be written name": [OK + 'Name = "another-editor"\n'],
    "'ENTRY' must be written entry": [OK + 'ENTRY = "sh -c anything"\n', 'name = "an-editor"\nENTRY = "python editor.py"\n'],
    "'Assets' must be written assets": [OK + '[[Assets]]\nurl = "http://x.test/model.bin"\n'],
    "must be written license": [OK + 'LICENSE = "proprietary"\n', OK + '"licen\u017fe" = "x"\n'],
    r"must be written as \[\[assets\]\] tables": [OK + "assets = 5\n", OK + 'assets = "none"\n', OK + "assets = false\n", OK + "assets = {}\n", OK + "assets = [1]\n"],
    "asset 1: missing path": [OK + '[[assets]]\nurl = "x"\n'],
}
AS_THE_SERVER_READS_IT = [
    OK + "budget = 9223372036854775807\nfloor = -9223372036854775808\nrate = 1e308\nnone = inf\nodd = nan\nflag = true\n",
    OK + "assets = []\n",
    OK + 'names = ["Name", "ENTRY"]\n[tool]\nName = "inside a table of its own, nobody matches it against anything"\n',
    OK + 'nickname = "x"\nentry_point = "y"\n',
]


def test_the_client_reads_a_manifest_as_the_server_does(tmp_path):
    # Checked against github.com/BurntSushi/toml v1.6.0 decoding into the server's Manifest
    # struct: every entry of NOT_AS_THE_SERVER_READS_IT is one it either refuses or takes a
    # field from that this client would not, and every entry of AS_THE_SERVER_READS_IT is
    # one it accepts with the same name and entry.
    n = 0
    for why, manifests in NOT_AS_THE_SERVER_READS_IT.items():
        for text in manifests:
            n += 1
            root = tmp_path / f"m{n}"
            root.mkdir()
            (root / "stet.toml").write_text(text, encoding="utf-8")
            with pytest.raises(ProjectError, match=why):
                load_project(root)
    for text in AS_THE_SERVER_READS_IT:
        n += 1
        root = tmp_path / f"m{n}"
        root.mkdir()
        (root / "stet.toml").write_text(text, encoding="utf-8")
        p = load_project(root)
        assert (p.name, p.entry) == ("an-editor", "python editor.py")


@pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="POSIX only")
def test_a_manifest_that_is_a_link_is_refused_before_the_server_misses_it(tmp_path):
    # Links are left out of a bundle. A project whose stet.toml is one would pass every
    # local check and arrive at the server with no manifest at all.
    real = tmp_path / "shared.toml"
    real.write_text(OK)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "editor.py").write_text("print('hi')\n")
    os.symlink(real, root / "stet.toml")
    with pytest.raises(ProjectError, match="stet.toml is a symbolic link"):
        load_project(root)
    (root / "stet.toml").unlink()
    os.symlink(tmp_path / "nowhere.toml", root / "stet.toml")  # and one that points nowhere is a link first
    with pytest.raises(ProjectError, match="symbolic link"):
        load_project(root)
    (root / "stet.toml").unlink()
    (root / "stet.toml").mkdir()
    with pytest.raises(ProjectError, match="not a plain file"):
        load_project(root)
    (root / "stet.toml").rmdir()
    (root / "stet.toml").write_text(OK)
    p = load_project(root)
    assert "stet.toml" in [f.name for f in survey(p)[0]] and unpack(pack(p)[0], tmp_path / "out") == []


def test_a_bundle_names_each_file_once_and_plainly(tmp_path):
    manifest = ("stet.toml", b'name = "x-y"\nentry = "python editor.py"\n')
    harmless, real = ("editor.py", b"print('nothing to see')\n"), ("editor.py", b"print('what runs')\n")
    # Of two files with one name the anchor would run the last, and the site shows the first.
    with pytest.raises(ProjectError, match="twice"):
        unpack(bundle([manifest, harmless, real]), tmp_path / "a")
    for n, roundabout in enumerate(("./editor.py", "x/../editor.py", "pkg//m.py", "pkg/./m.py")):
        with pytest.raises(ProjectError, match="twice, or by a roundabout path"):
            unpack(bundle([manifest, harmless, (roundabout, b"")]), tmp_path / f"b{n}")
    with pytest.raises(ProjectError, match="a file and for a directory"):
        unpack(bundle([manifest, ("pkg", b"x"), ("pkg/m.py", b"")]), tmp_path / "c")


def test_every_frame_of_a_download_is_read_and_a_damaged_one_is_an_error(tmp_path):
    from stet.project import unzstd

    z = zstandard.ZstdCompressor()
    two = z.compress(b"first\n") + z.compress(b"second\n")  # what pzstd writes, or two files joined
    assert unzstd(two, 1000) == b"first\nsecond\n"
    for damaged in (two[:-3], two + b"trailing junk", b"not zstd"):
        with pytest.raises(zstandard.ZstdError):
            unzstd(damaged, 1000)
    with pytest.raises(ProjectError, match="zstd"):
        unpack(two[:-3], tmp_path / "x")


def test_the_dry_run_refuses_what_the_server_would(tmp_path, monkeypatch):
    def project_with(n, toml):
        root = tmp_path / f"m{n}"
        root.mkdir()
        (root / "stet.toml").write_text(toml)
        return root

    ok = 'name = "an-editor"\nentry = "python editor.py"\n'
    load_project(project_with(0, ok + 'description = "café, 中文 and ordinary punctuation."\n'))
    refused = {
        "one command line": 'name = "an-editor"\nentry = "python editor.py ' + "x" * 200 + '"\n',
        "one command line of": 'name = "an-editor"\nentry = "python a.py\\npython b.py"\n',
        "ordinary text": ok + 'description = "nice \\u001b]0;owned\\u0007 editor"\n',
        "ordinary text each": ok + 'technique = "two\\nlines"\n',
        "must be text": 'name = 123\nentry = "python editor.py"\n',
        "entry must be text": 'name = "an-editor"\nentry = ["python", "editor.py"]\n',
        "over 64 KB": ok + "# " + "x" * (65 << 10) + "\n",
    }
    for n, (why, toml) in enumerate(refused.items(), 1):
        with pytest.raises(ProjectError, match=why):
            load_project(project_with(n, toml))

    # A plain file may be called what a left-out directory is called. The server takes it,
    # so the client packs it, and an anchor unpacks it.
    p = make(tmp_path, {"runs": "a file, not a directory", "data/venv": "likewise", "runs2/x": "x"})
    files, skipped = survey(p)
    assert [f.relative_to(p.path).as_posix() for f in files] == ["data/venv", "editor.py", "runs", "runs2/x", "stet.toml"] and not skipped
    assert unpack(pack(p)[0], tmp_path / "out") == [] and (tmp_path / "out" / "runs").is_file()

    # Paths: each is bounded, and so is the list of them.
    (p.path / ("n" * 250 + ".txt")).write_text("x")
    pack(p)
    deep = p.path / "a" / ("n" * 250 + ".txt")
    deep.parent.mkdir()
    deep.write_text("x")
    with pytest.raises(ProjectError, match="at most 255 bytes"):
        pack(p)
    deep.unlink()
    monkeypatch.setattr(project, "MAX_PATHS_BYTES", 100)
    with pytest.raises(ProjectError, match="paths of the"):
        pack(p)
    monkeypatch.undo()

    # The size limit is on the tar, which is what the server and the anchor unpack: a
    # header and padding for every file, on top of the files.
    q = make(tmp_path / "q", {f"f{i}": "x" for i in range(20)})
    monkeypatch.setattr(project, "MAX_UNPACKED_BYTES", 8 << 10)  # the files are a few hundred bytes; their tar is over 20 KB
    with pytest.raises(ProjectError, match="before compression"):
        pack(q)
