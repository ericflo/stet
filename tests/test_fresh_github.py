"""Fresh text from GH Archive: the window, who wrote it, what is prose, and how few of it any one
repository gives. Offline: the archive's hours are written here as gzipped JSON lines and read
from disk through the same code that reads them from the archive."""

import calendar
import gzip
import json
import re
import time
import zlib
from pathlib import Path

import pytest

from stet.newexam.fresh import github
from stet.newexam.rng import Rng

T0 = calendar.timegm((2026, 11, 19, 13, 0, 0))  # the window opens at 13:00 UTC
START, END = T0 + 1800, T0 + 5400  # 13:30 to 14:30: three hours of the archive cover it

WORDS = ["parser", "cache", "scheduler", "exporter", "importer", "renderer", "updater", "indexer", "uploader", "resolver"]


def prose(i: int, n: int = 4) -> str:
    """A few sentences of someone's own writing, different from anyone else's in every line."""
    w = WORDS[i % len(WORDS)]
    s = [f"The {w} in build {i} stops reading a file when the last line has no newline, so record {i} is lost without a warning.",
         f"I saw it on two machines after updating to version {i}.2, and going back to {i}.1 brings the old behaviour back.",
         f"It would help if the {w} said which file it gave up on, because with {i * 7} files the log does not make it clear.",
         f"Thanks for looking at this; I can test a fix on my side whenever one is ready for build {i}."]
    return "\n\n".join(s[:n])


def iso(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def user(login: str, kind: str = "User") -> dict:
    return {"login": login, "id": zlib.crc32(login.encode()), "type": kind}


def issue(n: int, repo: str, login: str, t: int, body: str, title: str = "", pr: bool = False, app=None, kind="User") -> dict:
    o = {"id": 7_000_000 + n, "number": n, "title": title or f"Problem number {n}", "user": user(login, kind), "body": body,
         "created_at": iso(t), "updated_at": iso(t), "html_url": f"https://github.com/{repo}/{'pull' if pr else 'issues'}/{n}",
         "performed_via_github_app": app}
    if pr:
        o["pull_request"] = {"url": f"https://api.github.com/repos/{repo}/pulls/{n}", "html_url": o["html_url"]}
    return o


def comment(cid: int, repo: str, n: int, login: str, t: int, body: str, kind="User") -> dict:
    return {"id": cid, "user": user(login, kind), "body": body, "created_at": iso(t), "updated_at": iso(t),
            "html_url": f"https://github.com/{repo}/issues/{n}#issuecomment-{cid}", "performed_via_github_app": None}


_ids = iter(range(50_000_000, 60_000_000))


def event(kind: str, t: int, repo: str, payload: dict, actor: str = "someone") -> dict:
    return {"id": str(next(_ids)), "type": kind, "actor": user(actor), "repo": {"id": zlib.crc32(repo.encode()), "name": repo},
            "payload": payload, "public": True, "created_at": iso(t)}


def opened(o: dict, repo: str, t: int | None = None, action: str = "opened") -> dict:
    return event("IssuesEvent", t if t is not None else github._ts(o["created_at"]), repo, {"action": action, "issue": o},
                 actor=o["user"]["login"])


def commented(c: dict, on: dict, repo: str) -> dict:
    return event("IssueCommentEvent", github._ts(c["created_at"]), repo, {"action": "created", "issue": on, "comment": c},
                 actor=c["user"]["login"])


def released(rid: int, repo: str, login: str, t: int, body: str, name: str = "", kind: str = "User", created: int | None = None) -> dict:
    r = {"id": rid, "tag_name": f"v{rid}", "name": name or f"Version {rid}", "author": user(login, kind), "body": body, "draft": False,
         "created_at": iso(created if created is not None else t), "published_at": iso(t),
         "html_url": f"https://github.com/{repo}/releases/tag/v{rid}"}
    return event("ReleaseEvent", t, repo, {"action": "published", "release": r}, actor=login)


def write_archive(root: Path, events: list[dict]) -> None:
    """The events as the archive keeps them: a gzipped file of JSON lines for each hour."""
    by_hour: dict[str, list[dict]] = {}
    for e in events:
        by_hour.setdefault(github.hour_urls(github._ts(e["created_at"]), github._ts(e["created_at"]) + 1)[0].rsplit("/", 1)[1], []).append(e)
    for name, evs in by_hour.items():
        with gzip.open(root / name, "wt") as f:
            f.writelines(json.dumps(e) + "\n" for e in evs)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """Point the source at an archive on disk, and name languages without the model."""
    monkeypatch.setattr(github, "BASE", tmp_path.as_uri())
    monkeypatch.setattr(github, "_lang", lambda text: "zh" if re.search(r"[一-鿿]", text) else "en")

    def run(events, cap=50, seed=b"seed"):
        write_archive(tmp_path, events)
        logs: list[str] = []
        docs = github.harvest(START, END, cap, Rng(seed, "fresh/github"), log=logs.append)
        return docs, logs

    return run


def test_the_archive_is_read_by_the_hours_that_cover_the_window():
    names = [u.rsplit("/", 1)[1] for u in github.hour_urls(START, END)]
    assert names == ["2026-11-19-13.json.gz", "2026-11-19-14.json.gz"]
    early = calendar.timegm((2026, 11, 3, 0, 0, 0))
    assert [u.rsplit("/", 1)[1] for u in github.hour_urls(early, early + 3600)] == ["2026-11-03-0.json.gz"]  # no leading zero
    assert github.hour_urls(START, END)[0].startswith("https://data.gharchive.org/")


def test_only_writing_made_in_the_window_counts(archive):
    old = issue(1, "a/old", "olga", START - 86400 * 30, prose(1))  # opened a month ago
    events = [
        opened(issue(2, "a/before", "bea", START - 60, prose(2)), "a/before"),  # a minute too early
        opened(issue(3, "a/inside", "ines", START + 60, prose(3)), "a/inside"),
        opened(issue(4, "a/end", "enzo", END, prose(4)), "a/end"),  # the window is [start, end)
        opened(old, "a/old", t=START + 120, action="labeled"),  # an old issue, labeled in the window: not new writing
        commented(comment(11, "a/old", 1, "carl", START + 300, prose(11)), old, "a/old"),  # a new reply to the old issue
        commented(comment(12, "a/old", 1, "dora", START - 10, prose(12)), old, "a/old"),
        opened(issue(5, "a/edited", "eve", START + 60, prose(5)), "a/edited", t=START + 90, action="edited"),
        released(21, "a/rel", "rita", START + 100, prose(21), created=START - 86400),  # its commit is older; its notes are new
    ]
    docs, _ = archive(events)
    got = {d.url: d for d in docs}
    assert sorted(got) == ["https://github.com/a/inside/issues/3", "https://github.com/a/old/issues/1#issuecomment-11",
                           "https://github.com/a/rel/releases/tag/v21"]
    reply = got["https://github.com/a/old/issues/1#issuecomment-11"]
    assert reply.source == "github-comment" and reply.question == prose(1) and reply.title == "Problem number 1"
    assert reply.text == prose(11) and reply.published == START + 300 and reply.extra["kind"] == "issue"
    new = got["https://github.com/a/inside/issues/3"]
    assert new.source == "github" and new.extra == {"kind": "issue", "repo": "a/inside", "event": new.extra["event"]}
    assert new.id == "issue:7000003" and new.lang == "en" and new.published == START + 60 and "GH Archive" in new.license
    assert got["https://github.com/a/rel/releases/tag/v21"].extra["kind"] == "release"


def test_bots_apps_and_agents_are_left_out(archive):
    t = START + 100
    events = [
        opened(issue(1, "b/one", "dependabot", t, prose(1)), "b/one"),
        opened(issue(2, "b/two", "renovate[bot]", t, prose(2)), "b/two"),
        opened(issue(3, "b/three", "helper", t, prose(3), kind="Bot"), "b/three"),
        opened(issue(4, "b/four", "acme-ci-bot", t, prose(4)), "b/four"),
        opened(issue(5, "b/five", "pat", t, prose(5), app={"slug": "claude"}), "b/five"),  # posted by an app for pat
        opened(issue(6, "b/six", "sam", t, prose(6) + "\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"), "b/six"),
        opened(issue(7, "b/seven", "kim", t, prose(7) + "\n\nThis issue has been automatically marked as stale."), "b/seven"),
        released(8, "b/eight", "github-actions[bot]", t, prose(8), kind="Bot"),
        opened(issue(9, "b/nine", "lee", t, prose(9)), "b/nine"),  # a person
    ]
    on = issue(10, "b/ten", "lee", t, prose(10))
    events += [commented(comment(31, "b/ten", 10, "codecov[bot]", t + 5, prose(31), kind="Bot"), on, "b/ten"),
               commented(comment(32, "b/ten", 10, "max", t + 6, "## [Codecov](https://codecov.io) Report\n" + prose(32)), on, "b/ten")]
    docs, logs = archive(events)
    assert sorted(d.url for d in docs if d.source == "github") == ["https://github.com/b/nine/issues/9", "https://github.com/b/ten/issues/10"]
    assert not [d for d in docs if d.source == "github-comment"]
    assert "'app': 1" in logs[-1] and "'bot': 6" in logs[-1] and "'machine': 3" in logs[-1]


TEMPLATE = """<!-- Thanks for reporting! Please fill in every section below. -->
### Describe the bug

A clear and concise description of what the bug is.

### Steps to reproduce

1. Go to the settings page of the application
2. Click on the button that saves the form

### Checklist

- [ ] I have searched the existing issues for this problem
- [x] I am using the latest version of the application
"""


def test_templates_code_quotes_and_generated_changelogs_are_not_prose(archive):
    t = START + 200
    events = [opened(issue(i, f"c/r{i}", f"user{i}", t, TEMPLATE), f"c/r{i}") for i in range(1, 5)]  # untouched templates
    filled = TEMPLATE.replace("A clear and concise description of what the bug is.", prose(5))
    events.append(opened(issue(5, "c/r5", "user5", t, filled), "c/r5"))
    code = "It fails here:\n\n```python\n" + "def f(x):\n    return x + 1\n" * 40 + "```\n\nSee https://example.org/log.txt"
    events.append(opened(issue(6, "c/r6", "user6", t, code), "c/r6"))
    quoted = ("> " + prose(70).replace("\n\n", "\n> ") + "\n\n<!-- a hidden note that nobody sees on the page -->\n" + prose(7)
              + "\n\n![screenshot](https://example.org/a.png) See [the docs](https://example.org/docs) and `config.yaml`.")
    events.append(opened(issue(7, "c/r7", "user7", t, quoted), "c/r7"))
    changelog = ("## What's Changed\n" + "".join(f"* Fix the {w} by @dev{i} in https://github.com/c/r8/pull/{i}\n" for i, w in enumerate(WORDS))
                 + "\n## New Contributors\n* @dev3 made their first contribution in https://github.com/c/r8/pull/3\n\n"
                 "**Full Changelog**: https://github.com/c/r8/compare/v1...v2")
    events.append(released(8, "c/r8", "rel", t, changelog))
    events.append(released(9, "c/r9", "rel9", t, "### Features\n\n" + "".join(f"* **{w}:** handle empty input ([#{i}](https://github.com/c/r9/issues/{i})) ([{i:07x}a](https://github.com/c/r9/commit/{i:07x}a))\n" for i, w in enumerate(WORDS))))
    docs, logs = archive(events)
    got = {d.extra["repo"].split("/")[1]: d for d in docs}
    assert sorted(got) == ["r5", "r7"], sorted(got)
    assert got["r5"].text == prose(5)  # what was left of the template was shared by five issues: none of it was anyone's
    text = got["r7"].text
    assert text.startswith(prose(7)) and text.endswith("See the docs and config.yaml.")
    for gone in ("> ", "hidden note", "http", "screenshot", "build 70"):
        assert gone not in text
    assert "'short': 7" in logs[-1]  # four templates, the code, two changelogs


def test_prose_lines():
    body = ("Intro line with `a <b> c` inline and a [link](https://x.org/a_(b)).\r\n\r\n"
            "```\ncode line one;\n```\n"
            "    indented_code = 1\n"
            "| a | b |\n|---|---|\n"
            "**Environment**\n"
            "_No response_\n"
            "- a real list item about the thing\n"
            "~~~\nunclosed fence swallows the rest\n")
    assert github.prose_lines(body) == ["Intro line with a <b> c inline and a link.", "", "- a real list item about the thing"]


def test_a_pull_requests_description_comes_from_a_comment_on_it_and_is_kept_once(archive):
    t = START + 400
    pr = issue(40, "d/pr", "pia", t, prose(40), title="Read the last line without a newline", pr=True)
    events = [commented(comment(41, "d/pr", 40, "rev", t + 50, prose(41)), pr, "d/pr")]
    # the archive before October 2025: the event that opens a pull request carries it whole, under its own id
    whole = {**{k: v for k, v in pr.items() if k != "pull_request"}, "id": 123456789, "base": {"repo": {"language": "Rust"}}}
    events.append(event("PullRequestEvent", t, "d/pr", {"action": "opened", "number": 40, "pull_request": whole}, actor="pia"))
    # since then it carries no title, no description and no address
    events.append(event("PullRequestEvent", t, "d/new", {"action": "opened", "number": 9, "pull_request": {"id": 5, "number": 9}}))
    hunk = "@@ -1,3 +1,3 @@\n-old line\n+new line"
    review = {"id": 77, "user": user("rev"), "body": prose(77), "created_at": iso(t + 60), "path": "src/read.rs", "diff_hunk": hunk,
              "html_url": "https://github.com/d/pr/pull/40#discussion_r77"}
    events.append(event("PullRequestReviewCommentEvent", t + 60, "d/pr", {"action": "created", "comment": review,
                                                                         "pull_request": {"id": 123456789, "number": 40}}))
    docs, _ = archive(events)
    pulls = [d for d in docs if d.source == "github"]
    assert len(pulls) == 1 and pulls[0].extra["kind"] == "pull" and pulls[0].title == "Read the last line without a newline"
    assert pulls[0].text == prose(40) and pulls[0].url == "https://github.com/d/pr/pull/40" and pulls[0].topic == "Rust"
    replies = {d.extra["kind"]: d for d in docs if d.source == "github-comment"}
    assert replies["pull"].question == prose(40) and replies["pull"].title == pr["title"]
    assert replies["review"].question == f"src/read.rs\n{hunk}" and replies["review"].id == "review-comment:77"


def test_a_repository_or_an_author_gives_only_a_few(archive):
    t = START + 500
    events = [opened(issue(100 + i, "e/busy", f"person{i}", t + i, prose(100 + i)), "e/busy") for i in range(12)]
    events += [opened(issue(200 + i, f"e/repo{i}", "prolific", t + i, prose(200 + i)), f"e/repo{i}") for i in range(12)]
    events += [opened(issue(300 + i, f"f/repo{i}", f"solo{i}", t + i, prose(300 + i)), f"f/repo{i}") for i in range(30)]
    docs, logs = archive(events, cap=1000)
    assert sum(d.extra["repo"] == "e/busy" for d in docs) == github.PER_REPO
    assert sum(d.url.startswith("https://github.com/e/repo") for d in docs) == github.PER_AUTHOR
    assert len(docs) == 30 + github.PER_REPO + github.PER_AUTHOR
    assert "  github technical: 54 written in the window, 54 by people in prose, 36 kept" in logs


def test_the_choice_is_the_seeds(archive):
    t = START + 600
    events = [opened(issue(400 + i, f"g/repo{i}", f"author{i}", t + i, prose(400 + i)), f"g/repo{i}") for i in range(40)]
    on = issue(999, "g/thread", "op", t, prose(999))
    events += [commented(comment(5000 + i, f"g/c{i}", 999, f"replier{i}", t + i, prose(5000 + i)), {**on, "id": 8000 + i}, f"g/c{i}")
               for i in range(40)]
    a, _ = archive(events, cap=10, seed=b"one")
    b, _ = archive(events, cap=10, seed=b"one")
    c, _ = archive(events, cap=10, seed=b"two")
    assert a == b and a != c
    for docs in (a, c):
        assert sum(d.source == "github" for d in docs) == 10 and sum(d.source == "github-comment" for d in docs) == 10


def test_an_hour_missing_from_the_archive_loses_only_itself(archive):
    docs, logs = archive([opened(issue(1, "h/one", "hana", START + 60, prose(1)), "h/one")])  # nothing at 14:00
    assert len(docs) == 1 and any("2026-11-19-14.json.gz is not in the archive" in x for x in logs)


def test_languages_are_named_by_lid176(monkeypatch):
    """The language is what lid.176 (langs.model) reads at the start of the text, as its ISO code."""
    from stet.newexam.fresh import langs

    seen = []

    class Model:
        def predict(self, text, k=1):
            seen.append(text)
            return ("__label__zh",), [0.98]

    monkeypatch.setattr(langs, "model", lambda: Model())
    assert github._lang("第一行\n第二行" + "字" * 2000) == "zh"
    assert seen[0].startswith("第一行 第二行") and len(seen[0]) == 1000  # one line, as fastText wants it
