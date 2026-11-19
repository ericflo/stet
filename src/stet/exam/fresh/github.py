"""Writing on GitHub, from GH Archive's hourly record of public events, written in the window.

Technical writing (source "github"): issues, pull requests' descriptions and release notes.
Discussion (source "github-comment"): comments on issues and pull requests, and on lines of a pull
request's diff, each kept with what it answers: the issue's or pull request's description, or the
lines of the diff. What a comment answers may be older; only the comment is scored, and the
comment is new.

GH Archive (https://www.gharchive.org) keeps GitHub's public event stream as one gzipped file of
JSON lines an hour, open to all without a key. Since October 2025 GitHub leaves a pull request's
title and description out of the event that opens it, so a description is taken from where the
stream still carries it whole: a comment on the pull request, which carries it as an issue.

Only people's own prose is kept. Bots, apps and the agents that sign their work go, and so do
code, logs, links, quoted lines, checklists, tables, headings, HTML comments, templates and
generated changelogs; what is left must be long enough to be someone's writing. A repository, and
an author, gives at most a few documents of each kind.
"""

from __future__ import annotations

import calendar
import gzip
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter

from . import GAP, UA, Doc

BASE = "https://data.gharchive.org"
# GitHub's terms let other users view and use public content through GitHub itself; they grant no
# licence to copy it elsewhere, and GH Archive adds none: the writing stays its authors'.
LICENSE = "GitHub user content, public; no licence granted beyond GitHub's Terms of Service; via GH Archive"
MIN_CHARS = 300  # of prose, once everything that is not the author's own prose is taken out
PER_REPO = 3  # documents of a kind from any one repository
PER_AUTHOR = 3  # and from any one author
COMMON = 3  # a line found in this many documents is a template's, not anyone's own
COMMON_CHARS = 20  # and lines shorter than this are too ordinary to tell
QUESTION_CHARS = 6000  # of what a comment answers, kept as it was given
TYPES = (b'"IssuesEvent"', b'"IssueCommentEvent"', b'"PullRequestEvent"', b'"PullRequestReviewCommentEvent"', b'"ReleaseEvent"')

# --- who wrote it ---------------------------------------------------------------------------------

# Accounts that automation runs as without GitHub's [bot] mark: CI, release tools, CLA checkers,
# merge queues, mirrors. ("ghost" is where GitHub puts a deleted account's writing: no one's now.)
BOTS = {
    "dependabot", "dependabot-preview", "renovate", "renovate-bot", "github-actions", "codecov", "codecov-io", "codecov-commenter",
    "coveralls", "sonarcloud", "sonarqubecloud", "netlify", "vercel", "stale", "allcontributors", "all-contributors", "greenkeeper",
    "greenkeeperio-bot", "snyk-bot", "pre-commit-ci", "mergify", "imgbot", "whitesource-bolt-for-github", "mend-bolt-for-github",
    "deepsource-autofix", "deepsource-io", "codeclimate", "lgtm-com", "google-cla", "googlebot", "copybara-service", "bors",
    "rustbot", "rust-highfive", "rust-log-analyzer", "rust-timer", "pytorchmergebot", "facebook-github-bot", "azure-pipelines",
    "azure-sdk", "microsoft-github-policy-service", "linux-foundation-easycla", "cla-assistant", "claassistantio", "changeset-bot",
    "release-please", "semantic-release-bot", "gitguardian", "socket-security", "coderabbitai", "copilot", "copilot-swe-agent",
    "copilot-pull-request-reviewer", "sourcery-ai", "sweep-ai", "codiumai-pr-agent", "qodo-merge-pro", "ellipsis-dev",
    "graphite-app", "chatgpt-codex-connector", "gemini-code-assist", "devin-ai-integration", "google-labs-jules", "openhands-agent",
    "cloudflare-workers-and-pages", "railway-app", "nx-cloud", "crowdin-bot", "transifex-integration", "restyled-io", "kodiakhq",
    "boring-cyborg", "pkg-pr-new", "sizebot", "github-advanced-security", "ghost",
}
_BOTNAME = re.compile(r"(?:^|[-_.])(?:bot|bots|robot|automation|autobot|buildbot|agent)(?:$|[-_.\d])", re.I)


def is_bot(user: dict | None) -> bool:
    """A login GitHub marks as a bot or an app, or that names itself one."""
    if not user:
        return True
    login = (user.get("login") or "").lower()
    return (not login or user.get("type") in ("Bot", "Organization") or login.endswith("[bot]") or login in BOTS
            or login.removesuffix("-bot") in BOTS or bool(_BOTNAME.search(login)))


# What bots, CI, deploy previews, coverage reports, stale-markers, CLA checks and the writing agents
# that sign their work leave in what they post, whoever's account it is posted from.
_MACHINE = re.compile(
    r"generated (?:with|by|using) \[?(?:claude|copilot|github copilot|cursor|codex|devin|jules|chatgpt|gpt-?\d|gemini|openhands|"
    r"aider|windsurf|an? (?:ai|llm)|ai\b)|co-authored-by: (?:claude|copilot|cursor|codex|devin|gemini|openhands|aider)|claude\.ai/code|"
    r"summary by coderabbit|coderabbit\.ai|chatgpt\.com/codex|cursor\.com/(?:agents|background)|devin\.ai|jules\.google|"
    r"\bcodecov\b|coverage (?:report|summary|diff)\b|coverage (?:increased|decreased|remained)|deploy preview|preview deployment|"
    r"successfully deployed|deployment (?:preview|status|ready)|the latest updates on your projects|\bsonar(?:cloud|qube)\b|"
    r"quality gate (?:passed|failed)|this (?:issue|pull request|pr|discussion) (?:has been|was|is being) (?:automatically|auto)|"
    r"automatically (?:marked|closed|created|opened) (?:as|by|because|this)|will be closed (?:in \d+ days|if no)|"
    r"(?:auto-?generated|automatically generated) (?:by|with|using|from|release notes|changelog|comment|summary|description)|"
    r"\[//\]: # \(|^/(?:assign|label|close|lgtm|approve|retest|ok-to-test|hold|kind|area|cc)\b|i have read the cla|"
    r"cla (?:assistant|signature)|all committers have signed|beep boop|i am a bot\b|i'm a bot\b|"
    r"this (?:comment|issue|pr|pull request|description|summary) was (?:generated|posted|written|created) (?:by|with|using)|"
    r"^\W*(?:model|reviewer|agent|written by|posted by|author)\W*:.*\b(?:claude|gpt|codex|gemini|deepseek|opus|sonnet|haiku|llama|"
    r"qwen|mistral|grok|kimi|glm|copilot)\b|\bas an ai\b|\bi(?:'m| am) (?:claude|codex|copilot|an ai)\b|"
    r"\b(?:run|written|drafted|posted|generated|reviewed|authored|prepared|made) by [^.\n]{0,40}\b(?:claude|gpt-?\d|codex|gemini|"
    r"copilot|deepseek|opus|sonnet|haiku|llama|qwen|grok|kimi|glm)\b|\bai acting on behalf of\b|\(\s*[\w.-]+'s (?:ai |coding )?agent\s*\)|"
    r"^\W*bot\s*:",
    re.I | re.M)

# --- what is prose ----------------------------------------------------------------------------------

_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
_DETAILS = re.compile(r"<details\b.*?(?:</details>|\Z)", re.S | re.I)  # folded away: logs, environments, screenshots
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})|^\s*(\{(?:code|noformat)(?::[^}]*)?\})\s*$")  # Markdown's, and Jira's
_DROP = [  # a line that is not prose, by its look before any markup is taken out
    re.compile(r"^\s{0,3}>"),  # quoted
    re.compile(r"^\s*(?:[-*+]|\d+[.)])?\s*\[[ xX]?\]\s"),  # a checklist
    re.compile(r"^\s*\||\|\s*$|^\s*:?-{3,}:?\s*\|"),  # a table
    re.compile(r"^\s{0,3}#{1,6}(?:\s|$)|^\s{0,3}(?:=+|-+)\s*$|^\s*(\*\*|__)[^*_]{1,80}\1:?\s*$"),  # a heading, or bold standing in for one
    re.compile(r"^\s*(?:[-*_=]\s*){3,}$"),  # a rule
    re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S"),  # a link's definition
    # a generated changelog: pull requests by their authors, commits by their hashes, dependency bumps
    re.compile(r"\bby @[\w.-]+ in (?:https?://\S+|#\d+)|\*\*full changelog\*\*|made (?:their|his|her|a) first contribution|"
               r"\(\[?[0-9a-f]{7,40}\]?(?:\([^)]*\))?\)|\[`?[0-9a-f]{7,40}`?\]\(|^\s*[-*]\s+[0-9a-f]{7,40}\b|"
               r"^\s*[-*]\s+(?:build|chore|ci)\(deps[^)]*\)|^\s*[-*]\s+bump \S+ from \S+ to \S+", re.I),
    # a log or a console: stack frames, timestamps, levels, prompts, paths, JSON
    re.compile(r"^\s*at [\w$.<>/]+[.(].*\)\s*$|^\s*File \".*\", line \d+|^\s*Traceback \(most recent call last\)|"
               r"^\s*\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|^\s*\[?(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL|PANIC)\]?[:\] ]|"
               r"^\s*[$%#] \S|^\s*(?:PS )?[A-Z]:\\|^\s*\w+@[\w.-]+:[~/]|^\s*\"[^\"]*\"\s*(?::|,?\s*$)"),
]
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]|<(?:img|picture|video|source|audio)\b[^>]*>", re.I)
_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()\s]|\([^)]*\))*(?:\s+\"[^\"]*\")?\)")
_REFLINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
_INLINE = re.compile(r"(`+)(.+?)\1")
_URL = re.compile(r"<?\b(?:https?|ftp)://[^\s<>]+>?|\bwww\.[^\s<>]+")
_TAG = re.compile(r"</?(?:details|summary|br|p|div|span|a|sub|sup|b|i|u|s|strong|em|kbd|code|pre|samp|var|del|ins|mark|small|big|"
                  r"font|center|table|thead|tbody|tr|td|th|ul|ol|li|h[1-6]|hr|blockquote|input|dl|dt|dd|section|article|header|"
                  r"footer|figure|figcaption|abbr|cite|q|tt|strike|nobr|wbr)\b[^>]*>", re.I)
_NO_ANSWER = re.compile(r"^\s*[_*]*(?:no response|n/?a|none|nothing|todo|tbd|-)[_*.]*\s*$", re.I)  # an issue form's empty field
_LETTERS = re.compile(r"[^\W\d_]")
# A line that is a statement of code, not a sentence about one: short, and shaped like code.
_CODE = re.compile(r"[{};]\s*$|^\s*[}\])]|^\s*(?:def|class|import|from\s+\S+\s+import|return|const|let|var|function|public|private|"
                   r"protected|package|#include|#define|func|fn|use|struct|impl|SELECT|INSERT|UPDATE|CREATE|if|for|while)\b.*[(:{=]|"
                   r"^\s*[\w.\[\]\"'$]+\s*=\s*\S")
CODE_LINE = 120  # characters: a longer line is a sentence
CODE_SHARE = 0.3  # of a text's lines, shaped like code, and the text is code


def _markup(s: str) -> str:
    """Markup out of a stretch of a line that is not inline code: links to their words, no URLs,
    no HTML tags, no emphasis, and nothing left of a link but its words."""
    s = _URL.sub("", _TAG.sub("", _REFLINK.sub(r"\1", s)))
    return re.sub(r"\*\*|__|~~|`", "", re.sub(r"\s*[(\[<]\s*[)\]>]", "", s))


def _code_line(line: str) -> bool:
    s = line.strip()
    if re.match(r"(?:\t| {4,})", line) and not re.match(r"(?:[-*+]|\d+[.)])\s", s):  # an indented code block
        return True
    visible = len(s.replace(" ", ""))
    return visible >= 8 and len(_LETTERS.findall(s)) < 0.5 * visible  # more symbols and digits than letters


def _closes(fence: str, m: re.Match, raw: str) -> bool:
    if fence.startswith("{"):  # Jira's {code} closes with {code}
        return bool(m.group(2))
    f = m.group(1) or ""  # Markdown's with as many of the same mark or more, and nothing after them
    return bool(f) and f[0] == fence[0] and len(f) >= len(fence) and not raw.strip()[len(f):].strip()


def prose_lines(body: str) -> list[str]:
    """What of a post is its author's own prose, a line at a time (an empty string between
    paragraphs): no code, logs, links, images, quotes, checklists, tables, headings, HTML comments
    or generated changelogs. Inline code stays, as words of the sentence it is in."""
    text = _DETAILS.sub("", _HTML_COMMENT.sub("", (body or "").replace("\r\n", "\n").replace("\r", "\n")))
    out: list[str] = []
    fence = ""
    for raw in text.split("\n"):
        m = _FENCE.match(raw)
        if fence:
            if m and _closes(fence, m, raw):
                fence = ""
            continue
        if m:
            fence = m.group(1) or m.group(2)
            continue
        if not raw.strip():
            out.append("")
            continue
        if any(p.search(raw) for p in _DROP) or _code_line(raw):
            continue
        line = _LINK.sub(r"\1", _IMG.sub("", raw))
        parts, pos = [], 0
        for c in _INLINE.finditer(line):
            parts += [_markup(line[pos : c.start()]), c.group(2).strip()]
            pos = c.end()
        line = "".join([*parts, _markup(line[pos:])]).rstrip()
        if _NO_ANSWER.match(line) or not _LETTERS.search(line):
            continue
        out.append(line)
    return out


def _join(lines: list[str]) -> str:
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _key(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _codeish(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and sum(1 for ln in lines if len(ln.strip()) < CODE_LINE and _CODE.search(ln)) > CODE_SHARE * len(lines)


# --- the archive --------------------------------------------------------------------------------------


def _ts(s: str | None) -> int | None:
    try:
        return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")) if s else None
    except ValueError:
        return None


def hour_urls(start: int, end: int) -> list[str]:
    """The archive's files that cover [start, end): one an hour, named by its UTC hour without a
    leading zero. An event is in the file of the hour it happened in."""
    out = []
    for t in range(start - start % 3600, end, 3600):
        g = time.gmtime(t)
        out.append(f"{BASE}/{g.tm_year:04d}-{g.tm_mon:02d}-{g.tm_mday:02d}-{g.tm_hour}.json.gz")
    return out


def _lines(url: str):
    """The file's lines as they arrive, read and decompressed a piece at a time, never whole."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, gzip.GzipFile(fileobj=r) as gz:
        yield from gz


def _scan(lines, start: int, end: int) -> list[dict]:
    """Every piece of writing the lines hold that was written in [start, end), before any judgment
    of it: who wrote it, where, when, and what it answers."""
    out = []
    for line in lines:
        head = line[:200]
        if b'"type"' in head and not any(t in head for t in TYPES):
            continue  # most events are not writing: skip them before parsing them
        try:
            e = json.loads(line)
        except ValueError:
            continue
        when = _ts(e.get("created_at"))
        p = e.get("payload") or {}
        if when is None or not start <= when < end or p.get("action") == "edited":
            continue
        kind, repo = e.get("type"), e.get("repo") or {}
        base = {"repo": repo.get("name", ""), "repo_id": repo.get("id"), "event": str(e.get("id", "")), "event_at": when}
        if kind in ("IssuesEvent", "IssueCommentEvent") and isinstance(p.get("issue"), dict):
            out += _issue(p["issue"], base, start, end)
        if kind == "IssueCommentEvent" and p.get("action") == "created" and isinstance(p.get("comment"), dict):
            issue = p.get("issue") or {}
            out += _comment(p["comment"], base, start, end, kind="pull" if issue.get("pull_request") else "issue",
                            title=issue.get("title") or "", context=issue.get("body") or "")
        elif kind == "PullRequestEvent" and p.get("action") == "opened" and isinstance(p.get("pull_request"), dict):
            pr = p["pull_request"]
            if "body" in pr and pr.get("html_url"):  # the archive before October 2025 has the whole pull request
                out += _issue({**pr, "pull_request": True}, base, start, end, topic=_language(pr))
        elif kind == "PullRequestReviewCommentEvent" and p.get("action") == "created" and isinstance(p.get("comment"), dict):
            pr, c = p.get("pull_request") or {}, p["comment"]
            hunk = c.get("diff_hunk") or ""
            out += _comment(c, base, start, end, kind="review", title=pr.get("title") or c.get("path") or "",
                            context=f"{c.get('path', '')}\n{hunk}" if hunk else pr.get("body") or "", topic=_language(pr))
        elif kind == "ReleaseEvent" and p.get("action") == "published" and isinstance(p.get("release"), dict):
            r = p["release"]
            at = _ts(r.get("published_at"))  # a release's created_at is its commit's date, not its notes'
            if not r.get("draft") and at is not None and start <= at < end and r.get("html_url") and "id" in r:
                out.append(_slim({**base, "group": "technical", "kind": "release", "id": f"release:{r['id']}", "url": r["html_url"],
                                  "user": r.get("author"), "app": None, "title": (r.get("name") or r.get("tag_name") or "").strip(),
                                  "body": r.get("body") or "", "at": at, "topic": ""}))
    return out


def _slim(c: dict) -> dict:
    """Only what judging needs: a bot's or an app's writing is judged by who wrote it, so its words
    are not kept while the rest of the hour is read."""
    user = c["user"] or {}
    c["user"] = {"login": user.get("login") or "", "type": user.get("type") or ""} if c["user"] else None
    app = c["app"]
    c["app"] = (app.get("slug") or "app") if isinstance(app, dict) else ("app" if app else None)
    if is_bot(c["user"]) or c["app"]:
        c["body"] = c["question"] = ""
    return c


def _language(pr: dict) -> str:
    return (((pr.get("base") or {}).get("repo") or {}).get("language")) or ""


def _issue(o: dict, base: dict, start: int, end: int, topic: str = "") -> list[dict]:
    """An issue or a pull request, if it was opened in the window. A pull request has one id
    however it is reached (its own event, or a comment's issue): its repository's and its number."""
    at = _ts(o.get("created_at"))
    if at is None or not start <= at < end or not o.get("html_url") or "id" not in o:
        return []
    if o.get("pull_request"):
        kind, oid = "pull", f"pull:{base['repo_id']}/{o.get('number')}"
    else:
        kind, oid = "issue", f"issue:{o['id']}"
    return [_slim({**base, "group": "technical", "kind": kind, "id": oid, "url": o["html_url"], "user": o.get("user"),
                   "app": o.get("performed_via_github_app"), "title": (o.get("title") or "").strip(), "body": o.get("body") or "",
                   "at": at, "topic": topic})]


def _comment(c: dict, base: dict, start: int, end: int, kind: str, title: str, context: str, topic: str = "") -> list[dict]:
    """A comment made in the window, with what it answers as the event gives it."""
    at = _ts(c.get("created_at"))
    if at is None or not start <= at < end or not c.get("html_url") or "id" not in c:
        return []
    context = _HTML_COMMENT.sub("", context.replace("\r\n", "\n")).strip()[:QUESTION_CHARS].strip()
    return [_slim({**base, "group": "discussion", "kind": kind, "id": f"{'review-comment' if kind == 'review' else 'comment'}:{c['id']}",
                   "url": c["html_url"], "user": c.get("user"), "app": c.get("performed_via_github_app"), "title": title.strip(),
                   "body": c.get("body") or "", "at": at, "topic": topic, "question": context or title.strip()})]


def _read_hour(url: str, start: int, end: int, log, tries: int = 4, pause: float = 5.0) -> list[dict]:
    """One hour's writing, read whole or not at all: a download that breaks is started over."""
    name = url.rsplit("/", 1)[-1]
    for i in range(tries):
        try:
            return _scan(_lines(url), start, end)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # not published yet, or one of the archive's gaps
                log(f"  github: {name} is not in the archive")
                return []
            if i == tries - 1:
                raise
            asked = e.headers.get("Retry-After", "")
            time.sleep(min(120.0, float(asked)) if asked.isdigit() else pause * 2**i)
        except urllib.error.URLError as e:
            if isinstance(e.reason, FileNotFoundError):  # an archive on disk, without this hour
                log(f"  github: {name} is not in the archive")
                return []
            if i == tries - 1:
                raise
            time.sleep(pause * 2**i)
        except (OSError, EOFError):  # a connection dropped, a file cut short (gzip.BadGzipFile is an OSError)
            if i == tries - 1:
                raise
            time.sleep(pause * 2**i)
    return []


# --- choosing -------------------------------------------------------------------------------------------


def _judge(cands: list[dict]) -> tuple[list[dict], Counter]:
    """The candidates that are someone's own prose, with that prose as their text; and why the
    rest are not."""
    why: Counter = Counter()
    first: dict[str, dict] = {}
    for c in sorted(cands, key=lambda c: (c["event_at"], len(c["event"]), c["event"])):  # each piece once, as first seen
        first.setdefault(c["id"], c)
    kept = []
    for c in sorted(first.values(), key=lambda c: c["id"]):
        if is_bot(c["user"]):
            why["bot"] += 1
        elif c["app"]:
            why["app"] += 1  # posted by an app on someone's behalf: agents, trackers, syncs
        elif _MACHINE.search(c["body"]):
            why["machine"] += 1
        else:
            kept.append({**c, "lines": prose_lines(c["body"])})
    # A line found in several documents is a template's: an issue form, a pull request's checklist,
    # a signature, an agent's boilerplate.
    seen = Counter(k for c in kept for k in {_key(ln) for ln in c["lines"] if len(_key(ln)) >= COMMON_CHARS})
    out = []
    for c in kept:
        text = _join([ln for ln in c["lines"] if seen[_key(ln)] < COMMON])
        if len(text) < MIN_CHARS:
            why["short"] += 1
        elif _codeish(text):
            why["code"] += 1
        else:
            out.append({**c, "text": text})
    return out, why


def _choose(cands: list[dict], cap: int, rng) -> list[dict]:
    """At most cap, at most PER_REPO from a repository and PER_AUTHOR from an author, chosen with
    rng: the same candidates and seed always give the same choice."""
    pool = sorted(cands, key=lambda c: c["id"])
    rng.shuffle(pool)
    repo: Counter = Counter()
    author: Counter = Counter()
    out = []
    for c in pool:
        r, a = c["repo"].lower(), ((c["user"] or {}).get("login") or "").lower()
        if repo[r] >= PER_REPO or author[a] >= PER_AUTHOR:
            continue
        repo[r] += 1
        author[a] += 1
        out.append(c)
        if len(out) >= cap:
            break
    return sorted(out, key=lambda c: c["id"])


def _lang(text: str) -> str:
    from .langs import model

    (label,), _ = model().predict(text[:1000].replace("\n", " "), k=1)
    return label.removeprefix("__label__")


def _doc(c: dict) -> Doc:
    common = {"id": c["id"], "url": c["url"], "lang": _lang(c["text"]), "license": LICENSE, "published": c["at"],
              "title": c["title"], "text": c["text"], "topic": c["topic"], "extra": {"kind": c["kind"], "repo": c["repo"], "event": c["event"]}}
    if c["group"] == "discussion":
        return Doc(source="github-comment", question=c["question"], **common)
    return Doc(source="github", **common)


def harvest(start: int, end: int, cap: int, rng, log=print) -> list[Doc]:
    """Issues, pull requests' descriptions and release notes (source "github"), and comments with
    what they answer (source "github-comment"), written by people in [start, end): at most cap of
    each of the two, from the archive's hours that cover the window, one download at a time."""
    t0 = time.time()
    cands: list[dict] = []
    for i, url in enumerate(hour_urls(start, end)):
        if i:
            time.sleep(GAP)
        try:
            cands += _read_hour(url, start, end, log)
        except Exception as e:  # one hour failing does not lose the rest
            log(f"  github {url.rsplit('/', 1)[-1]}: {type(e).__name__}: {e}")
    good, why = _judge(cands)
    docs = []
    for group in ("technical", "discussion"):
        pool = [c for c in good if c["group"] == group]
        chosen = _choose(pool, cap, rng.child(group))
        docs += [_doc(c) for c in chosen]
        log(f"  github {group}: {len({c['id'] for c in cands if c['group'] == group})} written in the window, {len(pool)} by people "
            f"in prose, {len(chosen)} kept")
    log(f"  github: {len(docs)} documents in {time.time() - t0:.0f}s; left out: {dict(sorted(why.items()))}")
    return docs
