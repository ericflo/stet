"""lore.kernel.org as a fresh source, against a stand-in for its search feed: no request leaves the machine."""

import time
import urllib.parse
from xml.sax.saxutils import escape

import pytest

from stet.exam.fresh import lore
from stet.exam.rng import Rng

START = 1_800_000_000
END = START + 3600
WHY = ("Requests that arrive after the queue is frozen are counted twice, once when they are queued and once when\n"
       "they are requeued, so the limit is reached with half the requests it allows.\n\n"
       "Count a request when it is first queued and never again.")
SIGN = "-- \n2.43.0\n\n_______________________________________________\nlinux-arm-kernel mailing list\n"


def iso(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def diff(path: str, added=("\tif (!t)", "\t\treturn;", "")) -> str:
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -10,4 +10,{4 + len(added)} @@ static void thing(void)\n {{\n \tstruct thing *t = get();\n"
            + "".join(f"+{a}\n" for a in added) + " \tdo_it(t);\n }\n")


def body(n: int = 0, path: str = "block/blk-core.c", why: str = WHY, tags: str = "", sob: str = "Signed-off-by: Ann Lee <ann@example.org>\n",
         front: str = "", diffs: str | None = None, after: str = SIGN) -> str:
    """A patch as git send-email writes it; n makes its added lines its own."""
    diffs = diff(path, (f"\tcount_{n}++;", "\tif (!t)", "\t\treturn;")) if diffs is None else diffs
    return f"{front}{why}\n\n{tags}{sob}---\n {path} | 3 +++\n 1 file changed, 3 insertions(+)\n\n{diffs}{after}"


def mail(msgid: str, subject: str, text: str, name: str = "Ann Lee", email: str = "ann@example.org", at: int = START + 100, parent: str = ""):
    return {"id": msgid, "subject": subject, "body": text, "name": name, "email": email, "at": at, "parent": parent}


def feed(mails: list[dict]) -> bytes:
    out = ['<?xml version="1.0" encoding="us-ascii"?>\n<feed\nxmlns="http://www.w3.org/2005/Atom"\n'
           'xmlns:thr="http://purl.org/syndication/thread/1.0"><title>search results</title>']
    for m in mails:
        parent = f'<thr:in-reply-to\nref="urn:uuid:0" href="{lore.BASE}{m["parent"]}/"/>' if m["parent"] else ""
        out.append(f"<entry><author><name>{escape(m['name'])}</name><email>{escape(m['email'])}</email></author>"
                   f"<title>{escape(m['subject'])}</title><updated>{iso(m['at'])}</updated><link\n"
                   f'href="{lore.BASE}{urllib.parse.quote(m["id"])}/"/><id>urn:uuid:{m["id"]}</id>{parent}<content\ntype="xhtml"><div\n'
                   f'xmlns="http://www.w3.org/1999/xhtml"><pre\nstyle="white-space:pre-wrap">{escape(m["body"])}</pre></div></content></entry>')
    return "".join(out + ["</feed>"]).encode()


def archive() -> list[dict]:
    """An hour of the lists: the cases a harvest must tell apart."""
    quoted = "> " + body(1).replace("\n", "\n> ")
    stable = body(20, front="6.6-stable review patch.  If anyone has any objections, please let me know.\n\n------------------\n\n"
                  "From: Ann Lee <ann@example.org>\n\ncommit 0123456789abcdef0123456789abcdef01234567 upstream.\n\n")
    return [
        mail("good@a", "[PATCH] block: count a requeued request once", body(1, front="From: Ann Lee <ann@example.org>\n\n")),
        mail("reply@a", "Re: [PATCH] block: count a requeued request once", f"{quoted}\n\nLooks right to me.\n", name="Bob Ray", email="bob@example.org",
             parent="good@a"),
        mail("cover@b", "[PATCH net-next 0/5] net: tidy the thing", f"{WHY}\n\nAnn Lee (5):\n  net: one\n  net: two\n", name="Cy Day", email="cy@example.org"),
        *[mail(f"s{i}@b", f"[PATCH net-next {i}/5] net: tidy the thing, part {i}", body(10 + i, "net/core/dev.c", sob="Signed-off-by: Cy Day <cy@example.org>\n"),
               name="Cy Day", email="cy@example.org", parent="cover@b") for i in range(1, 6)],
        *[mail(f"one{i}@c", f"[PATCH] mm: fix number {i}", body(30 + i, "mm/slub.c"), name="Di Fox", email="di@example.org") for i in range(5)],
        mail("v2@a", "[PATCH v2] block: count a requeued request once more", body(2)),
        mail("resend@a", "[PATCH RESEND] block: count another", body(3)),
        mail("rfc@a", "[RFC PATCH v1 1/1] rust: block: count in Rust", body(4, "rust/kernel/block.rs")),
        mail("dts@a", "[Intel-wired-lan] [PATCH iwl-next] arm64: dts: count a board", body(5, "arch/arm64/boot/dts/x/board.dts")),
        mail("stable@g", "[PATCH 6.6 012/200] block: count a requeued request once", stable, name="Greg Hale", email="greg@example.org"),
        mail("autosel@s", "[PATCH AUTOSEL 6.17-5.10] block: count it", body(21), name="Sam Low", email="sam@example.org"),
        mail("robot@i", "[PATCH] block: fix a warning the robot found", body(22), name="kernel test robot", email="lkp@intel.com"),
        mail("tipbot@k", "[PATCH] sched: count", body(23), name="tip-bot2 for Ann Lee", email="tip-bot2@linutronix.de"),
        mail("ip@n", "[PATCH iproute2-next] ip: count a link", body(24, "lib/utils.c"), name="Ed Gee", email="ed@example.org"),
        mail("uboot@u", "[PATCH] cmd: count a command", body(25, "cmd/count.c"), name="Flo Hay", email="flo@example.org"),
        mail("forwarded@m", "[PATCH net 1/9] ice: count a ring", body(26, "drivers/net/ethernet/intel/ice/ice_main.c", front="From: Ann Lee <ann@example.org>\n\n",
                                                                         sob="Signed-off-by: Ann Lee <ann@example.org>\nSigned-off-by: Guy Ito <guy@example.org>\n"),
             name="Guy Ito", email="guy@example.org"),
        mail("relay@h", "[PATCH] hwmon: count a fan", body(27, "drivers/hwmon/fan.c", front="From: Hal Jay <hal@example.org>\n\n",
                                                           sob="Signed-off-by: Hal Jay <hal@example.org>\n", after="\n---\nbase-commit: 0123456789ab\nchange-id: 2026-fan\n\nBest regards,\n-- \nHal\n"),
             name="Hal Jay via B4 Relay", email="devnull+hal.example.org@kernel.org", at=START + 200),
        mail("direct@h", "[PATCH] hwmon: count a fan", body(27, "drivers/hwmon/fan.c", sob="Signed-off-by: Hal Jay <hal@example.org>\n"), name="Hal Jay",
             email="hal@example.org", at=START + 190),
        mail("unsigned@j", "[PATCH] fs: count an inode", body(28, "fs/inode.c", sob=""), name="Ida Kay", email="ida@example.org"),
        mail("assisted@j", "[PATCH] fs: count a dentry", body(29, "fs/dcache.c", tags="Assisted-by: Claude:claude-fable-5\n"), name="Ida Kay", email="ida@example.org"),
        mail("binary@j", "[PATCH] fs: add a test image", body(40, "tools/testing/x.img", diffs=diff("tools/testing/x.c") + "diff --git a/tools/testing/x.img "
                                                              "b/tools/testing/x.img\nnew file mode 100644\nindex 0000000..1111111\nGIT binary patch\nliteral 12\nabcdef\n\n"),
             name="Ida Kay", email="ida@example.org"),
        mail("small@j", "[PATCH] fs: fix a typo", body(41, "fs/namei.c", diffs=diff("fs/namei.c", ("\tcount++;",))), name="Jo Lam", email="jo@example.org"),
        mail("terse@j", "[PATCH] fs: count", body(42, "fs/file.c", why="Count it."), name="Jo Lam", email="jo@example.org"),
        mail("early@j", "[PATCH] fs: count early", body(43, "fs/open.c"), name="Kit Moe", email="kit@example.org", at=START - 1),
        mail("late@j", "[PATCH] fs: count late", body(44, "fs/read_write.c"), name="Kit Moe", email="kit@example.org", at=END),
        mail("long@l", "[PATCH] lib: add a long table", body(45, "lib/table.c", diffs=diff("lib/table.c", [f"\t{{ {i}, {i * i} }}," for i in range(2000)])),
             name="Lu Ng", email="lu@example.org"),
    ]


class FakeLore:
    def __init__(self, mails):
        self.mails, self.calls = sorted(mails, key=lambda m: -m["at"]), []

    def __call__(self, url, params, **kw):
        assert url == lore.BASE and params["x"] == "A"
        self.calls.append(params)
        o = params.get("o", 0)
        return feed(self.mails[o : o + lore.PAGE]) if o < len(self.mails) else b"</feed>"


@pytest.fixture
def api(monkeypatch):
    fake = FakeLore(archive())
    monkeypatch.setattr(lore, "get_atom", fake)
    monkeypatch.setattr(lore, "GAP", 0)
    return fake


def run(cap=1000, seed=b"seed", log=lambda *a: None):
    return lore.harvest(START, END, cap, Rng(seed, "fresh/lore"), log=log)


def test_only_patches_first_sent_in_the_window_by_their_authors_are_kept(api):
    docs = run()
    ids = {d.id for d in docs}
    assert {"good@a", "rfc@a", "dts@a", "long@l"} <= ids and len(ids) == len(docs)
    for gone in ("reply@a", "cover@b", "v2@a", "resend@a", "stable@g", "autosel@s", "robot@i", "tipbot@k", "ip@n", "uboot@u", "forwarded@m",
                 "unsigned@j", "assisted@j", "binary@j", "small@j", "terse@j", "early@j", "late@j"):
        assert gone not in ids
    assert len(ids & {"relay@h", "direct@h"}) == 1 and "direct@h" in ids  # the same patch twice is kept once, as first sent
    assert all(START <= d.published < END and d.source == "lore" and d.license.startswith("GPL-2.0") and not d.question for d in docs)
    assert all(d.lang.startswith("code:") for d in docs)


def test_the_text_is_the_commit_message_and_the_diff(api):
    docs = {d.id: d for d in run()}
    doc = docs["good@a"]
    assert doc.url == "https://lore.kernel.org/all/good%40a/" and doc.title == "block: count a requeued request once"
    assert doc.lang == "code:c" and doc.topic == "block" and doc.published == START + 100
    assert doc.text == body(1, after="").rstrip("\n")  # no header from the front, no signature or list footer from the back
    assert doc.text.startswith("Requests that arrive") and "\n---\n block/blk-core.c | 3 +++" in doc.text and doc.text.endswith(" \tdo_it(t);\n }")
    assert docs["rfc@a"].lang == "code:rust" and docs["rfc@a"].topic == "rust/kernel" and docs["rfc@a"].title == "rust: block: count in Rust"
    assert docs["dts@a"].lang == "code:devicetree" and docs["dts@a"].topic == "arch/arm64" and docs["dts@a"].extra["tag"] == "PATCH iwl-next"
    assert docs["direct@h"].text.endswith(" \tdo_it(t);\n }")
    long = docs["long@l"].text
    assert lore.MAX_CHARS - 40 < len(long) <= lore.MAX_CHARS and long.endswith("},")  # a long diff is cut on a line


def test_an_author_and_a_series_give_at_most_a_few(api):
    docs = run()
    assert sum(d.id.startswith("one") for d in docs) == lore.PER_AUTHOR
    assert sum(d.id.startswith("s") and d.id.endswith("@b") for d in docs) == lore.PER_SERIES


def test_later_versions_can_be_kept(api, monkeypatch):
    monkeypatch.setattr(lore, "FIRST_ONLY", False)
    assert len({d.id for d in run()} & {"good@a", "v2@a", "resend@a", "rfc@a", "dts@a"}) == lore.PER_AUTHOR  # one author's, all the same
    monkeypatch.setattr(lore, "PER_AUTHOR", 10)
    assert {"good@a", "v2@a", "resend@a", "rfc@a", "dts@a"} <= {d.id for d in run()}


def test_the_search_asks_for_the_window_on_the_kernels_lists_a_page_at_a_time(api, monkeypatch):
    monkeypatch.setattr(lore, "PAGE", 10)
    docs = run()
    assert [c.get("o", 0) for c in api.calls] == [0, 10, 20, 30]  # 34 messages: three full pages and a short one
    q = api.calls[0]["q"]
    assert f"rt:{START}..{END - 1}" in q and "dt:20270115080000..20270115085959" in q and "s:PATCH" in q and "nq:Signed-off-by" in q
    assert "l:linux-kernel.vger.kernel.org" in q and "l:netdev.vger.kernel.org" in q and "u-boot" not in q and "git.vger" not in q
    monkeypatch.setattr(lore, "PAGE", 200)
    assert [d.id for d in run()] == [d.id for d in docs]


def test_an_archive_that_fails_gives_nothing(monkeypatch):
    def down(url, params, **kw):
        raise OSError("no answer")

    monkeypatch.setattr(lore, "get_atom", down)
    said = []
    assert run(log=said.append) == [] and "OSError" in said[0]
    monkeypatch.setattr(lore, "get_atom", lambda url, params, **kw: b"")  # a search that finds nothing
    assert run() == []


def test_the_choice_is_seeded(api):
    a = run(cap=5)
    assert len(a) == 5 and [d.id for d in run(cap=5)] == [d.id for d in a]
    assert {d.id for d in run(cap=5, seed=b"another")} != {d.id for d in a}


def test_subjects_and_diffs_are_read():
    assert lore.subject("[PATCH net-next v7 2/3] net: do it") == ("PATCH net-next v7 2/3", "PATCH net-next v7 2/3", "net: do it")
    assert lore.subject("[f2fs-dev] [RFC PATCH] f2fs: do it") == ("RFC PATCH", "f2fs-dev RFC PATCH", "f2fs: do it")
    assert lore.subject("[GIT PULL] things for 7.1") is None and lore.subject("net: do it") is None
    # a removed line of dashes is the diff's; the signature after its last hunk is not
    text = ("Why.\n\nSigned-off-by: A <a@example.org>\n---\n a.txt | 2 +-\n\ndiff --git a/a.txt b/a.txt\nindex 1..2 100644\n--- a/a.txt\n+++ b/a.txt\n"
            "@@ -1,3 +1,3 @@\n one\n--- \n+two\n three\n\\ No newline at end of file\n-- \n2.43.0\n")
    message, rest = lore.split(text)
    assert message == "Why.\n\nSigned-off-by: A <a@example.org>"
    assert rest.startswith("---\n a.txt | 2 +-\n\ndiff --git") and rest.endswith("--- \n+two\n three\n\\ No newline at end of file")
    assert lore.split("No diff here.\n") is None
