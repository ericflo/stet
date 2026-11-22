"""Patches e-mailed to the Linux kernel's mailing lists in the window: code, and why it was written.

lore.kernel.org archives the kernel's lists with public-inbox, whose search answers anyone
without a key or an account, as an Atom feed that carries every message whole:
/all/?q=<query>&x=A, two hundred messages a page, newest first, `o=<offset>` for the next. One
query finds the window's patches on the lists named below: received by the archive in the window
(rt:), dated in it by their sender (dt:), "PATCH" in the subject and a Signed-off-by of their own.

A message is kept if it is a patch as its author first sent it: a subject tagged [PATCH ...],
a commit message, a Signed-off-by and a diff. Replies go, and so do cover letters, later versions
and resends (v2, RESEND: most of their text was public before the window), patches sent on for
someone else (a maintainer's batch of what a sub-list has already seen), backports to stable
kernels (AUTOSEL, [PATCH 6.6], "commit ... upstream"), bots, binary patches, patches that add
next to nothing, and patches whose tags say an assistant wrote them. The text is the commit
message and the diff, as e-mailed: the in-body headers before it and the signature and list
footer after it are cut, the diffstat stays. A long diff is cut at MAX_CHARS, on a line.

The lists are the kernel's own, and a patch must touch only paths of the kernel's tree and carry
no other project's tag (iproute2, BlueZ, kvm-unit-tests and the like share those lists): what is
kept is a contribution to the kernel, GPL-2.0 under its Developer's Certificate of Origin.
lore.kernel.org archives other projects too (Git, U-Boot, QEMU, Buildroot, Xen), each under its
own licence; they are not asked for.

An author gives at most a few patches, and a series fewer still. The language is the code's:
"code:c", "code:rust", "code:devicetree", by where most of the added lines are.
"""

from __future__ import annotations

import calendar
import gzip
import hashlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

from . import GAP, UA, Doc

BASE = "https://lore.kernel.org/all/"
LICENSE = "GPL-2.0 (Linux kernel contributions under the DCO); via lore.kernel.org"
PAGE = 200  # messages in a full page of the feed
MAX_PAGES = 5  # of the feed read for one window
PER_AUTHOR = 3  # patches from any one author
PER_SERIES = 2  # and from any one series
MIN_MESSAGE = 100  # characters of commit message, its tags aside
MIN_ADDED = 3  # lines the diff adds
MAX_CHARS = 12000  # of a document: a longer diff is cut
FIRST_ONLY = True  # leave out later versions and resends, most of whose text is older than the window

# Lists whose patches are for the kernel's tree (List-Id, as the search's l: matches it).
_VGER = (
    "linux-kernel netdev bpf linux-fsdevel linux-wireless linux-media linux-usb linux-scsi linux-pm linux-iio linux-sound "
    "linux-doc devicetree kvm linux-s390 linux-mips rust-for-linux linux-pci linux-gpio linux-crypto linux-hardening "
    "linux-trace-kernel linux-perf-users linux-kselftest linux-arm-msm linux-acpi linux-input linux-hwmon linux-i2c linux-spi "
    "linux-mmc linux-clk linux-serial linux-watchdog platform-driver-x86 linux-renesas-soc linux-samsung-soc linux-tegra "
    "linux-omap linux-hyperv linux-security-module linux-efi linux-edac linux-cxl cgroups linux-can linux-kbuild linux-arch "
    "linux-api linux-block io-uring linux-xfs linux-btrfs linux-nfs linux-rdma linux-leds linux-pwm linux-rtc linux-fbdev "
    "linux-remoteproc linux-ide dmaengine sparclinux linux-parisc linux-m68k linux-sh linux-staging"
)
LISTS = tuple(f"{name}.vger.kernel.org" for name in _VGER.split()) + (
    "linux-arm-kernel.lists.infradead.org", "linux-riscv.lists.infradead.org", "linux-mediatek.lists.infradead.org",
    "linux-rockchip.lists.infradead.org", "linux-amlogic.lists.infradead.org", "linux-phy.lists.infradead.org",
    "linux-mtd.lists.infradead.org", "ath10k.lists.infradead.org", "ath11k.lists.infradead.org", "ath12k.lists.infradead.org",
    "linux-um.lists.infradead.org", "patches.lists.linux.dev", "iommu.lists.linux.dev", "kvmarm.lists.linux.dev",
    "linux-sunxi.lists.linux.dev", "imx.lists.linux.dev", "loongarch.lists.linux.dev", "llvm.lists.linux.dev",
    "damon.lists.linux.dev", "mptcp.lists.linux.dev", "linux-staging.lists.linux.dev", "sched-ext.lists.linux.dev",
    "driver-core.lists.linux.dev", "linux-rt-devel.lists.linux.dev", "virtualization.lists.linux.dev", "asahi.lists.linux.dev",
    "linux-mm.kvack.org", "dri-devel.lists.freedesktop.org", "intel-gfx.lists.freedesktop.org", "intel-xe.lists.freedesktop.org",
    "amd-gfx.lists.freedesktop.org", "nouveau.lists.freedesktop.org", "linuxppc-dev.lists.ozlabs.org", "intel-wired-lan.osuosl.org",
)
# The top of the kernel's tree: a patch that touches anything else is another project's.
ROOTS = frozenset(
    "arch block certs crypto Documentation drivers fs include init io_uring ipc kernel lib LICENSES mm net rust samples scripts "
    "security sound tools usr virt MAINTAINERS Makefile Kconfig Kbuild CREDITS .mailmap .gitignore .clang-format "
    ".get_maintainer.ignore .editorconfig .rustfmt.toml".split())
# Projects that share the kernel's lists and say so in the subject's tag.
_OTHER = re.compile(
    r"\b(?:iproute2|ethtool|bluez|nft|nftables|iptables|ebtables|libnftnl|libmnl|conntrack-tools|ipset|liburing|kvm-unit-tests|"
    r"i-g-t|igt|libdrm|xfsprogs|xfstests|fstests|blktests|btrfs-progs|e2fsprogs|f2fs-tools|erofs-utils|nfs-utils|cifs-utils|"
    r"rdma-core|ndctl|mdadm|dwarves|pahole|v4l-utils|libcamera|alsa-utils|alsa-lib|iw|wireless-regdb|hostap|libgpiod|dtc|"
    r"dtschema|dt-schema|trace-cmd|libtraceevent|libtracefs|kernelshark|rt-tests|util-linux|man-pages|ltp|kexec-tools|"
    r"makedumpfile|crash|mptcpd|ofono|ell|iwd|connman|qemu|u-boot|opensbi|tf-a|edk2|grub|barebox|buildroot|yocto|oe-core|"
    r"bitbake|poky|userspace|testsuite|sparse|smatch|cocci|git|b4|nvme-cli|libnvme|fio|xdp-tools|linux-firmware|firmware|"
    r"pciutils|usbutils|kmod|numactl|syzkaller|ima-evm-utils|keyutils|strace|glibc)\b", re.I)
_STABLE = re.compile(r"(?:^|\s)\d+\.\d+(?:\.\d+)?(?:\.y|-stable)?(?=\s|$)|\.y\b|\bAUTOSEL\b|\bstable\b|\bbackport", re.I)
_UPSTREAM = re.compile(r"^\s*commit [0-9a-f]{12,40} upstream\b|\[ Upstream commit [0-9a-f]{12,40} \]|"
                       r"\((?:cherry picked|backported) from commit [0-9a-f]{7,40}", re.I | re.M)
_LATER = re.compile(r"(?:(?<![a-z])|(?<=patch))v(\d+)\b", re.I)
_RESEND = re.compile(r"\b(?:RESEND|REPOST|RESENT)\b", re.I)
_COUNT = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
# Senders that are programs: test robots, CI, patch trackers, the tip tree's and the archive's own.
_BOT = re.compile(r"bot\d*\b|\brobot\b|patchwork|no-?reply|\bci\b|syzkaller|lkp@|replication probe|webhooks|github|"
                  r"kernelci|regzbot|tip-bot", re.I)
# The kernel asks for an Assisted-by: tag on what a coding assistant helped write.
_MACHINE = re.compile(r"^\s*(?:Assisted-by|Generated-by):|^\s*(?:Co-developed-by|Co-authored-by):.*\b(?:claude|chatgpt|gpt-?\d\S*|"
                      r"codex|copilot|gemini|cursor|deepseek|qwen|grok|kimi|sonnet|opus|llm)\b", re.I | re.M)
_TRAILER = re.compile(r"^[A-Z][A-Za-z-]+: \S")
_SOB = re.compile(r"^Signed-off-by: .*\S", re.M)
_IN_BODY = re.compile(r"^(From|Date|Subject): *(.*)$")
_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_BINARY = re.compile(r"^GIT binary patch$|^Binary files .* differ$", re.M)
_FILE = re.compile(r"^diff --git a/(\S+) b/\S+", re.M)
_META = ("index ", "--- ", "+++ ", "new file mode", "deleted file mode", "old mode", "new mode", "similarity index",
         "dissimilarity index", "rename from", "rename to", "copy from", "copy to")
# What the code is written in, by a file's name: only to say which language a patch is mostly in.
_EXT = {"c": "c", "h": "c", "rs": "rust", "py": "python", "sh": "shell", "tc": "shell", "dts": "devicetree", "dtsi": "devicetree",
        "dtso": "devicetree", "yaml": "yaml", "yml": "yaml", "json": "json", "rst": "rst", "txt": "text", "S": "asm", "pl": "perl",
        "awk": "awk", "lds": "ld", "cocci": "coccinelle", "config": "kconfig"}
_NAME = {"Makefile": "make", "Kbuild": "make", "Kconfig": "kconfig", "MAINTAINERS": "text"}
_A = "{http://www.w3.org/2005/Atom}"
_T = "{http://purl.org/syndication/thread/1.0}"


# --- the archive ----------------------------------------------------------------------------------------


def get_atom(url: str, params: dict, tries: int = 4, pause: float = 5.0) -> bytes:
    """GET a feed from the archive, compressed on the way; a search that finds nothing is a 404."""
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={**UA, "Accept-Encoding": "gzip"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                return gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            if i == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            asked = e.headers.get("Retry-After", "")
            time.sleep(min(120.0, float(asked)) if asked.isdigit() else pause * 2**i)
        except (OSError, EOFError):  # a connection dropped, a body cut short
            if i == tries - 1:
                raise
            time.sleep(pause * 2**i)
    return b""


def query(start: int, end: int) -> str:
    """The search for patches the archive received in [start, end), and dated in it."""
    day = "%Y%m%d%H%M%S"
    dated = f"dt:{time.strftime(day, time.gmtime(start))}..{time.strftime(day, time.gmtime(end - 1))}"
    lists = " OR ".join(f"l:{name}" for name in LISTS)
    return f"rt:{start}..{end - 1} AND {dated} AND s:PATCH AND nq:Signed-off-by AND ({lists})"


def _ts(s: str | None) -> int | None:
    try:
        return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")) if s else None
    except ValueError:
        return None


_ENTRY = re.compile(rb"<entry[\s>].*?</entry>", re.S)
_FEED = re.compile(rb"\s*(?:<\?xml[^>]*\?>\s*)?(<feed\b[^>]*>)")


def _elements(feed: bytes) -> list:
    """The feed's entries. A feed that is not XML as a whole (one message with a character XML
    does not allow is enough) gives those of its entries that are, each read on its own."""
    try:
        return list(ET.fromstring(feed).iter(f"{_A}entry"))
    except ET.ParseError:
        pass
    head = _FEED.match(feed)
    if head is None:
        return []
    out = []
    for m in _ENTRY.finditer(feed):
        try:
            out += ET.fromstring(head.group(1) + m.group(0) + b"</feed>").iter(f"{_A}entry")
        except ET.ParseError:
            continue
    return out


def entries(feed: bytes) -> list[dict]:
    """The feed's messages: subject, sender, date, the archive's URL for each, and its body whole."""
    if b"<entry" not in feed:  # nothing found, or a page past the last
        return []
    out = []
    for e in _elements(feed):
        link, content, parent = e.find(f"{_A}link"), e.find(f"{_A}content"), e.find(f"{_T}in-reply-to")
        if link is None or not link.get("href"):
            continue
        out.append({"subject": " ".join((e.findtext(f"{_A}title") or "").split()),
                    "name": " ".join((e.findtext(f"{_A}author/{_A}name") or "").split()),
                    "email": (e.findtext(f"{_A}author/{_A}email") or "").strip().lower(),
                    "at": _ts(e.findtext(f"{_A}updated")), "url": link.get("href"),
                    "parent": parent.get("href", "") if parent is not None else "",
                    "body": "".join(content.itertext()) if content is not None else ""})
    return out


def _window(get, start: int, end: int, why: Counter) -> list[dict]:
    out: list[dict] = []
    for n in range(MAX_PAGES):
        if n:
            time.sleep(GAP)
        feed = get(BASE, {"q": query(start, end), "x": "A", **({"o": n * PAGE} if n else {})})
        got = entries(feed)
        out += got
        if len(got) < PAGE and feed.count(b"</entry>") < PAGE:  # a page with messages that could not be read is still a full page
            break
    else:
        why[f"hours with more than {MAX_PAGES * PAGE} messages"] += 1
    return out


# --- what is a patch ------------------------------------------------------------------------------------


def _author(name: str) -> str:
    """A sender's name without the relay's: b4 sends "Ann Lee via B4 Relay" for Ann Lee."""
    return re.sub(r"\s+via\s+.*$", "", name, flags=re.I).strip(" \"'")


def subject(s: str) -> tuple[str, str, str] | None:
    """A patch's subject as (its [PATCH ...] tag, every tag before the title, the title); None
    for a subject that is not a patch's."""
    m = re.match(r"((?:\[[^\]]*\]\s*)+)(.*)$", s)
    if not m:
        return None
    tags = re.findall(r"\[([^\]]*)\]", m.group(1))
    patch = next((t for t in tags if re.search(r"\bPATCH\b", t, re.I)), None)
    return (patch, " ".join(tags), m.group(2).strip()) if patch and m.group(2).strip() else None


def split(body: str) -> tuple[str, str] | None:
    """A patch e-mail's body as (commit message, the rest up to the diff's last line): the headers
    git puts in the body go from the front, and whatever follows the diff (a signature, a base
    commit, a list's footer) from the back. None for a body without a diff."""
    lines = body.replace("\r\n", "\n").split("\n")
    first = next((i for i, ln in enumerate(lines) if _FILE.match(ln)), None)
    if first is None:
        return None
    head = lines[:first]
    while head and (not head[0].strip() or _IN_BODY.match(head[0])):
        head.pop(0)
    cut = next((i for i in range(len(head) - 1, -1, -1) if head[i] == "---"), len(head))
    i, old, new = first, 0, 0
    while i < len(lines):  # the diff ends where its hunks' counts say, not at the first line that looks like prose
        ln = lines[i]
        if old > 0 or new > 0:
            if ln.startswith("-"):
                old -= 1
            elif ln.startswith("+"):
                new -= 1
            elif ln.startswith(" ") or not ln:
                old, new = old - 1, new - 1
            elif not ln.startswith("\\"):
                break
        elif m := _HUNK.match(ln):
            old, new = int(m.group(1) or 1), int(m.group(2) or 1)
        elif not (_FILE.match(ln) or ln.startswith(_META) or ln.startswith("\\")):
            break
        i += 1
    return "\n".join(head[:cut]).strip(), "\n".join(head[cut:] + lines[first:i]).strip("\n")


def _language(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name in _NAME or name.startswith(("Kconfig", "Makefile")):
        return _NAME.get(name, "kconfig" if name.startswith("Kconfig") else "make")
    return _EXT.get(name.rsplit(".", 1)[-1], "text") if "." in name else "text"


def _added(diff: str) -> tuple[Counter, list[str]]:
    """The lines the diff adds, counted by the file they are added to; and the lines."""
    per: Counter = Counter()
    lines, path = [], ""
    for ln in diff.split("\n"):
        if m := _FILE.match(ln):
            path = m.group(1)
        elif path and ln.startswith("+") and not ln.startswith("+++ "):
            per[path] += 1
            lines.append(ln[1:])
    return per, lines


def _patch(e: dict, start: int, end: int, why: Counter) -> dict | None:
    """A message as a patch its author first sent in the window; None for anything else."""
    if e["at"] is None or not start <= e["at"] < end:
        return why.update(["outside the window"])
    if re.match(r"\s*(?:re|aw|fwd?|antw)\s*:", e["subject"], re.I):
        return why.update(["reply"])
    parts = subject(e["subject"])
    if parts is None:
        return why.update(["not a patch"])
    tag, tags, title = parts
    if _BOT.search(e["name"]) or _BOT.search(e["email"]):
        return why.update(["bot"])
    if _OTHER.search(tags):
        return why.update(["another project's"])
    if _STABLE.search(tag):
        return why.update(["backport"])
    if FIRST_ONLY and (_RESEND.search(tag) or any(int(v) > 1 for v in _LATER.findall(tag))):
        return why.update(["later version or resend"])
    count = _COUNT.search(tag)
    if count and int(count.group(1)) == 0:
        return why.update(["cover letter"])
    body = e["body"].replace("\r\n", "\n")
    sent_for = next((m.group(2) for ln in body.lstrip().split("\n")[:3] if (m := _IN_BODY.match(ln)) and m.group(1) == "From"), "")
    author = _author(e["name"])
    theirs = bool(e["email"]) and e["email"] in sent_for.lower()  # no address is no sign that it is the sender's own
    if sent_for and _author(sent_for.split("<")[0]).lower() != author.lower() and not theirs:
        return why.update(["sent on for someone else"])
    parts = split(body)
    if parts is None:
        return why.update(["no diff"])
    message, rest = parts
    if not _SOB.search(message):
        return why.update(["not signed off"])
    if _UPSTREAM.search(message):
        return why.update(["backport"])
    if _MACHINE.search(message):
        return why.update(["written with an assistant"])
    if _BINARY.search(body):
        return why.update(["binary"])
    per, added = _added(rest)
    if not per or any(path.split("/", 1)[0] not in ROOTS for path in _FILE.findall(rest) + list(per)):
        return why.update(["not the kernel's tree"])
    if len(added) < MIN_ADDED:
        return why.update(["adds next to nothing"])
    prose = " ".join(ln for ln in message.split("\n") if not _TRAILER.match(ln))
    if len(" ".join(prose.split())) < MIN_MESSAGE:
        return why.update(["no commit message to speak of"])
    text = f"{message}\n{rest}" if rest.startswith("---") else f"{message}\n\n{rest}"
    if len(text) > MAX_CHARS:
        cut = text.rfind("\n", 0, MAX_CHARS)
        text = text[: cut if cut > 0 else MAX_CHARS]  # at a line's end, or where there is none, at the limit
    by_lang: Counter = Counter()
    by_dir: Counter = Counter()
    for path, n in sorted(per.items()):
        by_lang[_language(path)] += n
        by_dir["/".join(path.split("/")[:-1][:2]) or path] += n  # a subsystem, more or less: drivers/net, fs/xfs, mm
    return {**e, "author": author, "title": title, "tag": tag, "text": text,
            "series": _COUNT.sub(lambda m: f"/{m.group(2)}", tag) if count else "",
            "lang": max(sorted(by_lang), key=lambda k: by_lang[k]), "topic": max(sorted(by_dir), key=lambda k: by_dir[k]),
            "files": len(per), "key": hashlib.sha256("\n".join([title, *added]).encode()).hexdigest()}


# --- the harvest ----------------------------------------------------------------------------------------


def _id(url: str) -> str:
    return urllib.parse.unquote(url.rstrip("/").rsplit("/", 1)[-1])


def harvest(start: int, end: int, cap: int, rng, log=print) -> list[Doc]:
    """Up to cap patches first sent to the kernel's lists in [start, end), chosen with rng: at
    most PER_AUTHOR from an author and PER_SERIES from a series."""
    t0 = time.time()
    why: Counter = Counter()
    try:
        found = _window(get_atom, start, end, why)
    except Exception as e:
        log(f"  lore: {type(e).__name__}: {e}")
        return []
    pool, seen = [], set()
    for e in sorted(found, key=lambda e: (e["at"] or 0, e["url"])):  # a patch sent twice is kept as first sent
        try:
            p = _patch(e, start, end, why)
        except Exception as err:  # one message that cannot be read does not lose the others
            why[f"error {type(err).__name__}"] += 1
            continue
        if p is None:
            continue
        if p["key"] in seen:
            why["sent twice"] += 1
            continue
        seen.add(p["key"])
        pool.append(p)
    pool.sort(key=lambda p: p["url"])
    rng.shuffle(pool)
    docs: list[Doc] = []
    per_author: Counter = Counter()
    per_series: Counter = Counter()
    for p in pool:
        if len(docs) >= cap:
            break
        author = p["author"].lower() or p["email"]
        series = (author, p["series"]) if p["series"] else (author, p["url"])
        if per_author[author] >= PER_AUTHOR:
            why["author full"] += 1
            continue
        if per_series[series] >= PER_SERIES:
            why["series full"] += 1
            continue
        per_author[author] += 1
        per_series[series] += 1
        docs.append(Doc(source="lore", id=_id(p["url"]), url=p["url"], lang=f"code:{p['lang']}", license=LICENSE, published=p["at"],
                        title=p["title"], text=p["text"], topic=p["topic"],
                        extra={"tag": p["tag"], "files": p["files"]}))
    langs = Counter(d.lang for d in docs)
    skipped = ", ".join(f"{k} {v}" for k, v in why.most_common())
    log(f"  lore: {len(docs)} patches of {len(pool)} first sent in the window, among {len(found)} messages, in "
        f"{dict(langs.most_common(6))} ({time.time() - t0:.0f}s; skipped: {skipped or 'none'})")
    return docs
