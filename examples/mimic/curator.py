"""This curator tries to buy the exam with selection. It is entered so that the board shows
what that buys.

The exam is known in outline: writing people published (reference, news, technical,
conversation) and generated skills (a session at a shell, tool calls and their JSON, reading,
transformation, logic). mimic keeps the pages of the corpus that look most like those, in any
language, and writes nothing of its own:

  1. Mark every page four times, with regular expressions over its opening and its address:
     how much it looks like a reference article, a news report, technical text (terminals,
     logs, JSON, code, patches) or a conversation (threads, replies, questions and answers).
  2. Inside a page, keep the lines that are writing or one of those forms, and the few short
     lines between two of them (a heading, what a command printed). Cut the rest.
  3. Fill a quarter of the budget from each kind, its most exam-like pages first. What a
     kind cannot fill goes to the best pages left of the others.
  4. Only if the budget is still not full, top it up with what `tidy` would have kept.
  5. Shuffle.

No model, no GPU, no ink: every kept piece is a run of whole lines at least min_copy_bytes
long. If this ever beats `tidy` by more than noise, the exam can be bought by looking like
it, and it is the exam that gets fixed.
"""

import hashlib
import random
import re
import unicodedata

from stet.curator import Task

# --- tidy's selection, copied from examples/tidy: the top-up -----------------------------------

# How a sentence ends: in scripts written with spaces, a mark and then a space; in Chinese and
# Japanese, the mark alone.
SENTENCE_END = re.compile(r"[.!?…؟۔।][\"')\]”’»]?\s|[。！？｡]")
SPACELESS = (("぀", "ヿ"), ("一", "鿿"), ("฀", "๿"))  # kana, Chinese characters, Thai


def looks_written(text: str) -> bool:
    """Mostly letters, of any script, and enough of them to say something. The vowel signs of
    Hindi, Thai and Arabic are marks and not letters, and count as part of the letters they sit on."""
    sample = text[:3000]
    if len(sample) < 200:
        return False
    return sum(c.isalpha() or c.isspace() or unicodedata.category(c)[0] == "M" for c in sample) / len(sample) >= 0.75


def spaceless(text: str) -> bool:
    """Written without spaces between words, so that a line's length is counted in characters."""
    sample = text[:2000]
    return sum(1 for c in sample if any(a <= c <= b for a, b in SPACELESS)) > 0.3 * len(sample)


def prose_ranges(raw: bytes, dense: bool) -> tuple[list[list[int]], int]:
    """Byte ranges of the lines worth keeping, each a whole line with its line break, and how
    many bytes they hold. A line is kept when it is long enough to be a sentence and has the
    end of one in it."""
    ranges: list[list[int]] = []
    pos = 0
    for line in raw.split(b"\n"):
        n = len(line) + 1  # + the line break
        body = line.decode("utf-8", "replace").strip()
        long_enough = len(body) >= 25 if dense else (len(body) >= 60 and body.count(" ") >= 9)
        if long_enough and SENTENCE_END.search(body + " "):
            if ranges and ranges[-1][1] == pos:
                ranges[-1][1] = pos + n
            else:
                ranges.append([pos, pos + n])
        pos += n
    if ranges and ranges[-1][1] > len(raw):  # the last line may have no line break
        ranges[-1][1] = len(raw)
    return ranges, sum(b - a for a, b in ranges)


def score(size: int, kept: int) -> float:
    """Higher is better: a page that is mostly writing, and long enough to say something."""
    return kept / max(1, size) + min(1.0, kept / 3000)


def tidy_candidates(task: Task, skip: set[int]) -> list[tuple[float, int, list[list[int]], int]]:
    """(score, document, kept ranges, kept bytes) of what tidy would keep, best first."""
    corpus = task.corpus
    seen: set[bytes] = set()
    scored = []
    for i in range(len(corpus)):
        if i in skip:
            continue
        text = corpus.text(i)
        if not looks_written(text):
            continue
        ranges, kept = prose_ranges(corpus.raw(i), spaceless(text))
        if kept < 400:
            continue
        key = hashlib.blake2b(text[:2000].encode("utf-8", "replace"), digest_size=8).digest()
        if key in seen:
            continue
        seen.add(key)
        scored.append((score(corpus.size(i), kept), i, ranges, kept))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


# --- what the exam looks like --------------------------------------------------------------------

# What mimic keeps, every language and code together, is about 2.8 bytes a token under the
# season tokenizer; English prose is about 3.2. On the generous side, as tidy is: a stream that
# runs over the budget is cut off, one that falls short repeats from the top.
BYTES_PER_TOKEN = 3.2
HEAD_BYTES = 12_000  # the marks are given on a page's opening
MIN_LIKE = 0.3  # a page marked lower than this in every kind is not exam-like at all
TEXT_SHARE, ADDRESS_SHARE = 0.7, 0.3  # of a mark: what the page says, and where it is from
BRIDGE = 4  # this many short lines between two kept ones stay: a heading, what a command printed


def rx(pattern: str) -> re.Pattern:
    """A pattern over a page's bytes, so that it reads every script alike and an offset is an
    offset. Each begins with a literal (a line break, a bracket, a digit) for the search to
    skip to: an alternation tried at every byte is ten times slower."""
    return re.compile(pattern.encode("utf-8"), re.M)


# A line of technical text, from its first byte: a prompt or a command, a log line, JSON, code,
# a patch, a mailing list's headers.
FORM = (
    r"[ \t]*(?:\$ \S|>>> |[\w.-]+@[\w.-]+:\S* ?[$#] |[A-Z]:\\"
    r"|(?:sudo|apt|apt-get|yum|dnf|brew|pip3?|npm|npx|yarn|git|docker|kubectl|curl|wget|ssh|chmod|mkdir|systemctl|cargo|make) [a-z./-]"
    r"|\[?\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d|\[?(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b|Traceback \(most recent|at [\w.$]+\(|File \""
    r"|[{\[]$|[}\]][,;]?$|\"[\w .-]{1,40}\" ?: ?\S"
    r"|(?:def |class |import |from [\w.]+ import |function\b|const |let |var |public |private |static |return\b|if ?\(|for ?\(|while ?\(|#include"
    r"|using |package |fn |func |SELECT |INSERT INTO |UPDATE |CREATE |<\?php|//|/\*|#!|</?[a-z][\w-]*[ >])"
    r"|diff --git|@@ -\d|[+-]{3} [ab]/|(?:Signed-off-by|Reviewed-by|Acked-by|Reported-by|Fixes|Closes|Resolves|Subject|Message-ID|In-Reply-To): ?\S)"
)
# The words a thread's furniture is made of, each alone on its line: reply, quote.
REPLY = (
    "Reply|Quote|Antworten|Zitat|Zitieren|Répondre|Citer|Responder|Citar|Rispondi|Cita|Ответить|Цитата|Цитировать"
    "|返信|引用|回复|回復|답글|Cevapla|Odpowiedz|Cytuj|Reageer|Svara|Vastaa"
)
# The headings an encyclopedia article ends with.
SOURCES = (
    "References|External links|See also|Bibliography|Further reading|Einzelnachweise|Weblinks|Literatur|Références|Liens externes"
    "|Voir aussi|Referencias|Enlaces externos|Véase también|Referências|Ligações externas|Voci correlate|Collegamenti esterni"
    "|Примечания|Ссылки|Литература|脚注|関連項目|外部リンク|参考文献|外部链接|각주|Przypisy|Linki zewnętrzne|Zobacz też|Kaynakça"
)


def words(*literals: str) -> tuple[bytes, ...]:
    """Literals to count, which is faster still."""
    return tuple(w.encode("utf-8") for w in literals)


# Somebody is reported as saying something, and somebody is quoted.
SAID = words(" said ", " told ", " dijo ", " sagte ", " a déclaré ", " disse ", " ha detto ", "сказал", "заявил", " dedi", "表示", "述べ", "말했다")
QUOTED = words("“", "«", "„", "「")

# For each kind: what its address tends to hold, how many marks make a page wholly of the
# kind, and the patterns that give them: (marks a match, most matches counted, pattern).
KINDS = {
    "reference": (
        re.compile(r"wiki|pedia\.|encyclo|britannica|lexikon|baike|dictionary|glossar|biograph"),
        8,
        [
            (1, 8, rx(r"\[(?:\d{1,3}|edit)\]")),  # a citation
            (0.5, 8, rx(r"\((?:[A-Za-z]{1,8}\.? )?\d{4}(?: ?(?:-|–) ?\d{0,4})?\)")),  # (1856–1923), (born 1960), (2019)
            (4, 1, rx(rf"\n(?:{SOURCES})\n")),
        ],
    ),
    "news": (
        re.compile(r"news|/20\d\d/\d\d?/|/articles?/|/story|/press|noticia|nachricht|actualit|novosti|haber|berita|nyheter|notizie|zeitung|diario|gazet|tribun|herald"),
        8,
        [
            (4, 1, rx(r"\((?:AP|Reuters|AFP|dpa|EFE|ANSA|TASS)\)")),  # a dateline
            (4, 1, rx(r"\nBy [A-Z]")),  # a byline
            (1, 4, SAID),
            (0.5, 4, QUOTED),
            (0.5, 4, rx(r"20[012]\d\b")),  # a date of this century
        ],
    ),
    "technical": (
        re.compile(
            r"github|gitlab|bitbucket|stackoverflow|stackexchange|/docs?/|docs?\.|/api/|/issues?/|/pull/|/releases?/|/commits?/|changelog"
            r"|kernel\.org|lists\.|bugzilla|bugs\.|sourceforge|npmjs|pypi|readthedocs|/man/|/manual|developer\.|tutorial|howto|gist\."
        ),
        8,
        [(1, 8, rx(r"\n" + FORM)), (0.5, 8, rx(r"[;{}]\n"))],
    ),
    "conversation": (
        re.compile(r"forum|foro|thread|topic|discuss|/comments?|/t/|/r/|/questions?/|answers\.|community\.|/boards?/|talk:|/talk|lemmy|mastodon|bsky|ycombinator|reddit|quora|/status/"),
        8,
        [
            (1, 8, rx(rf"\n(?:Re: |Posted (?:by|on) |Joined:? |Posts: ?\d|#\d+\n|@[\w.-]+|(?:Q|A|Question|Answer) ?[:.] |(?:{REPLY})\n)")),
            (1, 4, rx(r"\d (?:repl(?:y|ies)|comments?|points?|likes?|upvotes?|answers?|(?:min(?:ute)?s?|hours?|days?|weeks?|months?|years?) ago)\b")),
            (0.5, 6, rx(r"(?:\?|？|؟)\n")),  # a line that asks
        ],
    ),
}

FORM_LINE = rx(FORM + r"|(?:Q|A|Question|Answer) ?[:.] ")
# A sentence ends or pauses here, in any script.
WRITING = rx(r"[.!?:;][\"')\]]?(?: |$)|。|！|？|؟|۔|।|…")


def count(cue, head: bytes) -> int:
    return sum(head.count(w) for w in cue) if isinstance(cue, tuple) else len(cue.findall(head))


def likeness(head: bytes, url: str) -> tuple[float, str]:
    """How exam-like a page is, from 0 to 1, and the kind it is most like. `head` is the
    page's opening after a line break, `url` its address in lower case."""
    best, kind = 0.0, ""
    for name, (address, need, cues) in KINDS.items():
        marks = sum(weight * min(most, count(cue, head)) for weight, most, cue in cues)
        like = TEXT_SHARE * min(1.0, marks / need) + (ADDRESS_SHARE if address.search(url) else 0.0)
        if like > best:
            best, kind = like, name
    return best, kind


def content_ranges(raw: bytes, min_copy: int = 64) -> tuple[list[list[int]], int]:
    """Byte ranges of the lines that are writing or technical text, and how many bytes they hold."""
    ranges: list[list[int]] = []
    pos = 0
    gap = 0  # lines since the last one kept
    for line in raw.split(b"\n"):
        n = len(line) + 1  # + the line break
        if n > 200 or (n > 60 and WRITING.search(line)) or line[-1:] in (b";", b"{", b"}") or FORM_LINE.match(line):
            if ranges and gap <= BRIDGE:
                ranges[-1][1] = pos + n
            else:
                ranges.append([pos, pos + n])
            gap = 0
        else:
            gap += 1
        pos += n
    # No ink: a run shorter than min_copy bytes would be charged, and so would one that began
    # inside a character, since its start would be moved. The last line may have no line break.
    size = len(raw)
    ranges = [[a, min(b, size)] for a, b in ranges if min(b, size) - a >= min_copy and (raw[a] & 0xC0) != 0x80]
    return ranges, sum(b - a for a, b in ranges)


def exam_like(task: Task) -> dict[str, list[tuple[float, int, list[list[int]], int]]]:
    """For each kind, (mark, document, kept ranges, kept bytes) of the pages most like it, best first."""
    corpus = task.corpus
    min_copy = int(task.info.get("min_copy_bytes", 64))
    urls = corpus.urls
    seen: set[bytes] = set()
    pages: dict[str, list] = {name: [] for name in KINDS}
    for i in range(len(corpus)):
        raw = corpus.raw(i)
        if len(raw) < 400:
            continue
        like, kind = likeness(b"\n" + raw[:HEAD_BYTES], urls[i].lower() if i < len(urls) else "")
        if like < MIN_LIKE:
            continue
        key = hashlib.blake2b(raw[:4000], digest_size=8).digest()
        if key in seen:
            continue
        seen.add(key)
        ranges, kept = content_ranges(raw, min_copy)
        if kept < 400:
            continue
        # Twice the likeness, plus what tidy asks of any page: mostly content, and enough of it.
        mark = 2 * like + kept / len(raw) + min(1.0, kept / 3000)
        pages[kind].append((mark, i, ranges, kept))
    for found in pages.values():
        found.sort(key=lambda t: (-t[0], t[1]))
    return pages


def edit(task: Task):
    budget_bytes = task.train_tokens * BYTES_PER_TOKEN
    pages = exam_like(task)
    chosen = []
    used = 0
    left = []
    for found in pages.values():  # a quarter of the budget from each kind
        got = 0
        for n, (_, i, ranges, kept) in enumerate(found):
            if got >= budget_bytes / len(pages):
                left += found[n:]
                break
            chosen.append({"d": i, "k": ranges})
            got += kept
        used += got
    left.sort(key=lambda t: (-t[0], t[1]))
    for _, i, ranges, kept in left:  # what a kind could not fill, from the best of the others
        if used >= budget_bytes:
            break
        chosen.append({"d": i, "k": ranges})
        used += kept
    if used < budget_bytes:  # the exam-like pages ran short: tidy's selection tops the budget up
        for _, i, ranges, kept in tidy_candidates(task, {line["d"] for line in chosen}):
            if used >= budget_bytes:
                break
            chosen.append({"d": i, "k": ranges})
            used += kept
    random.Random(0).shuffle(chosen)
    return chosen


def main():
    task = Task.from_env()
    task.write(edit(task))


if __name__ == "__main__":
    main()
