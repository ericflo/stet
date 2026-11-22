"""The reference curator people fork, and the starter `stet init` writes, in a season whose exam
is written in every language of the crawl."""

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from conftest import FakeEnc, make_corpus

from stet.editors import tidy
from stet.script import materialize, write_script
from stet.spec import load_season

ROOT = Path(__file__).resolve().parents[1]
SEASON = load_season("s2")
SCALE = dataclasses.replace(SEASON.scale("nano"), train_tokens=4096, ctx=64, batch_size=4)

NAV = "Home | About | Contact\nCopyright 2019\n"
EN = "The court held that the statute did not apply to contracts signed before it passed. " * 12
RU = "Суд постановил, что закон не применяется к договорам, подписанным до его принятия. " * 12
JA = "裁判所は、その法律は成立前に署名された契約には適用されないと判断した。\n" * 14
AR = "قضت المحكمة بأن القانون لا ينطبق على العقود الموقعة قبل إقراره، وهذا ما أكده الحكم. " * 12
HI = "अदालत ने कहा कि यह कानून उसके पारित होने से पहले हस्ताक्षरित अनुबंधों पर लागू नहीं होता है। " * 12
SHOP = "BUY NOW | 19.99 | add to cart\n" * 60
TAGS = "news sport weather travel money\n" * 80


def _task(docs, tmp_path, tokens=4096):
    corpus = make_corpus(tmp_path / "corpus", docs)
    return SimpleNamespace(corpus=corpus, train_tokens=tokens, info={"min_copy_bytes": 64}), corpus


def test_tidy_keeps_writing_in_every_script_and_only_takes_lines_out(tmp_path):
    docs = [NAV + EN, NAV + RU, NAV + JA, NAV + AR, NAV + HI, SHOP, TAGS, "short", NAV + EN]
    task, corpus = _task(docs, tmp_path)
    script = tidy.edit(task)
    kept = {line["d"] for line in script}
    assert kept == {0, 1, 2, 3, 4}  # every language's page; not the shop, the tags, the stub or the copy
    for line in script:
        text = b"".join(corpus.raw(line["d"])[a:b] for a, b in line["k"]).decode()
        assert "Home | About" not in text and "Copyright" not in text and len(text) > 400
    # Whole lines of a page in the page's order: not a byte of ink, where the season allows none.
    path = tmp_path / "script.jsonl"
    write_script(path, script)
    _, stats = materialize(corpus, path, SEASON, SCALE, FakeEnc())
    assert stats.ink_tokens == 0 and stats.src_docs == 5 and stats.out_docs == 5
    assert tidy.edit(task) == script  # the same script every time


def test_tidy_cuts_lines_exactly_where_a_page_is_not_valid_text(tmp_path):
    # A page whose bytes are not all UTF-8: offsets are counted in the page's own bytes.
    corpus = make_corpus(tmp_path / "corpus", ["x"])
    raw = b"Menu \xff\xfe | Home\n" + (EN.encode() + b"\n") * 2 + b"\xff footer"
    ranges, kept = tidy.prose_ranges(raw, dense=False)
    assert ranges == [[len(b"Menu \xff\xfe | Home\n"), len(raw) - len(b"\xff footer")]] and kept == 2 * (len(EN.encode()) + 1)
    # The last line, with no line break after it, ends where the page does.
    ranges, _ = tidy.prose_ranges(EN.encode(), dense=False)
    assert ranges == [[0, len(EN.encode())]] and corpus is not None


def test_tidy_fills_the_budget_once_with_the_pages_that_are_most_writing(tmp_path):
    docs = [NAV * 40 + EN, EN * 3, RU * 3, NAV * 40 + RU]
    task, _ = _task(docs, tmp_path, tokens=1200)  # room for two of the clean pages
    assert {line["d"] for line in tidy.edit(task)} == {1, 2}


def test_the_starter_keeps_every_language_too(tmp_path):
    spec = importlib.util.spec_from_file_location("starter", ROOT / "src" / "stet" / "templates" / "curator.py")
    starter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter)
    assert all(starter.looks_written(t) for t in (EN, RU, JA, AR, HI))
    assert not starter.looks_written("19.99 | 24.50 | 7.25 | $$$\n" * 40) and not starter.looks_written("short")
    assert "English" not in (ROOT / "src" / "stet" / "templates" / "curator.py").read_text()
