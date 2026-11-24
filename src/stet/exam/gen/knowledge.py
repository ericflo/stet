"""Facts the web knows, asked in ten languages: where someone was born, a country's capital, who
wrote a book or directed a film, an element's symbol.

The facts are the season's list (data/facts-s1.jsonl.zst, built from Wikidata by
tools/build_wikidata_facts.py): well-known subjects, each with one undoubted answer. Which are
asked is the round's seed's choice. Knowing them is what a model learns from pages that say
them; a curator that drops such pages loses them, and one that keeps them gains.

The sentences are written so that they read right whatever the noun's gender or case: where a
language would need an article or an inflection the label does not have, the fact is put as
a heading and a field, as an infobox puts it.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources

from ..items import Item
from ..rng import Rng

NAME = "knowledge"
LANGS = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "ja", "zh"]
_COUNTRY = {"en": "{s}. Country:", "es": "{s}. País:", "fr": "{s}. Pays :", "de": "{s}. Staat:", "it": "{s}. Stato:", "pt": "{s}. País:",
            "nl": "{s}. Land:", "ru": "{s}. Страна:", "ja": "{s}の所在国：", "zh": "{s}所在国家："}
TEMPLATES = {
    "birthplace": {"en": "{s} was born in", "es": "{s} nació en", "fr": "Le lieu de naissance de {s} est", "de": "Der Geburtsort von {s} ist",
                   "it": "Il luogo di nascita di {s} è", "pt": "O local de nascimento de {s} é", "nl": "De geboorteplaats van {s} is",
                   "ru": "{s}. Место рождения:", "ja": "{s}の出生地は", "zh": "{s}的出生地是"},
    "birthyear": {"en": "{s} was born in the year", "es": "{s} nació en el año", "fr": "L'année de naissance de {s} est",
                  "de": "Das Geburtsjahr von {s} ist", "it": "L'anno di nascita di {s} è", "pt": "O ano de nascimento de {s} é",
                  "nl": "Het geboortejaar van {s} is", "ru": "{s}. Год рождения:", "ja": "{s}の生年は", "zh": "{s}的出生年份是"},
    "capital": {"en": "{s}. Capital:", "es": "La capital de {s} es", "fr": "La capitale de {s} est", "de": "Die Hauptstadt von {s} ist",
                "it": "La capitale di {s} è", "pt": "A capital de {s} é", "nl": "De hoofdstad van {s} is", "ru": "{s}. Столица:",
                "ja": "{s}の首都は", "zh": "{s}的首都是"},
    "currency": {"en": "{s}. Currency:", "es": "{s}. Moneda:", "fr": "{s}. Monnaie :", "de": "{s}. Währung:", "it": "{s}. Valuta:",
                 "pt": "{s}. Moeda:", "nl": "{s}. Munteenheid:", "ru": "{s}. Валюта:", "ja": "{s}の通貨は", "zh": "{s}的货币是"},
    "city-country": _COUNTRY,
    "mountain-country": _COUNTRY,
    "author": {"en": "{s} was written by", "es": "{s} fue escrito por", "fr": "{s} a été écrit par", "de": "{s} wurde geschrieben von",
               "it": "{s} è stato scritto da", "pt": "{s} foi escrito por", "nl": "{s} is geschreven door", "ru": "{s}. Автор:",
               "ja": "{s}の著者は", "zh": "{s}的作者是"},
    "director": {"en": "{s} was directed by", "es": "{s}. Dirección:", "fr": "{s} a été réalisé par", "de": "{s}. Regie:",
                 "it": "{s} è stato diretto da", "pt": "{s} foi dirigido por", "nl": "{s} werd geregisseerd door", "ru": "{s}. Режиссёр:",
                 "ja": "{s}の監督は", "zh": "{s}的导演是"},
    "painter": {"en": "{s} was painted by", "es": "{s} fue pintado por", "fr": "{s} a été peint par", "de": "{s} wurde gemalt von",
                "it": "{s} è stato dipinto da", "pt": "{s} foi pintado por", "nl": "{s} is geschilderd door", "ru": "{s}. Автор:",
                "ja": "{s}の作者は", "zh": "{s}的作者是"},
    "symbol": {"en": "The chemical symbol of {s} is", "es": "{s}. Símbolo químico:", "fr": "{s}. Symbole :", "de": "Das chemische Symbol von {s} ist",
               "it": "{s}. Simbolo:", "pt": "{s}. Símbolo:", "nl": "Het chemische symbool van {s} is", "ru": "{s}. Символ:",
               "ja": "{s}の元素記号は", "zh": "{s}的元素符号是"},
}
NO_SPACE = {"ja", "zh"}  # the answer follows without a space


@cache
def facts_with_ids() -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    """(fact, language) -> [(wikidata_qid, subject_label, answer_label)], in the list's order."""
    import zstandard

    raw = resources.files("stet.exam").joinpath("data/facts-s1.jsonl.zst").read_bytes()
    out: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for line in zstandard.ZstdDecompressor().decompress(raw, max_output_size=256 << 20).decode().splitlines():
        row = json.loads(line)
        qid = str(row["s"])
        for lang, (s, o) in row["labels"].items():
            out.setdefault((row["fact"], lang), []).append((qid, s, o))
    return out


@cache
def facts() -> dict[tuple[str, str], list[tuple[str, str]]]:
    """(fact, language) -> [(subject, answer)], in the list's order."""
    return {k: [(s, o) for _, s, o in rows] for k, rows in facts_with_ids().items()}


def corners() -> list[str]:
    return [f"{NAME}/{lang}/{fact}" for fact in TEMPLATES for lang in LANGS]


def make(rng: Rng, corner: str) -> Item:
    _, lang, fact = corner.split("/")
    pool = facts().get((fact, lang)) or facts()[(fact, "en")]
    if not facts().get((fact, lang)):
        lang = "en"
    s, o = rng.choice(pool)
    if fact == "birthyear":
        o = str(int(o))  # 0300 is 300
    ctx = TEMPLATES[fact][lang].format(s=s)
    return Item(kind="answer", corner=corner, fmt="prose", lang=lang, source=f"gen:{NAME}", ctx=ctx,
                cont=o if lang in NO_SPACE else f" {o}")


def _qid_parity(qid: str) -> int:
    import hashlib

    return hashlib.sha256(qid.encode("utf-8")).digest()[0] & 1


def split_pool_with_ids(fact: str, lang: str, parity: int) -> tuple[list[tuple[str, str, str]], str]:
    """Split a (fact, lang) pool deterministically by canonical Wikidata subject Q-ID parity:
    0 = exam pool (knowledge-mc), 1 = disjoint SFT probe pool (zero cross-language fact overlap)."""
    full = facts_with_ids().get((fact, lang))
    if not full:
        lang = "en"
        full = facts_with_ids()[(fact, "en")]
    part = [(qid, s, o) for qid, s, o in full if _qid_parity(qid) == parity]
    return part, lang


def split_pool(fact: str, lang: str, parity: int) -> tuple[list[tuple[str, str]], str]:
    """Split a (fact, lang) pool deterministically by canonical Wikidata subject Q-ID parity:
    0 = exam pool (knowledge-mc), 1 = disjoint SFT probe pool (zero cross-language fact overlap)."""
    part, lang = split_pool_with_ids(fact, lang, parity)
    return [(s, o) for _, s, o in part], lang


class KnowledgeMC:
    """Contrastive factual knowledge: right answer vs 3 same-relation, same-language distractors.

    Uses the even-parity split of Wikidata facts (disjoint from the SFT probe's odd-parity split).
    """

    NAME = "knowledge-mc"

    @staticmethod
    def corners() -> list[str]:
        return [f"{KnowledgeMC.NAME}/{lang}/{fact}" for fact in TEMPLATES for lang in LANGS]

    @staticmethod
    def make(rng: Rng, corner: str) -> Item:
        _, lang, fact = corner.split("/")
        pool, lang = split_pool(fact, lang, parity=0)
        s, o = rng.choice(pool)
        fmt_o = (lambda v: str(int(v))) if fact == "birthyear" else (lambda v: v)
        right_val = fmt_o(o)
        # Draw 3 distinct wrong answers from the exact same relation and language.
        other_vals = sorted({fmt_o(o2) for _, o2 in pool if fmt_o(o2) != right_val})
        if len(other_vals) < 3:
            full = facts().get((fact, lang)) or facts()[(fact, "en")]
            other_vals = sorted({fmt_o(o2) for _, o2 in full if fmt_o(o2) != right_val})
        wrongs_raw = rng.sample(other_vals, min(3, len(other_vals)))
        prefix = "" if lang in NO_SPACE else " "
        ctx = TEMPLATES[fact][lang].format(s=s)
        return Item(
            kind="choice",
            corner=corner,
            fmt="prose",
            lang=lang,
            source=f"gen:{KnowledgeMC.NAME}",
            ctx=ctx,
            cont=f"{prefix}{right_val}",
            wrongs=tuple(f"{prefix}{w}" for w in wrongs_raw),
        )
