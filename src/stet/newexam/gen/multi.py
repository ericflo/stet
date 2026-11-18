"""Arithmetic asked in other languages, so generated items are not all English."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng

NAME = "multi"
# language: (question template, answer label, words for plus, minus, times, space before the answer)
LANGS = {
    "es": ("¿Cuánto es {a} {op} {b}?", "Respuesta:", ("más", "menos", "por"), " "),
    "fr": ("Combien font {a} {op} {b} ?", "Réponse :", ("plus", "moins", "fois"), " "),
    "de": ("Was ist {a} {op} {b}?", "Antwort:", ("plus", "minus", "mal"), " "),
    "pt": ("Quanto é {a} {op} {b}?", "Resposta:", ("mais", "menos", "vezes"), " "),
    "it": ("Quanto fa {a} {op} {b}?", "Risposta:", ("più", "meno", "per"), " "),
    "nl": ("Hoeveel is {a} {op} {b}?", "Antwoord:", ("plus", "min", "keer"), " "),
    "pl": ("Ile to jest {a} {op} {b}?", "Odpowiedź:", ("plus", "minus", "razy"), " "),
    "ru": ("Сколько будет {a} {op} {b}?", "Ответ:", ("плюс", "минус", "умножить на"), " "),
    "tr": ("{a} {op} {b} kaç eder?", "Cevap:", ("artı", "eksi", "çarpı"), " "),
    "id": ("Berapa {a} {op} {b}?", "Jawaban:", ("ditambah", "dikurangi", "dikali"), " "),
    "ja": ("{a}{op}{b}はいくつですか？", "答え：", ("足す", "引く", "掛ける"), ""),
    "zh": ("{a}{op}{b}等于多少？", "答案：", ("加", "减", "乘以"), ""),
}
OPS = ["add", "sub", "mul"]
SIZES = ["small", "medium"]


def corners() -> list[str]:
    return [f"{NAME}/{lang}/{op}/{size}" for lang in LANGS for op in OPS for size in SIZES]


def make(rng: Rng, corner: str) -> Item:
    _, lang, op, size = corner.split("/")
    q, label, words, space = LANGS[lang]
    hi = 99 if size == "small" else 999
    a, b = rng.between(2, hi), rng.between(2, hi if op != "mul" else (12 if size == "small" else 99))
    if op == "sub" and a < b:
        a, b = b, a
    ans = {"add": a + b, "sub": a - b, "mul": a * b}[op]
    word = words[OPS.index(op)]
    return Item(kind="answer", corner=corner, fmt="math", lang=lang, source=f"gen:{NAME}",
                ctx=f"{q.format(a=a, op=f' {word} ' if space else word, b=b).replace('  ', ' ')}\n{label}", cont=f"{space}{ans}")
