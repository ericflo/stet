"""Grammatical minimal pairs, made from templates: two sentences that differ in one place, one
acceptable and one not. The score is how many bits the student prefers the acceptable one by.

Outside English a pair is made only where the wrong sentence is wrong beyond argument, from words
listed here by hand in every form they are used in:

- In Russian, German, Spanish, French, Portuguese, Italian, Polish and Dutch: a subject and its
  verb, one or many. The subject is listed with its article, so nothing is put together by rule.
  The verbs take no object. Where a language may leave its subject out (Spanish, Italian,
  Portuguese, Polish), a verb that could take one would let "the doctor wait" be read as "they
  wait for the doctor", and the wrong sentence would have a reading. Most sentences end by saying
  when (today, now, in the morning, in the evening), which goes with any of the verbs and makes
  it rare for a round to ask the same pair twice.
- In Japanese and Chinese: the word a thing is counted with, beside one that is never used of it
  (three volumes of books, never three vehicles of books); and one fixed rule each: Japanese marks
  what exists with が and never を, Chinese negates 有 with 没 and never 不.
"""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import GIVEN, NAMES

NAME = "grammar"
NOUNS = [("doctor", "doctors"), ("teacher", "teachers"), ("child", "children"), ("woman", "women"), ("man", "men"),
         ("student", "students"), ("dog", "dogs"), ("artist", "artists"), ("farmer", "farmers"), ("pilot", "pilots"),
         ("baker", "bakers"), ("singer", "singers"), ("mouse", "mice"), ("person", "people"), ("goose", "geese"),
         ("nurse", "nurses"), ("lawyer", "lawyers"), ("driver", "drivers"), ("neighbour", "neighbours"), ("guest", "guests")]
INTRANSITIVE = [("sleeps", "sleep"), ("laughs", "laugh"), ("waits", "wait"), ("sings", "sing"), ("works", "work"),
                ("smiles", "smile"), ("arrives", "arrive"), ("leaves", "leave"), ("complains", "complain"), ("rests", "rest")]
TRANSITIVE = [("sees", "see", "saw"), ("likes", "like", "liked"), ("helps", "help", "helped"), ("knows", "know", "knew"),
              ("visits", "visit", "visited"), ("admires", "admire", "admired"), ("thanks", "thank", "thanked"),
              ("praises", "praise", "praised")]
PARTICIPLES = [("taken", "took"), ("written", "wrote"), ("eaten", "ate"), ("broken", "broke"), ("chosen", "chose"),
               ("given", "gave"), ("seen", "saw"), ("forgotten", "forgot"), ("driven", "drove"), ("spoken", "spoke")]
OBJECTS = ["letter", "cake", "window", "song", "plan", "map", "note", "book"]
VOWEL_NOUNS = ["apple", "orange", "egg", "onion", "umbrella", "idea", "island", "elephant", "octopus", "envelope"]
CONSONANT_NOUNS = ["banana", "pear", "table", "lamp", "river", "garden", "kettle", "horse", "pencil", "ladder"]
PHENOMENA = ["agree-simple-sg", "agree-simple-pl", "agree-attractor-sg", "agree-attractor-pl", "agree-relative",
             "det-this", "det-these", "reflexive", "participle", "article-an", "article-a", "npi-ever", "there-are"]


# A subject, one and many, and verbs for one and for many.
AGREE = {
    "ru": ([("врач", "врачи"), ("учитель", "учителя"), ("ребёнок", "дети"), ("собака", "собаки"), ("женщина", "женщины"),
            ("мальчик", "мальчики"), ("кошка", "кошки"), ("сосед", "соседи")],
           [("спит", "спят"), ("работает", "работают"), ("смеётся", "смеются"), ("приходит", "приходят"), ("бежит", "бегут"),
            ("отдыхает", "отдыхают"), ("улыбается", "улыбаются"), ("молчит", "молчат")]),
    "de": ([("der Arzt", "die Ärzte"), ("die Lehrerin", "die Lehrerinnen"), ("das Kind", "die Kinder"), ("der Hund", "die Hunde"),
            ("die Frau", "die Frauen"), ("der Mann", "die Männer"), ("die Katze", "die Katzen"), ("der Nachbar", "die Nachbarn")],
           [("schläft", "schlafen"), ("arbeitet", "arbeiten"), ("singt", "singen"), ("wartet", "warten"), ("lacht", "lachen"),
            ("kommt", "kommen"), ("läuft", "laufen"), ("spielt", "spielen")]),
    "es": ([("el médico", "los médicos"), ("la profesora", "las profesoras"), ("el niño", "los niños"), ("el perro", "los perros"),
            ("la vecina", "las vecinas"), ("el gato", "los gatos"), ("la niña", "las niñas"), ("el vecino", "los vecinos")],
           [("llega", "llegan"), ("viene", "vienen"), ("sale", "salen"), ("entra", "entran"), ("sonríe", "sonríen"),
            ("viaja", "viajan"), ("tose", "tosen"), ("trabaja", "trabajan")]),
    "fr": ([("le médecin", "les médecins"), ("la voisine", "les voisines"), ("le chat", "les chats"), ("la fille", "les filles"),
            ("le garçon", "les garçons"), ("le professeur", "les professeurs"), ("le chien", "les chiens"),
            ("la boulangère", "les boulangères")],
           [("dort", "dorment"), ("travaille", "travaillent"), ("chante", "chantent"), ("attend", "attendent"), ("arrive", "arrivent"),
            ("sourit", "sourient"), ("part", "partent"), ("court", "courent")]),
    "pt": ([("o médico", "os médicos"), ("a professora", "as professoras"), ("o menino", "os meninos"), ("o cão", "os cães"),
            ("a menina", "as meninas"), ("o gato", "os gatos"), ("a vizinha", "as vizinhas"), ("o vizinho", "os vizinhos")],
           [("dorme", "dormem"), ("chega", "chegam"), ("ri", "riem"), ("trabalha", "trabalham"), ("viaja", "viajam"),
            ("caminha", "caminham"), ("sorri", "sorriem"), ("tosse", "tossem")]),
    "it": ([("il medico", "i medici"), ("la maestra", "le maestre"), ("il bambino", "i bambini"), ("il cane", "i cani"),
            ("la ragazza", "le ragazze"), ("il gatto", "i gatti"), ("la vicina", "le vicine"), ("il ragazzo", "i ragazzi")],
           [("dorme", "dormono"), ("arriva", "arrivano"), ("ride", "ridono"), ("parte", "partono"), ("cammina", "camminano"),
            ("viaggia", "viaggiano"), ("sorride", "sorridono"), ("tossisce", "tossiscono")]),
    "pl": ([("lekarz", "lekarze"), ("nauczyciel", "nauczyciele"), ("dziecko", "dzieci"), ("pies", "psy"), ("kobieta", "kobiety"),
            ("student", "studenci"), ("kot", "koty"), ("sąsiad", "sąsiedzi")],
           [("śpi", "śpią"), ("pracuje", "pracują"), ("czeka", "czekają"), ("biegnie", "biegną"), ("odpoczywa", "odpoczywają"),
            ("idzie", "idą"), ("płacze", "płaczą"), ("siedzi", "siedzą")]),
    "nl": ([("de dokter", "de dokters"), ("de lerares", "de leraressen"), ("het kind", "de kinderen"), ("de hond", "de honden"),
            ("de vrouw", "de vrouwen"), ("de man", "de mannen"), ("de kat", "de katten"), ("de jongen", "de jongens")],
           [("slaapt", "slapen"), ("werkt", "werken"), ("zingt", "zingen"), ("wacht", "wachten"), ("lacht", "lachen"),
            ("komt", "komen"), ("loopt", "lopen"), ("speelt", "spelen")]),
}
# When: today, now, in the morning, in the evening. After the verb, as each language may put it.
WHEN = {
    "ru": ["сегодня", "сейчас", "утром", "вечером"], "de": ["heute", "jetzt", "morgens", "abends"],
    "es": ["hoy", "ahora", "por la mañana", "por la noche"], "fr": ["aujourd'hui", "maintenant", "le matin", "le soir"],
    "pt": ["hoje", "agora", "de manhã", "à noite"], "it": ["oggi", "adesso", "di mattina", "di sera"],
    "pl": ["dziś", "teraz", "rano", "wieczorem"], "nl": ["vandaag", "nu", "'s ochtends", "'s avonds"],
}
# A thing, the word it is counted with, and words it is never counted with.
JA_COUNTED = [("本", "冊", ["匹", "台"]), ("雑誌", "冊", ["匹", "台"]), ("犬", "匹", ["冊", "台"]), ("猫", "匹", ["冊", "台"]),
              ("切手", "枚", ["匹", "台"]), ("皿", "枚", ["匹", "台"]), ("車", "台", ["匹", "冊"]), ("自転車", "台", ["匹", "冊"]),
              ("鉛筆", "本", ["匹", "台"]), ("傘", "本", ["匹", "台"])]
# None of them alive: they take あります.
JA_THINGS = ["本", "雑誌", "切手", "皿", "鉛筆", "傘", "鍵", "時計", "地図", "手紙", "財布", "眼鏡", "写真", "辞書", "新聞", "帽子"]
JA_PLACES = ["机の上", "部屋の中", "箱の中", "棚の上", "椅子の上", "かばんの中", "引き出しの中", "テーブルの上"]
ZH_COUNTED = [("书", "本", ["辆", "匹"]), ("杂志", "本", ["辆", "匹"]), ("马", "匹", ["辆", "本"]), ("汽车", "辆", ["匹", "本"]),
              ("自行车", "辆", ["匹", "本"]), ("桌子", "张", ["辆", "匹"]), ("猫", "只", ["辆", "本"]), ("铅笔", "支", ["辆", "匹"]),
              ("衣服", "件", ["辆", "本"]), ("纸", "张", ["辆", "匹"])]
ZH_THINGS = ["书", "汽车", "自行车", "铅笔", "钥匙", "地图", "手表", "雨伞"]
NUMERALS = ["三", "四", "五", "六", "七", "八", "九"]  # the same in both, and none of them has a second form before a counter
OTHER = {**{lang: ["agree-sg", "agree-pl"] for lang in AGREE}, "ja": ["counter", "exist"], "zh": ["classifier", "negation"]}


def corners() -> list[str]:
    return [f"{NAME}/en/{p}" for p in PHENOMENA] + [f"{NAME}/{lang}/{p}" for lang, ps in OTHER.items() for p in ps]


def _other(rng: Rng, lang: str, k: str) -> tuple[str, str]:
    """A pair in a language other than English: the acceptable sentence, then the other."""
    if k in ("agree-sg", "agree-pl"):
        nouns, verbs = AGREE[lang]
        (one, many), (v_one, v_many) = rng.choice(nouns), rng.choice(verbs)
        when = rng.choice(["", *WHEN[lang]])
        end = f" {when}." if when else "."
        return (f"{one} {v_one}{end}", f"{one} {v_many}{end}") if k == "agree-sg" else (f"{many} {v_many}{end}", f"{many} {v_one}{end}")
    who, n = rng.choice(GIVEN[lang]), rng.choice(NUMERALS)
    if k == "counter":
        thing, right, wrong = rng.choice(JA_COUNTED)
        return f"{who}は{thing}を{n}{right}買いました。", f"{who}は{thing}を{n}{rng.choice(wrong)}買いました。"
    if k == "exist":
        place, thing = rng.choice(JA_PLACES), rng.choice(JA_THINGS)
        return f"{place}に{thing}があります。", f"{place}に{thing}をあります。"
    if k == "classifier":
        thing, right, wrong = rng.choice(ZH_COUNTED)
        return f"{who}有{n}{right}{thing}。", f"{who}有{n}{rng.choice(wrong)}{thing}。"
    if k == "negation":
        thing = rng.choice(ZH_THINGS)
        return f"{who}没有{thing}。", f"{who}不有{thing}。"
    raise ValueError(k)


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def make(rng: Rng, corner: str) -> Item:
    _, lang, k = corner.split("/")
    if lang != "en":
        good, bad = _other(rng, lang, k)
        return Item(kind="pair", corner=corner, fmt="prose", lang=lang, source=f"gen:{NAME}", good=_cap(good), bad=_cap(bad))
    (n1s, n1p), (n2s, n2p) = rng.sample(NOUNS, 2)
    vs, vp = rng.choice(INTRANSITIVE)
    ts, tp, tpast = rng.choice(TRANSITIVE)
    name = rng.choice(NAMES)
    if k == "agree-simple-sg":
        good, bad = f"The {n1s} {vs}.", f"The {n1s} {vp}."
    elif k == "agree-simple-pl":
        good, bad = f"The {n1p} {vp}.", f"The {n1p} {vs}."
    elif k == "agree-attractor-sg":
        good, bad = f"The {n1s} near the {n2p} {vs}.", f"The {n1s} near the {n2p} {vp}."
    elif k == "agree-attractor-pl":
        good, bad = f"The {n1p} beside the {n2s} {vp}.", f"The {n1p} beside the {n2s} {vs}."
    elif k == "agree-relative":
        good, bad = f"The {n1s} that the {n2p} {tp} {vs}.", f"The {n1s} that the {n2p} {tp} {vp}."
    elif k == "det-this":
        good, bad = f"{name} {ts} this {n1s}.", f"{name} {ts} this {n1p}."
    elif k == "det-these":
        good, bad = f"{name} {ts} these {n1p}.", f"{name} {ts} these {n1s}."
    elif k == "reflexive":
        good, bad = f"The {n1p} {tpast} themselves.", f"The {n1p} {tpast} itself."
    elif k == "participle":
        part, past = rng.choice(PARTICIPLES)
        obj = rng.choice(OBJECTS)
        good, bad = f"{name} has {part} the {obj}.", f"{name} has {past} the {obj}."
    elif k == "article-an":
        n = rng.choice(VOWEL_NOUNS)
        good, bad = f"{name} found an {n}.", f"{name} found a {n}."
    elif k == "article-a":
        n = rng.choice(CONSONANT_NOUNS)
        good, bad = f"{name} found a {n}.", f"{name} found an {n}."
    elif k == "npi-ever":
        part, _ = rng.choice(PARTICIPLES)
        obj = rng.choice(OBJECTS)
        good, bad = f"Nobody has ever {part} that {obj}.", f"Somebody has ever {part} that {obj}."
    elif k == "there-are":
        good, bad = f"There are many {n1p} here.", f"There are many {n1s} here."
    else:
        raise ValueError(corner)
    return Item(kind="pair", corner=corner, fmt="prose", lang="en", source=f"gen:{NAME}", good=_cap(good), bad=_cap(bad))
