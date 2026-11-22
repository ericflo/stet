"""Words the generators draw from. Names come from many places, so no one culture's text is favoured.

The lists are short and public, so a word from them can be learned by heart. Where an answer
should rest on reading the question and nothing else, a generator coins its words from the seed.

The exam's skills are asked in the main languages of the web, not in English alone. Each of
those languages has here a short list of everyday nouns, of given names and of cities, in its own
script, and the few fixed words a question needs. A generator puts these words only where a
sentence takes them as they are written here: a name as the subject, a noun inside quotation
marks or in a file. None of them is ever inflected by code.
"""

_ONSET = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kl", "pl", "st", "tr"]
_VOWEL = ["a", "e", "i", "o", "u"]
_CODA = ["", "", "", "n", "r", "s", "l", "k", "m"]
# The same for a Russian reader, a Japanese one and a Chinese one. Every coined word begins with a
# consonant, in every script: French and Italian sentences rely on it (de Brano, never d'Arano).
_ONSET_RU = ["б", "д", "ф", "г", "к", "л", "м", "н", "п", "р", "с", "т", "в", "з", "бр", "др", "кл", "пл", "ст", "тр"]
_VOWEL_RU = ["а", "е", "и", "о", "у"]
_CODA_RU = ["", "", "", "н", "р", "с", "л", "к", "м"]
_KANA = list("カキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモラリルレロ")
# Characters Chinese writes foreign names with. Three of them read as a name from far away.
_HANZI = list("卡拉莫尼塔洛维萨米巴里诺伦达玛索")


def coin(rng, lang: str = "en", syllables: int | None = None) -> str:
    """A word nobody has written: two or three syllables that can be said, from the seed, in the
    script the language is written in."""
    if lang == "ja":
        return "".join(rng.choice(_KANA) for _ in range(syllables or rng.between(3, 4))) + rng.choice(["", "", "ン"])
    if lang == "zh":
        return "".join(rng.choice(_HANZI) for _ in range(syllables or 3))
    onset, vowel, coda = (_ONSET_RU, _VOWEL_RU, _CODA_RU) if lang == "ru" else (_ONSET, _VOWEL, _CODA)
    return "".join(rng.choice(onset) + rng.choice(vowel) for _ in range(syllables or rng.between(2, 3))) + rng.choice(coda)


def coin_name(rng, lang: str = "en") -> str:
    """A coined word as a name, with a capital where the script has capitals. Three syllables: two
    are too often a place that exists (Lima, Баку)."""
    return coin(rng, lang, 3).capitalize()


NAMES = [
    "Amara", "Bilal", "Chen", "Dara", "Elif", "Farah", "Goran", "Hana", "Ines", "Jomo", "Kaito", "Lena", "Mateo", "Nadia",
    "Oren", "Priya", "Quentin", "Rosa", "Sanjay", "Tariq", "Uma", "Viktor", "Wen", "Ximena", "Yusuf", "Zara", "Aiko",
    "Bruno", "Carmen", "Dmitri", "Esi", "Femi", "Greta", "Hugo", "Isha", "Jonas", "Kofi", "Leila", "Malik", "Nora",
    "Olu", "Paulo", "Rania", "Soren", "Thandi", "Ugo", "Vera", "Wanjiru", "Yara", "Zeynep", "Arjun", "Beatriz", "Cyrus",
    "Dalia", "Emeka", "Fatima", "Gustavo", "Hiroshi", "Ingrid", "Jamal", "Kavya", "Luca", "Mei", "Nikos", "Ola", "Pedro",
    "Rahel", "Sione", "Tomas", "Ulla", "Valentina", "Wiremu", "Xin", "Yuki", "Zainab", "Anan", "Bo", "Catalina", "Deniz",
]

OBJECTS = [
    ("apple", "apples"), ("pencil", "pencils"), ("notebook", "notebooks"), ("ticket", "tickets"), ("mango", "mangoes"),
    ("stamp", "stamps"), ("cup", "cups"), ("book", "books"), ("candle", "candles"), ("lemon", "lemons"), ("plant", "plants"),
    ("battery", "batteries"), ("brush", "brushes"), ("towel", "towels"), ("bottle of water", "bottles of water"),
    ("loaf of bread", "loaves of bread"), ("box of tea", "boxes of tea"), ("poster", "posters"), ("sticker", "stickers"),
    ("pair of socks", "pairs of socks"), ("onion", "onions"), ("marker", "markers"), ("bowl", "bowls"), ("kite", "kites"),
]

PLACES = [
    "Lagos", "Lima", "Oslo", "Hanoi", "Quito", "Nairobi", "Lyon", "Osaka", "Porto", "Tbilisi", "Accra", "Cusco", "Perth",
    "Bergen", "Dakar", "Kraków", "Medellín", "Pune", "Ghent", "Hobart", "Izmir", "Kyoto", "Mombasa", "Recife", "Tartu",
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

WORDS = [
    "river", "garden", "lantern", "harbor", "violet", "copper", "meadow", "pepper", "signal", "marble", "orbit", "canyon",
    "falcon", "ribbon", "summit", "pillow", "thunder", "basket", "cactus", "ember", "glacier", "hammer", "island", "jacket",
    "kettle", "ladder", "magnet", "needle", "oyster", "parrot", "quartz", "rocket", "saddle", "tunnel", "velvet", "walnut",
]

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "grey", "brown"]

# --- the same in the web's main languages ----------------------------------------------------------

# The languages the skills are asked in, the largest first (a Common Crawl sample's order).
LANGS = ["en", "ru", "de", "ja", "zh", "es", "fr", "pt", "it", "pl", "nl"]
NO_SPACE = {"ja", "zh"}  # written without spaces: an answer follows its label directly

# Everyday things, as a dictionary lists them: one word each, and nearly the same forty in every list.
NOUNS = {
    "en": WORDS,
    "ru": ["река", "сад", "окно", "стол", "книга", "дом", "мост", "лампа", "чашка", "ключ", "письмо", "дорога", "лес", "море",
           "озеро", "гора", "хлеб", "молоко", "яблоко", "цветок", "дерево", "птица", "рыба", "лошадь", "собака", "кошка", "часы",
           "дверь", "стул", "карта", "лодка", "поезд", "город", "остров", "камень", "песок", "снег", "дождь", "ветер", "облако"],
    "de": ["Fluss", "Garten", "Fenster", "Tisch", "Buch", "Haus", "Brücke", "Lampe", "Tasse", "Schlüssel", "Brief", "Straße", "Wald",
           "Meer", "See", "Berg", "Brot", "Milch", "Apfel", "Blume", "Baum", "Vogel", "Fisch", "Pferd", "Hund", "Katze", "Uhr", "Tür",
           "Stuhl", "Karte", "Boot", "Zug", "Stadt", "Insel", "Stein", "Sand", "Schnee", "Regen", "Wind", "Wolke"],
    "ja": ["川", "庭", "窓", "机", "本", "家", "橋", "ランプ", "コップ", "鍵", "手紙", "道", "森", "海", "湖", "山", "パン", "牛乳", "りんご", "花",
           "木", "鳥", "魚", "馬", "犬", "猫", "時計", "ドア", "椅子", "地図", "船", "電車", "町", "島", "石", "砂", "雪", "雨", "風", "雲"],
    "zh": ["河", "花园", "窗户", "桌子", "书", "房子", "桥", "灯", "杯子", "钥匙", "信", "路", "森林", "海", "湖", "山", "面包", "牛奶", "苹果",
           "花", "树", "鸟", "鱼", "马", "狗", "猫", "钟", "门", "椅子", "地图", "船", "火车", "城市", "岛", "石头", "沙子", "雪", "雨", "风", "云"],
    "es": ["río", "jardín", "ventana", "mesa", "libro", "casa", "puente", "lámpara", "taza", "llave", "carta", "camino", "bosque", "mar",
           "lago", "montaña", "pan", "leche", "manzana", "flor", "árbol", "pájaro", "pez", "caballo", "perro", "gato", "reloj", "puerta",
           "silla", "mapa", "barco", "tren", "ciudad", "isla", "piedra", "arena", "nieve", "lluvia", "viento", "nube"],
    "fr": ["rivière", "jardin", "fenêtre", "table", "livre", "maison", "pont", "lampe", "tasse", "clé", "lettre", "chemin", "forêt", "mer",
           "lac", "montagne", "pain", "lait", "pomme", "fleur", "arbre", "oiseau", "poisson", "cheval", "chien", "chat", "horloge", "porte",
           "chaise", "carte", "bateau", "train", "ville", "île", "pierre", "sable", "neige", "pluie", "vent", "nuage"],
    "pt": ["rio", "jardim", "janela", "mesa", "livro", "casa", "ponte", "lâmpada", "copo", "chave", "carta", "caminho", "floresta", "mar",
           "lago", "montanha", "pão", "leite", "maçã", "flor", "árvore", "pássaro", "peixe", "cavalo", "cão", "gato", "relógio", "porta",
           "cadeira", "mapa", "barco", "estrada", "cidade", "ilha", "pedra", "areia", "neve", "chuva", "vento", "nuvem"],
    "it": ["fiume", "giardino", "finestra", "tavolo", "libro", "casa", "ponte", "lampada", "tazza", "chiave", "lettera", "strada", "bosco",
           "mare", "lago", "montagna", "pane", "latte", "mela", "fiore", "albero", "uccello", "pesce", "cavallo", "cane", "gatto",
           "orologio", "porta", "sedia", "mappa", "barca", "treno", "città", "isola", "pietra", "sabbia", "neve", "pioggia", "vento", "nuvola"],
    "pl": ["rzeka", "ogród", "okno", "stół", "książka", "dom", "most", "lampa", "kubek", "klucz", "list", "droga", "las", "morze",
           "jezioro", "góra", "chleb", "mleko", "jabłko", "kwiat", "drzewo", "ptak", "ryba", "koń", "pies", "kot", "zegar", "drzwi",
           "krzesło", "mapa", "łódź", "pociąg", "miasto", "wyspa", "kamień", "piasek", "śnieg", "deszcz", "wiatr", "chmura"],
    "nl": ["rivier", "tuin", "raam", "tafel", "boek", "huis", "brug", "lamp", "kopje", "sleutel", "brief", "weg", "bos", "zee", "meer",
           "berg", "brood", "melk", "appel", "bloem", "boom", "vogel", "vis", "paard", "hond", "kat", "klok", "deur", "stoel", "kaart",
           "boot", "trein", "stad", "eiland", "steen", "zand", "sneeuw", "regen", "wind", "wolk"],
}

# Given names (in Chinese, full names, as a Chinese sentence would have them). The French and the
# Italian ones all begin with a consonant, so the "que" or "di" before one is never shortened.
GIVEN = {
    "en": NAMES,
    "ru": ["Анна", "Борис", "Вера", "Глеб", "Дарья", "Егор", "Жанна", "Захар", "Ирина", "Кирилл", "Лариса", "Максим", "Нина", "Олег",
           "Полина", "Роман", "Светлана", "Тимур", "Ульяна", "Фёдор"],
    "de": ["Anna", "Bernd", "Clara", "Dieter", "Emma", "Felix", "Greta", "Hans", "Ingrid", "Jonas", "Katrin", "Lukas", "Marie", "Niklas",
           "Paula", "Rolf", "Sabine", "Tobias", "Ulrike", "Werner"],
    "ja": ["太郎", "花子", "健太", "美咲", "翔太", "陽子", "大輔", "由美", "拓也", "恵子", "直樹", "明美", "和也", "智子", "雄一", "真由美",
           "浩二", "久美子", "一郎", "裕子"],
    "zh": ["王芳", "李伟", "张敏", "刘洋", "陈静", "杨磊", "赵丽", "黄强", "周杰", "吴霞", "徐明", "孙丽华", "马超", "朱婷", "胡军", "郭燕",
           "何平", "高建国", "林娜", "罗刚"],
    "es": ["Ana", "Carlos", "María", "Javier", "Lucía", "Miguel", "Carmen", "Pablo", "Elena", "Diego", "Sofía", "Andrés", "Marta",
           "Fernando", "Laura", "Sergio", "Paula", "Manuel", "Rosa", "Daniel"],
    "fr": ["Marie", "Pierre", "Camille", "Louis", "Sophie", "Julien", "Chloé", "Nicolas", "Manon", "Thomas", "Léa", "Paul", "Claire",
           "Mathieu", "Charlotte", "Bernard", "Nathalie", "Denis", "Julie", "François"],
    "pt": ["Ana", "João", "Maria", "Pedro", "Beatriz", "Tiago", "Carla", "Rui", "Inês", "Miguel", "Sofia", "André", "Marta", "Paulo",
           "Helena", "Bruno", "Teresa", "Nuno", "Luísa", "Ricardo"],
    "it": ["Marco", "Giulia", "Luca", "Sara", "Matteo", "Chiara", "Paolo", "Francesca", "Stefano", "Laura", "Davide", "Martina",
           "Giovanni", "Silvia", "Roberto", "Valentina", "Simone", "Federica", "Pietro", "Maria"],
    "pl": ["Anna", "Piotr", "Maria", "Tomasz", "Katarzyna", "Marek", "Agnieszka", "Paweł", "Ewa", "Krzysztof", "Magdalena", "Andrzej",
           "Joanna", "Jan", "Barbara", "Michał", "Zofia", "Adam", "Dorota", "Wojciech"],
    "nl": ["Anna", "Daan", "Sanne", "Bram", "Lotte", "Thijs", "Emma", "Ruben", "Sophie", "Lars", "Julia", "Sven", "Fleur", "Jeroen",
           "Iris", "Pieter", "Marieke", "Koen", "Eva", "Willem"],
}

CITIES = {
    "en": PLACES,
    "ru": ["Москва", "Казань", "Самара", "Омск", "Пермь", "Уфа", "Тула", "Сочи"],
    "de": ["Berlin", "Hamburg", "München", "Köln", "Leipzig", "Dresden", "Bremen", "Kiel"],
    "ja": ["東京", "大阪", "京都", "札幌", "福岡", "名古屋", "神戸", "仙台"],
    "zh": ["北京", "上海", "广州", "成都", "杭州", "南京", "武汉", "西安"],
    "es": ["Madrid", "Sevilla", "Valencia", "Bilbao", "Málaga", "Zaragoza", "Granada", "Toledo"],
    "fr": ["Paris", "Lyon", "Marseille", "Toulouse", "Nantes", "Bordeaux", "Lille", "Rennes"],
    "pt": ["Lisboa", "Porto", "Coimbra", "Braga", "Faro", "Recife", "Curitiba", "Salvador"],
    "it": ["Roma", "Milano", "Napoli", "Torino", "Firenze", "Bologna", "Venezia", "Genova"],
    "pl": ["Warszawa", "Kraków", "Gdańsk", "Poznań", "Wrocław", "Łódź", "Lublin", "Toruń"],
    "nl": ["Amsterdam", "Rotterdam", "Utrecht", "Leiden", "Groningen", "Haarlem", "Delft", "Zwolle"],
}

# What a question and its answer are headed with.
QUESTION = {"en": "Question:", "ru": "Вопрос:", "de": "Frage:", "ja": "質問：", "zh": "问题：", "es": "Pregunta:", "fr": "Question :",
            "pt": "Pergunta:", "it": "Domanda:", "pl": "Pytanie:", "nl": "Vraag:"}
ANSWER = {"en": "Answer:", "ru": "Ответ:", "de": "Antwort:", "ja": "答え：", "zh": "答案：", "es": "Respuesta:", "fr": "Réponse :",
          "pt": "Resposta:", "it": "Risposta:", "pl": "Odpowiedź:", "nl": "Antwoord:"}


def gap(lang: str) -> str:
    """What stands between a label or a sentence and what follows: a space, or nothing."""
    return "" if lang in NO_SPACE else " "
