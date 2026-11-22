"""Words the generators draw from. Names come from many places, so no one culture's text is favoured.

The lists are short and public, so a word from them can be learned by heart. Where an answer
should rest on reading the question and nothing else, a generator coins its words from the seed.
"""

_ONSET = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kl", "pl", "st", "tr"]
_VOWEL = ["a", "e", "i", "o", "u"]
_CODA = ["", "", "", "n", "r", "s", "l", "k", "m"]


def coin(rng) -> str:
    """A word nobody has written: two or three syllables that can be said, from the seed."""
    return "".join(rng.choice(_ONSET) + rng.choice(_VOWEL) for _ in range(rng.between(2, 3))) + rng.choice(_CODA)


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
