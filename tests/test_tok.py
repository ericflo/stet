from stet.spec import load_season
from stet.tok import load_encoding


def test_tokenizer_round_trips_and_matches_the_season():
    season = load_season("s0")
    enc = load_encoding(season)
    assert enc.n_vocab == season.vocab_size
    for text in ["Hello, world!", "naïve café — 東京", "def f(x):\n    return x ** 2\n", "<|endoftext|> is plain text here"]:
        ids = enc.encode_ordinary(text)
        assert enc.decode(ids) == text
        assert season.eot not in ids  # the separator can never be forged from document text
