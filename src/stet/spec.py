"""Seasons, scales and rounds.

A *season* freezes everything that must not move while scores are being compared: the
pools, the exam, the student, the budgets and the rules. A *round* is one draw from a
season: a corpus and an exam, both derived from a single seed. Everything here is a pure
function of public inputs, so any two parties derive byte-identical rounds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

HARNESS_VERSION = "0.1.0"


@dataclass(frozen=True)
class Scale:
    name: str
    corpus_units: int  # corpus size, in units of roughly 350 MB of raw text
    train_tokens: int
    n_layer: int
    n_head: int
    d_model: int
    ctx: int
    batch_size: int
    lr: float
    exam_items_per_source: int
    edit_seconds: int

    @property
    def steps(self) -> int:
        return self.train_tokens // (self.batch_size * self.ctx)


@dataclass(frozen=True)
class Shard:
    sha256: str
    size: int
    path: str
    url: str


@dataclass(frozen=True)
class Season:
    id: str
    title: str
    raw: dict = field(repr=False, compare=False)

    @property
    def digest(self) -> str:
        """Hash of the whole season definition, pool manifests included."""
        h = hashlib.sha256(json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode())
        h.update(resources.files("stet.seasons").joinpath(self.raw["tokenizer"]["file"]).read_bytes())
        for s in self.pool("practice"):
            h.update(bytes.fromhex(s.sha256))
        return h.hexdigest()

    @property
    def ink_frac(self) -> float:
        return float(self.raw["ink_frac"])

    @property
    def min_copy_bytes(self) -> int:
        return int(self.raw["min_copy_bytes"])

    @property
    def max_doc_bytes(self) -> int:
        return int(self.raw["max_doc_bytes"])

    @property
    def vocab_size(self) -> int:
        return int(self.raw["tokenizer"]["vocab_size"])

    @property
    def eot(self) -> int:
        return int(self.raw["tokenizer"]["eot"])

    @property
    def exam_sources(self) -> list[str]:
        return list(self.raw["exam"]["sources"])

    @property
    def official_scale(self) -> str:
        return self.raw["official_scale"]

    @property
    def train_cfg(self) -> dict:
        return dict(self.raw["train"])

    def scale(self, name: str) -> Scale:
        try:
            return Scale(name=name, **self.raw["scales"][name])
        except KeyError:
            raise ValueError(f"unknown scale {name!r}; choose from {sorted(self.raw['scales'])}") from None

    def pool(self, split: str) -> list[Shard]:
        return _pool(self.id, split)


@lru_cache(maxsize=None)
def load_season(season_id: str = "s0") -> Season:
    raw = json.loads(resources.files("stet.seasons").joinpath(f"{season_id}.json").read_text())
    return Season(id=raw["id"], title=raw["title"], raw=raw)


@lru_cache(maxsize=None)
def _pool(season_id: str, split: str) -> list[Shard]:
    cfg = load_season(season_id).raw["pools"][split]
    out = []
    for line in resources.files("stet.seasons").joinpath(cfg["manifest"]).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        sha, size, path = line.split(" ", 2)
        out.append(Shard(sha256=sha, size=int(size), path=path, url=cfg["base_url"] + path))
    return out


# --- deterministic randomness -------------------------------------------------------------
#
# SHA-256 in counter mode. Chosen over a library PRNG because it is trivially identical in
# every language, so the server, the CLI and anyone's independent reimplementation agree.


def prf_u64(seed: bytes, label: str, i: int) -> int:
    h = hashlib.sha256(seed + b"|" + label.encode() + b"|" + str(i).encode()).digest()
    return int.from_bytes(h[:8], "big")


def sample_without_replacement(seed: bytes, label: str, n: int, k: int) -> list[int]:
    """k distinct indices in range(n), in draw order."""
    if k > n:
        raise ValueError(f"cannot draw {k} from {n}")
    seen: set[int] = set()
    out: list[int] = []
    i = 0
    while len(out) < k:
        j = prf_u64(seed, label, i) % n
        i += 1
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


# --- rounds -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Round:
    season: Season
    kind: str  # "practice" or "official"
    index: int
    seed: bytes
    beacon: dict | None = None  # the drand round an official seed was drawn from
    crawl: dict | None = None  # the published crawl an official corpus is drawn from

    @property
    def id(self) -> str:
        return f"{self.season.id}-{self.kind[0]}{self.index:05d}"

    @property
    def split(self) -> str:
        return self.kind

    @property
    def pool_cfg(self) -> dict:
        """Where this round's corpus comes from. Practice and official rounds use different datasets."""
        if self.kind == "practice":
            return dict(self.season.raw["pools"]["practice"])
        return {**self.season.raw["pools"]["official"], **(self.crawl or {})}

    def shard_indices(self, scale: Scale, n_shards: int) -> list[int]:
        k = scale.corpus_units * int(self.pool_cfg["shards_per_unit"])
        return sample_without_replacement(self.seed, "shard", n_shards, k)

    def exam_indices(self, scale: Scale, source: str, n_available: int) -> list[int]:
        k = min(scale.exam_items_per_source, n_available)
        return sorted(sample_without_replacement(self.seed, f"exam:{source}", n_available, k))

    @property
    def init_seed(self) -> int:
        return prf_u64(self.seed, "init", 0) % (2**31 - 1)

    def describe(self) -> dict:
        d = {"id": self.id, "season": self.season.id, "kind": self.kind, "index": self.index, "seed": self.seed.hex()}
        if self.beacon:
            d["beacon"] = self.beacon
        if self.crawl:
            d["crawl"] = self.crawl
        return d


def practice_round(season: Season, index: int) -> Round:
    """Practice rounds have fixed public seeds: run them any time, as often as you like."""
    seed = hashlib.sha256(f"stet|{season.id}|practice|{index}".encode()).digest()
    return Round(season=season, kind="practice", index=index, seed=seed)


def official_seed(season_id: str, index: int, beacon: dict, crawl: dict) -> bytes:
    return hashlib.sha256(
        f"stet|{season_id}|official|{index}|{beacon['round']}|{beacon['randomness']}|{crawl['crawl']}|{crawl['paths_sha256']}".encode()
    ).digest()


def official_round(season: Season, index: int, beacon: dict, crawl: dict) -> Round:
    """Official rounds take their seed from a public randomness beacon (drand).

    `beacon` is {"round": int, "randomness": hex}: the drand round named when the round was
    announced. `crawl` is {"crawl": id, "paths_sha256": hex, "n_shards": int}: the newest
    published Common Crawl crawl at that time. Nobody, the organisers included, can know or
    choose the corpus before that drand round is emitted, and editors are committed before
    then.
    """
    beacon = {"round": int(beacon["round"]), "randomness": str(beacon["randomness"]).lower()}
    crawl = {"crawl": str(crawl["crawl"]), "paths_sha256": str(crawl["paths_sha256"]).lower(), "n_shards": int(crawl["n_shards"])}
    seed = official_seed(season.id, index, beacon, crawl)
    return Round(season=season, kind="official", index=index, seed=seed, beacon=beacon, crawl=crawl)


def round_from_description(season: Season, d: dict) -> Round:
    """Rebuild a round from its public description, checking that the seed follows from it."""
    if d["kind"] == "practice":
        rnd = practice_round(season, int(d["index"]))
    else:
        rnd = official_round(season, int(d["index"]), d["beacon"], d["crawl"])
    if "seed" in d and d["seed"] != rnd.seed.hex():
        raise ValueError(f"round {d.get('id')}: the seed does not follow from the beacon and the crawl")
    return rnd


def parse_round_id(round_id: str) -> tuple[str, str, int]:
    season_id, tail = round_id.rsplit("-", 1)
    kind = {"p": "practice", "o": "official"}[tail[0]]
    return season_id, kind, int(tail[1:])
