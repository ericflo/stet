"""Seasons, scales and rounds.

A *season* freezes everything that must not move while scores are being compared: the
pools, the exam, the student, the budgets and the rules. A *round* is one draw from a
season: a corpus and an exam, both derived from a single seed. Everything here is a pure
function of public inputs, so any two parties derive byte-identical rounds.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

HARNESS_VERSION = "0.3.1"


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


POLICY_KEYS = ("reference", "reference_curves", "tolerance", "noise", "record", "heldout_exam", "runner")


@dataclass(frozen=True)
class Season:
    id: str
    title: str
    raw: dict = field(repr=False, compare=False)

    @property
    def digest(self) -> str:
        """Hash of everything in the season that decides a score, pool manifests included.

        Left out: the numbers that only say how results are read (tolerances, the record
        rule, reference curves) and who may run editors (the floor for a runner's
        machine). Those are the server's to apply and can be re-measured without every
        verifier having to upgrade.
        """
        rules = {k: v for k, v in self.raw.items() if k not in POLICY_KEYS}
        h = hashlib.sha256(json.dumps(rules, sort_keys=True, separators=(",", ":")).encode())
        h.update(resources.files("stet.seasons").joinpath(self.raw["tokenizer"]["file"]).read_bytes())
        for s in self.pool("gym"):
            h.update(bytes.fromhex(s.sha256))
        return h.hexdigest()

    @property
    def ink_frac(self) -> float:
        return float(self.raw["ink_frac"])

    def ink_tokens(self, scale: Scale) -> int:
        """How many of the tokens the student trains on may be the editor's own text."""
        return int(self.ink_frac * scale.train_tokens)

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

    def beacon_time(self, beacon_round: int) -> int:
        """When a beacon round is emitted (unix seconds)."""
        cfg = self.raw["beacon"]
        return int(cfg["genesis"]) + (int(beacon_round) - 1) * int(cfg["period"])

    def beacon_round_at(self, t: int) -> int:
        """The first beacon round emitted at or after unix time t."""
        cfg = self.raw["beacon"]
        if t <= cfg["genesis"]:
            return 1
        return (int(t) - int(cfg["genesis"]) + int(cfg["period"]) - 1) // int(cfg["period"]) + 1

    def draw_slot(self, beacon_round: int) -> int | None:
        """The draw time a beacon round belongs to, or None if it belongs to none.

        Official rounds are drawn only at whole multiples of the season's grid, from the
        first beacon round at or after that moment. So which beacon rounds can seed a round
        is public and fixed in advance: nobody can go shopping among them for a seed.
        """
        grid = int(self.raw["draw_grid_seconds"])
        slot = self.beacon_time(beacon_round) // grid * grid
        return slot if self.beacon_round_at(slot) == int(beacon_round) else None

    def scale(self, name: str) -> Scale:
        try:
            return Scale(name=name, **self.raw["scales"][name])
        except KeyError:
            raise ValueError(f"unknown scale {name!r}; choose from {sorted(self.raw['scales'])}") from None

    def pool(self, split: str) -> list[Shard]:
        return _pool(self.id, split)


@cache
def load_season(season_id: str = "s0") -> Season:
    raw = json.loads(resources.files("stet.seasons").joinpath(f"{season_id}.json").read_text())
    return Season(id=raw["id"], title=raw["title"], raw=raw)


@cache
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

CRAWL_ID_RE = re.compile(r"^CC-MAIN-\d{4}-\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROUND_ID_RE = re.compile(r"^([a-z0-9]{1,16})-([pgo])(\d{1,9})$")


@dataclass(frozen=True)
class Round:
    season: Season
    kind: str  # "practice", "gym" or "official"
    index: int
    seed: bytes
    beacon: dict | None = None  # the drand round an official seed was drawn from
    crawl: dict | None = None  # the published crawl an official corpus is drawn from

    @property
    def id(self) -> str:
        return f"{self.season.id}-{self.kind[0]}{self.index:05d}"

    @property
    def split(self) -> str:
        """Which half of the exam pack this round is examined on."""
        return "official" if self.kind == "official" else "practice"

    @property
    def pool_cfg(self) -> dict:
        """Where this round's corpus comes from.

        official   the newest Common Crawl crawl at the time of the draw
        practice   one pinned crawl: the same kind of data, with seeds anyone can replay
        gym        a different dataset altogether (DCLM-Pool), for training agents
        """
        if self.kind == "official":
            return {**self.season.raw["pools"]["official"], **(self.crawl or {})}
        return dict(self.season.raw["pools"][self.kind])

    def shard_indices(self, scale: Scale, n_shards: int) -> list[int]:
        k = scale.corpus_units * int(self.pool_cfg["shards_per_unit"])
        return sample_without_replacement(self.seed, "shard", n_shards, k)

    def exam_indices(self, scale: Scale, source: str, n_available: int) -> list[int]:
        k = min(scale.exam_items_per_source, n_available)
        return sorted(sample_without_replacement(self.seed, f"exam:{source}", n_available, k))

    def doc_order(self, n_docs: int) -> list[int]:
        """The order the corpus's documents are written in: a seeded shuffle of arrival order.

        A crawl arrives grouped by host. Shuffling makes "the first N tokens" a fair sample,
        which is what the baseline trains on.
        """
        keys = [prf_u64(self.seed, "order", i) for i in range(n_docs)]
        return sorted(range(n_docs), key=keys.__getitem__)

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


def practice_round(season: Season, index: int, kind: str = "practice") -> Round:
    """Practice and gym rounds have fixed public seeds: run them any time, as often as you like."""
    if kind not in ("practice", "gym"):
        raise ValueError(f"{kind!r} rounds do not have fixed seeds")
    seed = hashlib.sha256(f"stet|{season.id}|{kind}|{index}".encode()).digest()
    return Round(season=season, kind=kind, index=index, seed=seed)


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
    # These come from the server and go into the seed, a URL and a file name. A crawl has
    # one name; anything looser would be a second spelling of it, and so a second seed.
    if not CRAWL_ID_RE.fullmatch(crawl["crawl"]) or not SHA256_RE.fullmatch(crawl["paths_sha256"]) or not SHA256_RE.fullmatch(beacon["randomness"]):
        raise ValueError(f"round {index}: the crawl or the beacon is not in the form this season uses")
    if not (0 < crawl["n_shards"] <= 10_000_000 and beacon["round"] > 0 and index >= 0):
        raise ValueError(f"round {index}: the crawl or the beacon is out of range")
    seed = official_seed(season.id, index, beacon, crawl)
    return Round(season=season, kind="official", index=index, seed=seed, beacon=beacon, crawl=crawl)


def round_from_description(season: Season, d: dict) -> Round:
    """Rebuild a round from its public description, checking that the seed follows from it."""
    if d["kind"] in ("practice", "gym"):
        rnd = practice_round(season, int(d["index"]), d["kind"])
    else:
        rnd = official_round(season, int(d["index"]), d["beacon"], d["crawl"])
    if "seed" in d and d["seed"] != rnd.seed.hex():
        raise ValueError(f"round {d.get('id')}: the seed does not follow from the beacon and the crawl")
    return rnd


def parse_round_id(round_id: str) -> tuple[str, str, int]:
    m = ROUND_ID_RE.fullmatch(round_id)
    if not m:
        raise ValueError(f"{round_id[:40]!r} is not a round id (they look like s0-o00012)")
    return m.group(1), {"p": "practice", "g": "gym", "o": "official"}[m.group(2)], int(m.group(3))
