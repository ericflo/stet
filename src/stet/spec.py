"""Seasons, scales and rounds.

A *season* freezes everything that must not move while scores are being compared: the
pools, the exam, the student, the budgets and the rules. A *round* is one draw from a
season: a corpus, the student's initial weights and the exam's generated questions, all
derived from a single seed (the exam's fresh text is three hours of the day before the draw,
which the seed picks).
Everything here is a pure function of public inputs, so any two parties derive
byte-identical rounds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

HARNESS_VERSION = "0.6.0"
# The oldest stet whose grades are this one's: the server refuses jobs and results from older
# ones, and they upgrade. Moved up only when what a grade measures changes, so that a release
# that changes nothing in a grade does not throw away work under way.
GRADES_SINCE = "0.6.0"
# The season the competition is in: what every command means when it names none. Changing it is
# the switch from one season to the next (see docs/OPERATING.md).
CURRENT_SEASON = "s3"


class TrainingError(RuntimeError):
    """Training or scoring diverged numerically (non-finite loss or score).

    A numerical divergence is the grader's failure, not a script rule violation: the grader
    retries once on the same machine and, if it still fails, hands the job back so another
    machine grades the script before anything is held against the entry.
    """


def hardware_backend(hw: dict | None) -> str | None:
    """Extract the concrete backend ('cuda', 'rocm', 'mps', 'cpu') from a hardware info dict.

    Returns None when CUDA metadata lacks an explicit backend (unless ROCm/AMD hints identify
    ROCm) so NVIDIA grades never reuse an untagged or ROCm-labelled legacy cache entry.
    """
    if not isinstance(hw, dict):
        return None
    dev = hw.get("device")
    if not isinstance(dev, str) or not dev:
        return None
    if dev == "cuda":
        bk = hw.get("backend")
        if bk in ("cuda", "rocm"):
            return bk
        torch_ver = str(hw.get("torch") or "").lower()
        gpu = str(hw.get("gpu") or "").lower()
        if "rocm" in torch_ver or "hip" in torch_ver or "amd" in gpu or "radeon" in gpu or "instinct" in gpu:
            return "rocm"
        return None
    return dev


@dataclass(frozen=True)
class Scale:
    name: str
    corpus_units: int  # corpus size, in units of roughly 300-350 MB of raw text
    train_tokens: int
    n_layer: int
    n_head: int
    d_model: int
    ctx: int
    batch_size: int
    lr: float
    edit_seconds: int
    micro_batch: int = 0  # 0: same as batch_size (no gradient accumulation)
    sft_steps: int = 0  # 0: no post-pretraining SFT probe
    warmup_steps: int = 0  # 0: derive from season warmup_frac (Season 1-2 default)

    @property
    def steps(self) -> int:
        return self.train_tokens // (self.batch_size * self.ctx)


@dataclass(frozen=True)
class Shard:
    sha256: str
    size: int
    path: str
    url: str


POLICY_KEYS = ("reference", "reference_curves", "tolerance", "noise", "record", "runner", "baselines", "references")


@dataclass(frozen=True)
class Season:
    id: str
    title: str
    raw: dict = field(repr=False, compare=False)

    @property
    def digest(self) -> str:
        """Hash of everything in the season that decides a score, pool manifests included.

        Left out: the numbers that only say how results are read (tolerances, the record
        rule, reference curves), who may run curators (the floor for a runner's machine),
        and how the server spreads a round's work (how many machines measure its unedited
        baseline, which reference curators run in it). Those are the server's to apply and
        can be changed without every verifier having to upgrade.
        """
        rules = {k: v for k, v in self.raw.items() if k not in POLICY_KEYS}
        h = hashlib.sha256(json.dumps(rules, sort_keys=True, separators=(",", ":")).encode())
        tok_cfg = self.raw["tokenizer"]
        if tok_cfg.get("kind") == "hf":
            # External HF tokenizer pinned by sha256 in rules (verified on download in tok.py).
            h.update(bytes.fromhex(tok_cfg["sha256"]))
        else:
            h.update(resources.files("stet.seasons").joinpath(tok_cfg["file"]).read_bytes())
        # Pinned base student files (if the season starts from a pretrained checkpoint).
        for f in (self.raw.get("student") or {}).get("files") or []:
            h.update(bytes.fromhex(f["sha256"]))
        # Pinned SFT probe pack (if the season bundles a fixed multilingual SFT probe).
        sft_cfg = self.raw.get("sft") or {}
        if sft_cfg.get("pack"):
            h.update(resources.files("stet.exam").joinpath("data", sft_cfg["pack"]).read_bytes())
        # the facts its knowledge items ask (when the season uses generated knowledge items)
        if facts_cfg := (self.raw.get("exam") or {}).get("facts"):
            h.update(resources.files("stet.exam").joinpath("data", facts_cfg["file"]).read_bytes())
        for s in self.pool("gym"):
            h.update(bytes.fromhex(s.sha256))
        return h.hexdigest()

    @property
    def min_copy_bytes(self) -> int:
        """The shortest copy a script may hold. See stet.script for the two exceptions."""
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
        """What the exam reports on: its pillars, in the order they are shown."""
        return list(self.raw["exam"]["pillars"])

    @property
    def answer_floor_bytes(self) -> int:
        """The shortest an answer is taken to be when its bits are put over its bytes: a one-word
        answer does not count as much as a sentence."""
        return int(self.raw["exam"].get("answer_floor_bytes", 1))

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
def load_season(season_id: str = CURRENT_SEASON) -> Season:
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
    exam_digest: str = ""  # an official round: the digest of the exam posted after its draw
    scale: str = ""  # published official round scale (e.g. "micro", "standard"), when reported

    @property
    def id(self) -> str:
        return f"{self.season.id}-{self.kind[0]}{self.index:05d}"

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
        if self.exam_digest:
            d["exam_digest"] = self.exam_digest
        if self.scale:
            d["scale"] = self.scale
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
    choose the corpus before that drand round is emitted, and curators are committed before
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
    if d.get("exam_digest"):  # a round whose exam has been posted
        if not SHA256_RE.fullmatch(str(d["exam_digest"])):
            raise ValueError(f"round {d.get('id')}: its exam's digest is not a sha256")
        rnd = dataclasses.replace(rnd, exam_digest=str(d["exam_digest"]))
    if d.get("scale"):
        rnd = dataclasses.replace(rnd, scale=str(d["scale"]).strip())
    return rnd


def parse_round_id(round_id: str) -> tuple[str, str, int]:
    m = ROUND_ID_RE.fullmatch(round_id)
    if not m:
        raise ValueError(f"{round_id[:40]!r} is not a round id (they look like s2-o00012)")
    return m.group(1), {"p": "practice", "g": "gym", "o": "official"}[m.group(2)], int(m.group(3))
