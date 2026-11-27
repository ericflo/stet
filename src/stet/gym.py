"""The gym: Stet as a reinforcement-learning environment.

One episode is one round. The observation is a raw corpus and a description of the task.
The action is an edit script. The reward is the gain: how much better the fixed student does
on the exam after training on your edit than after training on the unedited corpus, in bits
per byte, on the season's exam.

    from stet.gym import Env

    env = Env(scale="nano")
    obs = env.reset(0)                         # gym round 0
    keep = [i for i, text in obs.corpus if looks_good(text)]
    reward, info = env.step(keep)              # trains one student: about 40 s on a desktop GPU

Gym rounds are drawn from a different dataset (DCLM-Pool) than official rounds (the newest
Common Crawl). That is the point: a policy trained here and entered in the competition is
being tested on data, and a kind of data, it has never seen. Rounds are pure functions of
(season, split, index), so a reward can be reproduced on any machine, and there are as
many rounds as there are non-negative integers.

A gym round's exam is its generated questions: they follow from its seed, so a policy could
compute them. In the competition nobody can, since the exam is made at the draw and three
quarters of its score is fresh writing. A policy that peeks will score well in the gym and
nowhere else.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import data
from .pipeline import baseline
from .runner import task_info
from .script import write_script
from .spec import CURRENT_SEASON, Round, Scale, Season, load_season, practice_round
from .train import pick_device


@dataclass
class Observation:
    corpus: data.Corpus  # len(corpus), corpus.text(i), corpus.raw(i), corpus.size(i), corpus.urls[i]
    task: dict  # the same task.json a curator is given: budgets, limits, the student, the exam's pillars
    round: Round = field(repr=False)

    @property
    def train_tokens(self) -> int:
        return int(self.task["train_tokens"])


class Env:
    """`reset(index)` picks a round; `step(script)` grades an edit script on it.

    A round takes one action, but nothing stops you calling `step` again with another
    script for the same round, which is how you compare two edits of the same corpus.
    Each call trains one student from the same initial weights.
    """

    def __init__(self, scale: str = "nano", split: str = "gym", device=None, season: str = CURRENT_SEASON, quiet: bool = True):
        if split not in ("gym", "practice"):
            raise ValueError("split must be 'gym' or 'practice'")
        self.season: Season = load_season(season)
        self.scale: Scale = self.season.scale(scale)
        self.split = split
        self.device = pick_device(device)
        self.quiet = quiet
        self._round: Round | None = None
        self._next = 0

    def reset(self, index: int | None = None) -> Observation:
        """Start a round. Downloads and builds its corpus the first time (cached after that)."""
        if index is None:
            index = self._next
        self._next = index + 1
        self._round = practice_round(self.season, int(index), self.split)
        corpus = data.corpus(self._round, self.scale, quiet=self.quiet)
        return Observation(corpus, task_info(self._round, self.scale, float(self.scale.edit_seconds)), self._round)

    def _pin_exam(self, rnd: Round, scale: Scale):
        if rnd.kind == "practice":
            hours = data.practice_hours(rnd)
            pk = data.practice_key(rnd, hours)
            try:
                exam = data.round_exam(rnd, scale, hours=hours)
            except TypeError:
                exam = data.round_exam(rnd, scale)
            return data.corpus(rnd, scale, quiet=self.quiet), exam, pk
        corpus, exam = data.materialize(rnd, scale, quiet=self.quiet)
        return corpus, exam, None

    def baseline(self, *, exam=None, practice_key: str | None = None) -> dict:
        """The unedited corpus's result for the current round. Trained once per round, then cached."""
        if self._round is None:
            raise RuntimeError("call reset() first")
        import inspect

        kw: dict = {}
        try:
            params = inspect.signature(baseline).parameters
            accepts_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            params, accepts_var = {}, False
        if exam is not None and (accepts_var or "exam" in params):
            kw["exam"] = exam
        if practice_key is not None and (accepts_var or "practice_key" in params):
            kw["practice_key"] = practice_key
        return baseline(self._round, self.scale, self.device, **kw)

    def step(self, script) -> tuple[float, dict]:
        """Grade an edit script. `script` is a path to a script.jsonl or an iterable of lines.

        Returns (reward, info). The reward is the gain in bits per byte. A script that
        breaks a rule is not an error: it scores as the unedited corpus (reward 0) and
        `info["valid"]` is False with the reason in `info["invalid_reason"]`.
        """
        from .grade import grade

        if self._round is None:
            raise RuntimeError("call reset() first")
        rnd, scale = self._round, self.scale
        corpus, exam, pk = self._pin_exam(rnd, scale)
        base = self.baseline(exam=exam, practice_key=pk)
        with tempfile.TemporaryDirectory(prefix="stet-gym-") as tmp:
            if isinstance(script, (str, Path)):
                path = Path(script)
            else:
                path = Path(tmp) / "script.jsonl"
                write_script(path, script)
            result = grade(rnd, scale, corpus, exam, path, device=self.device)
        # The reward is the exam's aggregate of the item-by-item differences.
        from .exam.aggregate import gains, plan_of, round_score

        info = {"round": rnd.id, "valid": result["valid"], "invalid_reason": result["invalid_reason"],
                "baseline_mean": base["exam"]["mean"], "script": result.get("script")}
        if not result["valid"]:
            return 0.0, info
        want_sha = data.exam_digest(exam)
        res_sha = result.get("exam_sha256")
        base_sha = base.get("exam_sha256")
        if res_sha and res_sha != want_sha:
            raise RuntimeError(f"cannot compute reward: candidate exam ({res_sha[:12]}) does not match pinned exam ({want_sha[:12]})")
        if base_sha and base_sha != want_sha:
            raise RuntimeError(f"cannot compute reward: baseline exam ({base_sha[:12]}) does not match pinned exam ({want_sha[:12]})")
        if res_sha and base_sha and res_sha != base_sha:
            raise RuntimeError(f"cannot compute reward: candidate exam ({res_sha[:12]}) and baseline exam ({base_sha[:12]}) differ")
        if len(result["exam"]["item_scores"]) != len(exam) or len(base["exam"]["item_scores"]) != len(exam):
            raise RuntimeError(
                f"cannot compute reward: item score counts ({len(result['exam']['item_scores'])}, "
                f"{len(base['exam']['item_scores'])}) do not match pinned exam ({len(exam)})"
            )
        agg = round_score(exam, gains(result["exam"]["item_scores"], base["exam"]["item_scores"]), plan=plan_of(rnd.season))
        info.update(mean=result["exam"]["mean"], gain_by_pillar={k: round(v["score"], 5) for k, v in agg["pillars"].items()}, train=result["train"])
        return round(agg["score"], 5), info
