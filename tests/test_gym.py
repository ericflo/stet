"""The gym turns a round into (observation, action, reward) without changing what a score is."""

import pytest

from stet.exam.items import Item

pytest.importorskip("torch")


@pytest.fixture
def env(corpus, monkeypatch):
    from stet import data, gym

    seen = {}

    def fake_grade(rnd, scale, corpus, exam, script_path, device=None, progress=None, log=None):
        lines = open(script_path).read().split()
        seen["lines"] = lines
        if not lines:
            return {"valid": False, "invalid_reason": "the script is empty"}
        gain = 0.1 * len(lines)
        scores = [-5.0 + gain + 0.2, -5.0 + gain - 0.2]  # one item a pillar, each 0.2 either side
        return {
            "valid": True, "invalid_reason": None, "script": {"out_docs": len(lines)}, "train": {"tokens": scale.train_tokens},
            "exam": {"items": 2, "mean": -5.0 + gain, "item_scores": scores},
        }  # fmt: skip

    base = {"exam": {"items": 2, "mean": -5.0, "item_scores": [-5.0, -5.0]}}
    items = [Item(kind="answer", corner="agent/shell/ls", fmt="logs", lang="en", source="gen:agent", ctx="$ ls\na  b\n$ rm a\n$ ls\n", cont="b"),
             Item(kind="choice", corner="logic-mc/x", fmt="prose", lang="en", source="gen:logic-mc", ctx="All cats purr. Tom is a cat. Does Tom purr?", cont=" Yes", wrongs=(" No",))]
    monkeypatch.setattr(data, "corpus", lambda rnd, scale, quiet=False: corpus)
    monkeypatch.setattr(data, "materialize", lambda rnd, scale, quiet=False: (corpus, items))
    monkeypatch.setattr(gym, "baseline", lambda rnd, scale, device: base)
    monkeypatch.setattr("stet.grade.grade", fake_grade)
    e = gym.Env(scale="nano", device="cpu")
    e.seen = seen
    return e


def test_an_episode_is_a_round_and_the_reward_is_the_gain(env):
    obs = env.reset(3)
    assert obs.round.id == "s3-g00003" and obs.round.kind == "gym"
    assert len(obs.corpus) == 4 and obs.train_tokens == env.scale.train_tokens
    assert "exam_pillars" in obs.task and "seed" not in obs.task  # told what the exam covers, never what is in it
    reward, info = env.step([0, 3])
    assert env.seen["lines"] == ["0", "3"]
    assert reward == pytest.approx(0.2) and info["valid"] and info["baseline_mean"] == -5.0
    assert info["gain_by_pillar"] == {"agents": pytest.approx(0.4), "logic": pytest.approx(0.0)}
    # The same round can be stepped again to compare another edit of the same corpus.
    assert env.step([0])[0] == pytest.approx(0.1)


def test_a_script_that_breaks_a_rule_scores_as_the_unedited_corpus(env):
    env.reset()
    reward, info = env.step([])
    assert reward == 0.0 and not info["valid"] and info["invalid_reason"]


def test_rounds_advance_and_the_practice_split_is_available(env):
    from stet import gym

    assert env.reset().round.index == 0 and env.reset().round.index == 1
    assert gym.Env(split="practice", device="cpu").reset(2).round.id == "s3-p00002"
    with pytest.raises(ValueError):
        gym.Env(split="official")
    with pytest.raises(RuntimeError):
        gym.Env(device="cpu").step([0])
