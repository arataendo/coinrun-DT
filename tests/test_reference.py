import numpy as np
import pandas as pd
import pytest

from coinrun_dt.difficulty.reference import BASE, RandomPolicy, compare_references, load_levels


def write_levels(path, seeds, rng):
    ppo = rng.random(len(seeds))
    pd.DataFrame({"level_seed": seeds, "ppo_ref_success_rate": ppo, "expert_nll": 1 - ppo,
                  "difficulty_rollout": rng.random(len(seeds)), "difficulty_model": rng.random(len(seeds)),
                  "model_nll": 1 - ppo}).to_csv(path, index=False)


def test_load_levels_and_compare(tmp_path):
    rng = np.random.default_rng(0)
    write_levels(tmp_path / "a.csv", np.arange(20), rng)
    write_levels(tmp_path / "b.csv", np.arange(1000, 1020), rng)
    levels = load_levels({"train": str(tmp_path / "a.csv"), "holdout": str(tmp_path / "b.csv")})
    assert list(levels.columns[:3]) == ["level_set", "level_seed", BASE]
    assert "difficulty_model" not in levels and "model_nll" not in levels
    assert len(levels) == 40

    levels["ref_same"] = levels[BASE]
    levels["ref_opposite"] = 1 - levels[BASE]
    comparison, agreement = compare_references(levels, [BASE, "ref_same", "ref_opposite"],
                                               ["expert_nll", "difficulty_rollout"])
    c = comparison.set_index(["metric", "reference", "level_set"])
    assert set(comparison.level_set) == {"train", "holdout", "all"}
    assert c.loc[("expert_nll", "ref_same", "all"), "rho"] == pytest.approx(-1.0)
    assert c.loc[("expert_nll", "ref_opposite", "holdout"), "strength"] == "reversed"
    assert c.loc[("expert_nll", BASE, "train"), "n"] == 20
    a = agreement.set_index(["reference_a", "reference_b", "level_set"])
    assert a.loc[(BASE, "ref_same", "all"), "rho"] == pytest.approx(1.0)
    assert a.loc[(BASE, "ref_opposite", "all"), "rho"] == pytest.approx(-1.0)


def test_load_levels_rejects_overlapping_seeds(tmp_path):
    rng = np.random.default_rng(0)
    write_levels(tmp_path / "a.csv", np.arange(10), rng)
    write_levels(tmp_path / "b.csv", np.arange(5, 15), rng)
    with pytest.raises(ValueError):
        load_levels({"train": str(tmp_path / "a.csv"), "holdout": str(tmp_path / "b.csv")})


def test_random_policy():
    actions = RandomPolicy(0).act(np.zeros((64, 64, 64, 3), dtype=np.uint8))
    assert actions.shape == (64,) and actions.min() >= 0 and actions.max() < 15
