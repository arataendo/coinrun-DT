import numpy as np
import pandas as pd
import pytest

from coinrun_dt.difficulty.compare_dts import compare_dts


def levels(seeds, ppo, noise, rng):
    n = len(seeds)
    return pd.DataFrame({"level_seed": seeds, "ppo_ref_success_rate": ppo,
                         "success_rate": ppo + noise * rng.random(n),
                         "difficulty_rollout": 1 - ppo + noise * rng.random(n),
                         "expert_nll": 1 - ppo})


def test_compare_dts():
    rng = np.random.default_rng(0)
    train, hold = np.arange(30), np.arange(1000, 1030)
    p_train, p_hold = rng.random(30), rng.random(30)
    runs = {
        "mixed": {"train": levels(train, p_train, 1.0, rng), "holdout": levels(hold, p_hold, 1.0, rng)},
        "expert": {"train": levels(train, p_train, 0.0, rng), "holdout": levels(hold, p_hold, 0.0, rng)},
    }
    comparison, summary, per_level = compare_dts(runs)

    assert set(comparison.level_set) == {"train", "holdout", "all"}
    c = comparison.set_index(["metric", "level_set"])
    assert c.loc[("difficulty_rollout", "all"), "rho_expert"] == pytest.approx(-1.0)
    assert c.loc[("difficulty_rollout", "all"), "rho_mixed"] > -1.0
    assert c.loc[("expert_nll", "holdout"), "rho_mixed"] == pytest.approx(-1.0)
    s = summary.set_index(["dt", "level_set"])
    assert s.loc[("expert", "all"), "n"] == 60
    assert s.loc[("expert", "train"), "dt_success_rate"] == pytest.approx(round(p_train.mean(), 3))
    assert len(per_level) == 60
    assert {"difficulty_rollout_mixed", "difficulty_rollout_expert", "success_rate_expert"} <= set(per_level.columns)
