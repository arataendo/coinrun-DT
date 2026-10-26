import numpy as np
import pandas as pd
import pytest

from coinrun_dt.difficulty.compare import CompareConfig, compare, strength


def write_csv(path, n=30, seed=0):
    rng = np.random.default_rng(seed)
    ppo = rng.random(n)
    pd.DataFrame({
        "level_seed": np.arange(n),
        "difficulty_rollout": 1 - ppo + 0.1 * rng.random(n),
        "expert_nll": 1 - ppo,
        "rtg_kl": rng.random(n),
        "mean_action_entropy": ppo,     # 向きが逆の指標
        "difficulty_model": rng.random(n),
        "ppo_ref_success_rate": ppo,
    }).to_csv(path, index=False)


def test_compare_writes_ranked_table(tmp_path):
    csv = tmp_path / "difficulty.csv"
    write_csv(csv)
    out = tmp_path / "results"
    cfg = CompareConfig(csv=str(csv), out_dir=str(out), weight_sets={
        "nll": {"expert_nll": 1.0},
        "ent": {"mean_action_entropy": 1.0},
        "negent": {"mean_action_entropy": -1.0},
    })
    table = compare(cfg)

    assert table["rho_vs_ppo_success"].is_monotonic_increasing
    by_name = table.set_index("name")
    assert by_name.loc["model_nll", "rho_vs_ppo_success"] == pytest.approx(-1.0)
    assert by_name.loc["model_ent", "strength"] == "reversed"
    assert by_name.loc["model_negent", "strength"] == "strong"
    assert by_name.loc["expert_nll", "kind"] == "single"

    levels = pd.read_csv(out / "levels.csv")
    assert {"model_nll", "model_ent", "model_negent", "expert_nll"} <= set(levels.columns)
    assert len(levels) == 30
    assert (out / "weight_comparison.csv").exists() and (out / "weights.yaml").exists()


def test_strength_labels():
    assert [strength(r) for r in (-0.8, -0.5, -0.1, 0.2)] == ["strong", "moderate", "weak", "reversed"]
