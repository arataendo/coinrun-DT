import numpy as np
import pandas as pd
import pytest

from coinrun_dt.difficulty.analyze import analyze, main, zscore


def write_csv(path, n=20, seed=0):
    rng = np.random.default_rng(seed)
    ppo = rng.random(n)
    df = pd.DataFrame({
        "level_seed": np.arange(n),
        "difficulty_rollout": 1 - ppo + 0.1 * rng.random(n),
        "expert_nll": 1 - ppo,
        "rtg_kl": rng.random(n),
        "mean_action_entropy": rng.random(n),
        "difficulty_model": rng.random(n),
        "ppo_ref_success_rate": ppo,
    })
    df.to_csv(path, index=False)
    return df


def test_reweight_writes_to_separate_dir(tmp_path):
    csv = tmp_path / "difficulty.csv"
    write_csv(csv)
    original = pd.read_csv(csv)

    corr = analyze(str(csv), weights={"expert_nll": 2.0, "mean_action_entropy": 0.0})

    out = tmp_path / "reweighted"
    df = pd.read_csv(out / "difficulty.csv")
    np.testing.assert_allclose(df["difficulty_model"], 2.0 * zscore(original["expert_nll"]))
    assert corr.loc["difficulty_model", "ppo_ref_success_rate"] == pytest.approx(-1.0)
    assert (out / "weights.yaml").exists() and (out / "correlation_spearman.csv").exists()
    # 元の CSV は書き換えない
    pd.testing.assert_frame_equal(pd.read_csv(csv), original)


def test_cli_weights(tmp_path):
    csv = tmp_path / "difficulty.csv"
    write_csv(csv)
    main(["--csv", str(csv), "--weights", "expert_nll=1", "rtg_kl=-0.5", "--out-dir", str(tmp_path / "w")])
    assert (tmp_path / "w" / "difficulty.csv").exists()


def test_unknown_metric_is_rejected(tmp_path):
    csv = tmp_path / "difficulty.csv"
    write_csv(csv)
    with pytest.raises(ValueError, match="CSV にない指標"):
        analyze(str(csv), weights={"no_such_metric": 1.0})
