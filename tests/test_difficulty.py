import numpy as np
import pytest
import torch

pytest.importorskip("procgen")

from coinrun_dt.difficulty.analyze import analyze
from coinrun_dt.difficulty.score import DifficultyConfig, evaluate_levels
from coinrun_dt.dt.model import DecisionTransformer, DTConfig, save_dt


def test_pipeline_with_random_dt(tmp_path):
    torch.manual_seed(0)
    model = DecisionTransformer(DTConfig(context_len=4, n_layer=1, n_head=2, n_embd=32))
    ckpt = tmp_path / "dt.pt"
    save_dt(ckpt, model, reward_scale=0.1, train_config={}, iteration=0)

    cfg = DifficultyConfig(dt_ckpt=str(ckpt), target_returns=[10.0, 0.0], rollouts_per_level=2,
                           reference_source="dt_greedy", device="cpu")
    df = evaluate_levels(str(ckpt), [0, 1, 2], cfg)

    assert df["level_seed"].tolist() == [0, 1, 2]
    for col in ["difficulty_rollout", "success_rate", "rtg_sensitivity", "difficulty_model",
                "mean_action_entropy", "expert_nll", "rtg_kl"]:
        assert col in df
    assert df["difficulty_rollout"].between(0, 1).all()
    assert (df["mean_action_entropy"] <= np.log(15) + 1e-5).all()

    csv = tmp_path / "difficulty.csv"
    df.to_csv(csv, index=False)
    analyze(str(csv))
    assert (tmp_path / "correlation_spearman.csv").exists()
