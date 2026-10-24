"""A. ロールアウトベースの難易度指標。

1つのステージで、目標リターンごとに rollouts_per_level 回ずつ DT を（行動をサンプリングして）動かす。
すべての目標リターンを1つのベクトル化環境にまとめて、同時に実行する。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from coinrun_dt.dt.model import DecisionTransformer
from coinrun_dt.dt.rollout import DTPolicy
from coinrun_dt.envs import MAX_EPISODE_STEPS, make_level_env
from coinrun_dt.runner import run_episodes


def rollout_metrics(model: DecisionTransformer, reward_scale: float, level_seed: int,
                    target_returns: List[float], rollouts_per_level: int, device: torch.device,
                    distribution_mode: str = "easy", sample: bool = True, temperature: float = 1.0,
                    steps_weight: float = 0.5, seed: int = 0) -> Dict[str, float]:
    R = rollouts_per_level
    targets = np.repeat(np.asarray(target_returns, dtype=np.float32), R)
    env = make_level_env(level_seed, len(targets), distribution_mode, rand_seed=seed)
    policy = DTPolicy(model, env.num, targets, reward_scale, device, sample=sample,
                      temperature=temperature)
    episodes = run_episodes(env, policy, episodes_per_env=1)
    del env

    out: Dict[str, float] = {}
    by_target = {}
    for target in target_returns:
        eps = [e for e in episodes if targets[e.env_index] == np.float32(target)]
        sr = float(np.mean([e.success for e in eps]))
        by_target[target] = (sr, eps)
        out[f"success_rate_rtg{target:g}"] = sr

    main = max(target_returns)
    sr, eps = by_target[main]
    success_steps = [e.length for e in eps if e.success]
    out["success_rate"] = sr
    out["mean_return"] = float(np.mean([e.ret for e in eps]))
    out["mean_length"] = float(np.mean([e.length for e in eps]))
    out["mean_steps_to_success"] = float(np.mean(success_steps)) if success_steps else float("nan")
    out["rtg_sensitivity"] = (sr - by_target[min(target_returns)][0]
                              if len(target_returns) > 1 else float("nan"))

    # 成功しないほど難しい。成功しても時間がかかるほど少し難しい。範囲は [0, 1]
    steps_norm = (np.mean(success_steps) / MAX_EPISODE_STEPS) if success_steps else 1.0
    out["difficulty_rollout"] = float(1.0 - sr * (1.0 - steps_weight * steps_norm))
    return out
