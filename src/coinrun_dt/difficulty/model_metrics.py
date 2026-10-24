"""B. DT 内部指標（ロールアウトを繰り返さない、安価な難易度推定）。

ステージごとに参照軌跡を1本だけ用意し、その軌跡に沿って DT の行動分布を調べる。
- mean / p95_action_entropy: 高い目標リターンを条件にしたときの DT の迷い
- expert_nll: 参照軌跡の行動に対する負の対数尤度（DT が参照方策の行動をどれだけ予測できないか）
- rtg_kl: 高い目標と低い目標で条件付けたときの行動分布の KL 距離（条件付けがどれだけ効いているか）
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from coinrun_dt.dt.model import DecisionTransformer
from coinrun_dt.dt.rollout import DTPolicy, dt_logits_along_trajectory
from coinrun_dt.envs import make_level_env
from coinrun_dt.runner import Episode, run_episodes


def pick_reference(episodes) -> Episode:
    """成功した中で最短のエピソード。成功がなければ最初のエピソード。"""
    successes = [e for e in episodes if e.success]
    if successes:
        return min(successes, key=lambda e: e.length)
    return episodes[0]


def dt_greedy_reference(model: DecisionTransformer, reward_scale: float, level_seed: int,
                        target_return: float, device: torch.device,
                        distribution_mode: str = "easy") -> Episode:
    env = make_level_env(level_seed, 1, distribution_mode)
    policy = DTPolicy(model, 1, target_return, reward_scale, device, sample=False)
    return run_episodes(env, policy, episodes_per_env=1, record_obs=True)[0]


def model_metrics(model: DecisionTransformer, reward_scale: float, ref: Episode,
                  high_target: float, low_target: Optional[float], device: torch.device,
                  batch_size: int = 256) -> Dict[str, float]:
    logits_hi = dt_logits_along_trajectory(model, ref.obs, ref.actions, ref.rewards, high_target,
                                           reward_scale, device, batch_size)
    logp_hi = F.log_softmax(logits_hi, dim=-1)
    p_hi = logp_hi.exp()
    entropy = -(p_hi * logp_hi).sum(-1)
    actions = torch.as_tensor(ref.actions, device=device)
    nll = -logp_hi.gather(1, actions.unsqueeze(1)).squeeze(1)

    out = {
        "mean_action_entropy": entropy.mean().item(),
        "p95_action_entropy": torch.quantile(entropy, 0.95).item(),
        "expert_nll": nll.mean().item(),
        "rtg_kl": float("nan"),
        "ref_length": float(ref.length),
        "ref_success": float(ref.success),
    }
    if low_target is not None:
        logits_lo = dt_logits_along_trajectory(model, ref.obs, ref.actions, ref.rewards, low_target,
                                               reward_scale, device, batch_size)
        logp_lo = F.log_softmax(logits_lo, dim=-1)
        out["rtg_kl"] = (p_hi * (logp_hi - logp_lo)).sum(-1).mean().item()
    return out
