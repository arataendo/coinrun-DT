"""DQN の Q ネットワーク（IMPALA CNN + Dueling）。難易度の参照値を PPO とは別の種類のエージェントで取るために使う。"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from coinrun_dt.envs import NUM_ACTIONS
from coinrun_dt.ppo.model import ImpalaCNN, resolve_checkpoint


class QNetwork(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS, hidden: int = 256):
        super().__init__()
        self.encoder = ImpalaCNN(hidden=hidden)
        self.value_head = nn.Linear(hidden, 1)
        self.adv_head = nn.Linear(hidden, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 64, 64, 3) uint8 → Q 値 (B, A)"""
        h = self.encoder(obs)
        adv = self.adv_head(h)
        return self.value_head(h) + adv - adv.mean(-1, keepdim=True)


class DQNPolicy:
    """run_episodes 用の ε-greedy 方策（状態を持たない）。

    特定のステージを何度も動かすとき、greedy だと毎回ほぼ同じ軌跡になって成功率が 0 か 1 に
    なってしまうので、評価でも小さな ε を残す。
    """

    def __init__(self, model: QNetwork, device: torch.device, epsilon: float = 0.05,
                 seed: int = 0):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def reset(self, env_indices: np.ndarray) -> None:
        pass

    def observe(self, rew: np.ndarray, first: np.ndarray) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        q = self.model(torch.from_numpy(obs).to(self.device))
        actions = q.argmax(-1).cpu().numpy()
        explore = self.rng.random(len(actions)) < self.epsilon
        actions[explore] = self.rng.integers(0, q.shape[-1], explore.sum())
        return actions


def load_dqn(path: Union[str, Path], device: torch.device) -> Tuple[QNetwork, dict]:
    ckpt = torch.load(resolve_checkpoint(path), map_location=device)
    model = QNetwork().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
