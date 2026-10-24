"""IMPALA CNN の Actor-Critic（Procgen 論文の標準構成）。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from coinrun_dt.envs import NUM_ACTIONS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.conv0(F.relu(x))
        out = self.conv1(F.relu(out))
        return x + out


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = ResidualBlock(out_channels)
        self.res1 = ResidualBlock(out_channels)

    def forward(self, x):
        return self.res1(self.res0(self.pool(self.conv(x))))


class ImpalaCNN(nn.Module):
    def __init__(self, channels=(16, 32, 32), hidden: int = 256):
        super().__init__()
        seqs, in_c = [], 3
        for c in channels:
            seqs.append(ConvSequence(in_c, c))
            in_c = c
        self.seqs = nn.Sequential(*seqs)
        self.fc = nn.Linear(in_c * 8 * 8, hidden)  # 64 → 32 → 16 → 8

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.permute(0, 3, 1, 2).float() / 255.0
        x = self.seqs(x)
        x = F.relu(torch.flatten(x, 1))
        return F.relu(self.fc(x))


class ActorCritic(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS, hidden: int = 256):
        super().__init__()
        self.encoder = ImpalaCNN(hidden=hidden)
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, 64, 64, 3) uint8 → logits (B, A), value (B,)"""
        h = self.encoder(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class PPOPolicy:
    """run_episodes 用のラッパ（状態を持たない）。"""

    def __init__(self, model: ActorCritic, device: torch.device, sample: bool = True,
                 temperature: float = 1.0):
        self.model = model
        self.device = device
        self.sample = sample
        self.temperature = temperature

    def reset(self, env_indices: np.ndarray) -> None:
        pass

    def observe(self, rew: np.ndarray, first: np.ndarray) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        logits, _ = self.model(torch.from_numpy(obs).to(self.device))
        if self.sample:
            actions = Categorical(logits=logits / self.temperature).sample()
        else:
            actions = logits.argmax(-1)
        return actions.cpu().numpy()


_CKPT_RE = re.compile(r"step_(\d+)\.pt$")


def list_checkpoints(ckpt_dir: Union[str, Path]) -> List[Tuple[int, Path]]:
    """[(step, path)] を step の昇順で返す。"""
    out = []
    for p in Path(ckpt_dir).glob("step_*.pt"):
        m = _CKPT_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def resolve_checkpoint(path: Union[str, Path]) -> Path:
    """ディレクトリが渡されたら、その中で最も step が大きいチェックポイントを返す。"""
    path = Path(path)
    if path.is_dir():
        ckpts = list_checkpoints(path)
        if not ckpts:
            raise FileNotFoundError(f"チェックポイントがない: {path}")
        return ckpts[-1][1]
    return path


def load_ppo(path: Union[str, Path], device: torch.device) -> Tuple[ActorCritic, dict]:
    ckpt = torch.load(resolve_checkpoint(path), map_location=device)
    model = ActorCritic().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
