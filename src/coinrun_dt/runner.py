"""ベクトル化環境でエピソードを集める共通ループ。PPO と DT の両方の方策で使う。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

try:
    from typing import Protocol
except ImportError:  # Python 3.7
    Protocol = object


class VecPolicy(Protocol):
    def reset(self, env_indices: np.ndarray) -> None:
        """env_indices の環境で新しいエピソードが始まる。"""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (N, 64, 64, 3) uint8 → actions: (N,) int"""

    def observe(self, rew: np.ndarray, first: np.ndarray) -> None:
        """act 後に得た報酬と、エピソード終了フラグを受け取る。"""


@dataclass
class Episode:
    actions: np.ndarray            # (T,) int64
    rewards: np.ndarray            # (T,) float32
    level_seed: int
    success: bool
    env_index: int
    obs: Optional[np.ndarray] = None   # (T, 64, 64, 3) uint8（record_obs=True のときだけ）

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def ret(self) -> float:
        return float(self.rewards.sum())


@dataclass
class _Buffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)


def run_episodes(env, policy: VecPolicy, episodes_per_env: int, record_obs: bool = False,
                 on_episode: Optional[Callable[[Episode], None]] = None) -> List[Episode]:
    """各環境で最初の episodes_per_env 個のエピソードを集める。

    集める個数を環境ごとに固定するのは、「全体で E 個終わったら止める」方式だと
    短いエピソードに偏るため。on_episode を渡すと、エピソードを返り値に溜めずに
    逐次そのコールバックに渡す（大量収集時のメモリ対策）。
    """
    num_envs = env.num
    _, obs, _ = env.observe()
    obs = obs["rgb"]
    policy.reset(np.arange(num_envs))

    done = np.zeros(num_envs, dtype=np.int64)
    buffers = [_Buffer() for _ in range(num_envs)]
    episodes: List[Episode] = []

    while (done < episodes_per_env).any():
        actions = np.asarray(policy.act(obs))
        env.act(actions.astype(np.int32))
        rew, next_obs, first = env.observe()
        next_obs = next_obs["rgb"]
        infos = env.get_info()
        policy.observe(rew, first)

        for i in range(num_envs):
            if done[i] >= episodes_per_env:
                continue
            buf = buffers[i]
            if record_obs:
                buf.obs.append(obs[i].copy())
            buf.actions.append(int(actions[i]))
            buf.rewards.append(float(rew[i]))
            if first[i]:
                ep = Episode(
                    actions=np.asarray(buf.actions, dtype=np.int64),
                    rewards=np.asarray(buf.rewards, dtype=np.float32),
                    level_seed=int(infos[i]["prev_level_seed"]),
                    success=bool(infos[i]["prev_level_complete"]),
                    env_index=i,
                    obs=np.stack(buf.obs) if record_obs else None,
                )
                if on_episode is not None:
                    on_episode(ep)
                else:
                    episodes.append(ep)
                buffers[i] = _Buffer()
                done[i] += 1

        reset_idx = np.nonzero(first)[0]
        if len(reset_idx):
            policy.reset(reset_idx)
        obs = next_obs
    return episodes
