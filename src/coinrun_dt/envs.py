"""Procgen CoinRun 環境の生成ヘルパ（gym3 API を直接使う）。

gym3 のループ:
    rew, obs, first = env.observe()   # obs["rgb"]: (N, 64, 64, 3) uint8
    env.act(actions)
    rew, obs, first = env.observe()   # rew は直前の行動の報酬、first=True はその行動でエピソードが終わったことを表す
    env.get_info()[i]["prev_level_seed"], ["prev_level_complete"]  # 終わったエピソードの seed と成功
"""
from __future__ import annotations

from typing import Optional

from procgen import ProcgenGym3Env

ENV_NAME = "coinrun"
NUM_ACTIONS = 15
OBS_SHAPE = (64, 64, 3)
MAX_EPISODE_STEPS = 1000
COIN_REWARD = 10.0


def make_env(num_envs: int, distribution_mode: str = "easy", start_level: int = 0,
             num_levels: int = 0, rand_seed: Optional[int] = None) -> ProcgenGym3Env:
    """num_levels=0 なら、エピソードごとにランダムなステージになる。"""
    kwargs = dict(num=num_envs, env_name=ENV_NAME, num_levels=num_levels,
                  start_level=start_level, distribution_mode=distribution_mode)
    if rand_seed is not None:
        kwargs["rand_seed"] = rand_seed
    return ProcgenGym3Env(**kwargs)


def make_level_env(level_seed: int, num_envs: int, distribution_mode: str = "easy",
                   rand_seed: Optional[int] = None) -> ProcgenGym3Env:
    """特定のステージ（level_seed）だけを num_envs 並列で動かす環境。"""
    return make_env(num_envs, distribution_mode, start_level=level_seed, num_levels=1,
                    rand_seed=rand_seed)
