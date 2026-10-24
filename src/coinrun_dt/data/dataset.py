"""DT 用データセット。

エピソード単位の npz を、初回だけ1つのメモリマップ（<data_dir>/cache/obs.npy）にまとめる。
サンプル i は「時刻 i で終わる長さ K の文脈窓」で、エピソードの先頭側は左詰めでパディングしてマスクする。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from coinrun_dt.envs import MAX_EPISODE_STEPS, OBS_SHAPE


def compute_rtg(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """rtg[t] = sum_{t' >= t} gamma^(t'-t) * r[t']"""
    rtg = np.zeros(len(rewards), dtype=np.float32)
    acc = 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc
        rtg[t] = acc
    return rtg


def build_cache(data_dir: str, rebuild: bool = False) -> Path:
    root = Path(data_dir)
    cache = root / "cache"
    meta_path = cache / "meta.npz"
    index = pd.read_csv(root / "index.csv")
    lengths = index["length"].to_numpy(dtype=np.int64)
    total = int(lengths.sum())
    if meta_path.exists() and not rebuild:
        with np.load(meta_path) as meta:
            if ("total_frames" in meta.files and int(meta["num_episodes"]) == len(index)
                    and int(meta["total_frames"]) == total):
                return cache
    cache.mkdir(parents=True, exist_ok=True)

    ep_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    obs = np.lib.format.open_memmap(cache / "obs.npy", mode="w+", dtype=np.uint8,
                                    shape=(total,) + OBS_SHAPE)
    actions = np.zeros(total, dtype=np.int64)
    rewards = np.zeros(total, dtype=np.float32)
    rtg = np.zeros(total, dtype=np.float32)
    timesteps = np.zeros(total, dtype=np.int64)
    ep_of_step = np.zeros(total, dtype=np.int64)

    for e, (row, start) in enumerate(zip(index.itertuples(index=False), ep_starts)):
        with np.load(root / row.file) as ep:
            T = len(ep["actions"])
            assert T == row.length, f"{row.file}: index の length と一致しない"
            sl = slice(start, start + T)
            obs[sl] = ep["obs"]
            actions[sl] = ep["actions"]
            rewards[sl] = ep["rewards"]
            rtg[sl] = compute_rtg(ep["rewards"])
            timesteps[sl] = np.arange(T)
            ep_of_step[sl] = e
    obs.flush()
    del obs

    np.savez(meta_path, actions=actions, rewards=rewards, rtg=rtg, timesteps=timesteps,
             ep_of_step=ep_of_step, ep_starts=ep_starts, ep_lens=lengths,
             level_seeds=index["level_seed"].to_numpy(), success=index["success"].to_numpy(),
             ckpt_steps=index["ckpt_step"].to_numpy(), num_episodes=len(index),
             total_frames=total)
    print(f"built cache: {cache} ({len(index)} episodes, {total} frames)", flush=True)
    return cache


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, context_len: int, reward_scale: float = 0.1,
                 max_timestep: int = MAX_EPISODE_STEPS):
        self.cache = build_cache(data_dir)
        self.K = context_len
        self.reward_scale = reward_scale
        self.max_timestep = max_timestep
        with np.load(self.cache / "meta.npz") as meta:
            self.actions = meta["actions"]
            self.rtg = meta["rtg"]
            self.timesteps = meta["timesteps"]
            self.ep_of_step = meta["ep_of_step"]
            self.ep_starts = meta["ep_starts"]
            self.success = meta["success"]
        self._obs = None  # DataLoader の worker ごとに遅延して開く

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_obs"] = None
        return state

    @property
    def obs(self) -> np.ndarray:
        if self._obs is None:
            self._obs = np.load(self.cache / "obs.npy", mmap_mode="r")
        return self._obs

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, t: int) -> Dict[str, torch.Tensor]:
        K = self.K
        start = max(int(self.ep_starts[self.ep_of_step[t]]), t - K + 1)
        n = t - start + 1
        pad = K - n
        sl = slice(start, t + 1)

        states = np.zeros((K,) + OBS_SHAPE, dtype=np.uint8)
        states[pad:] = self.obs[sl]
        actions = np.zeros(K, dtype=np.int64)
        actions[pad:] = self.actions[sl]
        rtg = np.zeros((K, 1), dtype=np.float32)
        rtg[pad:, 0] = self.rtg[sl] * self.reward_scale
        timesteps = np.zeros(K, dtype=np.int64)
        timesteps[pad:] = np.minimum(self.timesteps[sl], self.max_timestep)
        mask = np.zeros(K, dtype=bool)
        mask[pad:] = True

        return {"states": torch.from_numpy(states), "actions": torch.from_numpy(actions),
                "rtg": torch.from_numpy(rtg), "timesteps": torch.from_numpy(timesteps),
                "mask": torch.from_numpy(mask)}
