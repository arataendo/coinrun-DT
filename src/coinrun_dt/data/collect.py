"""PPO の複数チェックポイントで軌跡を集める。

    python -m coinrun_dt.data.collect --config configs/collect_full.yaml

出力:
    <out_dir>/episodes/ep_XXXXXXX.npz   obs (T,64,64,3) uint8, actions (T,), rewards (T,)
    <out_dir>/index.csv                 file,length,return,success,level_seed,ckpt_step
既存の out_dir に追記できる。
"""
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from coinrun_dt.config import load_config
from coinrun_dt.envs import make_env
from coinrun_dt.ppo.model import PPOPolicy, list_checkpoints, load_ppo
from coinrun_dt.runner import Episode, run_episodes
from coinrun_dt.utils import get_device, set_seed

INDEX_FIELDS = ["file", "length", "return", "success", "level_seed", "ckpt_step"]


@dataclass
class CollectConfig:
    ckpt_dir: str = "checkpoints/ppo"
    ckpt_steps: Optional[List[int]] = None   # None なら ckpt_dir のすべてのチェックポイント。指定すると最も近いものを使う
    episodes_per_ckpt: int = 1000            # 実際の個数は ceil(E / num_envs) * num_envs
    num_envs: int = 64
    temperature: float = 1.0
    distribution_mode: str = "easy"
    start_level: int = 0
    num_levels: int = 0
    out_dir: str = "data/coinrun"
    overwrite: bool = False                  # True なら out_dir の既存データを消してから集める
    seed: int = 0
    device: str = "auto"


class EpisodeWriter:
    def __init__(self, out_dir: str):
        self.root = Path(out_dir)
        self.ep_dir = self.root / "episodes"
        self.ep_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.csv"
        is_new = not self.index_path.exists()
        self.count = 0 if is_new else len(pd.read_csv(self.index_path))
        self._f = open(self.index_path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=INDEX_FIELDS)
        if is_new:
            self._w.writeheader()

    def write(self, ep: Episode, ckpt_step: int) -> None:
        name = f"ep_{self.count:07d}.npz"
        np.savez_compressed(self.ep_dir / name, obs=ep.obs, actions=ep.actions, rewards=ep.rewards)
        self._w.writerow({"file": f"episodes/{name}", "length": ep.length, "return": ep.ret,
                          "success": int(ep.success), "level_seed": ep.level_seed,
                          "ckpt_step": ckpt_step})
        self._f.flush()
        self.count += 1

    def close(self) -> None:
        self._f.close()


def clear_output(out_dir: str) -> None:
    root = Path(out_dir)
    for d in ("episodes", "cache"):
        shutil.rmtree(root / d, ignore_errors=True)
    (root / "index.csv").unlink(missing_ok=True)


def select_checkpoints(ckpt_dir: str, steps: Optional[List[int]]):
    ckpts = list_checkpoints(ckpt_dir)
    if not ckpts:
        raise FileNotFoundError(f"チェックポイントがない: {ckpt_dir}")
    if steps is None:
        return ckpts
    chosen = {}
    for s in steps:
        step, path = min(ckpts, key=lambda c: abs(c[0] - s))
        chosen[step] = path
    return sorted(chosen.items())


def collect(cfg: CollectConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    if cfg.overwrite:
        clear_output(cfg.out_dir)
    writer = EpisodeWriter(cfg.out_dir)
    episodes_per_env = math.ceil(cfg.episodes_per_ckpt / cfg.num_envs)

    for k, (step, path) in enumerate(select_checkpoints(cfg.ckpt_dir, cfg.ckpt_steps)):
        model, _ = load_ppo(path, device)
        policy = PPOPolicy(model, device, sample=True, temperature=cfg.temperature)
        env = make_env(cfg.num_envs, cfg.distribution_mode, cfg.start_level, cfg.num_levels,
                       rand_seed=cfg.seed + k)
        stats = {"n": 0, "success": 0, "steps": 0}

        def on_episode(ep: Episode) -> None:
            writer.write(ep, step)
            stats["n"] += 1
            stats["success"] += int(ep.success)
            stats["steps"] += ep.length

        run_episodes(env, policy, episodes_per_env, record_obs=True, on_episode=on_episode)
        print(f"ckpt step={step}: episodes={stats['n']} "
              f"success_rate={stats['success'] / stats['n']:.3f} "
              f"mean_len={stats['steps'] / stats['n']:.1f}", flush=True)
        del env

    writer.close()
    index = pd.read_csv(writer.index_path)
    print("\n== データ全体（チェックポイント別） ==")
    print(index.groupby("ckpt_step").agg(episodes=("file", "count"), success_rate=("success", "mean"),
                                         mean_len=("length", "mean")).to_string())
    print(f"total episodes={len(index)} frames={int(index['length'].sum())} "
          f"success_rate={index['success'].mean():.3f}")


def main(argv=None) -> None:
    collect(load_config(CollectConfig, argv, description=__doc__))


if __name__ == "__main__":
    main()
