"""ステージ難易度評価関数のエントリポイント。

    python -m coinrun_dt.difficulty.score --config configs/difficulty_full.yaml

ステージ（level seed）ごとに、次の値を出す:
- A. ロールアウトベースの指標と difficulty_rollout（[0, 1]、大きいほど難しい）
- B. DT 内部指標と difficulty_model（評価した全ステージで z-score 化した指標の重み付き和）
- 参照値: 最終 PPO 方策のステージごとの成功率 ppo_ref_success_rate（ppo_ckpt を指定したとき）
結果は <out_dir>/difficulty.csv に書き、続けて相関分析（analyze.py）も行う。
評価中はステージごとに difficulty_partial.csv に追記し、中断したら続きから再開する。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from coinrun_dt.config import load_config
from coinrun_dt.difficulty.analyze import analyze, combine_model_metrics
from coinrun_dt.difficulty.model_metrics import dt_greedy_reference, model_metrics, pick_reference
from coinrun_dt.difficulty.rollout_metrics import rollout_metrics
from coinrun_dt.dt.model import load_dt
from coinrun_dt.envs import make_level_env
from coinrun_dt.ppo.model import PPOPolicy, load_ppo
from coinrun_dt.runner import run_episodes
from coinrun_dt.utils import get_device, set_seed


@dataclass
class DifficultyConfig:
    dt_ckpt: str = "checkpoints/dt/dt_final.pt"
    ppo_ckpt: Optional[str] = None            # ファイルまたはディレクトリ（ディレクトリなら最新のもの）
    level_seeds: Optional[List[int]] = None   # 指定がなければ level_start から num_eval_levels 個
    level_start: int = 0
    num_eval_levels: int = 100
    distribution_mode: str = "easy"
    # A. ロールアウト
    target_returns: List[float] = field(default_factory=lambda: [10.0, 5.0, 0.0])
    rollouts_per_level: int = 32
    sample: bool = True
    temperature: float = 1.0
    steps_weight: float = 0.5
    # B. 内部指標
    reference_source: str = "ppo"             # "ppo"（最終 PPO 方策の軌跡）または "dt_greedy"
    ppo_reference_rollouts: int = 16          # PPO の参照成功率と参照軌跡に使う回数
    # 既定は expert_nll 単体（results/cpu の比較で最良。エントロピーは向きが逆だった）
    model_metric_weights: Dict[str, float] = field(default_factory=lambda: {"expert_nll": 1.0})
    metrics_batch_size: int = 256
    # 出力
    out_dir: str = "runs/difficulty"
    seed: int = 0
    device: str = "auto"

    def seeds(self) -> List[int]:
        if self.level_seeds is not None:
            return list(self.level_seeds)
        return list(range(self.level_start, self.level_start + self.num_eval_levels))


def evaluate_levels(dt_ckpt: str, level_seeds: List[int], cfg: DifficultyConfig,
                    partial_csv: Optional[Path] = None) -> pd.DataFrame:
    """partial_csv を渡すと、ステージごとの結果をそこに追記し、そこにあるステージは飛ばす（中断からの再開用）。"""
    device = get_device(cfg.device)
    model, ckpt = load_dt(dt_ckpt, device)
    reward_scale = ckpt["reward_scale"]
    ppo_policy = None
    if cfg.ppo_ckpt:
        ppo_model, _ = load_ppo(cfg.ppo_ckpt, device)
        ppo_policy = PPOPolicy(ppo_model, device, sample=True)
    if cfg.reference_source == "ppo" and ppo_policy is None:
        raise ValueError("reference_source='ppo' には ppo_ckpt が必要")

    high = max(cfg.target_returns)
    low = min(cfg.target_returns) if len(cfg.target_returns) > 1 else None
    rows = []
    if partial_csv is not None and partial_csv.exists():
        rows = pd.read_csv(partial_csv).to_dict("records")
        print(f"resumed: {len(rows)} levels already in {partial_csv}", flush=True)
    done = {int(r["level_seed"]) for r in rows}
    t0 = time.time()
    for n, seed in enumerate(level_seeds):
        if seed in done:
            continue
        row = {"level_seed": seed}
        row.update(rollout_metrics(model, reward_scale, seed, cfg.target_returns,
                                   cfg.rollouts_per_level, device, cfg.distribution_mode,
                                   cfg.sample, cfg.temperature, cfg.steps_weight, cfg.seed))

        ppo_eps = None
        if ppo_policy is not None:
            env = make_level_env(seed, cfg.ppo_reference_rollouts, cfg.distribution_mode, rand_seed=cfg.seed)
            ppo_eps = run_episodes(env, ppo_policy, episodes_per_env=1,
                                   record_obs=cfg.reference_source == "ppo")
            del env
            row["ppo_ref_success_rate"] = float(np.mean([e.success for e in ppo_eps]))

        if cfg.reference_source == "ppo":
            ref = pick_reference(ppo_eps)
        elif cfg.reference_source == "dt_greedy":
            ref = dt_greedy_reference(model, reward_scale, seed, high, device, cfg.distribution_mode)
        else:
            raise ValueError(f"未知の reference_source: {cfg.reference_source}")
        row.update(model_metrics(model, reward_scale, ref, high, low, device, cfg.metrics_batch_size))
        rows.append(row)
        if partial_csv is not None:
            pd.DataFrame([row]).to_csv(partial_csv, mode="a", header=not partial_csv.exists(), index=False)
        print(f"[{n + 1}/{len(level_seeds)}] seed={seed} "
              f"difficulty_rollout={row['difficulty_rollout']:.3f} "
              f"success_rate={row['success_rate']:.3f} entropy={row['mean_action_entropy']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df = df.set_index("level_seed").loc[level_seeds].reset_index()
    df["difficulty_model"] = combine_model_metrics(df, cfg.model_metric_weights)
    return df


def main(argv=None) -> None:
    cfg = load_config(DifficultyConfig, argv, description=__doc__)
    set_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "difficulty_partial.csv"
    df = evaluate_levels(cfg.dt_ckpt, cfg.seeds(), cfg, partial_csv=partial)
    df.to_csv(out / "difficulty.csv", index=False)
    partial.unlink()
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)
    print(f"saved {out / 'difficulty.csv'}")
    cols = ["difficulty_rollout", "difficulty_model"] + (
        ["ppo_ref_success_rate"] if "ppo_ref_success_rate" in df else [])
    print(df[cols].describe().to_string())
    print()
    analyze(str(out / "difficulty.csv"))


if __name__ == "__main__":
    main()
