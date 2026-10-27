"""DT のデータ元とは別のエージェントで参照値（ステージごとの成功率）を取り直し、難易度指標と比べる。

    python -m coinrun_dt.difficulty.reference --config configs/cpu/reference.yaml

元の参照値 ppo_ref_success_rate は、DT の学習データを作った PPO 自身の成功率なので循環がある。
ここでは、独立に学習した別の PPO（seed 違い）、DQN、ランダム方策で同じステージを動かし、
各指標がそれらの成功率とも相関するかを見る。指標は評価し直さず、levels.csv（results/ の比較結果）を使う。

出力（out_dir。git add できる）:
    comparison.csv  指標 × 参照エージェント × ステージの組ごとの Spearman 相関（-1 に近いほど良い）
    agreement.csv   参照エージェント同士の Spearman 相関（指標が届きうる上限の目安）
    levels.csv      ステージごとの指標と、各エージェントの成功率（ref_<名前> 列）
途中結果は work_dir にエージェントごとに追記し、中断したら続きから再開する。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from coinrun_dt.config import load_config
from coinrun_dt.difficulty.compare import SINGLE_METRICS, strength
from coinrun_dt.dqn.model import DQNPolicy, load_dqn
from coinrun_dt.envs import NUM_ACTIONS, make_level_env
from coinrun_dt.ppo.model import PPOPolicy, load_ppo
from coinrun_dt.runner import run_episodes
from coinrun_dt.utils import get_device, set_seed

BASE = "ppo_base"   # DT のデータを作った PPO（元の参照値 ppo_ref_success_rate）
# difficulty_model は、ステージの組ごとに重みが違う（cpu は最初の既定の重み）ので比べない
METRICS = [m for m in SINGLE_METRICS if m != "difficulty_model"]


@dataclass
class ReferenceConfig:
    level_csvs: Dict[str, str] = field(default_factory=dict)       # ステージの組の名前 → levels.csv
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 名前 → {kind: ppo|dqn|random, ckpt, epsilon}
    rollouts_per_level: int = 16
    distribution_mode: str = "easy"
    work_dir: str = "runs/reference"
    out_dir: str = "results/reference"
    seed: int = 0
    device: str = "auto"


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env_indices: np.ndarray) -> None:
        pass

    def observe(self, rew: np.ndarray, first: np.ndarray) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, NUM_ACTIONS, len(obs))


def make_policy(spec: Dict[str, Any], device, seed: int):
    kind = spec.get("kind")
    if kind == "ppo":
        model, _ = load_ppo(spec["ckpt"], device)
        return PPOPolicy(model, device, sample=True)
    if kind == "dqn":
        model, _ = load_dqn(spec["ckpt"], device)
        return DQNPolicy(model, device, epsilon=spec.get("epsilon", 0.05), seed=seed)
    if kind == "random":
        return RandomPolicy(seed)
    raise ValueError(f"未知のエージェントの kind: {kind!r}（ppo / dqn / random）")


def evaluate_agent(name: str, policy, level_seeds: List[int], cfg: ReferenceConfig,
                   partial_csv: Optional[Path] = None) -> pd.DataFrame:
    """ステージごとの成功率。partial_csv にあるステージは飛ばし、新しく評価したものを追記する。"""
    rows = []
    if partial_csv is not None and partial_csv.exists():
        rows = pd.read_csv(partial_csv).to_dict("records")
        print(f"[{name}] resumed: {len(rows)} levels already in {partial_csv}", flush=True)
    done = {int(r["level_seed"]) for r in rows}
    todo = [s for s in level_seeds if s not in done]
    t0 = time.time()
    for n, seed in enumerate(todo):
        env = make_level_env(seed, cfg.rollouts_per_level, cfg.distribution_mode, rand_seed=cfg.seed)
        eps = run_episodes(env, policy, episodes_per_env=1)
        del env
        row = {"level_seed": seed, "success_rate": float(np.mean([e.success for e in eps])),
               "mean_length": float(np.mean([e.length for e in eps]))}
        rows.append(row)
        if partial_csv is not None:
            pd.DataFrame([row]).to_csv(partial_csv, mode="a", header=not partial_csv.exists(), index=False)
        if (n + 1) % 20 == 0 or n + 1 == len(todo):
            print(f"[{name}] {n + 1}/{len(todo)} levels ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows).set_index("level_seed").loc[level_seeds].reset_index()


def compare_references(levels: pd.DataFrame, references: List[str],
                       metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """levels: level_set 列、参照値の列（成功率）、指標の列を持つ。

    返り値:
        comparison  metric, reference, level_set（各組と all）, n, rho, strength
        agreement   reference_a, reference_b, level_set, rho
    """
    groups = [(s, g) for s, g in levels.groupby("level_set", sort=False)] + [("all", levels)]
    comp, agree = [], []
    for set_name, g in groups:
        for ref in references:
            for m in metrics:
                if m not in g or g[m].nunique() < 2:
                    continue
                rho = g[m].corr(g[ref], method="spearman")
                comp.append({"metric": m, "reference": ref, "level_set": set_name, "n": len(g),
                             "rho": round(rho, 3), "strength": strength(rho)})
        for i, a in enumerate(references):
            for b in references[i + 1:]:
                agree.append({"reference_a": a, "reference_b": b, "level_set": set_name,
                              "rho": round(g[a].corr(g[b], method="spearman"), 3)})
    return pd.DataFrame(comp), pd.DataFrame(agree)


def load_levels(level_csvs: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for set_name, path in level_csvs.items():
        df = pd.read_csv(path)
        keep = ["level_seed", "ppo_ref_success_rate"] + [m for m in METRICS if m in df]
        frames.append(df[keep].assign(level_set=set_name))
    levels = pd.concat(frames, ignore_index=True).rename(columns={"ppo_ref_success_rate": BASE})
    dup = levels["level_seed"].duplicated()
    if dup.any():
        raise ValueError(f"ステージの組の間で seed が重なっている: {levels.loc[dup, 'level_seed'].tolist()[:10]}")
    return levels[["level_set", "level_seed", BASE] + [c for c in levels if c in METRICS]]


def run(cfg: ReferenceConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    levels = load_levels(cfg.level_csvs)
    seeds = levels["level_seed"].astype(int).tolist()
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    references = [BASE]
    for name, spec in cfg.agents.items():
        policy = make_policy(spec, device, cfg.seed)
        res = evaluate_agent(name, policy, seeds, cfg, partial_csv=work / f"{name}.csv")
        levels[f"ref_{name}"] = res["success_rate"].values
        references.append(f"ref_{name}")
    comparison, agreement = compare_references(levels, references, METRICS)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "comparison.csv", index=False)
    agreement.to_csv(out / "agreement.csv", index=False)
    levels.to_csv(out / "levels.csv", index=False)
    with open(out / "reference.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)

    print("\n平均成功率:")
    print(levels.groupby("level_set")[references].mean().round(3).to_string())
    with pd.option_context("display.width", 200):
        for set_name in comparison["level_set"].unique():
            table = comparison[comparison.level_set == set_name].pivot(
                index="metric", columns="reference", values="rho")[references]
            print(f"\n指標と各エージェントの成功率との Spearman 相関（level_set={set_name}）:")
            # 新しいエージェントの相関の平均の順（上ほど、データ元の PPO に依存しない難易度として良い）
            print(table.loc[table[references[1:]].mean(axis=1).sort_values().index].to_string())
        print("\n参照エージェント同士の Spearman 相関（level_set=all）:")
        print(agreement[agreement.level_set == "all"].to_string(index=False))
    print(f"\n見方: -1 に近いほど、そのエージェントが苦戦するステージを難しいと判定できている。"
          f"{BASE} 以外の列でも強い負の相関なら、DT のデータ元の PPO に依存しない難易度と言える。"
          f"\nsaved {out / 'comparison.csv'}, {out / 'agreement.csv'}, {out / 'levels.csv'}")
    return comparison, agreement


def main(argv=None) -> None:
    run(load_config(ReferenceConfig, argv, description=__doc__))


if __name__ == "__main__":
    main()
