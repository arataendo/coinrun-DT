"""difficulty_model の重みの組み合わせを一括で比べ、結果を CSV で保存する。

    python -m coinrun_dt.difficulty.compare --config configs/cpu/weights.yaml

評価（score.py）はやり直さず、difficulty.csv の指標から合成し直すだけなので数秒で終わる。
出力（out_dir。既定は results/cpu で、git add できる）:
    weight_comparison.csv  指標単体と各合成の、参照値（PPO 成功率）・difficulty_rollout との Spearman 相関
    levels.csv             ステージごとの全指標と、各合成の値（model_<名前> 列）
    weights.yaml           比べた重みの組み合わせ
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from coinrun_dt.config import load_config
from coinrun_dt.difficulty.analyze import combine_model_metrics

# 単体でも比べる指標（大きいほど難しい向きのもの。success_rate は逆向きなので入れない）
SINGLE_METRICS = [
    "difficulty_rollout", "mean_steps_to_success", "rtg_sensitivity", "difficulty_model",
    "mean_action_entropy", "p95_action_entropy", "expert_nll", "rtg_kl",
]


@dataclass
class CompareConfig:
    csv: str = "runs/cpu/difficulty/difficulty.csv"
    out_dir: str = "results/cpu"
    reference: str = "ppo_ref_success_rate"     # 難しいほど小さくなる参照値
    weight_sets: Dict[str, Dict[str, float]] = field(default_factory=dict)


def strength(rho: float) -> str:
    """参照値（成功率）との相関から、難易度指標としての強さを付ける。負に大きいほど良い。"""
    if pd.isna(rho):
        return "n/a"
    if rho >= 0:
        return "reversed"
    if rho <= -0.7:
        return "strong"
    if rho <= -0.4:
        return "moderate"
    return "weak"


def compare(cfg: CompareConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv)
    if cfg.reference not in df:
        raise ValueError(f"CSV に参照値の列 {cfg.reference!r} がない（difficulty 設定で ppo_ckpt を指定する）")
    ref = df[cfg.reference]
    rollout = df["difficulty_rollout"]

    def row(name: str, kind: str, weights: str, values: pd.Series) -> dict:
        rho = values.corr(ref, method="spearman")
        return {"name": name, "kind": kind, "weights": weights,
                "rho_vs_ppo_success": round(rho, 3), "strength": strength(rho),
                "rho_vs_difficulty_rollout": round(values.corr(rollout, method="spearman"), 3)}

    rows = [row(m, "single", "", df[m]) for m in SINGLE_METRICS if m in df and df[m].nunique() > 1]
    levels = df.copy()
    for name, weights in cfg.weight_sets.items():
        values = combine_model_metrics(df, weights)
        levels[f"model_{name}"] = values
        desc = " ".join(f"{k}={w:g}" for k, w in weights.items())
        rows.append(row(f"model_{name}", "combined", desc, values))
    table = pd.DataFrame(rows).sort_values("rho_vs_ppo_success").reset_index(drop=True)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "weight_comparison.csv", index=False)
    levels.to_csv(out / "levels.csv", index=False)
    with open(out / "weights.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)

    print(f"levels={len(df)}  source={cfg.csv}\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table.to_string(index=False))
    print(f"\n見方: rho_vs_ppo_success が -1 に近いほど、PPO が苦戦するステージを難しいと判定できている"
          f"（上の行ほど良い）。strong ≦ -0.7 < moderate ≦ -0.4 < weak < 0 ≦ reversed（向きが逆）")
    print(f"\nsaved {out / 'weight_comparison.csv'}, {out / 'levels.csv'}")
    return table


def main(argv=None) -> None:
    compare(load_config(CompareConfig, argv, description=__doc__))


if __name__ == "__main__":
    main()
