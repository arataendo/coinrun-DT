"""C. 難易度指標どうし、および PPO 参照成功率との Spearman 相関を出す。

    python -m coinrun_dt.difficulty.analyze --csv runs/cpu/difficulty/difficulty.csv

--weights を渡すと、difficulty_model を CSV の指標から重みを変えて合成し直す（評価をやり直す必要はない）。
結果は既定で <csv のディレクトリ>/reweighted/ に出る。重みは負にもできる。

    python -m coinrun_dt.difficulty.analyze --csv runs/cpu/difficulty/difficulty.csv \\
        --weights expert_nll=1 rtg_kl=1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

METRICS = [
    "difficulty_rollout", "success_rate", "mean_steps_to_success", "rtg_sensitivity",
    "difficulty_model", "mean_action_entropy", "p95_action_entropy", "expert_nll", "rtg_kl",
    "ppo_ref_success_rate",
]
SCATTER_PAIRS = [
    ("difficulty_rollout", "difficulty_model"),
    ("ppo_ref_success_rate", "difficulty_rollout"),
    ("ppo_ref_success_rate", "difficulty_model"),
]


def zscore(x: pd.Series) -> pd.Series:
    std = x.std()
    if not np.isfinite(std) or std == 0:
        return x * 0.0
    return (x - x.mean()) / std


def combine_model_metrics(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """評価した全ステージで z-score 化した指標の重み付き和。"""
    unknown = [name for name in weights if name not in df]
    if unknown:
        raise ValueError(f"CSV にない指標: {unknown}")
    score = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        if w != 0:
            score = score + w * zscore(df[name].fillna(df[name].mean()))
    return score


def analyze(csv_path: str, out_dir: Optional[str] = None,
            weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if weights:
        out = Path(out_dir) if out_dir else Path(csv_path).parent / "reweighted"
        df["difficulty_model"] = combine_model_metrics(df, weights)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "difficulty.csv", index=False)
        with open(out / "weights.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"source": str(csv_path), "model_metric_weights": weights}, f, sort_keys=False)
        print("difficulty_model = " + " + ".join(f"{w:g} * z({k})" for k, w in weights.items()))
    else:
        out = Path(out_dir) if out_dir else Path(csv_path).parent
        out.mkdir(parents=True, exist_ok=True)

    cols = [c for c in METRICS if c in df and df[c].notna().sum() > 1 and df[c].nunique() > 1]
    corr = df[cols].corr(method="spearman")
    corr.to_csv(out / "correlation_spearman.csv")
    print(f"levels={len(df)}")
    print(corr.round(3).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return corr
    pairs = [(x, y) for x, y in SCATTER_PAIRS if x in cols and y in cols]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 4), squeeze=False)
        for ax, (x, y) in zip(axes[0], pairs):
            ax.scatter(df[x], df[y], s=12, alpha=0.7)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"Spearman ρ = {corr.loc[x, y]:.3f}")
        fig.tight_layout()
        fig.savefig(out / "scatter.png", dpi=120)
        plt.close(fig)
        print(f"saved {out / 'scatter.png'}")
    return corr


def parse_weights(items: List[str]) -> Dict[str, float]:
    weights = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--weights の形式は NAME=WEIGHT: {item!r}")
        weights[key] = float(value)
    return weights


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="runs/difficulty/difficulty.csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--weights", nargs="+", default=None, metavar="NAME=WEIGHT",
                        help="difficulty_model を合成し直す重み（例: expert_nll=1 rtg_kl=1）")
    args = parser.parse_args(argv)
    analyze(args.csv, args.out_dir, parse_weights(args.weights) if args.weights else None)


if __name__ == "__main__":
    main()
