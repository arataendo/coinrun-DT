"""C. 難易度指標どうし、および PPO 参照成功率との Spearman 相関を出す。

    python -m coinrun_dt.difficulty.analyze --csv runs/difficulty/difficulty.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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


def analyze(csv_path: str, out_dir: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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
        print(f"saved {out / 'scatter.png'}")
    return corr


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="runs/difficulty/difficulty.csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    analyze(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
