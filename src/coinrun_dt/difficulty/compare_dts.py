"""学習データの違う DT（例: PPO の各段階を混ぜたもの / 最終 PPO だけのもの）の難易度指標を並べて比べる。

    python -m coinrun_dt.difficulty.compare_dts --config configs/cpu/dt_comparison.yaml

各 DT の評価結果（levels.csv または difficulty.csv）を、ステージの組（train / holdout）ごとに読む。
出力（out_dir。git add できる）:
    comparison.csv  指標 × ステージの組ごとの、各 DT での PPO 成功率との Spearman 相関（rho_<DT 名> 列。-1 に近いほど良い）
    summary.csv     DT × ステージの組ごとの、DT と PPO の平均成功率
    levels.csv      ステージごとの、各 DT の主な指標（<指標>_<DT 名> 列）と PPO 成功率
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from coinrun_dt.config import load_config
from coinrun_dt.difficulty.compare import SINGLE_METRICS

REFERENCE = "ppo_ref_success_rate"
# difficulty_model は評価ごとに重みが違うことがあるので比べない
METRICS = [m for m in SINGLE_METRICS if m != "difficulty_model"]


@dataclass
class DTComparisonConfig:
    runs: Dict[str, Dict[str, str]] = field(default_factory=dict)   # DT 名 → {ステージの組 → CSV}
    out_dir: str = "results/dt_comparison"


def compare_dts(runs: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = list(runs)
    sets = list(dict.fromkeys(s for r in runs.values() for s in r))
    comp, summary, level_frames = [], [], []
    for set_name in sets + ["all"]:
        dfs = {}
        for name in names:
            parts = [df for s, df in runs[name].items() if set_name in (s, "all")]
            if parts:
                dfs[name] = pd.concat(parts, ignore_index=True)
        if len(dfs) < len(names):
            continue   # どれかの DT にこの組の結果がない
        for name, df in dfs.items():
            summary.append({"dt": name, "level_set": set_name, "n": len(df),
                            "dt_success_rate": round(df["success_rate"].mean(), 3),
                            "ppo_success_rate": round(df[REFERENCE].mean(), 3)})
        for m in METRICS:
            row = {"metric": m, "level_set": set_name}
            for name, df in dfs.items():
                ok = m in df and df[m].nunique() > 1
                row[f"rho_{name}"] = round(df[m].corr(df[REFERENCE], method="spearman"), 3) if ok else float("nan")
            comp.append(row)
        if set_name != "all":
            merged = None
            for name, df in dfs.items():
                cols = df[["level_seed"] + [m for m in METRICS + ["success_rate"] if m in df]]
                cols = cols.rename(columns={c: f"{c}_{name}" for c in cols if c != "level_seed"})
                if merged is None:
                    merged = df[["level_seed", REFERENCE]].merge(cols, on="level_seed")
                else:
                    merged = merged.merge(cols, on="level_seed")
            level_frames.append(merged.assign(level_set=set_name))
    levels = pd.concat(level_frames, ignore_index=True) if level_frames else pd.DataFrame()
    return pd.DataFrame(comp), pd.DataFrame(summary), levels


def run(cfg: DTComparisonConfig):
    runs = {name: {s: pd.read_csv(p) for s, p in sets.items()} for name, sets in cfg.runs.items()}
    comparison, summary, levels = compare_dts(runs)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "comparison.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    levels.to_csv(out / "levels.csv", index=False)
    with open(out / "dt_comparison.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)

    print("平均成功率（DT は目標リターン 10）:")
    print(summary.to_string(index=False))
    rho_cols = [f"rho_{n}" for n in cfg.runs]
    with pd.option_context("display.width", 200):
        for set_name in comparison["level_set"].unique():
            table = comparison[comparison.level_set == set_name].set_index("metric")[rho_cols]
            print(f"\nPPO 成功率との Spearman 相関（level_set={set_name}）:")
            print(table.loc[table.min(axis=1).sort_values().index].to_string())
    print(f"\n見方: -1 に近いほど、PPO が苦戦するステージを難しいと判定できている。"
          f"\nsaved {out / 'comparison.csv'}, {out / 'summary.csv'}, {out / 'levels.csv'}")
    return comparison, summary


def main(argv=None) -> None:
    run(load_config(DTComparisonConfig, argv, description=__doc__))


if __name__ == "__main__":
    main()
