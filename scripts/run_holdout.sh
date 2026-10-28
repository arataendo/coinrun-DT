#!/usr/bin/env bash
# ホールドアウト検証: 重みを選ぶのに使ったのとは別のステージで難易度評価をやり直し、指標を比べる。
#
#   nohup bash scripts/run_holdout.sh > holdout.log 2>&1 &     # configs/cpu（seed 1000〜1199）
#   bash scripts/run_holdout.sh configs/small                  # 動作確認
#
# 使う設定は <設定ディレクトリ>/difficulty_holdout.yaml と weights_holdout.yaml。
# 評価は途中で止まっても、同じコマンドで評価済みのステージの続きから再開する。
# 結果は results/<設定名>_holdout/ に CSV で出る（git add できる）。
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR=${1:-configs/cpu}
NAME=$(basename "$CONFIG_DIR")
source scripts/common.sh
lock holdout

echo "config=$CONFIG_DIR python=$PY threads=$NUM_THREADS host=$(hostname)"
echo "=== difficulty (holdout): 開始 $(date '+%F %T') ==="
t0=$SECONDS
"$PY" -m coinrun_dt.difficulty.score --config "$CONFIG_DIR/difficulty_holdout.yaml"
echo "=== difficulty (holdout): 完了 $(date '+%F %T')（$(( (SECONDS - t0) / 60 )) 分）==="
echo
bash scripts/compare_weights.sh "$CONFIG_DIR/weights_holdout.yaml"
