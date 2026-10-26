#!/usr/bin/env bash
# difficulty_model の重みの組み合わせを一括で比べ、結果を results/ に CSV で保存する（git add できる）。
#
#   bash scripts/compare_weights.sh                              # configs/cpu/weights.yaml
#   bash scripts/compare_weights.sh configs/small/weights.yaml
#
# 比べる組み合わせは weights.yaml の weight_sets に足す。評価はやり直さないので数秒で終わる。
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/cpu/weights.yaml}
source scripts/common.sh

"$PY" -m coinrun_dt.difficulty.compare --config "$CONFIG"
OUT=$("$PY" -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['out_dir'])" "$CONFIG")
echo
echo "GitHub に上げるとき: git add $OUT && git commit -m \"重みの比較結果\" && git push"
