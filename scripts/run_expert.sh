#!/usr/bin/env bash
# 最終 PPO の軌跡だけで DT（熟練者 DT）を作り、その DT で難易度を評価して、各段階を混ぜた DT と比べる。
#
#   nohup bash scripts/run_expert.sh > expert.log 2>&1 &     # configs/cpu
#   bash scripts/run_expert.sh configs/small                 # 動作確認
#
# 工程: 軌跡収集（collect_expert.yaml）→ DT 学習（dt_expert.yaml）→ 難易度評価
# （seed 0〜199 と 1000〜1199。difficulty_expert*.yaml）→ 比較。
# 先に run_pipeline.sh と run_holdout.sh を終えておく（PPO と、比べる相手の DT の結果を使う）。
# 止まっても、同じコマンドで続きから再開する（終わった工程は runs/<設定名>/.done/ の印で飛ばす）。
# 結果は results/<設定名>_expert/, _expert_holdout/, _dt_comparison/ に CSV で出る（git add できる）。
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR=${1:-configs/cpu}
NAME=$(basename "$CONFIG_DIR")
source scripts/common.sh
DONE_DIR=runs/$NAME/.done
mkdir -p "$DONE_DIR"

echo "config=$CONFIG_DIR python=$PY threads=$NUM_THREADS host=$(hostname)"
run_step collect_expert            -m coinrun_dt.data.collect     --config "$CONFIG_DIR/collect_expert.yaml" --set overwrite=true
run_step dt_expert                 -m coinrun_dt.dt.train         --config "$CONFIG_DIR/dt_expert.yaml"
run_step difficulty_expert         -m coinrun_dt.difficulty.score --config "$CONFIG_DIR/difficulty_expert.yaml"
run_step difficulty_expert_holdout -m coinrun_dt.difficulty.score --config "$CONFIG_DIR/difficulty_expert_holdout.yaml"

# 指標単体と重みの組み合わせの比較（weights*.yaml の組み合わせを、熟練者 DT の結果に当てる）
for h in "" _holdout; do
  "$PY" -m coinrun_dt.difficulty.compare --config "$CONFIG_DIR/weights$h.yaml" \
    --set csv=runs/$NAME/difficulty_expert$h/difficulty.csv out_dir=results/${NAME}_expert$h
done
"$PY" -m coinrun_dt.difficulty.compare_dts --config "$CONFIG_DIR/dt_comparison.yaml"
OUT=$("$PY" -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['out_dir'])" "$CONFIG_DIR/dt_comparison.yaml")
echo
echo "GitHub に上げるとき: git add results/${NAME}_expert results/${NAME}_expert_holdout $OUT && git commit -m \"熟練者 DT の結果\" && git push"
