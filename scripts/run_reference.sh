#!/usr/bin/env bash
# DT のデータ元とは別のエージェントで参照値を取り直し、難易度指標と比べる。
#
#   nohup bash scripts/run_reference.sh > reference.log 2>&1 &    # configs/cpu
#   bash scripts/run_reference.sh configs/small                   # 動作確認
#
# 1. 2本目の PPO（seed 違い、ppo_seed1.yaml）と DQN（dqn.yaml）を学習する。コア数に余裕があれば並列に回し、
#    それぞれのログは runs/<設定名>/logs/ に出る
# 2. 両者とランダム方策で、results/ の比較で使ったステージ（seed 0〜199 と 1000〜1199）を動かし、
#    成功率と各指標の相関を results/<設定名>_reference/ に CSV で出す（git add できる）
# 先に run_pipeline.sh と run_holdout.sh を終えておく。止まっても同じコマンドで続きから再開する。
#
# 環境変数:
#   PARALLEL  1 なら2つの学習を並列に回す（既定: コア数がスレッド数の2倍以上あれば 1）
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR=${1:-configs/cpu}
NAME=$(basename "$CONFIG_DIR")
source scripts/common.sh
DONE_DIR=runs/$NAME/.done
LOG_DIR=runs/$NAME/logs
mkdir -p "$DONE_DIR" "$LOG_DIR"
PARALLEL=${PARALLEL:-$(( CORES >= 2 * NUM_THREADS ? 1 : 0 ))}

echo "config=$CONFIG_DIR python=$PY threads=$NUM_THREADS parallel=$PARALLEL host=$(hostname)"
if [[ $PARALLEL == 1 ]]; then
  run_step ppo_seed1 -m coinrun_dt.ppo.train --config "$CONFIG_DIR/ppo_seed1.yaml" > "$LOG_DIR/ppo_seed1.log" 2>&1 &
  p1=$!
  run_step dqn -m coinrun_dt.dqn.train --config "$CONFIG_DIR/dqn.yaml" > "$LOG_DIR/dqn.log" 2>&1 &
  p2=$!
  echo "PPO (seed 1) と DQN を並列で学習中 $(date '+%F %T'): tail -f $LOG_DIR/ppo_seed1.log $LOG_DIR/dqn.log"
  status=0
  wait $p1 || status=1
  wait $p2 || status=1
  tail -n 1 "$LOG_DIR/ppo_seed1.log" "$LOG_DIR/dqn.log"
  [[ $status == 0 ]] || { echo "学習が失敗した。ログを見る: $LOG_DIR/" >&2; exit 1; }
else
  run_step ppo_seed1 -m coinrun_dt.ppo.train --config "$CONFIG_DIR/ppo_seed1.yaml"
  run_step dqn       -m coinrun_dt.dqn.train --config "$CONFIG_DIR/dqn.yaml"
fi

echo "=== reference: 開始 $(date '+%F %T') ==="
"$PY" -m coinrun_dt.difficulty.reference --config "$CONFIG_DIR/reference.yaml"
OUT=$("$PY" -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['out_dir'])" "$CONFIG_DIR/reference.yaml")
echo
echo "GitHub に上げるとき: git add $OUT && git commit -m \"別のエージェントでの検証結果\" && git push"
