#!/usr/bin/env bash
# 全工程（PPO → 軌跡収集 → DT → 難易度評価・相関分析）を一括実行する。
#
#   bash scripts/run_pipeline.sh                  # configs/cpu（研究室の CPU サーバ）
#   bash scripts/run_pipeline.sh configs/small    # 動作確認（数分）
#   nohup bash scripts/run_pipeline.sh > pipeline.log 2>&1 &   # ログアウトしても続ける
#
# 終わった工程は runs/<設定名>/.done/ に印を残し、再実行したときは飛ばす。途中で止まった工程は、
# PPO と DT は最新のチェックポイントから、難易度評価は評価済みのステージの続きから再開する。
# 軌跡収集は途中から再開できないので、最初からやり直す。
#
# 環境変数:
#   NUM_THREADS  torch の CPU スレッド数（既定: コア数と 16 の小さい方。この規模のモデルでは
#                スレッドを増やしすぎると、取り合いでかえって遅くなる）
#   FROM         この工程から後の印を消して再実行する（ppo / collect / dt / difficulty）。
#                PPO と DT はチェックポイントがあればそこから続けるので、最初から学習し直すときは
#                checkpoints/<設定名>/ppo や dt を消す
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR=${1:-configs/cpu}
NAME=$(basename "$CONFIG_DIR")
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
  PY=.venv/Scripts/python.exe
else
  PY=python
fi
CORES=$(nproc 2>/dev/null || echo 4)
NUM_THREADS=${NUM_THREADS:-$(( CORES < 16 ? CORES : 16 ))}
export OMP_NUM_THREADS=$NUM_THREADS MKL_NUM_THREADS=$NUM_THREADS PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

STEPS=(ppo collect dt difficulty)
DONE_DIR=runs/$NAME/.done
mkdir -p "$DONE_DIR"
if [[ -n ${FROM:-} ]]; then
  found=0
  for s in "${STEPS[@]}"; do
    [[ $s == "$FROM" ]] && found=1
    [[ $found == 1 ]] && rm -f "$DONE_DIR/$s"
  done
  [[ $found == 1 ]] || { echo "FROM は ${STEPS[*]} のどれか: $FROM" >&2; exit 1; }
fi

run_step() {
  local name=$1
  shift
  if [[ -f $DONE_DIR/$name ]]; then
    echo "=== $name: 完了済みなので飛ばす ==="
    return
  fi
  echo "=== $name: 開始 $(date '+%F %T') ==="
  local t0=$SECONDS
  "$PY" "$@"
  touch "$DONE_DIR/$name"
  echo "=== $name: 完了 $(date '+%F %T')（$(( (SECONDS - t0) / 60 )) 分）==="
}

echo "config=$CONFIG_DIR python=$PY threads=$NUM_THREADS host=$(hostname)"
run_step ppo        -m coinrun_dt.ppo.train        --config "$CONFIG_DIR/ppo.yaml"
run_step collect    -m coinrun_dt.data.collect     --config "$CONFIG_DIR/collect.yaml" --set overwrite=true
run_step dt         -m coinrun_dt.dt.train         --config "$CONFIG_DIR/dt.yaml"
run_step difficulty -m coinrun_dt.difficulty.score --config "$CONFIG_DIR/difficulty.yaml"
echo "すべて完了: runs/$NAME/difficulty/"
