# scripts/*.sh から source する共通設定（単体では実行しない）。
#   PY           使う Python（.venv があればそれ）
#   NUM_THREADS  torch の CPU スレッド数（既定: コア数と 16 の小さい方。この規模のモデルでは
#                スレッドを増やしすぎると、取り合いでかえって遅くなる）
#   run_step     工程の実行と完了の印（$DONE_DIR を先に決めておく）
#   lock         二重起動の防止（$NAME を先に決めておく）
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

# run_step 名前 Python の引数...: 工程を1つ実行し、終わったら $DONE_DIR/名前 に印を残す。印があれば飛ばす
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

# lock 名前: 同じ設定で同じスクリプトが動いていれば、二重に起動せずに止める（前回の PID が残っていても、
# そのプロセスが終わっていれば起動する）。$NAME を先に決めておく
lock() {
  local file=runs/$NAME/.lock_$1
  mkdir -p "runs/$NAME"
  if [[ -f $file ]] && kill -0 "$(cat "$file")" 2>/dev/null; then
    echo "すでに動いている: $1（設定 $NAME、PID $(cat "$file")）。止めるときは kill $(cat "$file")" >&2
    exit 1
  fi
  echo $$ > "$file"
  trap "rm -f '$file'" EXIT
}
