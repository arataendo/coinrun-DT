# scripts/*.sh から source する共通設定（単体では実行しない）。
#   PY           使う Python（.venv があればそれ）
#   NUM_THREADS  torch の CPU スレッド数（既定: コア数と 16 の小さい方。この規模のモデルでは
#                スレッドを増やしすぎると、取り合いでかえって遅くなる）
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
