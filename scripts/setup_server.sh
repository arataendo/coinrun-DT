#!/usr/bin/env bash
# 研究室サーバ（Linux, CPU）の初回セットアップ。.venv を作り、依存を入れてテストする。
#
#   bash scripts/setup_server.sh
#   PYTHON=/path/to/python3.10 bash scripts/setup_server.sh   # Python 3.10 の場所を指定する場合
#
# procgen 0.10.7 は Python 3.10 以下でしか入らない。python3.10 がなく uv があれば uv で用意する。
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3.10}
if [[ ! -x .venv/bin/python ]]; then
  if command -v "$PYTHON" >/dev/null 2>&1; then
    "$PYTHON" -m venv .venv
  elif command -v uv >/dev/null 2>&1; then
    uv venv --python 3.10 --seed .venv
  else
    echo "Python 3.10 が見つからない。PYTHON=/path/to/python3.10 を指定するか、uv か conda で用意する。" >&2
    exit 1
  fi
fi
PY=.venv/bin/python
"$PY" -c 'import sys; assert sys.version_info < (3, 11), f"Python {sys.version.split()[0]} では procgen が入らない（3.10 以下が必要）"'

"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版（CUDA 版の巨大な wheel を避ける）
"$PY" -m pip install --no-build-isolation gym==0.26.2                         # procgen の依存（sdist のみ）
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e . --no-deps --no-build-isolation
"$PY" -m pytest -q -W ignore

echo "setup OK: $("$PY" --version), torch $("$PY" -c 'import torch; print(torch.__version__)'), $(nproc) cores"
