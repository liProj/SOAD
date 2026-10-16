#!/bin/bash
# Rebuild the summary table, every figure, the paper PDF, the Word report and the code-patch diff from results/. Nothing here trains a model.
set -u
PY=/home/snakehand/Documents/mdpi_papers/.venv/bin/python
A=/home/snakehand/Documents/mdpi_papers/AIIA
export PYTHONWARNINGS=ignore
cd $A
step() { echo "== $1"; shift; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "FAILED rc=$rc: $*"; exit $rc; fi; }
step collect   $PY -u code/collect.py
step bootstrap $PY -u code/run_bootstrap.py
step figs_res  $PY -u code/figs_results.py
step figs_rob  $PY -u code/figs_robust.py
step collect2  $PY -u code/collect.py
step figs_ext  $PY -u code/figs_extra.py
step paper     $PY -u code/write_paper.py
step compile   $PY -u code/compile_paper.py
step report    $PY -u code/write_report.py
diff -ruN -x '__pycache__' -x '*.pyc' third_party/Geosphere-DETR/ultralytics third_party/Geosphere-DETR-run/ultralytics > results/fork_patch.diff
echo "fork patch: $(grep -c '^diff ' results/fork_patch.diff) files changed"
