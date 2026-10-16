#!/usr/bin/env bash
# Second reproduction queue: starts when queue_repro.sh (pid $2) has exited. Paper's full model first, then the gated-fusion rung.
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}; S=${1:-0}
while kill -0 $2 2>/dev/null; do sleep 60; done
run() { n=$1; shift; [ -s results/det/$n.json ] || $PY code/run_det.py --name $n --seed $S "$@" > logs/$n.log 2>&1; }
run E4_rescore_s$S  --model configs/E4_rescore.yaml --pretrained weights/rtdetr-l.pt --data configs/data_geomBN.yaml
run E3_gated_s$S    --model configs/E3_gated.yaml   --pretrained weights/rtdetr-l.pt --data configs/data_geomBN.yaml
