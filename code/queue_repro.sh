#!/usr/bin/env bash
# Reproduction with the authors' settings (README: imgsz 640, 100 epochs, batch 8, pretrained rtdetr-l.pt). usage: queue_repro.sh <seed>
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}; S=${1:-0}
run() { n=$1; shift; [ -s results/det/$n.json ] || $PY code/run_det.py --name $n --seed $S "$@" > logs/$n.log 2>&1; }
run E1_rtdetr_s$S   --model weights/rtdetr-l.pt --data configs/data_rgb.yaml &
run E3_gated_s$S    --model configs/E3_gated.yaml   --pretrained weights/rtdetr-l.pt --data configs/data_geomBN.yaml &
run E4_rescore_s$S  --model configs/E4_rescore.yaml --pretrained weights/rtdetr-l.pt --data configs/data_geomBN.yaml &
wait
run Y11m_s$S --kind yolo --model weights/yolo11m.pt --data configs/data_rgb.yaml &
run E2_concat_s$S   --model configs/E2_concat.yaml  --pretrained weights/rtdetr-l.pt --data configs/data_geomBN.yaml &
wait
