#!/usr/bin/env bash
# Uses the GPU slot freed by YOLO11m: 960-px fine-tune of the tiled model and its stitched evaluation, then the
# geometry-corruption evaluations. Replaces queue_960.sh and queue_robust.sh (same commands, no waiting).
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}; W=runs/T_E4_aligned_s0/weights/last.pt
[ -s results/det/T960_E4_aligned_s0.json ] || $PY code/run_det.py --name T960_E4_aligned_s0 --seed 0 --model $W --data configs/tiles_geomBN_aligned.yaml \
  --imgsz 960 --epochs 4 --extra '{"close_mosaic": 1, "save_period": 1, "warmup_epochs": 0.3}' > logs/T960_E4_aligned_s0.log 2>&1
[ -s results/det/T960_E4_aligned_s0_full.json ] || $PY code/eval_tiles.py --weights runs/T960_E4_aligned_s0/weights/last.pt --tag T960_E4_aligned_s0_full --imgsz 960 --batch 8 > logs/eval_T960_E4_aligned_s0_full.log 2>&1
for r in 0.1 0.3 0.5 0.7 1.0; do
  t=T_E4_aligned_s0_miss$r; [ -s results/det/$t.json ] || $PY code/eval_tiles.py --weights $W --tag $t --splits test --miss $r > logs/eval_$t.log 2>&1
done
for s in 0.02 0.05 0.1 0.2; do
  t=T_E4_aligned_s0_noise$s; [ -s results/det/$t.json ] || $PY code/eval_tiles.py --weights $W --tag $t --splits test --noise $s > logs/eval_$t.log 2>&1
done
