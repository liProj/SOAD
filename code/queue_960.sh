#!/usr/bin/env bash
# 960-px fine-tune of the tiled model (4 epochs from the 640-px tiled weights); starts when the tiled RGB ablation and YOLO11m are done.
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
bash code/wait_file.sh results/det/T_rgb_s0.json 1500 > /dev/null; bash code/wait_file.sh results/det/Y11m_s0.json 600 > /dev/null
[ -s results/det/T960_E4_aligned_s0.json ] || $PY code/run_det.py --name T960_E4_aligned_s0 --seed 0 --model runs/T_E4_aligned_s0/weights/last.pt --data configs/tiles_geomBN_aligned.yaml \
  --imgsz 960 --epochs 4 --extra '{"close_mosaic": 1, "save_period": 1, "warmup_epochs": 0.3}' > logs/T960_E4_aligned_s0.log 2>&1
[ -s results/det/T960_E4_aligned_s0_full.json ] || $PY code/eval_tiles.py --weights runs/T960_E4_aligned_s0/weights/last.pt --tag T960_E4_aligned_s0_full --imgsz 960 --batch 8 > logs/eval_T960_E4_aligned_s0_full.log 2>&1
