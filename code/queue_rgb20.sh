#!/usr/bin/env bash
# Second 10-epoch cycle of the tiled RGB model (starts from the first cycle's last weights, same settings), then the
# stitched single-view / mirrored / fused / re-scored evaluation. Takes the GPU slot of the stopped gated-fusion rung.
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
[ -s results/det/T_rgb20_s0.json ] || $PY code/run_det.py --name T_rgb20_s0 --seed 0 --epochs 10 --extra '{"close_mosaic": 1, "save_period": 1, "warmup_epochs": 0.3}' \
  --model runs/T_rgb_s0/weights/last.pt --data configs/tiles_rgb.yaml > logs/T_rgb20_s0.log 2>&1
[ -s runs/T_rgb20_s0/weights/last.pt ] && bash code/full_eval.sh T_rgb20_s0 rgb > logs/full_eval_T_rgb20_s0.log 2>&1
