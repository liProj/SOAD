#!/usr/bin/env bash
# Robustness of the tiled model to corrupted geometry on the held-out field (single view): block masking of the curvature maps
# (the protocol of the public code) and additive Gaussian noise. Starts when the tiled RGB ablation is done.
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}; W=runs/T_E4_aligned_s0/weights/last.pt
bash code/wait_file.sh results/det/T_rgb_s0.json 1500 > /dev/null
for r in 0.1 0.3 0.5 0.7 1.0; do
  t=T_E4_aligned_s0_miss$r; [ -s results/det/$t.json ] || $PY code/eval_tiles.py --weights $W --tag $t --splits test --miss $r > logs/eval_$t.log 2>&1
done
for s in 0.02 0.05 0.1 0.2; do
  t=T_E4_aligned_s0_noise$s; [ -s results/det/$t.json ] || $PY code/eval_tiles.py --weights $W --tag $t --splits test --noise $s > logs/eval_$t.log 2>&1
done
