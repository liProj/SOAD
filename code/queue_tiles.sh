#!/usr/bin/env bash
# After the tiled pilot: stitch + re-score it, then the tiled ablations (same 10 epochs = same number of iterations as 100 full-frame epochs).
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}; S=${1:-0}
bash code/wait_file.sh results/det/T_E4_aligned_s$S.json 900 > /dev/null
ev() { [ -s results/det/$1.json ] || $PY code/eval_tiles.py --weights $2 --tag $1 "${@:3}" > logs/eval_$1.log 2>&1; }
rs() { [ -s results/rescore_$1.json ] || $PY code/run_rescore.py $1 --ablate --seeds 3 > logs/rescore_$1.log 2>&1; }
tr() { n=$1; shift; [ -s results/det/$n.json ] || $PY code/run_det.py --name $n --seed $S --epochs 10 --extra '{"close_mosaic": 1, "save_period": 1}' "$@" > logs/$n.log 2>&1; }
ev T_E4_aligned_s${S}_full runs/T_E4_aligned_s$S/weights/last.pt; rs T_E4_aligned_s${S}_full
# ablation 1: tiling alone (RGB RT-DETR-L, no geometry, no sphere head)
tr T_rgb_s$S --model weights/rtdetr-l.pt --data configs/tiles_rgb.yaml
ev T_rgb_s${S}_full runs/T_rgb_s$S/weights/last.pt --data configs/tiles_rgb.yaml; rs T_rgb_s${S}_full
# ablation 2: tiling with the authors' loader (geometry dropped under mosaic, misaligned otherwise)
tr T_E4_unaligned_s$S --model configs/E4_rescore.yaml --pretrained weights/rtdetr-l.pt --data configs/tiles_geomBN.yaml
ev T_E4_unaligned_s${S}_full runs/T_E4_unaligned_s$S/weights/last.pt --data configs/tiles_geomBN.yaml; rs T_E4_unaligned_s${S}_full
