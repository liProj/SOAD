#!/usr/bin/env bash
# full_eval.sh <run name> <tile config suffix: geomBN_aligned | geomBN | rgb> [weights file, default last.pt]
# stitched single-view -> mirrored view -> box fusion -> physical-evidence re-scoring (fitted on val only). Tags: <run>_full, _flip, _wbf.
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}; R=$1; C=$2; W=runs/$R/weights/${3:-last.pt}
[ -s results/det/${R}_full.json ] || $PY code/eval_tiles.py --weights $W --tag ${R}_full --data configs/tiles_$C.yaml > logs/eval_${R}_full.log 2>&1
[ -s results/det/${R}_flip.json ] || $PY code/eval_tiles.py --weights $W --tag ${R}_flip --data configs/tiles_flip_$C.yaml --flip > logs/eval_${R}_flip.log 2>&1
$PY - <<P
import sys, pickle; sys.path.insert(0, "code")
from analysis import dets
from merge import wbf
for s in ("val", "test"):
    pickle.dump(wbf([dets("${R}_full", s), dets("${R}_flip", s)], 0.55), open(f"results/dets/${R}_wbf_{s}.pkl", "wb"))
P
for t in ${R}_full ${R}_wbf; do [ -s results/rescore_$t.json ] || $PY code/run_rescore.py $t --ablate --seeds 3 > logs/rescore_$t.log 2>&1; done
grep -hE '^base|^test PER' logs/rescore_${R}_full.log logs/rescore_${R}_wbf.log
