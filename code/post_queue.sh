#!/usr/bin/env bash
# Post-processing that follows the training queues: as each run finishes, save its full-frame detections, run the
# flip-fusion / re-scoring evaluation where it applies, and rebuild tables, figures, paper and report.
cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
fin() { bash code/finalize.sh > logs/finalize_$1.log 2>&1; }
bash code/wait_file.sh results/det/Y11m_s0.json 1500 > /dev/null
[ -s results/det/Y11m_s0.json ] && $PY code/dump_dets.py Y11m_s0 > logs/dump_Y11m_s0.log 2>&1 && fin Y11m
bash code/wait_file.sh results/rescore_T_rgb_s0_full.json 1500 > /dev/null
[ -s results/rescore_T_rgb_s0_full.json ] && bash code/full_eval.sh T_rgb_s0 rgb > logs/full_eval_T_rgb_s0.log 2>&1 && fin T_rgb
bash code/wait_file.sh results/det/E4_rescore_s0.json 2400 > /dev/null
[ -s results/det/E4_rescore_s0.json ] && $PY code/dump_dets.py E4_rescore_s0 > logs/dump_E4_rescore_s0.log 2>&1 && fin E4
bash code/wait_file.sh results/det/T960_E4_aligned_s0_full.json 2400 > /dev/null; fin T960
bash code/wait_file.sh results/rescore_T_E4_unaligned_s0_full.json 2400 > /dev/null
[ -s results/rescore_T_E4_unaligned_s0_full.json ] && bash code/full_eval.sh T_E4_unaligned_s0 geomBN > logs/full_eval_T_E4_unaligned_s0.log 2>&1 && fin T_unaligned
bash code/wait_file.sh results/det/E3_gated_s0.json 2400 > /dev/null
[ -s results/det/E3_gated_s0.json ] && $PY code/dump_dets.py E3_gated_s0 > logs/dump_E3_gated_s0.log 2>&1 && fin E3
