"""Convert the predictions.json written by run_det.py's final evaluation into full-frame detection pickles.
usage: dump_dets.py <run name>   ->  results/dets/<run>_{val,test}.pkl"""
import sys, pickle, json
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import load_gt, load_dets, evaluate
n = sys.argv[1]
for s in ("val", "test"):
    gt = load_gt(s); d = load_dets(f"{B}/runs/{n}/eval/{s}/predictions.json", gt)
    pickle.dump(d, open(f"{B}/results/dets/{n}_{s}.pkl", "wb"))
    e = evaluate(d, gt); print(n, s, round(e["map50"], 4), round(e["map5095"], 4), round(e["fppi"], 2), flush=True)
