"""Paired bootstrap for every comparison whose two detection sets exist; finished comparisons are kept from results/bootstrap.json."""
import os, sys, json
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from bootstrap import paired
from analysis import dets
E1, E4 = "E1_rtdetr_s0", "E4_rescore_s0"
P = [(E1, E4), ("T_rgb_s0_full", "T_E4_aligned_s0_full"), ("T_rgb_s0_wbf_per", "T_E4_aligned_s0_wbf_per"), ("T_E4_unaligned_s0_full", "T_E4_aligned_s0_full"),
     ("T_rgb_s0_full", "T_rgb20_s0_full"), ("T_rgb_s0_wbf_per", "T_rgb20_s0_wbf_per"), ("T_E4_aligned_s0_full", "T960_E4_aligned_s0_full")]
for b in ("T_rgb_s0", "T_rgb20_s0", "T_E4_aligned_s0"):
    P += [(E1, f"{b}_full"), (f"{b}_full", f"{b}_wbf"), (f"{b}_wbf", f"{b}_wbf_per"), (E1, f"{b}_wbf_per"), (E4, f"{b}_wbf_per"), (E4, f"{b}_full")]
p = f"{B}/results/bootstrap.json"
out = json.load(open(p)) if os.path.exists(p) else {}
for a, b in P:
    k = f"{a}->{b}"
    if k in out or not all(os.path.exists(f"{B}/results/dets/{t}_test.pkl") for t in (a, b)):
        continue
    out[k] = paired("test", dets(a, "test"), dets(b, "test"), 1000)
    print(k, {m: round(v["diff"], 4) for m, v in out[k].items()}, flush=True)
    json.dump(out, open(p, "w"), indent=1)
print("bootstrap comparisons:", len(out))
