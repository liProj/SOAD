"""Fit physical-evidence re-scoring on the validation detections of one detector and apply it to the held-out test field.

alpha is chosen on validation out-of-fold detections only; the test split is never used for any choice.
usage: run_rescore.py <tag> [--ablate]"""
import sys, json, pickle, argparse
import numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import load_gt, evaluate, fppi_at_recall
import rescore as R
ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--ablate", action="store_true"); ap.add_argument("--seeds", type=int, default=1)
a = ap.parse_args()
ALL = ["ctx", "box", "depth", "size", "curv", "colour"]
ALL = [g for g in ALL if g == "ctx" or g in R.GROUPS]
gt = {s: load_gt(s) for s in ("val", "test")}
det = {s: pickle.load(open(f"{B}/results/dets/{a.tag}_{s}.pkl", "rb")) for s in gt}
F = {s: R.featurize(s, det[s], gt[s], cache=f"{B}/results/dets/{a.tag}_{s}_feat.pkl") for s in gt}


def summ(d, s, rec):
    e = evaluate(d, gt[s]); e["fppi_at_base_recall"] = fppi_at_recall(d, gt[s], rec); return e


base = {s: evaluate(det[s], gt[s]) for s in gt}
rec = {s: base[s]["recall"] for s in gt}                       # recall of the raw detector at conf 0.25
for s in gt:
    base[s]["fppi_at_base_recall"] = fppi_at_recall(det[s], gt[s], rec[s])
out = {"tag": a.tag, "base": base, "groups": ALL}
print("base", {s: (round(base[s]["map50"], 4), round(base[s]["map5095"], 4), round(base[s]["fppi"], 2)) for s in gt}, flush=True)


def run(groups, alpha, seed=0):
    cols = R.columns(groups)
    m = R.fit(F["val"], sorted(gt["val"]), cols, seed)
    return R.apply(m, F["test"], det["test"], cols, alpha)


# alpha chosen on validation OOF
cols = R.columns(ALL); grid = {}
for al in (0.0, 0.25, 0.5, 0.75):
    grid[al] = summ(R.oof(F["val"], det["val"], gt["val"], cols, al), "val", rec["val"])
    print("val-oof alpha", al, round(grid[al]["map50"], 4), round(grid[al]["map5095"], 4), round(grid[al]["fppi_at_base_recall"], 2), flush=True)
best = max(grid, key=lambda k: grid[k]["map5095"])
out["val_oof_alpha"] = {str(k): v for k, v in grid.items()}; out["alpha"] = best
tests = []
for sd in range(a.seeds):
    d = run(ALL, best, sd); tests.append(summ(d, "test", rec["test"]))
    if sd == 0:
        pickle.dump(d, open(f"{B}/results/dets/{a.tag}_per_test.pkl", "wb"))
out["test"] = tests
print("test PER alpha", best, [(round(t["map50"], 4), round(t["map5095"], 4), round(t["fppi"], 2), round(t["fppi_at_base_recall"], 2)) for t in tests], flush=True)
if a.ablate:
    ab = {}
    sets = {"score only": ["score"], "ctx": ["ctx"]}
    for g in ALL[1:]:
        sets[f"ctx+{g}"] = ["ctx", g]
    for g in ALL[1:]:
        sets[f"all-{g}"] = [x for x in ALL if x != g]
    sets["all"] = ALL
    for k, gs in sets.items():
        ab[k] = dict(val_oof=summ(R.oof(F["val"], det["val"], gt["val"], R.columns(gs), best), "val", rec["val"]),
                     test=summ(run(gs, best), "test", rec["test"]))
        print(k, "val", round(ab[k]["val_oof"]["map5095"], 4), "test", round(ab[k]["test"]["map50"], 4), round(ab[k]["test"]["map5095"], 4),
              round(ab[k]["test"]["fppi_at_base_recall"], 2), flush=True)
    out["ablation"] = ab
    cols = R.columns(ALL); clf, reg = R.fit(F["val"], sorted(gt["val"]), cols, 0)
    names = [(R.CTX + R.NAMES)[i] for i in cols]
    out["importance"] = dict(names=names, clf=clf.booster_.feature_importance("gain").tolist(), reg=reg.booster_.feature_importance("gain").tolist())
json.dump(out, open(f"{B}/results/rescore_{a.tag}.json", "w"), indent=1)
