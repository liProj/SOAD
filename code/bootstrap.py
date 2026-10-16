"""Paired bootstrap over test frames: difference in mAP50, mAP50:95 and FPPI between two sets of full-frame detections.

usage: bootstrap.py <split> <detsA.pkl> <detsB.pkl> [n_boot]   (reports B - A)"""
import sys, json, pickle
import numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import load_gt, match
from ultralytics.utils.metrics import ap_per_class


def table(dets, gt, stems):
    """per-image match tables, so that a bootstrap resample is a concatenation."""
    return [(match(dets[k], gt[k]), dets[k][:, 4], dets[k][:, 5], gt[k][:, 0]) for k in stems]


def score(tab, idx, thr=0.25):
    tp = np.concatenate([tab[i][0] for i in idx]); conf = np.concatenate([tab[i][1] for i in idx])
    pc = np.concatenate([tab[i][2] for i in idx]); tc = np.concatenate([tab[i][3] for i in idx])
    ap = ap_per_class(tp, conf, pc, tc)[5]
    k = conf >= thr
    return ap[:, 0].mean(), ap.mean(), (~tp[k, 0]).sum() / len(idx)


def paired(split, da, db, n=1000, seed=0):
    gt = load_gt(split); stems = sorted(gt)
    ta, tb = table(da, gt, stems), table(db, gt, stems)
    rng = np.random.default_rng(seed); full = np.arange(len(stems))
    pa, pb = np.array(score(ta, full)), np.array(score(tb, full))
    diff = np.empty((n, 3))
    for j in range(n):
        idx = rng.integers(0, len(stems), len(stems))
        diff[j] = np.array(score(tb, idx)) - np.array(score(ta, idx))
    out = {}
    for i, name in enumerate(["map50", "map5095", "fppi"]):
        d = diff[:, i]; p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[name] = dict(a=float(pa[i]), b=float(pb[i]), diff=float(pb[i] - pa[i]), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                         p=float(max(p, 1.0 / n)))
    return out


if __name__ == "__main__":
    split, a, b = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    print(json.dumps(paired(split, pickle.load(open(a, "rb")), pickle.load(open(b, "rb")), n), indent=1))
