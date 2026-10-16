"""Analysis helpers on full-frame detections (dict stem -> [n, 6] x1 y1 x2 y2 score cls)."""
import sys, pickle
import numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import load_gt, iou_matrix, match, froc, evaluate

SIZE_BINS = [0, 12, 16, 20, 24, 32, 48, 400]            # box side in the 1920x1200 frame (px)


def dets(tag, split):
    return pickle.load(open(f"{B}/results/dets/{tag}_{split}.pkl", "rb"))


def gt_hits(d, g, thr=0.25, iou=0.5):
    """bool per GT box: matched by a detection with score >= thr (same greedy rule as the evaluator)."""
    hit = np.zeros(len(g), bool)
    dd = d[d[:, 4] >= thr]
    if not len(dd) or not len(g):
        return hit
    m = iou_matrix(g[:, 1:5], dd[:, :4]) * (g[:, 0][:, None] == dd[:, 5][None])
    p = np.array(np.nonzero(m >= iou)).T
    if len(p):
        p = p[m[p[:, 0], p[:, 1]].argsort()[::-1]]
        p = p[np.unique(p[:, 1], return_index=True)[1]]
        p = p[np.unique(p[:, 0], return_index=True)[1]]
        hit[p[:, 0]] = True
    return hit


def recall_by(dets_, gt, key, bins, thr=0.25):
    """recall of annotated fruit binned by a per-GT quantity; key(g) -> [m] values."""
    v, h = [], []
    for k, g in gt.items():
        if len(g):
            v.append(key(g)); h.append(gt_hits(dets_[k], g, thr))
    v, h = np.concatenate(v), np.concatenate(h)
    i = np.digitize(v, bins) - 1
    return np.array([h[i == j].mean() if (i == j).any() else np.nan for j in range(len(bins) - 1)]), np.array([(i == j).sum() for j in range(len(bins) - 1)])


def side(g):
    return np.sqrt((g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2]))


def per_frame_counts(dets_, gt, thr=0.25):
    """annotated fruit, detections and true positives per frame (for counting-error plots)."""
    out = []
    for k, g in gt.items():
        d = dets_[k]; dd = d[d[:, 4] >= thr]
        out.append((len(g), len(dd), int(match(dd, g)[:, 0].sum()) if len(dd) else 0))
    return np.array(out)


def summary(tag, split="test"):
    gt = load_gt(split); d = dets(tag, split)
    e = evaluate(d, gt); c = per_frame_counts(d, gt)
    e["count_mae"] = float(np.abs(c[:, 1] - c[:, 0]).mean()); e["count_bias"] = float((c[:, 1] - c[:, 0]).mean())
    return e
