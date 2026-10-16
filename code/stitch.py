"""Merge tile-level detections back into full 1920x1200 frames.

A detection that touches an interior tile edge is discarded (the neighbouring tile sees that fruit whole, because the
overlap of 213 / 280 px is ten times the fruit size); the rest are shifted to frame coordinates and de-duplicated
with class-wise NMS (IoU 0.5). At most 300 detections per frame are kept, like the single-pass detector."""
import json, os, sys, collections
import numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import load_gt, iou_matrix, evaluate
T, XS, YS = 640, [0, 427, 853, 1280], [0, 280, 560]


def nms(d, thr=0.5):
    keep = []
    for c in np.unique(d[:, 5]):
        idx = np.where(d[:, 5] == c)[0]; idx = idx[np.argsort(-d[idx, 4])]
        while len(idx):
            i = idx[0]; keep.append(i)
            if len(idx) == 1: break
            idx = idx[1:][iou_matrix(d[i:i + 1, :4], d[idx[1:], :4])[0] < thr]
    return d[sorted(keep, key=lambda i: -d[i, 4])]


def stitch(pred_json, gt, edge=2.0, iou=0.5, max_det=300, min_score=0.001, flip=False):
    per = collections.defaultdict(list)
    for r in json.load(open(pred_json)):
        if r["score"] < min_score: continue
        stem, yx = os.path.splitext(r["file_name"])[0].rsplit("__", 1)
        yi, xi = int(yx[0]), int(yx[1]); x, y, w, h = r["bbox"]
        if flip: x = T - x - w                           # predictions made on horizontally mirrored tiles
        if (x <= edge and xi > 0) or (y <= edge and yi > 0) or (x + w >= T - edge and xi < len(XS) - 1) or (y + h >= T - edge and yi < len(YS) - 1):
            continue
        per[stem].append([x + XS[xi], y + YS[yi], x + w + XS[xi], y + h + YS[yi], r["score"], r["category_id"] - 1])
    out = {}
    for k in gt:
        d = np.array(per.get(k, []), float).reshape(-1, 6)
        out[k] = nms(d, iou)[:max_det] if len(d) else d
    return out


if __name__ == "__main__":
    split, path = sys.argv[1], sys.argv[2]
    gt = load_gt(split); d = stitch(path, gt)
    print(json.dumps(evaluate(d, gt), indent=1))
