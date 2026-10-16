"""Weighted box fusion of several full-frame detection sets of the same frames (e.g. original + mirrored tiles).

Boxes of the same class are clustered greedily in score order (IoU >= thr with the running fused box); the fused box is the
score-weighted mean and the fused score is the sum of member scores divided by the number of sets, so a fruit found by
only one of the views is down-weighted."""
import sys
import numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import iou_matrix


def wbf_frame(ds, thr=0.55, max_det=300):
    n = len(ds); d = np.concatenate(ds)
    if not len(d):
        return d
    out = []
    for c in np.unique(d[:, 5]):
        x = d[d[:, 5] == c]; x = x[np.argsort(-x[:, 4])]
        fb = np.zeros((0, 4)); acc = []                       # fused boxes, [sum_w*box, sum_w, count]
        for b in x:
            j = -1
            if len(fb):
                iou = iou_matrix(b[None, :4], fb)[0]; j = int(iou.argmax()) if iou.max() >= thr else -1
            if j < 0:
                fb = np.vstack([fb, b[:4]]); acc.append([b[:4] * b[4], b[4], 1])
            else:
                acc[j][0] += b[:4] * b[4]; acc[j][1] += b[4]; acc[j][2] += 1; fb[j] = acc[j][0] / acc[j][1]
        out += [np.r_[fb[i], min(acc[i][1], n) / n, c] for i in range(len(fb))]
    out = np.array(out)
    return out[np.argsort(-out[:, 4])][:max_det]


def wbf(sets, thr=0.55):
    return {k: wbf_frame([s[k] for s in sets], thr) for k in sets[0]}
