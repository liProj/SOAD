"""Offline detection evaluation that mirrors the fork's validator (same matching rule, same AP routine),
so that re-scored detections can be evaluated without touching the detector.

Detections are a dict stem -> array [n, 6] = x1, y1, x2, y2, score, cls in original pixels (1920x1200)."""
import json, os, sys, collections
import numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/third_party/Geosphere-DETR-run")
from ultralytics.utils.metrics import ap_per_class
W, H = 1920, 1200
IOUV = np.linspace(0.5, 0.95, 10)


def load_gt(split):
    """stem -> [m, 5] = cls, x1, y1, x2, y2; images whose label file has a negative value are dropped, as the loader does."""
    d, gt = f"{B}/data/citrus/{split}/labels", {}
    for f in sorted(os.listdir(d)):
        a = np.loadtxt(os.path.join(d, f), ndmin=2).reshape(-1, 5) if os.path.getsize(os.path.join(d, f)) else np.zeros((0, 5))
        if (a < 0).any():
            continue
        a = np.unique(a, axis=0) if len(a) else a          # the loader removes duplicate rows
        x, y, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
        gt[f[:-4]] = np.stack([a[:, 0], x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
    return gt


def load_dets(path, gt=None):
    out = collections.defaultdict(list)
    for r in json.load(open(path)):
        x, y, w, h = r["bbox"]
        out[os.path.splitext(r["file_name"])[0]].append([x, y, x + w, y + h, r["score"], r["category_id"] - 1])
    out = {k: np.array(v, float) for k, v in out.items()}
    if gt is not None:
        for k in gt:
            out.setdefault(k, np.zeros((0, 6)))
        out = {k: out[k] for k in gt}
    return out


def iou_matrix(a, b):
    """a [m,4], b [n,4] -> [m,n]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-7)


def match(det, g):
    """Same greedy rule as BaseValidator.match_predictions -> bool [n, 10]."""
    correct = np.zeros((len(det), 10), bool)
    if len(det) == 0 or len(g) == 0:
        return correct
    iou = iou_matrix(g[:, 1:5], det[:, :4]) * (g[:, 0][:, None] == det[:, 5][None])
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def tp_table(dets, gt):
    """Concatenate per-image matches; detections are first sorted by score inside each image (as the detector emits them)."""
    tps, conf, pc, tc, img = [], [], [], [], []
    for k, g in gt.items():
        d = dets[k]
        tps.append(match(d, g)); conf.append(d[:, 4]); pc.append(d[:, 5]); tc.append(g[:, 0]); img.append(np.full(len(d), len(img)))
    return np.concatenate(tps), np.concatenate(conf), np.concatenate(pc), np.concatenate(tc), np.concatenate(img)


def evaluate(dets, gt, conf_thr=0.25):
    tp, conf, pc, tc, _ = tp_table(dets, gt)
    r = ap_per_class(tp, conf, pc, tc)
    ap = r[5]                                                # [nc, 10]
    keep = conf >= conf_thr
    t = tp[keep, 0]
    return dict(map50=float(ap[:, 0].mean()), map5095=float(ap.mean()), ap50_per_class=ap[:, 0].tolist(),
                ap_per_class=ap.mean(1).tolist(), n_images=len(gt), n_gt=int(len(tc)), pred=int(keep.sum()),
                tp=int(t.sum()), fp=int((~t).sum()), fppi=float((~t).sum() / len(gt)),
                precision=float(t.mean()) if keep.sum() else 0.0, recall=float(t.sum() / max(len(tc), 1)))


def froc(dets, gt, n=200):
    """recall and FPPI as the score threshold sweeps (IoU 0.5, class-aware)."""
    tp, conf, pc, tc, _ = tp_table(dets, gt)
    o = np.argsort(-conf); t = tp[o, 0]
    ctp, cfp = np.cumsum(t), np.cumsum(~t)
    return conf[o], ctp / max(len(tc), 1), cfp / len(gt)


def fppi_at_recall(dets, gt, target):
    c, rec, fp = froc(dets, gt)
    i = np.searchsorted(rec, target)
    return float(fp[i]) if i < len(rec) else float("nan")


if __name__ == "__main__":
    split, path = sys.argv[1], sys.argv[2]
    gt = load_gt(split); print(json.dumps(evaluate(load_dets(path, gt), gt), indent=1))
