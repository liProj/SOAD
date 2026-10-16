"""Physical-evidence re-scoring (PER): a small gradient-boosted calibrator that turns each detection's score into
P(true fruit) x expected IoU, using evidence the detector does not have at 640 px: metric size from depth, depth
contrast to the surround, curvature-map response, fruit colour, and the context of the other detections in the frame.

It is fitted on the detections of the *validation* frames only (image-grouped 5-fold, out-of-fold predictions for
model choice) and applied unchanged to the held-out field. The detector is not touched."""
import os, sys, pickle, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import load_gt, evaluate, match, iou_matrix
from boxfeat import box_features, NAMES, GROUPS
CTX = ["score", "logit", "cls", "rank_in_frame", "n_conf25", "max_iou_higher", "n_overlap", "score_rel_max", "score_gap_next"]
MIN_S = 0.02


def _one(args):
    split, stem, d, g = args
    k = d[:, 4] >= MIN_S
    dd = d[k]
    if not len(dd):
        return stem, np.zeros((0, len(CTX) + len(NAMES)), np.float32), np.zeros(0, bool), np.zeros(0, np.float32), np.where(k)[0]
    s = dd[:, 4]; o = np.argsort(-s); rank = np.empty(len(s)); rank[o] = np.arange(len(s))
    iou = iou_matrix(dd[:, :4], dd[:, :4]); np.fill_diagonal(iou, 0)
    higher = s[None, :] > s[:, None]
    ss = np.sort(s)[::-1]; nxt = np.r_[ss[1:], 0.0]; gap = np.empty(len(s)); gap[o] = ss - nxt
    ctx = np.stack([s, np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4))), dd[:, 5], rank, np.full(len(s), (d[:, 4] >= 0.25).sum()),
                    (iou * higher).max(1), (iou > 0.3).sum(1), s / s.max(), gap], 1)
    f = box_features(split, stem, dd[:, :4])
    tp = match(dd, g)[:, 0]
    best = (iou_matrix(g[:, 1:5], dd[:, :4]) * (g[:, 0][:, None] == dd[:, 5][None])).max(0) if len(g) else np.zeros(len(dd))
    return stem, np.hstack([ctx, f]).astype(np.float32), tp, best.astype(np.float32), np.where(k)[0]


def featurize(split, dets, gt, cache=None):
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    with ProcessPoolExecutor(14) as ex:
        r = list(ex.map(_one, [(split, k, dets[k], gt[k]) for k in gt], chunksize=4))
    out = {s: dict(X=x, tp=t, iou=i, idx=ix) for s, x, t, i, ix in r}
    if cache:
        pickle.dump(out, open(cache, "wb"))
    return out


def columns(groups):
    """indices of the feature columns for a list of group names ('ctx' + keys of boxfeat.GROUPS)."""
    names = CTX + NAMES; use = []
    for g in groups:
        use += CTX if g == "ctx" else (["score", "logit", "cls"] if g == "score" else GROUPS[g])
    return [names.index(n) for n in dict.fromkeys(use)]


def fit(F, stems, cols, seed=0):
    import lightgbm as lgb
    X = np.concatenate([F[s]["X"][:, cols] for s in stems]); y = np.concatenate([F[s]["tp"] for s in stems]); q = np.concatenate([F[s]["iou"] for s in stems])
    kw = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=5.0, random_state=seed, verbose=-1, n_jobs=8)
    return lgb.LGBMClassifier(**kw).fit(X, y), lgb.LGBMRegressor(**kw).fit(X, np.where(y, q, 0.0))


def apply(models, F, dets, cols, alpha=0.5):
    clf, reg = models; out = {}
    for s, d in dets.items():
        d = d.copy(); f = F[s]
        if len(f["idx"]):
            p = clf.predict_proba(f["X"][:, cols])[:, 1]; e = np.clip(reg.predict(f["X"][:, cols]), 1e-3, 1)
            new = p ** (1 - alpha) * e ** alpha if alpha > 0 else p
            rest = np.setdiff1d(np.arange(len(d)), f["idx"])
            d[rest, 4] *= 1e-3                                # detections below the 0.02 floor stay at the bottom, in order
            d[f["idx"], 4] = np.clip(new, 2e-5, 1)
        out[s] = d[np.argsort(-d[:, 4])]
    return out


def oof(F, dets, gt, cols, alpha=0.5, k=5, seed=0):
    stems = np.array(sorted(gt)); rng = np.random.default_rng(seed); fold = rng.permutation(len(stems)) % k; out = {}
    for j in range(k):
        m = fit(F, stems[fold != j], cols, seed)
        out.update(apply(m, {s: F[s] for s in stems[fold == j]}, {s: dets[s] for s in stems[fold == j]}, cols, alpha))
    return out
