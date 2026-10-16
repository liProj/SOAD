"""Per-box physical evidence from the metric depth map, the two curvature maps and the colour image.

Depth is uint16 millimetres (0 = no stereo match, 19999 = range cap). The quantities are chosen so that they do
not depend on which orchard the image came from: metric size (pixel extent x depth), front-to-surround depth
contrast (a fruit sits in front of what is behind it), bulge (centre closer than rim), curvature-map response,
and the share of fruit-coloured pixels."""
import functools, os
import numpy as np, cv2
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
D = f"{B}/data/citrus"
NAMES = ["w_px", "h_px", "log_area", "aspect", "cx", "cy",
         "z_c", "z_valid", "z_iqr", "z_ring", "z_contrast", "z_ring_valid", "bulge",
         "phys_w", "phys_h", "log_phys_area",
         "B_in", "B_ring", "B_diff", "B_std", "N_in", "N_ring", "N_diff",
         "hue_sin", "hue_cos", "sat", "val", "orange_frac", "orange_ring", "orange_diff"]
GROUPS = {"box": NAMES[0:6], "depth": NAMES[6:13], "size": NAMES[13:16], "curv": NAMES[16:23], "colour": NAMES[23:30]}


@functools.lru_cache(maxsize=4)
def load_maps(split, stem):
    im = cv2.imread(next(p for p in (f"{D}/{split}/images/{stem}.jpg", f"{D}/{split}/images/{stem}.png") if os.path.exists(p)))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    z = np.load(f"{D}/{split}/depth/{stem}.npy").astype(np.float32)
    z[(z <= 0) | (z >= 19999)] = np.nan
    b = cv2.imread(f"{D}/{split}/geom/B/{stem}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    n = cv2.imread(f"{D}/{split}/geom/N/{stem}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    orange = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 28) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 70)).astype(np.float32)
    return hsv, z, b, n, orange


def _med(a):
    a = a[np.isfinite(a)]
    return np.median(a) if a.size else np.nan


def box_features(split, stem, boxes):
    """boxes [n, 4] xyxy pixels -> [n, len(NAMES)] float32 (NaN where depth is missing)."""
    hsv, z, bm, nm, orange = load_maps(split, stem)
    Hh, Ww = z.shape
    out = np.full((len(boxes), len(NAMES)), np.nan, np.float32)
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, float)):
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b_, c, d = [int(round(v)) for v in (max(x1, 0), max(y1, 0), min(x2, Ww), min(y2, Hh))]
        if c - a < 2 or d - b_ < 2:
            continue
        ia, ib, ic, id_ = [int(round(v)) for v in (max(cx - w / 4, 0), max(cy - h / 4, 0), min(cx + w / 4, Ww), min(cy + h / 4, Hh))]
        ic, id_ = max(ic, ia + 1), max(id_, ib + 1)
        ra, rb, rc, rd = [int(round(v)) for v in (max(x1 - w / 2, 0), max(y1 - h / 2, 0), min(x2 + w / 2, Ww), min(y2 + h / 2, Hh))]
        ring = np.ones((rd - rb, rc - ra), bool); ring[b_ - rb:d - rb, a - ra:c - ra] = False
        zin, zbox, zring = z[ib:id_, ia:ic], z[b_:d, a:c], z[rb:rd, ra:rc][ring]
        zc, zr = _med(zin), _med(zring)
        rim = np.concatenate([zbox[0], zbox[-1], zbox[:, 0], zbox[:, -1]])
        fin = zbox[np.isfinite(zbox)]
        f = [w, h, np.log(w * h), w / h, cx / Ww, cy / Hh,
             zc, np.isfinite(zbox).mean(), (np.percentile(fin, 75) - np.percentile(fin, 25)) if fin.size > 3 else np.nan,
             zr, zr - zc, np.isfinite(zring).mean() if zring.size else 0.0, _med(rim) - zc,
             w * zc / 1000, h * zc / 1000, np.log(max(w * h * zc * zc / 1e6, 1e-6)) if np.isfinite(zc) else np.nan]
        for m in (bm, nm):
            mi, mr = m[b_:d, a:c].mean(), m[rb:rd, ra:rc][ring].mean() if ring.any() else np.nan
            f += [mi, mr, mi - mr] + ([m[b_:d, a:c].std()] if m is bm else [])
        hh = hsv[ib:id_, ia:ic].reshape(-1, 3).astype(np.float32)
        ang = hh[:, 0] * (np.pi / 90)
        oi, orr = orange[b_:d, a:c].mean(), orange[rb:rd, ra:rc][ring].mean() if ring.any() else np.nan
        f += [np.sin(ang).mean(), np.cos(ang).mean(), hh[:, 1].mean() / 255, hh[:, 2].mean() / 255, oi, orr, oi - orr]
        out[i] = f
    return out


if __name__ == "__main__":      # separability of annotated fruit vs. random same-size boxes, per split
    import sys, time
    sys.path.insert(0, f"{B}/code")
    from evalbox import load_gt, iou_matrix
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); res = {}
    for split in ["val", "test"]:
        gt = load_gt(split); P, N = [], []; t0 = time.time()
        for stem in list(gt)[::4]:
            g = gt[stem][:, 1:5]
            if not len(g): continue
            w, h = g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
            x = rng.uniform(0, 1920 - w); y = rng.uniform(0, 1200 - h)
            r = np.stack([x, y, x + w, y + h], 1)
            r = r[iou_matrix(g, r).max(0) < 0.1]
            P.append(box_features(split, stem, g)); N.append(box_features(split, stem, r))
        P, N = np.concatenate(P), np.concatenate(N)
        print(split, len(P), len(N), f"{time.time() - t0:.0f}s")
        for j, nme in enumerate(NAMES):
            v = np.concatenate([P[:, j], N[:, j]]); y = np.r_[np.ones(len(P)), np.zeros(len(N))]; ok = np.isfinite(v)
            res.setdefault(nme, []).append((roc_auc_score(y[ok], v[ok]), np.nanmedian(P[:, j]), np.nanmedian(N[:, j])))
    for k, v in res.items():
        print(f"{k:14s} AUC val {v[0][0]:.3f} test {v[1][0]:.3f} | median fruit/bg val {v[0][1]:.3g}/{v[0][2]:.3g} test {v[1][1]:.3g}/{v[1][2]:.3g}")
