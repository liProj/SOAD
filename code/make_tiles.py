"""Cut every 1920x1200 frame into overlapping native-resolution 640x640 tiles (4 x 3 grid, overlap 213 / 280 px),
together with its curvature maps, depth preview and labels, so the detector sees fruit at ~22 px instead of ~7 px.

A box is kept in a tile when at least half of its area is inside; it is clipped to the tile. For the training
split only a quarter of the fruit-free tiles are kept (seeded). Tiles of val/test are all kept (inference grid)."""
import os, sys, glob, json
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
SRC, DST = f"{B}/data/citrus", f"{B}/data/tiles"
T = 640
XS, YS = [0, 427, 853, 1280], [0, 280, 560]


def one(args):
    split, stem = args
    lab = f"{SRC}/{split}/labels/{stem}.txt"
    a = np.loadtxt(lab, ndmin=2).reshape(-1, 5) if os.path.getsize(lab) else np.zeros((0, 5))
    if (a < 0).any():
        return 0, 0, 0                                     # the loader drops these frames; so do we
    a = np.unique(a, axis=0) if len(a) else a
    im = cv2.imread(f"{SRC}/{split}/images/{stem}.jpg")
    g = {k: cv2.imread(f"{SRC}/{split}/geom/{k}/{stem}.png", cv2.IMREAD_GRAYSCALE) for k in "BND"}
    x1 = (a[:, 1] - a[:, 3] / 2) * 1920; x2 = (a[:, 1] + a[:, 3] / 2) * 1920
    y1 = (a[:, 2] - a[:, 4] / 2) * 1200; y2 = (a[:, 2] + a[:, 4] / 2) * 1200
    area = (x2 - x1) * (y2 - y1)
    rng = np.random.default_rng(abs(hash(stem)) % 2**32)
    nt = nb = ne = 0
    for yi, y0 in enumerate(YS):
        for xi, x0 in enumerate(XS):
            cx1, cy1, cx2, cy2 = np.clip(x1 - x0, 0, T), np.clip(y1 - y0, 0, T), np.clip(x2 - x0, 0, T), np.clip(y2 - y0, 0, T)
            keep = ((cx2 - cx1) * (cy2 - cy1) >= 0.5 * area) & (area > 0)
            if split == "train" and not keep.any():
                ne += 1
                if rng.random() > 0.25:
                    continue
            name = f"{stem}__{yi}{xi}"
            cv2.imwrite(f"{DST}/{split}/images/{name}.jpg", im[y0:y0 + T, x0:x0 + T], [cv2.IMWRITE_JPEG_QUALITY, 95])
            for k in "BND":
                cv2.imwrite(f"{DST}/{split}/geom/{k}/{name}.png", g[k][y0:y0 + T, x0:x0 + T])
            with open(f"{DST}/{split}/labels/{name}.txt", "w") as f:
                for c, u1, v1, u2, v2 in zip(a[keep, 0], cx1[keep], cy1[keep], cx2[keep], cy2[keep]):
                    f.write(f"{int(c)} {(u1 + u2) / 2 / T:.6f} {(v1 + v2) / 2 / T:.6f} {(u2 - u1) / T:.6f} {(v2 - v1) / T:.6f}\n")
            nt += 1; nb += int(keep.sum())
    return nt, nb, ne


if __name__ == "__main__":
    stats = {}
    for split in ["train", "val", "test"]:
        for k in ["images", "labels", "geom/B", "geom/N", "geom/D"]:
            os.makedirs(f"{DST}/{split}/{k}", exist_ok=True)
        stems = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{SRC}/{split}/labels/*.txt"))
        with ProcessPoolExecutor(12) as ex:
            r = np.array(list(ex.map(one, [(split, s) for s in stems], chunksize=8)))
        stats[split] = dict(frames=int((r[:, 0] > 0).sum()), tiles=int(r[:, 0].sum()), boxes=int(r[:, 1].sum()), empty_tiles_before_subsampling=int(r[:, 2].sum()))
        print(split, stats[split], flush=True)
    json.dump(stats, open(f"{B}/results/tiles.json", "w"), indent=1)
