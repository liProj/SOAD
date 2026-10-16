"""Inference-time variants of the val/test tiles.

flip : horizontally mirrored tiles (image + curvature maps + labels) for flip test-time augmentation.
gamma: label-free illumination normalisation -- one gamma per *frame* that moves the frame's median brightness (HSV V)
       to the median of the training frames; curvature maps and labels are unchanged (symlinked)."""
import os, sys, glob, json
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
SRC, TL = f"{B}/data/citrus", f"{B}/data/tiles"


def med_v(args):
    split, stem = args
    im = cv2.imread(f"{SRC}/{split}/images/{stem}.jpg")
    return float(np.median(im[::4, ::4].max(2)) / 255)


def flip_one(args):
    split, name = args
    D = f"{B}/data/tiles_flip/{split}"
    cv2.imwrite(f"{D}/images/{name}.jpg", cv2.imread(f"{TL}/{split}/images/{name}.jpg")[:, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    for k in "BND":
        cv2.imwrite(f"{D}/geom/{k}/{name}.png", cv2.imread(f"{TL}/{split}/geom/{k}/{name}.png", cv2.IMREAD_GRAYSCALE)[:, ::-1])
    with open(f"{TL}/{split}/labels/{name}.txt") as f, open(f"{D}/labels/{name}.txt", "w") as o:
        for l in f:
            c, x, y, w, h = l.split(); o.write(f"{c} {1 - float(x):.6f} {y} {w} {h}\n")


def gamma_one(args):
    split, name, g = args
    D = f"{B}/data/tiles_gamma/{split}"
    lut = (np.power(np.arange(256) / 255.0, g) * 255).round().astype(np.uint8)
    cv2.imwrite(f"{D}/images/{name}.jpg", lut[cv2.imread(f"{TL}/{split}/images/{name}.jpg")], [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    stems = {s: sorted({os.path.basename(f).rsplit("__", 1)[0] for f in glob.glob(f"{TL}/{s}/images/*.jpg")}) for s in ["train", "val", "test"]}
    with ProcessPoolExecutor(14) as ex:
        mv = {s: dict(zip(stems[s], ex.map(med_v, [(s, k) for k in stems[s]], chunksize=16))) for s in stems}
        target = float(np.median(list(mv["train"].values())))
        stat = {"target_median_v": target, **{f"{s}_median_v": float(np.median(list(mv[s].values()))) for s in mv}}
        for split in ["val", "test"]:
            names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{TL}/{split}/images/*.jpg"))
            for k in ["images", "labels", "geom/B", "geom/N", "geom/D"]:
                os.makedirs(f"{B}/data/tiles_flip/{split}/{k}", exist_ok=True)
            list(ex.map(flip_one, [(split, n) for n in names], chunksize=32))
            os.makedirs(f"{B}/data/tiles_gamma/{split}/images", exist_ok=True)
            for k in ["labels", "geom"]:
                if not os.path.exists(f"{B}/data/tiles_gamma/{split}/{k}"):
                    os.symlink(f"{TL}/{split}/{k}", f"{B}/data/tiles_gamma/{split}/{k}")
            gm = {k: float(np.clip(np.log(target) / np.log(np.clip(v, 0.02, 0.98)), 0.4, 2.5)) for k, v in mv[split].items()}
            stat[f"{split}_gamma_median"] = float(np.median(list(gm.values())))
            list(ex.map(gamma_one, [(split, n, gm[n.rsplit("__", 1)[0]]) for n in names], chunksize=32))
            print(split, len(names), flush=True)
    json.dump(stat, open(f"{B}/results/tile_variants.json", "w"), indent=1); print(stat)
