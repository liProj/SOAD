"""Does the published fork keep the geometry map aligned with the image under training augmentation?

A probe dataset is built in which the "geometry" file of every image is simply the image's own grey version.
If the pipeline transformed geometry together with the image, the two would stay (almost) perfectly correlated
after augmentation; a low correlation means the model is fed geometry from the wrong place."""
import os, sys, glob, json, shutil
import numpy as np, cv2, torch
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/third_party/Geosphere-DETR-run")
S = sys.argv[1]
for k in ["images", "labels", "geom/B", "geom/N"]:
    os.makedirs(f"{S}/train/{k}", exist_ok=True)
for f in sorted(glob.glob(f"{B}/data/citrus/val/images/*.jpg"))[:48]:
    st = os.path.basename(f)[:-4]
    im = cv2.imread(f); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    shutil.copy(f, f"{S}/train/images/{st}.jpg"); shutil.copy(f"{B}/data/citrus/val/labels/{st}.txt", f"{S}/train/labels/{st}.txt")
    cv2.imwrite(f"{S}/train/geom/B/{st}.png", g); cv2.imwrite(f"{S}/train/geom/N/{st}.png", g)
open(f"{S}/d.yaml", "w").write(f"path: {S}\ntrain: train/images\nval: train/images\nchannels: 3\ngeom_mode: BN\nnc: 2\nnames: [a, b]\n")
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.data import build_dataloader

open(f"{S}/d_align.yaml", "w").write(open(f"{S}/d.yaml").read() + "geom_align: true\n")

def corr(ds_kw, tag, n=40, yaml="d.yaml"):
    cfg = get_cfg(overrides=dict(data=f"{S}/{yaml}", imgsz=640, task="detect", **ds_kw.pop("hyp", {})))
    data = check_det_dataset(f"{S}/{yaml}")
    ds = RTDETRDataset(img_path=f"{S}/train/images", imgsz=640, batch_size=8, augment=ds_kw["augment"], hyp=cfg, rect=False,
                       cache=None, prefix="", data=data)
    cs, absent = [], 0
    for i in range(n):
        b = ds[i % len(ds)]
        if "geom" not in b:
            absent += 1; continue
        img = b["img"].float().mean(0).numpy()               # grey of the augmented image
        g = b["geom"]; g = g if isinstance(g, np.ndarray) else g.numpy()
        g = g[..., 0] if g.shape[-1] in (1, 2) else g[0]
        if g.shape != img.shape:
            g = cv2.resize(g.astype(np.float32), img.shape[::-1])
        cs.append(float(np.corrcoef(img.ravel(), g.astype(np.float32).ravel())[0, 1]))
    r = dict(setting=tag, samples=n, geom_absent=absent, mean_corr=float(np.mean(cs)) if cs else None,
             min_corr=float(np.min(cs)) if cs else None, frac_above_0p8=float(np.mean(np.array(cs) > 0.8)) if cs else None)
    print(r); return r

out = [corr(dict(augment=False), "no augmentation (validation pipeline)"),
       corr(dict(augment=True), "default training augmentation"),
       corr(dict(augment=True, hyp=dict(mosaic=0.0)), "training augmentation, mosaic off"),
       corr(dict(augment=True, hyp=dict(mosaic=0.0, fliplr=0.0, scale=0.0, translate=0.0)), "mosaic, flip, scale, translate all off")]
out += [corr(dict(augment=False), "ALIGNED loader: no augmentation", yaml="d_align.yaml"),
        corr(dict(augment=True), "ALIGNED loader: default training augmentation (mosaic on)", yaml="d_align.yaml"),
        corr(dict(augment=True, hyp=dict(mosaic=0.0)), "ALIGNED loader: mosaic off", yaml="d_align.yaml")]
json.dump(out, open(f"{B}/results/geom_alignment.json", "w"), indent=1)
