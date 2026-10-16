"""Train + evaluate one detector with the authors' modified ultralytics fork (third_party/Geosphere-DETR).

Evaluation follows the authors' validator: mAP from the full ranked list (conf 0.001), FPPI = false positives
per image at conf >= 0.25 and IoU 0.5 on the held-out test split.
"""
import argparse, json, os, sys, time
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/third_party/Geosphere-DETR-run")
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)          # yaml or .pt
ap.add_argument("--pretrained", default="")
ap.add_argument("--data", required=True)
ap.add_argument("--kind", default="rtdetr", choices=["rtdetr", "yolo"])
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--extra", default="{}")           # json dict of extra train args
ap.add_argument("--eval_only", action="store_true")
a = ap.parse_args()
from ultralytics import RTDETR, YOLO
M = RTDETR if a.kind == "rtdetr" else YOLO
run = f"{B}/runs/{a.name}"
best = f"{run}/weights/best.pt"
t0 = time.time()
if not a.eval_only and not os.path.exists(best):
    m = M(a.model)
    kw = dict(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, seed=a.seed, device=0, workers=a.workers,
              project=f"{B}/runs", name=a.name, exist_ok=True, plots=False, verbose=False, deterministic=False)
    if a.pretrained: kw["pretrained"] = a.pretrained
    kw.update(json.loads(a.extra))
    m.train(**kw)
train_s = time.time() - t0
import numpy as np
from ultralytics.utils.metrics import DetMetrics
# The fork's own FPPI code reads validator.stats, which this ultralytics version never fills (it is None),
# so the per-prediction statistics are captured here, just before the metrics object clears them.
_proc = DetMetrics.process
def _process(self, *args, **kw):
    self.last_stats = {k: np.concatenate(v, 0) for k, v in self.stats.items() if len(v)}
    return _proc(self, *args, **kw)
DetMetrics.process = _process
res = dict(name=a.name, seed=a.seed, imgsz=a.imgsz, epochs=a.epochs, train_seconds=round(train_s))
m = M(best)
res["params_M"] = round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2)
os.makedirs(f"{B}/results/det", exist_ok=True)
for split in ["val", "test"]:
    v = m.val(data=a.data, split=split, imgsz=a.imgsz, batch=a.batch, conf=0.001, device=0, plots=False, verbose=False, save_json=True,
              project=f"{run}/eval", name=split, exist_ok=True)
    st = v.last_stats
    # the loader drops every image whose label file holds a negative value (7 train, 1 test), as it did for the authors
    n_img = sum(not any(float(x) < 0 for x in open(os.path.join(f"{B}/data/citrus/{split}/labels", f)).read().split())
                for f in os.listdir(f"{B}/data/citrus/{split}/labels"))
    keep = st["conf"] >= 0.25
    tp = st["tp"][keep, 0]
    n_gt = len(st["target_cls"])
    r = dict(map50=float(v.box.map50), map5095=float(v.box.map), ap50_per_class=[float(x) for x in v.box.ap50],
             ap_per_class=[float(x) for x in v.box.ap], n_images=n_img, n_gt=int(n_gt), pred_at_025=int(keep.sum()),
             tp_at_025=int(tp.sum()), fp_at_025=int((~tp).sum()), fppi=float((~tp).sum() / n_img),
             precision_at_025=float(tp.mean()) if keep.sum() else 0.0, recall_at_025=float(tp.sum() / max(n_gt, 1)),
             speed_ms=float(v.speed.get("inference", 0)))
    res[split] = r
    np.savez_compressed(f"{B}/results/det/{a.name}_{split}_stats.npz", **st)
json.dump(res, open(f"{B}/results/det/{a.name}.json", "w"), indent=1)
print(json.dumps(res))
