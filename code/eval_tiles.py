"""Run a tile-trained detector over the tile grid of val/test, stitch to full frames, evaluate, save detections."""
import argparse, json, os, sys, pickle
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/third_party/Geosphere-DETR-run"); sys.path.insert(0, f"{B}/code")
ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--data", default=f"{B}/configs/tiles_geomBN_aligned.yaml")
ap.add_argument("--kind", default="rtdetr"); ap.add_argument("--splits", default="val,test"); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--flip", action="store_true"); ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--miss", type=float, default=0.0); ap.add_argument("--noise", type=float, default=0.0)
a = ap.parse_args()
from ultralytics import RTDETR, YOLO
from evalbox import load_gt, evaluate
from stitch import stitch
m = (RTDETR if a.kind == "rtdetr" else YOLO)(a.weights)
m.model.geom_missing_ratio = a.miss; m.model.geom_noise_sigma = a.noise   # test-time geometry corruption (0 = clean)
res = {"tag": a.tag, "weights": a.weights, "miss": a.miss, "noise": a.noise, "imgsz": a.imgsz}
os.makedirs(f"{B}/results/dets", exist_ok=True)
for split in a.splits.split(","):
    out = f"{B}/runs/_tile_eval/{a.tag}_{split}"
    v = m.val(data=a.data, split=split, imgsz=a.imgsz, batch=a.batch, conf=0.001, device=0, plots=False, verbose=False, save_json=True,
              project=f"{B}/runs/_tile_eval", name=f"{a.tag}_{split}", exist_ok=True)
    gt = load_gt(split); d = stitch(f"{out}/predictions.json", gt, flip=a.flip)
    pickle.dump(d, open(f"{B}/results/dets/{a.tag}_{split}.pkl", "wb"))
    res[split] = evaluate(d, gt); res[split]["tile_map50"] = float(v.box.map50); res[split]["tile_ms"] = float(v.speed.get("inference", 0))
    print(split, json.dumps(res[split]), flush=True)
json.dump(res, open(f"{B}/results/det/{a.tag}.json", "w"), indent=1)
