"""Collect every finished run into results/summary.json and results/tables/main.csv.

All rows are evaluated with the same offline evaluator on full 1920x1200 frames of the same split (499 test frames,
16,199 annotated fruit), so a tiled model and a full-frame model are directly comparable. A run without saved
detections is skipped and listed under `missing`."""
import os, sys, json
import numpy as np, pandas as pd
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from evalbox import load_gt, evaluate, fppi_at_recall
from analysis import dets, per_frame_counts
# tag, label, family, views per frame (640-px network passes)
TILED = [("T_rgb_s0", "Tiled RT-DETR-L (RGB)"), ("T_rgb20_s0", "Tiled RT-DETR-L (RGB), 20 epochs"), ("T_E4_aligned_s0", "Tiled GeoSphere-DETR, aligned geometry")]
ROWS = [("Y11m_s0", "YOLO11m (RGB)", "full frame, public code", 1), ("E1_rtdetr_s0", "RT-DETR-L (RGB)", "full frame, public code", 1),
        ("E3_gated_s0", "Gated geometry fusion", "full frame, public code", 1), ("E4_rescore_s0", "GeoSphere-DETR (full model)", "full frame, public code", 1)]
for _b, _l in TILED:
    if _b == "T_E4_aligned_s0":
        ROWS.append(("T_E4_unaligned_s0_full", "Tiled GeoSphere-DETR, public loader", "tiled", 12))
    ROWS += [(f"{_b}_full", _l, "tiled", 12), (f"{_b}_full_per", "  + re-scoring", "tiled", 12), (f"{_b}_wbf", "  + flip fusion", "tiled", 24), (f"{_b}_wbf_per", "  + flip fusion + re-scoring", "tiled", 24)]
ROWS.append(("T960_E4_aligned_s0_full", "Tiled GeoSphere-DETR, 960-px fine-tune", "tiled", 27))
PRINTED = [("YOLOv11m", 0.715, None, 8.735), ("RT-DETR", 0.786, 0.562, 7.24), ("E2 concat fusion", None, 0.605, 6.7), ("E3 gated fusion", None, 0.623, 4.6),
           ("GeoSphere-DETR", 0.863, 0.642, 3.29)]


def main():
    gt = {s: load_gt(s) for s in ("val", "test")}; out, missing = {}, []
    ref_recall = None
    for tag, lab, fam, views in ROWS:
        if not os.path.exists(f"{B}/results/dets/{tag}_test.pkl"):
            missing.append(tag); continue
        r = dict(label=lab, family=fam, views=views)
        for s in ("val", "test"):
            if not os.path.exists(f"{B}/results/dets/{tag}_{s}.pkl"):
                continue
            d = dets(tag, s); e = evaluate(d, gt[s]); c = per_frame_counts(d, gt[s])
            e["count_mae"] = float(np.abs(c[:, 1] - c[:, 0]).mean()); e["count_bias"] = float((c[:, 1] - c[:, 0]).mean())
            for t in (0.7, 0.8):
                e[f"fppi_at_recall_{int(t * 100)}"] = fppi_at_recall(d, gt[s], t)
            r[s] = e
        out[tag] = r
    # headline configuration: the tiled model whose flip-fused detections have the best validation mAP50:95 (the held-out field plays no part)
    cand = {b: out[f"{b}_wbf"]["val"]["map5095"] for b, _ in TILED if f"{b}_wbf" in out and "val" in out[f"{b}_wbf"]}
    head = max(cand, key=cand.get)
    json.dump(dict(runs=out, missing=missing, printed=PRINTED, headline=head, headline_val=cand), open(f"{B}/results/summary.json", "w"), indent=1)
    os.makedirs(f"{B}/results/tables", exist_ok=True)
    T = pd.DataFrame([dict(run=t, model=r["label"], family=r["family"], passes_per_frame=r["views"], val_mAP50=r.get("val", {}).get("map50"), val_mAP5095=r.get("val", {}).get("map5095"),
                           test_mAP50=r["test"]["map50"], test_mAP5095=r["test"]["map5095"], AP50_tree=r["test"]["ap50_per_class"][0], AP50_ground=r["test"]["ap50_per_class"][1],
                           FPPI_conf025=r["test"]["fppi"], recall_conf025=r["test"]["recall"], precision_conf025=r["test"]["precision"],
                           FPPI_at_recall70=r["test"]["fppi_at_recall_70"], FPPI_at_recall80=r["test"]["fppi_at_recall_80"], count_MAE=r["test"]["count_mae"], count_bias=r["test"]["count_bias"])
                      for t, r in out.items()])
    T.to_csv(f"{B}/results/tables/main.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T.drop(columns=["run", "family"]).round(3).to_string(index=False)); print("missing:", missing); print("headline (by validation):", head, {k: round(v, 4) for k, v in cand.items()})


if __name__ == "__main__":
    main()
