"""Result figures. Every panel is drawn from saved detections / result files; a run that is not finished is skipped."""
import os, sys, json, pickle, glob
import numpy as np, cv2
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from figstyle import plt, CAT, INK2, GRID, clean, save
from evalbox import load_gt, evaluate, froc, match
from analysis import dets, recall_by, side, SIZE_BINS, per_frame_counts
F = f"{B}/figures"
# tag -> (label, colour, line style); fixed across all figures
RUNS = {"Y11m_s0": ("YOLO11m (RGB, 640)", CAT[6], ":"), "E1_rtdetr_s0": ("RT-DETR-L (RGB, 640)", CAT[1], "-"),
        "E3_gated_s0": ("Gated fusion, public code", CAT[3], "--"), "E4_rescore_s0": ("GeoSphere-DETR, public code", CAT[7], "-"),
        "T_rgb_s0_full": ("Tiled RT-DETR-L (RGB)", CAT[2], "--"), "T_E4_unaligned_s0_full": ("Tiled, public loader", CAT[4], "--"),
        "T_rgb_s0_wbf_per": ("Tiled RGB + flip fusion + re-scoring", CAT[2], "-"),
        "T_rgb20_s0_full": ("Tiled RGB, 20 epochs", CAT[8 % len(CAT)], "--"), "T_rgb20_s0_wbf_per": ("Tiled RGB, 20 epochs + flip fusion + re-scoring", CAT[8 % len(CAT)], "-"),
        "T_E4_aligned_s0_full": ("Tiled GeoSphere-DETR, aligned geometry", CAT[0], "--"), "T_E4_aligned_s0_wbf_per": ("Tiled GeoSphere-DETR + flip fusion + re-scoring", CAT[0], "-")}
_sp = f"{B}/results/summary.json"
HEAD = json.load(open(_sp)).get("headline", "T_rgb_s0") if os.path.exists(_sp) else "T_rgb_s0"   # tiled configuration selected on validation
PAPER = {"YOLOv11m": (0.715, None, 8.735), "RT-DETR": (0.786, 0.562, 7.24), "GeoSphere-DETR": (0.863, 0.642, 3.29)}


def have(tag, split="test"):
    return os.path.exists(f"{B}/results/dets/{tag}_{split}.pkl")


def fig_froc():
    gt = load_gt("test"); fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for t, (lab, c, ls) in RUNS.items():
        if not have(t): continue
        d = dets(t, "test"); conf, rec, fp = froc(d, gt)
        ax[0].plot(fp, rec, color=c, ls=ls, lw=1.8, label=lab)
        e = evaluate(d, gt); ax[0].plot(e["fppi"], e["recall"], "o", color=c, ms=5, mec="white", mew=0.8)
        tp = np.cumsum(np.concatenate([match(d[k], gt[k])[:, 0] for k in gt])[np.argsort(-np.concatenate([d[k][:, 4] for k in gt]))])
        n = np.arange(1, len(tp) + 1); ax[1].plot(tp / e["n_gt"], tp / n, color=c, ls=ls, lw=1.8, label=lab)
    for name, (m50, m, fppi) in PAPER.items():
        ax[0].axvline(fppi, color=INK2, lw=0.7, ls=":"); ax[0].text(fppi, 0.27 - 0.12 * list(PAPER).index(name), f"printed FPPI \n{name} ", fontsize=6, color=INK2, ha="right")
    ax[0].set_xscale("log"); ax[0].set_xlim(0.3, 120); ax[0].set_ylim(0, 1); ax[0].set_xlabel("false positives per image (log)"); ax[0].set_ylabel("recall (IoU 0.5, class-aware)")
    ax[0].set_title("FROC on the held-out field (dot: confidence 0.25)"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Precision-recall, both classes pooled")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1.01); ax[1].legend(fontsize=7, loc="lower left")
    for a in ax: clean(a, "both")
    fig.tight_layout(); save(fig, f"{F}/fig_froc_pr")


def fig_size_count():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6)); gt = load_gt("test")
    use = [t for t in ("E1_rtdetr_s0", "E4_rescore_s0", "T_rgb_s0_full", "T_E4_aligned_s0_full", f"{HEAD}_wbf_per") if have(t)]
    x = np.arange(len(SIZE_BINS) - 1); w = 0.8 / len(use)
    for j, t in enumerate(use):
        r, n = recall_by(dets(t, "test"), gt, side, SIZE_BINS)
        ax[0].bar(x + (j - len(use) / 2 + 0.5) * w, r, w, color=RUNS[t][1], alpha=1 if RUNS[t][2] == "-" else 0.55, label=RUNS[t][0])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{a}-{b}" if b < 400 else f">{a}" for a, b in zip(SIZE_BINS[:-1], SIZE_BINS[1:])]); ax[0].set_xlabel("fruit side in the full frame (px)")
    ax[0].set_ylabel("recall at confidence 0.25"); ax[0].set_title("Recall by fruit size")
    ax[0].set_ylim(0, 1.45)
    ax[0].set_yticks(np.arange(0, 1.01, 0.2))
    ax[0].legend(fontsize=6.5, loc="upper left", borderaxespad=0.4)
    clean(ax[0])
    for a, t in zip(ax[1:], ("E1_rtdetr_s0", f"{HEAD}_wbf_per")):
        if not have(t): continue
        c = per_frame_counts(dets(t, "test"), gt); a.scatter(c[:, 0], c[:, 1], s=6, color=RUNS[t][1], alpha=0.5, lw=0)
        m = max(c[:, :2].max(), 1); a.plot([0, m], [0, m], color=INK2, lw=0.8, ls="--")
        a.set_xlabel("annotated fruit in the frame"); a.set_ylabel("detections at confidence 0.25")
        a.set_title(f"{RUNS[t][0] if t.startswith('E1') else 'Headline pipeline (tiles + flip fusion + re-scoring)'}\ncount MAE {np.abs(c[:, 1] - c[:, 0]).mean():.1f}, bias {np.mean(c[:, 1] - c[:, 0]):+.1f} per frame", fontsize=9); clean(a, "both")
    fig.tight_layout(); save(fig, f"{F}/fig_size_count")


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    R = {"E1_rtdetr_s0": "E1_rtdetr_s0", "E4_rescore_s0": "E4_rescore_s0", "E3_gated_s0": "E3_gated_s0", "_stopped_E2_concat_s0_ep13": None, "Y11m_s0": "Y11m_s0"}
    for run, tag in R.items():
        p = f"{B}/runs/{run}/results.csv"
        if not os.path.exists(p): continue
        import pandas as pd
        d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]
        lab, c, ls = RUNS[tag] if tag else ("Concat fusion, public code (stopped)", CAT[4], "-")
        ax[0].plot(d["epoch"], d["metrics/mAP50(B)"], color=c, ls=ls, lw=1.6, label=lab)
    ax[0].axvline(90, color=INK2, lw=0.7, ls=":"); ax[0].text(90, 0.03, " mosaic off", fontsize=7, color=INK2)
    ax[0].set_xlabel("epoch (full frames resized to 640 px)"); ax[0].set_ylabel("validation mAP50"); ax[0].set_title("Full-frame training with the public code"); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1)
    for run, tag in (("T_rgb_s0", "T_rgb_s0_full"), ("T_rgb20_s0", "T_rgb20_s0_full"), ("T_E4_unaligned_s0", "T_E4_unaligned_s0_full"), ("T_E4_aligned_s0", "T_E4_aligned_s0_full")):
        p = f"{B}/runs/{run}/results.csv"
        if not os.path.exists(p): continue
        import pandas as pd
        d = pd.read_csv(p); d.columns = [c.strip() for c in d.columns]
        ax[1].plot(d["epoch"], d["metrics/mAP50-95(B)"], color=RUNS[tag][1], ls="-", marker="o", ms=3, lw=1.6, label=RUNS[tag][0])
    ax[1].set_xlabel("epoch (native-resolution tiles; 10 epochs = the iterations of 100 full-frame epochs)"); ax[1].set_ylabel("validation mAP50:95 on tiles"); ax[1].set_title("Tiled training"); ax[1].legend(fontsize=7)
    for a in ax: clean(a, "both")
    fig.tight_layout(); save(fig, f"{F}/fig_training_curves")


def fig_per():
    p = f"{B}/results/rescore_{HEAD}_wbf.json"
    if not os.path.exists(p): return
    r = json.load(open(p))
    if "ablation" not in r: return
    ab = r["ablation"]; keys = list(ab); fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for a, (m, lab, ref) in zip(ax[:2], (("map5095", "test mAP50:95", r["base"]["test"]["map5095"]), ("fppi_at_base_recall", "test FPPI at the detector's recall", r["base"]["test"]["fppi_at_base_recall"]))):
        v = [ab[k]["test"][m] for k in keys]; y = np.arange(len(keys))[::-1]
        a.barh(y, v, color=[CAT[5] if k == "all" else (CAT[0] if k.startswith("ctx") or k == "score only" else CAT[3]) for k in keys], height=0.7)
        a.axvline(ref, color=INK2, ls="--", lw=1); a.text(ref, len(keys) - 0.4, " detector score", fontsize=7, color=INK2, va="bottom")
        a.set_yticks(y); a.set_yticklabels([k.replace("ctx", "context").replace("all-", "all - ") for k in keys], fontsize=7.5); a.set_xlabel(lab)
        lo, hi = min(v + [ref]), max(v + [ref]); a.set_xlim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.25); clean(a, "x")
    imp = r["importance"]; g = np.array(imp["clf"], float); o = np.argsort(g)[-12:]
    ax[2].barh(range(len(o)), g[o] / g.sum(), color=CAT[0]); ax[2].set_yticks(range(len(o))); ax[2].set_yticklabels([imp["names"][i] for i in o], fontsize=7.5)
    ax[2].set_xlabel("share of split gain (true-fruit classifier)"); clean(ax[2], "x")
    ax[0].set_title("Evidence groups: accuracy"); ax[1].set_title("Evidence groups: false positives"); ax[2].set_title("Most used evidence")
    fig.tight_layout(); save(fig, f"{F}/fig_rescoring_ablation")


def fig_qual(n=3):
    gt = load_gt("test")
    if not (have("E1_rtdetr_s0") and have(f"{HEAD}_wbf_per")): return
    a, b = dets("E1_rtdetr_s0", "test"), dets(f"{HEAD}_wbf_per", "test")
    gain = {k: (len(gt[k]), (a[k][:, 4] >= 0.25).sum() - (b[k][:, 4] >= 0.25).sum()) for k in gt}
    ks = sorted([k for k in gt if 25 <= gain[k][0] <= 60], key=lambda k: -gain[k][1])[:: max(1, 40 // n)][:n]
    fig, ax = plt.subplots(n, 2, figsize=(11, 3.45 * n))
    for r, k in enumerate(ks):
        im = cv2.cvtColor(cv2.imread(f"{B}/data/citrus/test/images/{k}.jpg"), cv2.COLOR_BGR2RGB)
        ys, xs = gt[k][:, [2, 4]], gt[k][:, [1, 3]]; cx, cy = int(np.median(xs)), int(np.median(ys))
        x0, y0 = int(np.clip(cx - 480, 0, 1920 - 960)), int(np.clip(cy - 300, 0, 1200 - 600))
        for c, (d, name) in enumerate(((a, "RT-DETR-L, full frame at 640 px"), (b, "Headline pipeline: tiles + flip fusion + re-scoring"))):
            x = d[k][d[k][:, 4] >= 0.25]; tp = match(x, gt[k])[:, 0] if len(x) else np.zeros(0, bool)
            ax[r, c].imshow(im[y0:y0 + 600, x0:x0 + 960]); ax[r, c].axis("off")
            for g in gt[k]:
                ax[r, c].add_patch(plt.Rectangle((g[1] - x0, g[2] - y0), g[3] - g[1], g[4] - g[2], fill=False, ec="white", lw=0.6, ls=":"))
            for bx, t in zip(x, tp):
                ax[r, c].add_patch(plt.Rectangle((bx[0] - x0, bx[1] - y0), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="#39ff14" if t else "#ff2d2d", lw=0.9))
            ax[r, c].set_xlim(0, 960); ax[r, c].set_ylim(600, 0)
            ax[r, c].set_title(f"{name}\n{int(tp.sum())} correct (green), {int((~tp).sum())} false (red), {len(gt[k])} annotated (dotted)", fontsize=8.5)
    fig.tight_layout(); save(fig, f"{F}/fig_qualitative")


if __name__ == "__main__":
    which = sys.argv[1:] or ["froc", "size_count", "curves", "per", "qual"]
    for w in which:
        globals()[f"fig_{w}"]()
