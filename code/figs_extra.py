"""Method and summary figures: pipeline schematic, geometry-alignment evidence, ablation ladder, printed vs. measured, cost."""
import os, sys, json
import numpy as np
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{B}/code")
from figstyle import plt, CAT, INK, INK2, GRID, clean, save
F = f"{B}/figures"
S = json.load(open(f"{B}/results/summary.json")); R = S["runs"]


def fig_pipeline():
    import runpy
    runpy.run_path(os.path.join(B, 'code', 'fig_pipeline_vector.py'), run_name='__main__')


def fig_alignment():
    A = json.load(open(f"{B}/results/geom_alignment.json")); P = json.load(open(f"{B}/results/E2_probe.json"))["val"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    rows = [("validation pipeline\n(no augmentation)", A[0], CAT[1]), ("public loader\ndefault training\n(mosaic on)", A[1], CAT[1]), ("public loader\nmosaic off\n(last 10 epochs)", A[2], CAT[1]),
            ("aligned loader\nno augmentation", A[4], CAT[0]), ("aligned loader\ndefault training\n(mosaic on)", A[5], CAT[0])]
    for i, (lab, r, c) in enumerate(rows):
        if r["mean_corr"] is None:
            ax[0].text(i, 0.04, "geometry absent\nfrom every sample", ha="center", fontsize=7.5, color=CAT[7]); ax[0].bar(i, 0.012, 0.6, color=CAT[7])
        else:
            ax[0].bar(i, r["mean_corr"], 0.6, color=c); ax[0].plot([i, i], [r["min_corr"], r["mean_corr"]], color=INK2, lw=1)
            ax[0].text(i, max(r["mean_corr"], 0) + 0.03, f"{r['mean_corr']:.2f}", ha="center", fontsize=8)
    ax[0].axhline(0, color=INK2, lw=0.6); ax[0].set_xticks(range(len(rows))); ax[0].set_xticklabels([r[0] for r in rows], fontsize=7.5); ax[0].set_ylim(-0.25, 1.12)
    ax[0].set_ylabel("correlation of image and its 'geometry'\n(probe: geometry file = grey image)"); ax[0].set_title("Does the geometry map follow the image through augmentation?"); clean(ax[0])
    v = [P["with_geometry"]["map50"], P["without_geometry"]["map50"]]
    ax[1].bar([0, 1], v, 0.55, color=[CAT[7], CAT[2]]); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["geometry fed\n(as in validation)", "geometry withheld"])
    for i, x in enumerate(v): ax[1].text(i, x + 0.02, f"{x:.3f}" if x > 0.01 else f"{x:.5f}", ha="center", fontsize=8)
    ax[1].set_ylim(0, 0.85); ax[1].set_ylabel("validation mAP50"); ax[1].set_title("Concat-fusion model after 13 epochs\nof the public code (same checkpoint)", fontsize=9); clean(ax[1])
    fig.tight_layout(); save(fig, f"{F}/fig_alignment")


def fig_ladder():
    from fig_ladder import draw_ladder
    draw_ladder()


def fig_printed():
    pairs = [("YOLOv11m", "Y11m_s0"), ("RT-DETR", "E1_rtdetr_s0"), ("E3 gated fusion", "E3_gated_s0"), ("GeoSphere-DETR", "E4_rescore_s0")]
    P = {n: (a, b, c) for n, a, b, c in S["printed"]}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5)); w = 0.36
    names = [n for n, t in pairs if t in R]
    for a, (i, m, lab) in zip(ax, ((0, "map50", "mAP50"), (1, "map5095", "mAP50:95"), (2, "fppi", "FPPI at confidence 0.25"))):
        for j, (n, t) in enumerate([p for p in pairs if p[1] in R]):
            pv = P[n][i]; mv = R[t]["test"][m]
            if pv is not None:
                a.bar(j - w / 2, pv, w, color=INK2); a.text(j - w / 2, pv, f"{pv:.3g}", ha="center", va="bottom", fontsize=7)
            a.bar(j + w / 2, mv, w, color=CAT[1]); a.text(j + w / 2, mv, f"{mv:.3g}", ha="center", va="bottom", fontsize=7)
        if f"{S['headline']}_wbf_per" in R:
            mv = R[f"{S['headline']}_wbf_per"]["test"][m]; j = len(names)
            a.bar(j, mv, w, color=CAT[0]); a.text(j, mv, f"{mv:.3g}", ha="center", va="bottom", fontsize=7)
        a.set_xticks(range(len(names) + 1)); a.set_xticklabels(names + ["Ours\n(SOAD)"], fontsize=7.5); a.set_title(lab)
        if i == 2: a.set_yscale("log")
        clean(a)
    ax[0].bar(0, 0, color=INK2, label="Reported in literature")
    ax[0].bar(0, 0, color=CAT[1], label="Baselines (this study)")
    ax[0].bar(0, 0, color=CAT[0], label="Ours (SOAD)")
    ax[0].set_ylim(0, 1.25)
    ax[0].set_yticks(np.arange(0, 1.01, 0.2))
    ax[0].legend(fontsize=8, loc="upper left", borderaxespad=0.6,
                 handlelength=1.6, labelspacing=0.45)
    fig.tight_layout(); save(fig, f"{F}/fig_printed_vs_measured")


def fig_cost():
    from fig_cost import draw_cost
    draw_cost(R, f"{F}/fig_cost")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["pipeline", "alignment", "ladder", "printed", "cost"]):
        globals()[f"fig_{w}"]()
