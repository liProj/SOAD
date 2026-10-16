"""SOAD component comparison using measured held-out results only."""
import json
from pathlib import Path
import numpy as np
from figstyle import plt, clean, save

ROOT = Path(__file__).resolve().parents[1]


def draw_ladder():
    runs = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))["runs"]
    rows = [
        ("Y11m_s0", "YOLO11m (full frame)", "#eb6834"),
        ("E1_rtdetr_s0", "RT-DETR-L (full frame)", "#eb6834"),
        ("E4_rescore_s0", "GeoSphere-DETR (full frame)", "#eb6834"),
        ("T_E4_aligned_s0_full", "GeoSphere-DETR (tiles, aligned)", "#8a78b8"),
        ("T_E4_aligned_s0_wbf_per", "GeoSphere-DETR (tiles + MF + EC)", "#8a78b8"),
        ("T_rgb_s0_full", "SOAD w/o MF & EC", "#8db9e8"),
        ("T_rgb_s0_full_per", "SOAD w/o MF", "#619bd9"),
        ("T_rgb_s0_wbf", "SOAD w/o EC", "#347fc8"),
        ("T_rgb_s0_wbf_per", "Ours (SOAD)", "#12569c"),
    ]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharey=True)
    for ax, metric, title, xmax in [
        (axes[0], "map5095", "(a) Detection accuracy", .65),
        (axes[1], "fppi_at_recall_70", "(b) False positives at matched recall", 15.5),
    ]:
        vals = [runs[tag]["test"][metric] for tag, _, _ in rows]
        ax.axhspan(-.45, .45, color="#12569c", alpha=.07, zorder=0)
        ax.barh(y, vals, height=.62, color=[c for _, _, c in rows], zorder=2)
        for idx, (yy, val) in enumerate(zip(y, vals)):
            ax.text(val + xmax * .013, yy,
                    f"{val:.3f}" if metric == "map5095" else f"{val:.2f}",
                    va="center", fontsize=9,
                    fontweight="bold" if idx == len(rows)-1 else "normal")
        for sep in (5.5, 3.5):
            ax.axhline(sep, color="#b5b5b5", lw=.7)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-.65, len(rows)-.35)
        ax.set_title(title, pad=12)
        ax.set_xlabel("mAP50:95 (higher is better)" if metric == "map5095"
                      else "FPPI at recall 0.70 (lower is better)")
        ax.set_yticks(y)
        clean(ax, "x")
    axes[0].set_yticklabels([label for _, label, _ in rows], fontsize=9)
    for tick in axes[0].get_yticklabels():
        if "SOAD" in tick.get_text():
            tick.set_fontweight("bold")
    axes[1].tick_params(axis="y", left=False)
    fig.subplots_adjust(left=.30, right=.98, top=.90, bottom=.14, wspace=.22)
    save(fig, str(ROOT / "figures/fig_ladder"))


if __name__ == "__main__":
    draw_ladder()
