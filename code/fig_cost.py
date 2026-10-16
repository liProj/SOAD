"""Accuracy versus inference cost, with a detail panel and numbered key."""
import json
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from figstyle import plt, clean, save

NAMES = [
    "YOLO11m", "RT-DETR-L", "GeoSphere-DETR",
    "SOAD w/o MF & EC", "SOAD w/o MF", "SOAD w/o EC", "Ours (SOAD)",
    "SOAD w/o MF & EC [20 epochs]", "SOAD w/o MF [20 epochs]",
    "SOAD w/o EC [20 epochs]", "SOAD [20 epochs]",
    "GeoSphere-DETR (T)", "GeoSphere-DETR (T+A)", "GeoSphere-DETR (T+A+EC)",
    "GeoSphere-DETR (T+A+MF)", "GeoSphere-DETR (T+A+MF+EC)",
    "GeoSphere-DETR (T+A, 960)"
]
TAGS = [
    "Y11m_s0", "E1_rtdetr_s0", "E4_rescore_s0",
    "T_rgb_s0_full", "T_rgb_s0_full_per", "T_rgb_s0_wbf", "T_rgb_s0_wbf_per",
    "T_rgb20_s0_full", "T_rgb20_s0_full_per", "T_rgb20_s0_wbf", "T_rgb20_s0_wbf_per",
    "T_E4_unaligned_s0_full", "T_E4_aligned_s0_full", "T_E4_aligned_s0_full_per",
    "T_E4_aligned_s0_wbf", "T_E4_aligned_s0_wbf_per", "T960_E4_aligned_s0_full"
]
COLORS = [
    "#2563EB", "#E87918", "#059669", "#A855F7", "#C83F49", "#009CB5",
    "#D09A00", "#8C564B", "#D653A3", "#64748B", "#78951C", "#9A5EBA",
    "#375873", "#E47C78", "#237D68", "#B8541A", "#5965B3"
]

def draw_cost(runs, output):
    assert set(TAGS) == set(runs)
    assert len(set(COLORS)) == len(TAGS)
    fig = plt.figure(figsize=(12, 6.7))
    overview = fig.add_axes([0.075, 0.49, 0.37, 0.43])
    detail = fig.add_axes([0.555, 0.49, 0.40, 0.43])
    axes = [overview, detail]
    handles = []
    for i, tag in enumerate(TAGS):
        r = runs[tag]
        x, y = r["views"], r["test"]["map5095"]
        marker = "*" if i == 6 else "o"
        size = 115 if i == 6 else 48
        for ax in [overview]:
            ax.scatter(x, y, s=size, marker=marker, color=COLORS[i],
                       edgecolor="white", linewidth=0.65, zorder=4)
        handles.append(Line2D([], [], color=COLORS[i], marker=marker, linestyle="none",
                              markersize=9 if i == 6 else 7,
                              label=f"{i+1:02d}  {NAMES[i]}"))
        if x == 1:
            overview.annotate(f"{i+1:02d}", (x,y), xytext=(9,0),
                              textcoords="offset points", va="center", fontsize=10,
                              color=COLORS[i], fontweight="bold")
    overview.set_xscale("log")
    overview.set_xlim(0.8, 34)
    overview.set_ylim(0.38, 0.59)
    overview.set_xticks([1,12,24])
    overview.set_xticklabels(["1","12","24"])
    overview.xaxis.set_minor_locator(NullLocator())
    overview.set_title("(a) All configurations", loc="left", fontsize=11, pad=10)
    overview.set_xlabel("640-px equivalent passes per frame (log)")
    overview.set_ylabel("Held-out mAP50:95")
    overview.add_patch(plt.Rectangle((10.8,0.548),18.2,0.036,
                       fill=False, edgecolor="#64748B", linestyle="--", linewidth=0.9))
    overview.annotate("Detail in (b)", (10.8,0.555), xytext=(2.3,0.515), fontsize=9,
                      color="#52514E", arrowprops=dict(arrowstyle="-", color="#64748B"))
    detail.set_ylim(0.546,0.588)
    detail.set_xlabel("Configuration ID (grouped by inference cost)")
    detail.set_ylabel("Held-out mAP50:95")
    detail.set_title("(b) Tiled configurations — grouped detail", loc="left", fontsize=11, pad=10)
    # A categorical detail axis separates configurations with identical cost.
    order=[]
    for cost in [12,24,27]:
        indices=sorted([i for i,t in enumerate(TAGS) if runs[t]["views"]==cost],
                       key=lambda i:runs[TAGS[i]]["test"]["map5095"])
        start=len(order)
        order.extend(indices)
        stop=len(order)
        if cost==24:
            detail.axvspan(start-0.5,stop-0.5,color="#F1F5F9",zorder=0)
        if start:
            detail.axvline(start-0.5,color="#CBD5E1",lw=0.8)
        detail.text((start+stop-1)/2,0.586,f"{cost} passes" if cost!=27 else "27",
                    ha="center",va="center",fontsize=8.5,color="#475569")
        for x, i in enumerate(indices,start):
            y=runs[TAGS[i]]["test"]["map5095"]
            detail.scatter(x,y,s=125 if i==6 else 48,marker="*" if i==6 else "o",
                           color=COLORS[i],edgecolor="white",lw=0.65,zorder=4)
    detail.set_xlim(-0.7,len(order)-0.3)
    detail.set_xticks(range(len(order)))
    detail.set_xticklabels([f"{i+1:02d}" for i in order],fontsize=8)
    clean(overview,"both")
    clean(detail,"y")
    legend=fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(0.065,0.015,0.91,0.34),
                      mode="expand",ncol=2,fontsize=9.5,frameon=False,
                      handletextpad=0.6,columnspacing=2,labelspacing=0.65,borderaxespad=0)
    legend.get_texts()[6].set_fontweight("bold")
    save(fig,str(output))

if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    runs=json.loads((root/"results/summary.json").read_text(encoding="utf-8"))["runs"]
    draw_cost(runs,root/"figures/fig_cost")
