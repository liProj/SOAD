"""Plot completed perturbation evaluations with shared, zoomed y-axis ranges."""
import json
import re
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from figstyle import plt, CAT, clean, save

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
clean0 = summary["runs"]["T_E4_aligned_s0_full"]["test"]
rgb = summary["runs"]["T_rgb_s0_full"]["test"]
series = {}
sources = {}
for kind in ("miss", "noise"):
    prefix = f"T_E4_aligned_s0_{kind}"
    values = []
    for log in (ROOT / "logs").glob(f"eval_{prefix}*.log"):
        matches = re.findall(r"^test (\{[^\r\n]+\})", log.read_text(encoding="utf-8"), re.M)
        if not matches:
            continue
        metrics = json.loads(matches[-1])
        assert metrics["n_images"] == clean0["n_images"]
        assert metrics["n_gt"] == clean0["n_gt"]
        strength = float(log.stem.removeprefix("eval_" + prefix))
        values.append((strength, metrics))
        sources[str(log.relative_to(ROOT))] = metrics
    if values:
        series[kind] = [(0.0, clean0)] + sorted(values)
assert len(series) == 2

fig, ax = plt.subplots(2, 2, figsize=(10.4, 6.2), squeeze=False)
metrics = (("map5095", "held-out mAP50:95"), ("fppi", "FPPI at confidence 0.25"))
for i, (metric, label) in enumerate(metrics):
    all_values = [r[metric] for values in series.values() for _, r in values]
    low, high = min(all_values), max(all_values)
    pad = max((high-low) * 0.20, 0.00005 if metric == "map5095" else 0.01)
    limits = (low-pad, high+pad)
    print(metric, "geometry range", (low, high), "axis", limits, "RGB", rgb[metric])
    for j, (kind, values) in enumerate(series.items()):
        a = ax[i, j]
        x = [strength for strength, _ in values]
        y = [r[metric] for _, r in values]
        a.plot(x, y, "o-", color=CAT[0], markersize=4)
        a.set_ylim(*limits)
        a.yaxis.set_major_locator(MaxNLocator(nbins=5))
        a.yaxis.set_major_formatter(FormatStrFormatter("%.5f" if metric == "map5095" else "%.2f"))
        a.set_ylabel(label)
        a.set_xlabel("masked share of the geometry maps" if kind == "miss"
                     else "noise sigma (geometry maps in [0, 1])")
        ref = f"{rgb[metric]:.5f}" if metric == "map5095" else f"{rgb[metric]:.2f}"
        a.set_title(("Block masking" if kind == "miss" else "Additive Gaussian noise")
                    + f"\nRGB reference = {ref} (outside zoom)", fontsize=9)
        clean(a, "both")
fig.suptitle("Aligned geometry model under input perturbations (zoomed y-axes)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, str(ROOT / "figures/fig_robustness"))
(ROOT / "results/robustness_plot_sources.json").write_text(
    json.dumps({"baseline_source": "results/summary.json", "clean": clean0,
                "rgb": rgb, "perturbation_logs": sources}, indent=2), encoding="utf-8")
