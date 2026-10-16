"""Data-only figures: what the benchmark looks like and how the held-out field differs from the training fields."""
import os, sys, pickle
import numpy as np, cv2
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from figstyle import plt, CAT, INK2, GRID, clean, save
from evalbox import load_gt
from boxfeat import box_features, NAMES
F = f"{B}/figures"
SP = ["train", "val", "test"]; COL = dict(train=CAT[0], val=CAT[2], test=CAT[1])
gt = {s: load_gt(s) for s in SP}
cache = f"{B}/results/gt_feats.pkl"
if os.path.exists(cache):
    feats = pickle.load(open(cache, "rb"))
else:
    feats = {}
    for s in SP:
        st = list(gt[s])[:: (6 if s == "train" else 2)]
        feats[s] = np.concatenate([box_features(s, k, gt[s][k][:, 1:5]) for k in st if len(gt[s][k])])
        print(s, feats[s].shape, flush=True)
    pickle.dump(feats, open(cache, "wb"))
ix = {n: i for i, n in enumerate(NAMES)}

# --- Fig: size and density ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
for s in SP:
    side = np.concatenate([np.sqrt((g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2])) for g in gt[s].values() if len(g)])
    ax[0].hist(side, bins=np.arange(0, 90, 2), histtype="step", lw=1.8, color=COL[s], density=True, label=f"{s} (median {np.median(side):.0f} px)")
    n = np.array([len(g) for g in gt[s].values()])
    ax[1].hist(n, bins=np.arange(0, 130, 5), histtype="step", lw=1.8, color=COL[s], density=True, label=f"{s} (mean {n.mean():.1f})")
ax[0].axvline(32, color=INK2, ls=":", lw=1); ax[0].text(33, ax[0].get_ylim()[1] * 0.9, "COCO 'small' limit", fontsize=7, color=INK2)
ax[0].set_xlabel("box side in the 1920x1200 frame (px)"); ax[0].set_ylabel("density"); ax[0].set_title("Fruit size"); ax[0].legend()
sec = ax[0].secondary_xaxis("top", functions=(lambda x: x / 3, lambda x: x * 3)); sec.set_xlabel("side after resizing the frame to 640 px", fontsize=8)
ax[1].set_xlabel("annotated fruit per frame"); ax[1].set_title("Density"); ax[1].legend()
w = 0.38
for j, s in enumerate(SP):
    c = np.concatenate([g[:, 0] for g in gt[s].values()])
    tot = len(c); tree = (c == 0).sum()
    ax[2].bar(j - w / 2, tree / tot * 100, w, color=CAT[5]); ax[2].bar(j + w / 2, (tot - tree) / tot * 100, w, color=CAT[3])
    ax[2].text(j - w / 2, tree / tot * 100 + 1, f"{tree:,}", ha="center", fontsize=7); ax[2].text(j + w / 2, (tot - tree) / tot * 100 + 1, f"{tot - tree:,}", ha="center", fontsize=7)
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(SP); ax[2].set_ylabel("share of instances (%)"); ax[2].set_title("Classes: on tree (green) / on ground (yellow)")
for a in ax: clean(a)
fig.tight_layout(); save(fig, f"{F}/fig_data_size_density")

# --- Fig: domain shift in physical quantities --------------------------------------------------
P = [("z_c", "depth of the fruit centre (m)", 1e-3, (0.5, 7)), ("phys_w", "metric width  w x Z  (px m)", 1, (10, 130)),
     ("z_valid", "share of valid depth pixels in the box", 1, (0, 1)), ("sat", "mean saturation in the box centre", 1, (0, 1)),
     ("val", "mean brightness in the box centre", 1, (0, 1)), ("orange_frac", "share of fruit-coloured pixels", 1, (0, 1))]
fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))
for a, (k, lab, sc, rg) in zip(ax.ravel(), P):
    for s in SP:
        v = feats[s][:, ix[k]] * sc; v = v[np.isfinite(v)]
        a.hist(v, bins=np.linspace(*rg, 50), histtype="step", lw=1.8, color=COL[s], density=True, label=f"{s}: median {np.median(v):.2f}")
    a.set_xlabel(lab); a.legend(fontsize=7); clean(a)
fig.suptitle("Annotated fruit in the training fields vs. the held-out field and day", y=1.0, fontsize=10)
fig.tight_layout(); save(fig, f"{F}/fig_data_domain_shift")

# --- Fig: modalities of one training and one test frame ----------------------------------------
fig, ax = plt.subplots(2, 4, figsize=(12, 4.2))
for r, s in enumerate(["train", "test"]):
    k = sorted(gt[s], key=lambda q: -len(gt[s][q]))[len(gt[s]) // 4]
    im = cv2.cvtColor(cv2.imread(f"{B}/data/citrus/{s}/images/{k}.jpg"), cv2.COLOR_BGR2RGB)
    z = np.load(f"{B}/data/citrus/{s}/depth/{k}.npy").astype(float); z[(z <= 0) | (z >= 19999)] = np.nan
    bm = cv2.imread(f"{B}/data/citrus/{s}/geom/B/{k}.png", 0); nm = cv2.imread(f"{B}/data/citrus/{s}/geom/N/{k}.png", 0)
    ax[r, 0].imshow(im)
    for c, x1, y1, x2, y2 in gt[s][k]:
        ax[r, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=CAT[3] if c else "#00e5ff", lw=0.5))
    ax[r, 1].imshow(z / 1000, cmap="viridis_r", vmin=0.5, vmax=6); ax[r, 2].imshow(bm, cmap="gray"); ax[r, 3].imshow(np.clip(nm.astype(float) * 6, 0, 255), cmap="magma")
    ax[r, 0].set_ylabel(f"{s}\n{len(gt[s][k])} fruit", fontsize=8)
    for a in ax[r]: a.set_xticks([]); a.set_yticks([])
for a, t in zip(ax[0], ["colour + annotation", "depth (0.5-6 m; white = no stereo match)", "convexity map B", "bumpiness map N (x6)"]): a.set_title(t, fontsize=9)
fig.tight_layout(); save(fig, f"{F}/fig_data_modalities")
