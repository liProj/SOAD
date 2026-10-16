"""Reconcile the downloaded citrus RGB-D dataset with what the paper prints (2,200 images, 102,719 instances)."""
import glob, os, hashlib, collections, json
import numpy as np, cv2
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
D = f"{B}/data/citrus"
out, tot_i, tot_n, md5 = [], 0, 0, collections.defaultdict(list)
stats = {}
for s in ["train", "val", "test"]:
    ims = sorted(f for f in glob.glob(f"{D}/{s}/images/*") if not os.path.basename(f).startswith("."))
    stem = {os.path.splitext(os.path.basename(f))[0] for f in ims}
    lab = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{D}/{s}/labels/*.txt")}
    cnt = {k: len({os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{D}/{s}/{k}/*") if not os.path.basename(f).startswith('.')} & stem)
           for k in ["depth", "geom/B", "geom/N", "geom/D"]}
    c, per, empty, dup_rows, oob, sizes, areas, negf, negi = collections.Counter(), [], 0, 0, 0, collections.Counter(), [], 0, 0
    for st in sorted(stem & lab):
        rows = [l.split() for l in open(f"{D}/{s}/labels/{st}.txt").read().strip().splitlines() if l.strip()]
        per.append(len(rows)); empty += not rows
        isneg = any(float(v) < 0 for r in rows for v in r)
        negf += isneg; negi += len(rows) * isneg
        dup_rows += len(rows) - len({tuple(r) for r in rows})
        for r in rows:
            c[int(r[0])] += 1
            x, y, w, h = map(float, r[1:5]); areas.append(w * h)
            oob += not (0 <= x - w / 2 + 1e-6 and x + w / 2 <= 1 + 1e-6 and 0 <= y - h / 2 + 1e-6 and y + h / 2 <= 1 + 1e-6)
    for f in ims:
        md5[hashlib.md5(open(f, "rb").read()).hexdigest()].append(f"{s}/{os.path.basename(f)}")
    for f in ims[:: max(1, len(ims) // 50)]:
        im = cv2.imread(f); sizes[f"{im.shape[1]}x{im.shape[0]}"] += 1
    n = sum(c.values()); tot_i += len(ims); tot_n += n
    stats[s] = dict(images=len(ims), labels=len(lab), img_without_label=len(stem - lab), label_without_img=len(lab - stem), **cnt,
                    instances=n, fruit_tree=c[0], fruit_ground=c[1], empty_label_files=empty, duplicate_label_rows=dup_rows,
                    boxes_outside_image=oob, files_with_negative_value=negf, instances_in_those_files=negi, per_image_mean=round(float(np.mean(per)), 1), per_image_max=int(max(per)),
                    median_box_area_frac=float(np.median(areas)), sizes=dict(sizes))
dups = {k: v for k, v in md5.items() if len(v) > 1}
stats["total"] = dict(images=tot_i, instances=tot_n, duplicate_image_groups=len(dups),
                      cross_split_duplicates=sum(len({x.split('/')[0] for x in v}) > 1 for v in dups.values()))
json.dump(stats, open(f"{B}/results/data_check.json", "w"), indent=1)
L = ["# Data check - citrus-fruit-rgbd (Hugging Face `ariel0818/citrus-fruit-rgbd`, CC BY 4.0) vs. GeoSphere-DETR paper\n",
     "| item | paper | downloaded | match |", "|---|---|---|---|",
     f"| images | 2,200 | {tot_i:,} | {'yes' if tot_i == 2200 else 'NO'} |",
     f"| annotated instances | 102,719 | {tot_n:,} | {'yes' if tot_n == 102719 else 'NO'} |",
     f"| split train/val/test (dataset card) | 1500/200/500 | {stats['train']['images']}/{stats['val']['images']}/{stats['test']['images']} | {'yes' if (stats['train']['images'], stats['val']['images'], stats['test']['images']) == (1500, 200, 500) else 'NO'} |",
     f"| duplicate images (md5) | not stated | {len(dups)} groups, {stats['total']['cross_split_duplicates']} across splits | - |", "",
     "| split | images | labels | depth | geom B | geom N | geom D | instances | on-tree | on-ground | empty label files | dup rows | files with a negative value (dropped by the loader) | instances in them | mean/img | max/img |", "|" + "---|" * 16]
for s in ["train", "val", "test"]:
    t = stats[s]
    L.append(f"| {s} | {t['images']} | {t['labels']} | {t['depth']} | {t['geom/B']} | {t['geom/N']} | {t['geom/D']} | {t['instances']:,} | {t['fruit_tree']:,} | {t['fruit_ground']:,} | {t['empty_label_files']} | {t['duplicate_label_rows']} | {t['files_with_negative_value']} | {t['instances_in_those_files']} | {t['per_image_mean']} | {t['per_image_max']} |")
L.append("\nImage sizes (sampled): " + "; ".join(f"{s}: {stats[s]['sizes']}" for s in ['train', 'val', 'test']))
open(f"{B}/data_check.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
