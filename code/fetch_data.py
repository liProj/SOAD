"""Download the public RGB-D citrus dataset (CC BY 4.0) used by GeoSphere-DETR. Labels first so counts can be reconciled early."""
import sys
from huggingface_hub import snapshot_download
OUT = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "citrus")
pat = {"labels": ["*/labels/*", "README.md"], "all": None}[sys.argv[1]]
p = snapshot_download("ariel0818/citrus-fruit-rgbd", repo_type="dataset", local_dir=OUT, allow_patterns=pat, max_workers=8)
print("done", p)
