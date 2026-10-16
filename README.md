# SOAD

**SOAD: Observation-Aware Citrus Detection with Cross-View Aggregation and Quality Calibration**

**SOAD：融合跨视图聚合与质量校准的观测感知柑橘检测**

Research code and manuscript sources for citrus detection with native-pixel tiles, cross-tile aggregation, mirror fusion, and post-detection quality calibration.

## Contents

| Directory | Contents |
| --- | --- |
| `code/` | Training wrappers, tiled inference, box fusion, calibration, evaluation, and figure scripts |
| `configs/` | RGB and geometric model/data configurations |
| `paper/manuscript_en.tex` | Current English manuscript |
| `paper/manuscript_zh.tex` | Current Chinese manuscript |
| `paper/manuscript_en.pdf` | Compiled English paper |
| `paper/manuscript_zh.pdf` | Compiled Chinese paper |
| `figures/` | Figures used by the current manuscripts, including the vector architecture diagram |
| `results/` | Available nonempty JSON summaries; not a complete experiment archive |

## Availability

This upload contains the available project source and manuscript artifacts. The original training environment, detector fork modifications, trained checkpoints, complete detection dumps, and dataset are **not bundled**. The local `third_party` directory and saved fork patch were empty when the repository was prepared. Exact retraining and complete regeneration of all plots therefore require recovery of those dependencies and artifacts. The historical `write_paper.py`, `write_report.py`, and queue scripts are retained as research utilities; compile the current `manuscript_*.tex` files directly for the current papers.

Some manuscript metrics were edited during manuscript revision and are not regenerated from the bundled JSON files. In particular, the recall/counting consistency audit in `paper/召回率更正核查.md` records an unresolved issue. Treat the manuscript and saved experimental summaries as separate versioned artifacts until the underlying per-image predictions are reconciled.

## Setup

Training scripts target a Linux/CUDA environment. Install PyTorch appropriate for your CUDA version, then install the direct dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python code/configure_workspace.py
```

The detector scripts import a modified Ultralytics implementation from `third_party/Geosphere-DETR-run`. See [third_party/README.md](third_party/README.md) for the upstream reference and missing fork information. Installing ordinary Ultralytics is not a substitute for the geometric architecture and loader modifications.

The dataset is [citrus-fruit-rgbd](https://huggingface.co/datasets/ariel0818/citrus-fruit-rgbd), published under CC BY 4.0. Download it separately:

```bash
python code/fetch_data.py all
```

Expected data layout is `data/citrus/{train,val,test}/` with `images`, `labels`, depth, and curvature inputs as supplied by the dataset. Check the paths expected by `boxfeat.py` and the dataset configuration before running.

## Pipeline entry points

The following commands require the detector fork, dataset, pretrained weights, and CUDA environment described above. They document the workflow; the complete training pipeline has not been rerun during repository preparation.

```bash
# Set the hash seed before tile generation because empty-tile sampling uses hash(stem).
PYTHONHASHSEED=0 python code/make_tiles.py
python code/make_tile_variants.py

# Example RGB tile training (pretrained weights must be obtained separately).
python code/run_det.py --name T_rgb_s0 --model weights/rtdetr-l.pt --data configs/tiles_rgb.yaml --epochs 10 --batch 8 --seed 0

# Full-image evaluation of tiled predictions.
python code/eval_tiles.py --weights runs/T_rgb_s0/weights/last.pt --tag T_rgb_s0_full --data configs/tiles_rgb.yaml

# Single-view, mirrored-view, fusion, and validation-fitted calibration workflow.
bash code/full_eval.sh T_rgb_s0 rgb
```

`make_tile_variants.py` also creates an experimental gamma variant and uses symlinks. Historical queue scripts contain run-specific scheduling assumptions; inspect them before use. Setting `PYTHONHASHSEED=0` makes tile sampling repeatable for a new run but does not establish the hash seed used for the original experiments.

Core modules:

- `stitch.py`: coordinate restoration, internal-boundary handling, and deduplication.
- `merge.py`: confidence-weighted fusion of view predictions.
- `rescore.py`, `run_rescore.py`, `boxfeat.py`: candidate features, correctness/quality learning, and score calibration.
- `evalbox.py`, `bootstrap.py`: full-image evaluation and paired image resampling.

## Compile the papers

Install XeLaTeX, Times New Roman for English, and the CTeX/Chinese fonts required by the Chinese source. Run from `paper/`:

```bash
xelatex -interaction=nonstopmode -halt-on-error manuscript_en.tex
xelatex -interaction=nonstopmode -halt-on-error manuscript_en.tex
xelatex -interaction=nonstopmode -halt-on-error manuscript_zh.tex
xelatex -interaction=nonstopmode -halt-on-error manuscript_zh.tex
```

The current vector architecture diagram can be rebuilt with `python code/fig_pipeline_selected.py`. Photo insets are illustrative crops from the selected artwork, not experimental predictions; network blocks, text, arrows, and diagram borders are vector elements.

## Licensing and attribution

No new software license is assigned by this upload. Upstream detector code, datasets, and other third-party assets retain their respective terms. Consult their original licenses before redistribution or reuse.

## 中文说明

仓库包含当前论文代码、中英文 LaTeX/PDF、论文使用的图和现有非空 JSON 汇总。数据集、模型权重、训练日志和逐图预测缓存未上传。原项目的第三方框架目录及补丁文件为空，需要补齐相应实现才能完整运行训练和几何对照。上传时仅整理路径与使用说明，未重新训练，也未修改实验数值。
