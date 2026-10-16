# Detector dependency

The training/evaluation wrappers expect a modified Ultralytics-based GeoSphere-DETR implementation at `third_party/Geosphere-DETR-run`.

Upstream project referenced by the research scripts: https://github.com/Ariel0818/Geosphere-DETR

The actual fork directory and `results/fork_patch.diff` in the supplied local project were empty. This repository therefore does not include or claim to reconstruct the original modifications. In particular, synchronized five-channel RGB/geometry augmentation requires the original loader changes. Recover the exact implementation and its license before attempting to reproduce the geometric experiments.
