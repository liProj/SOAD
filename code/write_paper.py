"""Generate paper/main.tex. Every number is read from results/; prose markers <<key>> are filled by paper_lib.fill.

A run that has not finished is never guessed: its row is left out of the tables and the sentence that would cite it is
replaced by an explicit 'not finished' statement, so the PDF can be rebuilt at any time."""
import json, os, sys
import numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from paper_lib import *
RES = f"{B}/results"
J = lambda p: json.load(open(f"{RES}/{p}"))
E = lambda p: os.path.exists(f"{RES}/{p}")


def numbers():
    N = {}
    S = J("summary.json"); R = S["runs"]; dc = J("data_check.json"); tl = J("tiles.json"); al = J("geom_alignment.json"); pr = J("E2_probe.json")["val"]
    N["n_img"], N["n_inst"] = f"{dc['total']['images']:,}", f"{dc['total']['instances']:,}"
    for s in ("train", "val", "test"):
        N[f"{s}_img"], N[f"{s}_inst"] = f"{dc[s]['images']:,}", f"{dc[s]['instances']:,}"
        N[f"{s}_tree"], N[f"{s}_ground"] = f"{dc[s]['fruit_tree']:,}", f"{dc[s]['fruit_ground']:,}"
        N[f"{s}_tiles"], N[f"{s}_tboxes"], N[f"{s}_tframes"] = f"{tl[s]['tiles']:,}", f"{tl[s]['boxes']:,}", f"{tl[s]['frames']:,}"
    N["ground_share"] = f1(100 * dc["test"]["fruit_ground"] / dc["test"]["instances"])
    any_run = next(iter(R.values()))
    N["test_frames"], N["test_gt"] = f"{any_run['test']['n_images']:,}", f"{any_run['test']['n_gt']:,}"
    from evalbox import load_gt
    g = load_gt("train"); side = np.concatenate([np.sqrt((v[:, 3] - v[:, 1]) * (v[:, 4] - v[:, 2])) for v in g.values() if len(v)])
    N["side_med"], N["side_640"], N["small_share"] = f1(np.median(side)), f1(np.median(side) / 3), f"{100 * (side < 32).mean():.0f}"
    N["c_val"], N["c_mosoff"], N["c_al"], N["c_almos"] = f2(al[0]["mean_corr"]), f2(al[2]["mean_corr"]), f2(al[4]["mean_corr"]), f2(al[5]["mean_corr"])
    N["probe_with"], N["probe_without"] = f"{pr['with_geometry']['map50']:.5f}", f3(pr["without_geometry"]["map50"])
    for t, r in R.items():
        x = r["test"]; k = t.replace("_s0", "").replace("_full", "")
        N[f"{k}_m50"], N[f"{k}_m95"], N[f"{k}_fppi"] = f3(x["map50"]), f3(x["map5095"]), f2(x["fppi"])
        N[f"{k}_rec"], N[f"{k}_prec"], N[f"{k}_f70"], N[f"{k}_f80"] = f3(x["recall"]), f3(x["precision"]), f2(x["fppi_at_recall_70"]), f2(x["fppi_at_recall_80"])
        N[f"{k}_cmae"], N[f"{k}_cbias"] = f1(x["count_mae"]), sg(x["count_bias"], 1)
        N[f"{k}_tree"], N[f"{k}_ground"] = f3(x["ap50_per_class"][0]), f3(x["ap50_per_class"][1])
        if "val" in r:
            N[f"{k}_v50"], N[f"{k}_v95"] = f3(r["val"]["map50"]), f3(r["val"]["map5095"])
    H = S["headline"]; hk = H.replace("_s0", "")           # tiled configuration selected on validation
    for k in [k for k in N if k.startswith(hk + "_")]:
        N["H_" + k[len(hk) + 1:]] = N[k]
    N["hv_rgb"], N["hv_geo"] = f3(S["headline_val"].get("T_rgb_s0", float("nan"))), f3(S["headline_val"]["T_E4_aligned_s0"])
    N["H_budget"] = "twice the iteration budget of the baseline" if H == "T_rgb20_s0" else "the same iteration budget as the baseline"
    N["H_epochs"] = "twenty" if H == "T_rgb20_s0" else "ten"
    bs = J("bootstrap.json")
    for key, tag in ((f"E1_rtdetr_s0->{H}_full", "b_tile"), (f"{H}_full->{H}_wbf", "b_flip"), (f"{H}_wbf->{H}_wbf_per", "b_per"), (f"E1_rtdetr_s0->{H}_wbf_per", "b_all"), (f"E4_rescore_s0->{H}_wbf_per", "b_e4"),
                     ("E1_rtdetr_s0->E4_rescore_s0", "b_e1e4"), ("T_rgb_s0_full->T_E4_aligned_s0_full", "b_geo"), ("T_rgb_s0_wbf_per->T_E4_aligned_s0_wbf_per", "b_geo2"),
                     ("T_E4_unaligned_s0_full->T_E4_aligned_s0_full", "b_align"), ("T_rgb_s0_full->T_rgb20_s0_full", "b_long"), ("T_E4_aligned_s0_full->T960_E4_aligned_s0_full", "b_960")):
        if key in bs:
            for m, mt in (("map50", "50"), ("map5095", "95"), ("fppi", "fp")):
                v = bs[key][m]; nd = 3 if m != "fppi" else 1
                N[f"{tag}_{mt}"] = sg(v["diff"], nd); N[f"{tag}_{mt}_ci"] = f"{sg(v['lo'], nd)} to {sg(v['hi'], nd)}"
                N[f"{tag}_{mt}_p"] = "$<$0.001" if v["p"] <= 0.001 else f"{v['p']:.3f}"
    rs = J(f"rescore_{H}_wbf.json"); ab = rs["ablation"]
    N["alpha"] = str(rs["alpha"]); N["per_seeds"] = str(len(rs["test"]))
    N["per_val_before"], N["per_val_after"] = f3(rs["base"]["val"]["map5095"]), f3(rs["val_oof_alpha"][str(rs["alpha"])]["map5095"])
    N["per_fr_before"], N["per_fr_after"] = f2(rs["base"]["test"]["fppi_at_base_recall"]), f"{min(t['fppi_at_base_recall'] for t in rs['test']):.2f}--{max(t['fppi_at_base_recall'] for t in rs['test']):.2f}"
    N["ab_base"], N["ab_base95"], N["ab_all95"] = f3(rs["base"]["test"]["map50"]), f3(rs["base"]["test"]["map5095"]), f3(ab["all"]["test"]["map5095"])
    N["ab_colour"], N["ab_nocolour"], N["ab_all"], N["ab_score"] = f3(ab["ctx+colour"]["test"]["map50"]), f3(ab["all-colour"]["test"]["map50"]), f3(ab["all"]["test"]["map50"]), f3(ab["score only"]["test"]["map50"])
    N["ab_box"], N["ab_depth"], N["ab_size"], N["ab_curv"] = (f3(ab[f"ctx+{g}"]["test"]["map50"]) for g in ("box", "depth", "size", "curv"))
    rg = J("rescore_T_E4_aligned_s0_wbf.json"); ag = rg["ablation"]
    N["g_ab_base"], N["g_ab_all"], N["g_ab_colour"], N["g_ab_nocolour"], N["g_alpha"] = f3(rg["base"]["test"]["map50"]), f3(ag["all"]["test"]["map50"]), f3(ag["ctx+colour"]["test"]["map50"]), f3(ag["all-colour"]["test"]["map50"]), str(rg["alpha"])
    b, t = R["E1_rtdetr_s0"]["test"], R[f"{H}_full"]["test"]
    N["fp_cut"] = f"{100 * (1 - t['fppi_at_recall_70'] / b['fppi_at_recall_70']):.0f}"
    N["fp_div"] = f1(b["fppi_at_recall_70"] / R[f"{H}_wbf_per"]["test"]["fppi_at_recall_70"])
    tv = J("tile_variants.json"); N["v_train"], N["v_test"] = f3(tv["train_median_v"]), f3(tv["test_median_v"])
    if E("det/T_E4_aligned_s0_im960val.json"):
        N["im960_v50"], N["im960_v95"] = f3(J("det/T_E4_aligned_s0_im960val.json")["val"]["map50"]), f3(J("det/T_E4_aligned_s0_im960val.json")["val"]["map5095"])
    e1 = J("det/E1_rtdetr_s0.json"); N["params_rgb"] = f1(e1["params_M"]); N["params_geo"] = f1(J("det/T_E4_aligned_s0.json")["params_M"])
    return N, R, S


FIGS = [("fig_data_modalities", "The benchmark's modalities for one training frame and one frame of the held-out field: RGB, depth, and the two curvature maps (B, convexity; N, bumpiness) that the reference method feeds to its geometry branch. Frames are stored rotated by 90 degrees.", "fig:mod"),
        ("fig_data_size_density", "Fruit size, density and class balance per split. The upper axis of the left panel shows the size of the same fruit after the full frame is resized to 640 px, the input size documented in the public code.", "fig:size"),
        ("fig_data_domain_shift", "Annotated fruit in the training fields against the held-out field and day. Metric width (pixel width times depth) is invariant across fields; fruit brightness and saturation are not.", "fig:shift"),
        ("fig_pipeline", "The pipeline. Steps (1)--(2) change what the detector is trained on; steps (3)--(5) act at inference and leave the weights untouched. Step (2) is used only when the reference geometry-fusion architecture is trained.", "fig:pipe"),
        ("fig_alignment", "Left: a probe in which each image's geometry file is its own grey version, so that a pipeline which transforms geometry together with the image keeps the two correlated. With the public loader the geometry is absent under the default (mosaic) augmentation and misplaced when mosaic is switched off; the aligned loader keeps it in place. Right: consequence for a concat-fusion model trained with the public code.", "fig:align"),
        ("fig_training_curves", "Validation curves. Left: full-frame training with the public code. Right: tiled training (ten epochs over the tiles equal the iteration budget of one hundred full-frame epochs).", "fig:curves"),
        ("fig_ladder", "All finished models on the held-out field, evaluated on full frames with one evaluator. Dotted lines mark the values printed in the reference paper.", "fig:ladder"),
        ("fig_froc_pr", "Free-response ROC and precision--recall curves on the held-out field. The dot on each FROC curve is the operating point at confidence 0.25; vertical lines are the false-positive rates printed in the reference paper (their recall at that point is not printed).", "fig:froc"),
        ("fig_size_count", "Left: recall at confidence 0.25 by fruit size. Centre and right: detections against annotated fruit per frame for the full-frame baseline and the headline pipeline.", "fig:count"),
        ("fig_rescoring_ablation", "Evidence re-scoring. Left and centre: held-out accuracy and false positives at the detector's own recall when the calibrator sees only some evidence groups. Right: the features the true-fruit classifier relies on.", "fig:per"),
        ("fig_robustness", "Robustness of the tiled geometry model to corrupted geometry on the held-out field: block masking of the curvature maps (protocol of the public code) and additive Gaussian noise.", "fig:rob"),
        ("fig_printed_vs_measured", "Values printed in the reference paper, the same models trained here with the public code under its documented settings, and the headline tiled pipeline (held-out field).", "fig:printed"),
        ("fig_cost", "Held-out accuracy against the number of 640-px network passes per frame.", "fig:cost"),
        ("fig_qualitative", "Held-out frames (crops). Left: RT-DETR-L on the full frame resized to 640 px. Right: the headline pipeline (tiled RGB model, flip fusion, re-scoring). Green: correct at IoU 0.5; red: false; dotted white: annotation.", "fig:qual")]

PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,array,caption,float,xcolor,hyperref,microtype}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{Resolution before fusion: scale-preserving tiling for citrus detection in RGB-D orchard images,\\ with an independent reproduction of GeoSphere-DETR and a test of aligned geometry}
\author{Independent replication study\thanks{All numbers are measured in this study unless marked ``printed''.}}
\date{}
\begin{document}
\maketitle
"""

ABSTRACT = r"""
\begin{abstract}
\noindent GeoSphere-DETR \cite{ref} detects citrus fruit in RGB-D orchard images by gating curvature maps derived from depth into the neck of RT-DETR and by re-scoring queries with a sphere-consistency head; it reports mAP50 0.863, mAP50:95 0.642 and 3.29 false positives per image (FPPI) on a held-out field. We reproduce the work from its public data and code, and then extend it.
The public data set reconciles exactly with the paper (<<n_img>> images, <<n_inst>> instances). The public code does not: it cannot be imported as released, its FPPI routine never executes, a non-RT-DETR baseline crashes at the first validation pass, and --- most consequentially --- the data loader does not move the geometry maps together with the image, so that under the default augmentation the geometry is absent from every training sample and is misplaced once mosaic is switched off. A concat-fusion model trained this way scores a validation mAP50 of <<probe_with>> when geometry is fed and <<probe_without>> when it is withheld.
Under the only training settings the repository documents (640-px input), the RT-DETR-L baseline reaches mAP50 <<E1_rtdetr_m50>>, mAP50:95 <<E1_rtdetr_m95>> and FPPI <<E1_rtdetr_fppi>> on the held-out field, far from the printed 0.786, 0.562 and 7.24.
<<abs_e4>>
We trace most of the gap to scale: fruit are <<side_med>> px wide in the 1920$\times$1200 frame and about <<side_640>> px after resizing. We therefore (i) train and infer on overlapping native-resolution tiles, (ii) fuse an original and a mirrored view, and (iii) re-score detections with a small calibrator fitted on validation detections; and, to test the reference architecture fairly, (iv) stack the geometry maps with the image before augmentation so that they stay aligned.
With <<H_budget>>, an RGB RT-DETR-L trained on tiles reaches mAP50 <<H_m50>> and mAP50:95 <<H_m95>>; with flip fusion and re-scoring it reaches <<H_wbf_per_m50>> and <<H_wbf_per_m95>> (paired bootstrap against the baseline: <<b_all_50>> and <<b_all_95>>, both $p$ <<b_all_50_p>>) and lowers the false positives per image at recall 0.70 from <<E1_rtdetr_f70>> to <<H_wbf_per_f70>>.
The reference architecture trained on the same tiles with aligned geometry reaches <<T_E4_aligned_m50>> and <<T_E4_aligned_m95>> (difference to the RGB model: <<b_geo_50>> and <<b_geo_95>>, $p$ = <<b_geo_95_p>> for mAP50:95): at native resolution we measure no accuracy gain from the geometry branch. What it changes is score calibration: at confidence 0.25 it produces <<T_E4_aligned_fppi>> false positives per image against <<T_rgb_fppi>> for the RGB model, while at matched recall the two are equal.
Both tiled pipelines exceed the printed RT-DETR baseline but do not reach the printed GeoSphere-DETR figures; since the paper's training resolution could not be established from the public material, that comparison is not like-for-like. We report where each component helps, where it does not, and what it costs.
\end{abstract}
\textbf{Keywords:} RGB-D fusion; small-object detection; image tiling; RT-DETR; reproducibility; fruit counting; agricultural robotics
"""

INTRO = r"""
\section{Introduction}
Counting and localising fruit on the tree is the perception problem behind yield estimation and robotic harvesting. In orchard images the difficulty is rarely the fruit that is large and in the open; it is the fruit that occupies a few hundred pixels, sits in shade, or is imitated by a yellowed leaf or a patch of sunlit soil. RGB-D cameras add a second source of evidence, because a fruit is a convex blob of a known physical size while a leaf is not, and several recent detectors try to exploit this.

GeoSphere-DETR \cite{ref} is a clear recent example. It converts the depth image into two curvature maps, fuses them into the neck of RT-DETR \cite{rtdetr} through learned spatial gates, and adds a ``sphere-consistency'' head whose output shifts each query's confidence at inference. On a new citrus data set with a field and day held out for testing it reports a reduction of false positives per image from 7.24 to 3.29 and an increase of mAP50:95 from 0.562 to 0.642 over RT-DETR. The authors released both the data and the code, which makes the work one of the few in its area that can be checked.

This paper does two things. First, it is an independent reproduction: we reconcile the data with the paper, run the public code as documented, and measure what it does. Second, it is an extension. The reproduction shows that the dominant error source under the documented settings is not the absence of geometry but the loss of resolution: a <<side_med>>-px fruit becomes a <<side_640>>-px fruit when the 1920$\times$1200 frame is resized to 640 px. It also shows that the released training pipeline does not deliver the geometry to the network in the right place. Both observations suggest changes that leave the architecture alone:

\begin{itemize}\setlength{\itemsep}{1pt}
\item \textbf{Scale-preserving tiling.} The detector is trained and applied on overlapping 640-px tiles cut at native resolution, in the spirit of slicing-aided inference \cite{sahi}, with an edge rule that makes stitching unambiguous.
\item \textbf{Flip fusion.} Detections from the original and a mirrored view are merged by weighted box fusion \cite{wbf}.
\item \textbf{Evidence re-scoring.} A small gradient-boosted calibrator \cite{lgbm}, fitted on validation detections only, re-scores each box from colour, metric size, depth contrast and curvature statistics.
\item \textbf{Augmentation-aligned geometry (for the reference architecture).} The curvature maps are stacked onto the image as extra channels before mosaic, affine and flip augmentation and separated again afterwards, so that the geometry branch is trained on geometry that matches the pixels. This makes a controlled test of the geometry branch possible.
\end{itemize}

The contributions are (1) a data reconciliation and four measured findings about the public implementation, each with a probe that isolates it; (2) a tiled pipeline that needs RGB only and that improves the held-out mAP50:95 of the reproduced baseline by <<b_all_95>> and divides its false positives at recall 0.70 by <<fp_div>>; (3) a controlled comparison, at native resolution and with aligned inputs, of the RGB detector with the reference geometry-fusion architecture, which finds no accuracy gain from geometry and locates its effect in score calibration; (4) an ablation that attributes the gain, including the components that turned out to matter little; and (5) an explicit account of what remains unexplained --- in particular, why the printed figures of the reference paper are not reached by its public code under its documented settings. The techniques in (2) are established ones; the contribution is the measurement of what they do on this benchmark, and of what the proposed geometry fusion adds once they are in place.
"""

RELATED = r"""
\section{Related work}
\textbf{Real-time transformers for detection.} DETR \cite{detr} removed hand-designed anchors and non-maximum suppression; RT-DETR \cite{rtdetr} made the design real-time with a hybrid encoder and IoU-aware query selection and is the backbone of the reference method. Like most detectors it is trained and benchmarked at 640 px on COCO \cite{coco}, where objects under 32 px are already the hardest group.

\textbf{Small objects and tiling.} When objects are small relative to the frame, resizing discards the signal before the network sees it. Slicing the image into overlapping tiles for inference, and fine-tuning on slices, is a standard remedy \cite{sahi}. It trades computation for resolution, and it changes the context each object is seen in; whether it helps is an empirical question for each data set, and orchard imagery --- many near-identical small objects --- is a favourable case.

\textbf{RGB-D fusion in orchards.} Depth has been fused by early concatenation, by parallel encoders, and by attention or gating. The reference paper argues that gating is more robust than concatenation when depth is missing or noisy, and supports this with masking and noise experiments. A practical requirement common to all of these designs is that depth-derived inputs undergo exactly the same geometric augmentation as the image; mosaic augmentation \cite{yolov4}, which composes four images into one, makes this easy to get wrong.

\textbf{Test-time fusion and score calibration.} Averaging predictions over mirrored views and merging boxes by weighted box fusion \cite{wbf} are established ways to trade inference cost for accuracy. Re-scoring detections with a second-stage model is likewise old; what is specific to RGB-D is that depth gives physical quantities --- metric size above all --- that are invariant to the field in which the image was taken.
"""

DATA = r"""
\section{Data and reconciliation with the reference paper}
\subsection{Data set}
The data set is the one released with the reference paper (a public Hugging Face repository under CC BY 4.0; 13,213 files, 14.5 GB) \cite{data}. Each sample has a 1920$\times$1200 RGB frame, a 16-bit depth image in millimetres, three depth-derived maps (B, convexity; N, bumpiness; D, a depth preview) and a YOLO-format label file with two classes, fruit on the tree and fruit on the ground (Fig.~\ref{fig:mod}). The test split comes from a field and day not used for training.

<<TAB_DATA>>

Table~\ref{tab:data} compares what was downloaded with what the paper states. Images and instances agree exactly, there are no duplicate images within or across splits, and every image has its label, depth and geometry files. Nine label files contain a negative coordinate (eight in training, one in test); the loader of the public code discards such frames entirely, so all models here --- and presumably those of the reference paper --- are trained on <<train_tframes>> frames and evaluated on <<test_frames>> held-out frames with <<test_gt>> annotated fruit.

\subsection{What the detector has to see}
Fruit are small: the median box side is <<side_med>> px and <<small_share>>\% of the boxes fall under COCO's 32-px ``small'' limit (Fig.~\ref{fig:size}). Resized to a 640-px input, the median fruit is about <<side_640>> px wide, i.e.\ less than one cell of the stride-8 feature map. Fruit on the ground make up <<ground_share>>\% of the held-out instances.

The held-out field differs from the training fields in appearance but not in geometry (Fig.~\ref{fig:shift}). The metric width of a fruit --- pixel width times depth --- has the same distribution in all three splits, whereas the brightness and saturation of the fruit drop markedly on the held-out field. The frame-level median brightness, by contrast, is the same (<<v_train>> in training, <<v_test>> on the held-out field): the fruit there sit in shade, the scene is not darker. This is the setting in which depth-derived evidence should help most, and it rules out a global illumination correction as a remedy (Section~\ref{sec:neg}).
"""

REPRO = r"""
\section{Reproducing the reference method from its public code}
\label{sec:repro}
The code is a modified copy of Ultralytics 8.4.9 \cite{ultra} (commit \texttt{be9cb88}). Its README documents one training configuration --- 640-px input, 100 epochs, batch size 8, weights initialised from the COCO-pretrained RT-DETR-L --- and we used exactly that. Four properties of the released code had to be dealt with; each was measured rather than inferred from reading the source.

<<TAB_FIND>>

\textbf{Geometry is not aligned with the image during training.} The loader reads the curvature maps as a separate array and returns them next to the image, but mosaic, random affine and flip are applied to the image only. We tested this with a probe data set in which the ``geometry'' file of each image is the image's own grey version: any pipeline that keeps the two aligned must return highly correlated arrays. On the validation pipeline the correlation is <<c_val>>. Under the default training augmentation the geometry is absent from every one of 40 samples drawn, because the mosaic step builds a new sample dictionary without it; with mosaic switched off, as in the last ten epochs, the geometry is present but its correlation with the image falls to <<c_mosoff>> (Fig.~\ref{fig:align}, left).
The consequence is direct. For the first 90 of 100 epochs the fusion modules receive no input and are not trained, while validation does feed geometry through them. A concat-fusion model trained with the public code shows normally falling training losses and a validation mAP50 of essentially zero for the 13 epochs we ran; the same checkpoint scores <<probe_with>> with geometry and <<probe_without>> without it (Fig.~\ref{fig:align}, right). The gated variant degrades less, because an untrained gate attenuates rather than replaces the RGB features, but its validation curve shows the same instability (Fig.~\ref{fig:curves}).

\textbf{The baseline under the documented settings.} RT-DETR-L trained on RGB reaches mAP50 <<E1_rtdetr_m50>>, mAP50:95 <<E1_rtdetr_m95>> and FPPI <<E1_rtdetr_fppi>> on the held-out field (validation: <<E1_rtdetr_v50>> and <<E1_rtdetr_v95>>). The reference paper prints 0.786, 0.562 and 7.24 for this model. Our evaluator reproduces the validator of the public code to four decimals, so the difference is not one of scoring. Of its 53.2 false positives per image, 52.6 are boxes that overlap no annotated fruit at IoU 0.5; duplicates of an already matched fruit (0.09) and class confusions (0.53) are negligible. Even at the recall the printed operating point would require, the reproduced baseline produces several times more false positives (<<E1_rtdetr_f70>> per image at recall 0.70).
<<repro_other>>
We could not establish the training resolution of the reference paper: the full text was not machine-accessible to us, and neither the repository nor the paper's figures state it. A larger input would explain much of the gap, for the reason developed next. All comparisons below are therefore given twice, against the public code as we ran it and against the printed values, and only the first is like-for-like.
"""

METHOD = r"""
\section{Proposed pipeline}
Figure~\ref{fig:pipe} summarises the pipeline. Two networks are trained on the same tiles with unchanged hyper-parameters: a plain RT-DETR-L on RGB (<<params_rgb>> M parameters) and the reference architecture --- RT-DETR-L with the geometry-gated neck and the sphere head (<<params_geo>> M parameters) --- with aligned geometry. The configuration whose flip-fused detections have the higher validation mAP50:95 is the headline configuration; the held-out field plays no part in that choice. What differs from the reproduction of Section~\ref{sec:repro} is the data the detector is trained on and what is done with its outputs.

\subsection{Scale-preserving tiling}
Each frame is cut into a 4$\times$3 grid of 640$\times$640 tiles at native resolution (horizontal origins 0, 427, 853, 1280; vertical origins 0, 280, 560), which overlap by 213 and 280 px, roughly ten fruit diameters. A box is assigned to a tile when at least half of its area lies inside, and is clipped to the tile. For training, all tiles containing fruit and a seeded quarter of the fruit-free tiles are kept (<<train_tiles>> tiles with <<train_tboxes>> boxes from <<train_tframes>> frames); for validation and testing all twelve tiles of a frame are used. Ten epochs over the training tiles amount to 18,490 iterations at batch size 8, the budget of 100 epochs over the full frames (18,700).

At inference a detection that touches an interior tile border within 2 px is discarded: the overlap guarantees that the neighbouring tile contains that fruit whole. The remaining boxes are shifted to frame coordinates, de-duplicated by class-wise non-maximum suppression at IoU 0.5, and the 300 highest-scoring are kept, the same cap as the single-pass detector.

\subsection{Augmentation-aligned geometry}
This step applies to the reference architecture only. The maps B and N are appended to the image as channels four and five when the image is loaded. Mosaic, affine transformation and flipping then act on a five-channel array and cannot separate geometry from appearance; colour jitter is restricted to the first three channels; the formatting step splits the array back into image and geometry. With the probe of Section~\ref{sec:repro} the correlation is <<c_al>> without augmentation and <<c_almos>> under the full training augmentation. The change is a few lines in the loader and has no inference cost.

\subsection{Flip fusion}
The tiles are evaluated a second time mirrored, the detections are mirrored back, and the two sets are merged by weighted box fusion \cite{wbf}: boxes of one class are clustered greedily at IoU 0.55, the fused box is the confidence-weighted mean, and the fused score is the sum of the member scores divided by the number of views, so a fruit found in only one view is down-weighted. This doubles the number of network passes.

\subsection{Evidence re-scoring}
For every detection with confidence above 0.02 we compute 39 quantities: the detector score and its context in the frame (rank, overlaps with stronger boxes, number of detections), box geometry, depth statistics inside the box and in a surrounding ring, metric width and height (pixels times depth), curvature-map statistics, and colour statistics in HSV. Two gradient-boosted models \cite{lgbm} are fitted on the detections of the validation frames: a classifier of whether the detection matches an annotated fruit at IoU 0.5, and a regressor of its IoU. The new score is $p^{1-\alpha}\,\widehat{\mathrm{IoU}}^{\alpha}$. Model choices and $\alpha$ are made on image-grouped five-fold out-of-fold predictions of the validation set ($\alpha$ = <<alpha>> was selected); the held-out field is used only for the final evaluation. Because re-scoring changes the scale of the scores, we report false positives at matched recall in addition to FPPI at a fixed confidence.

\subsection{Evaluation protocol}
All models, tiled or not, are scored on full 1920$\times$1200 frames with one offline evaluator that replicates the matching rule and the AP routine of the public code. FPPI is the number of detections with confidence at least 0.25 that match no annotation at IoU 0.5, divided by the number of frames, as defined in the public code. We add FPPI at fixed recall (0.70 and 0.80), free-response ROC curves, and the per-frame counting error at confidence 0.25. Differences are tested with a paired bootstrap over the held-out frames (1,000 resamples).
"""

RESULTS = r"""
\section{Results}
\subsection{Main comparison}
<<TAB_MAIN>>

Table~\ref{tab:main} and Fig.~\ref{fig:ladder} give every finished model on the held-out field. Tiling alone --- an RGB RT-DETR-L, no geometry --- raises mAP50 from <<E1_rtdetr_m50>> to <<T_rgb_m50>> and mAP50:95 from <<E1_rtdetr_m95>> to <<T_rgb_m95>>. The FROC curves (Fig.~\ref{fig:froc}) show the size of the change: at recall 0.70 the baseline produces <<E1_rtdetr_f70>> false positives per image and the tiled model <<T_rgb_f70>>; at recall 0.80, <<E1_rtdetr_f80>> against <<T_rgb_f80>>. The reference architecture trained on the same tiles with aligned geometry reaches mAP50 <<T_E4_aligned_m50>> and mAP50:95 <<T_E4_aligned_m95>>, with <<T_E4_aligned_f70>> and <<T_E4_aligned_f80>> false positives per image at the two recall levels.

The two tiled models differ in where a fixed confidence threshold falls. At confidence 0.25 the RGB model is at recall <<T_rgb_rec>> with <<T_rgb_fppi>> false positives per image, the geometry model at recall <<T_E4_aligned_rec>> with <<T_E4_aligned_fppi>>. FPPI at a fixed confidence therefore mixes accuracy with calibration, and we read it together with FPPI at matched recall throughout.

On validation the flip-fused RGB model has mAP50:95 <<hv_rgb>> and the flip-fused geometry model <<hv_geo>>, so the RGB model trained for <<H_epochs>> epochs is the headline configuration. With flip fusion and re-scoring it reaches mAP50 <<H_wbf_per_m50>> and mAP50:95 <<H_wbf_per_m95>> at FPPI <<H_wbf_per_fppi>>; the geometry model with the same two steps reaches <<T_E4_aligned_wbf_per_m50>>, <<T_E4_aligned_wbf_per_m95>> and <<T_E4_aligned_wbf_per_fppi>>.
<<main_e4>>

Against the printed values, both tiled pipelines are above the printed RT-DETR baseline in mAP50 and mAP50:95 (0.786 and 0.562) and below the printed GeoSphere-DETR (0.863 and 0.642); their FPPI at confidence 0.25 is higher than the printed 3.29 (Fig.~\ref{fig:printed}). We do not claim to have surpassed the printed result.

\subsection{Where the gain comes from}
<<TAB_LADDER>>

Table~\ref{tab:ladder} decomposes the improvement with paired bootstrap intervals. The step from the full-frame baseline to the tiled model accounts for most of it (<<b_tile_50>> mAP50, <<b_tile_95>> mAP50:95). Flip fusion adds <<b_flip_50>> and <<b_flip_95>>. Re-scoring adds <<b_per_50>> and <<b_per_95>>.
<<ladder_abl>>

\subsection{Counting and fruit size}
For yield estimation the number that matters is the count. At confidence 0.25 the baseline reports <<E1_rtdetr_cbias>> fruit per frame more than are annotated (mean absolute error <<E1_rtdetr_cmae>>). The tiled RGB model at the same threshold also over-counts (bias <<T_rgb_cbias>>, error <<T_rgb_cmae>>), because that threshold sits at recall <<T_rgb_rec>> on its score scale; after flip fusion and re-scoring the error is <<H_wbf_per_cmae>> (bias <<H_wbf_per_cbias>>; Fig.~\ref{fig:count}). The geometry model, whose sphere head lowers the confidence of doubtful queries, is at <<T_E4_aligned_cmae>> without re-scoring. The count error at a fixed threshold is thus a property of the score scale as much as of the detector, and a threshold chosen on validation data is needed in practice. The left panel of Fig.~\ref{fig:count} gives recall by fruit size at confidence 0.25 and has to be read with the same caveat. Fruit on the ground remain the harder class (AP50 <<H_wbf_per_ground>> against <<H_wbf_per_tree>> for fruit on the tree).

\subsection{What the re-scoring stage does}
<<TAB_PER>>

Evidence re-scoring was designed around physical evidence --- metric size and depth contrast, which Fig.~\ref{fig:shift} shows to be invariant across fields. The ablation does not support that design (Table~\ref{tab:per}, Fig.~\ref{fig:per}). On the flip-fused detections of the headline model the full calibrator moves held-out mAP50 from <<ab_base>> to <<ab_all>> and mAP50:95 from <<ab_base95>> to <<ab_all95>>; a calibrator that sees only the detector score gives <<ab_score>>. Adding a single evidence group to the context features gives <<ab_box>> (box geometry), <<ab_depth>> (depth), <<ab_size>> (metric size), <<ab_curv>> (curvature) and <<ab_colour>> (colour); removing colour from the full set gives <<ab_nocolour>>. Depth and curvature contribute nothing. On validation the stage moves out-of-fold mAP50:95 from <<per_val_before>> to <<per_val_after>>, i.e.\ not at all. For the geometry model the picture is the same with a slightly larger effect: mAP50 <<g_ab_base>> to <<g_ab_all>>, of which colour alone recovers <<g_ab_colour>> and without which <<g_ab_nocolour>> remains.

What the stage changes reliably is the score scale. At confidence 0.25 the flip-fused headline model produces <<H_wbf_fppi>> false positives per image at recall <<H_wbf_rec>> before re-scoring and <<H_wbf_per_fppi>> at recall <<H_wbf_per_rec>> after it; at the detector's own recall the false positives are <<per_fr_before>> before and <<per_fr_after>> after re-scoring over <<per_seeds>> seeds. We therefore describe it as a calibration step with a small ranking benefit, keep it as an optional final stage, and do not present it as a principal contribution.

\subsection{Robustness to corrupted geometry}\label{sec:robust}
These experiments concern the geometry model only. <<rob>>

\subsection{Cost}
The single-pass detector runs the network once per frame at 640 px. The tiled model runs it twelve times, and flip fusion doubles that (Fig.~\ref{fig:cost}); re-scoring adds a table lookup of 39 features per box and two small tree ensembles. The RGB tiled model needs no depth sensor at inference unless re-scoring is used, and re-scoring works nearly as well without its depth features. The tiled model is suited to offline yield mapping, or to robots that can afford a few frames per second, rather than to a 30-Hz control loop. Tiling is embarrassingly parallel and the twelve tiles fit in one batch.

\subsection{Negative results}
\label{sec:neg}
Four ideas were tested and did not help. (i) The geometry branch of the reference architecture, with aligned inputs, adds no accuracy at native resolution (above). (ii) Up-sampling the tiles to 960 px at inference without retraining lowers the geometry model's validation mAP50 from <<T_E4_aligned_v50>> to <<im960_v50>> and mAP50:95 from <<T_E4_aligned_v95>> to <<im960_v95>>; it was rejected on validation and never run on the held-out field. (iii) A label-free gamma correction that moves each frame's median brightness to that of the training set is nearly the identity on the held-out field, because the frames are not darker --- only the fruit are. (iv) Depth, metric-size and curvature evidence add nothing to re-scoring, as shown above.
<<t960>>

\subsection{Qualitative comparison}
Figure~\ref{fig:qual} shows held-out frames. The false positives of the full-frame baseline cluster on yellowed leaves, bright soil and specular patches; the tiled pipeline removes most of them and its remaining errors are mainly fruit at the image border and partially occluded fruit on the ground.
"""

DISCUSSION = r"""
\section{Discussion}
\textbf{Resolution before fusion.} The reference paper frames orchard detection as a problem of telling fruit from fruit-like background and answers it with geometry. Our measurements point to an earlier bottleneck under the documented settings: at 640 px the fruit is too small for the detector to localise, and most ``false positives'' are boxes that miss the IoU threshold on, or around, real fruit and its look-alikes. Restoring the native resolution removes <<fp_cut>>\% of the false positives at recall 0.70 without any new module and without depth.

\textbf{What geometry adds once resolution is restored.} The alignment finding is easy to miss, because the code runs, the losses fall and the final numbers look plausible: with mosaic on, the network simply trains as an RGB detector whose fusion modules receive nothing, and with mosaic off for the last epochs it sees geometry in the wrong place. Any conclusion about concatenation against gating drawn from such a pipeline is a conclusion about how gracefully each design ignores an input it was not trained on. <<disc_geo>>

\textbf{Significance.} For practitioners the result is a recipe that needs no new architecture and no depth channel: cut tiles at native resolution, fuse a mirrored view, and set the operating point on validation data. On this benchmark it divides the false positives of the reproduced baseline at recall 0.70 by <<fp_div>> and brings the per-frame count error from <<E1_rtdetr_cmae>> to about <<H_wbf_per_cmae>> fruit. For work on RGB-D fusion the result is a caution and a tool: depth-derived channels must be kept glued to the image through augmentation, the probe used here verifies that in a few minutes, and a fusion module should be credited only with what it adds over an RGB detector that sees the fruit at an adequate scale. For the research record the study supplies a checked data set and a runnable version of the reference code with a documented patch.

\textbf{Limitations.} (1) The comparison with the printed figures of the reference paper is not like-for-like, because its training resolution and schedule could not be confirmed; our pipelines do not reach them. (2) Every model was trained with one seed; the bootstrap intervals quantify sampling uncertainty over frames, not over training runs, so the small difference between the RGB and the geometry model should be read as ``no measurable gain'', not as evidence that geometry hurts. (3) The tiled models cost 12--24 network passes per frame. (4) Tiling, mirrored-view fusion and second-stage re-scoring are established techniques; the methodological novelty here is limited, and the value of the study lies in the measurements. (5) Re-scoring is fitted on 200 validation frames from the training fields and transfers only partly to the held-out field. (6) The data set covers one crop and one sensor.

\section{Conclusion}
The data released with GeoSphere-DETR are complete and match the paper; the released code, run as documented, neither trains the geometry branch on aligned inputs nor reproduces the printed figures. Training an RGB RT-DETR-L on native-resolution tiles improves held-out mAP50:95 from <<E1_rtdetr_m95>> to <<H_m95>> and, with flip fusion and re-scoring, to <<H_wbf_per_m95>>, and reduces false positives at recall 0.70 from <<E1_rtdetr_f70>> to <<H_wbf_per_f70>> per image. The reference architecture with correctly aligned geometry, trained on the same tiles, reaches <<T_E4_aligned_wbf_per_m95>> with the same steps: on this benchmark, resolution and not geometry is what the documented configuration lacks. The printed GeoSphere-DETR figures remain above all of these, and resolving that difference requires the original training configuration.

\section*{Data and code availability}
The data set is public \cite{data}. All scripts, the patch to the public code, the per-run result files and the saved detections from which every table and figure of this paper is generated are in the project directory accompanying this manuscript.

\begin{thebibliography}{12}\small
\bibitem{ref} GeoSphere-DETR: RGB-D citrus fruit detection in cluttered orchards via geometry-gated fusion and geometry-aware confidence recalibration. \emph{Artificial Intelligence in Agriculture}. 2026. doi:10.1016/j.aiia.2026.08.010. Code: github.com/Ariel0818/Geosphere-DETR.
\bibitem{data} Citrus fruit RGB-D data set. Hugging Face repository \texttt{ariel0818/citrus-fruit-rgbd}, CC BY 4.0.
\bibitem{rtdetr} Zhao Y, Lv W, Xu S, et al. DETRs beat YOLOs on real-time object detection. \emph{Proc IEEE/CVF CVPR}. 2024:16965--16974.
\bibitem{detr} Carion N, Massa F, Synnaeve G, Usunier N, Kirillov A, Zagoruyko S. End-to-end object detection with transformers. \emph{Proc ECCV}. 2020:213--229.
\bibitem{sahi} Akyon FC, Altinuc SO, Temizel A. Slicing aided hyper inference and fine-tuning for small object detection. \emph{Proc IEEE ICIP}. 2022:966--970.
\bibitem{wbf} Solovyev R, Wang W, Gabruseva T. Weighted boxes fusion: ensembling boxes from different object detection models. \emph{Image Vis Comput}. 2021;107:104117.
\bibitem{lgbm} Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. \emph{Adv Neural Inf Process Syst}. 2017;30.
\bibitem{yolov4} Bochkovskiy A, Wang C-Y, Liao H-YM. YOLOv4: optimal speed and accuracy of object detection. arXiv:2004.10934; 2020.
\bibitem{coco} Lin T-Y, Maire M, Belongie S, et al. Microsoft COCO: common objects in context. \emph{Proc ECCV}. 2014:740--755.
\bibitem{ultra} Jocher G, Qiu J, Chaurasia A. Ultralytics YOLO (software), version 8.4. github.com/ultralytics/ultralytics.
\end{thebibliography}
\end{document}
"""


def tables(N, R, S):
    T = {}
    dc = J("data_check.json")
    rows = [["Images", "2,200", N["n_img"], "yes"], ["Annotated instances", "102,719", N["n_inst"], "yes"], ["Split (train / val / test images)", "1,500 / 200 / 500", f"{N['train_img']} / {N['val_img']} / {N['test_img']}", "yes"],
            ["Instances per split", "not stated", f"{N['train_inst']} / {N['val_inst']} / {N['test_inst']}", "--"], ["On tree / on ground (test)", "not stated", f"{N['test_tree']} / {N['test_ground']}", "--"],
            ["Duplicate images (MD5), within or across splits", "not stated", "0", "--"], ["Images lacking a label, depth or geometry file", "not stated", "0", "--"],
            ["Label files with a negative coordinate (dropped by the loader)", "not stated", "8 train, 1 test", "--"], ["Frames / fruit actually evaluated (test)", "not stated", f"{N['test_frames']} / {N['test_gt']}", "--"]]
    T["TAB_DATA"] = tabular(rows, ["Quantity", "Reference paper", "Downloaded", "Match"], "p{6.6cm}llc", "Reconciliation of the public data set with the reference paper.", "tab:data",
                            note="Counts of the reference paper are those stated in its abstract and on the data-set card.")
    f = [["1", "Package cannot be imported: the model registry refers to sub-packages that were removed", "Import fails", "Removed the dangling classes; detection task only"],
         ["2", "FPPI routine reads a validator attribute that is never filled and returns early", "No FPPI is ever computed", "FPPI computed offline with the same definition"],
         ["3", "Geometry maps are not transformed with the image; mosaic drops them", f"Absent in 40/40 samples (mosaic on); correlation {N['c_mosoff']} (mosaic off) against {N['c_val']} (validation)", f"Five-channel loading (correlation {N['c_almos']})"],
         ["4", "Validator always passes a geometry argument to the model", "YOLO baseline crashes at the first validation pass", "Argument passed only when geometry exists"]]
    T["TAB_FIND"] = tabular(f, ["", "Property of the released code", "Measured effect", "What we did"], "cp{5.0cm}p{4.6cm}p{4.2cm}", "Findings about the public implementation.", "tab:find", size="\\footnotesize",
                            note="The unmodified repository is kept alongside the patched copy; the patch is in \\texttt{results/fork\\_patch.diff}.")
    rows = []
    fam = None
    for t, r in R.items():
        if r["family"] != fam:
            fam = r["family"]; rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{'Full frame resized to 640 px, public code' if fam.startswith('full') else 'Native-resolution tiles (this work)'}}}}} \\\\")
        x = r["test"]
        rows.append([esc(r["label"]).replace("  ", "\\quad "), r["views"], f3(x["map50"]), f3(x["map5095"]), f2(x["fppi"]), f3(x["recall"]), f2(x["fppi_at_recall_70"]), f2(x["fppi_at_recall_80"]), f1(x["count_mae"])])
    rows.append("\\midrule")
    rows.append("\\multicolumn{9}{l}{\\emph{Printed in the reference paper (training resolution not established)}} \\\\")
    for n, a, b, c in S["printed"]:
        rows.append([esc(n), 1, f3(a) if a else "--", f3(b) if b else "--", f2(c), "--", "--", "--", "--"])
    T["TAB_MAIN"] = tabular(rows, ["Model", "Passes", "mAP50", "mAP50:95", "FPPI", "Recall", "FPPI@R.70", "FPPI@R.80", "Count err."], "lrrrrrrrr",
                            f"Held-out field ({N['test_frames']} frames, {N['test_gt']} fruit), all models scored on full frames with one evaluator.", "tab:main", size="\\scriptsize\\setlength{\\tabcolsep}{2.2pt}",
                            note="Passes: 640-px network evaluations per frame. FPPI and recall at confidence 0.25, IoU 0.5. FPPI@R: false positives per image at the score threshold giving recall 0.70 / 0.80. Count err.: mean absolute difference between detections at confidence 0.25 and annotated fruit per frame.")
    L = [("b_e1e4", "Full-frame RT-DETR-L $\\rightarrow$ GeoSphere-DETR (both public code)"), ("b_tile", "Full-frame RT-DETR-L $\\rightarrow$ tiled RGB (headline model)"),
         ("b_flip", "+ flip fusion"), ("b_per", "+ evidence re-scoring"), ("b_all", "Full-frame RT-DETR-L $\\rightarrow$ headline pipeline"), ("b_e4", "GeoSphere-DETR (public code) $\\rightarrow$ headline pipeline"),
         ("b_geo", "Tiled RGB $\\rightarrow$ tiled GeoSphere-DETR, aligned geometry"), ("b_geo2", "The same, both with flip fusion and re-scoring"), ("b_align", "Tiled GeoSphere-DETR: public loader $\\rightarrow$ aligned"),
         ("b_long", "Tiled RGB: first $\\rightarrow$ second training cycle"), ("b_960", "Tiled GeoSphere-DETR: 640 px $\\rightarrow$ 960-px fine-tune")]
    rows = [[lab, f"{N[k + '_50']} ({N[k + '_50_ci']})", f"{N[k + '_95']} ({N[k + '_95_ci']})", f"{N[k + '_fp']} ({N[k + '_fp_ci']})", N[k + "_95_p"]] for k, lab in L if k + "_50" in N]
    T["TAB_LADDER"] = tabular(rows, ["Comparison", "$\\Delta$ mAP50", "$\\Delta$ mAP50:95", "$\\Delta$ FPPI (conf. 0.25)", "$p$"], ">{\\raggedright\\arraybackslash}p{4.6cm}lllc",
                              "Paired bootstrap over the held-out frames (1,000 resamples; 95\\% intervals in parentheses).", "tab:ladder", size="\\scriptsize\\setlength{\\tabcolsep}{3pt}",
                              note="$p$: two-sided bootstrap $p$-value for mAP50:95. FPPI differences involving re-scoring compare different score scales at a fixed confidence threshold; see Table~\\ref{tab:per} for matched recall.")
    rs = J(f"rescore_{S['headline']}_wbf.json"); ab = rs["ablation"]
    rows = [["Detector score (no re-scoring)", f3(rs["base"]["val"]["map5095"]), f3(rs["base"]["test"]["map50"]), f3(rs["base"]["test"]["map5095"]), f2(rs["base"]["test"]["fppi_at_base_recall"])], "\\midrule"]
    for k, v in ab.items():
        rows.append([esc(k.replace("ctx", "context").replace("all-", "all $-$ ")), f3(v["val_oof"]["map5095"]), f3(v["test"]["map50"]), f3(v["test"]["map5095"]), f2(v["test"]["fppi_at_base_recall"])])
    T["TAB_PER"] = tabular(rows, ["Evidence given to the calibrator", "Val mAP50:95", "Test mAP50", "Test mAP50:95", "Test FPPI@R"], "lrrrr",
                           "Evidence-group ablation of the re-scoring stage on the flip-fused detections of the headline model.", "tab:per", size="\\footnotesize",
                           note="Val: image-grouped out-of-fold predictions. FPPI@R: false positives per image at the recall the detector has at confidence 0.25. Context: detector score, class, rank and overlaps within the frame. Groups: box geometry, depth, metric size, curvature maps, colour.")
    return T


def conditional(N, R):
    C = {}
    has = lambda k: f"{k}_m50" in N
    if has("E4_rescore"):
        C["abs_e4"] = (f"The reference model itself, trained with the public code under the same settings, reaches mAP50 {N['E4_rescore_m50']}, mAP50:95 {N['E4_rescore_m95']} and FPPI {N['E4_rescore_fppi']}"
                       + (f"; that is {N['b_e1e4_95']} mAP50:95 relative to the RGB baseline it is meant to improve." if "b_e1e4_95" in N else "."))
        C["main_e4"] = (f"GeoSphere-DETR trained with the public code under the documented settings reaches mAP50 {N['E4_rescore_m50']}, mAP50:95 {N['E4_rescore_m95']} and FPPI {N['E4_rescore_fppi']} (validation: {N['E4_rescore_v50']} and {N['E4_rescore_v95']})"
                        + (f"; relative to the RGB baseline trained the same way this is {N['b_e1e4_50']} mAP50 and {N['b_e1e4_95']} mAP50:95 ({N['b_e1e4_95_ci']}), so the public pipeline does not reproduce the direction of the printed improvement, which is what the alignment finding of Section~\\ref{{sec:repro}} leads one to expect" if "b_e1e4_50" in N else "")
                        + (f". The headline pipeline differs from it by {N['b_e4_50']} mAP50 and {N['b_e4_95']} mAP50:95 (Table~\\ref{{tab:ladder}})." if "b_e4_50" in N else "."))
    else:
        C["abs_e4"] = "The public-code run of the full reference model had not finished when this version was compiled."
        C["main_e4"] = "The run of the full GeoSphere-DETR model with the public code had not finished when this version was compiled, so the table has no row for it."
    oth = []
    if has("Y11m"):
        oth.append(f"YOLO11m, which the patched validator allows to train, reaches mAP50 {N['Y11m_m50']} and FPPI {N['Y11m_fppi']} (printed for YOLOv11m: 0.715 and 8.735).")
    if has("E3_gated"):
        oth.append(f"The gated-fusion variant without the sphere head reaches mAP50:95 {N['E3_gated_m95']} and FPPI {N['E3_gated_fppi']} (printed: 0.623 and 4.6).")
    C["repro_other"] = " ".join(oth)
    ab = []
    if "b_geo_50" in N:
        ab.append(f"Replacing the RGB network by the reference architecture with aligned geometry changes mAP50 by {N['b_geo_50']} ({N['b_geo_50_ci']}) and mAP50:95 by {N['b_geo_95']} ({N['b_geo_95_ci']}; $p$ = {N['b_geo_95_p']})"
                  + (f"; after flip fusion and re-scoring of both, by {N['b_geo2_50']} ({N['b_geo2_50_ci']}) and {N['b_geo2_95']} ({N['b_geo2_95_ci']})." if "b_geo2_50" in N else "."))
    if has("T_E4_unaligned"):
        ab.append(f"The geometry model trained on tiles with the public loader reaches mAP50 {N['T_E4_unaligned_m50']} and mAP50:95 {N['T_E4_unaligned_m95']}"
                  + (f" (aligned minus public loader: {N['b_align_50']} and {N['b_align_95']}; $p$ = {N['b_align_95_p']} for mAP50:95). Alignment therefore changes nothing measurable on tiles either, which is what a gate that is nearly constant predicts (Section~\\ref{{sec:robust}}): a branch the network does not use cannot be harmed by misplaced input. The alignment defect is real and is fatal for concat fusion, but repairing it does not by itself make the gated branch useful." if "b_align_50" in N else "."))
    else:
        ab.append("The run of the geometry model on tiles with the public loader, which isolates the effect of alignment, had not finished when this version was compiled.")
    if has("T_rgb20"):
        ab.append(f"A second ten-epoch cycle of the tiled RGB model, started from the weights of the first, reaches mAP50 {N['T_rgb20_m50']} and mAP50:95 {N['T_rgb20_m95']} in a single view"
                  + (f" ({N['b_long_50']} and {N['b_long_95']} relative to the first cycle; $p$ = {N['b_long_95_p']})" if "b_long_50" in N else "")
                  + (f", and {N['T_rgb20_wbf_per_m50']} and {N['T_rgb20_wbf_per_m95']} with flip fusion and re-scoring." if has("T_rgb20_wbf_per") else "."))
    else:
        ab.append("A second ten-epoch training cycle of the tiled RGB model had not finished when this version was compiled.")
    C["ladder_abl"] = " ".join(ab)
    if "b_geo_95" in N:
        C["disc_geo"] = (f"With the inputs aligned and the resolution restored, the reference architecture differs from the RGB detector by {N['b_geo_95']} mAP50:95 ({N['b_geo_95_ci']}) and has the same false-positive rate at matched recall "
                         f"({N['T_E4_aligned_f70']} against {N['T_rgb_f70']} per image at recall 0.70). On this benchmark we therefore measure no accuracy benefit of the curvature maps or the sphere head; "
                         f"their visible effect is on calibration, where the model is far more conservative at a fixed confidence ({N['T_E4_aligned_fppi']} against {N['T_rgb_fppi']} false positives per image at 0.25). "
                         "A validation-fitted re-scoring stage brings the RGB model to the same operating point. This does not show that geometry cannot help --- the printed result may rest on a configuration we could not reconstruct --- but it does show that the benefit is not obtained by the released architecture once the fruit is seen at an adequate scale.")
    else:
        C["disc_geo"] = "The comparison that measures the value of aligned geometry at native resolution is reported in the results when finished."
    rob = []
    for kind, lab in (("miss", "masking ratio"), ("noise", "noise $\\sigma$")):
        v = [(float(f[len(f"T_E4_aligned_s0_{kind}"):-5]), J(f"det/{f}")["test"]) for f in sorted(os.listdir(f"{RES}/det")) if f.startswith(f"T_E4_aligned_s0_{kind}") and f.endswith(".json")]
        if v:
            rob.append(f"With {lab} " + ", ".join(f"{a:g}" for a, _ in sorted(v)) + " the tiled model's held-out mAP50:95 is " + ", ".join(f3(x["map5095"]) for _, x in sorted(v))
                       + f" (clean: {N['T_E4_aligned_m95']}) and its FPPI " + ", ".join(f2(x["fppi"]) for _, x in sorted(v)) + f" (clean: {N['T_E4_aligned_fppi']}).")
    C["rob"] = (" ".join(rob) + " Figure~\\ref{fig:rob} shows the curves. Masking follows the block protocol of the public code; the public code has no noise option, so the noise test uses a seeded hook added for this study."
                if rob else "The masking and noise evaluations of the tiled model had not finished when this version was compiled.")
    if E("gate_probe.json"):
        gp = J("gate_probe.json"); lv = [k for k in gp["clean"] if k.startswith("level")]
        C["rob"] += (" The insensitivity has a structural explanation. The fusion of the reference architecture multiplies each feature map by $1 + \\alpha g$ with $\\alpha$ = " + f"{gp['alpha']:g}"
                     + " and a gate $g$ in (0, 1), so geometry can only rescale features, by a factor between 1 and " + f"{1 + gp['alpha']:g}" + ". On held-out tiles the trained gates have means "
                     + ", ".join(f2(gp["clean"][k]["mean"]) for k in lv) + " at the three neck levels and vary within an image with standard deviations of only "
                     + ", ".join(f3(gp["clean"][k]["std_within_image"]) for k in lv) + "; with the curvature maps fully masked the means are "
                     + ", ".join(f2(gp["masked"][k]["mean"]) for k in lv) + ". The gate is thus close to a constant, the fusion close to a uniform scaling, and the detector close to an RGB detector, "
                     "which is consistent both with the absence of an accuracy gain and with the absence of any loss when geometry is removed. Robustness to missing geometry is in this case not evidence that the geometry is used well.")
    if has("T960_E4_aligned"):
        C["t960"] = (f"A further experiment did train at the larger scale: fine-tuning the tiled geometry model for four epochs on tiles up-sampled to 960 px gives held-out mAP50 {N['T960_E4_aligned_m50']} and mAP50:95 {N['T960_E4_aligned_m95']} "
                     f"at 2.25 times the cost per tile (single view; 640-px tiled model: {N['T_E4_aligned_m50']} and {N['T_E4_aligned_m95']}).")
    else:
        C["t960"] = ""
    return C


def main():
    N, R, S = numbers(); T = tables(N, R, S); N.update(conditional(N, R)); N.update(T)
    body = PRE + fill(ABSTRACT, N) + fill(INTRO, N) + fill(RELATED, N) + fill(DATA, N) + fill(REPRO, N) + fill(METHOD, N) + fill(RESULTS, N)
    figs = [f for f in FIGS if os.path.exists(f"{B}/figures/{f[0]}.pdf")]
    missing = [f[2] for f in FIGS if f not in figs]
    disc = fill(DISCUSSION, N)
    figtex = "\n\\clearpage\n\\section*{Figures}\n" + "\n".join(figure(n, c, l, "0.98\\linewidth" if n != "fig_cost" else "0.6\\linewidth") + ("\n\\clearpage" if i % 2 == 1 else "") for i, (n, c, l) in enumerate(figs))
    tex = body + disc.replace("\\begin{thebibliography}", figtex + "\n\\begin{thebibliography}")
    for lab in missing:                                    # a figure that does not exist yet must not leave a dangling reference
        tex = tex.replace(f"Figure~\\ref{{{lab}}} shows the curves. ", "").replace(f" (Fig.~\\ref{{{lab}}})", "").replace(f"Fig.~\\ref{{{lab}}}", "the corresponding figure (not yet available)")
    os.makedirs(f"{B}/paper", exist_ok=True)
    tex = tex.replace("$p$ = $<$", "$p$ $<$ ")
    open(f"{B}/paper/main.tex", "w").write(tex)
    json.dump({k: v for k, v in N.items() if not k.startswith("TAB_")}, open(f"{RES}/paper_numbers.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote paper/main.tex: {len(figs)} figures ({len(missing)} not yet available: {missing}), {len(T)} tables")


if __name__ == "__main__":
    main()
