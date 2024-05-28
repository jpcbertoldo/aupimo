"""Evaluate the anomaly score maps (asmaps) of a model on a dataset.

Important: overwritting output files is the default behavior.
"""
#!/usr/bin/env python

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import json
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
from anomalib.metrics import AUPR, AUPRO, AUROC
from PIL import Image
from torch import Tensor

# is it running as a notebook or as a script?
if (arg0 := Path(sys.argv[0]).stem) == "ipykernel_launcher":
    print("running as a notebook")
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell

    IS_NOTEBOOK = True

    # autoreload modified modules
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    # make a cell print all the outputs instead of just the last one
    InteractiveShell.ast_node_interactivity = "all"
    # show all warnings
    warnings.filterwarnings("always", category=Warning)

else:
    IS_NOTEBOOK = False


from aupimo import AUPIMOResult, aupimo_scores, per_image_iou_curves
from aupimo.oracles import IOUCurvesResult, max_avg_iou, max_iou_per_image

# %%
# Args

# collection [of datasets] = {mvtec, visa}
ACCEPTED_COLLECTIONS = {
    (MVTEC_DIR_NAME := "MVTec"),
    (VISA_DIR_NAME := "VisA"),
}

METRICS_CHOICES = [
    (METRIC_AUROC := "auroc"),
    (METRIC_AUPR := "aupr"),
    (METRIC_AUPRO := "aupro"),
    (METRIC_AUPIMO := "aupimo"),
    # =============================================================================
    # --------- iou ---------
    # curves
    (METRIC_IOU_CURVES_GLOBAL := "ioucurves_global"),
    (METRIC_IOU_CURVES_LOCAL := "ioucurves_local"),
    # oracle threshs
    (METRIC_MAX_AVG_IOU := "max_avg_iou"),
    (METRIC_MAX_IOU_PER_IMG := "max_iou_per_img"),
    (METRIC_MAX_AVG_IOU_MIN_THRESH := "max_avg_iou_min_thresh"),
    (METRIC_MAX_IOU_PER_IMG_MIN_THRESH := "max_iou_per_img_min_thresh"),
]

parser = argparse.ArgumentParser()
_ = parser.add_argument("--asmaps", type=Path, required=True)
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
_ = parser.add_argument("--not-debug", dest="debug", action="store_false")
_ = parser.add_argument("--metrics", "-me", type=str, action="append", choices=METRICS_CHOICES, default=[])
_ = parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--asmaps ../data/experiments/benchmark/patchcore_wr50/mvtec/bottle/asmaps.pt",
                # "--metrics auroc",
                # "--metrics aupr",
                # "--metrics aupro",
                # "--metrics aupimo",
                # "--metrics ioucurves_global",
                # "--metrics ioucurves_local",
                "--metrics max_avg_iou",
                "--metrics max_iou_per_img",
                "--metrics max_avg_iou_min_thresh",
                "--metrics max_iou_per_img_min_thresh",
                "--mvtec-root ../data/datasets/MVTec",
                "--visa-root ../data/datasets/VisA",
                "--not-debug",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

rundir = args.asmaps.parent

# %%
# Load `asmaps.pt`

print("loading asmaps.pt")

assert args.asmaps.exists(), str(args.asmaps)

asmaps_dict = torch.load(args.asmaps)
assert isinstance(asmaps_dict, dict), f"{type(asmaps_dict)=}"

asmaps = asmaps_dict["asmaps"]
assert isinstance(asmaps, torch.Tensor), f"{type(asmaps)=}"
assert asmaps.ndim == 3, f"{asmaps.shape=}"
print(f"{asmaps.shape=}")

images_relpaths = asmaps_dict["paths"]
assert isinstance(images_relpaths, list), f"{type(images_relpaths)=}"

assert len(asmaps) == len(images_relpaths), f"{len(asmaps)=}, {len(images_relpaths)=}"

# collection [of datasets] = {mvtec, visa}
collection = {p.split("/")[0] for p in images_relpaths}
assert collection.issubset(ACCEPTED_COLLECTIONS), f"{collection=}"

collection = collection.pop()

if collection == MVTEC_DIR_NAME:
    assert args.mvtec_root is not None, "please provide the argument `--mvtec-root`"
    collection_root = args.mvtec_root

if collection == VISA_DIR_NAME:
    assert args.visa_root is not None, "please provide the argument `--visa-root`"
    collection_root = args.visa_root

assert collection_root.exists(), f"{collection=} {collection_root=!s}"

dataset = {"/".join(p.split("/")[:2]) for p in images_relpaths}
assert len(dataset) == 1, f"{dataset=}"

dataset = dataset.pop()
print(f"{dataset=}")

print("sorting images and their asmaps")
images_argsort = np.argsort(images_relpaths)
images_relpaths = np.array(images_relpaths)[images_argsort].tolist()
asmaps = asmaps[images_argsort]

print("getting masks paths from images paths")


def _image_path_2_mask_path(image_path: str) -> str | None:
    if "good" in image_path:
        # there is no mask for the normal images
        return None

    path = Path(image_path.replace("test", "ground_truth"))

    if (collection := path.parts[0]) == VISA_DIR_NAME:
        path = path.with_suffix(".png")

    elif collection == MVTEC_DIR_NAME:
        path = path.with_stem(path.stem + "_mask").with_suffix(".png")

    else:
        msg = f"Unknown collection: {collection=}"
        raise NotImplementedError(msg)

    return str(path)


masks_relpaths = [_image_path_2_mask_path(p) for p in images_relpaths]

print(f"converting relative paths to absolute paths\n{collection_root=!s}")


def _convert_path(relative_path: str, collection_root: Path) -> str | None:
    if relative_path is None:
        return None
    relative_path = Path(*Path(relative_path).parts[1:])
    return str(collection_root / relative_path)


images_abspaths = [_convert_path(p, collection_root) for p in images_relpaths]
masks_abspaths = [_convert_path(p, collection_root) for p in masks_relpaths]

for path in images_abspaths + masks_abspaths:
    assert path is None or Path(path).exists(), path

# %%
# Load masks

print("loading masks")
masks_pils = [Image.open(p).convert("L") if p is not None else None for p in masks_abspaths]

masks_resolution = {p.size for p in masks_pils if p is not None}
assert len(masks_resolution) == 1, f"assumed single-resolution dataset but found {masks_resolution=}"
masks_resolution = masks_resolution.pop()
masks_resolution = (masks_resolution[1], masks_resolution[0])  # [W, H] --> [H, W]
print(f"{masks_resolution=} (HEIGHT, WIDTH)")

masks = torch.stack(
    [
        torch.tensor(np.asarray(pil), dtype=torch.bool)
        if pil is not None
        else torch.zeros(masks_resolution, dtype=torch.bool)
        for pil in masks_pils
    ],
    dim=0,
)
print(f"{masks.shape=}")

# %%
# DEBUG: only keep 2 images per class if in debug mode
if args.debug:
    print("debug mode --> only using 2 images")
    imgclass = (masks == 1).any(dim=-1).any(dim=-1).to(torch.bool)
    NUM_IMG_PER_CLASS = 5
    some_norm = torch.where(imgclass == 0)[0][:NUM_IMG_PER_CLASS]
    some_anom = torch.where(imgclass == 1)[0][:NUM_IMG_PER_CLASS]
    some_imgs = torch.cat([some_norm, some_anom])
    asmaps = asmaps[some_imgs]
    masks = masks[some_imgs]
    images_relpaths = [images_relpaths[i] for i in some_imgs]
    images_abspaths = [images_abspaths[i] for i in some_imgs]
    masks_abspaths = [masks_abspaths[i] for i in some_imgs]

# %%
# Resize asmaps to match the resolution of the masks

asmaps_resolution = asmaps.shape[-2:]
print(f"{asmaps_resolution=} (HEIGHT, WIDTH)")

if asmaps_resolution == masks_resolution:
    print("asmaps and masks have the same resolution")
else:
    print("resizing asmaps to match the resolution of the masks")
    asmaps = torch.nn.functional.interpolate(
        asmaps.unsqueeze(1),
        size=masks_resolution,
        mode="bilinear",
    ).squeeze(1)
    print(f"{asmaps.shape=}")

# %%
# Move data to device

if args.device == "cpu":
    print("using CPU")

elif args.device in ("cuda", "gpu"):
    print("moving data to GPU")
    masks = masks.cuda()
    asmaps = asmaps.cuda()

else:
    msg = f"Unknown device: {args.device=}"
    raise NotImplementedError(msg)


# %%
# Save single-valued metrics


def _save_value(jsonpath: Path, value: float, debug: bool) -> None:
    # prepend `debug_` to the filename if in debug mode
    if debug:
        jsonpath = jsonpath.with_stem("debug_" + jsonpath.stem)
    with Path(jsonpath).open("w") as f:
        json.dump({"value": value}, f, indent=4)


# %%
# AUROC


def _compute_auroc(asmaps: Tensor, masks: Tensor) -> float:
    metric = AUROC()
    metric.update(asmaps, masks)
    return metric.compute().item()


if METRIC_AUROC in args.metrics:
    print("getting auroc")
    auroc = _compute_auroc(asmaps, masks)
    print(f"{auroc=:.2%}")
    _save_value(rundir / "auroc.json", auroc, args.debug)

# %%
# AUPR


def _compute_aupr(asmaps: Tensor, masks: Tensor) -> float:
    metric = AUPR()
    metric.update(asmaps, masks)
    return metric.compute().item()


if METRIC_AUPR in args.metrics:
    print("getting aupr")
    aupr = _compute_aupr(asmaps, masks)
    print(f"{aupr=:.2%}")
    _save_value(rundir / "aupr.json", aupr, args.debug)


# %%
# AUPRO


def _compute_aupro(asmaps: Tensor, masks: Tensor) -> float:
    metric = AUPRO()
    metric.update(asmaps, masks)
    return metric.compute().item()


if METRIC_AUPRO in args.metrics:
    print("computing aupro")
    aupro = _compute_aupro(asmaps, masks)
    print(f"{aupro=}")
    _save_value(rundir / "aupro.json", aupro, args.debug)


# %%
# AUPIMO

if METRIC_AUPIMO in args.metrics:
    print("computing aupimo")

    # TODO(jpcbertoldo): retry with increasing num threshs (up to 1_000_000) GET FROM THE OTHER BRANCH  # noqa: TD003
    pimoresult, aupimoresult = aupimo_scores(
        asmaps,
        masks,
        fpr_bounds=(1e-5, 1e-4),
        paths=images_relpaths,  # relative, not absolute paths!
        num_threshs=30_000,
    )

    print("saving aupimo")

    aupimo_dir = rundir / "aupimo"
    if args.debug:
        aupimo_dir = aupimo_dir.with_stem("debug_" + aupimo_dir.stem)
    aupimo_dir.mkdir(exist_ok=True)

    pimoresult.save(aupimo_dir / "curves.pt")
    aupimoresult.save(aupimo_dir / "aupimos.json")

# %%
# iou curves with shared thresholds

iou_oracle_threshs_dir = rundir / "iou_oracle_threshs"

if METRIC_IOU_CURVES_GLOBAL in args.metrics:
    ioucurves = per_image_iou_curves(asmaps, masks, num_threshs=10_000, common_threshs=True, paths=images_relpaths)
    ioucurves.save(iou_oracle_threshs_dir / "ioucurves_global_threshs.pt")

# %%
# iou curves with local thresholds
if METRIC_IOU_CURVES_LOCAL in args.metrics:
    ioucurves = per_image_iou_curves(asmaps, masks, num_threshs=10_000, common_threshs=False, paths=images_relpaths)
    ioucurves.save(iou_oracle_threshs_dir / "ioucurves_local_threshs.pt")

# %%
# max avg iou withOUT min thresh
if METRIC_MAX_AVG_IOU in args.metrics:
    from aupimo import IOUCurvesResult
    from aupimo.oracles import max_avg_iou

    ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_global_threshs.pt")
    if args.debug:
        ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]

    max_avg_iou_result = max_avg_iou(
        ioucurves.threshs,
        ioucurves.per_image_ious,
        ioucurves.image_classes,
        paths=images_relpaths,
    )
    max_avg_iou_result.save(iou_oracle_threshs_dir / "max_avg_iou.json")


# %%
# max avg iou WITH min thresh


def _get_aupimo_thresh_lower_bound(aupimoresult_fpath: Path) -> float:
    """Threshold lower is FPR upper bound (1e-4)."""
    aupimoresult = AUPIMOResult.load(aupimoresult_fpath)
    return aupimoresult.thresh_lower_bound


_get_aupimo_thresh_lower_bound = partial(
    _get_aupimo_thresh_lower_bound,
    aupimoresult_fpath=rundir / "aupimo" / "aupimos.json",
)

if METRIC_MAX_AVG_IOU_MIN_THRESH in args.metrics:
    ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_global_threshs.pt")
    if args.debug:
        ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]

    max_avg_iou_result = max_avg_iou(
        ioucurves.threshs,
        ioucurves.per_image_ious,
        ioucurves.image_classes,
        paths=images_relpaths,
        min_thresh=_get_aupimo_thresh_lower_bound(),
    )
    max_avg_iou_result.save(iou_oracle_threshs_dir / "max_avg_iou_min_thresh.json")


# %%
# max iou per image withOUT min thresh
if METRIC_MAX_IOU_PER_IMG in args.metrics:
    ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_local_threshs.pt")
    if args.debug:
        ioucurves.threshs = ioucurves.threshs[some_imgs, :]
        ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]

    ious_maxs_result = max_iou_per_image(
        ioucurves.threshs,
        ioucurves.per_image_ious,
        ioucurves.image_classes,
        paths=images_relpaths,
    )
    ious_maxs_result.save(iou_oracle_threshs_dir / "max_iou_per_img.json")


# %%
# max iou per image WITH min thresh

if METRIC_MAX_IOU_PER_IMG_MIN_THRESH in args.metrics:
    from aupimo import IOUCurvesResult
    from aupimo.oracles import max_iou_per_image

    ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_local_threshs.pt")
    if args.debug:
        ioucurves.threshs = ioucurves.threshs[some_imgs, :]
        ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]

    ious_maxs_result = max_iou_per_image(
        ioucurves.threshs,
        ioucurves.per_image_ious,
        ioucurves.image_classes,
        min_thresh=_get_aupimo_thresh_lower_bound(),
        paths=images_relpaths,
    )
    ious_maxs_result.save(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

# %%
# generate superpixels

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk

from aupimo._validate_tensor import safe_tensor_to_numpy

image_idx = 6
asmap = asmaps[image_idx]
mask = masks[image_idx]
if (img := plt.imread(images_abspaths[image_idx])).ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

num_pixels = img.shape[0] * img.shape[1]
makers_pixels_ratio = 0.01
num_markers = int(num_pixels * makers_pixels_ratio)
print(f"{num_pixels=} {makers_pixels_ratio=} {num_markers=}")

suppixs = sk.segmentation.watershed(
    (gradient := sk.filters.sobel(sk.color.rgb2gray(img))),
    markers=num_markers,
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    compactness=0.001,
)
suppixs_labels = sorted(set(np.unique(suppixs.flatten())) - {0})
num_superpixels = len(set(np.unique(suppixs)) - {0})
print(f"number of superpixels: {num_superpixels}")

suppixs_valid = suppixs * (valid_asmap_mask := safe_tensor_to_numpy(asmap) > _get_aupimo_thresh_lower_bound())
num_superpixels_valid = len(set(np.unique(suppixs_valid)) - {0})
print(f"number of superpixels valid: {num_superpixels_valid}")

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ts = np.linspace(0, 1, 40)
ts = np.concatenate(list(zip((ts_even := ts[0::2]), (ts_odd := ts[1::2][::-1]))))  # noqa: B905
suppixs_colors = list(map(tuple, mpl.colormaps["jet"](ts)))

ax = axrow[0]
_ = ax.imshow(
    sk.segmentation.mark_boundaries(img, suppixs, color=mpl.colors.to_rgb("magenta")),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=(lw := 2.5),
    linestyles="--",
)

ax = axrow[1]
_ = ax.imshow(
    sk.color.label2rgb(suppixs, colors=suppixs_colors),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)

ax = axrow[2]
_ = ax.imshow(
    sk.segmentation.mark_boundaries(img, suppixs_valid, color=mpl.colors.to_rgb("magenta")),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)

ax = axrow[3]
_ = ax.imshow(
    sk.color.label2rgb(suppixs_valid, colors=suppixs_colors, bg_color="gray"),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)


# %%
# best achievable iou with superpixels (initial state)

suppixs_on_gt_labels = set(np.unique(suppixs * safe_tensor_to_numpy(mask.int())))
suppixs_on_gt = suppixs * np.isin(suppixs, suppixs_on_gt_labels)
num_suppixs_on_gt = len(set(suppixs_on_gt_labels) - {0})
print(f"number of superpixels on gt: {num_suppixs_on_gt}")

# find segments 100% inside the ground truth mask
suppixs_in_gt_labels = sorted(
    {suppix_label for suppix_label in suppixs_on_gt_labels if (mask[suppixs == suppix_label]).all()},
)
suppixs_in_gt = suppixs * np.isin(suppixs, suppixs_in_gt_labels)
num_superpixels_in_gt = len(set(suppixs_in_gt_labels) - {0})
print(f"number of superpixels in gt: {num_superpixels_in_gt}")

initial_selected_suppixs_labels = set(suppixs_in_gt_labels)
initial_available_suppixs_labels = set(suppixs_on_gt_labels) - set(suppixs_in_gt_labels)
print(f"AT START: {len(initial_selected_suppixs_labels)=} {len(initial_available_suppixs_labels)=}")
# TODO re-add validation

# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ax = axrow[0]
_ = ax.imshow(
    sk.segmentation.mark_boundaries(img, suppixs_on_gt, color=mpl.colors.to_rgb("magenta")),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=(lw := 2.5),
    linestyles="--",
)
ax = axrow[1]
_ = ax.imshow(
    sk.segmentation.mark_boundaries(img, suppixs_in_gt, color=mpl.colors.to_rgb("magenta")),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)

# %%
# viz segment selection

import matplotlib as mpl
from numpy import ndarray


def suppix_selection_map(
    suppixs: ndarray,
    selected_suppixs_labels: set[int],
    available_suppixs_labels: set[int],
) -> ndarray:
    # 0 means "out of scope", meaning it is not in the initial `segments`
    suppixs_selection_viz = np.full_like(suppixs, np.nan, dtype=float)
    # 1 means "selected"
    suppixs_selection_viz[np.isin(suppixs, list(selected_suppixs_labels))] = 1
    # 2 means "available"
    suppixs_selection_viz[np.isin(suppixs, list(available_suppixs_labels))] = 2
    return suppixs_selection_viz


# suppixs_selection_viz = suppix_selection_map(suppixs, initial_selected_suppixs_labels, initial_available_suppixs_labels)
suppixs_selection_viz = suppix_selection_map(suppixs, initial_selected_suppixs_labels, initial_available_suppixs_labels)


def iou_of_segmentation(segmentation: ndarray, mask: ndarray) -> float:
    assert segmentation.shape == mask.shape, f"{segmentation.shape=} {mask.shape=}"
    assert segmentation.dtype == bool, f"{segmentation.dtype=}"
    assert mask.dtype == bool, f"{mask.dtype=}"
    return (segmentation & mask).sum() / (segmentation | mask).sum()


# -----------------------------------------------------------------------------

# define a custom colormap for the values above (0, 1, 2, 3)
cmap = mpl.colors.ListedColormap([(0, 0, 0, 0), "tab:blue", "tab:olive"])
norm = mpl.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()

ax = axrow[0]
_ = ax.imshow(
    sk.segmentation.mark_boundaries(img, suppixs & ~np.isnan(suppixs_selection_viz), color=mpl.colors.to_rgb("magenta")),
)
_ = ax.imshow(suppixs_selection_viz, cmap=cmap, norm=norm, alpha=0.5)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=(lw := 2.5),
    linestyles="--",
)

ax = axrow[1]
_ = ax.imshow(img)
cs_sel_suppixs = ax.contour(
    suppixs_selection_viz,
    levels=[1.5, 2.5],
    colors="tab:blue",
    linewidths=(lw := 1.5),
)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="black",
    linewidths=lw,
    linestyles="--",
)
_ = ax.annotate(
    f"initial iou={iou_of_segmentation(suppixs_selection_viz == 1, mask.numpy()):.0%}",
    xy=(0, 1),
    xycoords="axes fraction",
    xytext=(10, -10),
    textcoords="offset points",
    ha="left",
    va="top",
    fontsize=20,
    bbox=dict(  # noqa: C408
        facecolor="white",
        alpha=1,
        edgecolor="black",
        boxstyle="round,pad=0.2",
    ),
)
try:
    segm
except NameError:
    pass
else:
    _ = ax.contour(
        segm,
        levels=[0.5],
        colors="green",
        linewidths=lw,
        linestyles="--",
    )
    _ = ax.annotate(
        f"final iou={iou_of_segmentation(segm, mask.numpy()):.0%}",
        xy=(0, 0), xycoords="axes fraction",
        xytext=(10, 10), textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=20,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )

# %%toothbrush
# best achievable iou with superpixels (search)
import copy

initial_num_selected = len(initial_selected_suppixs_labels)
sel = copy.deepcopy(initial_selected_suppixs_labels)
avail = copy.deepcopy(initial_available_suppixs_labels)
segm = suppix_selection_map(suppixs, sel, set()) == 1

history = [
    {
        "selected": copy.deepcopy(sel),
        "iou": iou_of_segmentation(segm, safe_tensor_to_numpy(mask)),
    },
]

from progressbar import progressbar

# this can be much faster by computing each suppix's iou with the mask and sorting them
# best first search
for _ in progressbar(range(int(1.10 * (num_suppixs_on_gt - initial_num_selected)))):
    # find the segment that maximizes the iou of the current segmentation
    best = max(
        avail,
        key=lambda label: iou_of_segmentation(
            segm | suppixs == label,
            safe_tensor_to_numpy(mask),
        ),
    )
    segm |= suppixs == best
    iou = iou_of_segmentation(segm, safe_tensor_to_numpy(mask))
    sel.add(best)
    avail.remove(best)
    history.append(
        {
            "best": best,
            "selected": copy.deepcopy(sel),
            "iou": iou,
        },
    )
    hist_last = history[-1]
    hist_prev = history[-2]
    if hist_last["iou"] < hist_prev["iou"]:
        break


# %%
# =============================================================================
# search history

num_suppixs_hist = np.array([len(h["selected"]) for h in history])
iou_hist = np.array([h["iou"] for h in history])
best_candidate_hist = np.array([h.get("best", None) for h in history])

best_iou_stepidx = np.argmax(iou_hist)
best_dict = history[best_iou_stepidx]
best_iou = best_dict["iou"]
best_selection = best_dict["selected"]
best_segmentation = suppix_selection_map(suppixs, best_selection, set()) == 1

best_candidate_mean_ascore_hist = np.array([
    asmap[suppixs == best_candidate].mean().item()
    for best_candidate in best_candidate_hist
])

fig, ax = plt.subplots(1, 1, figsize=np.array((6, 6)))

_ = ax.plot(num_suppixs_hist, iou_hist, label="iou history", marker="o", markersize=5, color="black")
_ = ax.axhline(best_iou, color="black", linestyle="--", label="best iou")

twinax = ax.twinx()
_ = twinax.scatter(
    num_suppixs_hist, best_candidate_mean_ascore_hist,
    label="superpixel mean ascore", marker="x",
    color="tab:blue",
)
_ = twinax.axhline(_get_aupimo_thresh_lower_bound(), color="tab:blue", linestyle="--", label="aupimo thresh")

_ = ax.set_xlabel("Number of superpixels selected")
# integer format
_ = ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1)
_ = ax.set_yticks(np.linspace(0, 1, 6))
_ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.grid(axis="y")

_ = twinax.set_ylabel("Superpixel anomaly score")
_ = twinax.set_yticks(np.linspace(*twinax.get_ylim(), 6))
_ = twinax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

# TODO curve of avg contour distance
# avg_cont_dist_of_selection = compute_avg_contour_distance(current_segmentation, mask.numpy())
# avg_cont_dist_of_best = compute_avg_contour_distance(best_segmentation, mask.numpy())

# %%
from progressbar import progressbar

# +1 is for the background
binclf_per_suppix = np.zeros((num_superpixels + 1, 2, 2), dtype=int)
# but it has dummy values
binclf_per_suppix[0, :] = -1

for label in progressbar(suppixs_labels):
    suppix_mask = suppixs == label
    tp = (mask & suppix_mask).sum()
    fp = (~mask & suppix_mask).sum()
    tn = (~mask & ~suppix_mask).sum()
    fn = (mask & ~suppix_mask).sum()
            # - `tp`: `[... , 1, 1]`
            # - `fp`: `[... , 0, 1]`
            # - `fn`: `[... , 1, 0]`
            # - `tn`: `[... , 0, 0]`
    binclf_per_suppix[label, :] = np.array([[tn, fp], [fn, tp]])

# %%
from aupimo.binclf_curve_numpy import per_image_iou

iou_per_suppix = np.concatenate([[-1], per_image_iou(binclf_per_suppix[1:, None, ...])[:, 0]])

# %%
# best achievable iou with superpixels (search)
import copy

initial_num_selected = len(initial_selected_suppixs_labels)
sel = copy.deepcopy(initial_selected_suppixs_labels)
avail = copy.deepcopy(initial_available_suppixs_labels)
segm = suppix_selection_map(suppixs, sel, set()) == 1

def iou_of_selection(sel: set[int], binclf_per_suppix: ndarray, mask: ndarray) -> float:
    binclfs = binclf_per_suppix[list(sel)]
    tp = binclfs[:, 1, 1].sum()
    fp = binclfs[:, 0, 0].sum()
    gt_size = mask.sum()
    return tp / (fp + gt_size)

history = [
    {
        "selected": copy.deepcopy(sel),
        "iou": iou_of_selection(sel, binclf_per_suppix, mask.numpy()),
    },
]

from progressbar import progressbar

# this can be much faster by computing each suppix's iou with the mask and sorting them
# best first search
for _ in progressbar(range(int(1.10 * (num_suppixs_on_gt - initial_num_selected)))):
    # find the segment that maximizes the iou of the current segmentation
    best = max(
        avail,
        key=lambda label: iou_of_selection(
            sel | {label},
            binclf_per_suppix,
            mask.numpy(),
        ),
    )
    segm |= suppixs == best
    iou = iou_of_selection(
        sel | {label},
        binclf_per_suppix,
        mask.numpy(),
    )
    sel.add(best)
    avail.remove(best)
    history.append(
        {
            "best": best,
            "selected": copy.deepcopy(sel),
            "iou": iou,
        },
    )
    hist_last = history[-1]
    hist_prev = history[-2]
    if hist_last["iou"] < hist_prev["iou"]:
        break

# %%
