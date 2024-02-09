"""Evaluate the anomaly score maps (asmaps) of a model on a dataset.

Important: overwritting output files is the default behavior.

author: jpcbertoldo
"""
#!/usr/bin/env python

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import json
import sys
import warnings
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


from aupimo import aupimo_scores, AUPIMOResult

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
    (AUPIMO := "aupimo"),
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
                "--asmaps ../data/experiments/benchmark/patchcore_wr50/mvtec/metal_nut/asmaps.pt",
                # "--metrics auroc",
                # "--metrics aupr",
                # "--metrics aupro",
                # "--metrics aupimo",
                "--mvtec-root ../data/datasets/MVTec",
                "--visa-root ../data/datasets/VisA",
                # "--not-debug",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

savedir = args.asmaps.parent

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
    _save_value(savedir / "auroc.json", auroc, args.debug)

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
    _save_value(savedir / "aupr.json", aupr, args.debug)


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
    _save_value(savedir / "aupro.json", aupro, args.debug)


# %%
# AUPIMO

if AUPIMO in args.metrics:
    print("computing aupimo")

    # TODO retry with increasing number of thresholds (up to 1_000_000)
    pimoresult, aupimoresult = aupimo_scores(
        asmaps,
        masks,
        fpr_bounds=(1e-5, 1e-4),
        paths=images_relpaths,  # relative, not absolute paths!
        num_threshs=30_000,
    )

    print("saving aupimo")

    aupimo_dir = savedir / "aupimo"
    if args.debug:
        aupimo_dir = aupimo_dir.with_stem("debug_" + aupimo_dir.stem)
    aupimo_dir.mkdir(exist_ok=True)

    pimoresult.save(aupimo_dir / "curves.pt")
    aupimoresult.save(aupimo_dir / "aupimos.json")

# %%
# ================================================================================================================
# new stuff (not in aupimo paper)

try:
    asmaps_tensor  # type: ignore
except NameError:
    asmaps_tensor = asmaps
    masks_tensor = masks

# %%

asmaps = asmaps_tensor.numpy()
masks = masks_tensor.numpy()

from aupimo.utils_numpy import valid_anomaly_score_maps

def _get_aupimo_thresh_upper_bound(aupimoresult_fpath: Path) -> float:
    aupimoresult = AUPIMOResult.load(aupimoresult_fpath)
    aupimo_thresh_upper_bound = aupimoresult.thresh_bounds[1]
    return aupimo_thresh_upper_bound

vasmaps = torch.from_numpy(valid_anomaly_score_maps(
    asmaps, _get_aupimo_thresh_upper_bound(args.asmaps.parent / "aupimo" / "aupimos.json")
))

# get gray cmap and make nan values gray
from matplotlib import pyplot as plt
VASMAPS_CMAP = plt.cm.get_cmap("gray")
VASMAPS_CMAP.set_bad("gray")

def _plot_gt_contour(ax, mask, **kwargs):
    kwargs = {**dict(colors="magenta", linewidths=(lw := 0.8)), **kwargs}
    ax.contour(mask, levels=[0.5], **kwargs)

# %%
# viz an anomalous image and its asmap
from matplotlib import pyplot as plt
from aupimo.pimo_numpy import _images_classes_from_masks

image_classes = _images_classes_from_masks(masks)

# get the first anomalous image
image_idx = np.where(image_classes == 1)[0][0]
asmap = asmaps[image_idx]
vasmap = vasmaps[image_idx]
mask = masks[image_idx]

img = plt.imread(images_abspaths[image_idx])
fig, axrow = plt.subplots(1, 3, figsize=(12, 4))
axrow[0].imshow(img)
axrow[1].imshow(mask)
axrow[2].imshow(asmap)
# %%
aupimoresult_fpath = savedir / "aupimo" / "aupimos.json"
from aupimo import AUPIMOResult

fig, axrow = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, constrained_layout=True)
_ = axrow[0].imshow(img)
_plot_gt_contour(axrow[0], mask)
_ = axrow[1].imshow(vasmap, cmap=VASMAPS_CMAP)
_plot_gt_contour(axrow[1], mask)

# %%
# TFPR

from aupimo.oracles_numpy import per_image_tfpr_curves

per_image_threshs, per_image_tfprs = per_image_tfpr_curves(
    asmaps, masks,
    min_valid_score=_get_aupimo_thresh_upper_bound(args.asmaps.parent / "aupimo" / "aupimos.json"),
    num_threshs=1000,
)

# %%
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# MAKE A FUNCTION TO FIND APPROXIMATIONS TO PIVOT POINTS IN THE CURVE
# AND PUT IT INT utils_numpy.py

# ideas to place somewhere 
# 1) avalanche
# 2) watershed on the image gradient with markers from the asmap's maxima 

# first approximation for the superpixel selection: exclude all superpixels touching the border

is_anom = ~tfpr_curves.isnan().all(dim=-1).all(dim=-1)

# TFPR_PIVOTS = [3e0, 1e1, 3e1]
TFPR_PIVOTS = [1e0, 3e0, 1e1]
# TFPR_PIVOTS = [1e-1, 3e-1, 1e0]

tfpr_pivots_thresh_idx = torch.stack(
    [torch.argmin(torch.abs(tfpr_curves[:, 1, :] - pivot), dim=1) for pivot in TFPR_PIVOTS],
    dim=1,
)
tfpr_pivots_thresh = torch.stack(
    [tfpr_curve[0, :][idxs] for tfpr_curve, idxs in zip(tfpr_curves, tfpr_pivots_thresh_idx, strict=False)],
    dim=0,
)
t2pr_pivots_actual = torch.stack(
    [tfpr_curve[1, :][idxs] for tfpr_curve, idxs in zip(tfpr_curves, tfpr_pivots_thresh_idx, strict=False)],
    dim=0,
)

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.plot(thrsh_idx, tfpr_curves[:, 1].T, label=np.arange(len(precision_curves)))
# ax.scatter(tfpr_pivots_thresh_idx[is_anom], t2pr_pivots_actual[is_anom])
# ax.set_xlabel("threshold index")
# ax.set_ylabel("tfpr")
# ax.set_yscale("log")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = ax.plot(precision_curves[:, 0].T, tfpr_curves[:, 1].T, label=np.arange(len(precision_curves)))
_ = ax.scatter(tfpr_pivots_thresh[is_anom], t2pr_pivots_actual[is_anom])
_ = ax.set_xlabel("threshold")
_ = ax.set_ylabel("tfpr")
_ = ax.set_yscale("log")
_ = ax.set_ylim(1e-1, 1e3)
_ = ax.legend(loc="upper left")

# %%
# viz tfpr pivots' contours

import numpy as np

image_idx = 9
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
threshs = tfpr_pivots_thresh[image_idx]

# hotfix to make sure they are increasing
threshs *= np.linspace(0.999, 1, len(threshs))

tfpr_of_threshs = t2pr_pivots_actual[image_idx]
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)
axrow = axes.ravel()
_ = axrow[0].imshow(img)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[1].imshow(asmap, cmap="gray")
_ = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
cs = axrow[1].contour(asmap, threshs, colors=["red", "blue", "green"], linewidths=lw)
fmt = {l: f"{int(np.ceil(val))}" for l, val in zip(cs.levels, tfpr_of_threshs, strict=False)}
axrow[1].clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
# get gray cmap and make nan values black
cmap = plt.cm.get_cmap("gray")
_ = cmap.set_bad("gray")
_ = axrow[2].imshow(vasmap, cmap=cmap)
_ = axrow[2].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
axrow[2].contour(vasmap, threshs, colors=["red", "blue", "green"], linewidths=lw)

# %%
from aupimo import per_image_iou
per_image_iou_values = per_image_iou(per_image_binclfs)
iou_curves = torch.stack([threshs_per_image, per_image_iou_values], dim=1)

# %%
# plot iou curves

from scipy.signal import argrelextrema

# find local maxima
iou_local_maxima_idxs = np.array(argrelextrema(iou_curves[:, 1].numpy(), np.greater, order=10, axis=1))

# mask to select thresholds as iou values from iou curves where there is a local maxima
iou_local_maxima_mask = np.zeros(iou_curves[:, 1].shape, dtype=bool)
_flat_mask = np.ravel_multi_index(iou_local_maxima_idxs, iou_local_maxima_mask.shape)
np.ravel(iou_local_maxima_mask)[_flat_mask] = True

# convert to torch
iou_local_maxima_idxs = torch.from_numpy(iou_local_maxima_idxs)
iou_local_maxima_mask = torch.from_numpy(iou_local_maxima_mask)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# by doing this, i am losing the information of "from which image" the local maxima is
# see the dicts in the cell below
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
iou_local_maxima = iou_curves[:, 1][iou_local_maxima_mask]
iou_local_maxima_threshs = iou_curves[:, 0][iou_local_maxima_mask]

iou_global_maxima_idx = torch.argmax(iou_curves[:, 1], dim=1)
iou_global_maxima = iou_curves[:, 1][torch.arange(len(iou_curves)), iou_global_maxima_idx]
iou_global_maxima_threshs = iou_curves[:, 0][torch.arange(len(iou_curves)), iou_global_maxima_idx]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = ax.plot(iou_curves[:, 0].T, iou_curves[:, 1].T, label=np.arange(len(iou_curves)))
_ = ax.scatter(iou_local_maxima_threshs, iou_local_maxima, s=50, c='blue')
_ = ax.scatter(iou_global_maxima_threshs, iou_global_maxima, s=50, c='red')
_ = ax.set_xlabel("threshold")
_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1)
_ = ax.legend(loc="upper right")

# %%
# viz max IoU contours

import numpy as np

image_idx = 6
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
if img.ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

# get the global maximum
thresh = iou_global_maxima_threshs[image_idx]
thresh_idx = iou_global_maxima_idx[image_idx]

iou_of_thresh = iou_global_maxima[image_idx]
tfpr_of_thresh = tfpr_curves[image_idx, 1, thresh_idx]
print(f"{image_idx=} {thresh=:.2g} {iou_of_thresh=:.0%} {int(tfpr_of_thresh)=}")

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)
axrow = axes.ravel()
_ = axrow[0].imshow(img)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[1].imshow(asmap, cmap="gray")
_ = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
cs = axrow[1].contour(asmap, [thresh], colors=["red"], linewidths=lw)
fmt = {l: f"{val:.0%}" for l, val in zip(cs.levels, [iou_of_thresh], strict=False)}
_ = axrow[1].clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
# get gray cmap and make nan values black
cmap = plt.cm.get_cmap("gray")
_ = cmap.set_bad("gray")
_ = axrow[2].imshow(vasmap, cmap=cmap)
_ = axrow[2].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[2].contour(vasmap, [thresh], colors=["red"], linewidths=lw)

# %%
# viz contour distance map
from scipy.ndimage import distance_transform_edt
import skimage.morphology as skm

image_idx = 5
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
if img.ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

thresh = iou_global_maxima_threshs[image_idx]

def get_contour_mask(mask, type="inner"):
    assert mask.dtype == bool, f"{mask.dtype=}"
    if type == "inner":
        return skm.binary_dilation(~mask, skm.square(3)) * mask  # contour of class 1 (anomalous)
    elif type == "outter":
        return skm.binary_dilation(mask, skm.square(3)) * (~mask)  # contour of class 1 (anomalous)
    raise ValueError(f"{type=}")

def saturate_distance_map(distance_map, mask, OUTTER_INNER_MAX_RATIO = 1):

    max_inner_distance = distance_map[mask].max()
    max_outter_distance = distance_map[~mask].max()

    # deal with edge cases
    if max_inner_distance == 0:
        max_inner_distance = 1

    if max_outter_distance == 0:
        max_outter_distance = 1

    if max_inner_distance > (max_outter_distance * OUTTER_INNER_MAX_RATIO):
        saturation_distance = max_outter_distance * OUTTER_INNER_MAX_RATIO

    elif max_outter_distance > (max_inner_distance * OUTTER_INNER_MAX_RATIO):
        saturation_distance = max_inner_distance * OUTTER_INNER_MAX_RATIO

    else:
        saturation_distance = max(max_inner_distance, max_outter_distance)

    return np.clip(
        distance_map, a_min=None, a_max=saturation_distance
    ), saturation_distance

# =============================================================================
# (PRED --> REF) NOPE!!!

# dist pred -> ref (it is not the same as dist ref -> pred)
distance_to_mask_contour_map = distance_transform_edt(~get_contour_mask(mask.to(bool).numpy()))

saturated_distance_to_mask_contour_map, saturation_distance = saturate_distance_map(distance_to_mask_contour_map, mask)
pred_contour_binmask = get_contour_mask((asmap >= thresh).numpy(), type="inner")

distances_on_line = saturated_distance_to_mask_contour_map[pred_contour_binmask]
norm_distances_on_line = distances_on_line / saturation_distance
avg_norm_distance_on_line = norm_distances_on_line.mean()

distance_line_map = saturated_distance_to_mask_contour_map.copy()
distance_line_map_viz = distance_line_map.copy()
distance_line_map[~pred_contour_binmask] = np.nan
distance_line_map_viz[~skm.binary_dilation(pred_contour_binmask, skm.square(7))] = np.nan

mindist_1didx = np.nanargmin(distance_line_map)
mindist_2didx = np.unravel_index(mindist_1didx, distance_line_map.shape)

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False, constrained_layout=True)
axrow = axes.ravel()
_ = axrow[0].imshow(img)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[0].contour(vasmap, levels=[thresh], colors="red", linewidths=(lw := 0.8))
_ = axrow[1].imshow(distance_to_mask_contour_map, cmap="viridis", vmax=saturation_distance)
_ = axrow[1].contour(vasmap, levels=[thresh], colors="red", linewidths=(lw := 0.8))
cmap = plt.cm.get_cmap("viridis")
_ = cmap.set_bad("gray")
_ = axrow[3].imshow(distance_line_map_viz, cmap=cmap, vmin=0,vmax=saturation_distance)
print("disclaimer: the line of distances viz is dilated to make it more visible")
_ = axrow[3].scatter(mindist_2didx[1], mindist_2didx[0], s=150, c="black")
_ = axrow[2].hist(distances_on_line, bins=20, density=True)
_ = axrow[2].axvline(avg_norm_distance_on_line * saturation_distance, c="red")
_ = axrow[2].set_xlabel("distance")
_ = axrow[2].set_ylabel("density")
_ = axrow[2].set_title(f"avg norm distance: {avg_norm_distance_on_line:.0%}")

_ = fig.suptitle("contour distance from PRED --> REF")

# =============================================================================
# (REF --> PRED)

# dist ref -> pred
pred = (asmap >= thresh).numpy()
distance_to_pred_contour_map = distance_transform_edt(~get_contour_mask(pred))

saturated_distance_to_pred_contour_map, saturation_distance = saturate_distance_map(distance_to_pred_contour_map, pred)

mask_contour_binmask = get_contour_mask(mask.numpy(), type="inner")

distances_on_line = saturated_distance_to_pred_contour_map[mask_contour_binmask]
norm_distances_on_line = distances_on_line / saturation_distance
avg_norm_distance_on_line = norm_distances_on_line.mean()

distance_line_map = saturated_distance_to_pred_contour_map.copy()
distance_line_map_viz = distance_line_map.copy()
distance_line_map[~mask_contour_binmask] = np.nan
distance_line_map_viz[~skm.binary_dilation(mask_contour_binmask, skm.square(5))] = np.nan

mindist_1didx = np.nanargmin(distance_line_map)
mindist_2didx = np.unravel_index(mindist_1didx, distance_line_map.shape)

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False, constrained_layout=True)
axrow = axes.ravel()
_ = axrow[0].imshow(img)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[0].contour(vasmap, levels=[thresh], colors="red", linewidths=(lw := 0.8))
_ = axrow[1].imshow(distance_to_pred_contour_map, cmap="viridis", vmax=saturation_distance)
_ = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
cmap = plt.cm.get_cmap("viridis")
_ = cmap.set_bad("gray")
_ = axrow[3].imshow(distance_line_map_viz, cmap=cmap, vmin=0,vmax=saturation_distance)
print("disclaimer: the line of distances viz is dilated to make it more visible")
_ = axrow[3].scatter(mindist_2didx[1], mindist_2didx[0], s=150, c="black")
_ = axrow[2].hist(distances_on_line, bins=20, density=True)
_ = axrow[2].axvline(avg_norm_distance_on_line * saturation_distance, c="red")
_ = axrow[2].set_xlabel("distance")
_ = axrow[2].set_ylabel("density")
_ = axrow[2].set_title(f"avg norm distance: {avg_norm_distance_on_line:.0%}")

_ = fig.suptitle("contour distance from REF --> PRED")

# %%
%%time
# avg contour distance curve

image_idx = 5
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
threshs = threshs_per_image[image_idx]
threshs_sampled = torch.linspace(threshs.min(), threshs.max(), 5)

def compute_avg_contour_distance(pred: Tensor, mask: Tensor) -> float:
    """use the ref --> pred contour distance"""
    assert isinstance(pred, np.ndarray), f"{pred=}"
    assert isinstance(mask, np.ndarray), f"{mask=}"
    assert pred.shape == mask.shape, f"{pred.shape=} {mask.shape=}"
    assert pred.dtype == bool, f"{pred.dtype=}"
    assert mask.dtype == bool, f"{mask.dtype=}"
    distance_to_pred_contour_map = distance_transform_edt(~get_contour_mask(pred))
    saturated_distance_to_pred_contour_map, saturation_distance = saturate_distance_map(distance_to_pred_contour_map, pred)
    mask_contour_binmask = get_contour_mask(mask, type="inner")
    distances_on_line = saturated_distance_to_pred_contour_map[mask_contour_binmask]
    norm_distances_on_line = distances_on_line / saturation_distance
    avg_norm_distance_on_line = norm_distances_on_line.mean()
    return avg_norm_distance_on_line

# debug with the instance from the previous cell
compute_avg_contour_distance((asmap >= thresh).numpy(), mask.numpy())

def compute_avg_contour_distance_curve(
    asmap: Tensor,
    mask: Tensor,
    threshs: Tensor,
) -> Tensor:
    return torch.tensor([
        compute_avg_contour_distance(
            (asmap >= thresh).numpy().astype(bool),
            mask.numpy().astype(bool),
        )
        for thresh in threshs
    ]) if not mask.sum() == 0 else torch.full_like(threshs, np.nan)

compute_avg_contour_distance_curve(asmap, mask, threshs_sampled)

## %%
# compute avg contour distance curves

threshs_per_image_sampled = torch.stack([
    torch.linspace(threshs.min(), threshs.max(), 50)
    for threshs in threshs_per_image
], dim=0)

def compute_avg_contour_distance_curves(
    asmaps: Tensor,
    masks: Tensor,
    threshs_per_image: Tensor,
) -> Tensor:
    return torch.stack([
        compute_avg_contour_distance_curve(asmap, mask, threshs)
        for asmap, mask, threshs in zip(asmaps, masks, threshs_per_image)
    ], dim=0)

avg_contour_distance_curves_values = compute_avg_contour_distance_curves(
    asmaps, masks, threshs_per_image_sampled
)

avg_contour_distance_curves = torch.stack([
    threshs_per_image_sampled,
    avg_contour_distance_curves_values,
], dim=1)

# %%
# plot avg contour distance curves

avg_cont_dist_global_minima_idx = torch.argmin(avg_contour_distance_curves[:, 1], dim=1)
avg_cont_dist_global_minima = avg_contour_distance_curves[:, 1][
    torch.arange(len(avg_contour_distance_curves)),
    avg_cont_dist_global_minima_idx
]
avg_cont_dist_global_minima_threshs = avg_contour_distance_curves[:, 0][
    torch.arange(len(avg_contour_distance_curves)),
    avg_cont_dist_global_minima_idx
]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = ax.plot(
    avg_contour_distance_curves[:, 0].T,
    avg_contour_distance_curves[:, 1].T,
    label=np.arange(len(avg_contour_distance_curves))
)
_ = ax.scatter(avg_cont_dist_global_minima_threshs, avg_cont_dist_global_minima, s=50, c='red')
_ = ax.set_xlabel("threshold")
_ = ax.set_ylabel("avg contour distance")
_ = ax.set_ylim(0, 1)
_ = ax.legend(loc="lower right")

# %%
# viz min avg contour distance contours

import numpy as np

image_idx = 5
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
if img.ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

# get the global maximum
thresh = avg_cont_dist_global_minima_threshs[image_idx]
thresh_idx = avg_cont_dist_global_minima_idx[image_idx]

avg_cont_dist_of_thresh = avg_cont_dist_global_minima[image_idx]
tfpr_of_thresh = tfpr_curves[image_idx, 1, thresh_idx]
print(f"{image_idx=} {thresh=:.2g} {avg_cont_dist_of_thresh=:.0%} {int(tfpr_of_thresh)=}")

fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True)
axrow = axes.ravel()
_ = axrow[0].imshow(img)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[1].imshow(asmap, cmap="gray")
_ = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
cs = axrow[1].contour(asmap, [thresh], colors=["red"], linewidths=lw)
fmt = {l: f"{val:.0%}" for l, val in zip(cs.levels, [avg_cont_dist_of_thresh], strict=False)}
_ = axrow[1].clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
# get gray cmap and make nan values black
cmap = plt.cm.get_cmap("gray")
_ = cmap.set_bad("gray")
_ = axrow[2].imshow(vasmap, cmap=cmap)
_ = axrow[2].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[2].contour(vasmap, [thresh], colors=["red"], linewidths=lw)


# %%
# superpixel segmentation example
# src: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html#comparison-of-segmentation-and-superpixel-algorithms

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, mark_boundaries, quickshift, slic, watershed

image_idx = 5
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
threshs = tfpr_pivots_thresh[image_idx]

# using info from aupimo's x-axis bounds to determine `min_size`
aupimo_min_size = int(np.ceil(1e-5 * np.prod(img.shape[:2])))
# 5 here

segments_fz = felzenszwalb(
    img,
    # higher scale --> larger clusters
    # img, scale=100,  # from the example
    scale=200,
    sigma=0.5,
    min_size=50,  # from the example
    # too small (~5)
    # min_size=aupimo_min_size,
)

# K-means in the 5d space: Color-(x,y,z) + image location
segments_slic = slic(
    img,
    # nb of k-means clusters/centers
    n_segments=(n_segments:=2500),
    # tradeoff between color and space proximity; higher --> + weight to space
    compactness=10,
    # use LAB color space; useful?
    # convert2lab=True,
    # preprocessing blur smoothing in the 5d space, can have per-channel values
    sigma=0,  # deactivated
    # sigma=1,  # from the example
    # could it useful?
    # min_size_factor=aupimo_min_size / (supposed_size := np.prod(img.shape[:2]) / n_segments),
    # doc recommends 3; is it useful?
    # max_size_factor=3,
    start_label=1,
    # zero-parameter mode
    # slic_zero=True,
)

# mode-seeking algorithm (mean-shift) in color-(x, y) space
segments_quick = quickshift(
    img,
    # gaussian kernel used for smoothing the sample density
    kernel_size=3,
    # preprocessing blur smoothing
    sigma=0,  # deactivated
    # cut off point for the neighborhood in the mean-shift procedure (not sure?)
    # higher --> less clusters
    max_dist=6,
    # color-space tradeoff; higher --> + weight to color
    ratio=.1,
    convert2lab=False,
    rng=42,
)

segments_watershed = watershed(
    (gradient := sobel(rgb2gray(img))),
    markers=2500,
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    compactness=0.001
)

segments_watershed_in_vasmap_mask = watershed(
    (gradient := sobel(rgb2gray(img))),
    # markers=2500,
    markers=(vasmap_mask := ~vasmap.isnan().numpy()).sum() / 100,
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    # compactness=0.001,  # from the example
    compactness=3e-4,
    mask=vasmap_mask,
)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
print(f"Watershed in vasmap mask number of segments: {len(np.unique(segments_watershed_in_vasmap_mask))}")

fig, ax = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True, constrained_layout=True)
ax = ax.T
_ = ax[0, 0].imshow(mark_boundaries(img, segments_fz))
_ = ax[0, 0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[0, 0].set_title("Felzenszwalbs's method")
_ = ax[0, 1].imshow(mark_boundaries(img, segments_slic))
_ = ax[0, 1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[0, 1].set_title("SLIC")
_ = ax[1, 0].imshow(mark_boundaries(img, segments_quick))
_ = ax[1, 0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[1, 0].set_title("Quickshift")
_ = ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
_ = ax[1, 1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[1, 1].set_title("Compact watershed")
_ = ax[2, 1].imshow(mark_boundaries(img, segments_watershed_in_vasmap_mask))
_ = ax[2, 1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = ax[2, 1].set_title("Compact watershed in vasmap mask")

# %%
# best achievable iou with superpixel segmentation (COMPACT WATERSHED)

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, mark_boundaries, quickshift, slic, watershed

image_idx = 5
asmap = asmaps[image_idx]
vasmap = _asmap2vasmap_aupimo_bounded(asmap)
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
threshs = tfpr_pivots_thresh[image_idx]

# resolution of the original image
# each superpixel is a zone with an unique value, like a semantic segmentation
segments = watershed(
    (gradient := sobel(rgb2gray(img))),
    # markers=2500,
    markers=(vasmap_mask := ~vasmap.isnan().numpy()).sum() / 100,
    # makes it harder for markers to flood faraway pixels --> regions more regularly shaped
    # compactness=0.001,  # from the example
    compactness=3e-4,
    mask=vasmap_mask,
)
segment_values = np.unique(segments[segments != 0])
num_segments = len(segment_values)
print(f"number of segments: {num_segments}")

# =============================================================================
# init segment selection

# find segments 100% inside the ground truth mask
# TODO speed up by converting to sets of pixel indexes and using set operations?
segments_inside_mask = {
    segment_value
    for segment_value in segment_values
    if mask[segments == segment_value].all()
}

# find segments 100% outside the ground truth mask
segments_outside_mask = np.array([
    segment_value
    for segment_value in segment_values
    if (~mask[segments == segment_value]).all()
])

selected_segments = set(segments_inside_mask)
available_segments = set(segment_values) - set(selected_segments) - set(segments_outside_mask)

print(f"{len(selected_segments)=} {len(available_segments)=}")

# =============================================================================
# viz segment selection (initial state)

import matplotlib as mpl
from numpy import ndarray

# zero means "out of scope", meaning it is not in the initial `segments`
segments_selection_viz = np.zeros_like(segments, dtype=int)

# 1 means "selected"
for segment_value in selected_segments:
    segments_selection_viz[segments == segment_value] = 1

# 2 means "available"
for segment_value in available_segments:
    segments_selection_viz[segments == segment_value] = 2

# 3 means "outside"
for segment_value in segments_outside_mask:
    segments_selection_viz[segments == segment_value] = 3

current_segmentation = segments_selection_viz == 1

def iou_of_segmentation(segmentation: ndarray, mask: ndarray) -> float:
    assert segmentation.shape == mask.shape, f"{segmentation.shape=} {mask.shape=}"
    assert segmentation.dtype == bool, f"{segmentation.dtype=}"
    assert mask.dtype == bool, f"{mask.dtype=}"
    return (segmentation & mask).sum() / (segmentation | mask).sum()

iou_of_selection = iou_of_segmentation(current_segmentation, mask.numpy())
avg_cont_dist_of_selection = compute_avg_contour_distance(current_segmentation, mask.numpy())

# define a custom colormap for the values above (0, 1, 2, 3)
colors = [(0, 0, 0, 0), "tab:green", "tab:blue", "tab:red"]
cmap = mpl.colors.ListedColormap(colors)
bounds = [0, 1, 2, 3, 4]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, axrow = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)

_ = axrow[0].imshow(mark_boundaries(img, segments))
_ = axrow[0].imshow(segments_selection_viz, cmap=cmap, norm=norm, alpha=0.5)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[0].set_title("segment selection initial state")

csmask = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8),)
cspred = axrow[1].contour(current_segmentation, levels=[0.5], colors="green", linewidths=(lw := 0.8),)
labels = [
    "ground truth mask",
    f"oracle segmentation\nIoU={iou_of_selection:.0%}\navg contour dist={avg_cont_dist_of_selection:.0%}",
]
for cs, label in zip([csmask, cspred], labels):
    cs.collections[0].set_label(label)
_ = axrow[1].legend()
_ = axrow[1].set_title("oracle segmentation")

_ = fig.suptitle("oracle segmentation (inital state)")

# =============================================================================
# find optimal segment selection to optimize iou
import copy

_current_segmentation = current_segmentation.copy()
_selected = copy.deepcopy(selected_segments)
_available = copy.deepcopy(available_segments)

history = [
    {
        "selected": copy.deepcopy(_selected),
        "iou": iou_of_segmentation(_current_segmentation, mask.numpy()),
    }
]

# best first search
while _available:
    # find the segment that maximizes the iou of the current segmentation
    best_segment = max(
        _available,
        key=lambda segment_value: iou_of_segmentation(
            _current_segmentation | (segments == segment_value),
            mask.numpy()
        )
    )
    _selected.add(best_segment)
    _available.remove(best_segment)
    _current_segmentation = _current_segmentation | (segments == best_segment)
    history.append({
        "best": best_segment,
        "selected": copy.deepcopy(_selected),
        "iou": iou_of_segmentation(_current_segmentation, mask.numpy()),
    })

# =============================================================================
# search history

iou_hist = np.array([h["iou"] for h in history])
best_iou_stepidx = np.argmax(iou_hist)

best_dict = history[best_iou_stepidx]
best_iou = best_dict["iou"]
best_selection = best_dict["selected"]

best_segmentation = np.zeros_like(segments, dtype=bool)
for segment_value in best_selection:
    best_segmentation[segments == segment_value] = True

iou_of_best = iou_of_segmentation(best_segmentation, mask.numpy())
avg_cont_dist_of_best = compute_avg_contour_distance(best_segmentation, mask.numpy())

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
_ = ax.plot(iou_hist)
_ = ax.scatter(best_iou_stepidx, best_iou, s=50, c='red')
_ = ax.set_xlabel("step index")
_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1)
_ = ax.legend(loc="lower right")

fig, axrow = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)

_ = axrow[0].imshow(mark_boundaries(img, segments))
_ = axrow[0].imshow(segments_selection_viz, cmap=cmap, norm=norm, alpha=0.5)
_ = axrow[0].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8))
_ = axrow[0].contour(best_segmentation, levels=[0.5], colors="black", linewidths=(lw := 1.2))
_ = axrow[0].set_title("segment selection best iou")

csmask = axrow[1].contour(mask, levels=[0.5], colors="magenta", linewidths=(lw := 0.8),)
cspred = axrow[1].contour(best_segmentation, levels=[0.5], colors="green", linewidths=(lw := 0.8),)
labels = [
    "ground truth mask",
    f"oracle segmentation\nIoU={iou_of_best:.0%}\navg contour dist={avg_cont_dist_of_best:.0%}",
]
for cs, label in zip([csmask, cspred], labels):
    cs.collections[0].set_label(label)
_ = axrow[1].legend()
_ = axrow[1].set_title("oracle segmentation")

_ = fig.suptitle("oracle segmentation (best iou)")

# %%