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
from functools import partial
from pathlib import Path

import numpy as np
import scipy as sp
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
_ = parser.add_argument("--checkpoints-dir", type=Path)
_ = parser.add_argument("--not-debug", dest="debug", action="store_false")
_ = parser.add_argument("--metrics", "-me", type=str, action="append", choices=METRICS_CHOICES, default=[])
_ = parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                # "--asmaps ../data/experiments/benchmark/patchcore_wr50/mvtec/metal_nut/asmaps.pt",
                "--asmaps ../data/experiments/benchmark/patchcore_wr50/mvtec/hazelnut/asmaps.pt",
                # "--metrics auroc",
                # "--metrics aupr",
                # "--metrics aupro",
                # "--metrics aupimo",
                # "--metrics ioucurves_global",
                # "--metrics ioucurves_local",
                # "--metrics max_avg_iou",
                # "--metrics max_iou_per_img",
                # "--metrics max_avg_iou_min_thresh",
                # "--metrics max_iou_per_img_min_thresh",
                "--mvtec-root ../data/datasets/MVTec",
                "--visa-root ../data/datasets/VisA",
                "--checkpoints-dir ../data/checkpoints",
                # "--not-debug",
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
# viz img vs {asmap, vasmap, mask}
import matplotlib.pyplot as plt

from aupimo.utils import valid_anomaly_score_maps

image_idx = 5
asmap = asmaps[image_idx]
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
if img.ndim == 2:
    img = img[..., None].repeat(3, axis=-1)

thresh_min = _get_aupimo_thresh_lower_bound()

vasmap = valid_anomaly_score_maps(asmap[None, ...], thresh_min)[0]

fig, axrow = plt.subplots(
    1,
    2,
    figsize=np.array((10, 5)) * 1.0,
    layout="constrained",
)

for ax in axrow:
    _ = ax.imshow(img)
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=(lw := 2.5),
        linestyles="--",
    )
    cs_thresh_min = ax.contour(
        asmap,
        levels=[thresh_min],
        colors=["white"],
        linewidths=lw,
    )

ax = axrow[0]
_ = ax.imshow(asmap, alpha=0.4, cmap="jet")
_ = ax.annotate(
    "Anomaly Score Map",
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

ax = axrow[1]
_ = ax.imshow(vasmap, alpha=0.4, cmap="jet")
_ = ax.annotate(
    "Valid Anomaly Score Map",
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

for ax in axrow:
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

# %%
# download the sam checkpoint
from huggingface_hub import hf_hub_download

# ckpt_name = "sam_vit_b_01ec64.pth"
ckpt_name = "sam_vit_h_4b8939.pth"

if not (ckpt_path := args.checkpoints_dir / ckpt_name).exists():
    print(f"downloading {ckpt_name=}")
    hf_hub_download(
        "ybelkada/segment-anything",
        f"checkpoints/{ckpt_name}",
        local_dir=args.checkpoints_dir.parent,
        local_dir_use_symlinks=False,
    )

else:
    print(f"{ckpt_name=} already exists")

from segment_anything import sam_model_registry

# %%
# load the sam model
sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)

# %%
# click points from the vasmap

import matplotlib.pyplot as plt
import skimage as sk
from segment_anything import SamPredictor

from aupimo._validate_tensor import safe_tensor_to_numpy
from aupimo.utils import valid_anomaly_score_maps

image_idx = 5
asmap = asmaps[image_idx]
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
if img.ndim == 2:
    img = img[..., None].repeat(3, axis=-1)
img_uint8 = plt.imread(images_abspaths[image_idx])
img_uint8 = (img_uint8 * 255).astype(np.uint8)
if img_uint8.ndim == 2:
    img_uint8 = img_uint8[..., None].repeat(3, axis=-1)

thresh_min = _get_aupimo_thresh_lower_bound()
vasmap = valid_anomaly_score_maps(asmap[None, ...], thresh_min)[0]

sam_predictor = SamPredictor(sam)
sam_predictor.set_image(img_uint8)

fig, axes = plt.subplots(
    3,
    2,
    figsize=np.array((20, 30)) * 0.5,
    layout="constrained",
)
axrow = axes.ravel()

for ax in axrow:
    _ = ax.imshow(img)
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=(lw := 2.5),
        linestyles="--",
    )
    cs_thresh_min = ax.contour(
        asmap,
        levels=[thresh_min],
        colors=["white"],
        linewidths=lw,
    )

ax = axrow[1]
_ = ax.imshow(vasmap, alpha=0.4, cmap="jet")

ax = axrow[2]
vasmap_maxima_coords = sk.feature.peak_local_max(
    safe_tensor_to_numpy(asmap),
    min_distance=1,
    threshold_abs=thresh_min,
    exclude_border=True,
)
normal_minima_coords = sk.feature.peak_local_max(
    safe_tensor_to_numpy(-asmap),
    min_distance=1,
    threshold_abs=-thresh_min,
    exclude_border=True,
)
_ = ax.scatter(
    vasmap_maxima_coords[:, 1],
    vasmap_maxima_coords[:, 0],
    c="black",
    s=100,
    alpha=1,
)
_ = ax.scatter(
    normal_minima_coords[:, 1],
    normal_minima_coords[:, 0],
    c="white",
    s=100,
    alpha=1,
)

ax = axrow[3]
sam_masks, sam_masks_qualities, _ = sam_predictor.predict(
    point_coords=np.concatenate(
        [
            vasmap_maxima_coords,
            normal_minima_coords,
        ],
        axis=0,
    ),
    point_labels=np.concatenate(
        [
            np.ones(len(vasmap_maxima_coords)),
            np.zeros(len(normal_minima_coords)),
        ],
    ),
    multimask_output=False,
    return_logits=False,
)
sam_mask_best = sam_masks[0]
sam_mask_best = sam_mask_best & safe_tensor_to_numpy(~vasmap.isnan())
_ = ax.imshow(sam_mask_best, cmap="jet")

ax = axrow[4]
gt_dist_borders = sp.ndimage.morphology.distance_transform_edt(mask)
gt_dist_borders_maxima = sk.feature.peak_local_max(
    gt_dist_borders,
    min_distance=1,
    threshold_rel=0.5,
    exclude_border=True,
)[:1]
gt_dist_borders_out = sp.ndimage.morphology.distance_transform_edt(~mask)
gt_dist_borders_out[safe_tensor_to_numpy(vasmap.isnan())] = 0
gt_dist_borders_out_maxima = sk.feature.peak_local_max(
    gt_dist_borders_out,
    min_distance=1,
    threshold_rel=0.5,
    exclude_border=False,
)[:1]
_ = ax.imshow(gt_dist_borders, cmap="jet", alpha=0.4)
_ = ax.scatter(
    gt_dist_borders_maxima[:, 1],
    gt_dist_borders_maxima[:, 0],
    c="black",
    s=100,
    alpha=1,
)
_ = ax.scatter(
    gt_dist_borders_out_maxima[:, 1],
    gt_dist_borders_out_maxima[:, 0],
    c="white",
    s=100,
    alpha=1,
)

ax = axrow[5]
sam_gt_masks, _, _ = sam_predictor.predict(
    point_coords=np.concatenate(
        [
            gt_dist_borders_maxima,
            gt_dist_borders_out_maxima,
        ],
        axis=0,
    ),
    point_labels=np.concatenate(
        [
            np.ones(len(gt_dist_borders_maxima)),
            np.zeros(len(gt_dist_borders_out_maxima)),
        ],
    ),
    multimask_output=False,
    return_logits=False,
)
sam_gt_mask_best = sam_gt_masks[0]
sam_gt_mask_best = sam_gt_mask_best & safe_tensor_to_numpy(~vasmap.isnan())
_ = ax.imshow(sam_gt_mask_best, cmap="jet",)

for ax in axrow:
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

# %%