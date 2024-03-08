#!/usr/bin/env python

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aupimo.oracles import IOUCurvesResult, MaxIOUPerImageResult
from aupimo.oracles_numpy import (
    open_image,
    upscale_image_asmap_mask,
    valid_anomaly_score_maps,
)

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


from aupimo._validate_tensor import safe_tensor_to_numpy
from aupimo.oracles_numpy import (
    calculate_levelset_mean_dist_to_superpixel_boundaries_curve,
)

# %%
# Args

# collection [of datasets] = {mvtec, visa}
ACCEPTED_COLLECTIONS = {
    (MVTEC_DIR_NAME := "MVTec"),
    (VISA_DIR_NAME := "VisA"),
}

parser = argparse.ArgumentParser()
_ = parser.add_argument("--asmaps", type=Path, required=True)
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
_ = parser.add_argument("--not-debug", dest="debug", action="store_false")
_ = parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
_ = parser.add_argument("--savedir", type=Path, default=None)

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                # "--asmaps ../../../data/experiments/benchmark/efficientad_wr101_s_ext/mvtec/capsule/asmaps.pt",
                "--asmaps ../../../data/experiments/benchmark/efficientad_wr101_s_ext/mvtec/transistor/asmaps.pt",
                # "--asmaps ../../../data/experiments/benchmark/rd++_wr50_ext/mvtec/bottle/asmaps.pt",
                "--mvtec-root ../../../data/datasets/MVTec",
                "--visa-root ../../../data/datasets/VisA",
                "--not-debug",
                "--savedir ../../../../2024-03-segm-paper/src/img",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

rundir = args.asmaps.parent
assert rundir.exists(), f"{rundir=}"

if args.savedir is not None:
    assert args.savedir.exists(), f"{args.savedir=}"

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


_convert_path = partial(_convert_path, collection_root=collection_root)

images_abspaths = [_convert_path(p) for p in images_relpaths]
masks_abspaths = [_convert_path(p) for p in masks_relpaths]

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

iou_oracle_threshs_dir = rundir / "iou_oracle_threshs"
superpixel_bound_dist_heuristic_dir = rundir / "superpixel_bound_dist_heuristic"
superpixel_oracle_selection_dir = Path(
    "/".join(rundir.parts[:-3] + ("patchcore_wr50",) + rundir.parts[-2:] + ("superpixel_oracle_selection",)),
)

# %%

ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_local_threshs.pt")
max_iou_per_image_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

payload_loaded = torch.load(superpixel_bound_dist_heuristic_dir / "superpixel_bound_dist_heuristic.pt")

# capsule
# image_idx = 95
# image_idx = 88

# transistor
image_idx = 29

threshs = payload_loaded["threshs_per_image"][image_idx]
num_levelsets = threshs.shape[0]
min_thresh = payload_loaded["min_thresh"]
upscale_factor = payload_loaded["upscale_factor"]

local_minima_idxs = payload_loaded["local_minima_idxs_per_image"][image_idx][:5]
local_minima_threshs = threshs[local_minima_idxs]
local_minima_ious = ioucurves.per_image_ious[image_idx][
    np.argmin(np.abs(local_minima_threshs[None, ...] - ioucurves.threshs[image_idx][..., None]), axis=0)
]

watershed_superpixel_relsize = payload_loaded["superpixels_params"]["superpixel_relsize"]
watershed_compactness = payload_loaded["superpixels_params"]["compactness"]

img = open_image(_convert_path(payload_loaded["paths"][image_idx]))
mask = safe_tensor_to_numpy(masks[image_idx])
asmap = safe_tensor_to_numpy(asmaps[image_idx])
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# resize image and asmap to double the resolution
img, asmap, mask = upscale_image_asmap_mask(img, asmap, mask, upscale_factor=upscale_factor)
valid_asmap, _ = valid_anomaly_score_maps(asmap[None, ...], min_thresh, return_mask=True)
valid_asmap = valid_asmap[0]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# reproduce the call of where the two above come from
(
    superpixels,
    superpixels_boundaries_distance_map,
    _,
    __,
    superpixels_original,
) = calculate_levelset_mean_dist_to_superpixel_boundaries_curve(
    img,
    asmap,
    min_thresh,
    watershed_superpixel_relsize,
    watershed_compactness,
    num_levelsets=num_levelsets,
    ret_superpixels_original=True,
)

# %%
import json

with (superpixel_oracle_selection_dir / "optimal_iou.json").open("r") as f:
    superpixel_oracle_selection_payload = json.load(f)


assert superpixel_oracle_selection_payload["superpixels_method"] == "watershed"

superpixel_oracle_selection = superpixel_oracle_selection_payload["results"][image_idx]

# %%
from aupimo.oracles_numpy import find_best_superpixels

history, selected_suppixs, available_suppixs = find_best_superpixels(superpixels_original, mask)

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage as sk


def _get_cmap_transparent_bad(cmap_name: str = "jet"):
    cmap = mpl.cm.get_cmap(cmap_name)
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def _get_binary_transparent_cmap(color) -> mpl.colors.ListedColormap:
    cmap = mpl.colors.ListedColormap([(0, 0, 0, 0), color, color, color])
    return cmap


fig, axes = plt.subplots(
    3,
    2,
    figsize=np.array((20, 30)),
    sharex=False,
    sharey=False,
    constrained_layout=True,
)
for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
axrow = axes.flatten()


def draw0(ax):
    imshow = ax.imshow(
        sk.segmentation.mark_boundaries(img, superpixels, color=mpl.colors.to_rgb("magenta")),
    )
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=3.5,
        linestyles="--",
    )
    _ = ax.contour(
        asmap,
        levels=[local_minima_threshs[2]],
        colors=["orange"],
        linewidths=3.5,
    )
    _ = ax.contour(
        asmap,
        levels=[local_minima_threshs[0]],
        colors=["yellow"],
        linewidths=3.5,
    )
    _ = ax.contour(
        asmap,
        levels=[local_minima_threshs[4]],
        colors=["red"],
        linewidths=3.5,
    )
    _ = ax.annotate(
        f"IoU: {local_minima_ious[0]:.0%} {local_minima_ious[1]:.0%} {local_minima_ious[2]:.0%}\n(yellow, orange, red)",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=40,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )


def draw1(ax):
    _ = ax.imshow(img)
    _ = ax.imshow(valid_asmap, cmap=_get_cmap_transparent_bad("jet"), alpha=0.45)
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=3.5,
        linestyles="--",
    )
    _ = ax.contour(
        asmap,
        levels=[max_iou_per_image_result.threshs[image_idx].item()],
        linewidths=3.5,
        colors=["w"],
    )
    _ = ax.annotate(
        f"IoU: {max_iou_per_image_result.ious[image_idx].item():.0%}",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=40,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )



def draw2(ax):
    _ = ax.imshow(valid_asmap, cmap=_get_cmap_transparent_bad("jet"), zorder=-10)
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=3.5,
        linestyles="--",
    )
    _ = ax.contour(
        asmap,
        levels=[local_minima_threshs[2]],
        colors=["orange"],
        linewidths=3.5,
    )
    _ = ax.contour(
        asmap,
        levels=[local_minima_threshs[0]],
        colors=["yellow"],
        linewidths=3.5,
    )
    _ = ax.contour(
        asmap,
        levels=[local_minima_threshs[4]],
        colors=["red"],
        linewidths=3.5,
    )


def draw3(ax):
    _ = ax.imshow(superpixels_boundaries_distance_map, cmap="cividis")
    _ = ax.contour(
        asmap,
        levels=[local_minima_threshs[3]],
        colors=["red"],
        linewidths=3.5,
    )
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=3.5,
        linestyles="--",
    )
    _ = ax.annotate(
        f"IoU: {local_minima_ious[2]:.0%}",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=40,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )



def draw4(ax):
    _ = ax.imshow(img)
    boundaries = sk.segmentation.find_boundaries(superpixels_original, mode="outer")
    _ = ax.imshow(boundaries, cmap=_get_binary_transparent_cmap("magenta"))
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=3.5,
        linestyles="--",
    )

    # smt went wrong with the superpixel selection SAVED
    # superpixel_selection_mask = np.isin(superpixels_original.astype(int), sorted(superpixel_oracle_selection["superpixels_selection"]))

    superpixel_selection_mask = np.isin(superpixels_original.astype(int), sorted(selected_suppixs))
    # _ = ax.imshow(superpixel_selection_mask, alpha=.3, cmap=_get_binary_transparent_cmap("orange"))
    _ = ax.contour(
        superpixel_selection_mask,
        levels=[0.5],
        colors="orange",
        linewidths=1.5,
        linestyles="-",
    )
    _ = ax.annotate(
        f"IoU: {superpixel_oracle_selection['iou']:.0%}",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=40,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )



draw0(axrow[0])
draw1(axrow[1])
draw2(axrow[2])
draw3(axrow[3])
draw4(axrow[4])

for ax in axrow[:2]:
    _ = ax.set_xlim(2048 * 0.22, 2048 * 0.78)
    _ = ax.set_ylim(2048 * 0.10, 2048 * 0.66)
for ax in axrow[2:]:
    _ = ax.set_xlim(2048 * 0.22, 2048 * 0.78)
    _ = ax.set_ylim(2048 * 0.12, 2048 * 0.46)
for ax in axes.flatten():
    _ = ax.invert_yaxis()

# %%
# plot the same as above but in individual figures each plot

fig0, ax = plt.subplots(figsize=(10, 10))
draw0(ax)
_ = ax.set_xlim(axrow[0].get_xlim())
_ = ax.set_ylim(axrow[0].get_ylim())
_ = ax.axis("off")

fig1, ax = plt.subplots(figsize=(10, 10))
draw1(ax)
_ = ax.set_xlim(axrow[1].get_xlim())
_ = ax.set_ylim(axrow[1].get_ylim())
_ = ax.axis("off")

fig2, ax = plt.subplots(figsize=(10, 10))
draw2(ax)
_ = ax.set_xlim(axrow[2].get_xlim())
_ = ax.set_ylim(axrow[2].get_ylim())
_ = ax.axis("off")

fig3, ax = plt.subplots(figsize=(10, 10))
draw3(ax)
_ = ax.set_xlim(axrow[3].get_xlim())
_ = ax.set_ylim(axrow[3].get_ylim())
_ = ax.axis("off")

fig4, ax = plt.subplots(figsize=(10, 10))
draw4(ax)
_ = ax.set_xlim(axrow[4].get_xlim())
_ = ax.set_ylim(axrow[4].get_ylim())
_ = ax.axis("off")

if args.savedir is not None:
    fig0.savefig(
        args.savedir / "image_gt_superpixels_3best_heuristic_segm.pdf",
        bbox_inches="tight",
        pad_inches=1e-2,
    )
    fig1.savefig(
        args.savedir / "image_gt_asmap_oracle_segm.pdf",
        bbox_inches="tight",
        pad_inches=1e-2,
    )
    fig2.savefig(
        args.savedir / "gt_asmap_3best_heuristic_segm.pdf",
        bbox_inches="tight",
        pad_inches=1e-2,
    )
    fig3.savefig(
        args.savedir / "superpixel_bound_dist_1best_heuristic_segm.pdf",
        bbox_inches="tight",
        pad_inches=1e-2,
    )
    fig4.savefig(
        args.savedir / "image_gt_superpixels_original_oracle_superpixel_selection.pdf",
        bbox_inches="tight",
        pad_inches=1e-2,
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
# annotate iou and contour distance
