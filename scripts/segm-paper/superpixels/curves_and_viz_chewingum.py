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
                # "--asmaps ../../../data/experiments/benchmark/efficientad_wr101_s_ext/mvtec/transistor/asmaps.pt",
                "--asmaps ../../../data/experiments/benchmark/efficientad_wr101_s_ext/visa/chewinggum/asmaps.pt",
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
superpixel_oracle_selection_dir = Path("/".join(rundir.parts[:-3] + ("patchcore_wr50", ) + rundir.parts[-2:] + ("superpixel_oracle_selection",) ))

# %%


ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_local_threshs.pt")
max_iou_per_image_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

payload_loaded = torch.load(superpixel_bound_dist_heuristic_dir / "superpixel_bound_dist_heuristic.pt")

# transistor
image_idx = 18  # missing leg, good one
# image_idx = 94  # upsidedown
# image_idx = 21  # broken part, shows limitation of multi-region level set

# chewinggum
image_idx = 3

threshs = payload_loaded["threshs_per_image"][image_idx]
heuristic_curve_values = payload_loaded["levelset_mean_dist_curve_per_image"][image_idx]
num_levelsets = threshs.shape[0]
min_thresh = payload_loaded["min_thresh"]
upscale_factor = payload_loaded["upscale_factor"]

# local_minima_idxs = np.array(payload_loaded["local_minima_idxs_per_image"][image_idx][:5])
local_minima_idxs = np.array(payload_loaded["local_minima_idxs_per_image"][image_idx])
local_minima_threshs = threshs[local_minima_idxs]
local_minima_values = heuristic_curve_values[local_minima_idxs]

# local_minima_idxs_bis = np.searchsorted(ioucurves.threshs[image_idx], local_minima_threshs)
local_minima_idxs_bis = np.argmin(np.abs(ioucurves.threshs[image_idx][None, ...] - local_minima_threshs[..., None]), axis=1)
local_minima_ious = ioucurves.per_image_ious[image_idx][local_minima_idxs_bis]
local_minima_ious_argsorted = np.argsort(local_minima_ious)
chosen = [0, len(local_minima_ious_argsorted) // 2, len(local_minima_ious_argsorted)-1]
local_minima_idxs_bis = local_minima_idxs_bis[local_minima_ious_argsorted[chosen]]
local_minima_ious = ioucurves.per_image_ious[image_idx][local_minima_idxs_bis]
local_minima_threshs_bis = ioucurves.threshs[image_idx, local_minima_idxs_bis]

local_minima_args = np.argmin(np.abs(local_minima_threshs[None, ...] - local_minima_threshs_bis[..., None]), axis=1)
local_minima_idxs = local_minima_idxs[local_minima_args]
local_minima_threshs = threshs[local_minima_idxs]
local_minima_values = heuristic_curve_values[local_minima_idxs]

# local_minima_idxs = np.sorted(lo)

img = open_image(_convert_path(payload_loaded["paths"][image_idx]))
mask = safe_tensor_to_numpy(masks[image_idx])
asmap = safe_tensor_to_numpy(asmaps[image_idx])
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# resize image and asmap to double the resolution
img, asmap, mask = upscale_image_asmap_mask(img, asmap, mask, upscale_factor=upscale_factor)
valid_asmap, _ = valid_anomaly_score_maps(asmap[None, ...], min_thresh, return_mask=True)
valid_asmap = valid_asmap[0]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import matplotlib as mpl
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=np.array((6, 4)) * .75)
_ = ax.plot(
    threshs,
    heuristic_curve_values,
    color="tab:red",
    label="Heuristic signal",
)
_ = ax.scatter(
    local_minima_threshs, local_minima_values,
    color="tab:red",  marker="o", s=150, zorder=10,
)
_ = ax.plot(ioucurves.threshs[image_idx], ioucurves.per_image_ious[image_idx], color="black", label="IoU")
_ = ax.scatter(
    local_minima_threshs_bis, local_minima_ious,
    color="black",  marker="o", s=150, zorder=10,
)
_ = ax.axvline(max_iou_per_image_result.threshs[image_idx], color="black", label="Oracle Thresh.", linestyle="--")
_ = ax.axvline(min_thresh, color="gray", label="Val. Thresh.", linestyle="--")

_ = ax.set_ylim(0, 1)
_ = ax.set_yticks(np.linspace(0, 1, 6))
_ = ax.set_yticks(np.linspace(0, 1, 11), minor=True)
_ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.grid(axis="y", linestyle="-", linewidth=1, alpha=0.5, which="both")
_ = ax.set_ylabel("IoU and Heuristic Signal")

_ = ax.set_xlabel("Thresholds")
_ = ax.set_xlim(left=0, right=2.5)
_ = ax.set_xticks(np.linspace(0, 2.5, 6))

leg = ax.legend(loc="upper right", framealpha=1, fontsize="small", ncol=2)

if args.savedir is not None:
    fig.savefig(args.savedir / "chewinggum_00_curves.pdf", bbox_inches="tight", pad_inches=1e-2,)


# %%


def _get_cmap_transparent_bad(cmap_name: str):
    cmap = mpl.cm.get_cmap(cmap_name)
    cmap.set_bad((0, 0, 0, 0))
    return cmap


fig, axes = plt.subplots(
    1,
    2,
    figsize=np.array((12, 6)) * 1,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axrow = axes.flatten()

def draw0(ax):
    _ = ax.imshow(img)
    _ = ax.imshow(valid_asmap, cmap=_get_cmap_transparent_bad("jet"), alpha=0.85)
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="--",
    )
    _ = ax.contour(
        asmap,
        levels=[max_iou_per_image_result.threshs[image_idx].item()],
        colors=["white"],
        linewidths=4.5,
    )


def draw1(ax):
    _ = ax.imshow(img)
    _ = ax.contour(
        asmap,
        levels=np.sort(local_minima_threshs),
        linewidths=3.5,
        colors=["yellow", "orange", "red"],
    )
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="--",
    )

draw0(axrow[0])
draw1(axrow[1])

for ax in axrow:
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_xlim(img.shape[0] * 0.22, img.shape[1] * .78)
    _ = ax.set_ylim(img.shape[0] * 0.30, img.shape[1] * .69)
    # _ = ax.invert_yaxis()

# %%

fig0, ax = plt.subplots(figsize=(8, 8))
draw0(ax)
_ = ax.set_xlim(axrow[0].get_xlim())
_ = ax.set_ylim(axrow[0].get_ylim())
_ = ax.set_xticks([])
_ = ax.set_yticks([])

fig1, ax = plt.subplots(figsize=(8, 8))
draw1(ax)
_ = ax.set_xlim(axrow[1].get_xlim())
_ = ax.set_ylim(axrow[1].get_ylim())
_ = ax.set_xticks([])
_ = ax.set_yticks([])

if args.savedir is not None:
    fig0.savefig(args.savedir / "chewinggum_00_heatmap_oracle.pdf", bbox_inches="tight", pad_inches=1e-2,)
    fig1.savefig(args.savedir / "chewinggum_00_heuristic_choices.pdf", bbox_inches="tight", pad_inches=1e-2,)

# %%
