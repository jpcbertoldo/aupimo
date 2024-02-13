"""TODO(jpcbertoldo): docstring in paper script.

author: jpcbertoldo
"""
#!/usr/bin/env python

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

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


from aupimo.oracles import IOUCurvesResult, MaxAvgIOUResult, MaxIOUPerImageResult
from aupimo.utils import per_image_scores_stats

# %%
# Args

# collection [of datasets] = {mvtec, visa}
ACCEPTED_COLLECTIONS = {
    (MVTEC_DIR_NAME := "MVTec"),
    (VISA_DIR_NAME := "VisA"),
}

parser = argparse.ArgumentParser()
_ = parser.add_argument("--rundir", type=Path, required=True)
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
_ = parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
_ = parser.add_argument("--save", action="store_true")

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--rundir ../../../data/experiments/benchmark/patchcore_wr50/mvtec/leather",
                "--mvtec-root ../../../data/datasets/MVTec",
                "--visa-root ../../../data/datasets/VisA",
                # "--save",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")
# %%
# verify that rundir contains a `asmaps.pt` file

if not (asmaps_pt := args.rundir / "asmaps.pt").is_file():
    msg = f"It looks like the rundir is not valid. {args.rundir=}"
    raise ValueError(msg)

else:
    print("rundir looks good")

# %%
# Load `asmaps.pt`

print("loading asmaps.pt")

assert (asmaps_path := args.rundir / "asmaps.pt").exists(), str(asmaps_path)  # noqa: RUF018

asmaps_dict = torch.load(asmaps_path)
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
# load iou stuff

iou_oracle_threshs_dir = args.rundir / "iou_oracle_threshs"

ioucurves_global = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_global_threshs.pt")

max_avg_iou_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou.json")
max_ious_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img.json")

max_avg_iou_min_thresh_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou_min_thresh.json")
max_ious_min_thresh_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

diff = max_ious_result.ious - max_avg_iou_result.ious_at_thresh
diff_min_thresh = max_ious_min_thresh_result.ious - max_avg_iou_min_thresh_result.ious_at_thresh

diff_stats = per_image_scores_stats(
    per_image_scores=diff,
    images_classes=max_ious_result.image_classes,
    only_class=1,
    repeated_replacement_atol=5e-2,
)
# remove the mean and the median
diff_stats = [stat for stat in diff_stats if stat["stat_name"] not in ("mean", "med")]

# %%
# diff stats viz v1 (with jet heatmap)

from aupimo.utils import valid_anomaly_score_maps

fig, axes = plt.subplots(
    2,
    2,
    figsize=np.array((4, 4)) * 3,
    sharex=True,
    sharey=True,
    layout="constrained",
)

for ax, stat in zip(axes.flatten(), diff_stats, strict=True):
    image_idx = stat["image_idx"]
    asmap = asmaps[image_idx]
    mask = masks[image_idx]
    img = plt.imread(images_abspaths[image_idx])
    if img.ndim == 2:
        img = img[..., None].repeat(3, axis=-1)

    # get the global maximum
    thresh_min = max_avg_iou_min_thresh_result.min_thresh
    thresh_global = max_avg_iou_result.thresh
    iou_at_global = max_avg_iou_result.ious_at_thresh[image_idx]
    thresh_local = max_ious_result.threshs[image_idx]
    iou_at_local = max_ious_result.ious[image_idx]

    vamap = valid_anomaly_score_maps(asmap[None, ...], thresh_min)[0]

    _ = ax.imshow(img)
    _ = ax.imshow(vamap, alpha=0.4, cmap="jet")
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=(lw := 2.5),
        linestyles="--",
    )
    cs_thresh_global = ax.contour(
        asmap,
        levels=[thresh_global],
        colors=["red"],
        linewidths=lw,
    )
    cs_thresh_local = ax.contour(
        asmap,
        levels=[thresh_local],
        colors=["blue"],
        linewidths=lw,
    )
    _ = ax.annotate(
        f"Image {image_idx}\nGlobal: {iou_at_global:.0%}\nLocal: {iou_at_local:.0%}",
        xy=(0, 1), xycoords="axes fraction",
        xytext=(10, -10), textcoords="offset points",
        ha="left", va="top",
        fontsize=20,
        bbox=dict(  # noqa: C408
            facecolor="white", alpha=1, edgecolor="black", boxstyle="round,pad=0.2",
        ),
    )

for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

# %%
# diff stats viz v2 (without heatmap, with validation level set)

from aupimo.utils import valid_anomaly_score_maps

fig, axes = plt.subplots(
    2,
    2,
    figsize=np.array((4, 4)) * 3,
    sharex=True,
    sharey=True,
    layout="constrained",
)

for ax, stat in zip(axes.flatten(), diff_stats, strict=True):
    image_idx = stat["image_idx"]
    asmap = asmaps[image_idx]
    mask = masks[image_idx]
    img = plt.imread(images_abspaths[image_idx])
    if img.ndim == 2:
        img = img[..., None].repeat(3, axis=-1)

    # get the global maximum
    thresh_min = max_avg_iou_min_thresh_result.min_thresh
    thresh_global = max_avg_iou_result.thresh
    iou_at_global = max_avg_iou_result.ious_at_thresh[image_idx]
    thresh_local = max_ious_result.threshs[image_idx]
    iou_at_local = max_ious_result.ious[image_idx]

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
    clabel_thresh_min = ax.clabel(
        cs_thresh_min, inline=True, fontsize=20,
        fmt="min. thresh",
    )
    cs_thresh_global = ax.contour(
        asmap,
        levels=[thresh_global],
        colors=["red"],
        linewidths=lw,
    )
    cs_thresh_local = ax.contour(
        asmap,
        levels=[thresh_local],
        colors=["blue"],
        linewidths=lw,
    )
    _ = ax.annotate(
        f"Image {image_idx}\nGlobal: {iou_at_global:.0%}\nLocal: {iou_at_local:.0%}",
        xy=(0, 1), xycoords="axes fraction",
        xytext=(10, -10), textcoords="offset points",
        ha="left", va="top",
        fontsize=20,
        bbox=dict(  # noqa: C408
            facecolor="white", alpha=1, edgecolor="black", boxstyle="round,pad=0.2",
        ),
    )

for ax in axes.flatten():
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
