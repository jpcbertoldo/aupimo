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


from aupimo import aupimo_scores, per_image_iou_curves

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
    (METRIC_IOU_CURVES_GLOBAL := "ioucurves_global"),
    (METRIC_IOU_CURVES_LOCAL := "ioucurves_local"),
    (METRIC_MAX_AVG_IOU := "max_avg_iou"),
    (METRIC_MAX_IOU_PER_IMG := "max_iou_per_img"),
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
                # "--metrics max_avg_iou",
                # "--metrics max_iou_per_img",
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

    aupimo_dir = savedir / "aupimo"
    if args.debug:
        aupimo_dir = aupimo_dir.with_stem("debug_" + aupimo_dir.stem)
    aupimo_dir.mkdir(exist_ok=True)

    pimoresult.save(aupimo_dir / "curves.pt")
    aupimoresult.save(aupimo_dir / "aupimos.json")

# %%
# iou curves with shared thresholds
if METRIC_IOU_CURVES_GLOBAL in args.metrics:
    ioucurves = per_image_iou_curves(asmaps, masks, num_threshs=10_000, common_threshs=True, paths=images_relpaths)
    ioucurves.save(savedir / "ioucurves_global_threshs.pt")

# %%
# iou curves with local thresholds
if METRIC_IOU_CURVES_LOCAL in args.metrics:
    ioucurves = per_image_iou_curves(asmaps, masks, num_threshs=10_000, common_threshs=False, paths=images_relpaths)
    ioucurves.save(savedir / "ioucurves_local_threshs.pt")

# %%
# iou max avg iou
if METRIC_MAX_AVG_IOU in args.metrics:
    from aupimo import IOUCurvesResult
    from aupimo.oracles import max_avg_iou

    ioucurves = IOUCurvesResult.load(savedir / "ioucurves_global_threshs.pt")
    if args.debug:
        ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]
    max_avg_iou_result = max_avg_iou(ioucurves.threshs, ioucurves.per_image_ious, paths=images_relpaths)
    max_avg_iou_result.save(savedir / "max_avg_iou.json")


# %%
# max iou per image
if METRIC_MAX_IOU_PER_IMG in args.metrics:
    from aupimo import IOUCurvesResult
    from aupimo.oracles import max_iou_per_image

    ioucurves = IOUCurvesResult.load(savedir / "ioucurves_local_threshs.pt")
    if args.debug:
        ioucurves.threshs = ioucurves.threshs[some_imgs, :]
        ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]
    ious_maxs_result = max_iou_per_image(ioucurves.threshs, ioucurves.per_image_ious, paths=images_relpaths)
    ious_maxs_result.save(savedir / "max_iou_per_img.json")



# %%
# TODO include option for min valid threshold


# %%
# viz iou strategies
import matplotlib as mpl
from matplotlib import pyplot as plt

from aupimo.oracles import IOUCurvesResult, MaxAvgIOUResult, MaxIOUPerImageResult
from aupimo.utils import per_image_scores_stats

max_avg_iou_result = MaxAvgIOUResult.load(savedir / "max_avg_iou.json")
max_ious_result = MaxIOUPerImageResult.load(savedir / "max_iou_per_img.json")
ioucurves = IOUCurvesResult.load(savedir / "ioucurves_global_threshs.pt")
if args.debug:
    ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]

diff = max_ious_result.ious - max_avg_iou_result.ious_at_thresh

fig, ax = plt.subplots()
ioucurves.plot_avg_iou_curve(ax)

_ = ax.scatter(
    max_avg_iou_result.thresh,
    max_avg_iou_result.avg_iou,
    color="black",
    s=200,
    marker="*",
    label="max avg iou",
)
_ = ax.legend(loc="upper left")

fig, axes = plt.subplots(
    2,
    1,
    figsize=(7, 5),
    height_ratios=[2, 1],
    sharex=True,
    layout="constrained",
)
ax = axes[0]
_ = ax.boxplot(
    [
        max_avg_iou_result.ious_at_thresh[max_avg_iou_result.image_classes == 1],
        max_ious_result.ious[max_ious_result.image_classes == 1],
    ],
    labels=[
        "@ max avg iou",
        "@ optimal",
    ],
    vert=False,
    notch=False,
)
_ = ax.scatter(
    max_avg_iou_result.avg_iou,
    1,
    color="black",
    s=200,
    marker="*",
)

ax = axes[1]
_ = ax.boxplot(
    diff[max_ious_result.image_classes == 1],
    labels=["diff"],
    vert=False,
    notch=False,
)

_ = ax.set_xlabel("IoU")

_ = ax.set_xlim(0 - (eps := 1e-2), 1 + eps)
_ = ax.set_xticks(np.linspace(0, 1, 5))
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

for ax in axes:
    _ = ax.grid(axis="x")

fig, ax = plt.subplots()
boxplot_stats = per_image_scores_stats(
    # per_image_scores=ious_at_max_avg_iou,
    per_image_scores=diff,
    images_classes=max_ious_result.image_classes,
    only_class=1,
    repeated_replacement_atol=5e-2,
)
for bp_stat in boxplot_stats:
    iou_curve = ioucurves.per_image_ious[bp_stat["image_idx"]]
    line = ax.plot(
        ioucurves.threshs,
        iou_curve,
        label=f"{bp_stat['stat_name']} idx={bp_stat['image_idx']}",
        alpha=0.7,
    )[0]
    _ = ax.scatter(
        max_ious_result.threshs[bp_stat["image_idx"]],
        max_ious_result.ious[bp_stat["image_idx"]],
        color=line.get_color(),
        s=200,
        marker="*",
    )

_ = ax.axvline(
    x=max_avg_iou_result.thresh,
    color="black",
    linestyle="--",
    alpha=0.5,
)
_ = ax.legend(loc="upper left")

_ = ax.set_ylim(0 - (eps := 1e-2), 1 + eps)
_ = ax.set_yticks(np.linspace(0, 1, 5))
_ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.grid(axis="y")

_ = ax.set_xlabel("Threshold")
_ = ax.set_ylabel("IoU")

# %%
