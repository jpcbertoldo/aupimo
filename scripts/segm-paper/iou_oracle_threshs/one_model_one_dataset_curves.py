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

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--rundir ../../../data/experiments/benchmark/patchcore_wr50/mvtec/bottle",
                "--mvtec-root ../../../data/datasets/MVTec",
                "--visa-root ../../../data/datasets/VisA",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

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

# %% avg iou curve
fig, ax = plt.subplots(figsize=[6, 4])

ioucurves_global.plot_avg_iou_curve(ax)
ioucurves_global.ax_cfg_xaxis(ax)
ioucurves_global.ax_cfg_yaxis(ax)

_ = ioucurves_global.axvline_min_thresh(ax, max_avg_iou_min_thresh_result.min_thresh)
_ = ioucurves_global.axvline_global_oracle(ax, max_avg_iou_min_thresh_result.thresh)
_ = ioucurves_global.scatter_global_oracle(
    ax,
    max_avg_iou_min_thresh_result.thresh,
    max_avg_iou_min_thresh_result.avg_iou,
)

_ = ax.legend(loc="upper left", framealpha=1, fontsize="small")

# %%
# diff stats curves

fig, ax = plt.subplots(figsize=np.array((6, 4)))

for bp_stat, color in zip(diff_stats, ["tab:orange", "tab:purple", "tab:brown", "tab:cyan"], strict=True):
    iou_curve = ioucurves_global.per_image_ious[bp_stat["image_idx"]]
    line = ioucurves_global.plot_iou_curve(ax, bp_stat["image_idx"], color=color)[0]
    _ = ioucurves_global.scatter_local_oracle(
        ax,
        max_ious_result.threshs[bp_stat["image_idx"]],
        max_ious_result.ious[bp_stat["image_idx"]],
        color=line.get_color(),
    )
_ = ioucurves_global.axvline_min_thresh(ax, max_avg_iou_min_thresh_result.min_thresh)
_ = ioucurves_global.axvline_global_oracle(ax, max_avg_iou_min_thresh_result.thresh)
leg = ax.legend(loc="upper left", framealpha=1, fontsize="small")
for line in leg.get_lines():
    line.set_linewidth(2.5)
ioucurves_global.ax_cfg_xaxis(ax)
ioucurves_global.ax_cfg_yaxis(ax)

# %%
