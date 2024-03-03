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
import torch
from PIL import Image

# %%

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
_ = parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--asmaps ../../data/experiments/benchmark/efficientad_wr101_m_ext/mvtec/pill/asmaps.pt",
                "--mvtec-root ../../data/datasets/MVTec",
                "--visa-root ../../data/datasets/VisA",
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
from pathlib import Path

from matplotlib import pyplot as plt


def open_image(image_abspath: str | Path) -> np.ndarray:
    if (img := plt.imread(image_abspath)).ndim == 2:
        img = img[..., None].repeat(3, axis=-1)
    return img

images = [open_image(image_abspath) for image_abspath in images_abspaths]


# %%
from aupimo import AUPIMOResult

aupro = json.loads((args.asmaps.parent / "aupro.json").read_text())["value"]
auroc = json.loads((args.asmaps.parent / "auroc.json").read_text())["value"]
aupimos = AUPIMOResult.load(args.asmaps.parent / "aupimo/aupimos.json")

# %%
is_anomaly_mask = masks.sum(dim=(-1, -2)) > 0
sorted(
    zip(torch.where(is_anomaly_mask)[0].tolist(), aupimos.aupimos[is_anomaly_mask].cpu().numpy().tolist(), strict=True),
    key=lambda x: x[1],
)

# %%
is_normal_mask = masks.sum(dim=(-1, -2)) == 0
mean_ascore = asmaps.mean(dim=(-1, -2))
sorted(
    zip(torch.where(is_normal_mask)[0].tolist(), mean_ascore[is_normal_mask].cpu().numpy().tolist(), strict=True),
    key=lambda x: x[1],
)


mean_ascore = asmaps.mean(dim=(-1, -2))
p99_ascore = np.percentile(asmaps.cpu().numpy().astype(float), 99, axis=(-1, -2))
sorted(
    zip(
        range(len(is_anomaly_mask)),
        # np.round(mean_ascore.numpy().astype(float), decimals=2).tolist(),
        np.round(p99_ascore, decimals=2).tolist(),
        aupimos.aupimos.round(decimals=3).tolist(), strict=True),
    key=lambda x: x[1],
)

# %%

fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")

# image_idx = 132
image_idx = 131
norm_image = images[image_idx]
norm_gt_mask = masks[image_idx].cpu().numpy()
norm_asmap = asmaps[image_idx].cpu().numpy()

# image_idx = 0
# image_idx = 148
image_idx = 64
anom1_image = images[image_idx]
anom1_gt_mask = masks[image_idx].cpu().numpy()
anom1_asmap = asmaps[image_idx].cpu().numpy()

viz_asmaps = np.stack([norm_asmap, anom1_asmap], axis=0)
ascore_min, ascore_max = np.percentile(viz_asmaps, (70, 100))

# crop the images and the asmaps
vertical_crop_propotion = 0.15
vertical_crop_size = int(norm_image.shape[0] * vertical_crop_propotion)
vertical_crop_slice = slice(vertical_crop_size, -vertical_crop_size)
norm_image = norm_image[vertical_crop_slice, :, :]
norm_gt_mask = norm_gt_mask[vertical_crop_slice, :]
norm_asmap = norm_asmap[vertical_crop_slice, :]
anom1_image = anom1_image[vertical_crop_slice, :, :]
anom1_gt_mask = anom1_gt_mask[vertical_crop_slice, :]
anom1_asmap = anom1_asmap[vertical_crop_slice, :]

ax = axes[0, 0]
_ = ax.imshow(norm_image)
_ = ax.annotate(
    "Normal",
    xy=(0, 0), xycoords="axes fraction",
    ha="left", va="bottom",
    xytext=(10, 10), textcoords="offset points",
    fontsize=24, color="k",
    bbox={"facecolor": "white", "alpha": 1, "edgecolor": "k", "boxstyle": "round,pad=0.2"},
)

ax = axes[0, 1]
_ = ax.imshow(norm_asmap, cmap="jet", vmin=ascore_min, vmax=ascore_max)
_ = ax.contour(norm_gt_mask, [0.5], colors="k", linewidths=1, linestyles="--")

ax = axes[1, 0]
_ = ax.imshow(anom1_image)
_ = ax.contour(anom1_gt_mask, [0.5], colors="red", linewidths=2, linestyles="--")
# fill in the mask region with red
anom1_gt_mask_fill = anom1_gt_mask.astype(float)
anom1_gt_mask_fill[anom1_gt_mask_fill == 0] = np.nan
anom1_gt_mask_fill[0, 0] = 0
_ = ax.imshow(anom1_gt_mask_fill, cmap="Reds", alpha=.4)
_ = ax.annotate(
    "Anomalous",
    xy=(0, 0), xycoords="axes fraction",
    ha="left", va="bottom",
    xytext=(10, 10), textcoords="offset points",
    fontsize=24, color="k",
    bbox={"facecolor": "white", "alpha": 1, "edgecolor": "k", "boxstyle": "round,pad=0.2"},
)

ax = axes[1, 1]
_ = ax.imshow(anom1_asmap, cmap="jet", vmin=ascore_min, vmax=ascore_max)
_ = ax.contour(anom1_gt_mask, [0.5], colors="red", linewidths=2, linestyles="--")

for ax in axes.flatten():
    _ = ax.axis("off")

savedir = Path("/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src/img")

fig.savefig(savedir / "asmaps-worst-cases.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)

# %%
