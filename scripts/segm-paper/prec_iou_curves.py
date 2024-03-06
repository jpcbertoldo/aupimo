"""Evaluate the anomaly score maps (asmaps) of a model on a dataset.

Important: overwritting output files is the default behavior.

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
from PIL import Image
from torch import Tensor

# TODO change this datasetwise vocabulary
# TODO refactor make it an arg
ROOT_SAVEDIR = Path("/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src")
assert ROOT_SAVEDIR.exists()
assert (IMG_SAVEDIR := ROOT_SAVEDIR / "img").exists()
(SAVEDIR := IMG_SAVEDIR / "prec-vs-iou").mkdir(parents=True, exist_ok=True)


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
_ = parser.add_argument("--savedir", type=Path, default=None)

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                # "--asmaps ../../data/experiments/benchmark/efficientad_wr101_s_ext/mvtec/metal_nut/asmaps.pt",
                "--asmaps ../../data/experiments/benchmark/patchcore_wr50/mvtec/metal_nut/asmaps.pt",
                "--metrics auroc",
                "--metrics aupr",
                "--metrics aupro",
                "--metrics aupimo",
                "--mvtec-root ../../data/datasets/MVTec",
                "--visa-root ../../data/datasets/VisA",
                "--not-debug",
                "--savedir ../../../2024-03-segm-paper/src/img",
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
if False and args.debug:
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
from matplotlib import pyplot as plt

image_idx = -1
asmap = asmaps[image_idx]
mask = masks[image_idx]
img = plt.imread(images_abspaths[image_idx])
fig, axrow = plt.subplots(1, 3, figsize=(12, 4))
axrow[0].imshow(img)
axrow[1].imshow(mask)
axrow[2].imshow(asmap)
# %%
aupimoresult_fpath = args.asmaps.parent / "aupimo" / "aupimos.json"
from aupimo import AUPIMOResult

aupimoresult_loaded = AUPIMOResult.load(aupimoresult_fpath)
aupimo_thresh_upper_bound = aupimoresult_loaded.thresh_bounds[1]
print(f"{aupimo_thresh_upper_bound=}")
# %%
from functools import partial


def _asmap2vasmap(asmap: Tensor, min_valid_score: float) -> Tensor:
    """`vasmap` stands for `valid anomaly score map`."""
    vasmap = asmap.clone()
    vasmap[vasmap < min_valid_score] = torch.nan
    return vasmap


_asmap2vasmap_aupimo_bounded = partial(_asmap2vasmap, min_valid_score=aupimo_thresh_upper_bound)

fig, axrow = plt.subplots(1, 3, figsize=(12, 4))
axrow[0].imshow(img)
axrow[1].imshow(mask)
axrow[2].imshow(_asmap2vasmap_aupimo_bounded(asmap))

# %%
# precision curves (per-image binclf)

from aupimo import per_image_binclf_curve, per_image_prec, per_image_tpr

threshs_per_image = torch.stack(
    [
        torch.linspace(
            aupimo_thresh_upper_bound,
            aupimo_thresh_upper_bound * 1.01
            if (asmap.isnan().all() or (in_image_max := asmap.max()) <= aupimo_thresh_upper_bound)
            else in_image_max,
            1000,
        )
        for asmap in asmaps
    ],
    dim=0,
)

threshs_shared, per_image_binclfs_shared = per_image_binclf_curve(
    asmaps,
    masks,
    algorithm="numba",
    threshs_choice="minmax-linspace",
    num_threshs=30_000,
)

_, per_image_binclfs = per_image_binclf_curve(
    asmaps,
    masks,
    algorithm="numba",
    threshs_choice="given-per-image",
    threshs_given=threshs_per_image,
)

per_image_prec_values = per_image_prec(per_image_binclfs)
per_image_prec_values_shared = per_image_prec(per_image_binclfs_shared)

per_image_tpr_values_shared = per_image_tpr(per_image_binclfs_shared)

# %%
from aupimo import per_image_fpr
from aupimo.pimo_numpy import _images_classes_from_masks

fprs = per_image_fpr(per_image_binclfs_shared)
fprs_normals = fprs[_images_classes_from_masks(masks.numpy()) == 0]
shared_fpr = fprs_normals.mean(axis=0)

# %%

import matplotlib as mpl

prec_pivots = {
    11: [0.6, 0.62],
    # 40: [0.5, 0.9, 0.99],
    # 43: [0.8, 0.9, 0.99,],
    67: [0.95, 0.98],
    # 102: [0.5, 0.6, 0.63],
}

images_idxs = sorted(prec_pivots.keys())

fig, ax = plt.subplots(1, 1, figsize=np.array((5, 2)) * 1, layout="constrained")
_ = ax.plot(threshs_shared, per_image_prec_values_shared[images_idxs].T, label=images_idxs)
_ = ax.set_xlim(threshs_shared.min(), threshs_shared.max())
_ = ax.legend()
_ = ax.grid(axis="y", which="major", alpha=0.8, linestyle="--")

_ = ax.set_ylim(0, 1)
_ = ax.set_yticks(np.linspace(0, 1, 6))
_ = ax.set_yticks(np.linspace(0, 1, 11), minor=True)
_ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.grid(axis="y", linestyle="-", linewidth=1, alpha=0.5, which="both")
_ = ax.set_ylabel("Precision")

_ = ax.set_xlabel("Thresholds")
_ = ax.set_xlim(10, 60)

leg = ax.legend(loc="upper left", title="Image", framealpha=1, fontsize="small", ncol=2)

LETTERS = "AB"
for iteridx, imgidx in enumerate(images_idxs):
    pivots = prec_pivots[imgidx]
    asmap = asmaps[imgidx]
    img = plt.imread(images_abspaths[imgidx])

    thresh_idxs = np.searchsorted(per_image_prec_values_shared[imgidx], pivots)
    threshs = threshs_shared[thresh_idxs]
    threshs = np.sort(np.unique(threshs))
    prec_values = per_image_prec_values_shared[imgidx][thresh_idxs]

    fig_imshow, ax_imshow = plt.subplots(1, 1, figsize=(9, 9), layout="constrained")
    _ = ax_imshow.imshow(img)
    _ = ax_imshow.imshow(asmap, cmap="jet", alpha=0.4)
    cs = ax_imshow.contour(
        masks[imgidx],
        levels=[0.5],
        colors="black",
        linewidths=6,
        linestyles="--",
    )
    # _ = ax.clabel(cs, cs.levels, inline=True, fmt={cs.levels[0]: "ground truth"}, fontsize=10)
    cs = ax_imshow.contour(
        asmap,
        threshs,
        colors=["white", "white"],
        linewidths=6,
    )
    # fmt = {l: f"{val:.0%}" for l, val in zip(cs.levels, prec_values, strict=False)}
    fmt = {l: LETTERS[enum] for enum, l in enumerate(cs.levels)}
    _ = ax_imshow.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=20)
    pivots_str = ", ".join(
        [
            f"{LETTERS[enum]} ({p:.0%}|{per_image_tpr_values_shared[imgidx, thidx]:.0%})"
            for enum, (p, thidx) in enumerate(zip(pivots, thresh_idxs, strict=False))
        ]
    )
    _ = ax_imshow.annotate(
        f"Image {imgidx} contours (precision|recall)\n{pivots_str}",
        xy=(0.5, 0),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=22,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )
    _ = ax_imshow.axis("off")

    ax.scatter(threshs, prec_values, s=50, c=ax.get_legend_handles_labels()[0][iteridx].get_color())

    if args.savedir is not None:
        fig_imshow.savefig(args.savedir / f"precision_contours_{imgidx}.pdf", bbox_inches="tight", pad_inches=1e-2)

if args.savedir is not None:
    fig.savefig(args.savedir / "precision_curves.pdf", bbox_inches="tight", pad_inches=1e-2)

# %%
from aupimo import per_image_iou

per_image_iou_values = per_image_iou(per_image_binclfs)
iou_curves = torch.stack([threshs_per_image, per_image_iou_values], dim=1)

per_image_iou_values_shared = per_image_iou(per_image_binclfs_shared)

# %%

fig, ax = plt.subplots(1, 1, figsize=np.array((5, 2)) * 1, layout="constrained")
_ = ax.plot(threshs_shared, per_image_iou_values_shared[images_idxs].T, label=images_idxs)
_ = ax.set_xlabel("Threshold")
_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1.02)
_ = ax.set_xlim(threshs_shared.min(), threshs_shared.max())
_ = ax.legend(loc="upper left", title="Image", framealpha=1, fontsize="small", ncol=2)
_ = ax.grid(axis="y", which="major", alpha=0.8, linestyle="--")

_ = ax.set_ylim(0, 1)
_ = ax.set_yticks(np.linspace(0, 1, 6))
_ = ax.set_yticks(np.linspace(0, 1, 11), minor=True)
_ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.grid(axis="y", linestyle="-", linewidth=1, alpha=0.5, which="both")
_ = ax.set_ylabel("IoU")

_ = ax.set_xlabel("Thresholds")
_ = ax.set_xlim(10, 60)

for iteridx, imgidx in enumerate(images_idxs):
    iou_curve = per_image_iou_values_shared[imgidx]
    max_thresh_idx = np.argmax(iou_curve).item()
    trhesh = threshs_shared[max_thresh_idx]
    maxiou = per_image_iou_values_shared[imgidx][max_thresh_idx]

    _ = ax.scatter(trhesh, maxiou, marker="*", s=100, c=ax.get_legend_handles_labels()[0][iteridx].get_color())

    pivot_value = maxiou * 0.90
    pivots_thresh_idxs = np.argsort(np.abs(iou_curve - pivot_value))[:10].numpy()
    pivots_thresh_idxs = np.sort(np.unique(pivots_thresh_idxs))[[0, -1]]
    _ = ax.scatter(
        threshs_shared[pivots_thresh_idxs],
        iou_curve[pivots_thresh_idxs],
        marker="o",
        s=50,
        c=ax.get_legend_handles_labels()[0][iteridx].get_color(),
    )

    all_pivots_thresh_idxs = sorted(np.unique(np.concatenate([pivots_thresh_idxs, [max_thresh_idx]])))
    threshs = threshs_shared[all_pivots_thresh_idxs]
    iou_values = iou_curve[all_pivots_thresh_idxs]

    asmap = asmaps[imgidx]
    img = plt.imread(images_abspaths[imgidx])

    fig_imshow, ax_imshow = plt.subplots(1, 1, figsize=(9, 9), layout="constrained")
    _ = ax_imshow.imshow(img)
    _ = ax_imshow.imshow(asmap, cmap="jet", alpha=0.4)

    cs = ax_imshow.contour(
        masks[imgidx],
        levels=[0.5],
        colors="black",
        linewidths=6,
        linestyles="--",
    )
    # _ = ax.clabel(cs, cs.levels, inline=True, fmt={cs.levels[0]: "ground truth"}, fontsize=10)

    cs = ax_imshow.contour(
        asmap,
        threshs,
        colors=["lightgray", "white", "lightgray"],
        linewidths=4,
    )  # _ = ax.clabel(cs, cs.levels, inline=True, fmt={cs.levels[0]: "ground truth"}, fontsize=10)

    # cs = ax_imshow.contour(asmap, threshs, lw=3, colors=["black", "black", "black"], labels=pivots)
    # fmt = {l: f"{let}" for l, let in zip(cs.levels, "ABC", strict=False)}
    # _ = ax_imshow.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=20)
    # pivots_str = [f"{p:.0%} ({per_image_tpr_values_shared[imgidx, thidx]:.0%})" for p, thidx in zip(iou_values, all_pivots_thresh_idxs)]
    pivots_str = [f"{p:.0%}" for p, thidx in zip(iou_values, all_pivots_thresh_idxs, strict=False)]
    pivots_str[0] += " [left]"
    pivots_str[1] += " [max]"
    pivots_str[-1] += " [right]"
    pivots_str = ", ".join(pivots_str)
    _ = ax_imshow.axis("off")
    _ = ax_imshow.annotate(
        f"Image {imgidx} contours\nIoU: {pivots_str}",
        xy=(0.5, 0),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=22,
        bbox=dict(  # noqa: C408
            facecolor="white",
            alpha=1,
            edgecolor="black",
            boxstyle="round,pad=0.2",
        ),
    )

    # fig_imshow.savefig(SAVEDIR / f"iou_contours_{imgidx}.pdf", bbox_inches="tight", pad_inches=1e-2)
    if args.savedir is not None:
        fig_imshow.savefig(args.savedir / f"iou_contours_{imgidx}.pdf", bbox_inches="tight", pad_inches=1e-2)

# fig.savefig(SAVEDIR / f"iou_curves.pdf", bbox_inches="tight", pad_inches=1e-2)

if args.savedir is not None:
    fig.savefig(args.savedir / "iou_curves.pdf", bbox_inches="tight", pad_inches=1e-2)

# %%
# shared fpr vs iou

fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.5), layout="constrained")
_ = ax.plot(shared_fpr, per_image_iou_values_shared[images_idxs].T, label=images_idxs)

_ = ax.axvspan(1e-5, 1e-4, alpha=0.2, color="tab:blue", label="AUC")

_ = ax.set_xscale("log")
_ = ax.minorticks_off()
_ = ax.set_xlim(1e-5, 1)
_ = ax.set_xlabel("Shared FPR (log)")

_ = ax.set_ylabel("IoU")
_ = ax.set_ylim(0, 1.02)
_ = ax.legend(loc="upper center", title="Image index", ncols=6, columnspacing=1)

# fig.savefig(SAVEDIR / f"shared_fpr_vs_iou.pdf", bbox_inches="tight", pad_inches=1e-2)

# %%
