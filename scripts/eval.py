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
    (METRIC_AUPRO_05 := "aupro_05"),
    (METRIC_IOU := "iou"),
]

parser = argparse.ArgumentParser()
_ = parser.add_argument("--asmaps", type=Path, required=True)
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
_ = parser.add_argument("--not-debug", dest="debug", action="store_false")
_ = parser.add_argument("--metrics", "-me", type=str, action="append", choices=METRICS_CHOICES, default=[])
_ = parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
_ = parser.add_argument("--seed", type=int, default=0)
_ = parser.add_argument("--add-tiny-regions", action="store_true")

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--asmaps ../data/experiments/benchmark/padim_r18/mvtec/bottle/asmaps.pt",
                "--metrics auroc",
                "--metrics aupr",
                "--metrics aupro",
                "--metrics aupimo",
                "--metrics aupro_05",
                "--metrics iou",
                "--mvtec-root ../data/datasets/MVTec",
                "--visa-root ../data/datasets/VisA",
                # "--add-tiny-regions",
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

if args.add_tiny_regions:
    print("modifying savedir to include `synthetic_tiny_regions`")
    savedir = savedir / "synthetic_tiny_regions"
    savedir.mkdir(exist_ok=True)

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
# Add tiny regions to the masks
# parameters are based on statistics from VisA 

if args.add_tiny_regions:

    RATIO_IMAGES_WITH_TINY_REGIONS = 0.15
    RATIO_REGIONS_SIZE_SMALLER_20PCT = 0.80
    MAX_NUM_ATTEMPTS_TO_PLACE_REGION = 10

    synthetic_tiny_regions_params_dir = savedir / "synthetic_tiny_regions_params"
    if args.debug:
        synthetic_tiny_regions_params_dir = synthetic_tiny_regions_params_dir.with_stem("debug_" + synthetic_tiny_regions_params_dir.stem)
    synthetic_tiny_regions_params_dir.mkdir(exist_ok=True)

    params_fpath = synthetic_tiny_regions_params_dir / f"seed={args.seed}.json"
    params_dict = {"seed": args.seed}

    generator = torch.Generator().manual_seed(args.seed)

    try:
        # get the number of images to add tiny regions to
        images_classes = torch.tensor([abspath is not None for abspath in masks_abspaths])
        num_anom = images_classes.sum().item()
        num_anom_to_add_tiny_regions = int(RATIO_IMAGES_WITH_TINY_REGIONS * num_anom)
        params_dict["num_anom_to_add_tiny_regions"] = num_anom_to_add_tiny_regions

        # randomly which images to add tiny regions to
        idxs_anom_images = torch.where(images_classes == 1)[0]
        idxs_to_add_tiny_regions = idxs_anom_images[torch.randperm(len(idxs_anom_images), generator=generator)]
        idxs_to_add_tiny_regions = idxs_to_add_tiny_regions[:num_anom_to_add_tiny_regions].tolist()
        params_dict["idxs_to_add_tiny_regions"] = idxs_to_add_tiny_regions

        # how many tiny regions to add to each image
        num_tiny_regions_per_image = torch.randint(low=1, high=5, size=(num_anom_to_add_tiny_regions,), generator=generator).tolist()
        params_dict["num_tiny_regions_per_image"] = num_tiny_regions_per_image

        # where to add the tiny regions
        params_dict[f"tiny_region_size_per_image"] = {}
        for image_idx, num_tiny_regions in zip(idxs_to_add_tiny_regions, num_tiny_regions_per_image):
            mask = masks[image_idx].clone()
            # get the size of the tiny regions
            # 80% of chance to have size from 1 to 9; 20% of chance to have size from 10 to 19
            smaller_than_10 = torch.rand(num_tiny_regions, generator=generator) < RATIO_REGIONS_SIZE_SMALLER_20PCT
            tiny_region_size_smaller_than_10 = torch.randint(low=1, high=10, size=(num_tiny_regions,), generator=generator)
            tiny_region_size_bigger_than_10 = torch.randint(low=10, high=20, size=(num_tiny_regions,), generator=generator)
            tiny_region_size = torch.where(smaller_than_10, tiny_region_size_smaller_than_10, tiny_region_size_bigger_than_10).tolist()
            params_dict[f"tiny_region_size_per_image"][image_idx] = tiny_region_size
            for abs_region_size in tiny_region_size:
                # get a random shape for the region
                square_size = torch.sqrt(torch.tensor(abs_region_size)).ceil().long().item()
                region = torch.zeros((square_size ** 2))
                region[:abs_region_size] = 1
                region = region[torch.randperm(len(region), generator=generator)]
                region = region.view(square_size, square_size)
                # get the position of the tiny region
                # make sure the position chosen is not outside the image
                not_outside_mask = torch.ones_like(mask)  
                not_outside_mask[(-square_size):, :] = 0
                not_outside_mask[:, (-square_size):] = 0
                possible_positions = torch.tensor(np.where((mask == 0) & not_outside_mask)).T
                possible_positions = possible_positions[torch.randperm(len(possible_positions), generator=generator)]
                for position in possible_positions[:MAX_NUM_ATTEMPTS_TO_PLACE_REGION]:
                    x, y = position
                    touches_existing_region = (region * mask[x:x+square_size, y:y+square_size]).sum() > 0
                    if touches_existing_region:
                        continue
                    mask[x:x+square_size, y:y+square_size] = region
                    break
            
            print(f"modifying mask {image_idx=}")
            masks[image_idx] = mask

    except Exception as ex:
        params_dict["exception"] = str(ex)
        # save the stack trace as a string in `params_dict`
        import traceback
        params_dict["stack_trace"] = traceback.format_exc()

    finally:
        # save the parameters
        with params_fpath.open("w") as f:
            json.dump(params_dict, f, indent=4)

# %%
# DEBUG: only keep 2 images per class if in debug mode
if args.debug:
    print("debug mode --> only using 2 images")
    imgclass = (masks == 1).any(dim=-1).any(dim=-1).to(torch.bool)
    some_norm = torch.where(imgclass == 0)[0][:2]
    some_anom = torch.where(imgclass == 1)[0][:2]
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
    masks = masks.cpu()
    asmaps = asmaps.cpu()

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

    for num_threshs in [
        # the number of thsreshs in the AUC cannot be predicted to 
        # we try different values of threhs in the curve until we find a good one
        30_000, 100_000, 300_000, 1_000_000,
    ]:
        print(f"trying {num_threshs=}")
        pimoresult, aupimoresult = aupimo_scores(
            asmaps, masks,
            fpr_bounds=(1e-5, 1e-4),
            paths=images_relpaths,  # relative, not absolute paths!
            num_threshs=num_threshs,
        )
        print(f"auc num threshs = {aupimoresult.num_threshs=}")
        if aupimoresult.num_threshs < 100:
            print("too few thresholds, trying again with more")
            continue
        break

    print("saving aupimo")

    aupimo_dir = savedir / "aupimo"
    if args.debug:
        aupimo_dir = aupimo_dir.with_stem("debug_" + aupimo_dir.stem)
    aupimo_dir.mkdir(exist_ok=True)

    pimoresult.save(aupimo_dir / "curves.pt")
    aupimoresult.save(aupimo_dir / "aupimos.json")

# =============================================================================
# REVIEW
# %%
# AUPRO_05

def _compute_aupro_05(asmaps: Tensor, masks: Tensor) -> float:
    metric = AUPRO(fpr_limit=0.05)
    metric.update(asmaps, masks)
    return metric.compute().item()

if METRIC_AUPRO_05 in args.metrics:
    print("computing aupro_05 (fpr upper bound of 5% instead of 30%)")
    aupro_05 = _compute_aupro_05(asmaps, masks)
    print(f"{aupro_05=}")
    _save_value(savedir / "aupro_05.json", aupro_05, args.debug)
    
# %%
# IOU

# TODO clean this up
from numpy import ndarray
from aupimo import per_image_binclf_curve, per_image_iou, per_image_fpr
from aupimo.pimo_numpy import aupimo_normalizing_factor

# def _compute_iou(asmaps: Tensor, masks: Tensor) -> float:

if METRIC_IOU in args.metrics:
    print("computing iou")
    
    # TODO this path reading could be done in a more elegant way
    assert (aupimo_auc_fpath := savedir / "aupimo" / "aupimos.json").exists(), f"{aupimo_auc_fpath=}"
    aupimoresult = AUPIMOResult.load(aupimo_auc_fpath)
    threshs = torch.linspace(*aupimoresult.thresh_bounds, IOU_NUM_THRESHS := 1000)
    _, binclf_curves = per_image_binclf_curve(
        anomaly_maps=asmaps,
        masks=masks,
        algorithm="numba",
        threshs_choice="given",
        threshs_given=threshs,
    )
    iou_curves = per_image_iou(binclf_curves)
        
    iou_dir = savedir / "iou"
    if args.debug:
        iou_dir = iou_dir.with_stem("debug_" + iou_dir.stem)
    iou_dir.mkdir(exist_ok=True)
    
    torch.save(iou_curves, iou_dir / "curves.pt")
    
    fprs = per_image_fpr(binclf_curves)
    images_classes = (masks == 1).any(dim=-1).any(dim=-1).to(torch.bool)
    # this is already in the AUPIMO's bounds
    shared_fpr = fprs[images_classes == 0].mean(dim=0)
    
    shared_fpr = shared_fpr.cpu().numpy()
    iou_curves = iou_curves.cpu().numpy()
    
    # -------------------------------------------------------------------------    
    # kind copied from `aupimo_scores`

    # `shared_fpr` is in descending order; `flip()` reverts to ascending order
    # `iou_curves` have to be flipped as well to match the new `shared_fpr`
    shared_fpr = np.flip(shared_fpr)
    iou_curves = np.flip(iou_curves, axis=1)

    # the log's base does not matter because it's a constant factor canceled by normalization factor
    shared_fpr_log = np.log(shared_fpr)

    # deal with edge cases
    invalid_shared_fpr = ~np.isfinite(shared_fpr_log)

    if invalid_shared_fpr.all():
        msg = (
            "Cannot compute AUPIMO because the shared fpr integration range is invalid). "
            "Try increasing the number of thresholds."
        )
        with (iou_dir / "error-00.txt").open("w") as f:
            f.write(msg)
        raise RuntimeError(msg)    

    if invalid_shared_fpr.any():
        msg = (
            "Some values in the shared fpr integration range are nan. "
            "The AUPIMO will be computed without these values."
        )
        with (iou_dir / "warning-00.txt").open("w") as f:
            f.write(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=1)

        # get rid of nan values by removing them from the integration range
        shared_fpr_log = shared_fpr_log[~invalid_shared_fpr]
        iou_curves = iou_curves[:, ~invalid_shared_fpr]

    num_points_integral = int(shared_fpr_log.shape[0])

    aucs: ndarray = np.trapz(iou_curves, x=shared_fpr_log, axis=1)

    # normalize, then clip(0, 1) makes sure that the values are in [0, 1] in case of numerical errors
    normalization_factor = aupimo_normalizing_factor(aupimoresult.fpr_bounds)
    aucs = (aucs / normalization_factor).clip(0, 1)

    # TODO fix to save inside `iou_dir`
    _save_value(savedir / "ious.json", aucs.tolist(), args.debug)


# %%
# Exit

print("exiting")
sys.exit(0)
