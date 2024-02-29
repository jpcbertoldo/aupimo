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

# from anomalib.metrics import AUPR, AUPRO, AUROC
from PIL import Image
from torch import Tensor

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


from aupimo import aupimo_scores, per_image_iou_curves
from aupimo._validate_tensor import safe_tensor_to_numpy
from aupimo.oracles import IOUCurvesResult, max_avg_iou, max_iou_per_image
from aupimo.oracles_numpy import (
    calculate_levelset_mean_dist_to_superpixel_boundaries_curve,
    get_superpixels_watershed,
)
from aupimo.pimo_numpy import compute_min_thresh_at_max_fpr_normal_images

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
    # oracle superpixels
    (METRIC_SUPERPIXEL_ORACLE := "superpixel_oracle"),
    # thresh selection with superpixels boundaries distance heuristic
    (METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC := "superpixel_bound_dist_heuristic"),
    (METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC_PARALLEL := "superpixel_bound_dist_heuristic_parallel"),
]

ANY_METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC = {
    METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC,
    METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC_PARALLEL,
}

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
                # "--asmaps ../data/experiments/benchmark/efficientad_wr101_m_ext/mvtec/metal_nut/asmaps.pt",
                "--asmaps ../data/experiments/benchmark/rd++_wr50_ext/mvtec/bottle/asmaps.pt",
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
                # "--metrics superpixel_oracle",
                # "--metrics superpixel_bound_dist_heuristic",
                # "--metrics superpixel_bound_dist_heuristic_parallel",
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

# %%
# validation min thresh at max fpr normal images

METRICS_THAT_USE_MIN_THRESH = {
    METRIC_MAX_AVG_IOU_MIN_THRESH,
    METRIC_MAX_IOU_PER_IMG_MIN_THRESH,
} | ANY_METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC

if len(set(args.metrics) & METRICS_THAT_USE_MIN_THRESH) > 0:
    min_thresh = compute_min_thresh_at_max_fpr_normal_images(
        safe_tensor_to_numpy(asmaps),
        safe_tensor_to_numpy(masks),
        fpr_metric=(fpr_metric := "mean-per-image-fpr"),
        max_fpr=(max_fpr_normal_images := 1e-2),
    )


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


if METRIC_MAX_AVG_IOU_MIN_THRESH in args.metrics:
    ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_global_threshs.pt")
    if args.debug:
        ioucurves.per_image_ious = ioucurves.per_image_ious[some_imgs, :]

    max_avg_iou_result = max_avg_iou(
        ioucurves.threshs,
        ioucurves.per_image_ious,
        ioucurves.image_classes,
        paths=images_relpaths,
        min_thresh=min_thresh,
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
        min_thresh=min_thresh,
        paths=images_relpaths,
    )
    ious_maxs_result.save(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

# %%
# best achievable iou with superpixels

if METRIC_SUPERPIXEL_ORACLE in args.metrics:
    import numpy as np
    from progressbar import progressbar

    from aupimo._validate_tensor import safe_tensor_to_numpy
    from aupimo.oracles_numpy import (
        find_best_superpixels,
        get_superpixels_watershed,
        open_image,
    )

    results = []

    for image_idx in progressbar(range(len(images_relpaths))):
        mask = masks[image_idx]

        if mask.sum() == 0:
            results.append(None)
            continue

        img = open_image(images_abspaths[image_idx])

        superpixels = get_superpixels_watershed(
            img,
            superpixel_relsize=(watershed_superpixel_relsize := 3e-4),
            compactness=(watershed_compactness := 1e-4),
        )
        history, selected_suppixs, available_suppixs = find_best_superpixels(
            superpixels.astype(int),
            safe_tensor_to_numpy(mask).astype(bool),
        )
        superpixel_best_iou = history[-1]["iou"]
        results.append(
            {
                "path": images_relpaths[image_idx],
                "iou": float(superpixel_best_iou),
                "superpixels_selection": sorted(map(int, selected_suppixs)),
            },
        )

    (superpixel_oracle_selection_dir := rundir / "superpixel_oracle_selection").mkdir(exist_ok=True)

    payload = {
        "superpixels_method": "watershed",
        "superpixels_params": {
            "superpixel_relsize": watershed_superpixel_relsize,
            "compactness": watershed_compactness,
        },
        "results": results,
    }
    with (superpixel_oracle_selection_dir / "optimal_iou.json").open("w") as f:
        json.dump(payload, f, indent=4)


# %%
# asmap-superpixels contour distance heuristic


def calculate_levelset_mean_dist_curve(
    images_absolute_paths: list[str | Path],
    anomaly_maps: np.ndarray,
    upscale_factor: int | float,
    min_thresh: float,
    watershed_superpixel_relsize: float = 3e-4,
    watershed_compactness: float = 1e-4,
    num_levelsets: int = 500,
):
    from aupimo.oracles_numpy import open_image, upscale_image_asmap_mask

    threshs_per_image = []
    levelset_mean_dist_curve_per_image = []

    for image_idx, asmap_original_size in enumerate(anomaly_maps):
        image_original_size = open_image(images_absolute_paths[image_idx])
        image, asmap, _ = upscale_image_asmap_mask(
            image_original_size,
            asmap_original_size,
            None,
            upscale_factor=upscale_factor,
        )
        _, __, threshs, levelset_mean_dist_curve = calculate_levelset_mean_dist_to_superpixel_boundaries_curve(
            image,
            asmap,
            min_thresh,
            watershed_superpixel_relsize,
            watershed_compactness,
            num_levelsets=num_levelsets,
        )
        threshs_per_image.append(threshs)
        levelset_mean_dist_curve_per_image.append(levelset_mean_dist_curve)

    threshs_per_image = np.array(threshs_per_image)
    levelset_mean_dist_curve_per_image = np.array(levelset_mean_dist_curve_per_image)

    return threshs_per_image, levelset_mean_dist_curve_per_image


if METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC in args.metrics:
    threshs_per_image, levelset_mean_dist_curve_per_image = calculate_levelset_mean_dist_curve(
        images_abspaths[:2],
        safe_tensor_to_numpy(asmaps)[:2],
        upscale_factor=(upscale_factor := 2),
        min_thresh=min_thresh,
        watershed_superpixel_relsize=(watershed_superpixel_relsize := 3e-4),
        watershed_compactness=(watershed_compactness := 1e-4),
    )

# %%
# asmap-superpixels contour distance heuristic WITH MULTIPROCESSING


def _worker_init(
    images_absolute_paths,
    mp_anomaly_maps,
    mp_threshs_per_image,
    mp_levelset_mean_dist_curve_per_image,
    shape_anomaly_maps: tuple[int, int, int],
    num_levelsets: int,
):
    from copy import deepcopy

    import numpy as np

    global shared_images_absolute_paths, shared_anomaly_maps, shared_threshs_per_image, shared_levelset_mean_dist_curve_per_image  # noqa: PLW0603

    shared_images_absolute_paths = deepcopy(images_absolute_paths)
    shared_anomaly_maps = np.frombuffer(mp_anomaly_maps, dtype=np.float32).reshape(shape_anomaly_maps)

    num_images = shape_anomaly_maps[0]

    shared_threshs_per_image = np.frombuffer(mp_threshs_per_image, dtype=np.float32).reshape(
        (num_images, num_levelsets),
    )
    shared_levelset_mean_dist_curve_per_image = np.frombuffer(
        mp_levelset_mean_dist_curve_per_image,
        dtype=np.float32,
    ).reshape((num_images, num_levelsets))


def _worker_do(
    image_idx: int,
    upscale_factor: int | float,
    min_thresh: float,
    watershed_superpixel_relsize: float,
    watershed_compactness: float,
    num_levelsets: int,
):
    from aupimo.oracles_numpy import (
        calculate_levelset_mean_dist_to_superpixel_boundaries_curve,
        open_image,
        upscale_image_asmap_mask,
    )

    global shared_images_absolute_paths, shared_anomaly_maps, shared_threshs_per_image, shared_levelset_mean_dist_curve_per_image  # noqa: PLW0602

    image_original_size = open_image(shared_images_absolute_paths[image_idx])
    anomaly_map_original_size = shared_anomaly_maps[image_idx]

    image, anomaly_map, _ = upscale_image_asmap_mask(
        image_original_size,
        anomaly_map_original_size,
        None,
        upscale_factor=upscale_factor,
    )
    _, __, threshs, levelset_mean_dist_curve = calculate_levelset_mean_dist_to_superpixel_boundaries_curve(
        image,
        anomaly_map,
        min_thresh=min_thresh,
        watershed_superpixel_relsize=watershed_superpixel_relsize,
        watershed_compactness=watershed_compactness,
        num_levelsets=num_levelsets,
    )

    shared_threshs_per_image[image_idx, :] = threshs[:]
    shared_levelset_mean_dist_curve_per_image[image_idx, :] = levelset_mean_dist_curve.astype(np.float32)[:]


def calculate_levelset_mean_dist_curve_multiprocessing(
    images_absolute_paths: list[str | Path],
    anomaly_maps: np.ndarray,
    upscale_factor: int | float,
    min_thresh: float,
    watershed_superpixel_relsize: float = 3e-4,
    watershed_compactness: float = 1e-4,
    num_levelsets: int = 500,
    num_procs: int | None = None,
):
    import multiprocessing as mp
    import multiprocessing.sharedctypes

    if num_procs is None:
        num_procs = mp.cpu_count() - 1

    mp_anomaly_maps = mp.sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(anomaly_maps.dtype), anomaly_maps.size)
    shared_anomaly_maps = np.frombuffer(mp_anomaly_maps, dtype=anomaly_maps.dtype).reshape(anomaly_maps.shape)
    shared_anomaly_maps[:, :, :] = anomaly_maps[:, :, :]

    num_images = anomaly_maps.shape[0]

    mp_threshs_per_image = mp.sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(np.float32), num_images * num_levelsets)
    shared_threshs_per_image = np.frombuffer(mp_threshs_per_image, dtype=np.float32).reshape(
        (num_images, num_levelsets),
    )

    mp_levelset_mean_dist_curve_per_image = mp.sharedctypes.RawArray(
        np.ctypeslib.as_ctypes_type(np.float32),
        num_images * num_levelsets,
    )
    shared_levelset_mean_dist_curve_per_image = np.frombuffer(
        mp_levelset_mean_dist_curve_per_image,
        dtype=np.float32,
    ).reshape((num_images, num_levelsets))

    _worker_do_partial = partial(
        _worker_do,
        upscale_factor=upscale_factor,
        min_thresh=min_thresh,
        watershed_superpixel_relsize=watershed_superpixel_relsize,
        watershed_compactness=watershed_compactness,
        num_levelsets=num_levelsets,
    )
    _worker_args = [(image_index,) for image_index in range(len(images_absolute_paths))]

    with mp.Pool(
        processes=num_procs,
        initializer=_worker_init,
        initargs=(
            images_absolute_paths,
            mp_anomaly_maps,
            mp_threshs_per_image,
            mp_levelset_mean_dist_curve_per_image,
            anomaly_maps.shape,
            num_levelsets,
        ),
    ) as pool:
        pool.starmap(_worker_do_partial, _worker_args)

    return shared_threshs_per_image, shared_levelset_mean_dist_curve_per_image


if METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC_PARALLEL in args.metrics:
    threshs_per_image, levelset_mean_dist_curve_per_image = calculate_levelset_mean_dist_curve_multiprocessing(
        images_abspaths,
        safe_tensor_to_numpy(asmaps).astype(np.float32),
        upscale_factor=(upscale_factor := 2.0),
        min_thresh=min_thresh,
        watershed_superpixel_relsize=(watershed_superpixel_relsize := 3e-4),
        watershed_compactness=(watershed_compactness := 1e-4),
        num_levelsets=500,
        num_procs=None,  # None means all cpus - 1
    )


# %%
# save asmap-superpixels contour distance heuristic

(superpixel_bound_dist_heuristic_dir := rundir / "superpixel_bound_dist_heuristic").mkdir(exist_ok=True)

if len(set(args.metrics) & ANY_METRIC_SUPERPIXEL_BOUND_DIST_HEURISTIC) > 0:
    # order = 2% of num_thresh is a heuristic i found manually (based on num_levelsets=500)
    # keep it as list of lists instead of ndarray because they may have different lengths
    num_levelsets = levelset_mean_dist_curve_per_image.shape[1]
    local_minima_idxs_per_image = [
        sp.signal.argrelmin(levelset_mean_dist_curve, order=int(2e-2 * num_levelsets))[0].astype(int).tolist()
        for levelset_mean_dist_curve in levelset_mean_dist_curve_per_image
    ]

    payload = {
        "min_thresh": float(min_thresh),
        "upscale_factor": float(upscale_factor),
        "superpixels_method": "watershed",
        "superpixels_params": {
            "superpixel_relsize": float(watershed_superpixel_relsize),
            "compactness": float(watershed_compactness),
        },
        "paths": images_relpaths,
        "threshs_per_image": torch.from_numpy(threshs_per_image),
        "levelset_mean_dist_curve_per_image": torch.from_numpy(levelset_mean_dist_curve_per_image),
        "local_minima_idxs_per_image": local_minima_idxs_per_image,
    }

    torch.save(payload, superpixel_bound_dist_heuristic_dir / "superpixel_bound_dist_heuristic.pt")

# %%
from anomalib.models import SuperpixelCore

module = SuperpixelCore(
    input_size=tuple(asmaps.shape[-2:]),
    # layers=["layer1"],
    # backbone="resnet18",
    layers=["layer2", "layer3"],
    backbone="wide_resnet50_2",
    superpixel_relsize=3e-4,
)

# %%
from functools import partial

import numpy as np
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image

task = TaskType.SEGMENTATION
datamodule = MVTec(
    root=args.mvtec_root,
    category="bottle",
    image_size=tuple(asmaps.shape[-2:]),
    train_batch_size=2,
    eval_batch_size=2,
    num_workers=8,
    task=task,
)
datamodule.setup()
i, data = next(enumerate(datamodule.test_dataloader()))
print(f'Image Shape: {data["image"].shape}\nMask Shape: {data["mask"].shape}\nImage Original Shape: {data["image_original"].shape}')

# %%

output = module.model(data["image"], data["image_original"])
output["embeddings"].shape
output["superpixels"].shape

# %%
# set logging to info level
import logging

logging.basicConfig(level=logging.INFO)

# %%

callbacks = [
    ModelCheckpoint(
        mode="max",
        monitor="pixel_AUROC",
    ),
    EarlyStopping(
        monitor="pixel_AUROC",
        mode="max",
        patience=3,
    ),
]

engine = Engine(
    callbacks=callbacks,
    pixel_metrics="AUROC",
    accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    devices=1,
    logger=False,
    #
    max_epochs=1,
)

engine.fit(datamodule=datamodule, model=module)

# %%
engine.test(datamodule=datamodule, model=module)

# %%
predictions = engine.predict(model=module, dataloaders=datamodule.test_dataloader())
tmp = {}
tmp["image"] = torch.concat([batch["image"] for batch in predictions], dim=0)
tmp["anomaly_maps"] = torch.concat([batch["anomaly_maps"] for batch in predictions], dim=0)
tmp["pred_masks"] = torch.concat([batch["pred_masks"] for batch in predictions], dim=0)
tmp["mask"] = torch.concat([batch["mask"] for batch in predictions], dim=0)
tmp["image_original"] = torch.concat([batch["image_original"] for batch in predictions], dim=0)
predictions = tmp
# %%
print(
    f'Image Shape: {predictions["image"].shape},\n'
    f'Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
    f'Predicted Mask Shape: {predictions["pred_masks"].shape}',
)

# %%
for image_idx in range(0, len(predictions["image"]), 10):
    image = predictions["image_original"][image_idx].cpu().numpy()
    anomaly_map = predictions["anomaly_maps"][image_idx].cpu().numpy().squeeze()
    gt_mask = predictions["mask"][image_idx].cpu().numpy().squeeze()
    fig, axrow = plt.subplots(1, 2, figsize=(10, 5))
    _ = axrow[0].imshow(image)
    _ = axrow[1].imshow(anomaly_map, cmap="jet")
    for ax in axrow:
        _ = ax.axis("off")
        _ = ax.contour(gt_mask, [0.5], colors="k", alpha=1, lw=5, ls="--")
    fig.savefig(f"{image_idx}.pdf")

# %%
