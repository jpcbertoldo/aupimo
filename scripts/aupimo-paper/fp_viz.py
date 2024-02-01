#!/usr/bin/env python
# coding: utf-8


# %%
# Pre-Setup (aux functions)

from __future__ import annotations

def set_ipython_autoreload_2():
    from IPython import get_ipython
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    
def running_mode() -> str:
    import pathlib
    import sys
    arg0 = pathlib.Path(sys.argv[0]).stem
    if arg0 == "ipykernel_launcher":
        print("running as a notebook")
        return "notebook"
    file_name = pathlib.Path(__file__).stem
    if arg0 == file_name:
        print("running as a script")
        return "script"
    else:
        print("running as an import")
        return "import"

def is_script():
    return running_mode() == "script"

def is_notebook():
    return running_mode() == "notebook"

def is_import():
    return running_mode() == "import"

import numpy as np
import torch


# %%
# DATA

from common import OUTPUTDIR, DATASETSDIR, adapt_imgpath_to_my_convention  # noqa: E402
print(f"{OUTPUTDIR=}")
assert OUTPUTDIR.exists()

records = [
    {
        "dir": str(category_dir),
        "model": model_dir.name,
        "dataset": dataset_dir.name,
        "category": category_dir.name,
    }
    for model_dir in OUTPUTDIR.iterdir()
    if model_dir.is_dir() and model_dir.name != "debug"
    for dataset_dir in model_dir.iterdir()
    if dataset_dir.is_dir()
    for category_dir in dataset_dir.iterdir()
    if category_dir.is_dir()
]
print(f"{len(records)=}")

import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402
from progressbar import progressbar  # noqa: E402
import json 

for record in progressbar(records):
    
    d = Path(record["dir"])
    
    assert (pimodir := d / "pimo").exists()
    
    curves_path = list(pimodir.glob("curves*.pt"))
    assert len(curves_path) == 1
    curves_path = curves_path[0]
    
    curves = torch.load(curves_path)
    
    record.update({
        k: v if not isinstance(v, torch.Tensor) else v.numpy()
        for k, v in curves.items()
    })
    
    if record["model"].startswith("efficientad"):
        
        # * is for the wandbe run id
        images_paths = d.glob(f"efficientad_output/**/images_paths.txt")
        images_paths = list(images_paths)
        assert len(images_paths) == 1

        # path to the files
        images_paths = images_paths[0]
        asmaps_path = Path(images_paths).with_name("asmaps.pt")
        assert asmaps_path.exists()
        
        # now it becomes a list of (relative) paths (from the file content)
        images_paths = images_paths.read_text().splitlines()
        
        record.update({
            "imgpaths": np.asarray(images_paths),
            "asmaps_path": str(asmaps_path),
        })
        
    elif record["model"] in (
        "simplenet_wr50_ext", 
        "pyramidflow_fnf_ext", "pyramidflow_r18_ext", 
        "rd++_wr50_ext",
        "uflow_ext",
    ):
        asmaps_path = d / "asmaps.pt"
        assert asmaps_path.exists()
        images_paths = (d / "key.txt").read_text().splitlines()
        images_paths = [
            adapt_imgpath_to_my_convention(record["dataset"], record["category"], p) 
            for p in images_paths
        ]
        record.update({
            "imgpaths": images_paths,
            "asmaps_path": str(asmaps_path),
        })
    else:
        predictions_df = pd.read_csv(d / "predictions.csv")
        asmaps_path = d / "asmaps.pt"
        assert asmaps_path.exists()
        record.update({
            "imgpaths": predictions_df["imgpath"].values,
            "asmaps_path": str(asmaps_path),
        })
    
df = pd.DataFrame.from_records(records)
df = df.replace({"dataset": {"mvtec_ad": "mvtec"}})
df = df.set_index(["dataset", "category", "model"])
df.head()

# %%
print("has any model with any NaN?")
df.isna().any(axis=1).any()

# %%

import numpy as np

TARGET_FPR_LEVELS = [1e-5, 1e-4, 1e-3, 1e-2,]

recs = []

for rowidx, row in df.iterrows():
    
    fprs = row["fprs"]
    imgclass = row["image_classes"]
    
    perlevel_choices = {}

    for target_fpr_level in TARGET_FPR_LEVELS:
        closest_thresh_idx = np.argmin(np.abs(fprs - target_fpr_level), axis=1)
        closest_actual_fpr = fprs[np.arange(len(fprs)), closest_thresh_idx]
        isclose = np.isclose(closest_actual_fpr, target_fpr_level, atol=1e-5)
        isnormal = imgclass == 0
        imgidxs = np.where(isclose & isnormal)[0] 
        perimg_thresh_idxs = closest_thresh_idx[imgidxs]
        assert imgidxs.shape == perimg_thresh_idxs.shape
        perlevel_choices[target_fpr_level] = {
            "imgidxs": imgidxs,
            "perimg_thresh_idxs": perimg_thresh_idxs,
        }

    imgidxs_in_all_levels = np.asarray(sorted(
        set.intersection(*[set(v["imgidxs"]) for v in perlevel_choices.values()])
    ))

    recs.extend([
        {
            **dict(zip(df.index.names, rowidx)),   
            "target": k,
            "imgidx": imgidx,
            "threshidx": thresh_idx,
            "thresh": row["thresholds"][thresh_idx],
            "imgpath": row["imgpaths"][imgidx],
            "asmaps_path": row["asmaps_path"],
        }
        for k, v in perlevel_choices.items()
        for imgidx, thresh_idx in zip(v["imgidxs"], v["perimg_thresh_idxs"])
        if imgidx in imgidxs_in_all_levels
    ])

df_fps = pd.DataFrame.from_records(recs)
# make it for this pc
df_fps["imgpath"] = df_fps["imgpath"].apply(lambda x: DATASETSDIR / x.split("/datasets/")[-1])  
df_fps

# %%

df_toplot = df_fps.copy()

df_toplot.sort_values(["dataset", "category", "imgidx", "target"], inplace=True)
df_toplot.head(5)

# get rid of per-image duplicates
# it can be from multiple models at a given fpr level, but we only want one
df_toplot = pd.concat([
    df_img[df_img.model == df_img.model.iloc[0]]
    for (dataset, category, imgidx), df_img in df_toplot.groupby(["dataset", "category", "imgidx"])
]).reset_index(drop=True)

df_toplot.sort_values(["dataset", "category", "imgidx", "target"], inplace=True)
df_toplot.head(5)

# ================================================================================
# ================================================================================

# %%

from matplotlib import pyplot as plt
import matplotlib as mpl
import skimage as skim
from PIL import Image


def get_binary_cmap(colorpositive="red"):
    return mpl.colors.ListedColormap(['#00000000', colorpositive])

def imshow_with_masks(ax, img, masks, kwargs):
    from copy import deepcopy
    kwargs = deepcopy(kwargs)  # there is a pop in there
    assert masks.ndim == 3
    num_masks = masks.shape[0]
    assert num_masks == len(kwargs)
    assert img.shape[:-1] == masks.shape[1:]
    ax.imshow(img)
    for mask, kws in zip(masks, kwargs):
        color = kws.pop("color")
        ax.imshow(mask, cmap=get_binary_cmap(color), **kws)    
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")

def imshow_with_3masks(ax, img, masks):
    imshow_with_masks(ax, img, masks, [
        dict(color="tab:pink", alpha=1),
        dict(color="black", alpha=1),
        dict(color="white", alpha=1),
    ])

def imshow_with_4masks(ax, img, masks):
    imshow_with_masks(ax, img, masks, [
        dict(color="tab:pink", alpha=1),
        dict(color="blue", alpha=1),
        dict(color="white", alpha=1),
        dict(color="blue", alpha=1),
    ])
    
def origin_size_2_bbox(origin, size):
    """ from origin and size to bbox 
    Args:
        origin (np.ndarray): (row, col) in [0, 1]
        size (float): in [0, 1]
    Returns:
        np.ndarray: (row, col, row, col) in [0, 1]
    """
    origin = np.asarray(origin)
    diagonal = origin + size
    origin = origin.clip(0, 1)
    diagonal = diagonal.clip(0, 1)
    return np.concatenate([origin, diagonal])

def bbox_2_origin_size(bbox):
    """ from bbox to origin and size
    Args:
        bbox (np.ndarray): (row, col, row, col) in [0, 1]
    Returns:
        np.ndarray: (row, col) in [0, 1]
        float: in [0, 1]
    """
    bbox = np.asarray(bbox)
    origin = bbox[:2]
    diagonal = bbox[2:]
    size = diagonal - origin
    return origin, size

def rel2abs(imgshape, bbox):
    """ from relative to absolute """
    resolution_x = imgshape[1]
    resolution_y = imgshape[0]
    return (np.array([resolution_y, resolution_x, resolution_y, resolution_x]) * bbox).astype(int)

def zoom(ax, origin, size, bbox=None):
    """zoom in the ax
    Args:
        origin (np.ndarray): (row, col) in [0, 1]
        size (float): in [0, 1]
    """
    if bbox is None:
        imgshape = ax.images[0].get_array().shape
        bbox = rel2abs(imgshape, origin_size_2_bbox(origin, size))
    ax.set_xlim(bbox[1], bbox[3])
    ax.set_ylim(bbox[0], bbox[2])
    ax.invert_yaxis()

def draw_zoom(ax, origin, size, bbox=None, color="w"):
    """draw a zoom in the ax
    Args:
        origin (np.ndarray): (row, col) in [0, 1]
        size (float): in [0, 1]
    """
    if bbox is None:
        imgshape = ax.images[0].get_array().shape
        bbox = rel2abs(imgshape, origin_size_2_bbox(origin, size))
    ax.add_patch(mpl.patches.Rectangle(
        (bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
        fill=False, edgecolor=color, lw=2, ls="--",
    ))

# ================================================================================

for (dataset, category, imgidx), df_img in df_toplot.groupby(["dataset", "category", "imgidx"]):
    break

imgpath = df_img["imgpath"].iloc[0]
imgpil = Image.open(imgpath, mode="r").convert("RGB").resize((256, 256), Image.BILINEAR)
img = np.asarray(imgpil)
asmap = torch.load(df_img["asmaps_path"].iloc[0])[imgidx].numpy()

threshs = df_img.set_index("target").loc[TARGET_FPR_LEVELS[::-1], "thresh"].values.astype(float)
masks = np.stack([(asmap >= thresh).astype(int) for thresh in threshs], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
for ax in axes:
    imshow_with_4masks(ax, img, masks)
    
size = 0.4
origin = np.asarray([0.25, 0.25])  # (row, col) in [0, 1]
draw_zoom(axes[0], origin, size)
zoom(axes[1], origin, size)
plt.show()

# %%

def get_mask_props(mask):
    import pandas as pd
    props = pd.DataFrame(
        skim.measure.regionprops_table(
            mask, properties=["area", "centroid",],
        )
    )
    if len(props) == 0:
        return pd.Series({"area": np.nan, "centroid": np.asarray([np.nan, np.nan])})
    props["centroid"] = props.apply(lambda row: np.asarray([row["centroid-0"], row["centroid-1"]]), axis=1)
    props.drop(columns=["centroid-0", "centroid-1"], inplace=True)
    props = props.iloc[0]
    # make the centroid relative
    props["centroid"] = props["centroid"] / np.asarray(mask.shape)
    return props

def get_zoom_candidates(mask, zoom_size, num_origin_jumps=10):
    import itertools
    possible_origins  = np.array(list(itertools.product(
        tmp := np.linspace(0, 1 - zoom_size, num_origin_jumps), tmp
    )))
    zoom_props = pd.DataFrame(
        [
            {
                "origin": origin,
                "bbox": (bbox := rel2abs(mask.shape, origin_size_2_bbox(origin, zoom_size))),
                **get_mask_props(mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]).to_dict(),
            }
            for origin in possible_origins
        ]
    ).dropna(axis=0)
    zoom_props["centroid_dist_center"] = zoom_props["centroid"].apply(lambda x: np.linalg.norm(x - 0.5))
    zoom_props.sort_values(["area", "centroid_dist_center"], ascending=[False, True], inplace=True)
    zoom_props = zoom_props.drop_duplicates("area", keep="first")
    return zoom_props


# ===============================================================================
# ================================================================================

# %%
# FPR LEVELS VIZ
# 
from common import DATADIR, DATASETS_LABELS, CATEGORIES_LABELS  # noqa: E402

for dcidx, ((dataset, category), df_dataset) in progressbar(
    enumerate(df_toplot.groupby(["dataset", "category"]))
):
    print(f"{dataset=} {category=}")
    (savedir := DATADIR / f"fp_levels/{dcidx:03}/").mkdir(exist_ok=True, parents=True)

    for plotidx, (imgidx, df_img) in enumerate(df_dataset.groupby("imgidx")):
        
        if plotidx >= 3: break
        
        # ================================================================================
        # args
        
        imgpath = df_img["imgpath"].iloc[0]
        asmaps_path = df_img["asmaps_path"].iloc[0]
        threshs = df_img.set_index("target").loc[TARGET_FPR_LEVELS[::-1], "thresh"].values.astype(float).tolist()
        zoom_size = 0.15
        
        # ================================================================================
        # load / compute
        
        assert Path(imgpath).exists()
        assert Path(asmaps_path).exists()
        
        imgpil = Image.open(imgpath, mode="r").convert("RGB")
        img = np.asarray(imgpil)

        # resize the asmap to the original resolution
        asmap = torch.nn.functional.interpolate(
            torch.load(asmaps_path)[imgidx].unsqueeze(0).unsqueeze(0),
            size=img.shape[:2],
            mode='bilinear',
        ).squeeze().numpy()
        masks = np.stack([(asmap >= thresh).astype(int) for thresh in threshs], axis=0)
        for idx in range(len(masks) - 1):
            masks[idx] = masks[idx] - masks[idx + 1]

        zoom_candidates = get_zoom_candidates(masks[0], zoom_size)
        best_zoom_origin = zoom_candidates.iloc[0].origin

        # ================================================================================
        # plot
        
        with mpl.rc_context({"font.family": "sans-serif"}):
            
            fig_full, ax_full = plt.subplots(figsize=(sz := 6, sz), dpi=100, layout="constrained")
            fig_zoom, ax_zoom = plt.subplots(figsize=(sz := 6, sz), dpi=100, layout="constrained")
            
            import matplotlib as mpl
            for ax in [ax_full, ax_zoom]:
                imshow_with_masks(ax, img, masks, [
                    dict(color=mpl.colormaps["Blues"](0.9), alpha=0.75),
                    dict(color=mpl.colormaps["Blues"](0.6), alpha=0.75),
                    dict(color="white", alpha=1),
                    dict(color="black", alpha=1),
                ])
                _ = ax.contour(masks[0],cmap=get_binary_cmap("navy"),)
            
            if category in ("bottle", "screw"):
                zoom_color = "black"
            else:
                zoom_color = "white"
                
            draw_zoom(ax_full, best_zoom_origin, zoom_size, color=zoom_color)
            zoom(ax_zoom, best_zoom_origin, zoom_size)
            
            # annotate the dataset and category
            _ = ax_full.annotate(
                f"{DATASETS_LABELS[dataset]} / {CATEGORIES_LABELS[category]} / Image {imgidx:03d}"
                if plotidx == 0 else
                f"Image {imgidx:03d}",
                xy=(0, 1), xycoords="axes fraction",
                xytext=(20, -20), textcoords="offset pixels",
                ha="left", va="top",
                bbox=dict(fc="w", alpha=1),
                fontsize=22,
            )
            _ = ax_zoom.annotate(
                "Zoom",
                xy=(0, 1), xycoords="axes fraction",
                xytext=(20, -20), textcoords="offset pixels",
                ha="left", va="top",
                bbox=dict(fc="w", alpha=1),
                fontsize=22,
            )
            
        fig_full
        fig_zoom
        _ = fig_full.savefig(savedir / f"fp_levels_{dcidx:03}_{plotidx:03d}_full.jpg", bbox_inches="tight")
        _ = fig_zoom.savefig(savedir / f"fp_levels_{dcidx:03}_{plotidx:03d}_zoom.jpg", bbox_inches="tight")
        # raise Exception("break")
        # break
    # break

