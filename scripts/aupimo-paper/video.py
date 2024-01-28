# %%
# Setup (pre args)

from __future__ import annotations

import sys
from functools import partial
import warnings
from pathlib import Path

import scipy as sp
import numpy as np
import torch
import skimage
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl

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

from aupimo import AUPIMOResult

# %%
def load_data(asmaps_fpath: Path):
    mvtec_root = Path.home() / "data/datasets/MVTec"
    visa_root = Path.home() / "data/datasets/VisA"
    ucsdped_root = Path.home() / "data/datasets/UCSDped"

    assert asmaps_fpath.exists(), str(asmaps_fpath)

    asmaps_dict = torch.load(asmaps_fpath)
    assert isinstance(asmaps_dict, dict), f"{type(asmaps_dict)=}"

    images_relpaths = asmaps_dict["paths"]
    assert isinstance(images_relpaths, list), f"{type(images_relpaths)=}"

    # collection [of datasets] = {mvtec, visa}
    collection = {p.split("/")[0] for p in images_relpaths}
    assert collection.issubset({"MVTec", "VisA", "UCSDped"}), f"{collection=}"
    collection = collection.pop()

    if collection == "MVTec":
        collection_root = mvtec_root

    elif collection == "VisA":
        collection_root = visa_root
    
    elif collection == "UCSDped":
        collection_root = ucsdped_root
        
    else:
        msg = f"Unknown collection: {collection=}"
        raise NotImplementedError(msg)

    assert collection_root.exists(), f"{collection=} {collection_root=!s}"

    print("getting masks paths from images paths")
    def _image_path_2_mask_path(image_path: str) -> str | None:
        if "good" in image_path:
            # there is no mask for the normal images
            return None

        image_path = Path(image_path.replace("test", "ground_truth"))

        if (collection := image_path.parts[0]) == "VisA":
            image_path = image_path.with_suffix(".png")

        elif collection == "MVTec":
            image_path = image_path.with_stem(image_path.stem + "_mask").with_suffix(".png")
            
        elif collection == "UCSDped":
            video_dir = image_path.parent
            gt_dir = video_dir.with_name(video_dir.name + "_gt")
            file_name = image_path.with_suffix(".bmp").name
            if video_dir.name in ("Test006", "Test009", "Test012"):
                file_name = f"frame{file_name}"
            image_path = gt_dir / file_name
        else:
            msg = f"Unknown collection: {collection=}"
            raise NotImplementedError(msg)

        return str(image_path)

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

    print("loading masks")
    masks_pils = [Image.open(p).convert("L") if p is not None else None for p in masks_abspaths]

    masks_resolution = {p.size for p in masks_pils if p is not None}
    assert len(masks_resolution) == 1, f"assumed single-resolution dataset but found {masks_resolution=}"
    masks_resolution = masks_resolution.pop()
    masks_resolution = (masks_resolution[1], masks_resolution[0])  # [W, H] --> [H, W]
    print(f"{masks_resolution=} (HEIGHT, WIDTH)")

    print("loading masks")
    masks = np.stack(
        [
            np.asarray(pil).astype(bool)
            if pil is not None
            else np.zeros(masks_resolution, dtype=bool)
            for pil in masks_pils
        ],
        axis=0,
    )
    print(f"{masks.shape=}")

    print("loading images")
    images_pils = [Image.open(p).convert("RGB") for p in images_abspaths]
    images = np.stack([np.asarray(pil) for pil in images_pils], axis=0)
    print(f"{images.shape=}")

    print("loading asmaps.pt")
    asmaps = asmaps_dict["asmaps"]
    assert isinstance(asmaps, torch.Tensor), f"{type(asmaps)=}"
    assert asmaps.ndim == 3, f"{asmaps.shape=}"
    assert len(asmaps) == len(images_relpaths), f"{len(asmaps)=}, {len(images_relpaths)=}"
    print(f"{asmaps.shape=}")
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
    asmaps = asmaps.cpu().numpy()
    
    # make sure all values in the asmaps are positive
    # `as` stands for `anomaly score`
    as_min = asmaps.min()
    if as_min < 0:
        print(f"shifting asmaps by {as_min=}")
        asmaps += abs(as_min)
        
    return images, masks, asmaps, images_abspaths, masks_abspaths

images, masks, asmaps, images_abspaths, masks_abspaths = load_data(
    Path.home() / "repos/aupimo/data/experiments/video/patchcore_wr50_every_2_frames/ucsdped/ucsdped2/asmaps.pt"
)

# %%
def load_aupimos(aupimos_fpath: Path):
    assert aupimos_fpath.exists(), str(aupimos_fpath)
    return AUPIMOResult.load(aupimos_fpath)

aupimos = load_aupimos(
    Path.home() / "repos/aupimo/data/experiments/video/patchcore_wr50_every_2_frames/ucsdped/ucsdped2/aupimo/aupimos.json"
)
aupimos.aupimos.shape
    
# %%
frames = [
    {
        "path": path,
        "image": path.name,
        "video": path.parent.name,      
        "aupimo": aupimo,
    }
    for path, aupimo in zip([
        Path(p)
        for p in aupimos.paths
    ], aupimos.aupimos.numpy().tolist())
]
import pandas as pd
frames = pd.DataFrame(frames).sort_values(["video", "image"]).set_index(["video"])
frames["frame_idx"] = frames["image"].map(lambda x: int(x.split(".")[0]))
frames["has_anomaly"] = frames["aupimo"].map(lambda x: int(not pd.isna(x)))
videos_keys = frames.index.unique().tolist()
max_frame_idx = frames["frame_idx"].max()

# %%
from skimage.morphology import label

# vk = videos_keys[0]
# ax = axes[0, 0]

fig, axes = plt.subplots(
    3, 4, figsize=(4 * 2.2, 3 * 2.2), dpi=150, 
    sharex=True, sharey=True, layout="constrained",
)

ax = axes[0, 0]
_ = ax.set_ylim(-.01, 1.01)
_ = ax.set_yticks(np.linspace(0, 1, 5))
# format the y-axis in %
_ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))

_ = ax.set_xlim(0, max_frame_idx)
_ = ax.set_xticks(np.linspace(0, max_frame_idx + 1, 4).astype(int))

for idx, vk in enumerate(videos_keys):
    ax = axes[idx // 4, idx % 4]
    video_frames = frames.loc[vk].reset_index(drop=True).set_index("image")
    
    _ = ax.plot(
        video_frames["frame_idx"], video_frames["aupimo"],
        color="black", linewidth=1, label="AUPIMO",
    )

    num_frames = len(video_frames)
    with_anomaly_segments = label(video_frames["has_anomaly"].values.astype(int))
    without_anomaly_segments = label(~video_frames["has_anomaly"].astype(bool))

    for segm_label in pd.unique(without_anomaly_segments):
        if segm_label == 0:
            continue
        frame_selection_mask = without_anomaly_segments == segm_label
        
        min_idx = np.where(frame_selection_mask)[0].min()
        max_idx = np.where(frame_selection_mask)[0].max()
        span_min_frame_idx = video_frames["frame_idx"].values[min_idx]
        span_max_frame_idx = video_frames["frame_idx"].values[min(max_idx + 1, num_frames - 1)]
        _ = ax.axvspan(
            span_min_frame_idx, span_max_frame_idx,
            color="tab:blue", alpha=0.4,
            label="Normal",
        )

    for segm_label in pd.unique(with_anomaly_segments):
        if segm_label == 0:
            continue
        frame_selection_mask = with_anomaly_segments == segm_label
        min_idx = np.where(frame_selection_mask)[0].min()
        max_idx = np.where(frame_selection_mask)[0].max()
        span_min_frame_idx = video_frames["frame_idx"].values[min_idx]
        span_max_frame_idx = video_frames["frame_idx"].values[min(max_idx + 1, num_frames - 1)]
        _ = ax.axvspan(
            span_min_frame_idx, span_max_frame_idx,
            label="Has anomaly",
            color="tab:red", alpha=0.4,
        )
        
    video_max_frame_idx = video_frames["frame_idx"].max()
    if video_max_frame_idx < max_frame_idx:
        _ = ax.axvspan(
            video_max_frame_idx, max_frame_idx,
            label="No frame",
            color="grey", alpha=0.5,
        )
        
    _ = ax.set_title(vk)

for ax in axes[-1, :]:
    _ = ax.set_xlabel("Frame Index")

for ax in axes[:, 0]:
    _ = ax.set_ylabel("AUPIMO")

for ax in axes.flatten():
    _ = ax.yaxis.grid(True, linestyle="--", linewidth=1, alpha=0.3, zorder=10, color="black")
    
# get legend handles and labels from axes[1, 0]
# then plot the legend in axes[0, 0]
handles, labels = axes[1, 0].get_legend_handles_labels()
_ = axes[0, 0].legend(handles, labels, loc="upper left",)

(SAVEDIR := Path("/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src/img/video")).mkdir(exist_ok=True, parents=True)
fig.savefig(SAVEDIR / "aupimo-vs-time.pdf", bbox_inches="tight", pad_inches=0.01)

# %%
s.plot?
# %%
from anomalib.data import UCSDped
from anomalib.data.utils.transforms import InputNormalizationMethod
# %%
datamodule = UCSDped(
    root=Path.home() / "data/datasets/UCSDped",
    category="UCSDped2",
    normalization=InputNormalizationMethod.NONE,
)
datamodule.prepare_data()
datamodule.setup()
# %%
RCPARAMS = {
    "font.family": "sans-serif", "text.usetex": True,
    "axes.titlesize": "xx-large", "axes.labelsize": "x-large",
    "ytick.labelsize": "x-large", "xtick.labelsize": "x-large",
    "legend.fontsize": "x-large", "legend.title_fontsize": "x-large",
    'text.latex.preamble': r'\usepackage{lmodern}',
}
plt.rcParams.update(RCPARAMS)

