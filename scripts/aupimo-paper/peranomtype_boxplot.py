#!/usr/bin/env python

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from anomalib.data.utils import read_image
from anomalib.metrics.pimo import AUPIMOResult
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

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
    # InteractiveShell.ast_node_interactivity = "all"
    InteractiveShell.ast_node_interactivity = "last"
    # show all warnings
    warnings.filterwarnings("always", category=Warning)

    print("setting numpy print precision")
    np.set_printoptions(floatmode="maxprec", precision=3, suppress=True)
    print("setting pandas print precision")
    pd.set_option("display.precision", 3)

else:
    IS_NOTEBOOK = False


# %%
# Args

parser = argparse.ArgumentParser()
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
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

args.mvtec_root = args.mvtec_root.resolve()
assert args.mvtec_root.exists()

args.visa_root = args.visa_root.resolve()
assert args.visa_root.exists()

print(f"{args=}")


# %%
# Setup (post args)

BENCHMARK_DIR = Path(__file__).parent / "../../data/experiments/benchmark"
BENCHMARK_DIR = BENCHMARK_DIR.resolve()
print(f"{BENCHMARK_DIR=}")
assert BENCHMARK_DIR.exists()

SAVE_DIR = Path("/home/jcasagrandebertoldo/repos/2024-these/ch2/img/per_anomtype")
print(f"{SAVE_DIR=}")
assert SAVE_DIR.exists()


# %%
# Load data

records = [
    {
        "path": str(dataset_dir.resolve()),
        "model": model_dir.name,
        "collection": collection_dir.name,
        "dataset": dataset_dir.name,
    }
    for model_dir in BENCHMARK_DIR.iterdir()
    if model_dir.is_dir() and model_dir.name != "debug"
    for collection_dir in model_dir.iterdir()
    if collection_dir.is_dir()
    for dataset_dir in collection_dir.iterdir()
    if dataset_dir.is_dir()
]
print(f"{len(records)=}")


def load_aupimo_result_from_json_dict(payload: dict[str, str | float | int | list[str]]) -> AUPIMOResult:
    """Convert the JSON payload to an AUPIMOResult dataclass."""
    if not isinstance(payload, dict):
        msg = f"Invalid payload. Must be a dictionary. Got {type(payload)}."
        raise TypeError(msg)
    try:
        return AUPIMOResult(
            fpr_lower_bound=payload["fpr_lower_bound"],
            fpr_upper_bound=payload["fpr_upper_bound"],
            # `num_threshs` vs `num_thresholds` is an inconsistency with an older version of the JSON file
            num_thresholds=payload["num_threshs"] if "num_threshs" in payload else payload["num_thresholds"],
            thresh_lower_bound=payload["thresh_lower_bound"],
            thresh_upper_bound=payload["thresh_upper_bound"],
            aupimos=torch.tensor(payload["aupimos"], dtype=torch.float64),
        )

    except KeyError as ex:
        msg = f"Invalid payload. Missing key {ex}."
        raise ValueError(msg) from ex

    except (TypeError, ValueError) as ex:
        msg = f"Invalid payload. Cause: {ex}."
        raise ValueError(msg) from ex


for record in records:
    d = Path(record["path"])
    record['dir'] = str(d.relative_to(BENCHMARK_DIR))
    try:
        assert (aupimo_dir := d / "aupimo").exists()  # noqa: RUF018
        with (aupimo_dir / "aupimos.json").open("r") as f:
            payload = json.load(f)
            aupimoresult = load_aupimo_result_from_json_dict(payload)
        scores = aupimoresult.aupimos.numpy()
        anomtype = [Path(p).parent.name for p in payload['paths']]
        record["imgidx_imgpath_aupimo_anomtype"] = list(zip(np.arange(scores.size), payload['paths'], scores, anomtype, strict=True))
        record['thresh_lbound'] = aupimoresult.thresh_lower_bound
        record['thresh_ubound'] = aupimoresult.thresh_upper_bound
        record['asmaps_path'] = str(aupimo_dir.parent / 'asmaps.pt')

    except AssertionError:
        record["aupimo"] = None

records_data = pd.DataFrame.from_records(records)
records_data["model"] = records_data["model"].astype("category")
records_data["collection"] = records_data["collection"].astype("category")
records_data["dataset"] = records_data["dataset"].astype("category")
records_data["path"] = records_data["path"].astype("string")
records_data["dir"] = records_data["dir"].astype("string")
records_data["asmaps_path"] = records_data["asmaps_path"].astype("string")

# %%
aupimo_results  = records_data.copy()
aupimo_results = aupimo_results.sort_values(["model", "collection", "dataset"])
aupimo_results = aupimo_results.reset_index(drop=True)
aupimo_results = aupimo_results.drop(columns=["imgidx_imgpath_aupimo_anomtype"])

print(f"has any model with any NaN? {aupimo_results.isna().any(axis=1).any()}")

aupimo_results.model.unique()
aupimo_results.collection.unique()
dataset_per_collection = aupimo_results.groupby("collection", observed=True)['dataset'].unique().explode()
dataset_per_collection.shape  # noqa: B018
aupimo_results.dtypes  # noqa: B018


# %%
data = records_data.copy()
data = data.drop(columns=["path", "thresh_lbound", "thresh_ubound", "dir", "asmaps_path"])
data = data.explode("imgidx_imgpath_aupimo_anomtype")
data = pd.concat(
    [
        data.drop(columns=["imgidx_imgpath_aupimo_anomtype"]),
        data[["imgidx_imgpath_aupimo_anomtype"]]
        .apply(lambda row: row.imgidx_imgpath_aupimo_anomtype, axis=1, result_type="expand")
        .rename(columns={0: "image_index", 1: "image_path", 2: "aupimo", 3: "anomtype"}),
    ],
    axis=1,
)
data["anomtype"] = data["anomtype"].astype("category")
data["image_path"] = data["image_path"].astype("string")
data["image_path_resolved"] = data.apply(
    lambda row: str((args.mvtec_root if row.collection == "mvtec" else args.visa_root).parent / row.image_path),
    axis=1,
).astype("string")
data = data.sort_values(["model", "collection", "dataset", "anomtype"])
data = data.reset_index(drop=True)

data.dtypes  # noqa: B018

anomtype_per_dataset = data.groupby(["collection", "dataset"], observed=True)['anomtype'].unique().explode()
anomtype_per_dataset.shape  # noqa: B018
num_anomtype_per_dataset = anomtype_per_dataset.groupby(["collection", "dataset"], observed=True).count().reset_index()
num_anomtype_per_dataset  # noqa: B018

# %%
from matplotlib import pyplot as plt

num_models = aupimo_results.model.nunique()

fig, axes = plt.subplots(1, num_models, figsize=np.array((num_models, 3.5)) * 4.5, layout="constrained")

for gbidx, data_modelgb in enumerate(data.groupby("model")):
    model, data_model = data_modelgb
    print(f"{model=}")
    print(f"{data_model.shape=}")
    ax = axes[gbidx]
    groups = list(data_model.groupby(["dataset", "anomtype"]))
    groups_names, groups_data = list(zip(*groups, strict=False))
    _ = ax.boxplot(
        [data["aupimo"].values for data in groups_data],
        labels=[f"{dataset}/{anomtype}" for (dataset, anomtype) in groups_names],
        vert=False,
    )
    # repeat the ticks from the left on the right
    _ = ax.set_title(model)
fig

# %%
table = data.query("anomtype != 'good'").groupby(["model", "collection", "dataset", "anomtype"], observed=True)["aupimo"].mean().unstack("model")

table.drop(columns=["padim_r18", "pyramidflow_fnf_ext"], inplace=True)

table_colored = table.style.background_gradient(cmap="bwr").format("{:.0%}")
table_colored

# %%
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================

images_per_anomtype = data.set_index(["model", "collection", "dataset", "anomtype"])

# %%
anomtype_per_dataset.loc["mvtec", "transistor"]
# %%
import torch
aupimo_result = aupimo_results.set_index(["model", "collection", "dataset"]).loc["patchcore_wr50", "mvtec", "transistor"]
asmaps = torch.load(aupimo_result.asmaps_path)['asmaps'].numpy()
asmaps.shape
aupimo_result

# %%


# %%

image_selection = (
    images_per_anomtype
    .loc["patchcore_wr50", "mvtec", "transistor"]
    [["aupimo", "image_index", "image_path_resolved"]]
    .query("anomtype != 'good'")
    .groupby("anomtype", observed=True)
    .apply(lambda df: df.iloc[(idx := 5):idx + 1])
    .reset_index(level=1, drop=True).reset_index()
)
image_selection


def mvtec_mask_path_from_image_path(image_path: str) -> str:
    image_path = Path(image_path)
    test_dir = image_path.parent.parent
    dataset_dir = image_path.parent.parent.parent
    return str(dataset_dir / "ground_truth" / image_path.relative_to(test_dir).with_stem(image_path.stem + "_mask"))

for _, row in image_selection.iterrows():
    fig, ax = plt.subplots(figsize=(5, 5))
    image = Image.open(row.image_path_resolved).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(1.6)
    image = ImageEnhance.Contrast(image).enhance(1.05)
    image = ImageEnhance.Sharpness(image).enhance(2.)
    mask = read_image(mvtec_mask_path_from_image_path(row.image_path_resolved))[..., 0]
    _ = ax.imshow(image)
    _ = ax.contour(mask, levels=[0.5], colors="red", linewidths=2)
    _ = ax.axis("off")
    _ = ax.annotate(
        f"{row.anomtype.replace('_', ' ').title()}",
        xy=(0.03, 0.97),  xycoords='axes fraction',
        ha='left', va='top',
        fontsize='xx-large',
        color='k',
        bbox={"facecolor": 'white', "alpha": 0.9, "pad": 5},
    )
    # fig.savefig(f"/home/jcasagrandebertoldo/Downloads/{row.anomtype}.jpg", bbox_inches="tight", dpi=200, pad_inches=0.)

# %%

image_selection = (
    images_per_anomtype
    .loc["patchcore_wr50", "mvtec", "transistor"]
    [["aupimo", "image_index", "image_path_resolved"]]
    .query("anomtype == 'good'")
    .iloc[[1, 30]]
    .reset_index()
)
image_selection

for index, row in image_selection.iterrows():
    fig, ax = plt.subplots(figsize=(5, 5))
    image = Image.open(row.image_path_resolved).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(1.6)
    image = ImageEnhance.Contrast(image).enhance(1.05)
    image = ImageEnhance.Sharpness(image).enhance(2.)
    _ = ax.imshow(image)
    _ = ax.axis("off")
    # fig.savefig(f"/home/jcasagrandebertoldo/Downloads/normal{index}.jpg", bbox_inches="tight", dpi=200, pad_inches=0.)
    fig.savefig(f"/home/jcasagrandebertoldo/Downloads/normal{index}nolabel.jpg", bbox_inches="tight", dpi=200, pad_inches=0.)


# %%

row = image_selection.query("anomtype == 'damaged_case'").iloc[0]

# ------------------

image = Image.open(row.image_path_resolved).convert("RGB")
image = ImageEnhance.Brightness(image).enhance(1.6)
image = ImageEnhance.Contrast(image).enhance(1.05)
image = ImageEnhance.Sharpness(image).enhance(2.)

mask = read_image(mvtec_mask_path_from_image_path(row.image_path_resolved))[..., 0]

asmap = asmaps[row.image_index]
asmap = cv2.resize(asmap, mask.shape, interpolation=cv2.INTER_NEAREST)

# ------------------

fig, ax = plt.subplots(figsize=(5, 5))
_ = ax.imshow(image)
_ = ax.contour(mask, levels=[0.5], colors="magenta", linewidths=2)
_ = ax.imshow(asmap, cmap="jet", alpha=0.5)
_ = ax.axis("off")
_ = ax.annotate(
    "image + heatmap",
    xy=(0.03, 0.97),  xycoords='axes fraction',
    ha='left', va='top',
    fontsize='xx-large',
    color='k',
    bbox={"facecolor": 'white', "alpha": 0.9, "pad": 5},
)

anomaly_indexes = np.stack(np.where(mask == 1), axis=1)
point_to = anomaly_indexes[np.argmin(-anomaly_indexes[:, 0] + anomaly_indexes[:, 1])][::-1]
_ = ax.annotate(
    "ground truth",
    xy=point_to,  xycoords='data',
    ha='left', va='top',
    fontsize='xx-large',
    color='k',
    bbox={"facecolor": 'white', "alpha": 0.9, "pad": 5},
)
fig.savefig(f"/home/jcasagrandebertoldo/Downloads/heatmap_{row.anomtype}_with_image.jpg", bbox_inches="tight", dpi=100, pad_inches=0)

# ------------------

fig, ax = plt.subplots(figsize=(5, 5))
_ = ax.contour(mask, levels=[0.5], colors="magenta", linewidths=2)
_ = ax.imshow(asmap, cmap="jet", alpha=0.5)
_ = ax.axis("off")
_ = ax.annotate(
    "heatmap",
    xy=(0.03, 0.97),  xycoords='axes fraction',
    ha='left', va='top',
    fontsize='xx-large',
    color='k',
    bbox={"facecolor": 'white', "alpha": 0.9, "pad": 5},
)
fig.savefig(f"/home/jcasagrandebertoldo/Downloads/heatmap_{row.anomtype}_with_image.jpg", bbox_inches="tight", dpi=100, pad_inches=0)

# %%