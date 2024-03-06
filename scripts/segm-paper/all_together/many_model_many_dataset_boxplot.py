"""TODO(jpcbertoldo): docstring in paper script.

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
from glob import glob
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

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
    # pandas/numpy print options
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option("display.precision", 8)

else:
    IS_NOTEBOOK = False


import const

from aupimo import _validate_tensor
from aupimo._validate_tensor import safe_tensor_to_numpy
from aupimo.oracles import IOUCurvesResult, MaxAvgIOUResult, MaxIOUPerImageResult

# %%
# Args

# collection [of datasets] = {mvtec, visa}
ACCEPTED_COLLECTIONS = {
    (MVTEC_DIR_NAME := "MVTec"),
    (VISA_DIR_NAME := "VisA"),
}

parser = argparse.ArgumentParser()
_ = parser.add_argument("--modelsdir", type=Path, required=True)
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
                "--modelsdir ../../../data/experiments/benchmark-segm-paper",
                "--mvtec-root ../../../data/datasets/MVTec",
                "--visa-root ../../../data/datasets/VisA",
                "--save",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

# %%
# verify `modelsdir`

if (asmaps_pt := args.modelsdir / "asmaps.pt").is_file():
    msg = f"It looks like the `modelsdir` is actually a rundir. {args.modelsdir=}"
    raise ValueError(msg)

# find all `rundirs` in any children of `modelsdir`
# `glob.glob` is necessary to make the `**` work with symlinks
rundirs = sorted(Path(p).parent for p in glob(f"{args.modelsdir!s}/**/asmaps.pt", recursive=True))  # noqa: PTH207

if len(rundirs) == 0:
    msg = f"It looks like the `modelsdir` does not contain any `rundirs`. {args.modelsdir=}"
    raise ValueError(msg)

num_rundirs = len(rundirs)
print(f"`modelsdir` '{args.modelsdir.name}' looks good, it contains {num_rundirs} `rundirs`")

rundirs_df = pd.DataFrame(rundirs, columns=["path"])
rundirs_df["relpath"] = [str(p.relative_to(args.modelsdir)) for p in rundirs]
rundirs_df = rundirs_df.set_index("relpath").sort_index()
rundirs_df = pd.concat(
    [
        rundirs_df,
        rundirs_df.apply(
            lambda row: dict(zip(["model", "collection", "dataset"], row.path.parts[-3:], strict=True)),
            axis=1,
            result_type="expand",
        ),
    ],
    axis=1,
)

# %%
# load results


def open_thresh_selection_oracle(rundir: Path) -> dict[str, np.ndarray | float]:
    iou_oracle_threshs_dir = rundir / "iou_oracle_threshs"

    max_avg_iou_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou.json")
    max_ious_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img.json")

    max_avg_iou_min_thresh_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou_min_thresh.json")
    max_ious_min_thresh_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

    return {
        name: _validate_tensor.safe_tensor_to_numpy(tensor)
        for name, tensor in {
            "global_thresh": max_avg_iou_result.ious_at_thresh,
            "local_thresh": max_ious_result.ious,
            "global_thresh_min_val": max_avg_iou_min_thresh_result.ious_at_thresh,
            "local_thresh_min_val": max_ious_min_thresh_result.ious,
        }.items()
    }


data_thresh_selection_oracle = rundirs_df.apply(
    lambda row: open_thresh_selection_oracle(row.path),
    axis=1,
    result_type="expand",
)

# nvm about global thresh min val
data_thresh_selection_oracle = data_thresh_selection_oracle.drop(columns="global_thresh_min_val")

# %%


def open_superpixel_oracle(rundir: Path) -> dict[str, np.ndarray | float]:
    superpixel_oracle_selection_dir = rundir / "superpixel_oracle_selection"

    # open `optimal_iou.json`
    file_path = superpixel_oracle_selection_dir / "optimal_iou.json"

    print(f"opening {file_path=}")
    with file_path.open("r") as f:
        payload = json.load(f)

    return {
        "superpixel_oracle": np.array(
            [record["iou"] if record is not None else np.nan for record in payload["results"]],
        ),
    }


# the superpixel oracle is independent of the model, so it was executed only for the `patchcore_wr50` model
data_superpixel_oracle = rundirs_df.query("model == 'patchcore_wr50'").apply(
    lambda row: open_superpixel_oracle(row.path),
    axis=1,
    result_type="expand",
)

data_superpixel_oracle_per_dataset = (
    pd.concat(
        [
            rundirs_df[["model", "collection", "dataset"]],
            data_superpixel_oracle,
        ],
        axis=1,
    )
    .query("model == 'patchcore_wr50'")
    .drop(columns="model")
    .reset_index(drop=True)
    .rename(columns={"superpixel_oracle": "ious"})
)

data_superpixel_oracle_per_dataset["avg_iou"] = data_superpixel_oracle_per_dataset.ious.map(np.nanmean)

print("data_superpixel_oracle_per_dataset")
data_superpixel_oracle_per_dataset

data_superpixel_oracle = data_superpixel_oracle_per_dataset.explode("ious").rename(
    columns={"ious": "iou"},
)
data_superpixel_oracle["image_idx"] = data_superpixel_oracle_per_dataset.ious.map(
    lambda ious: np.arange(len(ious)),
).explode()

print("data_superpixel_oracle")
data_superpixel_oracle

data_superpixel_oracle_per_collection = data_superpixel_oracle_per_dataset.groupby(["collection"]).agg(
    avg_iou=("avg_iou", "mean"),
)
data_superpixel_oracle_per_collection.loc["all"] = data_superpixel_oracle_per_dataset["avg_iou"].mean()
# data_superpixel_oracle_per_collection = data_superpixel_oracle_per_collection.reset_index()
data_superpixel_oracle_per_collection

# def change_relpath_model(relpath: str, model: str | None) -> str:
#     return "/".join([model] if model is not None else [] + relpath.split("/")[1:])

# data_superpixel_oracle_copies = []
# for model in rundirs_df.model.unique():
#     data_superpixel_oracle_copy = data_superpixel_oracle.copy().reset_index()
#     data_superpixel_oracle_copy["relpath"] = data_superpixel_oracle_copy["relpath"].map(
#         partial(change_relpath_model, model=model),
#     )
#     data_superpixel_oracle_copies.append(data_superpixel_oracle_copy.set_index("relpath"))
# data_superpixel_oracle = pd.concat(data_superpixel_oracle_copies, axis=0)


# %%


def open_superpixel_bound_dist_heuristic(rundir: Path, num_heuristic_choices: int) -> dict[str, np.ndarray | float]:
    superpixel_bound_dist_heuristic_dir = rundir / "superpixel_bound_dist_heuristic"

    # open `optimal_iou.json`
    file_path = superpixel_bound_dist_heuristic_dir / "superpixel_bound_dist_heuristic.pt"

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if not file_path.is_file():
    # return {"heuristic_5choices": np.array([[np.nan]])}

    payload = torch.load(file_path, map_location="cpu")

    heuristic_per_image = [
        (
            safe_tensor_to_numpy(threshs[minima_args]),
            safe_tensor_to_numpy(heuristic_signal[minima_args]),
        )
        for threshs, heuristic_signal, minima_args in zip(
            payload["threshs_per_image"],
            payload["levelset_mean_dist_curve_per_image"],
            payload["local_minima_idxs_per_image"],
            strict=True,
        )
    ]

    ioucurves = IOUCurvesResult.load(rundir / "iou_oracle_threshs/ioucurves_local_threshs.pt")

    ious_at_heuristic_threshs_per_image = []

    for (heuristic_threshs, heuristic_values), threshs, ious in zip(
        heuristic_per_image,
        ioucurves.threshs,
        ioucurves.per_image_ious,
        strict=True,
    ):
        # the heuristic is to get local minima with the contour distances, so
        # we keep the K lowest local minima
        heuristic_threshs = heuristic_threshs[np.argsort(heuristic_values)][:num_heuristic_choices]
        ious_at_heuristic_threshs = ious[np.searchsorted(threshs, heuristic_threshs)]
        ious_at_heuristic_threshs = np.pad(
            ious_at_heuristic_threshs,
            (0, num_heuristic_choices - len(ious_at_heuristic_threshs)),
            constant_values=np.nan,
        )
        ious_at_heuristic_threshs_per_image.append(ious_at_heuristic_threshs)

    return {
        "heuristic_5choices": np.array(ious_at_heuristic_threshs_per_image),
    }


data_heuristic = rundirs_df.apply(
    lambda row: open_superpixel_bound_dist_heuristic(row.path, num_heuristic_choices=5),
    axis=1,
    result_type="expand",
)

data_heuristic["heuristic_best"] = data_heuristic["heuristic_5choices"].map(lambda ious: np.nanmax(ious, axis=1))


# %%

data = pd.concat(
    [
        rundirs_df[["model", "collection", "dataset"]],
        data_thresh_selection_oracle,
        # data_superpixel_oracle,
        data_heuristic,
    ],
    axis=1,
)
data

data_per_rundir = data.melt(
    id_vars=["model", "collection", "dataset"],
    value_vars=[
        "global_thresh",
        "local_thresh",
        "local_thresh_min_val",
        # "superpixel_oracle",
        "heuristic_best",
    ],
    var_name="method",
    value_name="ious",
)
data = data_per_rundir.explode("ious").rename(columns={"ious": "iou"})
data["image_idx"] = data_per_rundir.ious.map(lambda ious: np.arange(len(ious))).explode()

print("data")
data

data_per_rundir["avg"] = data_per_rundir.ious.map(np.nanmean)

print("data_per_rundir")
data_per_rundir

data_per_model = data_per_rundir.groupby(["model", "collection", "method"]).agg(avg=("avg", "mean")).reset_index()

data_per_model_extra_mean = data_per_model.groupby(["model", "method"]).agg(avg=("avg", "mean")).reset_index()
data_per_model_extra_mean["collection"] = "mean"

data_per_model_extra_all = data_per_rundir.groupby(["model", "method"]).agg(avg=("avg", "mean")).reset_index()
data_per_model_extra_all["collection"] = "all"

data_per_model = pd.concat(
    [
        data_per_model,
        data_per_model_extra_mean,
        data_per_model_extra_all,
    ],
    axis=0,
)
data_per_model["collection"] = data_per_model["collection"].astype(
    pd.CategoricalDtype(categories=["mvtec", "visa", "mean", "all"], ordered=True),
)

data_per_model = data_per_model.sort_values(["model", "collection", "method"])

print("data_per_model")
data_per_model


model_pivot = data_per_model.pivot_table(
    index=["model", "collection"],
    columns="method",
    values="avg",
    aggfunc="first",
    observed=False,
).sort_index()

model_pivot = model_pivot[["global_thresh", "local_thresh", "local_thresh_min_val", "heuristic_best"]]

print("model_pivot")
model_pivot
#

# %%
# viz averages per model

fig, axrow = plt.subplots(
    1,
    2,
    figsize=np.array((10, 3)) * 0.95,
    layout="constrained",
    sharey=True,
    sharex=False,
)

# collections = ["mvtec", "visa", "all"]
collections = ["mvtec", "visa"]

for collec, ax in zip(collections, axrow, strict=False):
    collec_pivot = model_pivot.loc[(slice(None), collec), :].reset_index("collection", drop=True)
    _ = collec_pivot.plot.barh(
        ax=ax,
        color={
            "global_thresh": "tab:red",
            "local_thresh": "lightsteelblue",
            "local_thresh_min_val": "darkblue",
            "heuristic_best": "tab:green",
        },
        edgecolor="black",
        zorder=10,
        width=0.7,
    )
    _ = ax.axvline(
        data_superpixel_oracle_per_collection.loc[collec, "avg_iou"],
        color="black",
        ls="--",
        zorder=5,
        label="Oracle Superpixel Selection",
    )
    _ = ax.set_title(const.COLLECTION_2_LABEL[collec])

ax = axrow[0]
# _ = ax.set_ylabel("Model")
_ = ax.set_ylabel(None)
_ = ax.invert_yaxis()

_ = ax.set_yticklabels([const.MODEL_2_LABEL.get(key := txt.get_text(), key) for txt in ax.get_yticklabels()])

for ax in axrow[1:]:
    # hide yticks
    _ = ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

for ax in axrow:
    _ = ax.set_xticks(np.linspace(0.2, 0.8, 7))
    _ = ax.set_xticks(np.linspace(0, 0.8, 17), minor=True)
    _ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    _ = ax.set_xlabel("Cros-dataset Avg. of Avg. IoU")
    _ = ax.grid(axis="x", alpha=0.8, zorder=-10)
    _ = ax.grid(axis="x", alpha=0.8, zorder=-10, which="minor", ls="--")

_ = axrow[0].set_xlim(0.1, 0.85)
_ = axrow[1].set_xlim(0.1, 0.65)

handles, labels = axrow[0].get_legend_handles_labels()
labels = [const.METHOD_2_LABEL.get(label, label) for label in labels]
leg = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.55, -0.03),
)

for ax in axrow:
    _ = ax.get_legend().remove()

if args.save:
    fig.savefig(
        "../../../../2024-03-segm-paper/src/img/avg_avg_iou_per_model.pdf", bbox_inches="tight", pad_inches=1e-2,
    )

# %%
# viz one model in detail

best_model = "efficientad_wr101_s_ext"
data_best_model = data_per_rundir.query("model == @best_model").drop(columns="model").reset_index(drop=True)
data_best_model["collection_dataset"] = data_best_model["collection"] + "_" + data_best_model["dataset"]

fig, axes = plt.subplots(
    4,
    7,
    figsize=np.array((10, 3.5)) * 1.4,
    layout="constrained",
)
axrow = axes.flatten()

data_best_model_box = data_best_model.drop(columns=["collection", "dataset", "avg"])

# METHODS = ["global_thresh", "local_thresh", "local_thresh_min_val", "heuristic_best"]
# METHODS = ["local_thresh_min_val", "heuristic_best"]
METHODS = ["local_thresh", "heuristic_best"]
METHODS_LABELS = {
    "local_thresh": "Oracle",
    "heuristic_best": "Heuristic",
}


def color_code_boxplot(boxplots_dict: dict, bplot_idx: int, color: str):
    boxplots_dict["boxes"][bplot_idx].set_edgecolor(color)
    boxplots_dict["fliers"][bplot_idx].set(markeredgecolor=color, markerfacecolor="white")
    boxplots_dict["means"][bplot_idx].set(markeredgecolor=color, markerfacecolor="white")
    bplot_idx_pairs = slice(bplot_idx * 2, (bplot_idx + 1) * 2)
    for patch in (
        boxplots_dict["whiskers"][bplot_idx_pairs]
        + boxplots_dict["caps"][bplot_idx_pairs]
        + [boxplots_dict["medians"][bplot_idx]]
    ):
        patch.set_color(color)


for axidx, (collection_dataset, sub_data) in enumerate(data_best_model_box.groupby("collection_dataset")):
    if axidx >= 6:
        axidx += 1

    ax = axrow[axidx]
    sub_data = sub_data.set_index("method")

    boxplots_vs = ax.boxplot(
        [(x := sub_data.loc[method, "ious"])[~np.isnan(x)] for method in METHODS],
        labels=METHODS,
        positions=[0.25, .75],
        **(
            bp_kwargs_vs := dict(  # noqa: C408
                vert=True,
                notch=False,
                showmeans=True,
                meanprops={"marker": "d", "markerfacecolor": "white"},
                # make it possible to color the box with white (not transparent)
                patch_artist=True,
                boxprops={"facecolor": "white"},
                widths=0.2,
            )
        ),
    )

    color_code_boxplot(boxplots_vs, 0, "darkblue")
    color_code_boxplot(boxplots_vs, 1, "tab:green")

    if axidx in tuple(range(21, 28)):
        _ = ax.set_xticklabels([METHODS_LABELS.get(label, label) for label in METHODS])
    else:
        _ = ax.set_xticks([])

    _ = ax.set_xlim(0, 1)
    # _ = ax.invert_xaxis()

    _ = ax.set_yticks(np.linspace(0, 1, 5))
    if axidx % 7 == 0:
        _ = ax.set_yticklabels(["", "25%", "50%", "75%", ""])
    else:
        _ = ax.set_yticklabels([])
    _ = ax.set_ylim(0, 1)
    _ = ax.grid(axis="y", alpha=0.8, zorder=-10)

    collection, dataset = collection_dataset.split("_", 1)
    _ = ax.set_title(f"{const.COLLECTION_2_LABEL[collection]} / {const.DATASET_2_LABEL[dataset]}")

for ax in [axes[0, -1]]:
    _ = ax.axis("off")


if args.save:
    fig.savefig(
        "../../../../2024-03-segm-paper/src/img/iou_distrib_efficientad.pdf", bbox_inches="tight", pad_inches=1e-2,
    )


# %%

collection, dataset = "visa", "chewinggum"
heuristic_best_ious = data_best_model.query("dataset == @dataset and collection == @collection and method == 'heuristic_best'").ious.values[0]

local_thresh_ious = data_best_model.query("dataset == @dataset and collection == @collection and method == 'local_thresh'").ious.values[0]

# %%
argsorted = np.argsort(heuristic_best_ious)[:-np.isnan(heuristic_best_ious).sum()][::-1]
argsorted
idx = argsorted[0]
# idx = 69
idx
heuristic_best_ious[idx]
# %%

# tmp = data_heuristic.loc["efficientad_wr101_s_ext/mvtec/transistor", "heuristic_5choices"]
tmp = data_heuristic.loc["efficientad_wr101_s_ext/visa/chewinggum", "heuristic_5choices"]
signal = np.median(np.abs(np.diff(tmp, axis=1)), axis=1)
signal
argsorted = np.argsort(signal)
argsorted
signal[argsorted]
sum(np.isnan(signal))
image_idx = argsorted[-23]
image_idx
signal[image_idx]
tmp[image_idx]


# %%
