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
from glob import glob
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
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
    np.set_printoptions(precision=2, suppress=True)
    pd.set_option("display.precision", 8)

else:
    IS_NOTEBOOK = False


from aupimo import _validate_tensor
from aupimo.oracles import MaxAvgIOUResult, MaxIOUPerImageResult

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
rundirs_df["relpath"] = [p.relative_to(args.modelsdir) for p in rundirs]
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
# load iou stuff


def open_iou_oracle_threshs(rundir: Path) -> dict[str, MaxAvgIOUResult | MaxIOUPerImageResult]:
    iou_oracle_threshs_dir = rundir / "iou_oracle_threshs"

    max_avg_iou_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou.json")
    max_ious_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img.json")

    max_avg_iou_min_thresh_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou_min_thresh.json")
    max_ious_min_thresh_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

    return {
        name: _validate_tensor.safe_tensor_to_numpy(tensor)
        for name, tensor in {
            "global": max_avg_iou_result.ious_at_thresh,
            "local": max_ious_result.ious,
            "global_min_thresh": max_avg_iou_min_thresh_result.ious_at_thresh,
            "local_min_thresh": max_ious_min_thresh_result.ious,
        }.items()
    }


data = pd.concat(
    [
        rundirs_df[["model", "collection", "dataset"]],
        rundirs_df.apply(lambda row: open_iou_oracle_threshs(row.path), axis=1, result_type="expand"),
    ],
    axis=1,
)
data_per_rundir = data.melt(
    id_vars=["model", "collection", "dataset"],
    value_vars=["global", "local", "global_min_thresh", "local_min_thresh"],
    var_name="oracle",
    value_name="ious",
)
data = data_per_rundir.explode("ious").rename(columns={"ious": "iou"})
data["image_idx"] = data_per_rundir.ious.map(lambda ious: np.arange(len(ious))).explode()

print("data")
data

data_per_rundir["avg"] = data_per_rundir.ious.map(np.nanmean)

print("data_per_rundir")
data_per_rundir

data_per_model = data_per_rundir.groupby(["model", "collection", "oracle"]).agg(avg=("avg", "mean")).reset_index()

data_per_model_extra_mean = data_per_model.groupby(["model", "oracle"]).agg(avg=("avg", "mean")).reset_index()
data_per_model_extra_mean["collection"] = "mean"

data_per_model_extra_all = data_per_rundir.groupby(["model", "oracle"]).agg(avg=("avg", "mean")).reset_index()
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

data_per_model = data_per_model.sort_values(["model", "collection", "oracle"])

print("data_per_model")
data_per_model

model_pivot = data_per_model.pivot_table(
    index=["model", "collection"],
    columns="oracle",
    values="avg",
    aggfunc="first",
    observed=False,
).sort_index()

print("model_pivot")
model_pivot

# %%
# viz averages per model

fig, axrow = plt.subplots(
    1,
    3,
    figsize=np.array((9, 3)) * 1.3,
    layout="constrained",
    sharey=True,
    sharex=True,
)

collections = ["mvtec", "visa", "all"]

for collec, ax in zip(collections, axrow, strict=False):
    collec_pivot = model_pivot.loc[(slice(None), collec), :].reset_index("collection", drop=True)
    _ = collec_pivot.plot.barh(
        ax=ax,
        color={
            "global": "lightcoral",
            "local": "lightsteelblue",
            "global_min_thresh": "darkred",
            "local_min_thresh": "darkblue",
        },
        edgecolor="black",
        zorder=10,
        width=0.7,
    )
    _ = ax.set_title(collec)

ax = axrow[0]
_ = ax.set_ylabel("Model")
_ = ax.invert_yaxis()

for ax in axrow[1:]:
    # hide yticks
    _ = ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

_ = ax.set_xlim(0, 0.6)
_ = ax.set_xticks(np.linspace(0.1, 0.5, 5))
_ = ax.set_xticks(np.linspace(0, 0.6, 13), minor=True)
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

for ax in axrow:
    _ = ax.set_xlabel("Avg. Avg. IoU")
    _ = ax.grid(axis="x", alpha=0.8, zorder=-10)
    _ = ax.grid(axis="x", alpha=0.8, zorder=-10, which="minor", ls="--")

fig.legend(
    *axrow[0].get_legend_handles_labels(),
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.03),
)

for ax in axrow:
    _ = ax.get_legend().remove()

if args.save:
    fig.savefig(args.modelsdir / "avg_avg_iou_per_model.pdf", bbox_inches="tight", pad_inches=1e-2)

# %%
# viz averages per oracle

fig, axrow = plt.subplots(
    1,
    3,
    figsize=np.array((9, 3)) * 1.3,
    layout="constrained",
    sharey=True,
    sharex=True,
)

model_pivot_transposed = model_pivot.stack("oracle").unstack("model")

collections = ["mvtec", "visa", "all"]

for collec, ax in zip(collections, axrow, strict=False):
    # the slice(None) is for selecting all models
    collec_pivot = model_pivot_transposed.loc[(collec, slice(None)), :].reset_index("collection", drop=True)

    _ = collec_pivot.plot.barh(
        ax=ax,
        edgecolor="black",
        zorder=10,
        width=0.75,
    )
    _ = ax.set_title(collec)

ax = axrow[0]
_ = ax.set_ylabel("Oracle")
_ = ax.invert_yaxis()

for ax in axrow[1:]:
    # hide yticks
    _ = ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

_ = ax.set_xlim(0, 0.6)
_ = ax.set_xticks(np.linspace(0.1, 0.5, 5))
_ = ax.set_xticks(np.linspace(0, 0.6, 13), minor=True)
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

for ax in axrow:
    _ = ax.set_xlabel("Avg. Avg. IoU")
    _ = ax.grid(axis="x", alpha=0.8, zorder=-10)
    _ = ax.grid(axis="x", alpha=0.8, zorder=-10, which="minor", ls="--")

fig.legend(
    *axrow[0].get_legend_handles_labels(),
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.03),
)

for ax in axrow:
    _ = ax.get_legend().remove()

if args.save:
    fig.savefig(args.modelsdir / "avg_avg_iou_per_oracle.pdf", bbox_inches="tight", pad_inches=1e-2)

# %%
