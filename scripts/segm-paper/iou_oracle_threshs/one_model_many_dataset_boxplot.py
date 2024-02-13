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

import matplotlib as mpl
import numpy as np
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

else:
    IS_NOTEBOOK = False


from aupimo.oracles import MaxAvgIOUResult, MaxIOUPerImageResult

# %%
# Args

# collection [of datasets] = {mvtec, visa}
ACCEPTED_COLLECTIONS = {
    (MVTEC_DIR_NAME := "MVTec"),
    (VISA_DIR_NAME := "VisA"),
}

parser = argparse.ArgumentParser()
_ = parser.add_argument("--rundir_parent", type=Path, required=True)
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
                # "--rundir_parent ../../../data/experiments/benchmark/patchcore_wr50/mvtec",
                "--rundir_parent ../../../data/experiments/benchmark/efficientad_wr101_m_ext/mvtec",
                # "--rundir_parent ../../../data/experiments/benchmark/patchcore_wr50",
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
# verify that rundir_parent is actually a parent of rundirs

if (asmaps_pt := args.rundir_parent / "asmaps.pt").is_file():
    msg = f"It looks like the `rundir_parent` is actually a rundir. {args.rundir_parent=}"
    raise ValueError(msg)

# find all `rundirs` in any children of `rundir_parent`
rundirs = sorted(p.parent for p in args.rundir_parent.glob("**/asmaps.pt"))

if len(rundirs) == 0:
    msg = f"It looks like the `rundir_parent` does not contain any `rundirs`. {args.rundir_parent=}"
    raise ValueError(msg)

num_rundirs = len(rundirs)
print(f"`rundir_parent` '{args.rundir_parent.name}' looks good, it contains {num_rundirs} `rundirs`")

rundirs_relpaths = [str(p.relative_to(args.rundir_parent)) for p in rundirs]

# %%
# load iou stuff

data_per_relpath = {}

for rundir, relpath in zip(rundirs, rundirs_relpaths, strict=True):
    iou_oracle_threshs_dir = rundir / "iou_oracle_threshs"

    max_avg_iou_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou.json")
    max_ious_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img.json")

    max_avg_iou_min_thresh_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou_min_thresh.json")
    max_ious_min_thresh_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

    diff = max_ious_result.ious - max_avg_iou_result.ious_at_thresh
    diff_min_thresh = max_ious_min_thresh_result.ious - max_avg_iou_min_thresh_result.ious_at_thresh

    data_per_relpath[relpath] = {
        "max_avg_iou_result": max_avg_iou_result,
        "max_ious_result": max_ious_result,
        "max_avg_iou_min_thresh_result": max_avg_iou_min_thresh_result,
        "max_ious_min_thresh_result": max_ious_min_thresh_result,
        "diff": diff,
        "diff_min_thresh": diff_min_thresh,
    }

# %%
# averages

avg_iou_at_global = []
avg_iou_at_local = []
avg_iou_at_global_min_thresh = []
avg_iou_at_local_min_thresh = []

for dataset_idx, relpath in enumerate(rundirs_relpaths):
    data = data_per_relpath[relpath]
    max_avg_iou_result = data["max_avg_iou_result"]
    max_ious_result = data["max_ious_result"]
    max_avg_iou_min_thresh_result = data["max_avg_iou_min_thresh_result"]
    max_ious_min_thresh_result = data["max_ious_min_thresh_result"]
    diff = data["diff"]
    diff_min_thresh = data["diff_min_thresh"]

    avg_iou_at_global.append(
        max_avg_iou_result.ious_at_thresh[max_avg_iou_result.image_classes == 1].mean(),
    )

    avg_iou_at_local.append(
        max_ious_result.ious[max_ious_result.image_classes == 1].mean(),
    )
    avg_iou_at_global_min_thresh.append(
        max_avg_iou_min_thresh_result.ious_at_thresh[max_avg_iou_min_thresh_result.image_classes == 1].mean(),
    )

    avg_iou_at_local_min_thresh.append(
        max_ious_min_thresh_result.ious[max_ious_min_thresh_result.image_classes == 1].mean(),
    )

avg_avg_iou_at_global = np.mean(avg_iou_at_global)
avg_avg_iou_at_local = np.mean(avg_iou_at_local)
avg_avg_iou_at_global_min_thresh = np.mean(avg_iou_at_global_min_thresh)
avg_avg_iou_at_local_min_thresh = np.mean(avg_iou_at_local_min_thresh)

# %%
# plot averages
fig, axrow = plt.subplots(
    1,
    2,
    figsize=np.array((6, 4)) * 1.4,
    layout="constrained",
    sharey=True,
    sharex=True,
)

ind = np.arange(num_rundirs)
width = 0.4
actual_width = width * .65

ax = axrow[0]
_ = ax.barh(
    ind - width / 2, avg_iou_at_global, actual_width,
    label="global", zorder=10, color="tab:red", edgecolor="black",
)
_ = ax.barh(
    ind + width / 2, avg_iou_at_local, actual_width,
    label="local", zorder=10, color="tab:blue", edgecolor="black",
)
_ = ax.barh(
    num_rundirs - width / 2, avg_avg_iou_at_global, actual_width,
    zorder=10, color="tab:red", edgecolor="black",
)
_ = ax.barh(
    num_rundirs + width / 2, avg_avg_iou_at_local, actual_width,
    zorder=10, color="tab:blue", edgecolor="black",
)
_ = ax.set_title("without validation")

# +1 is for the overall average
_ = ax.set_ylim(-1 + width, num_rundirs + 1 - width)
_ = ax.set_yticks([*ind.tolist(), num_rundirs])
_ = ax.set_yticklabels([*rundirs_relpaths, "average"])
_ = ax.set_ylabel("Dataset")
_ = ax.invert_yaxis()

_ = ax.set_xlim(0, 1)
_ = ax.set_xticks(np.linspace(0, 1, 6))
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax = axrow[1]
_ = ax.barh(
    ind - width / 2, avg_iou_at_global_min_thresh, actual_width,
    label="global", zorder=10, color="tab:red", edgecolor="black",
)
_ = ax.barh(
    ind + width / 2, avg_iou_at_local_min_thresh, actual_width,
    label="local", zorder=10, color="tab:blue", edgecolor="black",
)
_ = ax.barh(
    num_rundirs - width / 2, avg_avg_iou_at_global_min_thresh, actual_width,
    zorder=10, color="tab:red", edgecolor="black",
)
_ = ax.barh(
    num_rundirs + width / 2, avg_avg_iou_at_local_min_thresh, actual_width,
    zorder=10, color="tab:blue", edgecolor="black",
)
_ = ax.set_title("with min thresh validation")

_ = ax.legend(loc="lower right")

for ax in axrow:
    _ = ax.xaxis.grid(alpha=0.8, zorder=-10)
    _ = ax.set_xlabel("Avg. IoU")

    _ = ax.yaxis.grid(alpha=0.8, zorder=-10, ls="--")

# --------------------- save ---------------------
if args.save:
    fig.savefig(args.rundir_parent / "avg_iou_barplot_v0.pdf", dpi=300, bbox_inches="tight", pad_inches=1e-2)

# %%
# boxplots

fig, axes = plt.subplots(
    num_rundirs,
    4,
    figsize=np.array((18, num_rundirs)),
    layout="constrained",
)


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


for dataset_idx, relpath in enumerate(rundirs_relpaths):
    data = data_per_relpath[relpath]
    max_avg_iou_result = data["max_avg_iou_result"]
    max_ious_result = data["max_ious_result"]
    max_avg_iou_min_thresh_result = data["max_avg_iou_min_thresh_result"]
    max_ious_min_thresh_result = data["max_ious_min_thresh_result"]
    diff = data["diff"]
    diff_min_thresh = data["diff_min_thresh"]

    ax_vs = axes[dataset_idx, 0]
    ax_diff = axes[dataset_idx, 1]
    ax_min_thresh_vs = axes[dataset_idx, 2]
    ax_min_thresh_diff = axes[dataset_idx, 3]

    # --------------------- without min thresh ---------------------
    # ------- global oracle
    ax = ax_vs
    boxplots_vs = ax.boxplot(
        [
            max_avg_iou_result.ious_at_thresh[max_avg_iou_result.image_classes == 1],
            max_ious_result.ious[max_ious_result.image_classes == 1],
        ],
        labels=[
            "global",
            "local",
        ],
        **(
            bp_kwargs_vs := dict(  # noqa: C408
                vert=False,
                notch=False,
                showmeans=True,
                meanprops={"marker": "d", "markerfacecolor": "white"},
                # make it possible to color the box with white (not transparent)
                patch_artist=True,
                boxprops={"facecolor": "white"},
                widths=0.5,
            )
        ),
    )
    color_code_boxplot(boxplots_vs, 0, "tab:red")
    color_code_boxplot(boxplots_vs, 1, "tab:blue")

    # ------- diff
    ax = ax_diff
    boxplots_diff = ax.boxplot(
        [diff[max_ious_result.image_classes == 1]],
        labels=[""],
        **(
            bp_kwargs_diff := dict(  # noqa: C408
                vert=False,
                notch=False,
                showmeans=True,
                meanprops={"marker": "d", "markerfacecolor": "white"},
                # make it possible to color the box with white (not transparent)
                patch_artist=True,
                boxprops={"facecolor": "white"},
                widths=0.5,
            )
        ),
    )
    color_code_boxplot(boxplots_diff, 0, "black")

    # --------------------- WITH min thresh ---------------------
    # ------- global oracle
    ax = ax_min_thresh_vs
    boxplots_vs = ax.boxplot(
        [
            max_avg_iou_min_thresh_result.ious_at_thresh[max_avg_iou_min_thresh_result.image_classes == 1],
            max_ious_min_thresh_result.ious[max_ious_min_thresh_result.image_classes == 1],
        ],
        labels=[
            "global",
            "local",
        ],
        **(
            bp_kwargs_vs := dict(  # noqa: C408
                vert=False,
                notch=False,
                showmeans=True,
                meanprops={"marker": "d", "markerfacecolor": "white"},
                # make it possible to color the box with white (not transparent)
                patch_artist=True,
                boxprops={"facecolor": "white"},
                widths=0.5,
            )
        ),
    )
    color_code_boxplot(boxplots_vs, 0, "tab:red")
    color_code_boxplot(boxplots_vs, 1, "tab:blue")

    # ------- diff
    ax = ax_min_thresh_diff
    boxplots_diff = ax.boxplot(
        [diff_min_thresh[max_ious_result.image_classes == 1]],
        labels=[""],
        **(
            bp_kwargs_diff := dict(  # noqa: C408
                vert=False,
                notch=False,
                showmeans=True,
                meanprops={"marker": "d", "markerfacecolor": "white"},
                # make it possible to color the box with white (not transparent)
                patch_artist=True,
                boxprops={"facecolor": "white"},
                widths=0.5,
            )
        ),
    )
    color_code_boxplot(boxplots_diff, 0, "black")

# --------------------- ax configs ---------------------
ax = axes[0, 0]
_ = ax.set_title("oracles without validation")

ax = axes[0, 1]
_ = ax.set_title("diff = local - global")

ax = axes[0, 2]
_ = ax.set_title("oracles with min thresh validation")

ax = axes[0, 3]
_ = ax.set_title("diff = local - global")

xlim = (0 - (eps := 1e-2), 1 + eps)
xticks = np.linspace(0, 1, 11)

for ax in axes[-1]:
    _ = ax.set_xlim(*xlim)
    _ = ax.set_xticks(xticks)
    _ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

for ax in axes[:-1].flatten():
    _ = ax.set_xlim(*xlim)
    _ = ax.set_xticks(xticks)
    _ = ax.set_xticklabels([])
    # hide the ticks
    _ = ax.tick_params(axis="x", length=0)

for ax, relpath in zip(axes[:, 0], rundirs_relpaths, strict=False):
    _ = ax.set_ylabel(relpath)

for ax in axes.flatten():
    _ = ax.grid(axis="x", alpha=0.4)
    _ = ax.grid(axis="y", ls="--", zorder=-10)
    _ = ax.invert_yaxis()

# --------------------- save ---------------------
if args.save:
    fig.savefig(args.rundir_parent / "iou_boxplot_v0.pdf", dpi=300, bbox_inches="tight", pad_inches=1e-2)

# %%
# TODO v1: put (num_datasets x 2) putting the diff bellow the vs, in the same subplot

# %%
# TODO v2: put without diff, show global vs local vs local w/ min thresh
