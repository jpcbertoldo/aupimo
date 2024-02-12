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


from aupimo.oracles import IOUCurvesResult, MaxAvgIOUResult, MaxIOUPerImageResult
from aupimo.utils import per_image_scores_stats

# %%
# Args

# collection [of datasets] = {mvtec, visa}
ACCEPTED_COLLECTIONS = {
    (MVTEC_DIR_NAME := "MVTec"),
    (VISA_DIR_NAME := "VisA"),
}

parser = argparse.ArgumentParser()
_ = parser.add_argument("--rundir", type=Path, required=True)
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
                "--rundir ../../../data/experiments/benchmark/patchcore_wr50/mvtec/bottle",
                "--mvtec-root ../../../data/datasets/MVTec",
                "--visa-root ../../../data/datasets/VisA",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

# %%

# verify that rundir contains a `asmaps.pt` file
if not (asmaps_pt := args.rundir / "asmaps.pt").is_file():
    msg = f"It looks like the rundir is not valid. {args.rundir=}"
    raise ValueError(msg)

else:
    print("rundir looks good")

# %%
# load iou stuff

iou_oracle_threshs_dir = args.rundir / "iou_oracle_threshs"

ioucurves_global = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_global_threshs.pt")

max_avg_iou_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou.json")
max_ious_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img.json")

max_avg_iou_min_thresh_result = MaxAvgIOUResult.load(iou_oracle_threshs_dir / "max_avg_iou_min_thresh.json")
max_ious_min_thresh_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

diff = max_ious_result.ious - max_avg_iou_result.ious_at_thresh
diff_min_thresh = max_ious_min_thresh_result.ious - max_avg_iou_min_thresh_result.ious_at_thresh

diff_stats = per_image_scores_stats(
    per_image_scores=diff,
    images_classes=max_ious_result.image_classes,
    only_class=1,
    repeated_replacement_atol=5e-2,
)
# remove the mean and the median
diff_stats = [stat for stat in diff_stats if stat["stat_name"] not in ("mean", "med")]

# %%
# boxplots

fig, axes = plt.subplots(
    4,
    1,
    figsize=np.array((6, 4)),
    height_ratios=[2, 1, 2, 1],
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

# --------------------- without min thresh ---------------------
# ------- global oracle
ax = axes[0]
boxplots_vs = ax.boxplot(
    [
        max_avg_iou_result.ious_at_thresh[max_avg_iou_result.image_classes == 1],
        max_ious_result.ious[max_ious_result.image_classes == 1],
    ],
    labels=[
        "global oracle",
        "local oracle",
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
_ = ax.set_xlim(0 - (eps := 1e-2), 1 + eps)
_ = ax.set_xticks(np.linspace(0, 1, 5))
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.set_title("without validation")

# ------- diff
ax = axes[1]
boxplots_diff = ax.boxplot(
    [diff[max_ious_result.image_classes == 1]],
    labels=["local - global"],
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
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

# --------------------- WITH min thresh ---------------------
# ------- global oracle
ax = axes[2]
boxplots_vs = ax.boxplot(
    [
        max_avg_iou_min_thresh_result.ious_at_thresh[max_avg_iou_result.image_classes == 1],
        max_ious_min_thresh_result.ious[max_ious_result.image_classes == 1],
    ],
    labels=[
        "global oracle",
        "local oracle",
    ],
    **bp_kwargs_vs,
)
color_code_boxplot(boxplots_vs, 0, "tab:red")
color_code_boxplot(boxplots_vs, 1, "tab:blue")
_ = ax.set_xlim(0 - (eps := 1e-2), 1 + eps)
_ = ax.set_xticks(np.linspace(0, 1, 5))
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
_ = ax.set_title("with min thresh validation")

# ------- diff
ax = axes[3]
boxplots_diff = ax.boxplot(
    [diff_min_thresh[max_ious_result.image_classes == 1]],
    labels=["local - global"],
    **bp_kwargs_diff,
)
color_code_boxplot(boxplots_diff, 0, "black")
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))


# --------------------- ax configs ---------------------
for ax in axes:
    _ = ax.grid(axis="x", alpha=0.4)
    _ = ax.grid(axis="y", ls="--", zorder=-10)
    _ = ax.invert_yaxis()
diff_xlim = axes[1].get_xlim()
diff_min_thresh_xlim = axes[3].get_xlim()
xlim = (min(diff_xlim[0], diff_min_thresh_xlim[0]), max(diff_xlim[1], diff_min_thresh_xlim[1]))
for ax in [axes[1], axes[3]]:
    _ = ax.set_xlim(xlim)

# --------------------- save ---------------------
if args.save:
    fig.savefig(iou_oracle_threshs_dir / "iou_boxplot.png", dpi=300, bbox_inches="tight")

# %%
