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

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

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

# %% avg iou curve
fig, ax = plt.subplots(figsize=[6, 4])

ioucurves_global.plot_avg_iou_curve(ax)
ioucurves_global.ax_cfg_xaxis(ax)
ioucurves_global.ax_cfg_yaxis(ax)

_ = ioucurves_global.axvline_min_thresh(ax, max_avg_iou_min_thresh_result.min_thresh)
_ = ioucurves_global.axvline_global_oracle(ax, max_avg_iou_min_thresh_result.thresh)
_ = ioucurves_global.scatter_global_oracle(
    ax,
    max_avg_iou_min_thresh_result.thresh,
    max_avg_iou_min_thresh_result.avg_iou,
)

leg = ax.legend(loc="upper left", framealpha=1, fontsize="small")
for line in leg.get_lines():
    line.set_linewidth(2.5)

# %%
# diff stats curves

fig, ax = plt.subplots(figsize=np.array((6, 4)))

for bp_stat, color in zip(diff_stats, ["tab:orange", "tab:purple", "tab:brown", "tab:cyan"], strict=True):
    iou_curve = ioucurves_global.per_image_ious[bp_stat["image_idx"]]
    _ = ioucurves_global.plot_iou_curve(ax, bp_stat["image_idx"], color=color)
    _ = ioucurves_global.scatter_local_oracle(
        ax,
        max_ious_result.threshs[bp_stat["image_idx"]],
        max_ious_result.ious[bp_stat["image_idx"]],
    )
_ = ioucurves_global.axvline_min_thresh(ax, max_avg_iou_min_thresh_result.min_thresh)
_ = ioucurves_global.axvline_global_oracle(ax, max_avg_iou_min_thresh_result.thresh)
leg = ax.legend(loc="upper left", framealpha=1, fontsize="small")
for line in leg.get_lines():
    line.set_linewidth(2.5)
ioucurves_global.ax_cfg_xaxis(ax)
ioucurves_global.ax_cfg_yaxis(ax)

# %%
