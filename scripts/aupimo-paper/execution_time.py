#!/usr/bin/env python
# coding: utf-8

# TODO revise it

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path
import pandas as pd
from progressbar import progressbar
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import matplotlib as mpl
from matplotlib import cm
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

    print("setting numpy print precision")
    np.set_printoptions(floatmode="maxprec", precision=5, suppress=True)
    print("setting pandas print precision")
    pd.set_option("display.precision", 5)

else:
    IS_NOTEBOOK = False


from aupimo.utils_numpy import compare_models_pairwise_wilcoxon
from aupimo import AUPIMOResult

import constants

# %%
# Args

parser = argparse.ArgumentParser()
_ = parser.add_argument("--rundir", type=Path)
if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--rundir ../../data/experiments/benchmark/padim_wr50/mvtec/screw",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

# %%
# Setup (post args)

# TODO change this datasetwise vocabulary
# TODO refactor make it an arg
ROOT_SAVEDIR = Path("/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src")
assert ROOT_SAVEDIR.exists()
assert (IMG_SAVEDIR := ROOT_SAVEDIR / "img").exists()

# %%
# Load data

records = [
    {
        "metric": json_file.stem,
        "json_path": str(json_file),
        "time_taken": json.loads(json_file.read_text())['time_taken'],
        "device": device_dir.name,
        "num_anom_images": int(num_anom_images_dir.name),
    }
    for device_dir in args.rundir.iterdir()
    if device_dir.is_dir()
    and device_dir.name in ("cpu", "cuda")
    for num_anom_images_dir in device_dir.iterdir()
    if num_anom_images_dir.is_dir()
    and num_anom_images_dir.name in ("119", "79", "39")
    for json_file in num_anom_images_dir.iterdir()
    if json_file.is_file() 
    and json_file.suffix == ".json"
]
print(f"{len(records)=}")

data = pd.DataFrame.from_records(records)
data["metric"] = data["metric"].map(lambda s: s.split("_")[0])
data["num_norm_images"] = 41
data["num_images"] = data["num_anom_images"] + data["num_norm_images"]
data["device"] = data["device"].map(lambda s: "gpu" if s == "cuda" else s)
data.query("metric not in ('aupr',)", inplace=True)
data.reset_index(drop=True, inplace=True)
data

# %%
# in seconds!!!!!!!!!!!!
seconds = data.set_index(["metric", "device", "num_anom_images"])["time_taken"].explode()
seconds.sort_index(inplace=True)
minutes = seconds / 60
seconds

# %%
mpl.rcParams.update(RCPARAMS := {
    "font.family": "sans-serif",
    "axes.labelsize": 'medium',
    "xtick.labelsize": 'medium',
    "ytick.labelsize": 'medium',
})

# %%

fig, ax = plt.subplots(figsize=np.array((5.5, 3))*1.1, layout="constrained", dpi=150)
ax2 = ax.twinx()

marker_of_device = {
    "cpu": "o",
    "gpu": "x",
}
color_of_metric = {
    "auroc": "tab:blue",
    "aupro": "tab:red",
    "aupimo": "tab:green",
}

for (metric, device), group in seconds.groupby(["metric", "device", ]):
    
    if metric == "aupro":
        group = minutes.loc[metric, device]
    
    mean = group.reset_index().groupby("num_anom_images")["time_taken"].mean()
    std = group.reset_index().groupby("num_anom_images")["time_taken"].std()
    
    ax_ = ax2 if metric == "aupro" else ax
    
    _ = ax_.plot(
        mean.index.values,
        mean.values,
        label=f"{metric.upper()} on {device.upper()}",
        marker=marker_of_device[device],
        color=color_of_metric[metric],
    )
    
_ = ax.set_xlabel("Number of images (normal | anomalous)")
_ = ax.set_xlim(33, 127)
_ = ax.xaxis.set_ticks(mean.index.values)
xticks_labels = data[["num_anom_images", "num_norm_images", "num_images"]].drop_duplicates()
xticks_labels = xticks_labels.sort_values("num_images").reset_index(drop=True)
xticks_labels = xticks_labels.apply(
    lambda row: f"{row['num_images']} ({row['num_norm_images']}|{row['num_anom_images']})",
    axis=1,
).values
_ = ax.xaxis.set_ticklabels(xticks_labels)

ax.set_ylabel("Execution time (SEC)")
ax.set_ylim(0, 25)
ax.set_yticks(np.linspace(0, 25, 6))
ax.grid(axis="y", which="major", alpha=0.8, linestyle="--")

# make the 2nd y-axis red 
ax2.set_ylim(0, 25)
ax2.set_yticks(np.linspace(0, 25, 6))
ax2.tick_params(axis='y')
ax2.set_ylabel("Execution time (MIN)")

leg = ax.legend(loc="upper left", ncol=2, framealpha=1, title="In seconds")
leg2 = ax2.legend(loc="lower right", ncol=1, framealpha=1, title="In minutes")

_ = fig.savefig(IMG_SAVEDIR / f"execution_time.pdf", bbox_inches="tight", pad_inches=0)
_ = fig.show()

# %%

fig, axes = plt.subplots(
    2, 1, figsize=np.array((6, 3))*1., 
    sharex=True, sharey=False, layout="constrained", dpi=150,
)

ax_top, ax_bottom = axes

marker_of_device = {
    "cpu": "o",
    "gpu": "x",
}
color_of_metric = {
    "auroc": "tab:blue",
    "aupro": "tab:red",
    "aupimo": "tab:green",
}

for (metric, device), group in seconds.groupby(["metric", "device",]):
    
    if metric == "aupro":
        ax = ax_top
    else:
        ax = ax_bottom    
    
    mean = group.reset_index().groupby("num_anom_images")["time_taken"].mean()
    std = group.reset_index().groupby("num_anom_images")["time_taken"].std()
        
    _ = ax.plot(
        mean.index.values,
        mean.values,
        label=f"{metric.upper()} on {device.upper()}",
        marker=marker_of_device[device],
        color=color_of_metric[metric],
    )

  
# X axis (shared)
_ = ax_bottom.set_xlabel("Number of images (normal | anomalous)")
_ = ax_bottom.set_xlim(33, 127)
_ = ax_bottom.xaxis.set_ticks(mean.index.values)
xticks_labels = data[["num_anom_images", "num_norm_images", "num_images"]].drop_duplicates()
xticks_labels = xticks_labels.sort_values("num_images").reset_index(drop=True)
xticks_labels = xticks_labels.apply(
    # lambda row: f"{row['num_images']} ({row['num_norm_images']}|{row['num_anom_images']})",
    lambda row: f"{row['num_images']}",
    axis=1,
).values
_ = ax_bottom.xaxis.set_ticklabels(xticks_labels)

# Y axis 

for ax in axes:
    ax.grid(axis="y", which="major", alpha=0.8, linestyle="--", color="tab:grey")

# (bottom)
ax_bottom.set_yticks(np.linspace(0, 25, 6))
ax_bottom.set_ylim(8, 22)

# (top)
ax_top.set_yticks([240, 600, 960])
ax_top.set_ylim(100, 1100)

# make annotations close to the grid lines of the top axis
# to show the conversion from seconds to minutes
for sec, min in zip([240, 600, 960], [4, 10, 16]):
    ax_top.annotate(
        f"{min} min",
        xy=(33, sec), xycoords="data",
        xytext=(4, 1), textcoords="offset points",
        ha="left", va="bottom", fontsize="x-small",
        color="tab:grey", alpha=1,
    )

# set y label in figure
fig.supylabel("Execution time (SEC)")

leg_bottom = ax_bottom.legend(loc="upper left", ncol=2, fontsize="small")
leg_top = ax_top.legend(loc="lower right", ncol=1, fontsize="small")

fig.savefig(IMG_SAVEDIR / f"execution_time.pdf", bbox_inches="tight", pad_inches=1e-2)

# %%
