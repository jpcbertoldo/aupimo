#!/usr/bin/env python
# coding: utf-8

# TODO revise it

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from pathlib import Path
from progressbar import progressbar
import json
import numpy as np

import numpy as np
import pandas as pd
import torch
from anomalib.metrics import AUPR, AUPRO, AUROC
from PIL import Image
from torch import Tensor
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats

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
    np.set_printoptions(floatmode="maxprec", precision=3, suppress=True)
    print("setting pandas print precision")
    pd.set_option("display.precision", 3)

else:
    IS_NOTEBOOK = False

import constants
from aupimo import AUPIMOResult

# %%
# Args
pass

# %%
# Setup (post args)

# TODO make this an arg?
BENCHMARK_DIR = Path(__file__).parent / "../../data/experiments/benchmark/"
print(f"{BENCHMARK_DIR.resolve()=}")
assert BENCHMARK_DIR.exists()

# TODO change this datasetwise vocabulary
# TODO make it an arg?
DATASETWISE_MAINTEXT_DIR = "/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src/img/datasetwise_maintext"
DATASETWISE_MAINTEXT_DIR = Path(DATASETWISE_MAINTEXT_DIR)
assert DATASETWISE_MAINTEXT_DIR.exists()

# %%
# Load data

records = [
    {
        "dir": str(category_dir),
        "model": model_dir.name,
        "dataset": dataset_dir.name,
        "category": category_dir.name,
    }
    for model_dir in BENCHMARK_DIR.iterdir()
    if model_dir.is_dir() and model_dir.name != "debug"
    for dataset_dir in model_dir.iterdir()
    if dataset_dir.is_dir()
    for category_dir in dataset_dir.iterdir()
    if category_dir.is_dir()
]
print(f"{len(records)=}")

for record in progressbar(records):

    d = Path(record["dir"])

    for m in [
        "auroc", "aupro",
        # "aupr",
        # "aupro_05",
    ]:
        try:
            record[m] = json.loads((d / f"{m}.json").read_text())['value']
        except FileNotFoundError:
            print(f"skipping `{m}` for {d}")

    try:
        assert (aupimodir := d / "aupimo").exists()
        aupimoresult = AUPIMOResult.load(aupimodir / "aupimos.json")
        record["aupimo"] = aupimoresult.aupimos.numpy()

    except AssertionError:
        print(f"skipping `aupimo` for {d}")


data = pd.DataFrame.from_records(records)
data = data.set_index(["model", "dataset", "category"]).sort_index().reset_index()

print("has any model with any NaN?")
data.isna().any(axis=1).any()

assert data.shape[0] == 351  # 27 models * 13 datasets 

print("df")
data.head(2)

# %%
# MAINTEXT DATASETWISE

REPLACE_LABELS = {"model": constants.MODELS_LABELS_MAINTEXT, "dataset": constants.DATASETS_LABELS, "category": constants.CATEGORIES_LABELS}
MIN_CONFIDENCE = 0.95
DATASETWISE_RC = {
    "font.family": "sans-serif",
    "axes.titlesize": "xx-large",
    "axes.labelsize": 'large',
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
}

data["avg_aupimo"] = data["aupimo"].apply(lambda x: np.nanmean(x))
data["p33_aupimo"] = data["aupimo"].apply(lambda x: np.nanpercentile(x, 33))

df_plot = data.reset_index()
df_plot = df_plot.set_index("model").loc[constants.MAINTEXT_MODELS].reset_index()
df_plot = df_plot.set_index(["dataset", "category", "model"]).sort_index()

# models order!!!!
# (-) sign is because higher is better
metric = "avg_aupimo"
pivot = df_plot.pivot_table(index=["dataset", "category"], columns="model", values=metric)
pivot_ranks = (-pivot).apply(scipy.stats.rankdata, axis=1, result_type="broadcast")
pivot_ranks_avg = pivot_ranks.mean(axis=0).sort_values().reset_index().rename(columns={0: "avg_rank"})
models_ordered = pivot_ranks_avg['model'].values.tolist()
num_models = len(models_ordered)

def get_metric_data(metric):

    # ====================
    # compute the average of the metric per model

    # scatter
    metric_plotdf = df_plot[metric].reset_index()

    metric_pmodel = metric_plotdf.groupby("model")[metric]
    metric_pmodel_avg = metric_pmodel.mean().sort_values().reset_index()

    metric_pmodel_pdataset = metric_plotdf.groupby(["dataset", "model"])[metric]
    metric_pmodel_pdataset_avg = metric_pmodel_pdataset.mean().sort_values().reset_index()

    # order models
    metric_plotdf = metric_plotdf.set_index("model").loc[models_ordered].reset_index()
    metric_pmodel_avg = metric_pmodel_avg.set_index("model").loc[models_ordered].reset_index()
    metric_pmodel_pdataset_avg = metric_pmodel_pdataset_avg.set_index("model").loc[models_ordered].reset_index()

    return metric_plotdf, metric_pmodel_avg, metric_pmodel_pdataset_avg

auroc_plotdf, auroc_pmodel_avg, auroc_pmodel_pdset_avg = get_metric_data("auroc")
aupro_plotdf, aupro_pmodel_avg, aupro_pmodel_pdset_avg = get_metric_data("aupro")
aupimo_avg_plotdf, aupimo_avg_pmodel_avg, aupimo_avg_pmodel_pdset_avg = get_metric_data("avg_aupimo")
aupimo_p33_plotdf, aupimo_p33_pmodel_avg, aupimo_p33_pmodel_pdset_avg = get_metric_data("p33_aupimo")

MODEL2Y = dict(map(reversed, enumerate(models_ordered[::-1])))

# auroc_plotdf IS ARBITRARY, JUST TO GET THE UNIQUE DATASET/CATEGORY PAIRS
unique_dc = auroc_plotdf[["dataset", "category"]].sort_values(by=["dataset", "category"]).drop_duplicates().values
DYS = np.linspace(-(DY := 0.3), DY, len(unique_dc))
DC2DY = {
    tuple(dc): DYS[dcidx]
    for dcidx, dc in enumerate(unique_dc)
}

def get_y(dfmetric):
    ys = dfmetric["model"].map(MODEL2Y)
    dys = dfmetric[["dataset", "category"]].apply(lambda row: DC2DY[tuple(row)], axis=1)
    return ys + dys

auroc_plotdf['y'] = get_y(auroc_plotdf)
aupro_plotdf['y'] = get_y(aupro_plotdf)
aupimo_avg_plotdf['y'] = get_y(aupimo_avg_plotdf)
aupimo_p33_plotdf['y'] = get_y(aupimo_p33_plotdf)

auroc_pmodel_avg['y'] = auroc_pmodel_avg["model"].map(MODEL2Y)
aupro_pmodel_avg['y'] = aupro_pmodel_avg["model"].map(MODEL2Y)
aupimo_avg_pmodel_avg['y'] = aupimo_avg_pmodel_avg["model"].map(MODEL2Y)
aupimo_p33_pmodel_avg['y'] = aupimo_p33_pmodel_avg["model"].map(MODEL2Y)

auroc_pmodel_pdset_avg['y'] = auroc_pmodel_pdset_avg["model"].map(MODEL2Y)
aupro_pmodel_pdset_avg['y'] = aupro_pmodel_pdset_avg["model"].map(MODEL2Y)
aupimo_avg_pmodel_pdset_avg['y'] = aupimo_avg_pmodel_pdset_avg["model"].map(MODEL2Y)
aupimo_p33_pmodel_pdset_avg['y'] = aupimo_p33_pmodel_pdset_avg["model"].map(MODEL2Y)


DATASET2MARKER = dict(mvtec=".", visa="x")

def get_fig_datasetwise_maintext():
    return plt.subplots(figsize=(7, 4), dpi=200, layout="constrained",)


with mpl.rc_context(rc=DATASETWISE_RC):

    def scatter(ax, dfmetric, col, **kwargs):
        mvtec_mask = dfmetric["dataset"] == "mvtec"
        visa_mask = dfmetric["dataset"] == "visa"
        _ = ax.scatter(
            x=dfmetric[col][mvtec_mask].values,
            y=dfmetric["y"][mvtec_mask].values,
            marker="^",
            s=20,
            alpha=0.6,
            **kwargs,
        )
        _ = ax.scatter(
            x=dfmetric[col][visa_mask].values,
            y=dfmetric["y"][visa_mask].values,
            marker="v",
            s=20,
            alpha=0.6,
            **kwargs,
        )

    def pdset_avg(ax, metric_pmodel_pdataset_avg, col, **kwargs):
        mvtec_mask = metric_pmodel_pdataset_avg["dataset"] == "mvtec"
        visa_mask = metric_pmodel_pdataset_avg["dataset"] == "visa"
        _ = ax.scatter(
            metric_pmodel_pdataset_avg[col][mvtec_mask].values,
            metric_pmodel_pdataset_avg["y"][mvtec_mask].values,
            marker="^",
            s=150,
            edgecolor="black",
            lw=2,  # border width
            **kwargs,
        )
        _ = ax.scatter(
            metric_pmodel_pdataset_avg[col][visa_mask].values,
            metric_pmodel_pdataset_avg["y"][visa_mask].values,
            marker="v",
            s=150,
            lw=2,  # border width
            edgecolor="black",
            **kwargs,
        )

    # ================================================================================
    # AUROC and AUPRO
    fig_auroc_aupro, ax = get_fig_datasetwise_maintext()

    scatter(ax, auroc_plotdf, "auroc", color="tab:blue")
    scatter(ax, aupro_plotdf, "aupro", color="tab:red")

    # plot the average of the averages per model
    _ = ax.scatter(
        auroc_pmodel_avg["auroc"].values,
        auroc_pmodel_avg["y"].values,
        marker="d",
        color="tab:blue",
        edgecolor="black",
        lw=2,  # border width
        s=150,
        zorder=100,
    )
    _ = ax.scatter(
        aupro_pmodel_avg["aupro"].values,
        aupro_pmodel_avg["y"].values,
        marker="d",
        color="tab:red",
        edgecolor="black",
        lw=2,  # border width
        s=150,
        zorder=100,
    )

    _ = ax.set_xlim(0.58, 1.02)
    _ = ax.set_xlabel(f"{constants.METRICS_LABELS['auroc']} (blue) / {constants.METRICS_LABELS['aupro']} (red)")
    _ = ax.set_xticks(np.linspace(0.6, 1.0, 5))
    _ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))

    _ = ax.set_yticks(np.arange(num_models))
    _ = ax.set_yticklabels([constants.MODELS_LABELS_MAINTEXT[name] for name in models_ordered[::-1]])
    _ = ax.set_ylim(-DY - (mrgn := 0.4), (num_models - 1) + DY + mrgn)
    YLIM = ax.get_ylim()
    _ = ax.set_ylabel(None)
    _ = ax.tick_params(axis="y", labelrotation=0, which="major")
    _ = ax.grid(axis="y", linestyle="--", which="major", color="grey", alpha=0.3)

    fig_auroc_aupro
    fig_auroc_aupro.savefig(DATASETWISE_MAINTEXT_DIR / "datasetwise_maintext_auroc_aupro.pdf", bbox_inches="tight")

    # ================================================================================
    # AUPIMO
    fig_aupimo, ax = get_fig_datasetwise_maintext()

    scatter(ax, aupimo_avg_plotdf, "avg_aupimo", color="tab:green")
    scatter(ax, aupimo_p33_plotdf, "p33_aupimo", color="tab:orange")

    _ = ax.scatter(
        aupimo_avg_pmodel_avg["avg_aupimo"].values,
        aupimo_avg_pmodel_avg["y"].values,
        marker="d",
        color="tab:green",
        edgecolor="black",
        lw=2,  # border width
        s=150,
        zorder=100,
    )
    _ = ax.scatter(
        aupimo_p33_pmodel_avg["p33_aupimo"].values,
        aupimo_p33_pmodel_avg["y"].values,
        marker="d",
        color="tab:orange",
        edgecolor="black",
        lw=2,  # border width
        s=150,
        zorder=100,
    )

    _ = ax.set_xlim(-.05, 1.05)
    _ = ax.set_xlabel("AUPIMO: Average (green) / P33 (orange)")
    _ = ax.set_xticks(np.linspace(0, 1.0, 5))
    _ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))

    _ = ax.set_yticks(np.arange(num_models))
    _ = ax.set_yticklabels([constants.MODELS_LABELS_MAINTEXT[name] for name in models_ordered[::-1]])
    _ = ax.set_ylim(*YLIM)
    _ = ax.set_ylabel(None)
    _ = ax.tick_params(axis="y", labelrotation=0, which="major")
    _ = ax.grid(axis="y", linestyle="--", which="major", color="grey", alpha=0.3)

    fig_aupimo
    fig_aupimo.savefig(DATASETWISE_MAINTEXT_DIR / "datasetwise_maintext_aupimo.pdf", bbox_inches="tight")


# %%
