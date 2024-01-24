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


from aupimo.utils_numpy import compare_models_pairwise_ttest_rel, compare_models_pairwise_wilcoxon
from aupimo import AUPIMOResult

import constants
from rank_diagram_display import RankDiagramDisplay
from boxplot_display import BoxplotDisplay



# %%
# Args

parser = argparse.ArgumentParser()
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
WHERE_SUPPMAT = "suppmat"
WHERE_MAINTEXT = "maintext"
WHERE_CHOICES = [WHERE_SUPPMAT, WHERE_MAINTEXT]
_ = parser.add_argument("--where", type=str, choices=WHERE_CHOICES, default="suppmat")
_ = parser.add_argument("--model", type=str, default="best")

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--mvtec-root ../data/datasets/MVTec",
                "--visa-root ../data/datasets/VisA",
                "--where maintext",
                # "--model patchcore_wr101"
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

# TODO make this an arg?
BENCHMARK_DIR = Path(__file__).parent / "../../data/experiments/benchmark/"
print(f"{BENCHMARK_DIR.resolve()=}")
assert BENCHMARK_DIR.exists()

# TODO change this datasetwise vocabulary
# TODO make it an arg?
PERMODEL_DIR = "/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src/img/permodel"
PERMODEL_DIR = Path(PERMODEL_DIR)
assert PERMODEL_DIR.exists()

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

match args.where:
    case "suppmat":
        print("using all models")
        MODELS_LABELS = constants.MODELS_LABELS

    case "maintext":
        print("using only main text models")
        data = data.query("model in @constants.MAINTEXT_MODELS")
        MODELS_LABELS = constants.MODELS_LABELS_MAINTEXT

    case _:
        raise ValueError(f"invalid `where` value: {args.where}")

# get the model with highest avg aupimo per dataset
data["avg_aupimo"] = data["aupimo"].apply(lambda x: np.nanmean(x))

if args.model == "best":
    data = data.groupby(["dataset", "category"]).apply(
        lambda df: df.loc[df["avg_aupimo"].idxmax()]
    ).reset_index(drop=True)
    
else:
    data = data.query("model == @args.model").reset_index(drop=True)

data.sort_values(["dataset", "category"], inplace=True)

data = data.set_index(["model", "dataset", "category"])

aupros = data["aupro"].values.astype(float)
aurocs = data["auroc"].values.astype(float)

aupimos = data["aupimo"].map(lambda arr: arr[~np.isnan(arr)])
avg_aupimos = aupimos.apply(np.nanmean).values.astype(float)

# %%
# TODO CSVS

# %%
RCPARAMS = {
    "font.family": "sans-serif",
    "axes.titlesize": "xx-large",
    "axes.labelsize": 'large',
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
}

# %%
# BOXPLOTS

with mpl.rc_context(rc=RCPARAMS):
    
    fig_boxplot, ax = plt.subplots(
        figsize=(5.5, 7.5) if args.where == "maintext" else (7, 4),
        dpi=200, layout="constrained"
    )
    
    BoxplotDisplay.plot_horizontal_functional(
        ax, aupimos,
        [
            f"{constants.DATALABELS_SHORT[dataset]}/{constants.CATEGORIES_LABELS_SHORT[category]}: {MODELS_LABELS[model]}"
            if args.model == "best" else
            f"{constants.DATALABELS_SHORT[dataset]} / {constants.CATEGORIES_LABELS[category]}" 
            for model, dataset, category in aupimos.index
        ],
        medianprops=dict(),
        meanprops=dict(),
        flierprops=dict(markersize=10,),
        widths=0.25,
    )
    
    scat = ax.scatter(
        aupros,
        np.arange(1, len(aupros) + 1),
        marker="|",
        color=(aupro_color := "tab:red"),
        linewidths=3,
        zorder=5,  # between boxplot (10) and grid (-10)
        s=100,
        label="AUPRO",
    )
    
    scat = ax.scatter(
        aurocs,
        np.arange(1, len(aurocs) + 1),
        marker="|",
        color=(auroc_color := "tab:blue"),
        linewidths=3,
        zorder=5,  # between boxplot (10) and grid (-10)
        s=100,
        label="AUROC",
    )
    
    # TODO add aupro 5%
    
    _ = ax.set_xlabel("AUROC (blue) / AUPRO (red) / AUPIMO (boxplot)")
    
    fig_boxplot
    if args.model == "best":
        fig_boxplot_fpath = PERMODEL_DIR / f"best_boxplot.pdf"
    else:
        fig_boxplot_fpath = PERMODEL_DIR / f"{args.model}_boxplot.pdf"
    _ = fig_boxplot.savefig(fig_boxplot_fpath, bbox_inches="tight")

# %%
