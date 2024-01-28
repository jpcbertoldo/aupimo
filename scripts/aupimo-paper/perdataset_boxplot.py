#!/usr/bin/env python
# coding: utf-8

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from progressbar import progressbar
import json
import numpy as np
import pandas as pd
import matplotlib as mpl

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
    np.set_printoptions(floatmode="maxprec", precision=3, suppress=True)
    print("setting pandas print precision")
    pd.set_option("display.precision", 3)

else:
    IS_NOTEBOOK = False


from aupimo import AUPIMOResult
import constants
from boxplot_display import BoxplotDisplay


# %%
# Args

parser = argparse.ArgumentParser()
_ = parser.add_argument("--dataset", type=str)
_ = parser.add_argument("--category", type=str)
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
WHERE_SUPPMAT = "suppmat"
WHERE_MAINTEXT = "maintext"
WHERE_CHOICES = [WHERE_SUPPMAT, WHERE_MAINTEXT]
_ = parser.add_argument("--where", type=str, choices=WHERE_CHOICES, default="suppmat")

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--dataset mvtec",
                "--category zipper",
                "--mvtec-root ../data/datasets/MVTec",
                "--visa-root ../data/datasets/VisA",
                "--where maintext"
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
# TODO refactor make it an arg
ROOT_SAVEDIR = Path("/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src")
assert ROOT_SAVEDIR.exists()
assert (IMG_SAVEDIR := ROOT_SAVEDIR / "img").exists()
assert (PERDATASET_CSVS_DIR := Path(IMG_SAVEDIR / "perdataset_csvs")).exists()

if args.where == WHERE_SUPPMAT:
    assert (PERDATASET_SAVEDIR := Path(IMG_SAVEDIR / "perdataset")).exists()
    
elif args.where == WHERE_MAINTEXT:
    assert (PERDATASET_SAVEDIR := Path(IMG_SAVEDIR / "perdataset_maintext")).exists()

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
    # !!!!!!!!!!!!!
    and dataset_dir.name == args.dataset
    for category_dir in dataset_dir.iterdir()
    if category_dir.is_dir()
    # !!!!!!!!!!!!!
    and category_dir.name == args.category
]
print(f"{len(records)=}")

for record in progressbar(records):

    d = Path(record["dir"])

    for m in [
        "auroc", "aupro",
        # "aupr",
        "aupro_05",
        "ious",
    ]:
        try:
            record[m] = json.loads((d / f"{m}.json").read_text())['value']
        except FileNotFoundError:
            record[m] = None

    try:
        assert (aupimodir := d / "aupimo").exists()
        aupimoresult = AUPIMOResult.load(aupimodir / "aupimos.json")
        record["aupimo"] = aupimoresult.aupimos.numpy()

    except AssertionError:
        record["aupimo"] = None


data = pd.DataFrame.from_records(records)
data = data.set_index(["dataset", "category"])
data = data.loc[(args.dataset, args.category)].reset_index(drop=True)
data = data.set_index(["model"]).sort_index()

# rename ious to iou
data = data.rename(columns={"ious": "iou"})
data["iou"] = data["iou"].apply(lambda x: np.asarray(x))

print(f"has any model with any NaN? {data.isna().any(axis=1).any()}")
assert data.shape[0] == 13  # 13 models
print("df")
data.head(2)

# %%
# maintext or suppmat
dcidx = constants.DS_CAT_COMBINATIONS.index((args.dataset, args.category))
table = pd.read_csv(PERDATASET_CSVS_DIR / f"perdataset_{dcidx:03}_table.csv", index_col=0)

# %%
# where

if args.where == WHERE_SUPPMAT:
    print("using all models")
    
elif args.where == WHERE_MAINTEXT:
    print("using only main text models")
    data = data.query("model in @constants.MAINTEXT_MODELS")
    
    # table
    row_is_metric = table.index.isin(constants.METRICS_LABELS)
    row_is_model_maintext = table.index.isin(constants.MAINTEXT_MODELS)
    keep_row = row_is_metric | row_is_model_maintext
    keep_col = table.columns.isin(constants.MAINTEXT_MODELS)
    table = table.loc[keep_row, keep_col]

# %%
# AVG AUPIMO AND IOU

is_anomalous_mask = ~np.isnan(data["aupimo"].iloc[0])

data["avg_aupimo"] = data["aupimo"].apply(
    lambda x: np.mean(x[is_anomalous_mask])
)

data["avg_iou"] = data["iou"].apply(
    lambda x: np.mean(x[is_anomalous_mask])
)

# best to worst (lowest to highest rank)
models_ordered = table.columns.tolist()[::-1]
avgrank_aupimo = table[models_ordered].loc["avgrank_aupimo"]

df_aupimo = data[["aupimo"]].explode("aupimo").dropna().reset_index()
df_aupimo_modelgb = df_aupimo["aupimo"].groupby(df_aupimo["model"]).apply(list)

# important: order the models in the same way as the nonparametric comparison
df_aupimo_modelgb = df_aupimo_modelgb.loc[models_ordered]  # should be "model", "score"

# replace model names with labels

models_labels_mapping = constants.MODELS_LABELS_MAINTEXT if args.where == WHERE_MAINTEXT else constants.MODELS_LABELS
models_labels_ordered = [models_labels_mapping[name] for name in models_ordered]
df_aupimo_modelgb = df_aupimo_modelgb.reset_index("model").replace({"model": models_labels_mapping}).set_index("model")["aupimo"]

# %%
# PER-METRIC DFS
aupros = data.loc[models_ordered, "aupro"].values.astype(float)
aurocs = data.loc[models_ordered, "auroc"].values.astype(float)
aupro05s = data.loc[models_ordered, "aupro_05"].values.astype(float)
aupimos = data.loc[models_ordered, "aupimo"]
avg_aupimos = aupimos.apply(np.nanmean).values.astype(float)
std_aupimos = aupimos.apply(np.nanstd).values.astype(float)
p33_aupimos = aupimos.apply(lambda x: np.nanpercentile(x, 33)).values.astype(float)
rank_avgs_ndarray = np.asarray(avgrank_aupimo)
avg_iou = data.loc[models_ordered, "avg_iou"].values.astype(float)

# %%
# AUPIMO CSV

if args.where == WHERE_SUPPMAT:
    aupimo_csv = data[["aupimo"]].copy()
    aupimo_csv.columns = ["aupimo"]
    aupimo_csv["aupimo"] = aupimo_csv["aupimo"].apply(lambda x: list(enumerate(x)))
    aupimo_csv = aupimo_csv.explode("aupimo")
    aupimo_csv["imgidx"] = aupimo_csv["aupimo"].apply(lambda x: x[0])
    aupimo_csv["aupimo"] = aupimo_csv["aupimo"].apply(lambda x: x[1])
    aupimo_csv = aupimo_csv.reset_index()[cols_sorted := ["model", "imgidx", "aupimo"]]
    aupimo_csv = aupimo_csv.sort_values(cols_sorted).reset_index(drop=True)
    aupimo_csv.to_csv(csv_fp := PERDATASET_CSVS_DIR / f"perdataset_{dcidx:03}_aupimo.csv", index=False, float_format="%.4f")
    pd.read_csv(csv_fp)

# %%
# BOXPLOTS

mpl.rcParams.update(RCPARAMS := {
    "font.family": "sans-serif",
    "axes.titlesize": "xx-large",
    "axes.labelsize": 'large',
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
})

figsize = np.array((7, 4)) * 1.15 if args.where == WHERE_SUPPMAT else np.array((7, 3)) * 1.15
fig_boxplot, ax = plt.subplots(figsize=figsize, dpi=200, layout="constrained")

BoxplotDisplay.plot_horizontal_functional(
    ax, df_aupimo_modelgb, models_labels_ordered,
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
    s=200,
    label="AUPRO",
)

scat = ax.scatter(
    aupro05s,
    np.arange(1, len(aupro05s) + 1),
    marker="|",
    color=(aupro_color5 := "tab:purple"),
    linewidths=3,
    zorder=5,  # between boxplot (10) and grid (-10)
    s=200,
    label="AUPRO05",
)

scat = ax.scatter(
    aurocs,
    np.arange(1, len(aurocs) + 1),
    marker="|",
    color=(auroc_color := "tab:blue"),  # TODO get from constants
    linewidths=3,
    zorder=5,  # between boxplot (10) and grid (-10)
    s=200,
    label="AUROC",
)


scat = ax.scatter(
    avg_iou,
    np.arange(1, len(avg_iou) + 1),
    marker="|",
    color=constants.METRICS_COLORS["avg_iou"],
    linewidths=3,
    zorder=5,  # between boxplot (10) and grid (-10)
    s=200,
    label="IoU",
)


_ = ax.set_xlabel("AUROC (blue) / AUPRO (30% red, 5% purple) / AUPIMO (boxplot) / IoU (orange)")

fig_boxplot.savefig(
    PERDATASET_SAVEDIR / f"perdataset_{dcidx:03}_boxplot.pdf", 
    bbox_inches="tight",
    pad_inches=0,
)

# %%
