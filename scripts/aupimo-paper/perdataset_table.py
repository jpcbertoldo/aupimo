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
_ = parser.add_argument("--dataset", type=str)
_ = parser.add_argument("--category", type=str)
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
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
assert (FIG_SAVEDIR := ROOT_SAVEDIR / "fig").exists()

assert (PERDATASET_SAVEDIR := Path(IMG_SAVEDIR / "perdataset")).exists()
assert (PERDATASET_CSVS_DIR := Path(IMG_SAVEDIR / "perdataset_csvs")).exists()
assert (PERDATASET_MAINTEXT_SAVEDIR := Path(IMG_SAVEDIR / "perdataset_maintext")).exists()
assert (PERDATASET_MAINTEXT_CSVS_DIR := Path(IMG_SAVEDIR / "perdataset_maintext_csvs")).exists()
assert (HEATMAPS_SAVEDIR := IMG_SAVEDIR / "heatmaps").exists()
# TODO refactor move to fig folder
assert (PERDATASET_FIGS_SAVEDIR := IMG_SAVEDIR / "perdataset_figs").exists()

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
    # !!!!!!!!!!!!!!!
    and dataset_dir.name == args.dataset
    for category_dir in dataset_dir.iterdir()
    if category_dir.is_dir()
    # !!!!!!!!!!!!!!!
    and category_dir.name == args.category
]
print(f"{len(records)=}")

for record in progressbar(records):

    d = Path(record["dir"])

    for m in [
        "auroc", "aupro",
        # "aupr",
        "aupro_05",
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

print(f"has any model with any NaN? {data.isna().any(axis=1).any()}")
assert data.shape[0] == 13  # 13 models

print("df")
data.head(2)

# %%
# Setup (post data)

# TODO rename
dcidx = constants.DS_CAT_COMBINATIONS.index((args.dataset, args.category))

# %%
# AVG AUPIMO
data["avg_aupimo"] = data["aupimo"].apply(lambda x: np.nanmean(x))

# %%
# RANK DATA

# rank and ordering
models_ordered, rank_avgs, confidences = compare_models_pairwise_wilcoxon(
    data["aupimo"].to_dict(), higher_is_better=True, alternative="greater"
)
models_ordered = list(models_ordered)
num_models = len(models_ordered)

h1_confidence_matrix = np.full((num_models, num_models), np.nan)
for i, j in itertools.combinations(range(num_models), 2):
    h1_confidence_matrix[i, j] = confidences[(models_ordered[i], models_ordered[j])]

# %%
# PER-METRIC DFS
aupros = data.loc[models_ordered, "aupro"].values.astype(float)
aurocs = data.loc[models_ordered, "auroc"].values.astype(float)
aupro05s = data.loc[models_ordered, "aupro_05"].values.astype(float)
aupimos = data.loc[models_ordered, "aupimo"]
avg_aupimos = aupimos.apply(np.nanmean).values.astype(float)
std_aupimos = aupimos.apply(np.nanstd).values.astype(float)
p33_aupimos = aupimos.apply(lambda x: np.nanpercentile(x, 33)).values.astype(float)
rank_avgs_ndarray = np.asarray(rank_avgs)

# %%
mpl.rcParams.update(RCPARAMS := {
    "font.family": "sans-serif",
    "axes.titlesize": "xx-large",
    "axes.labelsize": 'large',
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
})

# %%
# TABLE DATA

# cells
cells = h1_confidence_matrix.copy()

cells_num = cells.copy()
cells = np.asarray([
    x if np.isnan(x) else f"{xint}%" if (xint := int(np.round(x * 100))) > 1 else "<1%" for x in cells.flatten()
]).reshape(cells.shape)
cells[np.isnan(cells_num)] = ""

# columns and index
table_cols_csv = models_ordered
table_cols = [constants.MODELS_LABELS_SHORT[name] for name in models_ordered]

table_index_csv = models_ordered
table_index = [
    f"{constants.MODELS_LABELS_SHORT[name]} ({rank_avg:.1f})"
    for name, rank_avg in zip(models_ordered, rank_avgs)
]

# remove last model row (it's empty)
cells = cells[:-1]
cells_num = cells_num[:-1]
table_index_csv = table_index_csv[:-1]
table_index = table_index[:-1]

def score_values2str(values, higher_is_better=True):
    ranks = np.argsort(np.argsort(values * (-1 if higher_is_better else 1)))
    return np.array([
        "-" if np.isnan(x) else f"{x:.1%} ({sortidx + 1})"
        for x, sortidx in zip(values, ranks)
    ])

def std_values2str(values):
    return np.array(["-" if np.isnan(x) else f"{x:.1%}"  for x in values])

def rank_values2str(values):
    return np.array(["-" if np.isnan(x) else f"{x:.1f}"  for x in values])

# add extra rows
cells = np.concatenate([
    score_values2str(aurocs)[None, :],
    score_values2str(aupros)[None, :],
    score_values2str(aupro05s)[None, :],
    score_values2str(avg_aupimos)[None, :],
    std_values2str(std_aupimos)[None, :],
    score_values2str(p33_aupimos)[None, :],
    rank_values2str(rank_avgs_ndarray)[None, :],
    cells,
], axis=0)
cells_num = np.concatenate([
    aurocs[None, :],
    aupros[None, :],
    aupro05s[None, :],
    avg_aupimos[None, :],
    std_aupimos[None, :],
    p33_aupimos[None, :],
    rank_avgs_ndarray[None, :],
    cells_num,
], axis=0)
table_index_metrics_csv = [
    "auroc",
    "aupro",
    "aupro_05",
    "avg_aupimo",
    "std_aupimo",
    "p33_aupimo",
    "avgrank_aupimo",
]
table_index_csv = table_index_metrics_csv + table_index_csv
table_index_metrics = [constants.METRICS_LABELS[metric] for metric in table_index_metrics_csv] 
table_index = table_index_metrics + table_index

# revert columns order
cells = cells[:, ::-1]
cells_num = cells_num[:, ::-1]
table_cols_csv = table_cols_csv[::-1]
table_cols = table_cols[::-1]

# %%
# TABLE CSV
table_csv = pd.DataFrame(
    data=cells_num,
    index=table_index_csv,
    columns=table_cols_csv,
)
table_csv.to_csv(csv_fp := PERDATASET_CSVS_DIR / f"perdataset_{dcidx:03}_table.csv", float_format="%.4f")
pd.read_csv(csv_fp, index_col=0)

# %%
# TABLE
MIN_CONFIDENCE = .95
fig_tabl, ax = plt.subplots(figsize=(9, 2), dpi=200, layout="constrained")
table = ax.table(
    cellText=cells,
    colLabels=table_cols,
    rowLabels=table_index,
    loc="center",
    cellLoc="center",
    bbox=(0, 0, 1, 1),
)

# paint auroc cells in blue and aupro cells in red
for colidx in np.arange(-1, len(aurocs)):
    # auroc
    table.get_celld()[(1, colidx)].set_text_props(color=constants.METRICS_COLORS["auroc"], fontweight="bold")
    # aupro and aupro5
    table.get_celld()[(2, colidx)].set_text_props(color=constants.METRICS_COLORS["aupro"], fontweight="bold")
    table.get_celld()[(3, colidx)].set_text_props(color=constants.METRICS_COLORS["aupro5"], fontweight="bold")
    # avg/std/p33 aupimo/avgrank
    table.get_celld()[(4, colidx)].set_text_props(fontweight="bold")
    table.get_celld()[(5, colidx)].set_text_props(fontweight="bold")
    table.get_celld()[(6, colidx)].set_text_props(fontweight="bold")
    table.get_celld()[(7, colidx)].set_text_props(fontweight="bold")

low_confidence_cells = np.asarray(list(zip(*np.where(cells_num < MIN_CONFIDENCE))))
if len(low_confidence_cells) > 0:
    # mark table cells with low confidence with bold font
    for cell_position in low_confidence_cells + np.array([1, 0]):  # [1, 0] bc there is a shitf of indices
        cell = table.get_celld()[tuple(cell_position)]
        cell.set_text_props(fontweight="bold")

_ = ax.axis("off")

fig_tabl
_ = fig_tabl.savefig(PERDATASET_SAVEDIR / f"perdataset_{dcidx:03}_table.pdf", bbox_inches="tight")

# %%
