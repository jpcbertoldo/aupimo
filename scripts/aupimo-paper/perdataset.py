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

import matplotlib as mpl
import numpy as np
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

    print("setting numpy print precision")
    np.set_printoptions(floatmode="maxprec", precision=3, suppress=True)
    print("setting pandas print precision")
    pd.set_option("display.precision", 3)

else:
    IS_NOTEBOOK = False


from aupimo.utils_numpy import compare_models_pairwise_ttest_rel, compare_models_pairwise_wilcoxon
from aupimo import AUPIMOResult

import constants
from rank_diagram_display import RankDiagramDisplay
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

ROOT_SAVEDIR = Path("/home/jcasagrandebertoldo/repos/anomalib-workspace/adhoc/4200-gsoc-paper/latex-project/src/img")
assert (PERDATASET_SAVEDIR := Path(ROOT_SAVEDIR / "perdataset")).exists()
assert (PERDATASET_MAINTEXT_SAVEDIR := Path(ROOT_SAVEDIR / "perdataset_maintext")).exists()
assert (PERDATASET_CSVS_DIR := Path(ROOT_SAVEDIR / "perdataset_csvs")).exists()
assert (PERDATASET_MAINTEXT_CSVS_DIR := Path(ROOT_SAVEDIR / "perdataset_maintext_csvs")).exists()
assert (HEATMAPS_SAVEDIR := ROOT_SAVEDIR / "heatmaps").exists()
assert (PERDATASET_FIGS_SAVEDIR := DATADIR / "perdataset_figs").exists()

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

print("has any model with any NaN?")
data.isna().any(axis=1).any()

assert data.shape[0] == 13  # 13 models

print("df")
data.head(2)

# %%
# Setup (post data)

# TODO rename
dcidx = constants.DS_CAT_COMBINATIONS.index((args.dataset, args.category))

match args.where:
    case "suppmat":
        print("using all models")
        MODELS_LABELS = constants.MODELS_LABELS
        SAVEDIR_FIGS = PERDATASET_SAVEDIR
        SAVEDIR_CSVS = PERDATASET_CSVS_DIR

    case "maintext":
        print("using only main text models")
        data = data.query("model in @constants.MAINTEXT_MODELS")
        MODELS_LABELS = constants.MODELS_LABELS_MAINTEXT
        SAVEDIR_FIGS = PERDATASET_MAINTEXT_SAVEDIR
        SAVEDIR_CSVS = PERDATASET_MAINTEXT_CSVS_DIR

    case _:
        raise ValueError(f"invalid `where` value: {args.where}")

# %%
# AVG AUPIMO
data["avg_aupimo"] = data["aupimo"].apply(lambda x: np.nanmean(x))

# %%
# RANK DATA

# rank and ordering
rank_models_ordered, rank_avgs, confidences = compare_models_pairwise_wilcoxon(
    data["aupimo"].to_dict(), higher_is_better=True, alternative="greater"
)
rank_models_ordered = list(rank_models_ordered)
num_models = len(rank_models_ordered)

h1_confidence_matrix = np.full((num_models, num_models), np.nan)
for i, j in itertools.combinations(range(num_models), 2):
    h1_confidence_matrix[i, j] = confidences[(rank_models_ordered[i], rank_models_ordered[j])]

# %%
# BOXPLOT DATA
df_aupimo = data[["aupimo"]].explode("aupimo").dropna().reset_index()
df_aupimo_modelgb = df_aupimo["aupimo"].groupby(df_aupimo["model"]).apply(list)

# important: order the models in the same way as the nonparametric comparison
df_aupimo_modelgb = df_aupimo_modelgb.loc[rank_models_ordered]  # should be "model", "score"

# replace model names with labels
df_aupimo_modelgb = df_aupimo_modelgb.reset_index("model").replace({"model": MODELS_LABELS}).set_index("model")["aupimo"]

# %%
# PER-METRIC DFS
aupros = data.loc[rank_models_ordered, "aupro"].values.astype(float)
aurocs = data.loc[rank_models_ordered, "auroc"].values.astype(float)
aupimos = data.loc[rank_models_ordered, "aupimo"]
avg_aupimos = aupimos.apply(np.nanmean).values.astype(float)
std_aupimos = aupimos.apply(np.nanstd).values.astype(float)
p33_aupimos = aupimos.apply(lambda x: np.nanpercentile(x, 33)).values.astype(float)
rank_avgs_ndarray = np.asarray(rank_avgs)

# %%
# AUPIMO CSV
aupimo_csv = data[["aupimo"]].copy()
aupimo_csv.columns = ["aupimo"]
aupimo_csv["aupimo"] = aupimo_csv["aupimo"].apply(lambda x: list(enumerate(x)))
aupimo_csv = aupimo_csv.explode("aupimo")
aupimo_csv["imgidx"] = aupimo_csv["aupimo"].apply(lambda x: x[0])
aupimo_csv["aupimo"] = aupimo_csv["aupimo"].apply(lambda x: x[1])
aupimo_csv = aupimo_csv.reset_index()[cols_sorted := ["model", "imgidx", "aupimo"]]
aupimo_csv = aupimo_csv.sort_values(cols_sorted).reset_index(drop=True)
aupimo_csv = aupimo_csv.replace({"model": constants.MODELS_LABELS})
# TODO REVIEW-THEN-SAVE
# aupimo_csv.to_csv(csv_fp := SAVEDIR_CSVS / f"perdataset_{dcidx:03}_aupimo.csv", index=False)
# pd.read_csv(csv_fp)

# %%
# TABLE DATA

# cells
cells = h1_confidence_matrix.copy()
# cells = np.concatenate([np.full((num_models, 1), np.nan), cells,], axis=1)

cells_num = cells.copy()
cells = np.asarray([
    x if np.isnan(x) else f"{xint}%" if (xint := int(np.round(x * 100))) > 1 else "<1%" for x in cells.flatten()
]).reshape(cells.shape)
cells[np.isnan(cells_num)] = ""

# columns and index
table_cols = [constants.MODELS_LABELS_SHORT[name] for name in rank_models_ordered]
table_index = [
    f"{constants.MODELS_LABELS_SHORT[name]} ({rank_avg:.1f})"
    for name, rank_avg in zip(rank_models_ordered, rank_avgs)
]

# remove last model row (it's empty)
cells = cells[:-1]
cells_num = cells_num[:-1]
table_index = table_index[:-1]
table_cols = table_cols

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
    score_values2str(avg_aupimos)[None, :],
    std_values2str(std_aupimos)[None, :],
    score_values2str(p33_aupimos)[None, :],
    rank_values2str(rank_avgs_ndarray)[None, :],
    cells,
], axis=0)
cells_num = np.concatenate([
    aurocs[None, :],
    aupros[None, :],
    avg_aupimos[None, :],
    std_aupimos[None, :],
    p33_aupimos[None, :],
    rank_avgs_ndarray[None, :],
    cells_num,
], axis=0)
table_index = [
    constants.METRICS_LABELS["auroc"],
    constants.METRICS_LABELS["aupro"],
    constants.METRICS_LABELS["avg_aupimo"],
    constants.METRICS_LABELS["std_aupimo"],
    constants.METRICS_LABELS["p33_aupimo"],
    constants.METRICS_LABELS["avgrank_aupimo"],
] + table_index
table_cols = table_cols

# revert columns order
cells = cells[:, ::-1]
cells_num = cells_num[:, ::-1]
table_index = table_index
table_cols = table_cols[::-1]

# %%
# TABLE CSV
table_csv = pd.DataFrame(
    data=cells_num,
    index=table_index,
    columns=table_cols,
)
table_csv.to_csv(csv_fp := SAVEDIR_CSVS / f"perdataset_{dcidx:03}_table.csv", float_format="%.4f")
pd.read_csv(csv_fp, index_col=0)

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
        figsize=(7, 4) if args.where == "maintext" else (7, 4),
        dpi=200, layout="constrained"
    )
    
    BoxplotDisplay.plot_horizontal_functional(
        ax, df_aupimo_modelgb,
        [MODELS_LABELS[name] for name in rank_models_ordered],
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
        aurocs,
        np.arange(1, len(aurocs) + 1),
        marker="|",
        color=(auroc_color := "tab:blue"),
        linewidths=3,
        zorder=5,  # between boxplot (10) and grid (-10)
        s=200,
        label="AUROC",
    )
    
    _ = ax.set_xlabel("AUROC (blue) / AUPRO (red) / AUPIMO (boxplot)")
    
    fig_boxplot
    fig_boxplot_fpath = SAVEDIR_FIGS / f"perdataset_{dcidx:03}_boxplot.pdf"
    _ = fig_boxplot.savefig(fig_boxplot_fpath, bbox_inches="tight")

# %%
# DIAGRAM
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
# NEXt
with mpl.rc_context(rc=RCPARAMS):
    fig_diagram, ax = plt.subplots(figsize=(7 * 1.2, 3 * 1.2), dpi=100, layout="constrained")
    MIN_CONFIDENCE = 0.95

    RankDiagramDisplay(
        average_ranks=rank_avgs,
        methods_names=[constants.MODELS_LABELS_SHORT[name] for name in rank_models_ordered],
        confidence_h1_matrix=h1_confidence_matrix,
        min_confidence=MIN_CONFIDENCE,
    ).plot(
        ax=ax, mode='piramid',
        stem_min_height = 0.7,
        stem_vspace = 0.35,
        stem_kwargs=dict(fontsize='large'),
        bar_kwargs=dict(fontsize='medium'),
        low_confidence_position_low = 0.05,
        low_confidence_position_high = 0.50,
    )
    # ax.set_title("Rank Diagram", pad=20)
    fig_diagram
    _ = fig_diagram.savefig(SAVEDIR_FIGS / f"perdataset_{dcidx:03}_diagram.pdf", bbox_inches="tight")

    # -------------------------------------------------------------
    # TABLE
    fig_tabl, ax = plt.subplots(figsize=(9, 2), dpi=200, layout="constrained")
    table = ax.table(
        cellText=cells,
        colLabels=table_cols,
        rowLabels=table_index,
        loc="center",
        cellLoc="center",
        # colWidths=[0.127] * (num_models + 1),
        bbox=(0, 0, 1, 1),
    )
    # table.scale(0.8, 1)

    # all cells
    # for cell in table.get_celld().values():
    #     cell.set_height(0.06)
    #     # cell.set_width(0.12)

    # paint auroc cells in blue and aupro cells in red
    for colidx in np.arange(-1, len(aurocs)):
        # auroc
        table.get_celld()[(1, colidx)].set_text_props(color=auroc_color, fontweight="bold")
        # aupro
        table.get_celld()[(2, colidx)].set_text_props(color=aupro_color, fontweight="bold")
        # avg/std/p33 aupimo/avgrank
        table.get_celld()[(3, colidx)].set_text_props(fontweight="bold")
        table.get_celld()[(4, colidx)].set_text_props(fontweight="bold")
        table.get_celld()[(5, colidx)].set_text_props(fontweight="bold")
        table.get_celld()[(6, colidx)].set_text_props(fontweight="bold")

    low_confidence_cells = np.asarray(list(zip(*np.where(cells_num < MIN_CONFIDENCE))))
    if len(low_confidence_cells) > 0:
        # mark table cells with low confidence with bold font
        for cell_position in low_confidence_cells + np.array([1, 0]):  # [1, 0] bc there is a shitf of indices
            cell = table.get_celld()[tuple(cell_position)]
            cell.set_text_props(fontweight="bold")

    _ = ax.axis("off")

    fig_tabl
    _ = fig_tabl.savefig(SAVEDIR_FIGS / f"perdataset_{dcidx:03}_table.pdf", bbox_inches="tight")


# %%
# AUX DF (IMGPATHS, MASKSPATHS)

set_ipython_autoreload_2()

from pathlib import Path  # noqa: E402

import common  # noqa: E402

IMGPATHS = data.reset_index()[["dataset", "category"]].drop_duplicates().reset_index(drop=True)
IMGPATHS["dir"] = IMGPATHS.apply(
    lambda row: common.get_dataset_category_testimg_dir(row["dataset"], row["category"]),
    axis=1,
)
IMGPATHS["imgpath"] = IMGPATHS["dir"].apply(Path).apply(
    lambda d: sorted(d.glob("**/*.png")) + sorted(d.glob("**/*.JPG"))
).apply(lambda ps: [(idx, str(p)) for idx, p in enumerate(ps)])
IMGPATHS = IMGPATHS.drop(columns="dir").explode("imgpath").set_index(["dataset", "category"])
IMGPATHS = IMGPATHS.apply(
    lambda row: {"imgidx": row.imgpath[0], "imgpath": row.imgpath[1]}, axis=1, result_type='expand'
).reset_index().set_index(IMGPATHS.index.names + ["imgidx"])["imgpath"]

print("aux df IMGPATHS")
IMGPATHS.head(2)

MASKSPATHS = IMGPATHS.copy().apply(
    lambda p: (
        DATASETSDIR / relpath
        if (relpath := common.imgpath_2_maskpath(p.split("/datasets/")[1])) is not None
        else None
    )
).rename("maskpath")

print("aux df MASKSPATHS")
MASKSPATHS.head(2)

# %%
# DC ARGS / DC BEST MODEL ARGS

import torch  # noqa: E402
from anomalib.utils.metrics.perimg import compare_models_pairwise_wilcoxon  # noqa: E402

# `dc` stands for dataset/category
dc_args = data.reset_index()[cols := ["dataset", "category"]].drop_duplicates().sort_values(cols).reset_index(drop=True)
dc_args.columns.name = None

records = []

# `dc` stands for dataset/category
for dcidx, dcrow in dc_args.iterrows():

    print(f"{dcidx=} {dcrow.dataset=} {dcrow.category=}")
    data = data.loc[dcrow.dataset, dcrow.category]

    # RANK DIAGRAM DATA
    nonparametric = compare_models_pairwise_wilcoxon(data["aupimo"].to_dict(), higher_is_better=True, alternative="greater")

    rank_avgs = nonparametric.reset_index("model1").index.values.tolist()
    rank_models_ordered = nonparametric.reset_index("Average Rank").index.values.tolist()  # models order!!!!
    best_model = rank_models_ordered[0]

    records.append({
        **dcrow.to_dict(),
        "best_model": best_model,
    })

dc_bestmodel_args = pd.DataFrame.from_records(records)
dc_bestmodel_args

# %%
# HEATMAPS and PIMO CURVES

set_ipython_autoreload_2()

from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import matplotlib as mpl  # noqa: E402  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from anomalib.post_processing.post_process import get_contour_from_mask  # noqa: E402
from anomalib.utils.metrics.perimg import perimg_boxplot_stats  # noqa: E402
from anomalib.utils.metrics.perimg.pimo import AUPImOResult, PImOResult  # noqa: E402
from anomalib.utils.metrics.perimg.plot import _plot_perimg_curves  # noqa: E402
from common import DATADIR  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from progressbar import progressbar  # noqa: E402


def get_binary_cmap(colorpositive="red"):
    return mpl.colors.ListedColormap(['#00000000', colorpositive])

STATS_LABELS = dict(
    mean="Mean", whislo="Lower Whisker", q1="Q1", med="Median", q3="Q3", whishi="Upper Whisker",
)

def get_image_label(stat_name, value, imgidx):
    return f"{STATS_LABELS[stat_name]}: {value:.0%} ({imgidx:03})"

INSTANCES_COLORMAP = mpl.colormaps["tab10"]

def plot_boxplot_logpimo_curves(shared_fpr, tprs, image_classes, stats, ax):
    imgidxs = [dic["imgidx"] for dic in stats.values()]

    _plot_perimg_curves(
        ax, shared_fpr, tprs[imgidxs],
        *[
            dict(
                # label=f"{STATS_LABELS[stat]}: {dic['value']:.0%} ({dic['imgidx']:03})",
                label=get_image_label(stat, dic["value"], dic["imgidx"]),
                color=INSTANCES_COLORMAP(curvidx),
                lw=3,
            )
            for curvidx, (stat, dic) in enumerate(stats.items())
        ]
    )
    # x-axis
    _ = ax.set_xlabel("Avg. Nor. Img. FPR (log)")
    _ = ax.set_xscale("log")
    _ = ax.set_xlim(1e-5, 1)
    _ = ax.set_xticks(np.logspace(-5, 0, 6, base=10))
    _ = ax.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
    _ = ax.minorticks_off()
    _ = ax.axvspan(1e-5, 1e-4, color="tab:blue", alpha=0.3, zorder=-5, label="AUC range")
    # y-axis
    _ = ax.set_ylabel("Per-Image TPR")
    _ = ax.set_ylim(0, 1.02)
    _ = ax.set_yticks(np.linspace(0, 1, 11))
    _ = ax.set_yticklabels(["0", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
    _ = ax.grid(True, axis="y", zorder=-10)
    #
    # _ = ax.set_title("PIMO Curves (AUC boxplot statistics)")
    ax.legend(title="[Statistic]: [AUPIMO] ([Image Index])", loc="lower right")


# `dc` stands for dataset/category
for dcidx, dcrow in dc_bestmodel_args.iterrows():

    dataset, category = dcrow[["dataset", "category"]]
    best_model = dcrow["best_model"]

    # DATASET DF
    data = data.loc[dataset, category]

    # MODEL DF
    modeldf = data.loc[dataset, category, best_model]

    aupimos = torch.tensor(modeldf["aupimo"])
    asmaps = torch.tensor(torch.load(modeldf["asmaps_path"]))  # some where save as numpy array

    curves = PImOResult.load(modeldf["aupimo_curves_fpath"])
    aucurves = AUPImOResult.load(modeldf["aupimo_aucs_fpath"])

    imgclass = curves.image_classes

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if best_model == "rd++_wr50_ext":
        argsort = np.argsort((Path(modeldf.dir) / "key.txt").read_text().splitlines())
        asmaps = asmaps[argsort]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # lower/upper fpr bound is upper/lower threshold bound
    upper_threshold = curves.threshold_at(aucurves.lbound).item()
    lower_threshold = curves.threshold_at(aucurves.ubound).item()

    boxplot_stats_dicts = perimg_boxplot_stats(aupimos, imgclass, only_class=1, repeated_policy="avoid")
    stats = {
        stat_dict["statistic"]: {
            "value": stat_dict["nearest"],
            "imgidx": stat_dict["imgidx"],
        }
        for stat_dict in boxplot_stats_dicts
    }
    stats = OrderedDict([
        (k, stats[k])
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # images order in the paper
        for k in ["mean", "whislo", "q1", "med", "q3", "whishi"]
    ])

    # =======================================================================
    # HEATMAPS

    (heatmaps_dc_savedir := HEATMAPS_SAVEDIR / f"{dcidx:03}").mkdir(exist_ok=True, parents=True)

    for vizidx, (stat_name, stat_dict) in enumerate(stats.items()):

            imgidx = stat_dict["imgidx"]
            imgpil = Image.open(imgpath := IMGPATHS.loc[dataset, category, imgidx]).convert("RGB")
            maskpil = Image.open(maskpath := MASKSPATHS.loc[dataset, category, imgidx]).convert("L")
            assert (resolution := imgpil.size[::-1]) == maskpil.size[::-1]
            # [::-1] makes it (height, width)

            asmap = asmaps[imgidx]
            asmap_fullsize = torch.nn.functional.interpolate(
                asmap.unsqueeze(0).unsqueeze(0),
                size=resolution,
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()
            img = np.asarray(imgpil)
            gt = np.array(maskpil).astype(bool)

            fig, ax = plt.subplots(figsize=(sz := 6, sz), dpi=150, layout="constrained")
            _ = ax.imshow(img)
            asmap_viz = asmap_fullsize.copy()
            asmap_viz[asmap_fullsize < lower_threshold] = np.nan
            asmap_viz[asmap_fullsize > upper_threshold] = np.nan
            cmap_pimo_range = mpl.colors.ListedColormap(mpl.colormaps["Blues"](np.linspace(.6, 1, 1024)))
            cmap_pimo_range.set_bad("white", alpha=0)
            _ = ax.imshow(
                asmap_viz,
                cmap=cmap_pimo_range, alpha=0.4, vmin=lower_threshold, vmax=upper_threshold,
            )
            asmap_viz = asmap_fullsize.copy()
            asmap_viz[asmap_fullsize < upper_threshold] = np.nan
            cmap_anomaly_range = mpl.colors.ListedColormap(mpl.colormaps["Reds"](np.linspace(.4, 1, 1024)))
            cmap_anomaly_range.set_bad("white", alpha=0)
            ascores_beyond_ubound = asmap > upper_threshold
            ascores_vmax = (
                torch.quantile(asmap[ascores_beyond_ubound], 0.99)
                if ascores_beyond_ubound.any()
                else upper_threshold + 1e-5
            )
            _ = ax.imshow(
                asmap_viz,
                cmap=cmap_anomaly_range, alpha=.4, vmin=upper_threshold, vmax=ascores_vmax,
            )
            _ = ax.contour(
                get_contour_from_mask(gt, square_size=3, type="outter").astype(float),
                cmap=get_binary_cmap("white"),
            )
            _ = ax.annotate(
                get_image_label(stat_name, stat_dict["value"], stat_dict["imgidx"]),
                xy=((eps := 40), eps), xycoords='data',
                xytext=(0, 0), textcoords='offset fontsize',
                va="top", ha="left",
                fontsize=22,
                bbox=dict(facecolor='white', edgecolor='black')
            )
            _ = ax.axis("off")
            border = np.ones_like(gt).astype(bool)
            border = cv2.dilate(border.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
            _ = ax.contour(
                # get_contour_from_mask(, square_size=5, type="inner").astype(float),
                border.astype(float),
                cmap=get_binary_cmap("black"),
            )
            # add a square to frame the image
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            _ = ax.add_patch(
                mpl.patches.Rectangle((0, 0), *(np.array(gt.shape)[::-1]),
                linewidth=15, edgecolor=INSTANCES_COLORMAP(vizidx), facecolor='none')
            )
            _ = ax.set_xlim(xlim)
            _ = ax.set_ylim(ylim)
            fig
            fig.savefig(heatmaps_dc_savedir / f"{vizidx:03}.jpg", bbox_inches="tight")
            raise Exception("stop")
            break
    break
    continue

    # =======================================================================
    # PIMO CURVES
    with mpl.rc_context(rc=RCPARAMS):
        fig_curves, ax = plt.subplots(figsize=(7, 4), dpi=100, layout="constrained")
        plot_boxplot_logpimo_curves(curves.shared_fpr, curves.tprs, curves.image_classes, stats, ax)
        fig_curves
        fig_curves.savefig(PERDATASET_SAVEDIR / f"perdataset_{dcidx:03}_curves.pdf", bbox_inches="tight")

    break

print('done')


# %%
# TEX FILE

TEX_TEMPLATE = r"""
\begin{figure}[h]
    \centering
    \begin{subfigure}[b]{\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_table.pdf}
      \caption{Statistics and pairwise statistical tests.}
      \label{fig:benchmark-DATASETIDX-table}
    \end{subfigure}
    \\ \vspace{2mm}
    \begin{subfigure}[b]{0.5\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_diagram.pdf}
      \caption{Average rank diagram.}
      \label{fig:benchmark-DATASETIDX-diagram}
    \end{subfigure}
    \\ \vspace{2mm}
    \begin{subfigure}[b]{0.49\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_boxplot.pdf}
      \caption{Score distributions.}
      \label{fig:benchmark-DATASETIDX-boxplot}
    \end{subfigure}
    ~
    \begin{subfigure}[b]{0.49\linewidth}
      \includegraphics[width=\linewidth,valign=t,keepaspectratio]{src/img/perdataset/perdataset_DATASETIDX_curves.pdf}
      \caption{PIMO curves.}
      \label{fig:benchmark-DATASETIDX-pimo-curves}
    \end{subfigure}
    \\  \vspace{2mm}
    \begin{subfigure}[b]{\linewidth}

      \begin{minipage}{\linewidth}
        \centering
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/000.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/001.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/002.jpg}
      \end{minipage}
      \\
      \begin{minipage}{\linewidth}
        \centering
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/003.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/004.jpg}
        \includegraphics[height=32mm,valign=t,keepaspectratio]{src/img/heatmaps/DATASETIDX/005.jpg}
      \end{minipage}
      \caption{
        Heatmaps.
        Images selected according to AUPIMO's statistics.
        Statistic and image index annotated on upper left corner.
      }
      \label{fig:benchmark-DATASETIDX-heatmap}
    \end{subfigure}
    \caption{
      Benchmark on DATASETLABEL.
      PIMO curves and heatmaps are from MODELLABEL.
      NUMIMAGES images (NUMNORMALIMAGES normal, NUMANOMALYIMAGES anomalous).
    }
    \label{fig:benchmark-DATASETIDX}
\end{figure}
"""

# `dc` stands for dataset/category
for dcidx, dcrow in dc_bestmodel_args.iterrows():
    # args
    dataset, category = dcrow[["dataset", "category"]]
    best_model = dcrow["best_model"]
    print(f"{dcidx=} {dataset=} {category=} {best_model=}")
    # =======================================================================
    # data
    modeldf = data.loc[dataset, category, best_model]
    curves = PImOResult.load(modeldf["aupimo_curves_fpath"])
    imgclass = curves.image_classes
    num_images = len(imgclass)
    num_normal_images = (imgclass == 0).sum().item()
    num_anomaly_images = (imgclass == 1).sum().item()
    # break
    # =======================================================================
    # write tex file
    tex = TEX_TEMPLATE.replace("DATASETIDX", f"{dcidx:03}")
    tex = tex.replace("DATASETLABEL", f"{DATASETS_LABELS[dataset]} / {CATEGORIES_LABELS[category]}")
    tex = tex.replace("MODELLABEL", constants.MODELS_LABELS[best_model])
    tex = tex.replace("NUMIMAGES", f"{num_images:03}")
    tex = tex.replace("NUMNORMALIMAGES", f"{num_normal_images:03}")
    tex = tex.replace("NUMANOMALYIMAGES", f"{num_anomaly_images:03}")
    (PERDATASET_FIGS_SAVEDIR / f"{dcidx:03}.tex").write_text(tex)
    break
