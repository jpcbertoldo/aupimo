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
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
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

# %%
# Load data

records = [
    {
        "dir": str(category_dir),
        "model": model_dir.name,
        "dataset": dataset_dir.name,
        "category": category_dir.name,
        "noisy": False,
    }
    for model_dir in BENCHMARK_DIR.iterdir()
    if model_dir.is_dir() and model_dir.name != "debug"
    for dataset_dir in model_dir.iterdir()
    if dataset_dir.is_dir()
    for category_dir in dataset_dir.iterdir()
    if category_dir.is_dir()
]
records += [
    {
        "dir": str(Path(rec["dir"]) / "synthetic_tiny_regions"),
        "model": rec["model"],
        "dataset": rec["dataset"],
        "category": rec["category"],
        "noisy": True,
    }
    for rec in records
]
print(f"{len(records)=}")

for record in progressbar(records):
    d = Path(record["dir"])
    for m in ["auroc", "aupro","aupro_05",]:
        try:
            record[m] = json.loads((d / f"{m}.json").read_text())['value']
        except FileNotFoundError:
            record[m] = None

    try:
        assert (aupimodir := d / "aupimo").exists()
        aupimoresult = AUPIMOResult.load(aupimodir / "aupimos.json")
        record["aupimo"] = aupimoresult.aupimos.numpy()

    except AssertionError:
        record["aupimo"] = np.array([])

data = pd.DataFrame.from_records(records)
data = data.set_index(["model", "dataset", "category", "noisy"]).sort_index()
# replace aupro_05 with aupro5
data = data.rename(columns={"aupro_05": "aupro5"})
data = data.query("dataset == 'mvtec'").reset_index("dataset", drop=True)
data = data.drop(columns=["dir"])
data["avg_aupimo"] = data["aupimo"].apply(lambda x: np.nanmean(x))
data_aupimo = data["aupimo"].explode().dropna()
data = data.drop(columns=["aupimo"])

print(f"has any model with any NaN? {data.isna().any(axis=1).any()}")
print("df")
data.head(4)
data_aupimo.head(4)

# %%
# DIFF = REF (original) - EXPERIMENT (with synthetic_tiny_regions)
diff = data.loc[(slice(None), slice(None), False)] - data.loc[(slice(None), slice(None), True)]

diff_all_confounded = diff.reset_index(drop=True).T.stack().reset_index(-1, drop=True)
diff_all_confounded.index.name = "metric"
diff_all_confounded.name = "diff"

diff_stats = diff_all_confounded.abs().reset_index("metric").groupby("metric")["diff"].describe()
diff_stats["sem"] = diff_stats.apply(
    lambda row: row["std"] / np.sqrt(row["count"]),
    axis=1,
)
diff_stats["avg_sem_str"] = diff_stats.apply(
    lambda row: (
        f"{row['mean']:.1%} ± {row['sem']:.1%}"
        if row.name in ("aupro", "aupro5",) else
        f"{row['mean']:.2%} ± {row['sem']:.2%}"
        if row.name in ("aupimo",) else
        f"{row['mean']:.2%} ± {row['sem']:.1%}"
    ),
    axis=1,
)

# %%
# per metric histogram

mpl.rcParams.update(RCPARAMS := {
    "font.family": "sans-serif",
    "axes.titlesize": "xx-large",
    "axes.labelsize": 'large',
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
})

fig, ax = plt.subplots(figsize=np.array((6, 2.5))*1.2, layout="constrained", dpi=150)
for metric in diff_all_confounded.index.unique():
    _ = ax.hist(
        diff_all_confounded[metric], bins=10, 
        label=f"{constants.METRICS_LABELS[metric]} ({diff_stats.loc[metric, 'avg_sem_str']})", 
        alpha=.3, density=False, cumulative=False, log=True,
        color=constants.METRICS_COLORS[metric],
    )
    _ = ax.hist(
        diff_all_confounded[metric], bins=10, 
        alpha=1, density=False, cumulative=False, log=True,
        histtype="step", linewidth=2,
        color=constants.METRICS_COLORS[metric],
    )

_ = ax.set_xlabel("Difference in metric value (original - noisy annotated)")
_ = ax.set_ylabel("Frequency (log)")

_ = ax.set_xlim(-.02, .27)
_ = ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))

_ = ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
_ = ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
_ = ax.set_ylim(1, 4e2)

fig.legend(
    loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, ncol=2,
    title="Metric (avg. absolute difference ± SEM)",
    fontsize=11.5,
# reduce space between the legend columns
    columnspacing=0.5,
)


fig.tight_layout()
fig.savefig(IMG_SAVEDIR / "robustness.pdf", bbox_inches="tight", pad_inches=0.01)

# %%
