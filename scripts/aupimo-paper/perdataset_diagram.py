"""TODO use data from the table in csv"""

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

import pandas as pd
import matplotlib.pyplot as plt


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
from rank_diagram_display import RankDiagramDisplay

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
assert (PERDATASET_SAVEDIR := Path(IMG_SAVEDIR / "perdataset")).exists()
assert (PERDATASET_CSVS_DIR := Path(IMG_SAVEDIR / "perdataset_csvs")).exists()

# %%

# TODO rename
dcidx = constants.DS_CAT_COMBINATIONS.index((args.dataset, args.category))
table = pd.read_csv(PERDATASET_CSVS_DIR / f"perdataset_{dcidx:03}_table.csv", index_col=0)

# %%
# Setup (post data)

# best to worst (lowest to highest rank)
models_ordered = table.columns.tolist()[::-1]
avgrank_aupimo = table[models_ordered].loc["avgrank_aupimo"]
num_models = len(models_ordered)

h1_confidence_matrix = table[models_ordered].iloc[-(num_models-1):, :].values
# concat a nan row for the worst model
h1_confidence_matrix = np.vstack([h1_confidence_matrix, np.full((1, num_models), np.nan)])

# %%
fig_diagram, ax = plt.subplots(figsize=(7 * 1.2, 3 * 1.2), dpi=100, layout="constrained")
MIN_CONFIDENCE = 0.95

RankDiagramDisplay(
    average_ranks=avgrank_aupimo,
    methods_names=[constants.MODELS_LABELS_SHORT[name] for name in models_ordered],
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
fig_diagram
_ = fig_diagram.savefig(PERDATASET_SAVEDIR / f"perdataset_{dcidx:03}_diagram.pdf", bbox_inches="tight")

# %%
