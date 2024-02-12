"""Inspect benchmark data to detect missing or unexpected data.

This script is preferably intended to be run as an ipython notebook.
"""

# ruff: noqa: ERA001

# %%
# Setup

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

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

else:
    IS_NOTEBOOK = False


# %%
# Constants

HERE = Path(__file__).parent
BENCHMARK_ROOT_DIR = HERE / "benchmark"

MODELS = [
    "padim_r18",
    "padim_wr50",
    "patchcore_wr50",
    "patchcore_wr101",
    "fastflow_wr50",
    "fastflow_cait_m48_448",
    "efficientad_wr101_s_ext",
    "efficientad_wr101_m_ext",
    "simplenet_wr50_ext",
    "pyramidflow_fnf_ext",
    "pyramidflow_r18_ext",
    "uflow_ext",
    "rd++_wr50_ext",
]

COLLECTION_MVTEC = "mvtec"
COLLECTION_VISA = "visa"
COLLECTIONS = [COLLECTION_MVTEC, COLLECTION_VISA]

MVTEC_DATASETS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_DATASETS = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

DATASETS_PER_COLLECTION = {
    COLLECTION_MVTEC: MVTEC_DATASETS,
    COLLECTION_VISA: VISA_DATASETS,
}

-rw-r--r--  1 jcasagrandebertoldo CMM 6,4M févr. 11 15:03
-rw-r--r--  1 jcasagrandebertoldo CMM 9,6M févr. 11 15:03
-rw-r--r--  1 jcasagrandebertoldo CMM 7,0K févr. 12 15:34
-rw-r--r--  1 jcasagrandebertoldo CMM 7,0K févr. 12 15:34
-rw-r--r--  1 jcasagrandebertoldo CMM 8,9K févr. 12 15:34
-rw-r--r--  1 jcasagrandebertoldo CMM 8,9K févr. 12 15:34

# %%
# Find run dirs (1 model, 1 dataset) and check if they are complete

rundirs = pd.DataFrame.from_records(
    [
        {
            "model": model_dir.name,
            "collection": collection_dir.name,
            "dataset": dataset_dir.name,
            "abspath": dataset_dir,
            "relpath": dataset_dir.relative_to(BENCHMARK_ROOT_DIR),
        }
        for model_dir in BENCHMARK_ROOT_DIR.iterdir()
        if model_dir.is_dir()
        for collection_dir in model_dir.iterdir()
        if collection_dir.is_dir()
        for dataset_dir in collection_dir.iterdir()
        if dataset_dir.is_dir()
    ],
).sort_values(["model", "collection", "dataset"])
rundirs["model"] = rundirs.model.astype("category")
rundirs["collection"] = rundirs.collection.astype("category")
rundirs["dataset"] = rundirs.dataset.astype("category")

rundirs["has_iou_curves"] = rundirs.model.isin((
    "patchcore_wr50",
    "patchcore_wr101",
    "efficientad_wr101_s_ext",
    "efficientad_wr101_m_ext",
    "rd++_wr50_ext",
    "simplenet_wr50_ext",
    "uflow_ext",
))
rundirs = rundirs.query("has_iou_curves")

for idx, row in rundirs.iterrows():
    print(f"{row.relpath}")
    (iou_oracle_threshs_dir := row.abspath / "iou_oracle_threshs").mkdir(exist_ok=True, parents=True)
    files_names = [
        "ioucurves_global_threshs.pt",
        "ioucurves_local_threshs.pt",
        "max_avg_iou.json",
        "max_avg_iou_min_thresh.json",
        "max_iou_per_img.json",
        "max_iou_per_img_min_thresh.json",
    ]
    for fname in files_names:
        src = row.abspath / fname
        dst = iou_oracle_threshs_dir / fname
        if not src.exists():
            continue
        # move from src to dst
        src.rename(dst)

# %%

