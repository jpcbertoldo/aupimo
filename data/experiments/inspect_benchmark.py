"""Inspect benchmark data to detect missing or unexpected data.

This script is preferably intended to be run as an ipython notebook.
"""

# ruff: noqa: ERA001

# %%
# Setup

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import torch

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

from aupimo import AUPIMOResult, PIMOResult

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

# %%
# Args

parser = argparse.ArgumentParser()
parser.add_argument("--check-paths", action="store_true")
parser.add_argument("--check-aupimo-thresh-bounds", action="store_true")
parser.add_argument(
    "--check-missing-optional",
    type=str,
    action="append",
    choices=["asmaps.pt", "aupimo/curves.pt", "model"],
    default=[],
)
# "model" (in the choices above) is a bit special; it doesnt have a file extension
# because some models (efficientad) are saved as a directory with multiple files

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                "--check-paths",
                "--check-aupimo-thresh-bounds",
                "--check-missing-optional asmaps.pt",
                "--check-missing-optional aupimo/curves.pt",
                "--check-missing-optional model",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

# %%
# Find run dirs (1 model, 1 dataset) and check if they are complete

rundirs = pd.DataFrame.from_records(
    [
        {
            "model": model_dir.name,
            "collection": collection_dir.name,
            "dataset": dataset_dir.name,
            "path": dataset_dir.relative_to(BENCHMARK_ROOT_DIR),
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

print(f"found {len(rundirs)} run dirs")

models_found = sorted(rundirs.model.unique().tolist())
unexpected_models = sorted(set(models_found) - set(MODELS))

if unexpected_models:
    print(f"unexpected models found: {len(unexpected_models)} (will be ignored)")
    sorted(unexpected_models)
    rundirs = rundirs[rundirs.model.isin(MODELS)]

missing_models = sorted(set(MODELS) - set(models_found))
if missing_models:
    print(f"missing models: {len(missing_models)}")
    sorted(missing_models)

collections_found = rundirs.groupby("model", observed=True).collection.unique().apply(sorted)
unexpected_collections = collections_found.apply(lambda x: sorted(set(x) - set(COLLECTIONS)))
missing_collections = collections_found.apply(lambda x: sorted(set(COLLECTIONS) - set(x)))

if unexpected_collections.any():
    print(f"unexpected collections found: {unexpected_collections.apply(len).sum()} (will be ignored)")
    unexpected_collections[unexpected_collections.astype(bool)]
    rundirs = rundirs[rundirs.collection.isin(COLLECTIONS)]

if missing_collections.any():
    print(f"missing collections: {missing_collections.apply(len).sum()}")
    missing_collections[missing_collections.astype(bool)]

datasets_found = rundirs.groupby(["model", "collection"], observed=True).dataset.unique().apply(sorted)
unexpected_datasets = datasets_found.to_frame().apply(
    lambda row: sorted(set(row.dataset) - set(DATASETS_PER_COLLECTION[row.name[1]])),
    axis=1,
)
missing_datasets = datasets_found.to_frame().apply(
    lambda row: sorted(set(DATASETS_PER_COLLECTION[row.name[1]]) - set(row.dataset)),
    axis=1,
)

if unexpected_datasets.any():
    print(f"unexpected datasets found: {unexpected_datasets.apply(len).sum()} (will be ignored)")
    unexpected_datasets[unexpected_datasets.astype(bool)]
    dataset_is_known = rundirs.apply(lambda row: row.dataset in DATASETS_PER_COLLECTION[row.collection], axis=1)
    rundirs = rundirs[dataset_is_known]

if missing_datasets.any():
    print(f"missing datasets: {missing_datasets.apply(len).sum()}")
    missing_datasets[missing_datasets.astype(bool)]

print(f"{len(rundirs)} run dirs are left after filtering data")

# rundirs = rundirs.reset_index(drop=True).sort_values(["model", "collection", "dataset"])
rundirs = rundirs.reset_index(drop=True).set_index(["model", "collection", "dataset"]).sort_index()


def expected_files_exist(rundir: Path, model: str, _collection: str, _dataset: str) -> dict[str, bool]:
    aupimo_dir = rundir / "aupimo"
    missing = {
        "auroc.json": (rundir / "auroc.json").is_file(),
        "aupr.json": (rundir / "aupr.json").is_file(),
        "aupro.json": (rundir / "aupro.json").is_file(),
        "aupimo/aupimos.json": aupimo_dir.is_dir() and (aupimo_dir / "aupimos.json").is_file(),
        # optionals
        "asmaps.pt": (rundir / "asmaps.pt").is_file(),
        "aupimo/curves.pt": aupimo_dir.is_dir() and (aupimo_dir / "curves.pt").is_file(),
        # model is special because some models (efficientad) are saved as a directory with multiple files
    }

    if model in ("efficientad_wr101_m_ext", "efficientad_wr101_s_ext"):
        model_dir = rundir / "model"
        missing["model"] = (
            model_dir.is_dir()
            and (model_dir / "autoencoder.pt").is_file()
            and (model_dir / "student.pt").is_file()
            and (model_dir / "teacher.pt").is_file()
        )

    elif model in ("fastflow_wr50", "patchcore_wr101", "patchcore_wr50", "simplenet_wr50_ext"):
        missing["model"] = (rundir / "model.pt").is_file()

    return missing


FILES_SCORES = ["auroc.json", "aupr.json", "aupro.json", "aupimo/aupimos.json"]

rundirs_files = (
    (BENCHMARK_ROOT_DIR / rundirs[["path"]])
    .apply(
        lambda row: expected_files_exist(row.path, *row.name),
        axis=1,
    )
    .apply(pd.Series)
)
rundirs_files.columns.name = "file"

# SCORES
rundirs_files_scores = rundirs_files[FILES_SCORES]

missing_score_files = rundirs_files_scores.apply(
    lambda row: sorted(row[row == False].index.tolist()),  # noqa: E712
    axis=1,
).to_frame("files")
missing_score_files["num"] = missing_score_files.map(len)

if missing_score_files["num"].sum() > 0:
    print(f"missing score files: {missing_score_files['num'].sum()}")
    print(missing_score_files.explode("files").files.value_counts())
    missing_score_files[missing_score_files["num"] > 0]

    # kind of transpose the dataframe (who is missing each file)
    who_is_missing_what = missing_score_files.explode("files").reset_index()
    who_is_missing_what["run"] = who_is_missing_what.apply(
        lambda row: f"{row.model}/{row.collection}/{row.dataset}",
        axis=1,
    )
    who_is_missing_what = who_is_missing_what.drop(columns=["model", "collection", "dataset", "num"])
    who_is_missing_what = who_is_missing_what.groupby("files").run.unique().apply(sorted).to_frame("runs")
    who_is_missing_what["num"] = who_is_missing_what["runs"].map(len)
    who_is_missing_what = who_is_missing_what.sort_values("num", ascending=True)
    who_is_missing_what

    print(
        "report of missing score files in 'missing_score_files.html' and "
        "'missing_score_files-who_is_missing_what.html'",
    )
    missing_score_files.to_html(HERE / "missing_score_files.html")
    who_is_missing_what[["runs"]].explode("runs").to_html(
        HERE / "missing_score_files-who_is_missing_what.html",
    )

# OPTIONALS
if len(args.check_missing_optional) > 0:
    print(f"checking optional files: {args.check_missing_optional}")

    rundirs_files_optionals = rundirs_files[args.check_missing_optional]

    missing_optional_files = rundirs_files_optionals.apply(
        lambda row: sorted(row[row == False].index.tolist()),  # noqa: E712
        axis=1,
    ).to_frame("files")
    missing_optional_files["num"] = missing_optional_files.map(len)

    if missing_optional_files["num"].sum() > 0:
        print(f"missing optional files: {missing_optional_files['num'].sum()}")
        print("per file type:")
        print(missing_optional_files.explode("files").files.value_counts())
        missing_optional_files[missing_optional_files["num"] > 0]


# %%
# Check content of files

# index: (model, collection, dataset, file)
files = rundirs_files.stack(sort=True)  # noqa: PD013
files = files[files == True]  # noqa: E712
print(f"found {len(files)} files, now checking their content")

# name[:3] = (model, collection, dataset), name[3] = file
files = files.to_frame().apply(lambda row: rundirs.loc[row.name[:3]] / row.name[3], axis=1)


def check_single_value_json(fpath: Path):
    with fpath.open() as f:
        data = json.load(f)
    return (
        ("missing-key", None)
        if "value" not in data
        else ("wrong-tye", None)
        if not isinstance(data["value"], float)
        else (None, None)
    )


def check_aupimos(fpath: Path, check_paths: bool, check_thresh_bounds: bool):
    try:
        aupimoresult = AUPIMOResult.load(fpath)
    except Exception as ex:
        return (type(ex).__name__, ex)
    if check_paths and aupimoresult.paths is None:
        return ("missing-paths", str(fpath))
    if check_thresh_bounds:
        missing_lower = aupimoresult.thresh_lower_bound is None
        missing_upper = aupimoresult.thresh_upper_bound is None
        if missing_lower and missing_upper:
            return ("missing-thresh-bound", "lower and upper")
        if missing_lower:
            return ("missing-thresh-bound", "lower")
        if missing_upper:
            return ("missing-thresh-bound", "upper")
    return (None, None)


def check_curves(fpath: Path, check_paths: bool):
    try:
        pimoresult = PIMOResult.load(fpath)
    except Exception as ex:
        return (type(ex).__name__, ex)
    if check_paths and pimoresult.paths is None:
        return ("missing-paths", str(fpath))
    return (None, None)


def check_asmaps(fpath: Path, check_paths: bool):
    try:
        data = torch.load(fpath)
    except Exception as ex:
        return (type(ex).__name__, ex)
    return (
        ("missing-key", "asmaps")
        if "asmaps" not in data
        else ("missing-key", "paths")
        if check_paths and "paths" not in data
        else ("wrong-type", "asmaps")
        if not isinstance(data["asmaps"], torch.Tensor)
        else ("wrong-type", "paths")
        if not isinstance(data["paths"], list)
        else ("wrong-type", "paths elements")
        if not all(isinstance(p, str) for p in data["paths"])
        else (None, None)
    )


def dumb_ok(_):
    return (None, None)


# name[3] = file
files_errors = (
    (BENCHMARK_ROOT_DIR / files["path"])
    .to_frame()
    .apply(
        lambda row: {
            "auroc.json": check_single_value_json,
            "aupr.json": check_single_value_json,
            "aupro.json": check_single_value_json,
            "aupimo/aupimos.json": partial(
                check_aupimos,
                check_paths=args.check_paths,
                check_thresh_bounds=args.check_aupimo_thresh_bounds,
            ),
            "asmaps.pt": partial(check_asmaps, check_paths=args.check_paths),
            "aupimo/curves.pt": partial(check_curves, check_paths=args.check_paths),
        }[row.name[3]](row.path),
        axis=1,
        result_type="expand",
    )
    .rename(columns={0: "error", 1: "detail"})
)

files_errors = files_errors[files_errors["error"].notnull()]

# %%
if files_errors.empty:
    print("no errors found")

else:
    print(f"found {len(files_errors)} files with errors")

    files_per_rundir = (
        files_errors.reset_index("file")["file"]
        .groupby(files_errors.index.names[:3], observed=True)
        .apply(sorted)
        .to_frame("files")
    )
    files_per_rundir["num"] = files_per_rundir["files"].map(len)
    files_per_rundir = files_per_rundir[["num", "files"]]

    summary_fpath = HERE / "files_errors_summary.html"
    details_fpath = HERE / "files_errors.html"
    if summary_fpath.exists() or details_fpath.exists():
        print("overwriting existing error files")

    print("report of problematic files per run dir in 'files_errors_summary.html'")
    files_per_rundir.to_html(summary_fpath)

    print("details in 'files_errors.html'")
    files_errors.to_html(details_fpath)

# %%
