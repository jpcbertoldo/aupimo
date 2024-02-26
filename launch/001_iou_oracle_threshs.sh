#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

# ==========================================================
# GENERIC SETUP

function get_this_script_dir() {
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    SCRIPT_DIR=$(realpath $SCRIPT_DIR)
}

get_this_script_dir
echo "[debug] SCRIPT_DIR=${SCRIPT_DIR}"

function setup_conda() {

    local INIT_CONDA_FPATH=${HOME}/init-conda-bash

    if [ -f ${INIT_CONDA_FPATH} ]; then
        source ${INIT_CONDA_FPATH}

    elif ! command -v conda > /dev/null; then
        echo "[error] conda is not found!"
        exit 1
    fi

    # src: https://stackoverflow.com/a/56155771/9582881
    # i have to do this because otherwise `conda activate` fails
    eval "$(conda shell.bash hook)"
    echo "[debug] conda=$(which conda)"
}

echo "setting up conda"
setup_conda

# ==========================================================
# SPECIFIC SETUP

AUPIMO_SCRIPTS_DIR=$(realpath ${SCRIPT_DIR}/../scripts)
echo "[debug] AUPIMO_SCRIPTS_DIR=${AUPIMO_SCRIPTS_DIR}"

echo "going to scripts dir"
cd ${AUPIMO_SCRIPTS_DIR}
echo "[debug] pwd=$(pwd)"

BENCHMARK_DIR=$(realpath ${SCRIPT_DIR}/../data/experiments/benchmark-segm-paper)
echo "[debug] BENCHMARK_DIR=${BENCHMARK_DIR}"

if [ ! -d ${BENCHMARK_DIR} ]; then
    echo "error: BENCHMARK_DIR=${BENCHMARK_DIR} does not exist"
    exit 1
fi

CONDA_ENV_NAME="aupimo-dev"
echo "[debug] CONDA_ENV_NAME=${CONDA_ENV_NAME}"

conda activate ${CONDA_ENV_NAME}
echo "[debug] python=$(which python)"

# ==========================================================
# MAIN

MODELS=( "patchcore_wr50" "patchcore_wr101" "efficientad_wr101_s_ext" "efficientad_wr101_m_ext" "rd++_wr50_ext" "uflow_ext" )
echo "[debug] MODELS=${MODELS[@]}"

ARG_METRICS="--metrics max_avg_iou --metrics max_iou_per_img --metrics max_avg_iou_min_thresh --metrics max_iou_per_img_min_thresh"
echo "[debug] ARG_METRICS=${ARG_METRICS}"

ARG_DATASETS_ROOTS="--mvtec-root ../data/datasets/MVTec --visa-root ../data/datasets/VisA"
echo "[debug] ARG_DATASETS_ROOTS=${ARG_DATASETS_ROOTS}"

for MODEL in "${MODELS[@]}"
do
    echo "MODEL=${MODEL}"

    MODEL_DIR=${BENCHMARK_DIR}/${MODEL}
    echo "MODEL_DIR=${MODEL_DIR}"

    # find all `asmaps.pt` files in the model directory
    # -L to follow symlinks (IMPORTANT!)
    ASMAPS_PTS=$(find -L ${MODEL_DIR} -name "asmaps.pt")
    ASMAPS_PTS=$(echo ${ASMAPS_PTS} | tr ' ' '\n' | sort)
    echo "ASMAPS_PTS count=$(echo ${ASMAPS_PTS} | wc -w)"  

    for ASMAPS_PT in ${ASMAPS_PTS}
    do
        echo "ASMAPS_PT=${ASMAPS_PT}"

        # call with the models as arguments 
        # and forward all other arguments from the command line
        python eval.py ${ARG_METRICS} ${ARG_DATASETS_ROOTS} --asmaps ${ASMAPS_PT} --not-debug
    done
done