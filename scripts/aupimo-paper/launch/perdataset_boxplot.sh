#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

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

function get_this_script_dir() {
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    SCRIPT_DIR=$(realpath $SCRIPT_DIR)
}

get_this_script_dir
echo "[debug] SCRIPT_DIR=${SCRIPT_DIR}"

cd ${SCRIPT_DIR}/..
echo "[debug] pwd=$(pwd)"

CONDA_ENV_NAME="aupimo-dev"
echo "[debug] CONDA_ENV_NAME=${CONDA_ENV_NAME}"

conda activate ${CONDA_ENV_NAME}
echo "[debug] python=$(which python)"

DATASETS_CATEGORIES=("mvtec/bottle" "mvtec/cable" "mvtec/capsule" "mvtec/carpet" "mvtec/grid" "mvtec/hazelnut" "mvtec/leather" "mvtec/metal_nut" "mvtec/pill" "mvtec/screw" "mvtec/tile" "mvtec/toothbrush" "mvtec/transistor" "mvtec/wood" "mvtec/zipper" "visa/candle" "visa/capsules" "visa/cashew" "visa/chewinggum" "visa/fryum" "visa/macaroni1" "visa/macaroni2" "visa/pcb1" "visa/pcb2" "visa/pcb3" "visa/pcb4" "visa/pipe_fryum")

echo "launching suppmat"
for DATASET_CATEGORY in ${DATASETS_CATEGORIES[@]}; do
    echo "[debug] DATASET_CATEGORY=${DATASET_CATEGORY}"

    ARGS=""
    ARGS="${ARGS} --mvtec-root ../data/datasets/MVTec"
    ARGS="${ARGS} --visa-root ../data/datasets/VisA"

    DATASET=$(echo ${DATASET_CATEGORY} | cut -d'/' -f 1)
    ARGS="${ARGS} --dataset ${DATASET}"

    CATEGORY=$(echo ${DATASET_CATEGORY} | cut -d'/' -f 2)
    ARGS="${ARGS} --category ${CATEGORY}"

    echo "[debug] ARGS=${ARGS}"

    python perdataset_boxplot.py ${ARGS} --where suppmat
    python perdataset_boxplot.py ${ARGS} --where maintext
    
done
