#!/usr/bin/env bash
set -e  # stops the execution of a script if a command or pipeline has an

function get_this_script_dir() {
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    SCRIPT_DIR=$(realpath $SCRIPT_DIR)
}

get_this_script_dir
echo "[debug] SCRIPT_DIR=${SCRIPT_DIR}"

cd ${SCRIPT_DIR}
echo "[debug] pwd=$(pwd)"

MODELS=( "patchcore_wr50" "patchcore_wr101" "efficientad_wr101_s_ext" "efficientad_wr101_m_ext" "rd++_wr50_ext" "simplenet_wr50_ext" "uflow_ext" )
echo "[debug] MODELS=${MODELS[@]}"

MODELS_ARG="--models"
for MODEL in "${MODELS[@]}"
do
    MODELS_ARG="${MODELS_ARG} ${MODEL}"
done

# call with the models as arguments 
# and forward all other arguments from the command line
python inspect_benchmark.py ${MODELS_ARG} $@
