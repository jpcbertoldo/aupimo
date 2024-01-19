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

MODEL=$1

if [ -z "$MODEL" ]
then
    echo "Please provide a model name as argument"
    exit 1
fi

# read the directory names in `benchmark`
AVAILABLE_MODELS=$(ls -d benchmark/* | xargs -n 1 basename)
AVAILABLE_MODELS=(${AVAILABLE_MODELS[@]//benchmark-/})

# check if the model name is valid
if [[ ! " ${AVAILABLE_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Invalid model name: ${MODEL}"
    echo "Available models: ${AVAILABLE_MODELS[@]}"
    exit 1
fi

echo "[debug] MODEL=${MODEL}"

ZIP_FNAME="benchmark-models::${MODEL}.zip"
echo "[debug] ZIP_FNAME=${ZIP_FNAME}"

SINGLE_FILE_MODELS=(
    fastflow_wr50
    patchcore_wr101
    patchcore_wr50
    simplenet_wr50_ext
)

MULTI_FILE_MODELS=(
    efficientad_wr101_m_ext
    efficientad_wr101_s_ext
)

if [[ " ${SINGLE_FILE_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Single file model: ${MODEL}"
    # first * is for the collection (mvtec, visa), second * is for the dataset (bottle, cable, capsule, ...)
    zip -r ${ZIP_FNAME} benchmark/${MODEL}/*/*/model.pt

# ========================
elif [[ " ${MULTI_FILE_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Multi file model: ${MODEL}"
    zip -r ${ZIP_FNAME} benchmark/${MODEL}/*/*/model/*.pt

# ========================
else
    ALL_MODELS=(
        ${SINGLE_FILE_MODELS[@]}
        ${MULTI_FILE_MODELS[@]}
    )
    # sort
    ALL_MODELS=($(echo "${ALL_MODELS[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

    echo "Model not implemented yet: ${MODEL}"
    echo "Choices: ${ALL_MODELS[@]}"
    exit 1
fi

ls -h ${ZIP_FNAME}

echo "Done."