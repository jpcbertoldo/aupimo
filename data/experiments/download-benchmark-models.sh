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

declare -A AVAILABLE_MODELS_MD5=( 
    ["efficientad_wr101_m_ext"]="788d46f8b0d16d0e00de2575d96c29e8" 
    ["efficientad_wr101_s_ext"]="47cb3142b059e7c99208c3899d9a5f1b" 
    ["fastflow_wr50"]="f185b28fee7d81d69d894749360455d2" 
    ["patchcore_wr101"]="b2e95b0fe419f3b3c518ee33dc4e2892" 
    ["patchcore_wr50"]="8f0b6bc77915b0a7ca982bfaacc3545b" 
    ["simplenet_wr50_ext"]="a5439168b00d396c5e1892d735728f93" 
)

# verify the model is available
if [[ ! " ${!AVAILABLE_MODELS_MD5[@]} " =~ " ${MODEL} " ]]; then
    echo "Invalid model name: ${MODEL}"
    echo "Available models: ${!AVAILABLE_MODELS_MD5[@]}"
    exit 1
fi

echo "[debug] MODEL=${MODEL}"

ZIP_FNAME="benchmark-models::${MODEL}.zip"
echo "[debug] ZIP_FNAME=${ZIP_FNAME}"

DOWNLOAD_URL="https://zenodo.org/records/10532861/files/${ZIP_FNAME}"
echo "[debug] DOWNLOAD_URL=${DOWNLOAD_URL}"

echo "Downloading..."
wget ${DOWNLOAD_URL}

ls -hl ${ZIP_FNAME}

echo "Verifying MD5 checksum..."
MD5SUM=$(md5sum ${ZIP_FNAME} | awk '{print $1}')

echo "[debug] expected MD5SUM=${AVAILABLE_MODELS_MD5[$MODEL]}"
echo "[debug] computed MD5SUM=${MD5SUM}"

if [ "$MD5SUM" != "${AVAILABLE_MODELS_MD5[$MODEL]}" ]; then
    echo "MD5 checksum verification failed. Please try again."
    exit 1
else
    echo "MD5 checksum verification passed."
fi

echo "Done."