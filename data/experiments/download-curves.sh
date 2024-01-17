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

EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]
then
    echo "Please provide an experiment name as argument ('ablation' or 'benchmark')"
    exit 1
elif [ "$EXPERIMENT" != "ablation" ] && [ "$EXPERIMENT" != "benchmark" ]
then
    echo "Please provide a valid experiment name as argument ('ablation' or 'benchmark')"
    exit 1
fi

echo "[debug] EXPERIMENT=${EXPERIMENT}"

# ablation is not implemented
if [ "$EXPERIMENT" == "ablation" ]
then
    echo "Ablation experiment is not implemented yet."
    exit 1
fi

ZIP_FNAME="${EXPERIMENT}-curves.zip"
echo "[debug] ZIP_FNAME=${ZIP_FNAME}"

# exit with error if the zip file already exists
if [ -f "$ZIP_FNAME" ]; then
    echo "File ${ZIP_FNAME} already exists. Please rename it and try again."
    exit 1
fi

echo "Downloading ${ZIP_FNAME}..."
wget https://zenodo.org/records/10523657/files/benchmark-curves.zip

ls -hl ${ZIP_FNAME}

echo "Done."