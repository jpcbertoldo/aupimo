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

ZIP_FNAME="${EXPERIMENT}-curves.zip"
echo "[debug] ZIP_FNAME=${ZIP_FNAME}"

zip -r ${ZIP_FNAME} ${EXPERIMENT} -i "**/aupimo/curves.pt"
ls -hl ${ZIP_FNAME}

echo "Done."