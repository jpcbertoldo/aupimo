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

ZIP_FNAME="benchmark-segm-paper-superpixel-oracle.zip"
echo "[debug] ZIP_FNAME=${ZIP_FNAME}"

zip -q -r ${ZIP_FNAME} benchmark-segm-paper -i "**/optimal_iou.json"
ls -hl ${ZIP_FNAME}

echo "Done."