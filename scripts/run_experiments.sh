#!/bin/bash

# normal images
DEVICES=("cuda" "cpu")
Anomalous=(39 79 119)
N_COUNT=4

for device in "${DEVICES[@]}"; do
    for n_anomalous in "${Anomalous[@]}"; do
        for ((group=1; group<=N_COUNT; group++)); do
            python scripts/eval.py --asmaps data/experiments/benchmark/padim_wr50/mvtec/screw/asmaps.pt --mvtec-root ~/datasets/MVTec/ --visa-root ~/datasets/visa --anomalous_images "$n_anomalous" --device "$device" --not-debug
        done
    done
done
