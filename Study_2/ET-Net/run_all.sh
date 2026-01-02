#!/usr/bin/env bash
set -euo pipefail

# GPU selection (use 4 GPUs: 0,1,2,3)
export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
# Single learning rate and modes; append extra flags after this script call.
LRS=("5e-5")
MODES=("both" "text" "eye")

for LR in "${LRS[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "[run_all] Starting run with lr=${LR} mode=${MODE}"
    EXTRA=""
    if [[ "${MODE}" == "text" ]]; then
      EXTRA="--disable_eye"
    elif [[ "${MODE}" == "eye" ]]; then
      EXTRA="--disable_text"
    fi
    python3 main_text_eye.py --lr "${LR}" ${EXTRA} --device_map_auto "$@"
  done
done
