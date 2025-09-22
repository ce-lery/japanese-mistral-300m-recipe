#! /bin/bash

# set log
# mkdir -p results/log/$(basename "$0" .sh)
# log=results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
# exec &> >(tee -a $log)
# set -x

# inference
uv run python ../../src/inference/sft_inference.py \
    --device "cuda" \

# set +x