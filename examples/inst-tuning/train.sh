#! /bin/bash

# set log
mkdir -p results/log/$(basename "$0" .sh)
log=results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
exec &> >(tee -a $log)
set -x

# set parameters
export TOKENIZERS_PARALLELISM=false
#refer: https://zenn.dev/bilzard/scraps/5b00b74984831f
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ./results/

# intialize process counter
SECONDS=0

# ------------------------------------------
#   train
# ------------------------------------------
#rm -r ./results/pretrain/$DIR_NAME/mistral2b_trial2
uv run python ../../src/train/sft_train.py \
    --model_name_or_path  ../pretrain/results/train/mistral_300m/ \
    --output_dir ./results/train1/ \

time=$SECOND
echo "process_time: $time sec"

set +x
