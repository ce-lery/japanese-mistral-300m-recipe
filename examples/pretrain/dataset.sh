#! /bin/bash

# Please run the command "cd pretrain/dataset/ ; bash dataset.sh"

# ------------------------------------------
#   Set log configuration
# ------------------------------------------

# set log
mkdir -p ./results/log/$(basename "$0" .sh)
log=./results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
exec &> >(tee -a $log)
set -x

# output start time

echo ">>> start time: $(date '+%Y-%m-%d %I:%M:%S %p')"

mkdir -p results/dataset/
cd results/dataset/

# ------------------------------------------
#   Download training corpus
# ------------------------------------------
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/ce-lery/corpus-ja-11b.git
cd corpus-ja-11b
git lfs pull 
bash merge_train.sh
mv ./train.jsonl ../
cd ../
rm -r corpus-ja-11b

# ------------------------------------------
#   Download tokenizer corpus
# ------------------------------------------
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/ce-lery/wiki.git
cd wiki
git lfs pull
mv ./wiki.jsonl ../
rm -r wiki

echo ">>> end time: $(date '+%Y-%m-%d %I:%M:%S %p')"
