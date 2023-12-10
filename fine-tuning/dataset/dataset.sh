#!/bin/bash

if [! -d ".env" ];then
    echo "Please run setup/setup.sh first."
    exit 1
fi
source ../.env/bin/activate

cd dataset

# dataset download
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja.git
cd databricks-dolly-15k-ja
git lfs pull --include "databricks-dolly-15k-ja.json"
cd ../

# dataset reformating
python alpaca_preprocess.py

cd ../