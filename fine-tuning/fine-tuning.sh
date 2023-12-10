#!/bin/bash

# activate venv
if [ ! -d ".env" ];then
    echo "Please execute "bash setup.sh" first."
    exit 1
fi
source .env/bin/activate

cd fine-tuning

# create dataset for fine-tuning
bash dataset/dataset.sh

# execute fine-tuning
bash train/train.sh

# inference with fine-tuned models
cd inference
python inference.py
cd ../

cd ../