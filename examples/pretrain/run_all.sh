#!/bin/bash

# activate venv
if [ ! -d ".env" ]; then bash setup.sh; fi
source .env/bin/activate

cd pretrain

# create dataset for pretrain
bash dataset/dataset.sh

# # train sentencepice tokenizer
bash tokenizer/tokenizer.sh

# # execute pretrain
bash train/train.sh

# # inference with fine-tuned models
bash inference/inference.sh

cd ../