#!/bin/bash

# create dataset for pretrain
bash dataset.sh

# train sentencepice tokenizer
bash tokenizer.sh

# execute pretrain
bash train.sh

# inference with fine-tuned models
bash inference.sh

cd ../