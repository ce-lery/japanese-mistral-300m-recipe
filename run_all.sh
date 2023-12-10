#!/bin/bash

# create python virtual environment
bash setup.sh

# execute pretrain
bash pretrain/pretrain.sh 

# execute fine-tuning
bash fine-tuning/fine-tuning.sh