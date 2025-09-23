#!/bin/bash

# execute pretrain
cd examples/pretrain
bash run_all.sh 
cd ../../

# execute fine-tuning
cd examples/inst-tuning
bash run_all.sh 
cd ../../
