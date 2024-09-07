#!/bin/bash

#setting log output
#touch log/train.log
#set LOG = "log/train.log"
# exec > >(awk '{print strftime("[%Y/%m/%d %H:%M:%S] "),$0} {fflush()}' | tee -a $LOG) 2>&1

# activate venv
if [ ! -d "../.env" ];then
    echo "Please run setup/setup.sh first."
    exit 1
fi
source ../.env/bin/activate


cd train

# refer: https://huggingface.co/docs/transformers/main/main_classes/deepspeed#zero3-config
# refer: https://note.com/npaka/n/n26a587be962d
# refer: https://discuss.huggingface.co/t/customized-tokenization-files-in-run-clm-script/21460/3
# deepspeed --no_local_rank run_clm.py hf_config_3b.json --deepspeed --deepspeed_config ds_config_zero.json 
# python run_clm.py hf_config_3b.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --standalone --nnodes=1 --nproc-per-node=7 run_clm.py hf_config_39m.json
cd ../


