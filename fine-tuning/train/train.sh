#!/bin/bash

# activate venv
if [ ! -d "../.env" ];then
    echo "Please run setup/setup.sh first."
    exit 1
fi
source ../.env/bin/activate

# for sentencepice
# if [ ! -e "sentencepiece_model_pb2.py" ];then
#     wget https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece/sentencepiece_model_pb2.py
# fi

cd train

# export CUDA_VISIBLE_DEVICES=0

# fine-tuning
# refer: https://huggingface.co/docs/transformers/main/main_classes/deepspeed#zero3-config
# refer: https://note.com/npaka/n/n26a587be962d
# refer: https://discuss.huggingface.co/t/customized-tokenization-files-in-run-clm-script/21460/3
deepspeed --no_local_rank ../../pretrain/train/run_clm.py hf_config_ft.json --deepspeed --deepspeed_config ../../pretrain/train/ds_config_zero.json 


cd ../


