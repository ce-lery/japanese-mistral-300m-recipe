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

# for sentencepice
# if [ ! -e "sentencepiece_model_pb2.py" ];then
#     wget https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece/sentencepiece_model_pb2.py
# fi

cd train

# refer: https://huggingface.co/docs/transformers/main/main_classes/deepspeed#zero3-config
# refer: https://note.com/npaka/n/n26a587be962d
# refer: https://discuss.huggingface.co/t/customized-tokenization-files-in-run-clm-script/21460/3
deepspeed --no_local_rank run_clm.py hf_config.json --deepspeed --deepspeed_config ds_config_zero.json 

cd ../


