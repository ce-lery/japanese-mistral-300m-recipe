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

# export CUDA_VISIBLE_DEVICES=0

# 24GB GPUでも1.3b以上の学習は難しいため、deepspeedのZeROの使用を検討
# refer: https://huggingface.co/docs/transformers/main/main_classes/deepspeed#zero3-config
# refer: https://note.com/npaka/n/n26a587be962d
# refer: https://discuss.huggingface.co/t/customized-tokenization-files-in-run-clm-script/21460/3
# BS=15; deepspeed transformers/examples/pytorch/language-modeling/run_clm.py \
BS=32; deepspeed ../../pretrain/train/run_clm.py \
    --model_name_or_path "../../pretrain/train/checkpoints-retnet-300M_wiki_constant_spm_neologdn_bytefallback_t5" \
    --train_file ../dataset/databricks-dolly-15k-ja.txt \
    --trust_remote_code True \
    --validation_split_percentage 10\
    --output_dir checkpoints-finetuning\
    --do_train --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --learning_rate 5.0e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --num_train_epochs 40 \
    --logging_dir "logs-finetuning" \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 100 \
    --load_best_model_at_end \
    --save_total_limit 2 \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --block_size 256 \
    --adam_epsilon 1.0e-4 \
    --fp16 \
    --gradient_accumulation_steps 8 \
    --push_to_hub False\
    --dataloader_num_workers  8\
    --optim "adamw_bnb_8bit" \
    --torch_compile \
    --torch_compile_backend "eager" \
    --torch_compile_mode "max-autotune" \
    --deepspeed ../../pretrain/train/ds_config_zero2.json
    # gradient_checkpointing 1
    # --dataloader_num_workers  8\
    #--metric_for_best_model="loss" #This is Default
    #--greater_is_better=False #This is Default


cd ../


