#! /bin/bash

# set log
mkdir -p results/log/$(basename "$0" .sh)
log=results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
exec &> >(tee -a $log)
set -x

# set parameters
export TOKENIZERS_PARALLELISM=false
#refer: https://zenn.dev/bilzard/scraps/5b00b74984831f
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TODO: for debug. To perform up, please remove here.
# export CUDA_LAUNCH_BLOCKING=1

BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=20
EPOCH=1
DIR_NAME=mistral_300m
mkdir -p ./results/train/

# intialize process counter
SECONDS=0

# ------------------------------------------
#   train
# ------------------------------------------
#rm -r ./results/train/$mistral_300m
uv run python ../../src/train/run_clm.py \
    --model_type "mistral" \
    --config_name ../../src/config/config_mistral_300m.json \
    --tokenizer_name ./results/tokenizer/llamatokenizer \
    --train_file ./results/dataset/train.jsonl \
    --output_dir ./results/train/$DIR_NAME \
    --cache_dir ./results/train/.cache/ \
    --do_train \
    --do_eval \
    --validation_split_percentage 1 \
    --prediction_loss_only \
    --remove_unused_columns False \
    --learning_rate 3.0e-4 \
    --num_train_epochs $EPOCH \
    --logging_dir ./results/train/logs/$DIR_NAME \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_strategy "steps" \
    --eval_steps 10000 \
    --save_total_limit 3 \
    --warmup_steps 1000 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --block_size 1024 \
    --torch_dtype "bfloat16" \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --push_to_hub False \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 12 \
    --optim "adamw_bnb_8bit" \
    --adam_epsilon 1.0e-8 \
    --lr_scheduler_type cosine_with_min_lr \
    --dataloader_pin_memory True \
    --trust_remote_code True \
    --weight_decay 0.1 \
    --adam_beta2 0.95  \
    --min_lr_rate 0.1 \
    #--gradient_checkpointing True \
    #--torch_compile True \
    #--torch_compile_backend "eager" \
    #--torch_compile_backend "inductor" \
    #--min_lr 8.0e-6 \
    #--load_best_model_at_end \

time=$SECONDS
echo "process_time: $time sec"

# deactivate

set +x