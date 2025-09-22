#! /bin/bash

# set log
# mkdir -p results/log/$(basename "$0" .sh)
# log=results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
# exec &> >(tee -a $log)
# set -x

# pretrain
# refer: https://note.com/npaka/n/n26a587be962d
# refer: https://discuss.huggingface.co/t/customized-tokenization-files-in-run-clm-script/21460/3
uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "香川県の名物は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "香川県の県庁所在地は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "兵庫県の県庁所在地は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "栃木県の県庁所在地は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "日本の首都は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "アメリカの首都は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "日本で一番高い山は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "日本で二番目に高い山は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "日本で二番目に高い山は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "日本で一番大きな湖は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "世界で一番高い山は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "世界で一番大きな湖は、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "赤信号" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "ジョークとは、" 

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "自然言語処理とは、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "自動車を運転する際、青信号は進む、" \

uv run python ../../src/inference/inference.py \
    --model_path "ce-lery/mistral-300m-base" \
    --device "cuda" \
    --prompt "人工知能" \

# set +x