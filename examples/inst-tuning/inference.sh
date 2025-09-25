#! /bin/bash

# set log
# mkdir -p results/log/$(basename "$0" .sh)
# log=results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
# exec &> >(tee -a $log)
# set -x

# inference
uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "日本で一番高い山は何？" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "日本で一番大きな湖の名前は？" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "世界で一番高い山は何？" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "世界で一番大きな湖の名前は？" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "冗談を言ってください。" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "香川県の名物は何？" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "日本の首都はどこ？" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "こんにちは！" \

uv run python ../../src/inference/sft_inference.py \
    --model_path "./results/train/mistral_300m_sft/checkpoint-1323" \
    --device "cuda" \
    --prompt "兵庫県の県庁所在地の名前は？" \

# set +x