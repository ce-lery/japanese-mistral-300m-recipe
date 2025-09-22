#! /bin/bash

# set log
mkdir -p results/log/$(basename "$0" .sh)
log=results/log/$(basename "$0" .sh)/$(date +%Y%m%d_%H%M%S).log
exec &> >(tee -a $log)
set -x

mkdir -p results/tokenizer

# create corpus for torkenizer
uv run python ../../src/tokenizer/extract_text_of_jsonl.py \
    --input_path="./results/dataset/train.jsonl" \
    --output_path="./results/tokenizer/tokenizer_corpus.txt"

# create tokenizer
mkdir -p ./results/tokenizer/spm
mkdir -p ./results/tokenizer/llamatokenizer
uv run python ../../src/tokenizer/unigram.py \
    --model_prefix="./results/tokenizer/spm" \
    --output_dir="./results/tokenizer/llamatokenizer" \
    --corpus_path="./results/tokenizer/tokenizer_corpus.txt"

set +x