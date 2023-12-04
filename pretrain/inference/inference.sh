#!/bin/bash

#activate venv
if [ ! -d "../.env" ];then
    echo "Please run setup/setup.sh first."
    exit 1
fi
source ../.env/bin/activate

export CUDA_VISIBLE_DEVICES=0

cd inference
python inference.py
cd ../
