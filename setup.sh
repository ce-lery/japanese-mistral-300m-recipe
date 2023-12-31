#!/bin/bash

#setting log output
# log="../log/setup.log"
# exec &> >(awk '{print strftime("[%Y/%m/%d %H:%M:%S] "),$0} {fflush()}' | tee -a $log)

# make & activate venv
if [ -d ".env" ];then
    echo "Already exist .env"
    exit 1
fi

# create python virtual environment
mkdir -p log
mkdir -p .env
cd .env
pyenv local 3.11.6
python -m venv ./
cd bin
source activate
./python -m pip install --upgrade pip
cd ../../

# install python library
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# refer:https://github.com/Dao-AILab/flash-attention/issues/453
pip install flash-attn==2.3.4 --no-build-isolation

# # create python virtual environment for creating dataset
deactivate
mkdir -p .env_py310
cd .env_py310
pyenv local 3.10.10
python -m venv ./
cd bin
source activate
./python -m pip install --upgrade pip
cd ../../
pip install wikiextractor==3.0.6 neologdn==0.5.2 tqdm==4.66.1

# install python library from github
mkdir -p module 
cd module
git clone https://github.com/ku-nlp/python-textformatting.git
cd python-textformatting
python setup.py install
cd ../../
