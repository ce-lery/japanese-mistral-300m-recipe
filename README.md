# japanese-mistral-300m-recipe

## Overview

Welcome to my repository!   
This repository is the recipe for create japanese-mistral-300m.

The feature is ...

- Suppression of unknown word generation by using byte fallback in SentencePiece tokenizer and conversion to huggingface Tokenizers format
- Speed ​​up training with torch.compile (about 2 times)
- Speeding up training with flash attention 2 (about 1.2 times)
- RAM offloading using DeepSpeed ​​ZERO also supports learning with small-scale VRAM
- Use of Mistral 300M

Yukkuri shite ittene!
<!-- 
## Receipe

If you  want to restruct this model , you can refer this Github repository.

I write the receipe for struction this model. For example,

- the mixture ratio of pretraining dataset
- preprocess with sentencepiece
- pretraining with retnet
- about evaluation case

If you find my mistake,error,...etc, please create issue.
If you create pulreqest, I'm very happy! -->

## Quick Started

If you want to try out the contents of this repository quickly and easily, please use [this ipynb file](https://colab.research.google.com/github/ce-lery/japanese-mistral-300m-recipe/blob/main/quick_start.ipynb).


## Getting Started

Build a Python environment using Docker files.

```bash
git clone https://github.com/ce-lery/japanese-mistral-300m-recipe.git
cd japanese-mistral-300m-recipe
docker compose build
docker compose run mistral300m
```

Run the shell script with the following command.  
Execute python virtual environment construction, pretrain, and fine tuning in order.  

```bash
bash run_all.sh
```


## User Guide

The User Guide for this repository is published [here](https://zenn.dev/selllous/articles/transformers_pretrain_to_ft). It is written in Japanese
