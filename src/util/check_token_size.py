from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
from argparse import Namespace
from datasets import load_dataset
from tqdm import tqdm
from functools import partial

def count_token_size(dataset_path:str, cache_dir_path:str, tokenizer_path:str):

    extension = dataset_path.split(".")[-1]
    if extension == "txt":
        extension = "text"
    elif extension == "jsonl":
        extension = "json"
    elif extension != "csv" and extension != "json" and extension != "text":
        extension = None

    data_files={}
    data_files["train"] = dataset_path

    ds = load_dataset(
        extension,
        data_files=data_files,
        #path=dataset_path,
        # streaming=True, 
        cache_dir=cache_dir_path,
        split="train"#"train[:1%]"
    )    

    # print(ds["train"]["text"][0])

    # load tokenizer
    tokenizer_kwargs = {
        "cache_dir": cache_dir_path,
        #"token": model_args.token,
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    # tokenize dataset & count token size
    text_column_name = "text"
    def tokenize_function(examples):
        processed_texts = [text + tokenizer.eos_token for text in examples[text_column_name]]
        output = tokenizer(processed_texts)
        output["token_size"] = [len(ids) for ids in output["input_ids"]]
        return output

    tokenized_ds = ds.map(
        tokenize_function,
        num_proc=8,
        batched=True,
    )

    token_size = sum(tokenized_ds["token_size"])

    print(">>> count token size")
    print(f"\tdataset_path: {dataset_path}")
    print(f"\tdataset token size: {token_size/10**9:.2f}B params")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--dataset_path",help="",default="./examples/pretrain_mistral_3b/results/dataset/wikipedia/cleaned/wiki.jsonl")
    # parser.add_argument("--dataset_path",help="",default="./externals/corpus-cleaner/results/dataset/cleaned/cc100.jsonl")
    parser.add_argument("--dataset_path",help="",default="../../examples/pretrain/results/dataset/train.jsonl")
    parser.add_argument("--cache_dir_path",help="",default="./.cache")
    parser.add_argument("--tokenizer_path",help="",default="../../examples/pretrain/results/tokenizer/llamatokenizer")

    args = parser.parse_args()
    count_token_size(args.dataset_path, args.cache_dir_path, args.tokenizer_path)


