import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))

def replate_newline(text:str) -> str:
    return text.replace("\n","\\n")

def change_dataset_format():
    # アルパカ学習データの書式に合わせる
    # https://note.com/npaka/n/n91e6dfecd034

    # json file open
    json_open = open(current_path+"/databricks-dolly-15k-ja/databricks-dolly-15k-ja.json",'r', encoding="utf-8_sig")
    json_load = json.load(json_open)

    print(json_load[0])
    # 書き込み対象のdatabricks-dolly-15k-ja.txt open
    f = open(current_path+'/databricks-dolly-15k-ja.txt', 'w', encoding="utf-8_sig")

    for v in tqdm(json_load):
        string = r"<s>\n"
        if(v["input"]==""):
            # 入力がInstrcutionのみの場合
            string += r"以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n[SEP]\n"
            string += r"指示:\n"
            string += replate_newline(v["instruction"])
            string += r"\n[SEP]\n応答:\n"
            string += replate_newline(v["output"])
            string += r"\n</s>"
        else:
            # 入力がInstructionとInputの場合
            string += r"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n[SEP]\n"
            string += r"指示:\n"
            string += replate_newline(v["instruction"])
            string += r"\n[SEP]\n入力:\n"
            string += replate_newline(v["input"])
            string += r"\n[SEP]\n応答:\n"
            string += replate_newline(v["output"])
            string += r"\n</s>"
        # txtファイル書き込み
        f.write(string)
        f.write("\n")

    # txtを閉じる
    f.close()

# def check_length():
#     #  

if __name__ == '__main__':
    change_dataset_format()
