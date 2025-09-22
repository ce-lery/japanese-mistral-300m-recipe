from transformers import AutoModelForCausalLM, AutoConfig
import argparse
from argparse import Namespace

def check_model_size(config_name):
    config = AutoConfig.from_pretrained(config_name)
    print(config)

    model = AutoModelForCausalLM.from_config(config)
    print(model)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size/10**6:.2f}M params")

def print_state_dict_keys(state_dict):
    for key in state_dict.keys():
        print(key)

if __name__ == "__main__": 

    # Please run `uv run python src/util/check_model_size.py` at top directory.` 


    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_name",help="",default="./src/config/config_mistral_300m.json")
    # parser.add_argument("--output_path",help="",default="")
    
    args = parser.parse_args()
    check_model_size(args.config_name)

