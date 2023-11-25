# from transformers import Pre

from transformers import AutoModelForCausalLM, MistralForCausalLM, MistralConfig   
import json

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MistralConfig.from_dict(config)
    return config

config = load_config_from_json(config_file = "config.json")    
print(config)

model = MistralForCausalLM(config)
print(model)

model_size = sum(t.numel() for t in model.parameters())
print(f"Mistral-300m size: {model_size/1000**2:.1f}M parameters")


from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2-medium"
)

model = GPT2LMHeadModel(config)
print(config)
print(model)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")