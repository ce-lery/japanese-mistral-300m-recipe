# from transformers import Pre

from transformers import AutoModelForCausalLM, MistralForCausalLM, MistralConfig   
import json

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MistralConfig.from_dict(config)
    return config

config = load_config_from_json(config_file = "../mistral-39m/config.json")    
print(config)

model = MistralForCausalLM(config)
print(model)

model_size = sum(t.numel() for t in model.parameters())
print(f"Total Mistral size={model_size/2**20:.2f}M params")


from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig,GPTNeoXForCausalLM

config = AutoConfig.from_pretrained(
    "cyberagent/open-calm-3b"
)
# model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-3b", device_map="auto", torch_dtype=torch.float16)


model = GPTNeoXForCausalLM(config)
print(config)
print(model)
model_size = sum(t.numel() for t in model.parameters())
print(f"open-calm-3b size: {model_size/1000**3:.1f}B parameters")