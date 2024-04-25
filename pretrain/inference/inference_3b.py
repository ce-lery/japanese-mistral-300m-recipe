from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
# from transformers import PreTrainedTokenizerFast
import sentencepiece as spm

MODEL_NAME = "../train/checkpoints-mistral-3B"
torch.set_float32_matmul_precision('high')

DEVICE = "cuda"
if torch.cuda.is_available():
    print("cuda")
    DEVICE = "cuda"
else:
    print("cpu")
    DEVICE = "cpu"
# DEVICE = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
).to(DEVICE)

# streamer = TextStreamer(tokenizer)

prompt = "大規模言語モデルとは、"

inputs = tokenizer(prompt, add_special_tokens=False,return_tensors="pt").to(model.device)
with torch.no_grad():

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        do_sample=True,
        early_stopping=False,
        top_p=0.95,
        top_k=50,
        temperature=0.9,
        # streamer=streamer,
        no_repeat_ngram_size=2,
        num_beams=3
    )

# print(outputs[0])
print(outputs.tolist()[0])
outputs_txt = tokenizer.decode(outputs[0])
# outputs_txt = tokenizer.decode_ids(outputs[0])

print(outputs_txt)
