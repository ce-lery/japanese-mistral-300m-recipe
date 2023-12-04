import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

current_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# model.safetensorを読み込むには、pip install safetensorsが必要
tokenizer = AutoTokenizer.from_pretrained(current_path+"/../train/checkpoints-finetuning/", use_fast=False,trust_remote_code=True)
# model.safetensorを読み込むには、pip install safetensorsが必要
model = AutoModelForCausalLM.from_pretrained(current_path+"/../train/checkpoints-finetuning/",trust_remote_code=True).to(device)

# tokenizer = AutoTokenizer.from_pretrained("inu-ai/dolly-japanese-gpt-1b", use_fast=False)
# model = AutoModelForCausalLM.from_pretrained("inu-ai/dolly-japanese-gpt-1b").to(device)

# """
MAX_ASSISTANT_LENGTH = 100
MAX_INPUT_LENGTH = 1024
INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n入力:\n{input}\n[SEP]\n応答:\n'
NO_INPUT_PROMPT = r'<s>\n以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n[SEP]\n指示:\n{instruction}\n[SEP]\n応答:\n'

def prepare_input(instruction, input_text):
    if input_text != "":
        prompt = INPUT_PROMPT.format(instruction=instruction, input=input_text)
    else:
        prompt = NO_INPUT_PROMPT.format(instruction=instruction)
    return prompt

def format_output(output):
    output = output.lstrip("<s>").rstrip("</s>").replace("[SEP]", "").replace("\\n", "\n")
    return output

def generate_response(instruction, input_text):
    prompt = prepare_input(instruction, input_text)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    n = len(token_ids[0])
    # print(n)

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            min_length=n,
            max_length=min(MAX_INPUT_LENGTH, n + MAX_ASSISTANT_LENGTH),
            top_p=0.95,
            top_k=50,
            temperature=0.4,
            do_sample=True,
            no_repeat_ngram_size=2,
            num_beams=2,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.unk_token_id]]
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    formatted_output_all = format_output(output)
    response = f"Assistant:{formatted_output_all.split('応答:')[-1].strip()}"

    return formatted_output_all, response 

instruction = "あなたは何でも正確に答えられるAIです。"
questions = [
    "日本で一番高い山は？",
    "日本で一番広い湖は？",
    "世界で一番高い山は？",
    "世界で一番広い湖は？",
    "冗談を言ってください。",
]

# 各質問に対して応答を生成して表示
for question in questions:
    formatted_output_all, response = generate_response(instruction, question)
    print(response)
# """