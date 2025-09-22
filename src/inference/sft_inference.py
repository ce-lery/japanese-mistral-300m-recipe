import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

current_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device=torch.device("cpu")

model_path = current_path+"/../../examples/inst-tuning/results/train1/checkpoint-1323"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(device)

print("model:\n",model)

def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    # token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    # print("token_ids:",token_ids)
    print("messages:",messages)
    print("prompt:",prompt)

    with torch.no_grad():
        generated_tokens = model.generate(
            tokenized_chat.to("cuda"), 
            use_cache=True, 
            early_stopping=False,
            max_new_tokens=1024,
            top_p=0.95,
            top_k=50,
            temperature=0.2,
            do_sample=True,
            no_repeat_ngram_size=2,
            num_beams=3,
            # pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # bad_words_ids=[[tokenizer.unk_token_id]]
        )
    generated_text = tokenizer.decode(generated_tokens[0])
    # print(generated_tokens[0])
    print(generated_text.replace(tokenizer.eos_token, "\n"))
    # print(generated_text)

    # output = tokenizer.decode(output_ids.tolist()[0])
    # formatted_output_all = format_output(output)
    # response = f"Assistant:{formatted_output_all.split('応答:')[-1].strip()}"

    # return formatted_output_all, response 

prompts = [
    "日本で一番高い山は何？",
    "日本で一番大きな湖の名前は？",
    "世界で一番高い山は何？",
    "世界で一番大きな湖の名前は？",
    "冗談を言ってください。",
    "香川県の名物は何？",
    "日本の首都はどこ？",
    "こんにちは！",
    "兵庫県の県庁所在地の名前は？",

]

# 各質問に対して応答を生成して表示
for prompt in prompts:
    generate_response(prompt)

