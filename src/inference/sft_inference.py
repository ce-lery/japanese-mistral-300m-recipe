import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def inference(model_path:str, device: str = "cpu", prompt:str = ""):

    if (device != "cuda" and device != "cpu"):
        device = "cpu"
    if not torch.cuda.is_available():
        device = "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(device)

    # streamer = TextStreamer(tokenizer)
    messages = [{"role": "user", "content": prompt}]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    # token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    # print("token_ids:",token_ids)

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
        )
    generated_text = tokenizer.decode(generated_tokens[0])
    # print(generated_tokens[0])
    print(generated_text.replace(tokenizer.eos_token, "\n"))
    # print(generated_text)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path",help="",default="")
    parser.add_argument("--device",help="",default="")
    parser.add_argument("--prompt",help="",default="")

    args = parser.parse_args()
    inference(args.model_path, args.device, args.prompt)
