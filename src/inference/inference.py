from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sentencepiece as spm
import argparse
from argparse import Namespace

torch.set_float32_matmul_precision('high')

def inference(model_path:str, device: str = "cpu",prompt:str = "",):
    # MODEL_NAME = "../train/checkpoints-mistral-300M-jsonl"

    if (device != "cuda" and device != "cpu"):
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 ).to(device)
    # streamer = TextStreamer(tokenizer)

    inputs = tokenizer(prompt, add_special_tokens=True,return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=1024,
            do_sample=True,
            early_stopping=False,
            top_p=0.90,
            top_k=50,
            temperature=0.1,
            # streamer=streamer,
            no_repeat_ngram_size=2,
            num_beams=3
        )

    print("output token length:", len(outputs[0]))
    print(outputs[0])    
    # print(outputs.tolist()[0])
    outputs_txt = tokenizer.decode(outputs[0])
    # outputs_txt = tokenizer.decode_ids(outputs[0])
    print(outputs_txt)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path",help="",default="")
    parser.add_argument("--device",help="",default="")
    parser.add_argument("--prompt",help="",default="")

    args = parser.parse_args()
    inference(args.model_path, args.device, args.prompt)
