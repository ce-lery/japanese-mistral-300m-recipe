from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer,BitsAndBytesConfig
import torch
# import sentencepiece as spm
import argparse
from argparse import Namespace
from transformers.trainer_utils import set_seed
set_seed(42)

from pprint import pprint
from datasets import load_dataset, concatenate_datasets
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
import itertools
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

torch.set_float32_matmul_precision('high')


def instruction_tune(model_name_or_path:str, device: str = "cpu",prompt:str = "",output_dir:str=""):
    """
        Refs: https://github.com/ghmagazine/llm-book/blob/main/chapter11/11-2-instruction_tuning-train.ipynb
    """

    if (device != "cuda" and device != "cpu"):
        device = "cpu"

    """ load dataset """ 
    # load swallow-instruct dataset
    ds = load_dataset("llm-book/oasst1-21k-ja", split="train", cache_dir=output_dir+"/.cache/")
    print(ds)
    pprint(ds[0])

    """ set tokenizer """
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    tokenizer.chat_template = """
    {%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ bos_token + 'User:' + message['content'] + eos_token }}
    {%- elif message['role'] == 'assistant' -%}
        {% generation %}{{ bos_token + 'Assistant:'  + message['content'] + eos_token }}{% endgeneration %}
    {%- endif -%}
    {%- if loop.last and add_generation_prompt -%}
        {{ bos_token + 'Assistant:' }}
    {%- endif -%}
    {%- endfor -%}
    """

    # output sample of text
    chat_text = tokenizer.apply_chat_template(
        ds[0]["conversation"], tokenize=False
    )
    print(chat_text.replace(tokenizer.eos_token, "\n"))

    # output sample of text without last assistant message
    chat_text = tokenizer.apply_chat_template(
        ds[0]["conversation"][:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(chat_text.replace(tokenizer.eos_token, "\n"))

    """ apply tokenizer to dataset """
    # change dataset to token id with chat template
    tokenized_dataset = [
        tokenizer.apply_chat_template(item["conversation"],return_dict=True,return_assistant_tokens_mask=True)
        for item in ds
    ]
    # output sample text
    token_ids = tokenized_dataset[0]
    # print("token id:", token_ids)
    # print("token   :", tokenizer.convert_ids_to_tokens(token_ids["input_ids"]))
    # print("assistant mask:",token_ids["assistant_masks"])

    #tokenizer.pad_token = tokenizer.unk_token
    print(tokenizer.pad_token)
    print(tokenizer.unk_token)

    """ apply datacollator to dataset """
    # apply complementation
    # refs: https://huggingface.co/docs/trl/v0.23.0/en/sft_trainer#trl.trainer.sft_trainer.DataCollatorForLanguageModeling
    #https://huggingface.co/docs/transformers/ja/chat_templating
    #https://huggingface.co/docs/transformers/v4.56.2/ja/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template
    bos = tokenizer.bos_token
    collator = DataCollatorForLanguageModeling(
        completion_only_loss=True,
        pad_token_id=0 # TODO: check here
    )
    #https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
    # completion_mask==0 data, the labels are to be -100, and excluded from train
    # https://huggingface.co/docs/trl/v0.23.0/sft_trainer
    # print(tokenized_dataset[0])
    # pprint(tokenized_dataset[0])
    # print([{"input_ids":tokenized_dataset[0]}])
    batch = collator([tokenized_dataset[0]])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    # print("token id:", input_ids)
    # print("labels:", labels)

    segments_to_fit: list[list[int]] = []
    segments_to_ignore: list[list[int]] = []
    # grouping by labels
    for key, group in itertools.groupby(
        range(len(input_ids)), key=lambda i: labels[i] == -100
    ):
        group = list(group)
        if key:
            segments_to_ignore.append(group)
        else:
            segments_to_fit.append(group)

    print("---- the part that isn't used by calculating loss ----")
    for seg in segments_to_ignore:
        print(tokenizer.decode(input_ids[seg]))

    print("---- the part that is used by calculating loss  ----")
    for seg in segments_to_fit:
        print(tokenizer.decode(input_ids[seg]))

    """ Setting of model parameters """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        use_cache=False, 
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,  # this is needed for gradient checkpointing
        trust_remote_code=True,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        # target_modules=[
        #     "q_proj",
        #     "k_proj",
        #     "v_proj",
        #     "o_proj",
        #     "gate_proj",
        #     "up_proj",
        #     "down_proj",
        # ],
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    """ Train """
    # training configuration parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        num_train_epochs=1,
        #max_steps=50,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=300,
        report_to="none",
    )

    trainer = Trainer(
        model,
        train_dataset=tokenized_dataset,
        data_collator=collator,  # the module that execute labeling and creating mini-batch
        args=training_args,
        processing_class=tokenizer,  # if you set this option, model is saved with tokenizer
    )

    trainer.train()

    print("SFT Training Completed.")

    """ inference """
    """
    messages = [{"role": "user", "content": prompt}]
    model = AutoModelForCausalLM.from_pretrained(output_dir+"/checkpoint-1323", trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(output_dir+"/checkpoint-1323", trust_remote_code=True, use_fast=False)

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_tokens = model.generate(tokenized_chat.to("cuda"), 
                                        use_cache=True, 
                                        max_new_tokens=1024,
                                        do_sample=True,
                                        early_stopping=False,
                                        top_p=0.95,
                                        top_k=50,
                                        temperature=0.7,
                                        # streamer=streamer,
                                        no_repeat_ngram_size=2,
                                        num_beams=3,
    )
    generated_text = tokenizer.decode(generated_tokens[0])
    print(generated_text.replace(tokenizer.eos_token, "\n"))
    """

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_name_or_path",help="",default="ce-lery/mistral-300m-base")
    parser.add_argument("--device",help="",default="cuda")
    parser.add_argument("--prompt",help="",default="学習が完了しました。何か一言お願いします。")
    parser.add_argument("--output_dir",help="",default="")

    args = parser.parse_args()
    instruction_tune(args.model_name_or_path, args.device, args.prompt,args.output_dir)
