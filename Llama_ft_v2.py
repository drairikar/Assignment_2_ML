import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Dict, List
from datasets import load_dataset, Dataset, disable_caching
disable_caching()
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset
from IPython.display import Markdown
from functools import partial
# import copy
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
import bitsandbytes
from peft import TaskType

# Load the dataset
ds = load_dataset("Deysi/spam-detection-dataset", split="train")
print("Dataset", ds.column_names)

from huggingface_hub import login
login(token="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")
# small_ds = ds.select([i for i in range(1000)])

# prompt_template = "This is a {} message: {}"
# answer_template = """{response}"""

def _add_text(batch):
    label_map = {0: "not_spam", 1: "spam"}
    results = {
        "Prompt": [],
        "Answer": [],
        "text" : []
    }

    for message, label in zip(batch["text"], batch["label"]):
        message = message.strip()
        # label = rec.get("label")
        instruction = label_map.get(label, "unknown")

        if label == 1:
            results["Prompt"].append("Generate a spam message")
            results["Answer"].append(message)
            results["text"].append("Generate a spam message" + "\n\n" + message)

        results["Prompt"].append(f"Classify the following message as spam or not:\n\n{message}\n\nAnswer:")
        results["Answer"].append(instruction)
        results["text"].append(f"Classify the following message as spam or not:\n\n{message}\n\nAnswer:" + "\n\n" + instruction)

    return results

    # message = rec.get("text", "").strip()
    # label = rec.get("label")
    # instruction = label_map.get(label, "unknown")

    # if label == 1:
    #     results.append({
    #         "Prompt": "Generate a spam message",
    #         "Answer": message,
    #         "text": "Generate a spam message" + "\n\n" + message
    #     })

    # results.append({
    #     "Prompt": f"Classify the following message as spam or not:\n\n{message}\n\nAnswer:",
    #     "Answer": instruction,
    #     "text": f"Classify the following message as spam or not:\n\n{message}\n\nAnswer:" + "\n\n" + instruction
    # })

    # return results

# ds = ds.map(_add_text, batched=True)
ds = ds.map(_add_text, batched=True)
print("Columns", ds.column_names)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model_name = "meta-llama/Llama-3.2-1B"
access_token = "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"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                              device_map = {'':torch.cuda.current_device()}, 
                                             load_in_8bit = True, 
                                             torch_dtype = torch.float16)

model.resize_token_embeddings(len(tokenizer))

MAX_LENGTH = 128

def _preprocess_batch(batch: Dict[str, List[str]]):
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    outputs = tokenizer(batch["Answer"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    inputs["labels"] = outputs["input_ids"]

    # inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    # inputs["input_ids"] = inputs["input_ids"]
    # inputs["attention_mask"] = inputs["attention_mask"]
    
    # # Create labels from input_ids
    # inputs["labels"] = inputs["input_ids"]
    return inputs

_preprocesing_fn = partial(_preprocess_batch)
encode_ds = ds.map(_preprocesing_fn, batched=True, remove_columns = ["text", "label", "Prompt", "Answer"])
print(encode_ds.column_names)


processed_ds = encode_ds.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)

split_ds = processed_ds.train_test_split(test_size= 100, seed=42)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=MAX_LENGTH, padding="max_length", pad_to_multiple_of=8)

LORA_R = 128
LORA_Alpha = 256
Lora_dropout = 0.05

lora_config = LoraConfig(
    r = LORA_R,
    lora_alpha = LORA_Alpha,
    lora_dropout = Lora_dropout,
    bias = "none",
    task_type= TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=MAX_LENGTH, padding="max_length", pad_to_multiple_of=8)

EPOCHS = 3
learning_rate = 1e-4
model_name = "Llama_ft"
output_dir = f"./{model_name}"

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=EPOCHS,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=split_ds["train"],
    eval_dataset=split_ds["test"],
    data_collator=data_collator,
)

model.config.use_cache = False
trainer.train()
trainer.model.save_pretrained(output_dir)
trainer.save_model(output_dir)
trainer.model.config.save_pretrained(output_dir)

def generate_and_detect_spam(task_type = "generate", input_message = None):

    inference = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=128)

    if task_type == "generate":
        prompt = "Generate a spam message"
    
    elif task_type == "detect":
        if not input_message:
            raise ValueError("input_message is required for spam detection")
        prompt = f"Classify the following message as spam or not:\n\n{input_message}\n\nAnswer:"

    else:
        raise ValueError("task_type should be either generate or detect")
    
    
    result = inference(prompt)[0]["generated_text"]
    return result

generated_spam = generate_and_detect_spam(task_type="generate")
print("Generated Spam", generated_spam)

detected_spam = generate_and_detect_spam(task_type="detect", input_message=generated_spam)
print("Detected Spam", detected_spam)


# def post_process(response):
#     ##generate response for spam
#     response = response.replace("This is a spam message: ", "")
#     inference = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=128)
#     response = inference(prompt_template.format(instruction="spam", response=response)[len("This is a spam message: "):])[0]["generated_text"]

#     formatted_response = post_process(response)

#     formatted_response

   

    





