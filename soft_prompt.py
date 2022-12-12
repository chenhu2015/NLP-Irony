from model import GPT2PromptTuningLM, BertPromptTuningLM
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizerFast
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import os
from datasets import load_metric
torch.cuda.empty_cache()
import wandb
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ["WANDB_DISABLED"] = "true"


wandb.init(mode="disabled")

n_prompt_tokens = 20

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   return {"accuracy": load_accuracy, "f1": load_f1}

device = "cuda" if torch.cuda.is_available() else "cpu"

Chinese_test = load_dataset('csv', data_files='Chinese-test.csv')
Chinese_train = load_dataset('csv', data_files='Chinese-train.csv')
English_test = load_dataset('csv', data_files='English-test.csv')
English_train = load_dataset('csv', data_files='English-train.csv')
Mix_train = load_dataset('csv', data_files='Mix-train.csv')
Mix_test = load_dataset('csv', data_files='Mix-test.csv')

tokenizer = AutoTokenizer.from_pretrained("gpt2")
#if tokenizer.pad_token is None:
#    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token
def preprocess_function(examples):
    return tokenizer(examples["content"], truncation=True)

tokenized_Chinese_train = Chinese_train['train'].map(preprocess_function)
tokenized_Chinese_test = Chinese_test['train'].map(preprocess_function)
tokenized_English_train = English_train['train'].map(preprocess_function)
tokenized_English_test = English_test['train'].map(preprocess_function,batched = True)
tokenized_Mix_train = Mix_train['train'].map(preprocess_function,batched = True)
tokenized_Mix_test = Mix_test['train'].map(preprocess_function)
# Only update soft prompt'weights for prompt-tuning. ie, all weights in LM are set as `require_grad=False`. 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize GPT2LM with soft prompt
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2PromptTuningLM.from_pretrained("gpt2",n_tokens=n_prompt_tokens,num_labels=2).to(device)
# model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
# model.config.pad_token_id = model.config.eos_token_id
# print("Current model is")
# print(model)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_Chinese_train,
    eval_dataset=tokenized_Mix_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.evaluate())