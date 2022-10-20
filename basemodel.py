import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import wandb
import os
import torch
import numpy as np
from datasets import load_metric

os.environ["WANDB_DISABLED"] = "true"

wandb.init(mode="disabled")

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

device = "cuda" if torch.cuda.is_available() else "cpu"

Chinese_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/Chinese-test.csv')
Chinese_train = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/Chinese-train.csv')
English_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/English-test.csv')
English_train = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/English-train.csv')
Mix_train = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/Mix-train.csv')
Mix_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/Mix-test.csv')


## pre-train process with xlm-roberta
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_function(examples):
    return tokenizer(examples["content"], truncation=True)

def preprocess_function_1(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_Chinese_train = Chinese_train['train'].map(preprocess_function, batched=True)
tokenized_Chinese_test = Chinese_test['train'].map(preprocess_function, batched=True)

tokenized_English_train = English_train['train'].map(preprocess_function, batched=True)
tokenized_English_test = English_test['train'].map(preprocess_function, batched=True)

tokenized_Mix_train = Mix_train['train'].map(preprocess_function, batched=True)
tokenized_Mix_test = Mix_test['train'].map(preprocess_function, batched=True)

# imdb = load_dataset("imdb")
# tokenized_imdb = imdb.map(preprocess_function_1, batched=True)
# print(tokenized_imdb['train'][0])
# print(tokenized_Mix_train[0])
# exit()


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_Mix_train,
    eval_dataset=tokenized_English_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.evaluate())