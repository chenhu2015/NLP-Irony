import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from soft_embedding import SoftEmbedding

import wandb
import os
import torch
import numpy as np
from datasets import load_metric

os.environ["WANDB_DISABLED"] = "true"
mode = 1

wandb.init(mode="disabled")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
   load_precision = load_metric("precision")
   load_recall = load_metric("recall")

   logits, labels = eval_pred
   np.save("/home/kumarv/xuan0008/NLP_project/softmax/english_mix_bert.npy",logits)
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   precision = load_precision.compute(predictions=predictions, references=labels)["precision"]
   recall = load_recall.compute(predictions=predictions, references=labels)["recall"]
   return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall }

device = "cuda" if torch.cuda.is_available() else "cpu"

Chinese_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/dataset/Chinese-test.csv')
Chinese_train = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/dataset/Chinese-train.csv')
English_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/dataset/English-test.csv')
English_train = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/dataset/English-train.csv')
Mix_train = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/dataset/Mix-train.csv')
Mix_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/dataset/Mix-test.csv')

English_error_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/English_W.csv')
Chinese_error_test = load_dataset('csv', data_files='/home/kumarv/xuan0008/NLP_project/Chinese_W.csv')

## pre-train process with xlm-roberta
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
print(Chinese_test['train']['content'][0])

def preprocess_function(examples):
   

    return tokenizer(examples["content"], truncation=True)


tokenized_Chinese_train = Chinese_train['train'].map(preprocess_function, batched=True)
tokenized_Chinese_test = Chinese_test['train'].map(preprocess_function, batched=True)

tokenized_English_train = English_train['train'].map(preprocess_function, batched=True)
tokenized_English_test = English_test['train'].map(preprocess_function, batched=True)

tokenized_Mix_train = Mix_train['train'].map(preprocess_function, batched=True)
tokenized_Mix_test = Mix_test['train'].map(preprocess_function, batched=True)



data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2).to(device)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_English_train,
    eval_dataset = tokenized_Mix_test,
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
)

if mode == 1:
    trainer.train()
    trainer.save_model("Mix_Bert")

print(trainer.evaluate())
