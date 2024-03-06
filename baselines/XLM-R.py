import torch
from datasets import load_dataset
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, TrainingArguments
import os
import random

data_root_path = "../data"
model_root_path = "../models"
data_path = os.path.join(data_root_path, "zh-ru-trip-train-data-5k-no-id.csv")
model_path = os.path.join(model_root_path, "xlm-roberta-base")


dataset = load_dataset("csv", data_files=data_path, split="train")
dataset = dataset.train_test_split(test_size=0.2)

def split_examples(batch):
    queries = []
    passages = []
    labels = []
    for label in ["positive", "negative"]:
        for (query, passage) in zip(batch["queries"], batch[label]):
            queries.append(query)
            passages.append(passage)
            labels.append(int(label == "positive"))
    return {"query": queries, "passage": passages, "label": labels}

dataset = dataset.map(split_examples, batched=True, remove_columns=["queries", "positive", "negative"])

train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_indices = list(range(len(train_dataset)))
random.shuffle(train_indices)
train_dataset = train_dataset.select(train_indices)

test_indices = list(range(len(test_dataset)))
random.shuffle(test_indices)
test_dataset = test_dataset.select(test_indices)

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
config.num_labels = 1
config.problem_type = "multi_label_classification"
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

def tokenize(batch):
    tokenized = tokenizer(
        batch["query"],
        batch["passage"],
        padding=True,
        truncation=True,
        max_length=128,
    )
    tokenized["labels"] = [[float(label)] for label in batch["label"]]
    return tokenized

train_tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["query", "passage", "label"])
test_tokenized_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["query", "passage", "label"])

train_tokenized_dataset.set_format("torch")
test_tokenized_dataset.set_format("torch")

training_args = TrainingArguments(output_dir='./XLM-R_output',
                                  fp16=True,
                                  half_precision_backend="auto",
                                  per_device_train_batch_size=32,
                                  num_train_epochs=3,
                                  save_strategy="epoch",
                                  save_total_limit=3,
                                  logging_steps=50,
                                  warmup_steps=100)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_tokenized_dataset,
                  tokenizer=tokenizer)

trainer.train()