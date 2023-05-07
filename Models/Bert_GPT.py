# !pip install transformers
# !pip install datasets
# !pip install evaluate

import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json


path_datasets = "../Dataset/"


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


with open(path_datasets + "new_dataset_complete.json", 'r') as f:
  data = json.load(f)

dataset = []
for text in data.keys():
  for topic in data[text].keys():
    dataset.append((text, topic, data[text][topic]))

texts = []
labels = []
for i in range(len(dataset)):
  post, topic, label = dataset[i]
  text = post + ' [SEP] ' + topic
  if "disagree" in label.lower():
    labels.append(0)
  elif "agree" in label.lower():
    labels.append(1)
  else:
    labels.append(2)
  texts.append(text)


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)


dataset = DatasetDict({'train':Dataset.from_dict({'label':y_train,'text':X_train}),
                        'test':Dataset.from_dict({'label':y_test,'text':X_test})
                        })


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


small_train_dataset = dataset["train"].map(tokenize_function, batched=True)
small_eval_dataset = dataset["test"].map(tokenize_function, batched=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

TOKEN = "Hugging Face TOKEN"
model.push_to_hub("bert_gpt_train",  use_auth_token=TOKEN)
tokenizer.push_to_hub("bert_gpt_train",  use_auth_token=TOKEN, commit_message="Upload Tokenizer")