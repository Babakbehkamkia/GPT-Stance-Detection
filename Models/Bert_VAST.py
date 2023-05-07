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


path_datasets = "../Dataset/"


data_train = pd.read_csv(path_datasets + "VAST/vast_train.csv")
data_test = pd.read_csv(path_datasets + "VAST/vast_dev.csv")


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# print(model)

# def print_net_parameters(net):
#     for name, para in net.named_parameters():
#         print("-"*20)
#         print(f"name: {name}")
#         print("values: ")
#         print(para)
# print_net_parameters(model)

# print(model.classifier)

# for name, param in model.named_parameters():
#      if param.requires_grad and 'classifier' not in name:
#          param.requires_grad = False

# print_net_parameters(model)


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def prepare_text(row):
  post = row['post']
  topic = row['new_topic']
  text = post + ' [SEP] ' + topic
  return text

data_train['text'] = data_train.apply(prepare_text, axis=1)
data_test['text'] = data_test.apply(prepare_text, axis=1)

dataset = DatasetDict({'train':Dataset.from_dict({'label':data_train['label'],'text':data_train['text']}),
                        'test':Dataset.from_dict({'label':data_test['label'],'text':data_test['text']})
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
model.push_to_hub("bert_VAST_train_only",  use_auth_token=TOKEN)
tokenizer.push_to_hub("bert_VAST_train_only",  use_auth_token=TOKEN, commit_message="Upload Tokenizer")