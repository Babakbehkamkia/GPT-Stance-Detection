# !pip install -q openai
# !pip install transformers
# !pip install datasets
# !pip install evaluate

import json
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm
import time
import openai

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import pickle


path_datasets = "../Dataset/"

tokenizer = AutoTokenizer.from_pretrained("Babak-Behkamkia/bert_gpt_train")

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


data_test = pd.read_csv(path_datasets + "VAST/vast_dev.csv")

def prepare_text(row):
  post = row['post']
  topic = row['new_topic']
  text = post + ' [SEP] ' + topic
  return text

data_test['text'] = data_test.apply(prepare_text, axis=1)

def generate_sample(text, topic):
  openai.api_key = 'API KEY'
  samples = []
  flag = True

  for label in ["agree", "disagree"]:
    if label == "disagree" and flag:
      flag = False
      time.sleep(61)
    messages = [ {"role": "system", "content":
            "You are a intelligent assistant."} ]
    
    for i in tqdm(range(3)):
        
      try:
        prompt = f'''Your task is to generate a human written post. Do not mention that your are an intelligent assistant
                    The generate post must discuss about the given topic in some point of itself. 
                    The generated post must have a stance toward this topic. 
                    the stance could be "agree" or "disagree".
                    the post should be at most 2 paragraphs.
                    just return the post.
                    topic: {topic}
                    stance: {label}

                    Here is an example post, but we do not know the stance of the text blow toward the topic. you can learn from its structure.
                    text: ```{text}```'''
        
          
        message = prompt
        if message:
          messages.append(
            {"role": "user", "content": message},
          )
          chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
          )
        reply = chat.choices[0].message.content        
        samples.append(reply)
        messages.append({"role": "assistant", "content": reply})
      except Exception as e: 
        print(e)
        samples.append("error")
        print("There is an error in : ", i)
    
  return samples

# sample_text = "Without government to ensure their behavior, companies will attempt to make a profit even to the DETRIMENT of the society that supports the business. We have seen this in the environment, in finances, in their treatment of workers and customers. Enough."

# sample_topic = "Role of government"

# generated_text= generate_sample(sample_text, sample_topic)

# try:
#   prompt = f'''What is the stance of the text which is delimited by triple backticks toward the given topic. 
#               your answer should agree, disagree or neutral. keep your answer 1 word long. 
#               please double check your answer before response and be sure about it.
#               topic: {sample_topic}
#               text: ```{generated_text[1]}```'''
  
#   messages = [ {"role": "system", "content":
#     "You are a intelligent assistant."} ]
#   message = prompt
#   if message:
#     messages.append(
#       {"role": "user", "content": message},
#     )
#     chat = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo", messages=messages
#     )
#   reply = chat.choices[0].message.content

#   print(reply)
  
#   messages.append({"role": "assistant", "content": reply})
# except Exception as e: 
#   print(e)

def fine_tune_model(post, topic, model):
  new_texts = generate_sample(post, topic)

  new_data = [new_texts[i] + ' [SEP] ' + topic for i in range(len(new_texts))]
  labels = [1]*3 + [0]*3
  dataset = DatasetDict({'train':Dataset.from_dict({'label':labels,'text':new_data})})

  small_train_dataset = dataset["train"].map(tokenize_function, batched=True)
  
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_train_dataset,
    compute_metrics=compute_metrics,
  )
  trainer.train()

TOKEN = "Hugging Face TOKEN"

def evaluate(posts, topics, texts, labels):
  preds = []
  model = AutoModelForSequenceClassification.from_pretrained("Babak-Behkamkia/bert_gpt_train")
  for i in tqdm(range(len(texts))):
    fine_tune_model(posts[i], topics[i], model)
    model.push_to_hub("bert_gpt_train",  use_auth_token=TOKEN)

    inputs = texts[i]
    label = labels[i]
    
    classifier = pipeline("text-classification", model="Babak-Behkamkia/bert_gpt_train")
    output = classifier(inputs)
    preds.append((output[0]['label'], label))

    
    # with open(f"predictions_bert_gpt_{i}", "wb") as fp:
    #   pickle.dump(preds, fp)
  return preds

preds = evaluate(data_test['post'], data_test['new_topic'], data_test['text'], data_test['label'])

import pickle
with open("predictions_bert_gpt", "wb") as fp:   #Pickling
  pickle.dump(preds, fp)