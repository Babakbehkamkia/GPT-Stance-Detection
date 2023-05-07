import json
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm
import time
import pickle
import json
import openai


openai.api_key = 'API KEY'


with open("new_dataset_complete.json", 'r') as f:
  data = json.load(f)

dataset = []
for text in data.keys():
  for topic in data[text].keys():
    dataset.append((text, topic, data[text][topic]))



preds = []
for i in tqdm(len(dataset)):
  if i % 3 == 0 and i != 0:
    time.sleep(61)
  try:
    text, topic, label = dataset[i] 
    prompt = f'''What is the stance of the text which is delimited by triple backticks toward the given topic. 
                your answer should agree, disagree or neutral. keep your answer 1 word long. 
                please double check your answer before response and be sure about it.
                topic: {topic}
                text: ```{text}```'''
    
    if i % 10 ==0:
      messages = [ {"role": "system", "content":
        "You are a intelligent assistant."} ]
    message = prompt
    if message:
      messages.append(
        {"role": "user", "content": message},
      )
      chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
      )
    reply = chat.choices[0].message.content
    
    preds.append(reply)
    messages.append({"role": "assistant", "content": reply})
  except Exception as e: 
    print(e)
    preds.append("error")
    print("There is an error in : ", i)
    with open(f"predictions_{i}", "wb") as f:
      pickle.dump(preds, f)


with open("predictions", "wb") as fp:
  pickle.dump(preds, fp)