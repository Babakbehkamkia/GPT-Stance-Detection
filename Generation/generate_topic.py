import json
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm
import time
import openai

openai.api_key = 'API KEY'



data = pd.read_csv("../Dataset/VAST/vast_train.csv")
posts = data['post'].unique()


new_dataset = {}
for i in tqdm(range(len(posts))):
  if i % 3 == 0 and i != 0:
    time.sleep(61)
  try:
    text = posts[i]
    prompt = f'''list the most potential topics of the given text with their label. 
                a label can be "agree" or "disagree" only. try to find both labels. 
                summarize each topic in 2 to 4 words. 
                return a json in which topics are keys and labels are values:
                example: {{topic here: label here}}
                text: ```{text}```'''
    message = prompt
    messages = [ {"role": "system", "content":
			"You are a intelligent assistant."} ]
    if message:
      messages.append(
        {"role": "user", "content": message},
      )
      chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
      )
    reply = chat.choices[0].message.content
    j = reply
    j = j.split("{")[1]
    j = j.split("}")[0]
    j = "{" + j + "}"
    j = j.replace('\n','')
    json_object = json.loads(j)
    new_dataset[posts[i]] = json_object
  except:
    print("There is an error in ",i)


with open("new_dataset_complete.json", "w") as f:
    json.dump(new_dataset, f)