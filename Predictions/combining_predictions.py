import pandas as pd
import numpy as np
import pickle
import json



path_datasets = "../Dataset/gpt_predictions/"

with open(path_datasets + "predictions_2591", "rb") as fp: 
  predictions_2591 = pickle.load(fp)

with open(path_datasets + "predictions_first_half", "rb") as fp: 
  predictions_first_half = pickle.load(fp)

with open(path_datasets + "predictions_second_6000", "rb") as fp:
  predictions_second_6000 = pickle.load(fp)

with open(path_datasets + "predictions_second_half", "rb") as fp: 
  predictions_second_half = pickle.load(fp)

with open("../Dataset/new_dataset_complete.json", 'r') as f:
  dataset = json.load(f)

data = {
    "text": [],
    "topic": [],
    "label": []
}
for text in dataset.keys():
  for topic in dataset[text].keys():
    data["text"].append(text)
    data["topic"].append(topic)
    if "disagree" in dataset[text][topic].lower():
      data["label"].append(0)
    elif "agree" in dataset[text][topic].lower():
      data["label"].append(1)
    else:
      data["label"].append(2)

data = pd.DataFrame(data)


len(predictions_2591)

len(predictions_first_half)

len(predictions_second_6000)

len(predictions_second_half)


def count_errors(dataset):
  count = 0
  for i in range(len(dataset)):
    if dataset[i] == 'error':
      count += 1
  return count

print(count_errors(predictions_2591))
print(count_errors(predictions_first_half))
print(count_errors(predictions_second_6000))
print(count_errors(predictions_second_half))

all_preds = []
for i in range(2364):
  all_preds.append(predictions_2591[i])
for i in range(2364, 3658): 
  all_preds.append(predictions_first_half[i-2364])
for i in range(3658, 6000):
  all_preds.append(predictions_second_6000[i-3658])
for i in range(6000, 7317):
  all_preds.append(predictions_second_half[i-6000])

len(all_preds)

labels = data['label']

predictions = []
for i in range(len(all_preds)):
  if "disagree" in all_preds[i].lower():
    predictions.append(0)
  elif "agree" in all_preds[i].lower():
    predictions.append(1)
  elif "error" == all_preds[i]:
    predictions.append(2)
  else:
    predictions.append(2)

num = 0
for i in range(len(predictions)):
  if predictions[i] == 2:
    num += 1
print(num)

count = 0
agree_count = 0
agree_true_count = 0
disagree_count = 0
disagree_true_count = 0



for i in range(len(predictions)):
  if predictions[i] == 2:
    continue
  if labels[i] == 0:
    disagree_count += 1
  elif labels[i] == 1:
    agree_count += 1
  if predictions[i] == labels[i]:
    count += 1
    if labels[i] == 0:
      disagree_true_count += 1
    elif labels[i] == 1:
      agree_true_count += 1
    
print(count)
print((count-10)/(len(predictions)-1080))
print("===========")
print(agree_true_count)
print(agree_true_count/agree_count)
print("===========")
print(disagree_true_count)
print(disagree_true_count/disagree_count)




