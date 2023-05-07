from transformers import pipeline
import json
from tqdm import tqdm
import pickle


path_datasets = "../Dataset/"

classifier = pipeline("text-classification", model="Babak-Behkamkia/bert_gpt_train")


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

def evaluate(texts, labels):
  preds = []
  for i in tqdm(range(len(texts))):
    inputs = texts[i]
    label = labels[i]
    output = classifier(inputs)
    preds.append((output[0]['label'], label))
  return preds

preds = evaluate(texts, labels)


with open(path_datasets + "predictions_bert_gpt", "wb") as fp:
  pickle.dump(preds, fp)

with open(path_datasets + "predictions_bert_gpt", "rb") as fp:
  predictions = pickle.load(fp)

preds = []
labels = []
for i in range(len(predictions)):
  pred, label = predictions[i]
  if pred == "LABEL_0":
    preds.append(0)
    labels.append(label)

  elif pred == "LABEL_1":
    preds.append(1)
    labels.append(label)
  elif pred == "LABEL_2":
    preds.append(2)
    labels.append(label)


count = 0
for i in range(len(preds)):
  if preds[i] == labels[i]:
    count += 1

print(count)
print(count/len(predictions))

