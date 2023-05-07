import pickle

with open("file_name", "rb") as fp:
  predictions = pickle.load(fp)

preds = []
labels = []
for i in range(len(predictions)):
  pred, label = predictions[i]
  labels.append(label)
  if pred == "LABEL_0":
    preds.append(0)
  elif pred == "LABEL_1":
    preds.append(1)
  elif pred == "LABEL_2":
    preds.append(2)

num = 0
for i in range(len(preds)):
  if preds[i] == 2:
    num += 1
print(num)

count = 0
agree_count = 0
agree_true_count = 0
disagree_count = 0
disagree_true_count = 0



for i in range(len(preds)):
  if preds[i] == 2:
    continue
  if labels[i] == 0:
    disagree_count += 1
  elif labels[i] == 1:
    agree_count += 1
  if preds[i] == labels[i]:
    count += 1
    if labels[i] == 0:
      disagree_true_count += 1
    elif labels[i] == 1:
      agree_true_count += 1
    
print(count)
print((count)/len(preds))
print("===========")
print(agree_true_count)
print(agree_true_count/agree_count)
print("===========")
print(disagree_true_count)
print(disagree_true_count/disagree_count)

count = 0

for i in range(len(preds)):
  if preds[i] == labels[i]:
    count += 1

print(count)
print(count/len(predictions))

