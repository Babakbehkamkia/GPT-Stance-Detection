from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




bert_model = BertModel.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class AttentionModel(nn.Module):
    def __init__(self, input_size):
        super(AttentionModel, self).__init__()
        self.W = nn.Linear(input_size, input_size, bias=False)
        self.v = nn.Linear(input_size, 1, bias=False)

    def forward(self, input1, input2):
        h = torch.tanh(self.W(input2.unsqueeze(1)))
        a = torch.softmax(self.v(h), dim=1)
        c = torch.sum(a * input1.unsqueeze(2),dim=1)
        return c

class LinearLayer(nn.Module):
    def __init__(self, input_size, num_labels):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(1, num_labels)
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, num_labels)

    def forward(self, x):
        out = self.linear(x)
        return out

class DeepLearningModel(nn.Module):
    def __init__(self, bert_model, tokenizer, input_size, num_labels):
        super(DeepLearningModel, self).__init__()
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.attention = AttentionModel(input_size)
        self.linear = LinearLayer(input_size, num_labels)

    def forward(self, text1, text2):

        bert_out1 = self.bert(**text1)[0][:, 0, :]
        bert_out2 = self.bert(**text2)[0][:, 0, :]

        attention_out = self.attention(bert_out1, bert_out2)
        out = self.linear(attention_out)
        out = torch.sigmoid(out)
        return out


def train(model, train_loader, val_loader, epochs=10, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_true = 0
        val_true = 0

        model.train()
        for input1, input2, label in tqdm(train_loader):
            input_ids1 = tokenizer(list(input1), padding=True, truncation=True, return_tensors="pt")
            input_ids2 = tokenizer(list(input2), padding=True, truncation=True, return_tensors="pt")
            optimizer.zero_grad()
            outputs = model(input_ids1, input_ids2)
            outputs = outputs.squeeze(1)
            for i in range(len(outputs)):
              pred = None
              if outputs[i] >= 0.5:
                pred = 1
              else:
                pred = 0
              if pred == label[i]:
                train_true += 1
            loss = criterion(outputs, label.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for input1, input2, label in val_loader:
                input_ids1 = tokenizer(list(input1), padding=True, truncation=True, return_tensors="pt")
                input_ids2 = tokenizer(list(input2), padding=True, truncation=True, return_tensors="pt")

                outputs = model(input_ids1, input_ids2)

                outputs = outputs.squeeze(1)
                for i in range(len(outputs)):
                  pred = None
                  if outputs[i] >= 0.5:
                    pred = 1
                  else:
                    pred = 0
                  if pred == label[i]:
                    val_true += 1

                loss = criterion(outputs, label.float())
                val_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc = train_true / len(train_loader.dataset)
        vall_acc = val_true / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc}, Val Loss: {val_loss:.4f}, Val Acc: {vall_acc}')


class TextTargetDataset(Dataset):
    def __init__(self, texts, targets, labels):
        self.texts = texts
        self.targets = targets
        self.labels = labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        label = self.labels[idx]
        return text, target, label

    def __len__(self):
        return len(self.texts)

path_datasets = "../Dataset/VAST/"

train_data = pd.read_csv(path_datasets + "gpt-3_train.csv").iloc[:500]
test_data = pd.read_csv(path_datasets + "gpt-3_test.csv")


train_dataset = TextTargetDataset(train_data['post'], train_data['new_topic'], train_data['label'])
val_dataset = TextTargetDataset(test_data['post'], test_data['new_topic'], test_data['label'])


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

model = DeepLearningModel(bert_model, tokenizer, 768, 1)
train(model, train_loader, val_loader)

TOKEN = "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"
model.push_to_hub("bert_attention_stance",  use_auth_token=TOKEN)
tokenizer.push_to_hub("bert_attention_stance",  use_auth_token=TOKEN, commit_message="Upload Tokenizer")
