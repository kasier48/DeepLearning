import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'distilbert-base-uncased')

ds = load_dataset("fancyzhx/ag_news")

label_info = ds['train'].features['label']
print(f"라벨의 종류: {label_info.names}")

num_labels = len(label_info.names)

def collate_fn(batch):
  max_len = 400
  texts, labels = [], []
  for row in batch:
    labels.append(row['label'])
    texts.append(row['text'])

  encoding = tokenizer(texts, padding=True, truncation=False, max_length=max_len, return_tensors="pt")
  labels = torch.LongTensor(labels)

  return encoding.input_ids, labels, encoding.attention_mask


train_loader = DataLoader(
    ds['train'], batch_size=64, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    ds['test'], batch_size=64, shuffle=False, collate_fn=collate_fn
)

from torch import nn

class TextClassifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', 'distilbert-base-uncased')
    self.classifier = nn.Linear(768, num_labels)

  def forward(self, x, attention_mask):
    x = self.encoder(x, attention_mask)['last_hidden_state']
    x = self.classifier(x[:, 0])

    return x


model = TextClassifier()

for param in model.encoder.parameters():
  param.requires_grad = False
  
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')
lr = 0.001
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=lr)
n_epochs = 10

for epoch in range(n_epochs):
  total_loss = 0.
  model.train()
  for data in train_loader:
    model.zero_grad()
    inputs, labels, attention_mask = data
    inputs, labels, attention_mask = inputs.to(device), labels.to(device), attention_mask.to(device)

    preds = model(inputs, attention_mask)
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  print(f"Epoch {epoch:3d} | Train Loss: {total_loss}")
  
def accuracy(model, dataloader):
  cnt = 0
  acc = 0

  for data in dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    preds = model(inputs)
    preds = torch.argmax(preds, dim=-1)
    # preds = (preds > 0).long()[..., 0]

    cnt += labels.shape[0]
    acc += (labels == preds).sum().item()

  return acc / cnt


with torch.no_grad():
  model.eval()
  train_acc = accuracy(model, train_loader)
  test_acc = accuracy(model, test_loader)
  print(f"=========> Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")