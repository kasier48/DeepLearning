import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
# from transformers import BertTokenizerFast
# from tokenizers import (
#     decoders,
#     models,
#     normalizers,
#     pre_tokenizers,
#     processors,
#     trainers,
#     Tokenizer,
# )


ds = load_dataset("stanfordnlp/imdb")
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
  max_len = 400
  texts, labels = [], []
  for row in batch:
    # [MYCODE] label에는 -2를 주어 마지막 단어를 주도록 설정
    # texts에는 -2를 주어 마지막 단어를 제외한 문장을 주도록 설정
    labels.append(tokenizer(row['text'], truncation=True, max_length=max_len).input_ids[-2])
    texts.append(torch.LongTensor(tokenizer(row['text'], truncation=True, max_length=max_len).input_ids[:-2]))

  texts = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)
  labels = torch.LongTensor(labels)

  return texts, labels


train_loader = DataLoader(
    ds['train'], batch_size=64, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    ds['test'], batch_size=64, shuffle=False, collate_fn=collate_fn
)

from torch import nn
from math import sqrt

class MultiHeadAttention(nn.Module):
  def __init__(self, input_dim, d_model, num_heads):
    super().__init__()

    self.input_dim = input_dim
    self.d_model = d_model
    
    # [MOYCODE] d_k에 num_heads 만큼의 차원 단위 부여
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.wq = nn.Linear(input_dim, d_model)
    self.wk = nn.Linear(input_dim, d_model)
    self.wv = nn.Linear(input_dim, d_model)
    self.dense = nn.Linear(d_model, d_model)

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x, mask):
    batch_size = x.size(0)
    
    # [MYCODE] split_heads를 통해 num_heads 만큼 차원으로 확장.
    # (B, S, D) -> (B, H, S, D_K)
    q = self.__split_heads(self.wq(x))
    k = self.__split_heads(self.wk(x))
    v = self.__split_heads(self.wv(x))
    
    # (B, H, S, D_K) * (B, H, D_K, S) = (B, H, S, S)
    score = torch.matmul(q, k.transpose(-1, -2))
    
    # [MYCODE] d_k = d_model / num_heads 단위로 처리되므로 변경
    score = score / sqrt(self.d_k)

    if mask is not None:
      score = score + (mask * -1e9)

    # (B, H, S, S) * (B, H, S, D_K) = (B, H, S, D_K)
    score = self.softmax(score)
    result = torch.matmul(score, v)
    
    # [MYCODE] num_heads 만큼 다시 결합하여 d_model 차원으로 복원한다.
    # (B, S * H, D)
    result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    # (B, S, D)
    result = self.dense(result)

    return result
  
  def __split_heads(self, x):
    batch_size, seq_len, d_model = x.size()
    x = x.view(batch_size, self.num_heads, seq_len, self.d_k)
    return x

class TransformerLayer(nn.Module):
  def __init__(self, input_dim, d_model, dff):
    super().__init__()

    self.input_dim = input_dim
    self.d_model = d_model
    self.dff = dff

    self.sa = MultiHeadAttention(input_dim, d_model)
    self.ffn = nn.Sequential(
      nn.Linear(d_model, dff),
      nn.ReLU(),
      nn.Linear(dff, d_model)
    )

  def forward(self, x, mask):
    x = self.sa(x, mask)
    x = self.ffn(x)

    return x

import numpy as np

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, None], np.arange(d_model)[None, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[None, ...]

    return torch.FloatTensor(pos_encoding)


max_len = 400
print(positional_encoding(max_len, 256).shape)

class TextClassifier(nn.Module):
  def __init__(self, vocab_size, d_model, n_layers, dff):
    super().__init__()

    self.vocab_size = vocab_size
    self.d_model = d_model
    self.n_layers = n_layers
    self.dff = dff

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = nn.parameter.Parameter(positional_encoding(max_len, d_model), requires_grad=False)
    self.layers = nn.ModuleList([TransformerLayer(d_model, d_model, dff) for _ in range(n_layers)])
    
    # [MYCODE] 마지막 단어를 예측하는 것이므로 총 토큰의 길이를 주도록 설정
    self.classification = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    mask = (x == tokenizer.pad_token_id)
    mask = mask[:, None, :]
    seq_len = x.shape[1]

    x = self.embedding(x)
    x = x * sqrt(self.d_model)
    x = x + self.pos_encoding[:, :seq_len]

    for layer in self.layers:
      x = layer(x, mask)

    x = x[:, -1]
    x = self.classification(x)

    return x


model = TextClassifier(len(tokenizer), 32, 2, 32)

from torch.optim import Adam

lr = 0.001
model = model.to('cuda')

# [MYCODE] 마지막 단어에 대한 예측이기 때문에 다중 분류할 수 있도록 설정
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

import numpy as np
import matplotlib.pyplot as plt

def accuracy(model, dataloader):
  cnt = 0
  acc = 0

  for data in dataloader:
    inputs, labels = data
    inputs, labels = inputs.to('cuda'), labels.to('cuda')

    preds = model(inputs)
    preds = torch.argmax(preds, dim=-1)
    # preds = (preds > 0).long()[..., 0]

    cnt += labels.shape[0]
    acc += (labels == preds).sum().item()

  return acc / cnt

n_epochs = 50

for epoch in range(n_epochs):
  total_loss = 0.
  model.train()
  for data in train_loader:
    model.zero_grad()
    inputs, labels = data
    inputs, labels = inputs.to('cuda'), labels.to('cuda').float()

    preds = model(inputs)[..., 0]
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  print(f"Epoch {epoch:3d} | Train Loss: {total_loss}")

  with torch.no_grad():
    model.eval()
    train_acc = accuracy(model, train_loader)
    test_acc = accuracy(model, test_loader)
    print(f"=========> Train acc: {train_acc:.3f} | Test acc: {test_acc:.3f}")