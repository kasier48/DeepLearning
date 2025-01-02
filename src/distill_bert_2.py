import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'distilbert-base-uncased')

ds = load_dataset("glue", "mnli")
train_dataset = ds['train']
print(f"Train 데이터 확인: {train_dataset[0]}")

label_info = train_dataset.features['label']
print(f"라벨의 종류: {label_info.names}")

num_labels = len(label_info.names)

dataset_length = 10000
train_dataset = train_dataset.select(range(dataset_length))

validation_matched_dataset = ds['validation_matched']

test_dataset = ds['test_matched']

def collate_fn(batch):
  max_len = 400
  texts, labels = [], []
  for row in batch:
    labels.append(row['label'])
    
    # [MYCODE] 전제와 가설 문장을 합체
    texts.append(row['premise'] + " " + row['hypothesis'])

  encoding = tokenizer(texts, padding=True, truncation=False, max_length=max_len, return_tensors="pt")
  labels = torch.LongTensor(labels)

  return encoding.input_ids, labels, encoding.attention_mask

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
)
validation_loader = DataLoader(
    validation_matched_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
)

device = torch.device('cuda')

def train_model(model, optimizer, n_epochs):
  loss_list = []
  train_acc_list = []
  validation_acc_list = []
  
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
      
    loss_list.append(total_loss)
    print(f"Epoch {epoch:3d} | Train Loss: {total_loss}")

    with torch.no_grad():
      model.eval()
      train_acc = accuracy(model, train_loader)
      train_acc_list.append(train_acc)
      
      validation_acc = accuracy(model, validation_loader)
      validation_acc_list.append(validation_acc)
      print(f"=========> Train acc: {train_acc:.3f} | Validation acc: {validation_acc:.3f}")
      
  
  return (loss_list, train_acc_list, validation_acc_list)
  
def accuracy(model, dataloader):
  cnt = 0
  acc = 0

  for data in dataloader:
    inputs, labels, attention_mask = data
    inputs, labels, attention_mask = inputs.to(device), labels.to(device), attention_mask.to(device)

    preds = model(inputs, attention_mask)
    preds = torch.argmax(preds, dim=-1)

    cnt += labels.shape[0]
    acc += (labels == preds).sum().item()

  return acc / cnt

from torch import nn

class FineTunningTextClassifier(nn.Module):
  def __init__(self, num_labels, dropout_rate):
    super(FineTunningTextClassifier, self).__init__()

    self.encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', 'distilbert-base-uncased')
    self.dropout = nn.Dropout(dropout_rate)
    self.classifier = nn.Linear(768, num_labels)

  def forward(self, x, attention_mask):
    x = self.encoder(x, attention_mask)['last_hidden_state']
    x = self.dropout(x[:, 0])
    x = self.classifier(x)
    return x

from torch.optim import Adam
from torch.optim import AdamW
import numpy as np

fine_tunning_model = FineTunningTextClassifier(num_labels, dropout_rate=0.1)

lr = 1e-5
fine_tunning_model = fine_tunning_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(fine_tunning_model.parameters(), lr=lr, weight_decay=0.01)
# optimizer = Adam(fine_tunning_model.parameters(), lr=lr)
n_epochs = 10

fine_tunning_result = train_model(fine_tunning_model, optimizer, n_epochs)

# =============================================================================================>

from transformers import DistilBertModel, DistilBertConfig

# [MYCODE] Non-traiend 되지 않은 모델 정의
class NonTrainedTextClassifier(nn.Module):
    def __init__(self, num_labels, config, dropout_rate):
        super(NonTrainedTextClassifier, self).__init__()
        self.encoder = DistilBertModel(config)  # 사전 학습되지 않은 DistilBERT
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x, attention_mask):
        x = self.encoder(x, attention_mask)['last_hidden_state']
        x = self.dropout(x[:, 0])
        x = self.classifier(x)
        return x

non_trained_config = DistilBertConfig()
non_traiend_text_classifier = NonTrainedTextClassifier(num_labels, non_trained_config, dropout_rate=0.1)
non_traiend_text_classifier = non_traiend_text_classifier.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(fine_tunning_model.parameters(), lr=lr, weight_decay=0.01)

non_trained_result = train_model(non_traiend_text_classifier, optimizer, n_epochs)

import matplotlib.pyplot as plt

def plot_acc(first_accs, second_accs, label1='train', label2='test'):
  x = np.arange(len(first_accs))

  plt.plot(x, first_accs, label=label1)
  plt.plot(x, second_accs, label=label2)
  plt.legend()
  plt.show()
  
fine_tunning_train_acc_list = fine_tunning_result[1]
fine_tunning_validation_acc_list = fine_tunning_result[2]
plot_acc(fine_tunning_train_acc_list, fine_tunning_validation_acc_list, label1="Fine-Tunning-Acc", label2="Validation-Acc")

non_trained_train_acc_list = non_trained_result[1]
non_traiend_validation_acc_list = non_trained_result[2]
plot_acc(non_trained_train_acc_list, non_traiend_validation_acc_list, label1="Non-Train-Acc", label2="Validation-Acc")

def plot_loss(n_epochs, model1, model2, mode1_label, model2_label):
  # 손실 그래프 그리기
  plt.figure(figsize=(10, 6))
  plt.plot(range(n_epochs), model1, label=mode1_label, color='blue')
  plt.plot(range(n_epochs), model2, label=model2_label, color='red')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training Loss Comparison between Model 1 and Model 2')
  plt.legend()
  plt.show()

fine_tunning_loss_list = fine_tunning_result[0]
non_trained_loss_list = non_trained_result[0]
plot_loss(n_epochs, fine_tunning_loss_list, non_trained_loss_list, mode1_label="Fine-Tunning-Loss", model2_label="Non-Train-Loss")

d = 1
