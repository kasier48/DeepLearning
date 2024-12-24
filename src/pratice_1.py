import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device('cuda')

def accuracy(model, dataloader):
  cnt = 0
  acc = 0

  for data in dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    preds = model(inputs)
    preds = torch.argmax(preds, dim=-1)

    cnt += labels.shape[0]
    acc += (labels == preds).sum().item()

  return acc / cnt

def plot_acc(train_accs, test_accs, label1='train', label2='test'):
  x = np.arange(len(train_accs))

  plt.plot(x, train_accs, label=label1)
  plt.plot(x, test_accs, label=label2)
  plt.legend()
  plt.show()

# [MYCODE] 데이터 전처리: 이미지를 텐서로 변환하고 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST의 평균과 표준편차 적용
])

# [MYCODE] MNIST 데이터셋 불러오기
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"train_dataset len: {len(train_dataset)}")
print(f"train size: {train_dataset[0][0].shape}")
plt.imshow(train_dataset[0][0][0], cmap='gray')

batch_size = 256

# [MYCODE] train, test에 대한 배치 사이즈에 따른 dataLoader 정의
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self, input_dim, n_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)

        # [MYCODE] 0 ~ 9의 이미지를 분류하기 때문에 10으로 설정
        self.fc3 = nn.Linear(n_dim, 10)

        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model(1 * 28 * 28, 1024).to(device)

lr = 0.001

# [MYCODE] 다중 클래스 분류에서 CrossEntropyLoss 사용
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# [MYCODE] 정확도를 매 epoch 타이밍 때 마다 리스트에 저장
train_acc_list = []
test_acc_list = []
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch: {epoch:3d} | Lost: {running_loss}")

    train_acc = accuracy(model, dataloader=train_loader)
    train_acc_list.append(train_acc)

    test_acc = accuracy(model, dataloader=test_loader)
    test_acc_list.append(test_acc)
    
plot_acc(train_accs=train_acc_list, test_accs=test_acc_list)
     