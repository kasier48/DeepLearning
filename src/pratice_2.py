import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device('cuda')

def plot_acc(first_accs, second_accs, label1='train', label2='test'):
  x = np.arange(len(first_accs))

  plt.plot(x, first_accs, label=label1)
  plt.plot(x, second_accs, label=label2)
  plt.legend()
  plt.show()

# [MYCODE] 데이터 전처리: 이미지를 텐서로 변환하고 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print(f"train_dataset len: {len(train_dataset)}")
print(f"train size: {train_dataset[0][0].shape}")
plt.imshow(train_dataset[0][0][0], cmap='gray')

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# [MYCODE] 10개의 이미지를 분류할 수 있는 모델 생성
class Model(nn.Module):
    def __init__(self, input_dim, n_dim, criterion, act, dropoutProbability = 0.0):
        super(Model, self).__init__()
        
        self.criterion = criterion
        
        self.fc1 = nn.Linear(input_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.fc3 = nn.Linear(n_dim, 10)  # 10개의 클래스를 분류
    
        self.dropout = nn.Dropout(p=dropoutProbability)
        self.act = act

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        
        x = self.act(self.fc1(x))
        if self.dropout.p > 0.0:
            x = self.dropout(x)
            
        x = self.act(self.fc2(x))
        if (self.dropout.p > 0.0):
            x = self.dropout(x)
            
        x = self.fc3(x)
        return x
    
    def learn(self, num_epochs, train_loader, optimizer):
        acc_list = []
        
        for epoch in range(num_epochs):
            running_loss = self.__learn_internal(optimizer=optimizer, train_loader=train_loader)
            print(f"Epoch: {epoch:3d} | Lost: {running_loss}")
            
            acc = self.__accuracy(dataloader=train_loader)
            acc_list.append(acc)
            
        return acc_list
        
    def learn_with_test(self, num_epochs, optimizer, train_loader, test_loader):
        train_acc_list = []
        test_acc_list = []
        
        for epoch in range(num_epochs):
            running_loss = self.__learn_internal(optimizer=optimizer, train_loader=train_loader)
            print(f"Epoch: {epoch:3d} | Lost: {running_loss}")
            
            train_acc = self.__accuracy(dataloader=train_loader)
            train_acc_list.append(train_acc)
            
            test_acc = self.__accuracy(dataloader=test_loader)
            test_acc_list.append(test_acc)
            
        return (train_acc_list, test_acc_list)

    def __learn_internal(self, optimizer, train_loader):
        running_loss = 0.0
        if self.dropout.p > 0.0:
            self.train()
            
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
            
        return running_loss
            
    def __accuracy(self, dataloader):
        cnt = 0
        acc = 0

        if self.dropout.p > 0.0:
           self.dropout.eval()
           
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                preds = self(inputs)
                preds = torch.argmax(preds, dim=-1)

                cnt += labels.shape[0]
                acc += (labels == preds).sum().item()

        return acc / cnt

lr = 0.001
num_epochs = 50
criterion = nn.CrossEntropyLoss()
input_dim = 3 * 32 * 32
act = nn.ReLU()

# [MYCODE] SGD, Adam 간의 정확도 비교
model = Model(input_dim=input_dim, n_dim=1024, criterion=criterion, act=act).to(device)

optimizer = optim.SGD(params=model.parameters(), lr=lr)
sgd_acc_list = model.learn(num_epochs=num_epochs, train_loader=train_loader, optimizer=optimizer)

optimizer = optim.Adam(params=model.parameters(), lr=lr)
adam_acc_list = model.learn(num_epochs=num_epochs, train_loader=train_loader, optimizer=optimizer)

plot_acc(first_accs=sgd_acc_list, second_accs=adam_acc_list, label1="SGD", label2="Adam")

# [MYCODE] 활성화 함수 LeakyReLU, Sigmoid 간의 정확도 비교
act = nn.LeakyReLU()
model = Model(input_dim=input_dim, n_dim=1024, criterion=criterion, act=act).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
leakyReLU_acc_list = model.learn(num_epochs=num_epochs, train_loader=train_loader, optimizer=optimizer)

act = nn.Sigmoid()
model = Model(input_dim=input_dim, n_dim=1024, criterion=criterion, act=act).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
sigmoid_acc_list = model.learn(num_epochs=num_epochs, train_loader=train_loader, optimizer=optimizer)

plot_acc(first_accs=leakyReLU_acc_list, second_accs=sigmoid_acc_list, label1="LeakyReLU", label2="Sigmoid")

# [MYCODE] dropout 적용한 train, test 데이터간의 정확도 비교
act = nn.LeakyReLU()
model = Model(input_dim=input_dim, n_dim=1024, criterion=criterion, act=act, dropoutProbability=0.1).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
train_acc_list, test_acc_list = model.learn_with_test(num_epochs=num_epochs, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader)

plot_acc(first_accs=train_acc_list, second_accs=test_acc_list, label1="Train", label2="Test")

        

