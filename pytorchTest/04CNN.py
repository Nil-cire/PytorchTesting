import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from pathlib import Path
import requests
import gzip
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

path = Path('data')
dataPath = path / 'mnist'
dataPath.mkdir(parents=True, exist_ok=True)
url = "http://deeplearning.net/data/mnist/"
fileName = "mnist.pkl.gz"

if not (dataPath / fileName).exists():
    content = requests.get(url + fileName).content
    (dataPath / fileName).open('wb').write(content)

with gzip.open((dataPath / fileName).as_posix(), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid),_) = pickle.load(f, encoding='latin-1')

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

lr = 0.1

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr= lr, momentum=0.9)
loss_func = F.cross_entropy

bs = 1000
train_dt = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_dt, batch_size=bs, shuffle=True)
valid_dt = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_dt, batch_size=bs*2)

epochs = 1


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for i in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()

            opt.step()
            opt.zero_grad()

        model.eval()
        for xb, yb in valid_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            print(loss)


fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# print(x_train.shape)

