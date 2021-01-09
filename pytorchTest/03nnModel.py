import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
import requests
import gzip
import pickle
import math
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Minst_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


# def loss_func(pred, target):
#     return -pred[range(target.shape[0]), target].mean()

lr = 0.5

def get_model():
    model = Minst_Logistic()
    opt = optim.SGD(model.parameters(), lr=lr)
    return model, opt

# input data
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

x_train, y_train, x_valid, y_valid = map(torch.tensor,(x_train, y_train, x_valid, y_valid))

bs = 10000
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

# start training

# model = Minst_Logistic()

model, opt = get_model()

loss_func = F.cross_entropy

def accuracy(input, out):
    input = torch.argmax(input)
    return (input == out).float().mean()


bs = 5000
epochs = 2

def fit():
    for epoch in range(epochs):
        model.train()
        print("train_loss")
        # for i in range(x_train.shape[0] // bs):
        for xb, yb in train_dl:
            # xb = x_train[i*bs: i*bs+bs]
            # yb = y_train[i*bs: i*bs+bs]
            # xb, yb = train_ds[i*bs: i*bs+bs]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            # with torch.no_grad():
            #     for p in model.parameters():
            #         p -= p.grad * lr

            opt.step()
            opt.zero_grad()
            # model.zero_grad()

            print(loss)

        model.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                valid_loss = loss_func(model(xb), yb).mean()
                print("valid_loss")
                print(valid_loss)


fit()

# print(loss)