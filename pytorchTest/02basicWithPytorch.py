import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from IPython.core.debugger import set_trace

import torch
from pathlib import Path
import requests
import pickle
import gzip
import math
from matplotlib import pyplot
import numpy as np
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

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

# pyplot.imshow(x_train[0].reshape(28, 28), cmap='gray')
# pyplot.show()
# print(x_train.shape)

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid) )

weights = torch.randn(784, 10, requires_grad=True)
# weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
        return log_softmax(xb @ weights + bias)

bs = 64

# xb = x_train[0: bs]
# preds = model(xb)

# z = preds[range(y_train.shape[0]), y_train]
# z = range(y_train.shape[0])
# z = range(10)
# print(z)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

def accuracy(out, yb):
    pred = torch.argmax(out, dim=1)
    return (pred == yb).float().mean()


lr = 1
epochs = 2

for epoch in range(epochs):
    for i in range(x_train.shape[0] // bs):
        # set_trace()
        xb = x_train[i*bs: i*bs+bs]
        yb = y_train[i*bs: i*bs+bs]
        pred = model(xb)
        loss = loss_func(pred, yb)


        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

