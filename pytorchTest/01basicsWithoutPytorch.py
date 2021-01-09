import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

weight = torch.randn(784, 10)
weight.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
        return log_softmax(xb @ weight + bias)

bs = 64

xb = x_train[0: bs]
preds = model(xb)

# z = preds[range(y_train.shape[0]), y_train]
# z = range(y_train.shape[0])
# z = range(10)
# print(z)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))

def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

print(accuracy(preds, yb))

