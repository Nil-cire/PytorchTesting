from pathlib import Path
import os
import torch

PATH = Path('data')
path2 = PATH / 'mnist'
print(path2)
print(path2.as_posix())
# print([x for x in PATH.iterdir()])

# print(os.getcwd())

# print(torch.zeros(10))

