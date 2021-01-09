import torch

# xIn = torch.randn(500, 784)
xIn = torch.randn(6, 64)

weight = torch.randn(3, 3)

xIn_re = xIn.view(-1, 3 , 8 , 8)

print(xIn)

print("re")

print(xIn_re)
