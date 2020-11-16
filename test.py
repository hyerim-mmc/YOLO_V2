import torch
import torch.nn as nn
import network

darknet19 = network.Darknet19()

darknet19.load_state_dict(torch.load("./dataset/Darknet19/Darknet19.pth", map_location=torch.device("cpu")))

block = []
temp = []
x = []
k = 0
for layer in darknet19.children():
    if k == 0:
        maxpool = layer
    else:
        temp = []
        temp.append(layer)
        block = nn.Sequential(*list(temp))
        x.append(block)
    k += 1

k = [1, 3, 7, 11, 17]
for idx in k:
    x.insert(idx, maxpool)
# print(x)
pretrain1 = nn.Sequential(*list(x)[:17]).to(torch.device("cpu"))
pretrain2 = nn.Sequential(*list(x)[17:-2]).to(torch.device("cpu"))
# print(pretrain2)

