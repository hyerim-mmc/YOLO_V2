import torch


pth = torch.load("./dataset/Darknet19.pth", map_location="cpu")
print(pth)
