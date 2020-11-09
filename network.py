import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
from dataset import VOCDataset
from dataset import ImageNetDataset
from torch.utils.data import DataLoader

def layer(input,output,kernel_size=3,padding=1,stride=1,eps=1e-5,momentum=0.9,negative_slope=0.01):
    set = nn.Sequential(
        nn.Conv2d(input,output,kernel_size=kernel_size,padding=padding,stride=stride,bias=False),
        nn.BatchNorm2d(output,eps=eps,momentum=momentum),
        nn.LeakyReLU(negative_slope=negative_slope)
    )
    return set

class Darknet19:
    def __init__(self, batch_size=64, epoch=10, learning_rate=0.1, weight_decay=0.0005, momentum=0.9,division=1, burn_in=1000, load_path=None):
        self.batch_size = batch_size
        self.mini_batch_size = int(self.batch_size / division)
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.division = division
        self.burn_in = burn_in
        self.load_path = load_path
        self.train_dataset = DataLoader(ImageNetDataset, batch_size=self.mini_batch_size, shuffle=True)
        self.val_dataset = DataLoader(ImageNetDataset, shuffle=True)
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(),lr= learning_rate,weight_decay=weight_decay,momentum=momentum)
        self.criterion

    def run(self):
        print(
                "Epoch {:4d}/{} Batch {}/{} Cost : {:.6f}".format(
                    epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
                )
            )
