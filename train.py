import torch
import numpy as np

from torch.utils.data import DataLoader


class Yolov2_train:
    def __init__(self, epoch=160, lr=0.001, weight_decay=0.0005, momentum=0.9):
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

    def decay_lr(self):
        pass

    def run(self):
        pass

