import torch
import utils
import loss
import torch.nn as nn
import numpy as np
import network
import utils

from network import Yolov2
from dataset import VOCDataset
from torch.utils.data import DataLoader


class Yolov2_train:
    def __init__(
        self,
        batch_size=64,
#         epoch=160,
        epoch=10,
        lr=0.1,
        device="cpu",
        weight_decay=0.0005,
        momentum=0.9,
        division=1,
        burn_in=1000,
    ):
        self.batch_size = batch_size
        self.mini_batch_size = int(self.batch_size / division)
        self.epoch = epoch
        self.lr = lr
        self.device = torch.device(device)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.division = division
        self.burn_in = burn_in


        self.train_dataset = DataLoader(VOCDataset(), batch_size=self.mini_batch_size, shuffle=True, num_workers=8)
        self.val_dataset = DataLoader(
            VOCDataset(train_mode=False), batch_size=self.mini_batch_size, shuffle=True, num_workers=8
        )
        self.model = network.Yolov2().to(self.device)
        self.log_path = "./dataset/tensorboard/"

        param = {}
        param["name"] = "sgd"
        param["lr"] = lr
        param["weight_decay"] = weight_decay
        param["momentum"] = momentum

        self.optimizer = utils.get_optim(param, self.model)
        self.train_dataset = DataLoader(VOCDataset(), batch_size=self.mini_batch_size, shuffle=True)
        self.val_dataset = DataLoader(VOCDataset(train_mode=False), batch_size=self.mini_batch_size, shuffle=True)
        self.model = Yolov2(pretrained=True).to(self.device)

        param = {}
        param["name"] = "sgd"
        param["lr"] = self.lr
        param["weight_decay"] = self.weight_decay
        param["momentum"] = self.momentum

        self.optimizer = utils.get_optim(param, self.model)
        self.criterion = criterion

    def decay_lr(self, step, epoch):
        if (epoch == 0) and (step <= self.burn_in):
            power = 4
            lr = 1e-3 * (step / self.burn_in) ** power
            for param in self.optimizer.param_groups:
                param["lr"] = lr

        if (epoch + 1 == 60) or (epoch + 1 == 90):
            for param in self.optimizer.param_groups:
                param["lr"] /= 10

    def run(self):
        pass


if __name__ == "__main__":
    yolov2 = Yolov2_train()
    yolov2.run()
