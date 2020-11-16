import torch
import torch.nn as nn
import numpy as np
import network
import utils

from dataset import VOCDataset
from torch.utils.data import DataLoader


class Yolov2_train:
    def __init__(
        self,
        batch_size=64,
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

    def decay_lr(self):
        pass

    def run(self):
        pass

