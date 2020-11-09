import torch
import torch.nn as nn
import numpy as np

from dataset import VOCDataset
from dataset import ImageNetDataset
from torch.utils.data import DataLoader


class Darknet19:
    def __init__(self, batch_size, epoch, learning_rate, weight_decay, momentum, device, division, burn_in, load_path):
        self.batch_size = batch_size
        self.minibatch_size = int(self.batch_size / division)
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device
        self.division = division
        self.burn_in = burn_in
        self.load_path = load_path
        self.train_dataset = DataLoader(ImageNetDataset, batch_size=self.minibatch_size, shuffle=True)
        self.val_dataset = DataLoader(ImageNetDataset, shuffle=True)

        self.optimizer
        self.criterion

    def run(self):
        # 퍼옴
        nb_epochs = 20
        for epoch in range(nb_epochs + 1):
            for batch_idx, samples in enumerate(self.train_dataset):
                x_train, y_train = samples

            prediction = model(x_train)
            cost = F.mse_loss(prediction, y_train)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print(
                "Epoch {:4d}/{} Batch {}/{} Cost : {:.6f}".format(
                    epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
                )
            )
