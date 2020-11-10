import torch
import utils
import numpy as np
import torch.nn as nn
from dataset import VOCDataset
from dataset import ImageNetDataset
from torch.utils.data import DataLoader


def conv_layer(
    input, output, kernel_size=3, padding=1, stride=1, eps=1e-5, momentum=0.9, negative_slope=0.01,
):
    conv = nn.Sequential(
        nn.Conv2d(
            input, output, kernel_size=kernel_size, padding=padding, stride=stride, bias=False,
        ),
        nn.BatchNorm2d(output, eps=eps, momentum=momentum),
        nn.LeakyReLU(negative_slope=negative_slope),
    )
    return conv


class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = conv_layer(3, 32)
        self.conv2 = conv_layer(32, 64)
        self.conv3 = conv_layer(64, 128)
        self.conv4 = conv_layer(128, 64, kernel_size=1, padding=0)
        self.conv5 = conv_layer(64, 128)
        self.conv6 = conv_layer(128, 256)
        self.conv7 = conv_layer(256, 128, kernel_size=1, padding=0)
        self.conv8 = conv_layer(128, 256)
        self.conv9 = conv_layer(256, 512)
        self.conv10 = conv_layer(512, 256, kernel_size=1, padding=0)
        self.conv11 = conv_layer(256, 512)
        self.conv12 = conv_layer(512, 256, kernel_size=1, padding=0)
        self.conv13 = conv_layer(256, 512)
        self.conv14 = conv_layer(512, 1024)
        self.conv15 = conv_layer(1024, 512, kernel_size=1, padding=0)
        self.conv16 = conv_layer(512, 1024)
        self.conv17 = conv_layer(1024, 512, kernel_size=1, padding=0)
        self.conv18 = conv_layer(512, 1024)
        self.conv19 = nn.Conv2d(1024, 1000, kernel_size=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        x8 = self.maxpool(x7)
        x9 = self.conv6(x8)
        x10 = self.conv7(x9)
        x11 = self.conv8(x10)
        x12 = self.maxpool(x11)
        x13 = self.conv9(x12)
        x14 = self.conv10(x13)
        x15 = self.conv11(x14)
        x16 = self.conv12(x15)
        x17 = self.conv13(x16)
        x18 = self.maxpool(x17)
        x19 = self.conv14(x18)
        x20 = self.conv15(x19)
        x21 = self.conv16(x20)
        x22 = self.conv17(x21)
        x23 = self.conv18(x22)
        x24 = self.conv19(x23)
        x25 = self.avgpool(x24)
        x25 = x25.view((x.shape[0], 1000))

        return x25


class Pretrain_model:
    def __init__(
        self,
        batch_size=64,
        epoch=10,
        lr=0.1,
        weight_decay=0.0005,
        momentum=0.9,
        division=1,
        burn_in=1000,
        load_path=None,
    ):
        self.batch_size = batch_size
        self.mini_batch_size = int(self.batch_size / division)
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.division = division
        self.burn_in = burn_in
        self.load_path = load_path
        self.train_dataset = DataLoader(
            ImageNetDataset, batch_size=self.mini_batch_size, shuffle=True
        )
        self.val_dataset = DataLoader(ImageNetDataset, shuffle=True)
        self.model = model

        param = {}
        param["name"] = "sgd"
        param["lr"] = lr
        param["weight_decay"] = weight_decay
        param["momentum"] = momentum

        self.optimizer = utils.optim(param, self.model)
        self.criterion = nn.CrossEntropyLoss()

    def decay_lr(self):
        pass

    def run(self):
        print(
            "Epoch {:4d}/{} Batch {}/{} Cost : {:.6f}".format(
                epoch, nb_epochs, batch_idx + 1, len(dataloader), cost.item()
            )
        )
