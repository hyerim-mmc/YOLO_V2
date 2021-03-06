import torch
import utils
import numpy as np
import torch.nn as nn
from dataset import VOCDataset
from dataset import ImageNetDataset
from torch.utils.data import DataLoader


def conv_net(input, output, kernel_size=3, padding=1, stride=1, eps=1e-5, momentum=0.9, negative_slope=0.01):
    conv = nn.Sequential(
        nn.Conv2d(input, output, kernel_size=kernel_size, padding=padding, stride=stride, bias=False,),
        nn.BatchNorm2d(output, eps=eps, momentum=momentum),
        nn.LeakyReLU(negative_slope=negative_slope),
    )
    return conv


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride

        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))

        return x


class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = conv_net(3, 32)
        self.conv2 = conv_net(32, 64)
        self.conv3 = conv_net(64, 128)
        self.conv4 = conv_net(128, 64, kernel_size=1, padding=0)
        self.conv5 = conv_net(64, 128)
        self.conv6 = conv_net(128, 256)
        self.conv7 = conv_net(256, 128, kernel_size=1, padding=0)
        self.conv8 = conv_net(128, 256)
        self.conv9 = conv_net(256, 512)
        self.conv10 = conv_net(512, 256, kernel_size=1, padding=0)
        self.conv11 = conv_net(256, 512)
        self.conv12 = conv_net(512, 256, kernel_size=1, padding=0)
        self.conv13 = conv_net(256, 512)
        self.conv14 = conv_net(512, 1024)
        self.conv15 = conv_net(1024, 512, kernel_size=1, padding=0)
        self.conv16 = conv_net(512, 1024)
        self.conv17 = conv_net(1024, 512, kernel_size=1, padding=0)
        self.conv18 = conv_net(512, 1024)
        self.conv19 = nn.Conv2d(1024, 1000, kernel_size=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        # block1
        x1 = self.conv1(x)
        # block2
        x2 = self.maxpool(x1)
        x3 = self.conv2(x2)
        # block3
        x4 = self.maxpool(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        # block4
        x8 = self.maxpool(x7)
        x9 = self.conv6(x8)
        x10 = self.conv7(x9)
        x11 = self.conv8(x10)
        # block5
        x12 = self.maxpool(x11)
        x13 = self.conv9(x12)
        x14 = self.conv10(x13)
        x15 = self.conv11(x14)
        x16 = self.conv12(x15)
        x17 = self.conv13(x16)
        # block6
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
        batch_size=128,
        epoch=10,
        lr=0.1,
        device="cpu",
        weight_decay=0.0005,
        momentum=0.9,
        division=2,
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

        self.train_dataset = DataLoader(ImageNetDataset(), batch_size=self.mini_batch_size, shuffle=True, num_workers=8)
        self.val_dataset = DataLoader(ImageNetDataset(val_mode=True), batch_size=1, shuffle=True, num_workers=8)
        self.model = Darknet19().to(self.device)

        param = {}
        param["name"] = "sgd"
        param["lr"] = lr
        param["weight_decay"] = weight_decay
        param["momentum"] = momentum

        self.optimizer = utils.get_optim(param, self.model)
        self.criterion = nn.CrossEntropyLoss()

    def decay_lr(self, step, epoch):
        if (epoch == 0) and (step <= self.burn_in):
            power = 4
            lr = 1e-3 * (step / self.burn_in) ** power
            for param in self.optimizer.param_groups:
                param["lr"] = lr

    def run(self):
        step = 1
        print_size = 100
        for epoch in range(self.epoch):
            divi = 0
            Loss, Val_Loss, Train_Precision, Val_Precision = [], [], [], []

            for data in self.train_dataset:
                # train mode
                self.model.train()
                image, annotation = data[0].to(self.device), data[1].to(self.device)

                hypothesis = self.model.forward(image)
                loss = self.criterion(hypothesis, annotation)
                loss.backward()
                divi += 1

                # mini batch is over
                if divi == self.division:
                    self.decay_lr(step, epoch)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    divi = 0
                    step += 1

                # calculate precision for train_dataset
                with torch.no_grad():
                    Loss.append(loss.detach().cpu().numpy())
                    idx = torch.argmax(hypothesis, dim=1)
                    total = len(annotation)
                    total_correct = (idx == annotation).float().sum()
                    train_precision = total_correct / total
                    Train_Precision.append(
                        train_precision.detach().cpu().numpy() if torch.cuda.is_available() else train_precision.numpy()
                    )

                if step % print_size == 0 and divi == 0:
                    with torch.no_grad():
                        # eval mode
                        self.model.eval()
                        k = 0
                        for val_data in self.val_dataset:
                            val_image, val_annotation = (
                                val_data[0].to(self.device),
                                val_data[1].to(self.device),
                            )

                            val_hypothesis = self.model.forward(val_image)
                            val_loss = self.criterion(val_hypothesis, val_annotation)

                            # calculate precision for val_dataset
                            Val_Loss.append(val_loss.detach().cpu().numpy())
                            idx = torch.argmax(val_hypothesis, dim=1)
                            total = len(val_annotation)
                            total_correct = (idx == val_annotation).float().sum()
                            val_precision = total_correct / total

                            Val_Precision.append(
                                val_precision.detach().cpu().numpy()
                                if torch.cuda.is_available()
                                else val_precision.numpy()
                            )
                            k += 1
                            if k == 10:
                                break

                    loss = np.array(Loss).mean()
                    val_loss = np.array(Val_Loss).mean()
                    train_precision = np.array(Train_Precision).mean()
                    val_precision = np.array(Val_Precision).mean()

                    print(
                        "Epoch: {}/{} | Step: {} | Loss: {:.5f} | Val_Loss: {:.5f} | Train_Precision: {:.4f} | Val_Precision: {:.4f}".format(
                            epoch + 1, self.epoch, step, loss, val_loss, train_precision, val_precision,
                        )
                    )

                    Loss, Val_Loss, Train_Precision, Val_Precision = [], [], [], []

            save_path = "./dataset/Darknet19/epoch_{0}.pth".format(epoch + 1)
            torch.save(self.model.state_dict(), save_path)

        save_path = "./dataset/Darknet19/Darknet19.pth"
        torch.save(self.model.state_dict(), save_path)


class Yolov2(nn.Module):
    def __init__(self, n_bbox=5, n_class=20, device="cpu", pretrained=True):
        super().__init__()
        self.device = torch.device(device)
        self.n_bbox = n_bbox
        self.n_class = n_class

        if pretrained:
            darknet19 = Darknet19()
            print("Load pretrained Darknet19 model...")
            darknet19.load_state_dict(torch.load("./dataset/Darknet19/Darknet19.pth", map_location=self.device))

            # state_dict parsing
            block = []
            temp = []
            pretrain = []
            k = 0
            for layer in darknet19.children():
                if k == 0:
                    maxpool = layer
                else:
                    temp = []
                    temp.append(layer)
                    block = nn.Sequential(*list(temp))
                    pretrain.append(block)
                k += 1

            k = [1, 3, 7, 11, 17]
            for idx in k:
                pretrain.insert(idx, maxpool)
        else:
            print("There is no pretrained model!")

        self.pretrain1 = nn.Sequential(*list(pretrain)[:17]).to(self.device)
        self.pretrain2 = nn.Sequential(*list(pretrain)[17:-2]).to(self.device)

        self.conv1 = conv_net(1024, 1024)
        self.conv2 = conv_net(1024, 1024)
        self.conv3 = conv_net(512, 64, kernel_size=1, padding=0)
        self.reorg = Reorg()
        self.conv4 = conv_net(1024 + 256, 1024)
        self.conv5 = nn.Conv2d(1024, self.n_bbox * (5 + self.n_class), kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.pretrain1(x)
        x2 = self.pretrain2(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        pass_through = self.conv3(x1)
        pass_through = self.reorg(pass_through)
        x2 = torch.cat((pass_through, x2), dim=1)
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        return x2


if __name__ == "__main__":
    # darknet19 = Pretrain_model(device="cuda:2")
    # darknet19.run()
    yolov2 = Yolov2(pretrained=True)
