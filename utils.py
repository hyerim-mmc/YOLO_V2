import sys
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def load_pth(pth_path="./dataset/Darknet19/Darknet19.pth", save_path="darknet19.txt"):
    sys.stdout(save_path, "w")
    pth = torch.load(pth_path, map_location="cpu")
    print(pth)


def tensorboard(log_path, results, step):
    writer = SummaryWriter(log_path)
    loss, val_loss, train_precision, val_precision = results
    writer.add_scalars("Loss/Loss", loss, step)
    writer.add_scalars("Loss/Validation Loss", val_loss, step)
    writer.add_scalars("Precision/Train Precision", train_precision, step)
    writer.add_scalars("Precision/Validation Precision", val_precision, step)


def get_optim(param, model):
    if param["name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=param["lr"], momentum=param["momentum"], weight_decay=param["weight_decay"],
        )
    elif param["name"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=param["lr"], weight_decay=param["weight_decay"], eps=param["eps"],
        )
    return optimizer


def show_image(data):
    image, annotation = data["image"], data["annotation"]

    if torch.is_tensor(image):
        image = transforms.ToPILImage()(image)

    draw = ImageDraw.Draw(image)
    bboxes = annotation["bbox"]
    for bbox in bboxes:
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="red", width=2)
        # draw.text((bbox[0], bbox[1]), name)

    image.show()


def calc_IOU(self, pred, truth):
    w_min = np.minimum(pred[0], truth[0])
    h_min = np.minimum(pred[1], truth[1])

    overlap = w_min * h_min
    iou = overlap / (pred[0] * pred[1] + truth[0] * truth[1] - overlap)

    return iou
