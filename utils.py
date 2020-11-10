import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms


def optim(param, model):
    if param["name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=param["lr"],
            momentum=param["momentum"],
            weight_decay=param["weight_decay"],
        )
    elif param["name"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param["lr"],
            weight_decay=param["weight_decay"],
            eps=param["eps"],
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


def load_npy(path):
    file = np.load(path)
    print("%s : " % path)
    print(file)


def calc_IOU(self, box):
    w_min = np.minimum(box[0], self.centroid[:, 0])
    h_min = np.minimum(box[1], self.centroid[:, 1])

    overlap = w_min * h_min
    iou = overlap / (box[0] * box[1] + self.centroid[:, 0] * self.centroid[:, 1] - overlap)
    idx = np.argmax(iou)

    return idx
