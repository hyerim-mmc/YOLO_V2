import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms


def show_image(data):
    image, annotation = data["image"], data["annotation"]

    if torch.is_tensor(image):
        image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    bboxes = annotation["bbox"]

    for bbox in bboxes:
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="red", width=2)
        # draw.text((bbox[0], bbpx[1]), name)

    image.show()
