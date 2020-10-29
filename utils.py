from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
import numpy as np


def show_image_raw(data):
    img_file, annotation = data["image"], data["annotation"]
    draw = ImageDraw.Draw(img_file)

    for idx in range(len(annotation)):
        name = annotation[idx][0]
        xmin = annotation[idx][1]
        ymin = annotation[idx][2]
        xmax = annotation[idx][3]
        ymax = annotation[idx][4]

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name)

    img_file.show()


def show_image(data):
    img_file, annotation = data["image"], data["annotation"]
    img_file = transforms.ToPILImage()(img_file)
    annotation = annotation.tolist()
    draw = ImageDraw.Draw(img_file)

    for idx in range(len(annotation)):
        xmin = annotation[idx][0]
        ymin = annotation[idx][1]
        xmax = annotation[idx][2]
        ymax = annotation[idx][3]

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        # draw.text((xmin, ymin), name)

    img_file.show()
