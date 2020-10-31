import os
import random
import torch
import numpy as np
import utils
import xml.etree.ElementTree as Et
import torch.utils.data.dataset
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image


class Resize:
    def __init__(self, output_size, scale_factor=1.15):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.scale_factor = scale_factor

    def __call__(self, data):
        image, annotation = data
        width, height = image.size

        if height > width:
            new_h, new_w = self.output_size * height / width, self.output_size
        elif height == width:
            new_h, new_w = height, width
        else:
            new_h, new_w = self.output_size, self.output_size * width / height

        new_h, new_w = int(new_h * self.scale_factor), int(new_w * self.scale_factor)
        new_image = TF.resize(image, (new_h, new_w), Image.BILINEAR)

        bboxes = annotation["bbox"]
        for idx in range(len(bboxes)):
            _bbox = bboxes[idx]
            _bbox[0] *= new_w / width
            _bbox[1] *= new_h / height
            _bbox[2] *= new_w / width
            _bbox[3] *= new_h / height

        return (new_image, annotation)


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        image, annotation = data
        width, height = image.size
        rand_w = random.randint(0, width - self.output_size)
        rand_h = random.randint(0, height - self.output_size)

        crop_image = TF.crop(image, rand_h, rand_w, self.output_size, self.output_size)

        bboxes = annotation["bbox"]
        check = torch.ones((len(bboxes)))
        for idx in range(len(bboxes)):
            _bbox = bboxes[idx]
            _bbox[0] -= rand_w
            _bbox[1] -= rand_h
            _bbox[2] -= rand_w
            _bbox[3] -= rand_h

            center_x, center_y = int((_bbox[0] + _bbox[2]) / 2), int((_bbox[1] + _bbox[3]) / 2)

            if _bbox[0] < 0:
                _bbox[0] = 1
            if _bbox[1] < 0:
                _bbox[1] = 1
            if _bbox[2] > self.output_size:
                _bbox[2] = self.output_size - 1
            if _bbox[3] > self.output_size:
                _bbox[3] = self.output_size - 1
            if center_x < 0 or center_x > self.output_size:
                check[idx] = 0
            if center_y < 0 or center_y > self.output_size:
                check[idx] = 0

        check = check == 1
        annotation["bbox"] = bboxes[check]
        annotation["label"] = annotation["label"][check]

        print("# of bbox : %d\n" % len(annotation["bbox"]))

        return (crop_image, annotation)


class VOC_DataLoad(torch.utils.data.Dataset):
    def __init__(self, output_size=416, train=True):
        if os.path.exists("./dataset/train.npy") and os.path.exists("./dataset/val.npy"):
            self.train_set = np.load("./dataset/train.npy")
            self.test_set = np.load("./dataset/test.npy")
        else:
            self.load_dataset()

        self.train = train
        self.resize_bd = Resize(output_size=output_size)
        self.crop_bd = RandomCrop(output_size=output_size)
        self.transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.75, hue=0.1, saturation=0.75),
                transforms.ToTensor(),
            ]
        )
        self.val_transform = transforms.Compose([transforms.ToTensor()])

        self.category = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def __len__(self):
        return len(self.data_set)

    def load_dataset(self):
        train_path = "./PASCAL_VOC_2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
        validate_path = "./PASCAL_VOC_2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt"

        self.train_files = np.loadtxt(train_path, delimiter="\n", dtype="str")
        self.validate_files = np.loadtxt(validate_path, delimiter="\n", dtype="str")
        self.data_set = np.concatenate((self.train_files, self.validate_files))

        random.shuffle(self.data_set)
        split = int(np.round(0.2 * len(self.data_set)))

        self.train_set, self.test_set = self.data_set[split:], self.data_set[:split]

        np.save("dataset/train.npy", self.train_set)
        np.save("dataset/test.npy", self.test_set)

    def __getitem__(self, idx):
        """
        :return: { 'image' : image, 
                   'annotation' : {
                                'label' : [label_index]
                                'bbox' : [[xmin, ymin, xmax, ymax], [...]]
                                }
                }
        """
        if self.train:
            self.image_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages/" + "{0}.jpg".format(
                self.train_set[idx]
            )
            self.xml_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/" + "{0}.xml".format(
                self.train_set[idx]
            )

        else:
            self.image_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages/" + "{0}.jpg".format(
                self.test_set[idx]
            )
            self.xml_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/" + "{0}.xml".format(
                self.test_set[idx]
            )

        img_name = self.image_path
        xml_name = self.xml_path

        image = Image.open(img_name)
        xml = open(xml_name, "r")
        tree = Et.parse(xml)
        root = tree.getroot()

        objects = root.findall("object")
        annotation = {}
        labels = []
        bboxes = []

        for _object in objects:
            name = _object.find("name").text
            label = self.category.index(name)
            labels.append(label)

            bndbox = _object.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            bboxes.append([xmin, ymin, xmax, ymax])

        annotation["label"] = torch.tensor(labels).float()
        annotation["bbox"] = torch.tensor(bboxes).float()

        if self.train:
            image, annotation = self.resize_bd((image, annotation))
            image, annotation = self.crop_bd((image, annotation))
            image = self.transform(image)
        else:
            # need to check
            image, annotation = self.resize_bd((image, annotation))
            image, annotation = self.crop_bd((image, annotation))
            image = self.val_transform(image)

        return {"image": image, "annotation": annotation}


if __name__ == "__main__":
    VOC_dataset = VOC_DataLoad()

    for idx in range(VOC_dataset.__len__()):
        print("#%d image" % idx)
        utils.show_image(VOC_dataset.__getitem__(idx))
    # print(VOC_dataset.__getitem__(idx))
