import os
import random
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

    def __call__(self, sample):
        image_path, bbox = sample["image"], sample["annotation"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if height > width:
            new_h, new_w = self.output_size * height / width, self.output_size
        elif height == width:
            new_h, new_w = height, width
        else:
            new_h, new_w = self.output_size, self.output_size * width / height

        new_h, new_w = int(new_h * self.scale_factor), int(new_w * self.scale_factor)
        new_image = TF.resize(image, (new_h, new_w), Image.BILINEAR)

        for idx in range(len(bbox)):
            _bbox = bbox[idx]
            _bbox[0] *= new_w / width
            _bbox[1] *= new_h / height
            _bbox[2] *= new_w / width
            _bbox[3] *= new_h / height

        return {"image": new_image, "annotation": bbox}


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample["image"], sample["annotation"]
        width, height = image.size

        rand_w = random.randint(0, width - self.output_size)
        rand_h = random.randint(0, height - self.output_size)

        crop_image = TF.crop(image, rand_h, rand_w, self.output_size, self.output_size)
        check = []
        temp = False
        for idx in range(len(bbox)):
            _bbox = bbox[idx]

            _bbox[0] -= rand_w
            _bbox[1] -= rand_h
            _bbox[2] -= rand_w
            _bbox[3] -= rand_h

            # if (
            #     (_bbox[0] < 0)
            #     and (_bbox[0] > self.output_size)
            #     and (_bbox[1] < 0)
            #     and (_bbox[1] > self.output_size)
            #     and (_bbox[2] < 0)
            #     and (_bbox[2] > self.output_size)
            #     and (_bbox[3] < 0)
            #     and (_bbox[3] > self.output_size)
            # ):
            #     del _bbox
            min_v = [_bbox[0], _bbox[1]]
            max_v = [_bbox[2], _bbox[3]]

            for i, v in enumerate(min_v):
                if v < 0 and v > self.output_size:
                    temp = True
                else:
                    temp = False

            for i, v in enumerate(max_v):
                if v < 0 and v > self.output_size:
                    temp = True
                else:
                    temp = False

            if temp:
                print("#%d bbox can't be bbox" % idx)
                check.append(0)
            else:
                check.append(1)

        for idx in range(len(bbox)):
            if check[idx] == 0:
                del _bbox[idx]

        print("# of bbox : %d\n" % len(bbox))
        return {"image": crop_image, "annotation": bbox}


class ToTensor:
    def __call__(self, sample):
        image, bbox = sample["image"], sample["annotation"]
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        bbox = np.array(bbox)
        return {"image": torch.from_numpy(image), "annotation": torch.from_numpy(bbox)}


class VOC_DataLoad(torch.utils.data.Dataset):
    def __init__(self, base_dir="./PASCAL_VOC_2012/VOCdevkit/VOC2012", train=True, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.train = train

        train_files = np.loadtxt(
            os.path.join(base_dir, "ImageSets/Main/train.txt"), delimiter="\n", dtype="str"
        )
        val_files = np.loadtxt(
            os.path.join(base_dir, "ImageSets/Main/val.txt"), delimiter="\n", dtype="str"
        )
        self.data_files = np.concatenate((train_files, val_files))

        if os.path.exists("dataset/train.npy") & os.path.exists("dataset/val.npy"):
            pass

        else:
            os.mkdir("dataset")
            random.shuffle(self.data_files)
            split = int(np.round(0.1 * len(self.data_files)))
            train_files, val_files = self.data_files[split:], self.data_files[:split]

            np.save("dataset/train.npy", train_files)
            np.save("dataset/val.npy", val_files)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        :return: { 'image' : img_file, 'annotation' :  [[xmin, ymin, xmax, ymax], [...]]
                }
        """

        img_folder = "JPEGImages"
        annot_folder = "Annotations"

        if self.train:
            file_name = np.load("dataset/train.npy")[idx]
        else:
            file_name = np.load("dataset/val.npy")[idx]

        img_file = os.path.join(self.base_dir, img_folder, "{0}.jpg".format(file_name))
        xml = open(os.path.join(self.base_dir, annot_folder, "{0}.xml".format(file_name)), "r")
        tree = Et.parse(xml)
        root = tree.getroot()

        objects = root.findall("object")
        annotation = []

        obj_idx = 0
        for _object in objects:
            tmp = []
            xml_bndbox = _object.find("bndbox")
            tmp.append(float(xml_bndbox.find("xmin").text))
            tmp.append(float(xml_bndbox.find("ymin").text))
            tmp.append(float(xml_bndbox.find("xmax").text))
            tmp.append(float(xml_bndbox.find("ymax").text))

            annotation.append(tmp)
            obj_idx += 1

        sample = {"image": img_file, "annotation": annotation}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    VOC_dataset = VOC_DataLoad(
        train=True,
        transform=transforms.Compose([Resize(416, scale_factor=1.15), RandomCrop(416), ToTensor()]),
    )
    # utils.show_image(VOC_dataset.__getitem__(4))

    for idx in range(VOC_dataset.__len__()):
        print("#%d image" % idx)
        utils.show_image(VOC_dataset.__getitem__(idx))
    # print(VOC_dataset.__getitem__(idx))
