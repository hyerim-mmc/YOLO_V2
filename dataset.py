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
from PIL import ImageDraw

category = [
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


def show_image(data):
    image, annotation = data["image"], data["annotation"]

    if torch.is_tensor(image):
        image = transforms.ToPILImage()(image)

    draw = ImageDraw.Draw(image)
    bboxes = annotation["bbox"]
    for i, bbox in enumerate(bboxes):
        name = annotation["label"][i].item()
        name_string = category[name]
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="red", width=2)
        draw.text((bbox[0], bbox[1]), name_string, fill=(255, 0, 0))

    image.show()


class Resize:
    def __init__(self, resize_size, scale_factor=1.15):
        assert isinstance(resize_size, int)
        self.resize_size = resize_size
        self.scale_factor = scale_factor

    def __call__(self, data):
        image, annotation = data
        width, height = image.size

        if height > width:
            new_h, new_w = self.resize_size * height / width, self.resize_size
        elif height == width:
            new_h, new_w = height, width
        else:
            new_h, new_w = self.resize_size, self.resize_size * width / height

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
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, data):
        image, annotation = data
        width, height = image.size
        rand_w = random.randint(0, width - self.crop_size)
        rand_h = random.randint(0, height - self.crop_size)

        crop_image = TF.crop(image, rand_h, rand_w, self.crop_size, self.crop_size)

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
            if _bbox[2] > self.crop_size:
                _bbox[2] = self.crop_size - 1
            if _bbox[3] > self.crop_size:
                _bbox[3] = self.crop_size - 1
            if center_x < 0 or center_x > self.crop_size:
                check[idx] = 0
            if center_y < 0 or center_y > self.crop_size:
                check[idx] = 0

        check = check == 1
        annotation["bbox"] = bboxes[check]
        annotation["label"] = annotation["label"][check]

        # print("# of bbox : %d\n" % len(annotation["bbox"]))

        return (crop_image, annotation)


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, resize_size=416, crop_size=416, train_mode=True):
        if os.path.exists("./dataset/train.npy") and os.path.exists("./dataset/val.npy"):
            self.train_set = np.load("./dataset/train.npy")
            self.test_set = np.load("./dataset/test.npy")
        else:
            self.load_dataset()

        self.train_mode = train_mode
        self.resize_bd = Resize(resize_size=resize_size)
        self.crop_bd = RandomCrop(crop_size=crop_size)
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
        :return: { "image" : image,
                   "annotation" : {
                                "label" : [label_index]
                                "bbox" : [[xmin, ymin, xmax, ymax], [...]]
                                }
                }
        """
        if self.train_mode:
            self.image_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages/" + "{0}.jpg".format(self.train_set[idx])
            self.xml_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/" + "{0}.xml".format(self.train_set[idx])

        else:
            self.image_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages/" + "{0}.jpg".format(self.test_set[idx])
            self.xml_path = "./Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/" + "{0}.xml".format(self.test_set[idx])

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

        annotation["label"] = torch.tensor(labels).long()
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


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, img_size=256, val_mode=False):
        super(ImageNetDataset, self).__init__()

        self.base_path = "/home/mmc-server3/Server/server2/hyerim/yolov2/"
        self.data_path = "/home/mmc-server3/Server/dataset/ILSVRC2012_img_train/"
        self.val_mode = val_mode

        if val_mode:
            self.data_path = "/home/mmc-server3/Server/dataset/ILSVRC2012_img_val/"
            self.val_label = np.loadtxt("./ImageNet/ILSVRC2011_devkit-2.0/data/ILSVRC2011_validation_ground_truth.txt")
            self.val_cat = np.empty((0), dtype="str")

        data_set = np.empty((0), dtype="str")
        self.category = os.listdir(self.data_path)
        self.category.sort()

        for i in self.category:
            path = self.data_path + i
            img_set = os.listdir(path)
            img_set.sort()
            img_set = np.array(img_set)
            data_set = np.concatenate((data_set, img_set), 0)

            if val_mode:
                temp = np.stack([i for j in range(len(img_set))])
                self.val_cat = np.concatenate((self.val_cat, temp), 0)

        self.transformation = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.75, saturation=0.75, hue=0.1),
                transforms.ToTensor(),
            ]
        )
        self.va_transformation = transforms.Compose([transforms.Resize([img_size, img_size]), transforms.ToTensor()])
        self.data_set = data_set

        print("Total Image : {0} // Total Category : {0}".format(self.data_set.shape[0], len(self.category)))

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        data = self.data_set[idx]

        if self.val_mode:
            val_cat = self.val_cat[idx]
            cat_idx = self.category.index(val_cat)
            path = self.data_path + val_cat + "/" + data

        else:
            train_cat = data[:9]
            cat_idx = self.category.index(train_cat)
            path = self.data_path + train_cat + "/" + data

        image = Image.open(path).convert("RGB")

        if self.val_mode:
            image = self.va_transformation(image)
        else:
            image = self.transformation(image)

        return (image, cat_idx)


class Anchor_Box:
    def __init__(self, k=5, path="./dataset/train.npy"):
        self.k = k
        box = []
        data_set = np.load(path)
        xml_set = {
            "./Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/" + "{0}.xml".format(data_set[i])
            for i in range(len(data_set))
        }

        for xml_path in xml_set:
            xml = open(xml_path, "r")
            tree = Et.parse(xml)
            root = tree.getroot()
            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)

            objects = root.findall("object")
            for _object in objects:
                bndbox = _object.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                box.append([(xmax - xmin) / width, (ymax - ymin) / height])
                # box.append([xmax - xmin, ymax - ymin])

        self.box = box
        self.box = np.array(box, dtype=np.float32)
        self.centroid = np.array(random.sample(list(self.box), self.k))
        self.cluster = np.empty(len(self.box))

    def calc_dist(self, box):
        w_min = np.minimum(box[0], self.centroid[:, 0])
        h_min = np.minimum(box[1], self.centroid[:, 1])

        overlap = w_min * h_min
        iou = overlap / (box[0] * box[1] + self.centroid[:, 0] * self.centroid[:, 1] - overlap)
        dist = 1 - iou
        idx = np.argmin(dist)

        return idx

    def assign_cluster(self):
        for i, box in enumerate(self.box):
            idx = self.calc_dist(box)
            self.cluster[i] = idx

    def update_centroid(self):
        for i in range(self.k):
            idx = self.cluster == i
            kth_boxes = self.box[idx]
            next_centroid = np.mean(kth_boxes, axis=0)
            self.centroid[i, :] = next_centroid

    def save_anchor(self):
        anchor = self.centroid.copy()
        area = anchor[:, 0] * anchor[:, 1]

        sorted_anchor = np.array(anchor)

        for i in range(self.k):
            order = np.argsort(area)[i]
            sorted_anchor[i] = list(anchor[order])

        np.save("./dataset/anchor.npy", sorted_anchor)
        print("anchor.npy saved!")

    def gen_anchor(self):
        old_cluster = self.cluster.copy()

        while True:
            self.assign_cluster()
            self.update_centroid()

            if (self.cluster == old_cluster).all():
                self.save_anchor()
                return

            old_cluster = self.cluster.copy()


if __name__ == "__main__":
    # VOC = VOCDataset()
    # for idx in range(VOC.__len__()):
    #     print("#%d image" % idx)
    #     show_image(VOC.__getitem__(idx))

    ImageNet = ImageNetDataset(val_mode=True)
    print(ImageNet.__getitem__(0))

    # anchor = Anchor_Box()
    # anchor.gen_anchor()
    # utils.load_npy("./dataset/anchor.npy")
