import os
import random
import numpy as np
import utils

import xml.etree.ElementTree as Et
import torch.utils.data.dataset
import torch.utils.data.dataloader

from PIL import Image


class Resize:
    def __int__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        image_path, annotation = sample["image"], sample["annotation"]
        image = Image.open(image_path).convert('RGB')
        height, width = np.shape(image)



        return


class RandomCrop:
    def __init__(self):
        pass


class VOC_DataLoad(torch.utils.data.Dataset):
    def __init__(self, base_dir='./PASCAL_VOC_2012/VOCdevkit/VOC2012', train=True, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.train = train

        train_files = np.loadtxt(os.path.join(base_dir, 'ImageSets/Main/train.txt'), delimiter='\n', dtype='str')
        val_files = np.loadtxt(os.path.join(base_dir, 'ImageSets/Main/val.txt'), delimiter='\n', dtype='str')
        self.data_files = np.concatenate((train_files, val_files))

        if os.path.exists('train.npy') & os.path.exists('val.npy'):
            pass

        else:
            random.shuffle(self.data_files)
            split = int(np.round(0.1 * len(self.data_files)))
            train_files, val_files = self.data_files[split:], self.data_files[:split]

            np.save('train.npy', train_files)
            np.save('val.npy', val_files)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        :return: { 'image' : img_file path,
                   'annotation' :
                   {"size" :
                                {   "width" : <string>
                                    "height" : <string>
                                    "depth" : <string>    }
                    "objects" :
                                {   "num_obj" : <int>
                                    "<index>" :
                                                {   "name" : <string>
                                                    "bndbox" :
                                                                {   "xmin" : <float>
                                                                    "ymin" : <float>
                                                                    "xmax" : <float>
                                                                    "ymax" : <float>    }
                                                }
                                }
                }
        """

        img_folder = 'JPEGImages'
        annot_folder = 'Annotations'

        if self.train:
            file_name = np.load('train.npy')[idx]
        else:
            file_name = np.load('val.npy')[idx]

        img_file = os.path.join(self.base_dir, img_folder, "{0}.jpg".format(file_name))
        xml = open(os.path.join(self.base_dir, annot_folder, "{0}.xml".format(file_name)), "r")
        tree = Et.parse(xml)
        root = tree.getroot()

        xml_size = root.find("size")
        size = {
            "width": xml_size.find("width").text,
            "height": xml_size.find("height").text,
            "channels": xml_size.find("depth").text
        }

        objects = root.findall("object")
        if len(objects) == 0:
            return False, "number of objects is zero"
        obj = {
            "num_obj": len(objects)
        }

        obj_idx = 0
        for _object in objects:
            tmp = {
                "name": _object.find("name").text
            }

            xml_bndbox = _object.find("bndbox")
            bndbox = {
                "xmin": float(xml_bndbox.find("xmin").text),
                "ymin": float(xml_bndbox.find("ymin").text),
                "xmax": float(xml_bndbox.find("xmax").text),
                "ymax": float(xml_bndbox.find("ymax").text)
            }
            tmp["bndbox"] = bndbox
            obj[str(obj_idx)] = tmp

            obj_idx += 1

        annotation = {
            "size": size,
            "objects": obj
        }

        if self.transform:
            pass

        sample = {'image': img_file, 'annotation': annotation}
        return sample


if __name__ == '__main__':
    VOC_dataset = VOC_DataLoad()
    for idx in range(VOC_dataset.__len__()):
        utils.show_image(VOC_dataset.__getitem__(idx)["image"], VOC_dataset.__getitem__(idx)["annotation"])
