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

    def __call__(self, sample):
        image_path, bbox = sample['image'], sample['annotation']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        if height > width:
            new_h, new_w = self.output_size * height / width, self.output_size
        elif height == width:
            new_h, new_w = height, width
        else:
            new_h, new_w = self.output_size, self.output_size * width / height

        new_h, new_w = int(new_h * self.scale_factor), int(new_w * self.scale_factor)
        new_image = TF.resize(image, (new_h, new_w), Image.BICUBIC)

        for idx in range(len(bbox)):
            _bbox = bbox[idx]
            _bbox[0] *= new_w / width
            _bbox[1] *= new_h / height
            _bbox[2] *= new_w / width
            _bbox[3] *= new_h / height

        return {'image': new_image, 'annotation': bbox}


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['annotation']
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
                print('#%d bbox can't be bbox' % idx)
                check.append(0)
            else:
                check.append(1)

        for idx in range(len(bbox)):
            if check[idx] == 0:
                del _bbox[idx]

        print('# of bbox : %d\n' % len(bbox))
        return {'image': crop_image, 'annotation': bbox}


class ToTensor:
    def __call__(self, sample):
        image, bbox = sample['image'], sample['annotation']
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        bbox = np.array(bbox)
        return {'image': torch.from_numpy(image), 'annotation': torch.from_numpy(bbox)}


class VOC_DataLoad(torch.utils.data.Dataset):
    def __init__(self, output_size = 416, train=True):
        if os.path.exists('./dataset/train.npy') and os.path.exists('./dataset/val.npy'):
            self.train_set = np.load('./dataset/train.npy')
            self.test_set = np.load('./dataset/test.npy')
        else:
            self.load_dataset()

        if train:
            self.image_path = os.path.join('./Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages/', '{0}.jpg'.format(self.train_set))
            self.xml_path = os.path.join('./Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/', '{0}.xml'.format(self.train_set))
        else:
            self.image_path = os.path.join('./Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages/', '{0}.jpg'.format(self.test_set))
            self.xml_path = os.path.join('./Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/', '{0}.xml'.format(self.test_set))
        
        self.train = train
        self.resize_bd = Resize(output_size = output_size)
        self.crop_bd = RandomCrop(output_size = output_size)
        self.transform = transforms.Compose([self.resize_bd, self.crop_bd, transforms.ColorJitter(brightness=0.75, hue=0.1, saturation=.75), ToTensor()]),
        self.va_transform = transforms.Compose([ToTensor()])
        
        self.category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                         'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']
    def __len__(self):
        return len(self.data_set)

    def load_dataset(self):
        train_path = './PASCAL_VOC_2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
        validate_path = './PASCAL_VOC_2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt'

        self.train_files = np.loadtxt(train_path, delimiter='\n', dtype='str')
        self.validate_files = np.loadtxt(validate_path, delimiter='\n', dtype='str')
        self.data_set = np.concatenate((self.train_files, self.validate_files))

        random.shuffle(self.data_set)
        split = int(np.round(0.2 * len(self.data_set)))

        self.train_set, self.test_set = self.data_set[split:], self.data_set[:split]

        np.save('dataset/train.npy', self.train_set)
        np.save('dataset/test.npy', self.test_set)



    def __getitem__(self, idx):
        '''
        :return: { 'image' : img_file, 'annotation' :  [[xmin, ymin, xmax, ymax], [...]]
                }
        '''

        self.xml = open(os.path.join(self.base_dir, annot_folder, '{0}.xml'.format(file_name)), 'r')
        img_folder = 'JPEGImages'
        annot_folder = 'Annotations'

        if self.train:
            file_name = np.load('dataset/train.npy')[idx]
        else:
            file_name = np.load('dataset/val.npy')[idx]

        img_file = os.path.join(self.base_dir, img_folder, '{0}.jpg'.format(file_name))
        xml = open(os.path.join(self.base_dir, annot_folder, '{0}.xml'.format(file_name)), 'r')
        tree = Et.parse(xml)
        root = tree.getroot()

        objects = root.findall('object')
        annotation = []

        obj_idx = 0
        for _object in objects:
            tmp = []
            xml_bndbox = _object.find('bndbox')
            tmp.append(float(xml_bndbox.find('xmin').text))
            tmp.append(float(xml_bndbox.find('ymin').text))
            tmp.append(float(xml_bndbox.find('xmax').text))
            tmp.append(float(xml_bndbox.find('ymax').text))

            annotation.append(tmp)
            obj_idx += 1

        sample = {'image': img_file, 'annotation': annotation}

        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    VOC_dataset = VOC_DataLoad(
        train=True
        )
    # utils.show_image(VOC_dataset.__getitem__(4))

    for idx in range(VOC_dataset.__len__()):
        print('#%d image' % idx)
        utils.show_image(VOC_dataset.__getitem__(idx))
    # print(VOC_dataset.__getitem__(idx))
