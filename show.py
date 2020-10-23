import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et

from PIL import Image
from PIL import ImageDraw

base_dir = './PASCAL_VOC_2012/VOCdevkit/VOC2012'
img_folder = 'JPEGImages'
annot_folder = 'Annotations'

annot_root, annot_dir, annot_files = next(os.walk(os.path.join(base_dir, annot_folder)))
img_root, img_dir, img_files = next(os.walk(os.path.join(base_dir, img_folder)))

for xml_file in annot_files:
    img_name = img_files[img_files.index(".".join([xml_file.split(".")[0], "jpg"]))]
    img_file = os.path.join(img_root, img_name)
    img = Image.open(img_file).convert('RGB')
    draw = ImageDraw.Draw(img)

    xml = open(os.path.join(annot_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    # get "size" tag
    size = root.find("size")
    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    # get "object" tag
    objects = root.findall("object")
    for obj in objects:
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        draw.rectangle(((xmin,ymin), (xmax,ymax)), outline = "red")
        draw.text((xmin,ymin), name)

    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.close()

