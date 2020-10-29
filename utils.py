from PIL import Image
from PIL import ImageDraw

# input : file path
def show_image(img_file, annotation):
    img = Image.open(img_file).convert('RGB')
    draw = ImageDraw.Draw(img)

    for idx in range(0, int(annotation["objects"]["num_obj"])):
        name = annotation["objects"][str(idx)]["name"]
        xmin = annotation["objects"][str(idx)]["bndbox"]["xmin"]
        ymin = annotation["objects"][str(idx)]["bndbox"]["ymin"]
        xmax = annotation["objects"][str(idx)]["bndbox"]["xmax"]
        ymax = annotation["objects"][str(idx)]["bndbox"]["ymax"]

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name)

    img.show()

# input : raw image
def show_image_raw(img_file, annotation):
    draw = ImageDraw.Draw(img_file)

    for idx in range(0, int(annotation["objects"]["num_obj"])):
        name = annotation["objects"][str(idx)]["name"]
        xmin = annotation["objects"][str(idx)]["bndbox"]["xmin"]
        ymin = annotation["objects"][str(idx)]["bndbox"]["ymin"]
        xmax = annotation["objects"][str(idx)]["bndbox"]["xmax"]
        ymax = annotation["objects"][str(idx)]["bndbox"]["ymax"]

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name)

    img_file.show()