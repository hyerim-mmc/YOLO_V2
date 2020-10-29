from PIL import Image
from PIL import ImageDraw

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