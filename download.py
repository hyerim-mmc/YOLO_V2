import os
import wget
import tarfile


def load_PASCAL_VOC2012():
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    filename = "VOCtrainval_11-May-2012.tar"
    data_dir = "./PASCAL_VOC_2012"

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        wget.download(url, data_dir)
    else:
        if filename in os.listdir(data_dir):
            pass
        else:
            wget.download(url, data_dir)
    if filename not in os.listdir(data_dir):
        raise RuntimeError("Tarfile is not found")

    if "VOCdevkit" in os.listdir(data_dir):
        print("Dataset already exists")
    else:
        with tarfile.open(os.path.join(data_dir, filename), "r") as untar:
            print("Start extracting tarfile")
            untar.extractall(data_dir)
            print("PASCAL_VOC2012 is downloaded")


def load_ImageNet():
    train_url = (
        "http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar"
    )
    val_url = "http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar"
    train_file = "ILSVRC2012_img_train.tar"
    val_file = "ILSVRC2012_img_val.tar"
    google_drive_train_path = "C:/Users/LeeHyerim/Google 드라이브/ImageNet/train"
    google_drive_val_path = "C:/Users/LeeHyerim/Google 드라이브/ImageNet/val"

    # if not os.path.exists(google_drive_train_path):
    #     os.mkdir(google_drive_train_path)
    #     wget.download(train_url, google_drive_train_path)
    # else:
    #     with tarfile.open(os.path.join(google_drive_train_path,train_file),"r") as untar:
    #         untar.extractall(google_drive_train_path)

    # if not os.path.exists(google_drive_val_path):
    #     os.mkdir(google_drive_val_path)
    #     wget.download(val_url, google_drive_val_path)
    # else:
    # with tarfile.open(os.path.join(google_drive_val_path, val_file), "r") as untar:
    #     untar.extractall(google_drive_val_path)

    imagefile_name = "ILSVRC2012_img_val.tar"
    labelfile_name = "ILSVRC2011_devkit-2.0.tar.gz"
    data_dir = "./ImageNet"
    image_dir = data_dir + "/ImageSets"

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if "ImageSets" in os.listdir(data_dir):
        print("Image file already exists")
    else:
        with tarfile.open(imagefile_name, "r") as untar:
            print("Start extracting image tarfile")
            untar.extractall(image_dir)
            print("Image of ImageNet is downloaded")

    if "ILSVRC2011_devkit-2.0" in os.listdir(data_dir):
        print("Label file already exists")
    else:
        with tarfile.open(labelfile_name, "r") as untar:
            print("Start extracting label tarfile")
            untar.extractall(data_dir)
            print("Label of ImageNet is downloaded")


if __name__ == "__main__":
    # load_PASCAL_VOC2012()
    load_ImageNet()
