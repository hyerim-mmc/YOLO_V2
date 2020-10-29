import os
import wget
import tarfile


def load_PASCAL_VOC2012():
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    filename = 'VOCtrainval_11-May-2012.tar'
    data_dir = './PASCAL_VOC_2012'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        wget.download(url, data_dir)
    else:
        if filename in os.listdir(data_dir):
            pass
        else:
            wget.download(url, data_dir)
    if filename not in os.listdir(data_dir):
        raise RuntimeError('Tarfile is not found')

    if 'VOCdevkit' in os.listdir(data_dir):
        print("Dataset already exists")
    else:
        with tarfile.open(os.path.join(data_dir, filename), 'r') as untar:
            print("Start extracting tarfile")
            untar.extractall(data_dir)
            print("PASCAL_VOC2012 is downloaded")


if __name__ == '__main__':
    load_PASCAL_VOC2012()
