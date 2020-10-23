import os
import tarfile
import wget

def PASCAL_VOC_2012_LOAD():
    URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    PATH = './PASCAL_VOC_2012'

    if not os.path.exists(PATH):
        os.mkdir(PATH)
        wget.download(URL,PATH)
    else:
        wget.download(URL,PATH)

    os.listdir()
    untar = tarfile.TarFile(file_untar)
    untar.extractall()
    untar.close()

    print("Download finish!")


if __name__ == '__main__':
    PASCAL_VOC_2012_LOAD()

