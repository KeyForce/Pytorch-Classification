# -*- coding: utf-8 -*-
# 剪裁标注数据
import os
import random
import shutil
import xml.dom.minidom

import cv2
import matplotlib.pyplot as plt
import numpy
from PIL import Image
from torchvision import transforms as tfs

image_path = '/home/hanwei-1/data/usg/cvted1-3/JPEGImages'
annotation_path = '/home/hanwei-1/data/usg/cvted1-3/Annotations'
benign_path = '/home/hanwei-1/data/usg/cvted1-3/ROI/benign_aug'
suspicion_path = '/home/hanwei-1/data/usg/cvted1-3/ROI/suspicious_aug'

file = os.listdir(annotation_path)


def devide_all_file():
    for xmlfile in file:
        print(xmlfile)
        dom = xml.dom.minidom.parse(os.path.join(annotation_path, xmlfile))
        root = dom.documentElement

        name = root.getElementsByTagName('name')
        filename = root.getElementsByTagName('filename')

        a = name[0].firstChild.data
        b = filename[0].firstChild.data
        image_name = b.replace('_', '').replace('png', 'jpg')

        if a == 'suspicious':
            shutil.copy(os.path.join(image_path, image_name), suspicion_path)

        if a == 'benign':
            shutil.copy(os.path.join(image_path, image_name), benign_path)


def create_roi_pic():
    for xmlfile in file:
        print(xmlfile)
        dom = xml.dom.minidom.parse(os.path.join(annotation_path, xmlfile))
        root = dom.documentElement

        name = root.getElementsByTagName('name')
        filename = root.getElementsByTagName('filename')

        label_name = name[0].firstChild.data
        file_name = filename[0].firstChild.data
        image_name = file_name.replace('_', '').replace('png', 'jpg')

        xmin = root.getElementsByTagName('xmin')[0].firstChild.data
        ymin = root.getElementsByTagName('ymin')[0].firstChild.data
        xmax = root.getElementsByTagName('xmax')[0].firstChild.data
        ymax = root.getElementsByTagName('ymax')[0].firstChild.data

        image_p = os.path.join(image_path, image_name)
        image = cv2.imread(image_p)
        # plt.imshow(image)
        # plt.show()
        corpimage = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        # plt.imshow(corpimage)
        # plt.show()
        if label_name == 'benign':
            cv2.imwrite('/home/hanwei-1/data/usg/cvted1-3/ROI/benign/' + image_name, corpimage)
        else:
            cv2.imwrite('/home/hanwei-1/data/usg/cvted1-3/ROI/suspicious/' + image_name, corpimage)


def create_txt():
    path = os.listdir(benign_path)
    data = []

    for p in path:
        data.append(" ".join(['/benign_aug', p, 'benign']))
    sus_path = os.listdir(suspicion_path)

    for p in sus_path:
        data.append(" ".join(['/suspicious_aug', p, 'suspicious']))

    random.shuffle(data)
    root_dir = '/home/hanwei-1/data/usg/cvted1-3/ROI'

    with open('{}/data_aug.txt'.format(root_dir), 'w') as f:
        for image in data:
            f.write('{}\r\n'.format(image))

    train = data[:int(len(data) * 0.7)]
    test = data[int(len(data) * 0.7):]

    with open('{}/train_aug.txt'.format(root_dir), 'w') as f:
        for image in train:
            f.write('{}\r\n'.format(image))

    with open('{}/test_aug.txt'.format(root_dir), 'w') as f:
        for image in test:
            f.write('{}\r\n'.format(image))


im_aug = tfs.Compose([
    tfs.Resize([96, 96]),
    tfs.RandomHorizontalFlip(),
    # tfs.RandomCrop(32),
])


def data_aug():
    nrows = 1
    ncols = 1
    figsize = (8, 8)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    path = os.listdir('/home/hanwei-1/data/usg/cvted1-3/ROI/suspicious')
    # plt.imshow(im_aug(im))
    for im_name in path:
        im = Image.open('/home/hanwei-1/data/usg/cvted1-3/ROI/suspicious/' + im_name)
        im_name = im_name.replace('.jpg', '')
        for i in range(nrows):
            for j in range(ncols):
                image = im_aug(im)
                # figs[i][j].imshow(image)
                # figs[i][j].axes.get_xaxis().set_visible(False)
                # figs[i][j].axes.get_yaxis().set_visible(False)
                opencvImage = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
                file_name = '/home/hanwei-1/data/usg/cvted1-3/ROI/suspicious_aug/' + im_name + str(i + j) + '.jpg'
                print(file_name)
                cv2.imwrite(file_name, opencvImage)
    # plt.show()


if __name__ == '__main__':
    img = Image.open('/home/hanwei-1/data/usg/cvted1-3/ROI/benign/20190121165048.jpg')
    # new_img = img.resize((96, 96), Image.BILINEAR)
    # plt.imshow(new_img)
    # plt.show()
    # data_aug()
    create_txt()
