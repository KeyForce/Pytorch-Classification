# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw

image_path = '/home/hanwei-1/data/usg/cvted4/JPEGImages'
annotation_path = '/home/hanwei-1/data/usg/cvted4/Annotations'
printstyle_path = '/usr/shar/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
save_path = '/home/hanwei-1/data/usg/cvted4/all_pic_roi4'


def _load_pascal_annotation(_data_path, xml_index):
    """
    加载VOC的标注数据
    :param _data_path: 标注的文件夹
    :param xml_index: xml文件名称
    :return: 字典{'boxes': boxes,'label': label,}
    """
    filename = os.path.join(_data_path, xml_index)
    tree = ET.parse(filename)
    objs = tree.findall('object')

    boxes = []

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')

        y1 = float(bbox.find('xmin').text)
        x1 = float(bbox.find('ymin').text)
        y2 = float(bbox.find('xmax').text)
        x2 = float(bbox.find('ymax').text)

        boxes = [x1, y1, x2, y2]

    label = objs[0].find('name').text

    return {'boxes': boxes,
            'label': label,
            }


if __name__ == '__main__':
    xml_path = os.listdir(annotation_path)
    anno_info = {}
    for xml_index in xml_path:
        anno_info = _load_pascal_annotation(annotation_path, xml_index)
        image = Image.open(os.path.join(image_path, xml_index.replace('xml', 'jpg')))

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font=printstyle_path,
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        label_size = draw.textsize(anno_info['label'], font)

        top, left, bottom, right = anno_info['boxes']

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(3):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=(255, 255, 0)
            )
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 0)
        )
        draw.text(text_origin, anno_info['label'], fill=(0, 0, 0), font=font)
        del draw
        file_name = os.path.join(save_path, xml_index.replace('xml', 'jpg'))
        print(file_name)
        # image.save(file_name, 'jpeg')
        plt.imshow(image)
        plt.show()
