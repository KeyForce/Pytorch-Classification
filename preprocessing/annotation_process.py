# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET

image_path = '/home/hanwei-1/data/usg/cvted1-4/JPEGImages'
annotation_path = '/home/hanwei-1/data/usg/cvted1-4/Annotations'


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

    objs[0].find('name').text = 'suspicious'

    label = objs[0].find('name')

    if label != 'suspicious':
        print('error')

    return {'boxes': boxes,
            'label': label,
            }


if __name__ == '__main__':
    xml_path = os.listdir(annotation_path)
    anno_info = {}
    for xml_index in xml_path:
        anno_info = _load_pascal_annotation(annotation_path, xml_index)
