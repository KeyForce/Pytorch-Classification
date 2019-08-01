# -*- coding: utf-8 -*-
import os
import xml.dom.minidom

image_path = '/home/hanwei-1/data/usg/cvted4/JPEGImages'
annotation_path = '/home/hanwei-1/data/usg/cvted4/Annotations'

if __name__ == '__main__':
    file = os.listdir(annotation_path)

    sus_count = 0
    be_count = 0

    for xml_file in file:
        print(xml_file)
        dom = xml.dom.minidom.parse(os.path.join(annotation_path, xml_file))
        root = dom.documentElement

        name = root.getElementsByTagName('name')
        class_name = name[0].firstChild.data

        if class_name == 'suspicious':
            sus_count += 1

        if class_name == 'benign':
            be_count += 1

    print("sus:", sus_count)
    print("be", be_count)
    print("all", file.__len__())
