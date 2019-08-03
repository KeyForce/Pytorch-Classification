# -*- coding: utf-8 -*-
from fastai import *
from fastai.vision import *
from fastai.vision import image as im
import torch.nn as nn
from torch.nn.functional import mse_loss
import json
import re
import torch


image_path = '/home/hanwei-1/data/hand_labels_synth/synth2'
json_path = '/home/hanwei-1/data/hand_labels_synth/synth2_json'

transforms = get_transforms(do_flip=False, max_zoom=1.1, max_warp=0.01, max_rotate=45)


def get_y_func(x):
    pre, ext = os.path.splitext(x)
    hand_data_out = []
    hand_data = json.load(open(pre + '.json'))
    for i in range(21):
        hand_data_out.append(hand_data['hand_pts'][i][:2])
    return torch.tensor(hand_data_out, dtype=torch.float)

data = (PointsItemList.from_folder(path=image_path, extensions=['.jpg'])
        .split_by_rand_pct()                 # setting training and testing dataset folders paths
        .label_from_func(get_y_func)                                     # using get_y_func() to get coordinates for each image
        .transform(transforms,size=224, tfm_y=True, remove_out=False,  # very important!!!: setting remove_out to False,
                                                                         # prevents from discarding coordinates that may
                                                                         # disappear after data augmentation
                   padding_mode='border', resize_method=ResizeMethod.PAD)
        .databunch(bs=8)                                                 # Setting your batch size.
        .normalize(imagenet_stats))                                      # Normalizing the data to help the model converging faster