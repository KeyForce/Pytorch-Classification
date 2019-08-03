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

transforms = get_transforms(do_flip=False, max_zoom=1.1, max_warp=0.01, max_rotate=45)


def get_y_func(x):
    pre, ext = os.path.splitext(x)
    hand_data_out = []
    hand_data = json.load(open(pre + '.json'))
    for i in range(21):
        hand_data_out.append(hand_data['hand_pts'][i][:2])
    return torch.tensor(hand_data_out, dtype=torch.float)


data = (PointsItemList.from_folder(path=image_path, extensions=['.jpg'])
        .split_by_rand_pct()
        .label_from_func(get_y_func)
        .transform(transforms,size=368, tfm_y=True, remove_out=False,
                   padding_mode='border', resize_method=ResizeMethod.PAD)
        .databunch(bs=8)
        .normalize(imagenet_stats))


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


head_reg = nn.Sequential(
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(73728, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, 42),
    Reshape(-1,21,2),
    nn.Tanh())


class MSELossFlat(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor):
     return super().forward(input.view(-1), target.view(-1))

mse_loss_flat = MSELossFlat()
learn = cnn_learner(data, models.resnet34,custom_head=head_reg, loss_func=mse_loss_flat)
learn.fit_one_cycle(cyc_len = 100,max_lr = 1e-4)