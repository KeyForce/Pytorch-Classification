# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from fastai import *
from fastai.vision import *


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class handpose_model(nn.Module):
    def __init__(self):
        super().__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3', \
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict({
            'conv1_1': [3, 64, 3, 1, 1],
            'conv1_2': [64, 64, 3, 1, 1],
            'pool1_stage1': [2, 2, 0],
            'conv2_1': [64, 128, 3, 1, 1],
            'conv2_2': [128, 128, 3, 1, 1],
            'pool2_stage1': [2, 2, 0],
            'conv3_1': [128, 256, 3, 1, 1],
            'conv3_2': [256, 256, 3, 1, 1],
            'conv3_3': [256, 256, 3, 1, 1],
            'conv3_4': [256, 256, 3, 1, 1],
            'pool3_stage1': [2, 2, 0],
            'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1],
            'conv4_3': [512, 512, 3, 1, 1],
            'conv4_4': [512, 512, 3, 1, 1],
            'conv5_1': [512, 512, 3, 1, 1],
            'conv5_2': [512, 512, 3, 1, 1],
            'conv5_3_CPM': [512, 128, 3, 1, 1]})

        block1_1 = OrderedDict({
            'conv6_1_CPM': [128, 512, 1, 1, 0],
            'conv6_2_CPM': [512, 22, 1, 1, 0]
        })

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict({
                'Mconv1_stage%d' % i: [150, 128, 7, 1, 3],
                'Mconv2_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d' % i: [128, 22, 1, 1, 0]})

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(1, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        return x


def generate_data(size):
    x = np.random.uniform(size=(size, 1))
    y = x * 2.0
    return torch.FloatTensor(x), torch.FloatTensor(y)


train_x, train_y = generate_data(1000)
val_x, val_y = generate_data(100)

train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)

train_dl = DataLoader(train_ds, batch_size=8)
val_dl = DataLoader(val_ds, batch_size=8)

data_bunch = DataBunch(train_dl, val_dl)

model = handpose_model()
learn = Learner(data_bunch, model, loss_func=F.mse_loss)
learn.fit_one_cycle(1)

