'''VGG11/13/16/19 in Pytorch.'''
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    name = 'VGG19'
    net = VGG(name, 2)
    print(name)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    print(net)

    # # -----------------------------------------------------
    # # 可疑和良性 两类 输入图片尺寸480*465
    # n_classes = 2
    # image_width = 480
    # image_height = 465
    # model = VGG('VGG19', num_classes=2)
    # x = Variable(torch.randn(1, 3, image_height, image_width))
    #
    # start = time.time()
    # pred = model(x)
    # end = time.time()
    # print(end - start)




