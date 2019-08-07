# -*- coding: utf-8 -*-
import time

import torch
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 使用mobilenet V2 作为主干网络，只使用feature层，不使用分类层
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# Keypoint RCNN 需要知道主干网络的输出通道数
backbone.out_channels = 1280

# RPN 生成5种不同尺寸的大小 3种比例 Tuple[Tuple[int]]
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
# 让我们来定义我们将使用哪些特性映射来执行感兴趣的裁剪区域，以及重新缩放后的裁剪大小
# 如果主干返回一个张量，featmap_names应该是[0]
# 更一般地说，主干应该返回一个OrderedDict[Tensor]
# 在featmap_names中，您可以选择使用哪个feature maps
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                         output_size=14,
                                                         sampling_ratio=2)
# 输入参数
model = KeypointRCNN(backbone,
                     num_classes=2,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     keypoint_roi_pool=keypoint_roi_pooler)
model.eval()
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

start = time.time()
predictions = model(x)
end = time.time()
print(end - start)

