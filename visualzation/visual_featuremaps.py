# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)


def normalize_invert(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


vis_layer = 'conv2'
log_dir = '../Logger/visual_featuremaps'
txt_path = './visual.txt'
pretrained_path = './net_params_72p.pkl'

net = Net()
pretrained_dict = torch.load(pretrained_path)
net.load_state_dict(pretrained_dict)

# 数据预处理
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)
testTransform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normTransform
])
# 载入数据
test_data = MyDataset(txt_path=txt_path, transform=testTransform)
print(test_data)
test_loader = DataLoader(dataset=test_data, batch_size=1)
img, label = iter(test_loader).next()

x = img
writer = SummaryWriter(log_dir=log_dir)
for name, layer in net._modules.items():

    # 为fc层预处理x
    x = x.view(x.size(0), -1) if "fc" in name else x

    # 对x执行单层运算
    x = layer(x)
    print(x.size())

    # 由于__init__()相较于forward()缺少relu操作，需要手动增加
    x = F.relu(x) if 'conv' in name else x

    # 依据选择的层，进行记录feature maps
    if name == vis_layer:
        # 绘制feature maps
        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
        plt.imshow(x1[2, 0, :, :].detach().numpy(), cmap="hot")
        plt.colorbar()
        plt.show()
        heatmap = x1[2, 0, :, :].detach().numpy()
        img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=2)  # B，C, H, W
        writer.add_image(vis_layer + '_feature_maps', img_grid)

        # 绘制原始图像
        img_raw = normalize_invert(img, normMean, normStd)  # 图像去标准化
        img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8')
        writer.add_image('raw img', img_raw)  # j 表示feature map数
writer.close()
