# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # 创建网路与加载预训练数据
    net = Net()
    pretrained_dict = torch.load('./net_params.pkl')
    net.load_state_dict(pretrained_dict)

    writer = SummaryWriter('../Logger')
    params = net.state_dict()

    for k, v in params.items():
        print(k)
        if 'conv' in k and 'weight' in k:
            print(v.size())                 # [out, in, h, w]
            channel_in = v.size()[1]        # 输入层通道数
            channel_out = v.size()[0]       # 输出层通道数
            print(v)
            print('chanel in', channel_in)
            print('channel out', channel_out)
            # 通道单独输出
            for j in range(channel_out):
                kernel_j = v[j, :, :, :].unsqueeze(1)
                kernel_grid = vutils.make_grid(kernel_j, nrow=channel_in, normalize=True, scale_each=True)
                writer.add_image(k + '_split_in_channel', kernel_grid, global_step=j)
            # 全部通道输出
            k_w, k_h = v.size()[3], v.size()[2]
            print(k_w, k_h)
            kernel_all = v.view(-1, 1, k_w, k_h)
            kernel_grid = vutils.make_grid(kernel_all, nrow=channel_in, normalize=True, scale_each=True)
            writer.add_image(k + '_all', kernel_grid)

    writer.close()
