# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/7 15:53
FileName: ResidualNet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # 88 = 24x3 + 16

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.mp = nn.MaxPool2d(2)
        # self.fc = nn.Linear(512, 10)

    def forward(self, x):
        print(x.shape)
        in_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        print(x.shape)

        x = self.rblock1(x)
        print(x.shape)

        x = self.mp(F.relu(self.conv2(x)))
        print(x.shape)

        x = self.rblock2(x)
        print(x.shape)

        # x = x.view(in_size, -1)  # (64,32×4×4=512)
        # x = self.fc(x)
        return x


x = torch.randn(100, 1, 15*4, 25*4)
model = Net()
out = model(x)
print('resnet:', out.shape)