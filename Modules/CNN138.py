# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/11 21:28
FileName: CNN138.py
"""
import numpy as np
import torch.nn as nn
import torch
torch.set_printoptions(threshold=np.inf)


class Net(nn.Module):
    def __init__(self, t=100, h=8, w=13, hidden_size=4):
        super(Net, self).__init__()
        self.downsample = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.downsample4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.downsample1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.downsample3 = nn.Conv2d(32, 1, 1, stride=1, padding=0)
        self.upsample1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = self.relu1(x)
#         x = self.dropout(x)
        print(x.shape)
        x = self.downsample(x)
        x = self.relu2(x)
        print(x.shape)

        x = self.downsample4(x)
        x = self.relu8(x)
        print(x.shape)

        x = self.downsample1(x)
        x = self.relu3(x)
        print(x.shape)

        x = self.upsample1(x, output_size=[4, 7])
        x = self.relu4(x)
        print(x.shape)

        x = self.downsample2(x)
        x = self.relu5(x)
        print(x.shape)

        x = self.upsample(x, output_size=[8, 13])
        x = self.relu6(x)
        print(x.shape)

        x = self.downsample3(x)
        x = self.relu7(x)
        print(x.shape)

#         x = self.dropout(x)
        return x


x = torch.randn(100, 1, 8, 13)
model = Net()
out = model(x)
print('resnet:', out.shape)