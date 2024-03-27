# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/11 21:31
FileName: CNN5251.py
"""
import numpy as np
import torch.nn as nn
import torch
torch.set_printoptions(threshold=np.inf)


class Net(nn.Module):
    def __init__(self, t=100, h=15, w=25, hidden_size=4):
        super(Net, self).__init__()
        self.downsample = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.downsample1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(w, w)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = self.relu1(x)
        x = self.downsample(x)
        x = self.relu2(x)
        x = self.downsample1(x)
        x = self.relu3(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x


# a2 = torch.zeros([1, 1, 15, 25])
# model = Net()
# out = model(a2)
# print(a2.shape)