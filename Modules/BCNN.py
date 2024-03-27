# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2022/11/6 14:23
FileName: BCNN.py
"""

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import os
import math
import matplotlib.pyplot as plt
import time
from BBBConv import BBBConv2d
from BBBLinear import BBBLinear
from misc import ModuleWrapper, FlattenLayer


class BCNN1525(nn.Module):
    def __init__(self, t=100, h=15, w=25, hidden_size=4):
        super(BCNN1525, self).__init__()
        self.downsample = BBBConv2d(1, 32, 3, stride=2, padding=1)
        self.downsample4 = BBBConv2d(32, 32, 3, stride=1, padding=1)
        self.downsample1 = BBBConv2d(32, 64, 3, stride=2, padding=1)
        self.downsample2 = BBBConv2d(32, 32, 3, stride=1, padding=1)
        self.downsample3 = BBBConv2d(32, 1, 1, stride=1, padding=0)
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
        x.cuda()
        x = self.relu1(x)
        x = self.downsample(x)
        print(x)
        print(x.shape)
        x = self.relu2(x)
        x = self.downsample4(x)
        print(x)
        print(x.shape)
        x = self.relu8(x)
        x = self.downsample1(x)
        F.interpolate(input=x, size=(15, 25), mode='bilinear')
        print(x)
        print(x.shape)
        x = self.relu3(x)
        x = self.upsample1(x, output_size=[8, 13])
        print(x)
        print(x.shape)
        x = self.relu4(x)
        x = self.downsample2(x)
        print(x)
        print(x.shape)
        x = self.relu5(x)
        x = self.upsample(x, output_size=[15, 25])
        print(x)
        print(x.shape)
        x = self.relu6(x)
        x = self.downsample3(x)
        print(x)
        print(x.shape)
        x = self.relu7(x)
        x = self.dropout(x)
        return x
