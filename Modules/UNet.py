# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/2 21:28
FileName: UNet.py
"""
import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, num_convs, inchannels, outchannels, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
            blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

    def forward(self, x):
        return self.layer(x)


class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.convt(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, nchannels=1, nclasses=1):
        super(UNet, self).__init__()
        self.down1 = DownBlock(1, nchannels, 64, pool=False)
        self.down2 = DownBlock(1, 64, 128)
        # self.down3 = DownBlock(1, 128, 256)
        # self.down4 = DownBlock(1, 256, 512)
        # self.down5 = DownBlock(1, 512, 1024)
        # self.up1 = UpBlock(1024, 512)
        # self.up2 = UpBlock(512, 256)
        # self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, nclasses, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        # x3 = self.down3(x2)
        # x4 = self.down4(x3)
        # x5 = self.down5(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x3, x2)
        x = self.up4(x2, x1)
        x = self.out(x)
        return x


a2 = torch.zeros([1, 1, 15*4, 25*4])
model = UNet()
out = model(a2)
