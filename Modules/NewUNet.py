# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/11 14:28
FileName: NewUNet.py
"""
import torch
from torch import nn
import torch.nn.functional as F


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
        print(nn.Sequential(*blk))
        self.layer = nn.Sequential(*blk)

    def forward(self, x):
        return self.layer(x)


class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(inchannels, outchannels, 3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1 = self.convt(x1)
        if x1[0][0].shape == torch.Size([8, 13]):
            x1 = self.convt(x1, output_size=[15, 25])
        elif x1[0][0].shape == torch.Size([9, 7]):
            x1 = self.convt(x1, output_size=[18, 14])
        elif x1[0][0].shape == torch.Size([4, 7]):
            x1 = self.convt(x1, output_size=[8, 13])
        # print("x1:", x1.shape)
        # print("x2:", x2.shape)
        x = torch.cat([x2, x1], dim=1)
        # print("x:", x.shape)
        x = self.conv(x)
        # print("x1:", x.shape)
        # print("end")
        return x


class UNet(nn.Module):
    def __init__(self, nchannels=1, nclasses=1):
        super(UNet, self).__init__()
        self.down1 = DownBlock(1, nchannels, 64, pool=False)
        # self.down2 = DownBlock(1, 64, 128)
        # self.down2 = F.max_pool2d()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2d = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
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
        # print(x.shape)
        x1 = self.down1(x)
        # print(x1.shape)
        # x2 = self.down2(x1)
        # x2 = self.maxpool2d(x1)
        # print('x2:', x2.shape)
        x2 = self.Conv2d(x1)
        # print('x2:', x2.shape)
        x2 = self.relu(x2)
        # print('x2:', x2.shape)
        # x3 = self.down3(x2)
        # print(x3.shape)
        # x4 = self.down4(x3)
        # print(x4.shape)
        # x5 = self.down5(x4)
        # print(x5.shape)
        # x = self.up1(x5, x4)
        # print(x.shape)
        # x = self.up2(x, x3)
        # print(x.shape)
        # x = self.up3(x3, x2)
        # print(x.shape)
        x = self.up4(x2, x1)
        # print(x.shape)
        x = self.out(x)
        # print(x.shape)
        return x


a2 = torch.zeros([1, 1, 18, 14])
model = UNet()
out = model(a2)

# print('resnet:', out.shape)