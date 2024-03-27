# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/6 15:48
FileName: ResNet1.py
"""
import torch
from torch import nn
import torch.nn.functional as F


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=(1, 1)):
        """
        :param ch_in:
        :param ch_in:
        """
        super(ResBlk, self).__init__()
        # ResNet 里面加上batch normalization更稳定
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=stride, padding=1)
        # 卷积结束后对本层进行batch normalize
        self.bn1 = nn.BatchNorm2d(ch_out)
        # 第二个卷积层，并再进行batch normalize
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 设置短接层，假如输出的通道不等于输入的通道，就通过1*1卷积核，将其变回ch_in
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        # resnet 单元重的前向传播：输入-conv1-norm1-relu-conv2-norm2
        # [2, 64, 32, 32] => [2, 128, 8, 8]
        out = F.relu(self.bn1(self.conv1(x)))
        # [2, 128, 8, 8] => [2, 128, 8, 8]
        out = self.bn2(self.conv2(out))
        # sortcut
        # extra module:[b, ch_in, h, w] + [b, ch_out, h, w]
        # element-wise add
        # 短接的地方：输入的x，直接加上输出，再次进入relu，并返回
        # 如果通道数不相同，通过1*1卷积核转换通道，再标准化，再相加
        out = self.extra(x) + out
        out = F.relu(out)
        return out


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


class ResNet(nn.Module):
    def __init__(self, nclasses=1):
        super(ResNet, self).__init__()

        # 第一次卷积把输入1通道变成64通道
        # [2, 1, 32, 32] => [2, 64, 10, 10]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # 接下来使用四次resnet模块，最终输出512通道
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h, w] => [b, 512, h, w]
        self.blk4 = ResBlk(512, 256, stride=2)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, nclasses, kernel_size=1)
        )
        self.U1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.U2 = nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        # 最终输出，把512个高级特征转换为10个预测
        # self.outlayer = nn.Linear(256 * 1 * 1, 10)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        # [b, 1, h, w] => [b, 64, h, w]
        x = F.relu(self.conv1(x))
        # [b, 64, h, w] => [b, 512, h, w]
        x = self.blk1(x)
        # x = self.blk2(x)
        if x[0][0].shape == torch.Size([8, 13]):
            x = self.U1(x, output_size=[15, 25])
        elif x[0][0].shape == torch.Size([9, 7]):
            x = self.U1(x, output_size=[18, 14])
        elif x[0][0].shape == torch.Size([4, 7]):
            x = self.U1(x, output_size=[8, 13])
        # x = self.blk3(x)
        # x = self.blk4(x)
        x = self.U2(x)
        # 通过平均池化，把512*h*w变成512*1*1
        # x = F.adaptive_avg_pool2d(x, [1, 1])
        # x = x.view(x.size(0), -1)
        # x = self.outlayer(x)
        return x


x = torch.randn(100, 1, 8, 13)
model = ResNet()
out = model(x)
