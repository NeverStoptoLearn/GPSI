# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2022/7/29 8:54
FileName: CNN_source.py
"""
import torch.nn as nn
import torch


# class CNN_source(nn.Module):
#     def __init__(self, t=100, h=18, w=14, hidden_size=4):
#         super(CNN_source, self).__init__()
#         self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=3, stride=2)
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_size * 2),
#             nn.LeakyReLU()
#         )
#         self.cnn2 = nn.Sequential(
#             nn.Conv2d(in_channels=hidden_size * 2, out_channels=hidden_size, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_size),
#             nn.LeakyReLU()
#         )
#         self.cnn3 = nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=1, padding=(5, 4))
#         self.h = h
#         self.w = w
#         self.dropout = nn.Dropout(p=0.15)
#
#     def forward(self, x):
#         print(x.shape)
#         x = self.cnn(x)
#         print(x.shape)
#         x = self.dropout(x)
#         x = self.cnn1(x)
#         print(x.shape)
#         x = self.cnn2(x)
#         print(x.shape)
#         x = self.cnn3(x)
#         print(x.shape)
#         return x

class CNN_source(nn.Module):
    def __init__(self, t=100, h=18, w=14, hidden_size=4):
        super(CNN_source, self).__init__()
        self.downsample = nn.Conv2d(1, 2, 3, stride=2, padding=1)
        self.downsample1 = nn.Conv2d(2, 4, 3, stride=2, padding=1)
        self.upsample1 = nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1)
        self.h = h
        self.w = w
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = self.downsample(x)
        x = self.downsample1(x)
        x = self.upsample1(x, output_size=[9, 7])
        x = self.upsample(x, output_size=[18, 14])
        x = self.dropout(x)
        return x
