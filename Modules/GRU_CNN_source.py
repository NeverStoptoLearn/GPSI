# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2022/7/3 13:47
FileName: GRU_CNN_source.py
"""
import torch.nn as nn
import torch


class GRU_CNN_source(nn.Module):
    def __init__(self, t=13, n_well=6, h=18, w=14, hidden_size=32):
        super(GRU_CNN_source, self).__init__()
        self.gru = nn.GRU(input_size=n_well, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU()
        )
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=6, kernel_size=3, padding=1),
            nn.BatchNorm1d(6),
            nn.LeakyReLU()
        )
        self.cnn2 = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=5)
        self.h = h
        self.w = w
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        # print(x.shape)
        x, _ = self.gru(x)
        x = x.transpose(-1, -2)
        # print(x.shape)
        x = self.cnn(x)
        # print(x.shape)
        x = self.dropout(x)
        # x = self.upsample1(x)
        # print(x.shape)
        # print(x0.shape)
        #         x0 = torch.repeat_interleave(x0, repeats=x.size(0), dim=0)
        # print(x.shape)
        # print(x0.shape)
        # x = torch.cat((x0, x), 1)

        #         x = self.dropout(x)
        x = self.cnn1(x)
        # print(x.shape)
        x = x.transpose(-1, -2)
        # print(x.shape)
        x = self.cnn2(x)
        # print(x.shape)
        shape_x = x.shape
        return x

#         return torch.sigmoid(x)
#         return torch.softmax(x.view(shape_x[0], shape_x[1], -1), dim=-1).view(shape_x)
