# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2022/7/4 16:30
FileName: simple_net.py
"""
import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, dim_fea=6, hidden_size=16):
        super(SimpleNet, self).__init__()
        self.gru = nn.GRU(input_size=dim_fea, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.regressionHead = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, x):
        _, h = self.gru(x)
        h = h.transpose(0, 1)
        # print(h.shape)
        h = self.flatten(h)
        # print(h.shape)
        h = self.fc(h)
        # print(h.shape)
        h = self.regressionHead(h)
        # print(h.shape)
        return h


if __name__ == '__main__':
    a = torch.rand((7, 13, 6))
    mm = SimpleNet()
    print(mm(a).shape)