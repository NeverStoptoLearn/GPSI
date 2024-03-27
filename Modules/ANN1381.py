# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/11 21:35
FileName: ANN1381.py
"""
import numpy as np
import torch.nn as nn
import torch
torch.set_printoptions(threshold=np.inf)


class Net(nn.Module):
    def __init__(self, hidden_dim=16, width=8, height=13):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(height, height)
        self.relu1 = nn.ReLU()  # nn.Linear为线性关系，加上激活函数转为非线性
        self.fc2 = nn.Linear(height, height)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out