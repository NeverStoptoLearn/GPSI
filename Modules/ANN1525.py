# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2023/1/11 21:34
FileName: ANN1525.py
"""
import numpy as np
import torch.nn as nn
import torch
torch.set_printoptions(threshold=np.inf)


class Net(nn.Module):
    def __init__(self, hidden_dim=16, width=15, height=25):
        super(Net, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(height, hidden_dim)
        self.relu1 = nn.ReLU()  # nn.Linear为线性关系，加上激活函数转为非线性

        self.fc2 = nn.Linear(hidden_dim, width * hidden_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(width * hidden_dim, height)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out