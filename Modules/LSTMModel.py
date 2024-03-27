# -*- coding: utf-8 -*-   

""" 
author: Zhou Chen
Time: 2022/8/29 10:14
FileName: LSTMModel.py
"""
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, width, height):
        super(LSTMModel, self).__init__()
        # 定义层
        # self.fc1 = nn.Linear(height, hidden_dim)
        # self.relu1 = nn.ReLU()  # nn.Linear为线性关系，加上激活函数转为非线性
        #
        # self.fc2 = nn.Linear(hidden_dim, width * hidden_dim)
        # self.relu2 = nn.ReLU()
        #
        # self.fc3 = nn.Linear(width * hidden_dim, height)
        self.lstm = nn.LSTM(height, 20, 2)
        self.fc = nn.Linear(20, height)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        print(out.shape)
        out = self.fc(out)
        print(out.shape)
        return out
