# -*- coding: utf-8 -*-

""" 
author: Zhou Chen
Time: 2022/8/26 15:53
FileName: ANNModel.py
"""
import torch.nn as nn
import torch


# class ANNModel(nn.Module):
#     def __init__(self, input_dim, width, height):
#         super(ANNModel, self).__init__()
#         # 定义层
#         self.fc1 = nn.Linear(input_dim, width)
#         self.relu1 = nn.ReLU()  # nn.Linear为线性关系，加上激活函数转为非线性
#
#         self.fc2 = nn.Linear(1, width)
#         self.relu2 = nn.ReLU()
#
#         self.fc3 = nn.Linear(width, height)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         return out
# class ANNModel(nn.Module):
#     def __init__(self, input_dim=100, hidden_dim=252, width=18, height=14):
#         super(ANNModel, self).__init__()
#         # 定义层
#         self.fc1 = nn.Linear(input_dim, width * height)
#         self.relu1 = nn.ReLU()  # nn.Linear为线性关系，加上激活函数转为非线性
#
#         self.fc2 = nn.Linear(width * height, input_dim * hidden_dim)
#         self.relu2 = nn.ReLU()
#
#         self.fc3 = nn.Linear(input_dim * hidden_dim, width * height)
#
#     def forward(self, x):
#         # out = self.fc1(x)
#         # out = self.relu1(out)
#         out = self.relu1(self.relu1(x))
#         print(out.shape)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         print(out.shape)
#         out = self.fc3(out)
#         print(out.shape)
#         return out

class ANNModel(nn.Module):
    def __init__(self, hidden_dim, width, height):
        super(ANNModel, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(height, hidden_dim)
        self.relu1 = nn.ReLU()  # nn.Linear为线性关系，加上激活函数转为非线性

        self.fc2 = nn.Linear(hidden_dim, width*hidden_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(width*hidden_dim, height)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        print(out.shape)
        out = self.fc2(out)
        out = self.relu2(out)
        print(out.shape)
        out = self.fc3(out)
        print(out.shape)
        return out
