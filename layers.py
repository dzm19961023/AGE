import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np


## GCN层定义

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    # 定义图卷积网络里的重要参数
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # 输入到网络的特征
        self.out_features = out_features  # 网络输出的特征
        self.dropout = dropout  # 随意丢弃神经网络单元，防止过拟合，增加泛化能力还解决了费时的问题
        self.act = act  # 这里采用线性整流函数,(relu)Rectified Linear Unit,
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # parameter做为储存权重的类，对weights进行了初始化
        self.reset_parameters()

    # 定义参数（权重）
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    # 前向传播
    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)  # trainning的时候加drop out参数
        support = torch.mm(input, self.weight)  # 矩阵相乘
        output = torch.spmm(adj, support)  # 矩阵相乘
        output = self.act(output)  # 应用激活函数
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 构造解码器
class SampleDecoder(Module):
    def __init__(self, act=torch.sigmoid):  # 将sigmoid做为解码器输出的激活函数
        super(SampleDecoder, self).__init__()
        self.act = act

    def forward(self, zx, zy):
        sim = (zx * zy).sum(1)  # 求矩阵行的和
        sim = self.act(sim)  # 输出结果
        return sim
