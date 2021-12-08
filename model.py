import torch
import torch.nn as nn
import torch.nn.functional as F

## 模型结构定义，框架细节

from layers import *

class LinTrans(nn.Module):

    # 一般把具有可学习参数的层（全连接层，卷积层）放在init中
    # 把不具有可学习参数的层放在构造函数中
    # 所有放在构造函数init里的层都是这个模型的“固有属性”
    def __init__(self, layers, dims):
        super(LinTrans, self).__init__() #调用父类构造函数
        # moduleList 一个特殊的module，包含几个子module，像list一样使用，但不能直接将输入传给moduleList
        # 用于存储任意数量的nn.module
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):
        
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled

    # 必须重写forward方法
    # 实现模型的功能，实现各个层之间的连接关系的核心
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.scale(out)
        out = F.normalize(out)
        return out

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
