import torch
import torch.nn as nn
from architecture import *

class CBSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size = 3, stride = 1, 
                 padding = 1, groups = 1
                ):
        super(CBSBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2d(self.in_channels, 
                              self.out_channels, self.kernel_size, 
                              self.stride, self.padding, 
                              bias=False, groups=groups) # đặt tham số groups ở đây phục vụ cho việc có sử dụng tích chập chiều sâu hay không
                                                         # Ta cần có lý luận để thuyết phục cô Hải.
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        self.activation = nn.SiLU()
    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut = True):
        super(BottleNeck, self).__init__()
        
        self.c_ = int(out_channels)
        self.conv1 = CBSBlock(in_channels, self.c_ , kernel_size=1, padding=0)
        self.conv2 = CBSBlock(self.c_ , out_channels)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n = 1, shortcut = True, e_CSP = 0.5):
        super(C3, self).__init__()
        self.c_ = int(out_channels * e_CSP)
        self.conv1 = CBSBlock(in_channels, self.c_, 1, 1, 0)
        self.conv2 = CBSBlock(in_channels, self.c_, 1, 1, 0)
        self.conv3 = CBSBlock(int(self.c_ / e_CSP), out_channels, 1, 1, 0)
        self.m = nn.Sequential(*[BottleNeck(self.c_, self.c_, shortcut=shortcut) for _ in range(n)])
    
    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SPPF, self).__init__()
        self.c_ = int(in_channels / 2)
        
        # giam channels
        self.conv1 = CBSBlock(in_channels=in_channels, out_channels=self.c_, kernel_size=1, padding=0)
        self.conv2 = CBSBlock(in_channels=self. c_ * 4, out_channels=out_channels, kernel_size=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        
        return self.conv2(torch.cat((x, y1, y2, y3), dim=1))

class Upsample(nn.Module):
    def  __init__(self, out_channels = None, scale_fator = 2,  interpolation = "nearest"):
        super(Upsample, self).__init__()
        self.out_channels = out_channels
        self.scale_fator = scale_fator
        self.interpolation = interpolation
        
        self.upsample = nn.Upsample(scale_factor=scale_fator, mode="nearest")
    def forward(self, x):
        return self.upsample(x)
    
class Concat(nn.Module):
    def __init__(self, dim = 1):
        super(Concat, self).__init__()
        self.dim = dim
    def forward(self, inputs):
        return torch.concat(inputs, self.dim)