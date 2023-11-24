import torch
import torch.nn as nn
import math
from mmcv.cnn import ConvModule, xavier_init
import torch.nn.functional as F

from .SE_weight_module import SEWeightModule

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 3, 3, 3], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = ConvModule(inplans, planes//4, kernel_size=conv_kernels[0], padding=0,
                            stride=stride, groups=conv_groups[0], dilation=1)
        self.conv_2 = ConvModule(inplans, planes//4, kernel_size=conv_kernels[1], padding=6,
                            stride=stride, groups=conv_groups[1], dilation=6)
        self.conv_3 = ConvModule(inplans, planes//4, kernel_size=conv_kernels[2], padding=12,
                            stride=stride, groups=conv_groups[2], dilation=12)
        self.conv_4 = ConvModule(inplans, planes//4, kernel_size=conv_kernels[3], padding=18,
                            stride=stride, groups=conv_groups[3], dilation=18)
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        x1 = F.interpolate(x1, x4.size()[2:])
        x2 = F.interpolate(x2, x4.size()[2:])
        x3 = F.interpolate(x3, x4.size()[2:])
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out