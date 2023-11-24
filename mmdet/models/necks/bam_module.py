import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelGate, self).__init__()
        mid_channel = channel // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
        return out

class SpatiaGate(nn.Module):
    # dilation value and reduction ratio, set d = 4 r = 16
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatiaGate, self).__init__()
        self.gate_s = nn.Sequential()
        # 1x1 + (3x3)*2 + 1x1
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):#进行多个空洞卷积，丰富感受野
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                                             kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))  # 1×H×W

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatiaGate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor



# import numpy as np
# import torch
# from torch import nn
# from torch.nn import init
#
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(ChannelAttention, self).__init__()
#         mid_channel = channel // reduction
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.shared_MLP = nn.Sequential(
#             nn.Linear(in_features=channel, out_features=mid_channel),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=mid_channel, out_features=channel)
#         )
#
#     def forward(self, x):
#         avg = self.avg_pool(x).view(x.size(0), -1)
#         out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return out
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, channel, reduction=16, dilation_conv_num=2, dilation_rate=4):
#         super(SpatialAttention, self).__init__()
#         mid_channel = channel // reduction
#         self.reduce_conv = nn.Sequential(
#             nn.Conv2d(channel, mid_channel, kernel_size=1),
#             nn.BatchNorm2d(mid_channel),
#             nn.ReLU(inplace=True)
#         )
#         dilation_convs_list = []
#         for i in range(dilation_conv_num):
#             dilation_convs_list.append(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
#             dilation_convs_list.append(nn.BatchNorm2d(mid_channel))
#             dilation_convs_list.append(nn.ReLU(inplace=True))
#         self.dilation_convs = nn.Sequential(*dilation_convs_list)
#         self.final_conv = nn.Conv2d(mid_channel, 1, kernel_size=1)
#
#     def forward(self, x):
#         y = self.reduce_conv(x)
#         x = self.dilation_convs(y)
#         out = self.final_conv(y).expand_as(x)
#         return out
#
#
# class BAM(nn.Module):
#     """
#         BAM: Bottleneck Attention Module
#         https://arxiv.org/pdf/1807.06514.pdf
#     """
#     def __init__(self, channel):
#         super(BAM, self).__init__()
#         self.channel_attention = ChannelAttention(channel)
#         self.spatial_attention = SpatialAttention(channel)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         att = 1 + self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))
#         return att * x
#



