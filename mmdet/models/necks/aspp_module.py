# import torch
# import torch.nn as nn
# from mmcv.cnn import ConvModule
#
#
# class ASPPModule(nn.ModuleList):
#     """Atrous Spatial Pyramid Pooling (ASPP) Module.
#
#     Args:
#         dilations (tuple[int]): Dilation rate of each layer.
#         in_channels (int): Input channels.
#         channels (int): Channels after modules, before conv_seg.
#         conv_cfg (dict|None): Config of conv layers.
#         norm_cfg (dict|None): Config of norm layers.
#         act_cfg (dict): Config of activation layers.
#     """
#
#     def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
#                  act_cfg):
#         super(ASPPModule, self).__init__()
#         self.dilations = dilations
#         self.in_channels = in_channels
#         self.channels = channels
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg
#         for dilation in dilations:
#             self.append(
#                 ConvModule(
#                     self.in_channels,
#                     self.channels,
#                     1 if dilation == 1 else 3,
#                     dilation=dilation,
#                     padding=0 if dilation == 1 else dilation,
#                     conv_cfg=self.conv_cfg,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))
#
#     def forward(self, x):
#         """Forward function."""
#         aspp_outs = []
#         for aspp_module in self:
#             aspp_outs.append(aspp_module(x))
#
#         return aspp_outs
#
#
#
from torch import nn
import torch
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 512
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x),)
        res = torch.cat(res, dim=1)
        return self.project(res)
# aspp = ASPP(256,[6,12,18])
# x = torch.rand(2,256,13,13)
# print(aspp(x).shape)