import torch
import numpy
import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmdet.models.necks.aspp_module import ASPP

class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        # self.artous_rates = atrous_rates=(12, 24, 36)
        # self.aspp=ASPP(in_dim, atrous_rates)


        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //8 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):

        m_batchsize, C, height, width =x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        # aspp_k = self.aspp(self.key_conv(x))
        # proj_key = aspp_k.view(m_batchsize, -1, width * height)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        # aspp_v = self.aspp(self.value_conv(x))
        # proj_value = aspp_v.view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
