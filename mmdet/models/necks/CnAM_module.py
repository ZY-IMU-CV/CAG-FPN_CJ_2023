import torch
import torch.nn as nn
class CnAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CnAM, self).__init__()
        # 原文中对应的P, Z, S
        self.Z_conv = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.P_conv = nn.Conv2d(in_channels, out_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)

    # CnAM使用了FPN中的F5和CEM输出的特征图F
    def forward(self, F5, F):
        m_batchsize, C, width, height = F5.size()

        proj_query = self.P_conv(F5).view(m_batchsize, -1, width*height).permute(0, 2, 1)  # B x N x C''

        proj_key = self.Z_conv(F5).view(m_batchsize, -1, width * height)  # B x C'' x N

        S = torch.bmm(proj_query, proj_key).view(m_batchsize, width * height, width, height)  # B x N x W x H
        attention_S = self.sigmoid(self.avg(S).view(m_batchsize, -1, width, height))  # B x 1 x W x H

        proj_value = self.value_conv(F)

        out = proj_value * attention_S  # B x W x H

        return out