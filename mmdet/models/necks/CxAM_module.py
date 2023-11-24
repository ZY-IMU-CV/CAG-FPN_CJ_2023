import torch
import torch.nn as nn
class CxAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CxAM, self).__init__()
        self.key_conv = nn.Conv2d(in_channels, out_channels//reduction, 1)
        self.query_conv = nn.Conv2d(in_channels, out_channels//reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)   # B x N x C'

        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # B x C' x N

        R = torch.bmm(proj_query, proj_key).view(m_batchsize, width*height, width, height)  # B x N x W x H
        # 先进行全局平均池化, 此时 R 的shape为 B x N x 1 x 1, 再进行view, R 的shape为 B x 1 x W x H
        attention_R = self.sigmoid(self.avg(R).view(m_batchsize, -1, width, height))    # B x 1 x W x H

        proj_value = self.value_conv(x)

        out = proj_value * attention_R  # B x W x H

        return out