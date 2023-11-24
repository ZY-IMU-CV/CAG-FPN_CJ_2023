import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale

class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gamma = Scale(0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, c, height, width = x.size()
        y = x.view(batch_size, c, -1)#HW*c
        feat_y_transpose = y.view(batch_size, c, -1).permute(0, 2, 1)#c*HW
        attention_y = torch.bmm(y, feat_y_transpose)#c*c
        attention_new = torch.max(attention_y, dim=-1, keepdim=True)[0].expand_as(attention_y) - attention_y
        attention_y = self.softmax(attention_new)

        feat_a = x.view(batch_size, c, height * width)#HW*c
        attention = torch.bmm(attention_y, feat_a)#HW*c
        attention = attention.view(batch_size, c,  height, width)#H*W*c
        out = self.gamma(attention) + x
        return out
