import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sweight = Parameter(torch.zeros(1, channel //  groups, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // groups, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // groups, channel // groups)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)

        # spatial attention
        xs = self.gn(x)
        xs = self.sweight * xs + self.sbias
        xs = x * self.sigmoid(xs)

        out = xs.reshape(b, -1, h, w)


        out = self.channel_shuffle(out, 1)
        return  out
