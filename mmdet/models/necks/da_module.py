import torch.nn as nn
from mmdet.models.necks.dapam_module import PAM_Module
from mmdet.models.necks.dacam_module import _ChannelAttentionModule
# class DANetHead(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer):
#         super(DANetHead, self).__init__()
#         inter_channels = in_channels // 4  # in_channels=2018，通道数缩减为512
#
#         self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm_layer(inter_channels), nn.ReLU())
#         self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm_layer(inter_channels), nn.ReLU())
#
#         self.sa = PAM_Module(inter_channels)  # 空间注意力模块
#         self.sc = _ChannelAttentionModule(inter_channels)  # 通道注意力模块
#
#         self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm_layer(inter_channels), nn.ReLU())
#         self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
#                                     norm_layer(inter_channels), nn.ReLU())
#
#         # nn.Dropout2d(p,inplace)：p表示将元素置0的概率；inplace若设置为True，会在原地执行操作。
#         self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))  # 输出通道数为类别的数目
#         self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
#         self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
#
#     def forward(self, x):
#         # 经过一个1×1卷积降维后，再送入空间注意力模块
#         feat1 = self.conv5a(x)
#         sa_feat = self.sa(feat1)
#         # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
#         sa_conv = self.conv51(sa_feat)
#         sa_output = self.conv6(sa_conv)
#
#         # 经过一个1×1卷积降维后，再送入通道注意力模块
#         feat2 = self.conv5c(x)
#         sc_feat = self.sc(feat2)
#         # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
#         sc_conv = self.conv52(sc_feat)
#         sc_output = self.conv7(sc_conv)
#
#         feat_sum = sa_conv + sc_conv  # 两个注意力模块结果相加
#         sasc_output = self.conv8(feat_sum)  # 最后再送入1个有dropout的1×1卷积中
#
#         output = [sasc_output]
#         output.append(sa_output)
#         output.append(sc_output)
#         return tuple(output)  # 输出模块融合后的结果，以及两个模块各自的结果


class DANetHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        inter_channels = in_channels // 4  # in_channels=2018，通道数缩减为512
        self.position_attention_module = PAM_Module(inter_channels)
        self.channel_attention_module = _ChannelAttentionModule(inter_channels)

    def forward(self, input):
        bs, c, h, w = input.shape
        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)
        p_out = p_out.permute(0, 2, 1).view(bs, c, h, w)
        c_out = c_out.view(bs, c, h, w)
        return p_out + c_out