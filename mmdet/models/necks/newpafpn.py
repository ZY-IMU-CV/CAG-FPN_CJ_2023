# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN
from .xunhuanmodule import FPNModule

@NECKS.register_module()
class PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PAFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

            #初始化循环模块
            self.FPN=FPNModule()


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        used_backbone_levels = len(3)
        num = 2
        FPN_list = []
        for i in range(num):
            FPN_list.append(FPNModule())

        for i, fpn in FPN_list:
            X = fpn(X)
        X.extend([
            self.pafpn_convs[i](inter_outs = [
            self.fpn_convs[i](laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]) for i in range(used_backbone_levels)
        ])
            for i in range(0, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(X):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    X.append(F.max_pool2d(X[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    X.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    X.append(self.fpn_convs[used_backbone_levels](
                        laterals=[
                                     lateral_conv(inputs[-1 + self.start_level])
                                     for i, lateral_conv in enumerate(self.lateral_convs)
                                 ]))
                elif self.add_extra_convs == 'on_output':
                    X.append(self.fpn_convs[used_backbone_levels](X[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        X.append(self.fpn_convs[i](F.relu(X[-1])))
                    else:
                        X.append(self.fpn_convs[i](X[-1]))
        return tuple(X)
