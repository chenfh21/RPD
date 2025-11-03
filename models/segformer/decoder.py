import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from pdc_modules.batchnorm import NormLayer


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """Linear Embedding"""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvBNRelu(nn.Module):

    @classmethod
    def _same_paddings(cls, kernel):
        if kernel == 1:
            return 0
        elif kernel == 3:
            return 1

    def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel)

        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = NormLayer(outChannels, norm_type='sync_bn')
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self,
                 e_dim=256,
                 feature_strides=[4, 8, 16, 32],
                 embed_dims=[32, 64, 160, 256],
                 in_index=[0, 1, 2, 3]):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(embed_dims)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = embed_dims

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=e_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=e_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=e_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=e_dim)

        self.linear_fuse = ConvBNRelu(inChannels=e_dim * 4, outChannels=e_dim, kernel=1)

        self.in_index = in_index

    def forward(self, inputs):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x = [inputs[i] for i in self.in_index]

        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False).contiguous()

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False).contiguous()

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False).contiguous()

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()

        fused = torch.cat([_c4, _c3, _c2, _c1], dim=1).contiguous()  # 关键
        return self.linear_fuse(fused)
