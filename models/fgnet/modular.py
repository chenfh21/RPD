import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .timm_droppath import DropPath


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LCA(nn.Module):
    def __init__(self, dim: int):
        super(LCA, self).__init__()
        self.channels = dim
        self.k = self.compute_adaptive_k(dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv2d(dim, dim, kernel_size=self.k, padding=self.k // 2, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.gap(x)
        scale = self.conv1d(scale)
        scale = self.sigmoid(scale)
        return scale * x

    def compute_adaptive_k(self, C):
        if C <= 1:
            return 1 
        log2_C = math.log2(C)
        mid_val = (1 + log2_C) / 2
        nearest_int = round(mid_val) 

        if nearest_int % 2 == 1:
            return nearest_int
        else:
            return nearest_int + 1


class FFA(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.cross_chan_atten = LCA(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.cross_chan_atten(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x = self.gamma[:, None, None] * x
        x = input + self.drop_path(x)
        return x


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__()
        p = d * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CFF(nn.Module):
    def __init__(self, ca, cs, cd, dilation_rates=[1, 2, 3, 4]):
        super(CFF, self).__init__()
        self.total_c = ca + cs + cd
        self.num_branches = len(dilation_rates)

        self.total_c = _make_divisible(ch=self.total_c, divisor=self.num_branches, min_ch=None)
        self.c_split = self.total_c // self.num_branches

        self.cv_merge = Conv(self.total_c, self.total_c, 3, 1, 1)

        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(Conv(self.c_split, self.c_split, 3, 1, d=rate))
        self.cv_concat = Conv(self.total_c, cd, 1, 1, 1)

    def forward(self, fa: torch.Tensor, fs: torch.Tensor, fd: torch.Tensor) -> torch.Tensor:
        concat_f = torch.cat([fa, fs, fd], dim=1)
        fi = self.cv_merge(concat_f)

        splits = torch.chunk(fi, chunks=self.num_branches, dim=1)

        y = []
        for i, conv_i in enumerate(self.branches):
            branch_out = conv_i(splits[i])
            y.append(branch_out)

        concat_out = torch.cat(y, dim=1)
        return self.cv_concat(concat_out)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Up2(nn.Module):
    def __init__(self, ca, cs, cd):
        super(Up2, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cff = CFF(ca, cs, cd)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        target_size = x3.shape[2:]
        x1 = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)

        cff_out = self.cff(x1, x2, x3)
        out = self.up(cff_out)
        return out


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1))
