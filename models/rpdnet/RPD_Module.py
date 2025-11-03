import torch
import copy
import os
from torch import nn, Tensor
from typing import Tuple, List
from collections import OrderedDict
import math
from pdc_modules.RPD_ops import *


def _make_divisible(ch, divisor=8, min_ch=None):
    """
            This function is taken from the original tf repo.
            It ensures that all layers have a channel number that is divisible by 8
            It can be seen here:
            https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SEblock(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4):
        super(SEblock, self).__init__()
        squeeze_channels = _make_divisible(in_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


def conv_bn(op, in_channels, out_channels, kernel_size, padding, stride=1, groups=1, dilation=1, bias=False):
    if padding is not None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', Conv2d(op, in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def conv_bn_convert(op, in_channels, out_channels, kernel_size, groups=1, bias=False):
    padding = (kernel_size - 1) // 2
    result = nn.Sequential()
    if op == 'rd':
        result.add_module('conv',
                          Conv2d(op=None, in_channels=in_channels, out_channels=out_channels, kernel_size=5,
                                 padding=2, groups=groups, dilation=1, bias=bias))
    else:
        result.add_module('conv',
                          Conv2d(op=None, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 padding=padding, groups=groups, bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RPD(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True,
                 deploy=False,
                 use_se=False,
                 convert=False,  # 用于pdc训练后转换成为vanilla conv
                 num_conv_branches: int = 1):
        super(RPD, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2

        self.activation = nn.ReLU()

        if use_se:
            self.se = SEblock(in_channels=out_channels, squeeze_factor=16)
        else:
            self.se = nn.Identity()
        self.kel_convert = 5
        if deploy:  # 这里为了使用轻量化操作我使用的是深度卷积
            self.rep_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size,
                                      stride=stride, padding=padding, groups=groups, bias=bias)

        else:  # identity + 1x1 dconv + pdc(include 3x3 dconv) dconv
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            # adding pdc branches
            rbr_conv = list()
            if convert:
                self.pdc = config_block_converted(pdc)
                for op in self.pdc:
                    rbr_conv.append(
                        conv_bn_convert(op, in_channels, out_channels, kernel_size=kernel_size, groups=groups))
            else:
                self.pdc = config_block(pdc)
                for op in self.pdc:
                    rbr_conv.append(conv_bn(op, in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, groups=groups))

            rbr_conv = rbr_conv * num_conv_branches
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # 用kernel_size控制是否有pointwise卷积
            self.rbr_1x1 = conv_bn(op=None, in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=stride, padding=padding, groups=groups) if kernel_size > 1 else None

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, 'rep_conv'):
            # return self.activation(self.se(self.rep_conv(inputs)))
            return self.se(self.activation(self.rep_conv(inputs)))

        identity_out = self.rbr_identity(inputs) if self.rbr_identity is not None else 0

        if self.rbr_1x1 is not None:
            point_out = self.rbr_1x1(inputs)
        else:
            point_out = 0

        out = identity_out + point_out
        for ix in range(len(self.rbr_conv)):
            out += self.rbr_conv[ix](inputs)

        # return self.activation(self.se(out))
        return self.se(self.activation(out))

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    # (self.in_channels, input_dim, self.kel_convert, self.kel_convert),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device)

                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                    # kernel_value[i, i % input_dim, self.kel_convert // 2, self.kel_convert // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    # def _get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     kernel_1x1 = 0
    #     bias_1x1 = 0
    #     if self.rbr_1x1 is not None:
    #         kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
    #         kernel_1x1 = F.pad(kernel_1x1, [self.kel_convert // 2] * 4)
    #
    #     kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_identity)
    #
    #     # get weights and bias of conv branches
    #     kernel_conv = 0
    #     bias_conv = 0
    #     for ix in range(len(self.rbr_conv) - 1):
    #         _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
    #         bias_conv += _bias
    #         # add to the central part
    #         kernel_conv += nn.functional.pad(_kernel, [(self.kel_convert - self.kernel_size) // 2] * 4)
    #
    #     last_kernel, last_bias = self._fuse_bn_tensor(self.rbr_conv[-1])
    #
    #     kernel_conv += last_kernel
    #     bias_conv += last_bias
    #
    #     return kernel_conv + kernel_1x1 + kernel_identity, bias_conv + bias_1x1 + bias_identity
    def _get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel_1x1 = 0
        bias_1x1 = 0
        if self.rbr_1x1 is not None:
            kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernel_1x1 = F.pad(kernel_1x1, [self.kernel_size // 2] * 4)

        kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_identity)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(len(self.rbr_conv) - 1):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            bias_conv += _bias    # add to the central part
            kernel_conv += _kernel

        last_kernel, last_bias = self._fuse_bn_tensor(self.rbr_conv[-1])

        kernel_conv += rd_rep(last_kernel)
        bias_conv += last_bias

        return kernel_conv + kernel_1x1 + kernel_identity, bias_conv + bias_1x1 + bias_identity

    def switch_to_deploy(self):
        if hasattr(self, 'rep_conv'):
            return
        kernel, bias = self._get_equivalent_kernel_bias()

        self.rep_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                  out_channels=self.rbr_conv[0].conv.out_channels,
                                  kernel_size=self.rbr_conv[0].conv.kernel_size,  # kernel_size以radialconv为主，5
                                  stride=self.rbr_conv[0].conv.stride,
                                  padding=self.rbr_conv[0].conv.padding,  # kernel_size以radialconv为主，padding=2
                                  dilation=self.rbr_conv[0].conv.dilation,
                                  groups=self.rbr_conv[0].conv.groups,
                                  bias=True)

        self.rep_conv.weight.data = kernel
        self.rep_conv.bias.data = bias

        # delete un-used branches
        for param in self.parameters():
            param.detach_()

        self.__delattr__('rbr_conv')
        if hasattr(self, 'rbr_1x1'):
            self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        # 在repvgg种还删除了'id_tensor'
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

def rd_rep(kernel):
    if kernel.shape[-1] > 3:
        buffer = torch.zeros(kernel.shape[0], kernel.shape[1], 3, 3, device=kernel.device)
        buffer[:, :, 0, 0] = kernel[:, :, 0, 0] - kernel[:, :, 1, 1]
        buffer[:, :, 0, 1] = kernel[:, :, 0, 2] - kernel[:, :, 1, 2]
        buffer[:, :, 0, 2] = kernel[:, :, 0, 4] - kernel[:, :, 1, 3]
        buffer[:, :, 1, 0] = kernel[:, :, 2, 0] - kernel[:, :, 2, 1]
        buffer[:, :, 1, 2] = kernel[:, :, 2, 4] - kernel[:, :, 2, 3]
        buffer[:, :, 2, 0] = kernel[:, :, 4, 0] - kernel[:, :, 3, 1]
        buffer[:, :, 2, 1] = kernel[:, :, 4, 2] - kernel[:, :, 3, 2]
        buffer[:, :, 2, 2] = kernel[:, :, 4, 4] - kernel[:, :, 3, 3]
        buffer[:, :, 1, 1] = 0  # Placeholder for the missing element
    else:
        buffer = kernel
    return buffer

def RPD_model_deploy(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        save_path_convert = os.path.join(save_path, 'deploy_model.pth')
        torch.save(model.state_dict(), save_path_convert)
    return model
