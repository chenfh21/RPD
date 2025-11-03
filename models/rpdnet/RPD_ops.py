import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def adjust_out_channels(out_channels, groups):
    if out_channels % groups == 0:
        return out_channels
    else:
        return math.ceil(out_channels / groups) * groups


class Conv2d(nn.Module):
    def __init__(self, op, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        out_channels = adjust_out_channels(out_channels, groups)
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.op = op

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.op is None:
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return self.op(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


## cd, ad, rd convolutions
## theta could be used to control the vanilla conv components
## theta = 0 reduces the function to vanilla conv, theta = 1 reduces the function to pure pdc
def createConvFunc(op_type, theta=0.7):
    assert op_type in ['cv', 'cd', 'ad', 'rd', 'erd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    assert theta >= 0 and theta <= 1.0, 'theta should be within (0, 1]'

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True) * theta
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc

        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - theta * weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.tensor([0], dtype=torch.float32, device='cuda').repeat(shape[0], shape[1], 5 * 5)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:] * theta
            buffer[:, :, 12] = weights[:, :, 0] * (1 - theta)
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    elif op_type == 'erd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2]
            assert weights.size(2) == 3 and weights.size(3) == 3

            weights_conv = (weights - weights.sum(dim=[2, 3], keepdim=True)) * theta
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=2, dilation=2, groups=groups)
            return y

        return func

    else:
        print('impossible to be here unless you force that')
        return None


def convert_pdc(op_type, weight):
    if op_type == 'cv':
        return weight
    elif op_type == 'cd':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight
    elif op_type == 'ad':
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return weight_conv
    elif op_type == 'rd':
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return buffer
    elif op_type == 'erd':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight

    raise ValueError("wrong op {}".format(str(op_type)))


blocks_config = {
    'cv': {
        'block_opd': 'cv'
    },
    'cccc': {
        'block_op1': 'cv',
        'block_op2': 'cv',
        'block_op3': 'cv',
        'block_op4': 'cv',
    },

    'cdc': {
        'block_opd1': 'cv',
        'block_opd2': 'cv',
        'block_opd3': 'cv',
        'block_opc': 'cd'
    },

    'adc': {
        'block_opd1': 'cv',
        'block_opd2': 'cv',
        'block_opd3': 'cv',
        'block_opa': 'ad'
    },

    'rdc': {
        'block_opd1': 'cv',
        'block_opd2': 'cv',
        'block_opd3': 'cv',
        'block_opr': 'rd'
    },

    'cadc': {
        'block_opd1': 'cv',
        'block_opc': 'cd',
        'block_opd2': 'cv',
        'block_opa': 'ad'
    },

    'pdc': {
        'block_opd': 'cv',
        'block_opc': 'cd',
        'block_opa': 'ad',
        'block_opr': 'rd',
    },

    'epdc': {
        'block_opd': 'cv',
        'block_opc': 'cd',
        'block_opa': 'ad',
        'block_opr': 'erd',
    },

   'ccpdc': {
        'block_opd': 'cv',
        'block_1c': 'cd',
        'block_2c': 'cd',
        'block_opa': 'ad',
        'block_opr': 'rd',
    }
}


def config_block(block: str) -> List:
    block_options = list(blocks_config.keys())
    assert block in block_options, f'unrecognized block_ops, please choose from {block_options}'

    cv_list = [createConvFunc(v, theta=0.5) for k, v in blocks_config[block].items()]

    return cv_list


# 生成函数/方法操作
def config_block_converted(block: str) -> List:
    block_options = list(blocks_config.keys())
    assert block in block_options, f'unrecognized block_ops, please choose from {block_options}'

    cv_list = [op for op_type, op in blocks_config[block].items()]

    return cv_list


def convert_block(state_dict, config):  # state_dict -> block_config
    cv_ops = config_block_converted(config)  # config -> e.g. 'pdc'.操作选项
    new_dict = {}

    for pname, p in state_dict.items():
        if pname.endswith('rbr_conv.0.conv.weight'):
            new_dict[pname] = convert_pdc(cv_ops[0], p)
        elif pname.endswith('rbr_conv.1.conv.weight') and p.shape[2] != 1:
            new_dict[pname] = convert_pdc(cv_ops[1], p)
        elif pname.endswith('rbr_conv.2.conv.weight') and p.shape[2] != 1:
            new_dict[pname] = convert_pdc(cv_ops[2], p)
        elif pname.endswith('rbr_conv.3.conv.weight') and p.shape[2] != 1:
            new_dict[pname] = convert_pdc(cv_ops[3], p)
        else:
            new_dict[pname] = p

    return new_dict
