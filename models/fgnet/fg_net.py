from collections import OrderedDict
from typing import Dict

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from models.fgnet.modular import *
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FineGrainedBranch(nn.Module):
    def __init__(self, in_channels=3, target_channels=[32, 64, 128]):
        super(FineGrainedBranch, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, target_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(target_channels[0]),
            nn.GELU()
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(target_channels[0], target_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(target_channels[1]),
            nn.GELU(),
            FFA(target_channels[1])
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(target_channels[1], target_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(target_channels[2]),
            nn.GELU(),
            FFA(target_channels[2])
        )

    def forward(self, x):
        stage0_out = self.stage0(x)
        stage1_out = self.stage1(stage0_out)
        stage2_out = self.stage2(stage1_out)
        # stage3_out = self.stage3(stage2_out)

        return {
            "stage0": stage0_out,
            "stage1": stage1_out,
            "stage2": stage2_out,
        }


class FGNet(nn.Module):
    def __init__(self,
                 num_classes=2,
                 pretrained: bool = False,
                 target_channels=[32, 64, 128]
                 ):
        super().__init__()
        self.num_classes = num_classes
        if pretrained:
            backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        else:
            backbone = mobilenet_v3_large(weights=None)
        encoder = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        self.stage_out_channels = [encoder[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(
            stage_indices)])  # {'1': 'stage0', '3': 'stage1', '6': 'stage2', '12': 'stage3', '15': 'stage4'}
        self.encoder = IntermediateLayerGetter(encoder, return_layers=return_layers)

        # Fine_grained branches
        self.fine_grained = FineGrainedBranch(in_channels=3, target_channels=target_channels)

        c1 = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c1, self.stage_out_channels[3])

        self.up2 = Up2(target_channels[1], self.stage_out_channels[3], self.stage_out_channels[2])
        self.up3 = Up2(target_channels[2], self.stage_out_channels[2], self.stage_out_channels[1])

        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=self.num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        backbone_out = self.encoder(x)
        fg_out = self.fine_grained(x)

        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'], fg_out['stage1'])
        x = self.up3(x, backbone_out['stage1'], fg_out['stage2'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x
