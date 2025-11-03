from collections import OrderedDict

from typing import Dict

from torch import nn
from .resnet_backbone import *


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


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True))
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)  # 有的也设置为0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )


class DeepLabPlusHead(nn.Sequential):
    def __init__(self, in_channels: int,
                 num_classes: int,
                 low_level_inchannels: int,
                 low_level_outchannels: int = 48,
                 aspp_dilate=[12, 24, 36]) -> None:
        super(DeepLabPlusHead, self).__init__()
        self.ASPP = ASPP(in_channels, aspp_dilate)

        self.low_level_features = nn.Sequential(
            nn.Conv2d(low_level_inchannels, low_level_outchannels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_outchannels),
            nn.ReLU(inplace=True)
        )

        in_clachannels = low_level_outchannels + 256  # channels为304
        self.classifier = nn.Sequential(
            nn.Conv2d(in_clachannels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        low_level_features = self.low_level_features(x['low_level'])
        out_features = self.ASPP(x['out'])

        out_features = nn.functional.interpolate(out_features, size=low_level_features.shape[-2:], mode='bilinear',
                                                 align_corners=False)

        res = torch.cat([out_features, low_level_features], dim=1)

        return self.classifier(res)


class DeepLabV3Plus(nn.Module):
    """
    Implements DeepLabV3Plus model from
    '"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"'

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
    """

    def __init__(self, backbone, out_planes, num_classes, low_level_planes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.classifier = DeepLabPlusHead(
            out_planes,
            num_classes,
            low_level_planes
        )

    def forward(self, x: Tensor):  # x.shape= [1 3 480 480]
        input_shape = x.shape[-2:]
        # contact: features is a dict of tensors
        features = self.backbone(x)

        x = self.classifier(features)
        # 使用双线性插值还原回原图尺寸
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x


def deeplabv3plus_resnet50(output_stride, num_classes=21, pretrained_backbone=False):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet50(replace_stride_with_dilation=replace_stride_with_dilation)

    if pretrained_backbone:
        backbone.load_state_dict(torch.load("resnet50.pth"))

    out_planes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3Plus(backbone, out_planes, num_classes, low_level_planes)

    return model


def deeplabv3plus_resnet101(output_stride, num_classes=21, pretrained_backbone=False):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet101(replace_stride_with_dilation=replace_stride_with_dilation)

    if pretrained_backbone:
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_planes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3Plus(backbone, out_planes, num_classes, low_level_planes)

    return model

