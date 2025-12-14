from .deeplabv3plus.modeling import deeplabv3plus_resnet50
from .rpdnet import RPDNet
from .ERFNet import ERFNetModel
from .segnext import SegNext
from .segformer import SegFormer
from .EfficientVit import EfficientViTSeg
from .model_multimetrics import *
from .fgnet import FGNet
from .losses import *
from .RPD_ops import *


def get_backbone(cfg: Dict) -> nn.Module:
    num_classes = cfg['backbone']['num_classes']
    pretrained = cfg['backbone']['pretrained']

    if cfg['backbone']['name'] == 'deeplabv3plus_resnet50':
        return deeplabv3plus_resnet50(output_stride=16, num_classes=num_classes, pretrained_backbone=pretrained)

    if cfg['backbone']['name'] == 'erfnet':
        return ERFNetModel(num_classes, pretrained=pretrained)

    if cfg['backbone']['name'] == 'segnext':
        return SegNext(num_classes)

    if cfg['backbone']['name'] == 'segformer':
        return SegFormer(num_classes)

    if cfg['backbone']['name'] == 'efficientvitseg':
        return EfficientViTSeg(num_classes)

    if cfg['backbone']['name'] == 'fgnet':
        return FGNet(num_classes)

    if cfg['backbone']['name'] == 'RPDNet':
        deploy = cfg['backbone']['deploy']
        convert = cfg['backbone']['convert']
        return RPDNet(num_classes, deploy=deploy, convert=convert)

    raise ValueError('The requested backbone is not supported.')






