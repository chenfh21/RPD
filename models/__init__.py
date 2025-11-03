from .deeplabv3plus_model import *
from .ori_unet import UNet
from .RPDNet import RPDNet
from .erfnet import ERFNetModel
from .segnext import SegNext
from .segformer import SegFormer
# from .module import *
from .model_multimetrics import *
from .losses import *
from .RPD_ops import *


# from .ori_unet import *

def get_backbone(cfg: Dict) -> nn.Module:
    num_classes = cfg['backbone']['num_classes']
    pretrained = cfg['backbone']['pretrained']
    deploy = cfg['backbone']['deploy']
    convert = cfg['backbone']['convert']

    if cfg['backbone']['name'] == 'deeplabv3plus_resnet50':
        return deeplabv3plus_resnet50(output_stride=16, num_classes=num_classes, pretrained_backbone=pretrained)

    if cfg['backbone']['name'] == 'erfnet':
        return ERFNetModel(num_classes, pretrained=pretrained)

    if cfg['backbone']['name'] == 'baseline':
        return UNet(num_classes)

    if cfg['backbone']['name'] == 'RPDNet':
        return RPDNet(num_classes, deploy=deploy, convert=convert)
        # return RPDNet(num_classes)

    if cfg['backbone']['name'] == 'segnext':
        return SegNext(num_classes)

    if cfg['backbone']['name'] == 'segformer':
        return SegFormer(num_classes)

    raise ValueError('The requested backbone is not supported.')
