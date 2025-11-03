from functools import partial

from models.segformer.encoder import *
from models.segformer.decoder import *


class SegFormer(nn.Module):
    def __init__(self,
                 num_classes,
                 patch_size=4,
                 embed_dims=[32, 64, 160, 256],
                 depths=[2, 2, 2, 2],
                 dec_outChannels=256,
                 dropout_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=dropout_ratio),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))

        self.encoder = MixVisionTransformer(
            patch_size=patch_size, embed_dims=embed_dims, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths, sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=dropout_ratio)

        self.decoder = SegFormerHead(e_dim=dec_outChannels,
                                     feature_strides=[4, 8, 16, 32],
                                     embed_dims=embed_dims)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out), mean=0)

    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)  # here output will be B x C x H/8 x W/8
        output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True)  # now its same as input
        #  bilinear interpol was used originally
        return output


if __name__ == '__main__':
    from torchsummary import summary

    model = SegFormer(num_classes=2)
    # summary(model, (3,1024,2048))

    y = torch.randn((6, 3, 512, 512))  # .to('cuda' if torch.cuda.is_available() else 'cpu')
    x = model.forward(y)
    print(x.shape)
    print('seg_logits is contiguous:', x.is_contiguous())


