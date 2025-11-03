import torch
import math
from models.batchnorm import *


class StemConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(StemConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, kernel_size=3, stride=2, padding=1),
            NormLayer(out_ch // 2, norm_type='sync_bn'),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=3, stride=2, padding=1),
            NormLayer(out_ch, norm_type='sync_bn'))

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()

        return x, H, W


class DownSample(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, k=3, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=k, stride=stride, padding=k // 2)
        self.norm = NormLayer(embed_dim, norm_type='sync_bn')

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = self.norm(x)
        return x, H, W


class LayerScale(nn.Module):
    def __init__(self, inchannels, init_value=1e-2):
        super().__init__()
        self.inchannels = inchannels
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones(inchannels), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1)
            return scale * x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inchannels}, init_value={self.init_value}'


class MSCA(nn.Module):
    def __init__(self, dim):
        super(MSCA, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=(1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, kernel_size=(21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x.clone()
        x = self.conv0(x)

        attn_0 = self.conv0_1(x)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(x)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(x)
        attn_2 = self.conv2_2(attn_2)

        attn = x + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * identity


class BlockMSCA(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm = NormLayer(dim, norm_type='sync_bn')
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1)
        self.layer_scale = LayerScale(dim)
        self.drop_path = StochasticDepth(p=drop_path)

    def forward(self, x):
        identity = x.clone()

        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        out = x + identity

        return out


class FFN(nn.Module):
    def __init__(self, in_ch, out_ch, hid_ch):
        super().__init__()
        self.proj1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.dwconv = nn.Conv2d(hid_ch, hid_ch, 3, padding=1, dilation=1, bias=True, groups=hid_ch)
        self.act = nn.GELU()
        self.proj2 = nn.Conv2d(hid_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.proj1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.proj2(x)
        return x


class StochasticDepth(nn.Module):
    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        return self._stochastic_depth(x, self.p, self.mode, self.training)

    def _stochastic_depth(self,
                          input: torch.Tensor,
                          p: float,
                          mode: str,
                          training: bool = True):
        if not training or p == 0.0:
            return input

        survival_rate = 1.0 - p
        if mode == 'row':
            shape = [input.shape[0]] + [1] * (input.ndim - 1)
        elif mode == 'batch':
            shape = [1] * input.ndim

        noise = torch.empty(shape, dtype=input.dtype, device=input.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)

        return input * noise

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


class BlockFFN(nn.Module):
    def __init__(self, in_ch, out_ch, hid_ch, drop_path=0.):
        super().__init__()
        self.norm = NormLayer(in_ch, norm_type='sync_bn')
        self.ffn = FFN(in_ch, out_ch, hid_ch)
        self.layer_scale = LayerScale(in_ch)
        self.drop_path = StochasticDepth(p=drop_path)

    def forward(self, x):
        identity = x.clone()

        x = self.norm(x)
        x = self.ffn(x)
        x = self.layer_scale(x)
        x = self.drop_path(x)

        op = x + identity
        return op


class MSCAStage(nn.Module):
    def __init__(self, dim, ffn_ratio=4., drop_path=0.0):
        super().__init__()
        self.msca_block = BlockMSCA(dim, drop_path)
        ffn_hid_dim = int(dim * ffn_ratio)
        self.ffn_block = BlockFFN(in_ch=dim, out_ch=dim, hid_ch=ffn_hid_dim, drop_path=drop_path)

    def forward(self, x):
        x = self.msca_block(x)
        x = self.ffn_block(x)

        return x


class MSCANet(nn.Module):
    def __init__(self, in_ch=3,
                 embed_dims=[32, 64, 160, 256],
                 ffn_ratios=[4, 4, 4, 4],
                 depths=[3, 3, 5, 2],
                 num_stages=4,
                 drop_path=0.,
                 num_classes=1000,
                 as_classifier=False):
        super(MSCANet, self).__init__()
        self.as_classifier = as_classifier
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                input_embed = StemConv(in_ch, embed_dims[0])
            else:
                input_embed = DownSample(in_ch=embed_dims[i - 1], embed_dim=embed_dims[i])

            stage = nn.ModuleList([MSCAStage(dim=embed_dims[i], ffn_ratio=ffn_ratios[i],
                                             drop_path=dpr[cur + j]) for j in range(depths[i])])

            norm_layer = NormLayer(embed_dims[i], norm_type='sync_bn')
            cur += depths[i]

            setattr(self, f'input_embed{i + 1}', input_embed)
            setattr(self, f'stage{i + 1}', stage)
            setattr(self, f'norm_layer{i + 1}', norm_layer)

        if self.as_classifier:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            input_embed = getattr(self, f'input_embed{i + 1}')
            stage = getattr(self, f'stage{i + 1}')
            norm_layer = getattr(self, f'norm_layer{i + 1}')

            x, H, W = input_embed(x)

            for stg in stage:
                x = stg(x)

            x = norm_layer(x)
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        if self.as_classifier:
            x = self.avgpool(x[-1])
            x = x.flatten(1)
            x = self.head(x)
        return x


class ConvRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class _MatrixDecomposition2DBase(nn.Module):
    '''
    Base class for furhter implementing the NMF, VQ or CD as in paper

    https://arxiv.org/pdf/2109.04553.pdf

    this script only has NMF as it has best performance for semantic segmentation
    as mentioned in paper

    D (dictionery) in paper is bases
    C (codes) in paper is coef here
    '''

    def __init__(self,
                 spatial=True,
                 md_s=1,
                 md_d=512,
                 md_r=64,
                 t_steps=6, e_steps=6,
                 inv_t=1, eta=0.9,
                 rand_init=True):
        super().__init__()

        self.spatial = spatial

        self.S = md_s
        self.D = md_d
        self.R = md_r

        self.train_steps = t_steps
        self.eval_steps = e_steps

        self.inv_t = inv_t
        self.eta = eta

        self.rand_init = rand_init

    def _bild_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        # here: N = HW and D = C in case of spatial attention
        coef = torch.bmm(x.transpose(1, 2), bases)
        # column wise softmax ignore batch dim, i.e, on HW dim
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    @torch.no_grad()
    def online_update(self, bases):
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        # column wise normalization i.e. HW dim
        self.bases = F.normalize(self.bases, dim=1)
        return None

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        if self.spatial:
            # spatial attention k
            D = C // self.S  # reduce channels
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer('bases', bases)
        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)
        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self):
        super().__init__()

        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R)).to('cuda' if torch.cuda.is_available() else 'cpu')
        bases = F.normalize(bases, dim=1)  # column wise normalization i.e HW dim

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        '''
        Algorithm 2 in paper
        NMF with multiliplicative update.
        '''
        # coef (C/codes)update
        # (B*S, D, N)T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)  # D^T @ X
        # (BS, N, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))  # D^T @ D @ C
        # Multiplicative update
        coef = coef * (numerator / (denominator + 1e-7))  # updated C
        # bases (D/dict) update
        # (BS, D, N) @ (BS, N, R) -> (BS, D, R)
        numerator = torch.bmm(x, coef)  # X @ C^T
        # (BS, D, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))  # D @ D @ C^T
        # Multiplicative update
        bases = bases * (numerator / (denominator + 1e-7))  # updated D
        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B*S, D, N)T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)  # D^T @ X
        # (BS, N, R) @ [(BS, D, R)T @ (BS, D, R)] -> (BS, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))  # D^T @ D @ C
        # Multiplicative update
        coef = coef * (numerator / (denominator + 1e-7))
        return coef


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


class HamBurger(nn.Module):
    def __init__(self, inChannels):
        super().__init__()
        self.put_cheese = True
        C = 512

        # add Relu at end as NMF works of non-negative only
        self.lower_bread = nn.Sequential(nn.Conv2d(inChannels, C, 1),
                                         nn.ReLU(inplace=True)
                                         )
        self.ham = NMF2D()
        self.cheese = ConvBNRelu(C, C)
        self.upper_bread = nn.Conv2d(C, inChannels, 1, bias=False)

    def forward(self, x):
        skip = x.clone()

        x = self.lower_bread(x)
        x = self.ham(x)

        if self.put_cheese:
            x = self.cheese(x)

        x = self.upper_bread(x)
        x = F.relu(x + skip, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class HamDecoder(nn.Module):
    '''SegNext'''

    def __init__(self, outChannels, enc_embed_dims=[32, 64, 460, 256]):
        super().__init__()

        ham_channels = 512

        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels)
        self.align = ConvRelu(ham_channels, outChannels)

    def forward(self, features):
        features = features[1:]  # drop stage 1 features b/c low level
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)

        return x


class SegNext(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels=3,
                 embed_dims=[32, 64, 160, 256],
                 ffn_ratios=[4, 4, 4, 4],
                 depths=[3, 3, 5, 2],
                 num_stages=4,
                 dec_outChannels=256,
                 drop_path=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_ch=in_channels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths,
                               num_stages=num_stages, drop_path=drop_path)
        self.decoder = HamDecoder(
            outChannels=dec_outChannels, enc_embed_dims=embed_dims)
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

#
# if __name__ == '__main__':
#     from torchsummary import summary
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model = SegNext(num_classes=2, in_channels=3, embed_dims=[32, 64, 460, 256],
#                     ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2],
#                     num_stages=4, drop_path=0.0)
#     # summary(model, (3,1024,2048))
#     model.to(device)
#     y = torch.randn((6, 3, 1024, 1024)).to(device)  # .to('cuda' if torch.cuda.is_available() else 'cpu')
#     x = model.forward(y)
#     print(x.shape)

