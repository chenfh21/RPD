from pdc_modules.efficientvit.backbone import *


class DAGBlock(nn.Module):
    def __init__(
            self,
            inputs: Dict[str, nn.Module],
            merge: str,
            post_input: Optional[nn.Module],
            middle: nn.Module,
            outputs: Dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class SegHead(DAGBlock):
    def __init__(
            self,
            fid_list: List[str],
            in_channel_list: List[int],
            stride_list: List[int],
            head_stride: int,
            head_width: int,
            head_depth: int,
            expand_ratio: float,
            middle_op: str,
            final_expand: Optional[float],
            n_classes: int,
            dropout=0,
            norm="bn2d",
            act_func="hswish",
    ):
        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    (
                        None
                        if final_expand is None
                        else ConvLayer(head_width, head_width * final_expand, 1, norm=norm, act_func=act_func)
                    ),
                    ConvLayer(
                        head_width * (1 if final_expand is None else final_expand),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }
        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 width_list=[8, 16, 32, 64, 128],
                 depth_list=[1, 2, 2, 2, 2],
                 dim=8,
                 in_channels_list=[128, 64, 32],
                 stride_list=[32, 16, 8]):
        super().__init__()
        self.num_classes = num_classes

        # use lightweight efficientvit_backbone_b0
        self.backbone = EfficientViTBackbone(
            width_list=width_list,
            depth_list=depth_list,
            dim=dim)
        self.seghead = SegHead(
            fid_list=['stage4', 'stage3', 'stage2'],
            in_channel_list=in_channels_list,
            stride_list=stride_list,
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            middle_op='mbconv',
            final_expand=4,
            n_classes=self.num_classes
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.seghead(out)
        out = out["segout"]
        out = F.interpolate(out, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return out
